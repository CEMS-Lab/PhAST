import sys
import types

import torch

from phast.damage_solver import PhaseFieldDamageSolver
from phast.multigrid import AMGPreconditioner


class _TinyMesh:
    device = torch.device("cpu")
    n_nodes = 3
    n_elems = 1
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
    areas = torch.tensor([0.5], dtype=torch.float64)
    grad_phi = torch.tensor(
        [[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float64,
    )
    sparse_indices = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 2, 2, 2],
         [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        dtype=torch.long,
    )


class _TinyMaterial:
    Gc = 2.7
    l0 = 0.015
    pf_model = "AT2"
    gamma_correction = False
    at1_source = 0.0


class _TinyFEM:
    device = torch.device("cpu")
    mesh = _TinyMesh()
    material = _TinyMaterial()


class _ExplodingAMG:
    def __init__(self):
        self._P = object()
        self._R = object()
        self._A_coarse = object()
        self._A_coarse_chol = object()
        self._n_coarse = 1
        self._cheb_eig_max = 1.0
        self.cleared = False
        self.vcycle_calls = 0

    def _clear_hierarchy(self):
        self._P = None
        self._R = None
        self._A_coarse = None
        self._A_coarse_chol = None
        self._n_coarse = 0
        self._cheb_eig_max = None
        self.cleared = True

    def vcycle(self, *_args, **_kwargs):
        self.vcycle_calls += 1
        raise AssertionError("AMG vcycle must not be used during fallback")


def test_amg_update_failure_clears_stale_hierarchy(monkeypatch):
    """A failed PyAMG rebuild must not leave an old coarse hierarchy active."""

    fake_pyamg = types.SimpleNamespace(
        smoothed_aggregation_solver=lambda _: (_ for _ in ()).throw(
            ValueError("synthetic AMG failure")
        )
    )
    monkeypatch.setitem(sys.modules, "pyamg", fake_pyamg)

    mg = AMGPreconditioner(_TinyMesh(), Gc_l0=0.03, dtype=torch.float64)
    stale = torch.sparse_csr_tensor(
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1.0], dtype=torch.float64),
        size=(1, 1),
    )
    mg._P = stale
    mg._R = stale
    mg._A_coarse = torch.ones((1, 1), dtype=torch.float64)
    mg._A_coarse_chol = torch.ones((1, 1), dtype=torch.float64)
    mg._n_coarse = 1
    mg._cheb_eig_max = 1.0

    ok = mg.update(torch.ones(1, dtype=torch.float64))

    assert ok is False
    assert mg._P is None
    assert mg._R is None
    assert mg._A_coarse is None
    assert mg._A_coarse_chol is None
    assert mg._n_coarse == 0
    assert mg._cheb_eig_max is None
    assert mg._A_diag_inv is not None

    r = torch.ones(3, dtype=torch.float64)
    z = mg.vcycle(r, torch.ones(1, dtype=torch.float64))
    assert torch.isfinite(z).all()


def test_amg_generic_rebuild_error_clears_stale_hierarchy(monkeypatch):
    """Unexpected AMG setup failures must also clear old coarse operators."""

    fake_pyamg = types.SimpleNamespace(
        smoothed_aggregation_solver=lambda _: (_ for _ in ()).throw(
            RuntimeError("synthetic partial rebuild failure")
        )
    )
    monkeypatch.setitem(sys.modules, "pyamg", fake_pyamg)

    mg = AMGPreconditioner(_TinyMesh(), Gc_l0=0.03, dtype=torch.float64)
    stale = torch.sparse_csr_tensor(
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1.0], dtype=torch.float64),
        size=(1, 1),
    )
    mg._P = stale
    mg._R = stale
    mg._A_coarse = torch.ones((1, 1), dtype=torch.float64)
    mg._A_coarse_chol = torch.ones((1, 1), dtype=torch.float64)
    mg._n_coarse = 1
    mg._cheb_eig_max = 1.0

    ok = mg.update(torch.ones(1, dtype=torch.float64))

    assert ok is False
    assert mg._P is None
    assert mg._R is None
    assert mg._A_coarse is None
    assert mg._A_coarse_chol is None
    assert mg._n_coarse == 0
    assert mg._cheb_eig_max is None


def test_amg_nonfinite_reaction_uses_jacobi_without_pyamg_build(monkeypatch):
    """Non-finite coefficients should clear stale AMG and avoid PyAMG setup."""

    def _unexpected_build(_):
        raise AssertionError("pyAMG should not be called for non-finite input")

    fake_pyamg = types.SimpleNamespace(smoothed_aggregation_solver=_unexpected_build)
    monkeypatch.setitem(sys.modules, "pyamg", fake_pyamg)

    mg = AMGPreconditioner(_TinyMesh(), Gc_l0=0.03, dtype=torch.float64)
    stale = torch.sparse_csr_tensor(
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1.0], dtype=torch.float64),
        size=(1, 1),
    )
    mg._P = stale
    mg._R = stale
    mg._A_coarse = torch.ones((1, 1), dtype=torch.float64)
    mg._A_coarse_chol = torch.ones((1, 1), dtype=torch.float64)
    mg._n_coarse = 1
    mg._cheb_eig_max = 1.0

    ok = mg.update(torch.tensor([float("nan")], dtype=torch.float64))

    assert ok is False
    assert mg._P is None
    assert mg._R is None
    assert mg._A_coarse is None
    assert mg._A_coarse_chol is None
    assert mg._n_coarse == 0
    assert mg._cheb_eig_max is None
    assert mg._A_diag_inv is not None

    z = mg.vcycle(torch.ones(3, dtype=torch.float64),
                  torch.ones(1, dtype=torch.float64))
    assert torch.isfinite(z).all()


def test_high_damage_fallback_skips_amg_vcycle_in_projected_cg():
    """Near-fully damaged PPCG solves must use Jacobi after AMG is cleared."""

    solver = PhaseFieldDamageSolver(
        _TinyFEM(),
        tol=1e-8,
        max_iter=2,
        bounds_method="projected_cg",
        use_multigrid=False,
        preconditioner="jacobi",
    )
    amg = _ExplodingAMG()
    solver._multigrid = amg
    solver._preconditioner = "amg"
    solver._use_multigrid = True

    H = torch.ones(_TinyMesh.n_elems, dtype=torch.float64)
    d_prev = torch.full((_TinyMesh.n_nodes,), 0.96, dtype=torch.float64)

    d = solver.solve(H, d_prev)

    assert amg.cleared is True
    assert amg.vcycle_calls == 0
    assert solver._amg_fallback_active is True
    assert torch.isfinite(d).all()
    assert torch.all(d >= d_prev)
