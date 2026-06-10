import math

import pytest
import torch

from phast.damage_solver import PhaseFieldDamageSolver
from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mesh import FEMMesh


def _unit_square_mesh():
    nodes = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return FEMMesh.from_tensors(
        nodes, elements, device="cpu", dtype=torch.float64, element_type="T3")


def test_pfczm_requires_tensile_strength():
    with pytest.raises(ValueError, match="requires sigma_ts > 0"):
        Material(pf_model="PFCZM", sigma_ts=0.0)


def test_pfczm_wu_coefficients_match_tensile_strength_calibration():
    mat = Material(
        E=200.0, nu=0.3, Gc=1.5, l0=0.25,
        pf_model="PFCZM", sigma_ts=5.0, eta_residual=1e-9,
    )
    assert mat.pfczm_a1 == pytest.approx(
        4.0 * mat.E * mat.Gc / (math.pi * mat.l0 * mat.sigma_ts ** 2))
    assert mat.pfczm_c_alpha == pytest.approx(math.pi)

    d0 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    g0 = mat.degradation(d0).sum()
    g0.backward()
    assert d0.grad.item() == pytest.approx(
        -(1.0 - mat.eta_residual) * mat.pfczm_a1, rel=1e-12)


def test_pfczm_projected_solve_has_strength_threshold_and_finite_residual():
    mesh = _unit_square_mesh()
    mat = Material(
        E=200.0, nu=0.3, Gc=1.5, l0=0.25, rho=1.0,
        energy_split="isotropic", pf_model="PFCZM", sigma_ts=5.0,
        eta_residual=1e-9,
    )
    solver = PhaseFieldDamageSolver(
        FEMOperators(mesh, mat),
        tol=1e-8,
        max_iter=1000,
        use_multigrid=False,
        bounds_method="projected_cg",
    )

    d_prev = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    hcrit = mat.sigma_ts ** 2 / (2.0 * mat.E)

    H_low = torch.full((mesh.n_elems,), 0.5 * hcrit, dtype=torch.float64)
    d_low = solver.solve(H_low, d_prev)
    assert d_low.max().item() < 1e-10

    H_high = torch.full((mesh.n_elems,), 3.0 * hcrit, dtype=torch.float64)
    d_high = solver.solve(H_high, d_prev)
    assert torch.isfinite(d_high).all()
    assert d_high.min().item() >= -1e-12
    assert d_high.max().item() <= 1.0 + 1e-12
    assert d_high.max().item() > 1e-4
    residual = solver.compute_residual(H_high, d_high)
    assert torch.isfinite(residual).all()
    assert getattr(solver, "last_residual", float("inf")) < 1e-5


def test_pfczm_rejects_differentiable_damage_path():
    mesh = _unit_square_mesh()
    mat = Material(
        E=200.0, nu=0.3, Gc=1.5, l0=0.25, rho=1.0,
        energy_split="isotropic", pf_model="PFCZM", sigma_ts=5.0,
    )
    solver = PhaseFieldDamageSolver(
        FEMOperators(mesh, mat), use_multigrid=False)
    solver.differentiable = True
    H = torch.ones(mesh.n_elems, dtype=torch.float64)
    d_prev = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="Differentiable PF-CZM"):
        solver.solve(H, d_prev)
