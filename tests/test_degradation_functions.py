"""Unit tests for ``Material.degradation`` across all three g(d) presets.

W4 audit Tier-3, Gap 2: ``material.py`` exposes ``degradation_type`` with
literals ``'standard'`` (default), ``'cubic'``, ``'rational'`` (Wu 2017),
but the existing test suite only exercised ``'standard'``.

Note: the audit prompt named the default ``'quadratic'``; the actual
literal in ``material.py`` is ``'standard'``. We test the real literals.

Contract for every g:
  - g(0) = 1                         (un-degraded at d=0)
  - g(1) ≈ eta_residual              (residual stiffness at d=1)
  - g'(0) < 0 for the currently implemented AT2 degradation families
  - g monotone-decreasing on [0, 1]
  - At intermediate d, the Borden cubic family matches its closed form.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from phast.material import Material  # noqa: E402


@pytest.fixture(params=['standard', 'cubic', 'rational'])
def material(request):
    """Build a Material at the requested degradation_type."""
    return Material(
        E=210000.0, nu=0.3, Gc=2.7, l0=0.015, rho=7.8e-9,
        pf_model='AT2', energy_split='isotropic',
        degradation_type=request.param, eta_residual=1e-7,
    )


def test_degradation_endpoints(material):
    """g(0) = 1 exactly and g(1) ≈ eta_residual for all three types.

    All three forms now use the ``(1-eta) * R(d) + eta`` blending so
    g(0) = 1 exactly (R(0) = 1) and g(1) = eta exactly (R(1) = 0).
    See #279 — the rational form previously gave g(0) = 1 + eta.
    """
    eta = material.eta_residual
    d0 = torch.tensor([0.0], dtype=torch.float64)
    d1 = torch.tensor([1.0], dtype=torch.float64)
    g0 = material.degradation(d0).item()
    g1 = material.degradation(d1).item()
    assert g0 == 1.0, (
        f"g(0)={g0} != 1 exactly for {material.degradation_type}")
    assert g1 == pytest.approx(eta, abs=10 * eta), (
        f"g(1)={g1} != eta={eta} for {material.degradation_type}")


def test_degradation_negative_derivative_at_zero(material):
    """g'(0) < 0 — degradation must immediately reduce stiffness as d grows.

    The audit prompt asked for g'(0) = 0, which is a property of the
    AT1 'optimal' degradation g(d) = 1 - p(d) families — NOT of the
    forms implemented here:
      standard: g'(0) = -2 (1-eta)
      cubic   : g'(0) = -s (1-eta), with default s=1
      rational: g'(0) = -a1 = -4/(pi*l0)
    All three are strictly negative, which is the well-known reason AT2
    has no elastic threshold and AT1 needs the explicit S_H source term
    (see ``test_at1_threshold.py``). We pin the strict-negative property
    here so any future swap to a g'(0)=0 family is caught.
    """
    d = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    g = material.degradation(d).sum()
    g.backward()
    grad = d.grad.item()
    assert grad < 0.0, (
        f"g'(0)={grad} should be strictly negative for "
        f"{material.degradation_type}")


def test_degradation_monotone_decreasing(material):
    """g monotone-decreasing on [0, 1]: g(d_i) > g(d_{i+1}) for a dense sweep."""
    d = torch.linspace(0.0, 1.0, 51, dtype=torch.float64)
    g = material.degradation(d)
    diffs = g[1:] - g[:-1]
    # Strict decrease (allow tiny float slack near eta_residual saturation).
    assert (diffs <= 1e-15).all(), (
        f"g not monotone-decreasing for {material.degradation_type}: "
        f"max(diff)={diffs.max().item():.3e}")


def test_degradation_relative_shape_at_intermediate_d():
    """Closed-form values at d=0.5 with l0=0.015 (a1 = 4/(pi*l0) ≈ 85):

      standard: 0.25 + eta
      cubic   : (3-s)*0.25 - (2-s)*0.125 = 0.375 for s=1
      rational: 0.25 / (0.25 + 0.5 * 1.5 * 85) + eta ≈ 3.9e-3 + eta

    So at this (l0, d) combination the ranking is rational < standard < cubic:
    the rational form drops the fastest because a1 dominates the denominator
    (Wu 2017's calibration is for matching peak stress, not g(0.5) shape).
    """
    common = dict(E=210000.0, nu=0.3, Gc=2.7, l0=0.015, rho=7.8e-9,
                  pf_model='AT2', energy_split='isotropic',
                  eta_residual=1e-7)
    mat_std = Material(**common, degradation_type='standard')
    mat_cub = Material(**common, degradation_type='cubic')
    mat_rat = Material(**common, degradation_type='rational')

    d = torch.tensor([0.5], dtype=torch.float64)
    g_std = mat_std.degradation(d).item()
    g_cub = mat_cub.degradation(d).item()
    g_rat = mat_rat.degradation(d).item()

    # Closed-form pins for standard and cubic.
    assert g_std == pytest.approx(0.25, abs=1e-6)
    assert g_cub == pytest.approx(0.375, abs=1e-6)
    # Strict ranking at d=0.5, l0=0.015.
    assert g_rat < g_std < g_cub, (
        f"Expected g_rat < g_std < g_cub at d=0.5, l0=0.015; got "
        f"rational={g_rat:.4f}, cubic={g_cub:.4f}, standard={g_std:.4f}")


def test_cubic_s_parameter_controls_initial_slope():
    """Borden cubic has g'(0) = -s, so s=0 gives a flat initial slope."""
    common = dict(E=210000.0, nu=0.3, Gc=2.7, l0=0.015, rho=7.8e-9,
                  pf_model='AT2', energy_split='isotropic',
                  degradation_type='cubic', eta_residual=0.0)
    for s in (0.0, 0.5, 1.0):
        mat = Material(**common, cubic_s=s)
        d = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        g = mat.degradation(d).sum()
        g.backward()
        assert d.grad.item() == pytest.approx(
            -s * (1.0 - mat.eta_residual), abs=1e-12)


def test_pfczm_gamma_correction_preserves_strength_threshold():
    from phast.damage_solver import PhaseFieldDamageSolver
    from phast.fem_operators import FEMOperators
    from phast.mesh import FEMMesh

    nodes = torch.tensor(
        [[0.0, 0.0], [1.3, 0.0], [0.0, 0.7]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mesh = FEMMesh.from_tensors(
        nodes, elements, device="cpu", dtype=torch.float64, element_type="T3")
    mat = Material(
        E=3000.0, nu=0.3, Gc=0.12, l0=0.1,
        pf_model='PFCZM', sigma_ts=3.0,
        energy_split='isotropic', gamma_correction=True,
    )
    solver = PhaseFieldDamageSolver(
        FEMOperators(mesh, mat), tol=1.0e-10, max_iter=20,
        use_multigrid=False, bounds_method="projected_cg")

    assert float(solver._element_Gc_cg()[0]) < mat.Gc
    Hcrit = mat.sigma_ts * mat.sigma_ts / (2.0 * mat.E)
    H = torch.full((mesh.n_elems,), Hcrit, dtype=torch.float64)
    d0 = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    residual = solver.compute_residual(H, d0)
    assert float(torch.linalg.vector_norm(residual)) < 1.0e-12


def test_pfczm_solver_reports_convergence_flag():
    from phast.damage_solver import PhaseFieldDamageSolver
    from phast.fem_operators import FEMOperators
    from phast.mesh import FEMMesh

    nodes = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mesh = FEMMesh.from_tensors(
        nodes, elements, device="cpu", dtype=torch.float64, element_type="T3")
    mat = Material(
        E=3000.0, nu=0.3, Gc=0.12, l0=0.1,
        pf_model='PFCZM', sigma_ts=3.0,
        energy_split='isotropic',
    )
    H = torch.zeros(mesh.n_elems, dtype=torch.float64)
    d0 = torch.zeros(mesh.n_nodes, dtype=torch.float64)

    converging_solver = PhaseFieldDamageSolver(
        FEMOperators(mesh, mat), tol=1.0e-8, max_iter=10,
        use_multigrid=False, bounds_method="projected_cg")
    d = converging_solver.solve(H, d0)
    assert torch.allclose(d, d0)
    assert converging_solver.last_converged is True

    nonconverging_solver = PhaseFieldDamageSolver(
        FEMOperators(mesh, mat), tol=1.0e-16, max_iter=0,
        use_multigrid=False, bounds_method="projected_cg")
    nonconverging_solver.solve(H, d0)
    assert nonconverging_solver.last_converged is False


def test_damage_solver_rejects_nonstandard_degradation():
    from phast.damage_solver import PhaseFieldDamageSolver
    from phast.fem_operators import FEMOperators

    class _Mesh:
        device = 'cpu'
        dtype = torch.float64
        n_nodes = 3
        n_elems = 1
        elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
        areas = torch.tensor([0.5], dtype=torch.float64)
        grad_phi = torch.zeros(1, 3, 2, dtype=torch.float64)
        M_scalar = torch.ones(3, dtype=torch.float64)
        _elem_flat = elements.flatten()
        h_min = 1.0

    mat = Material(
        E=210.0, nu=0.3, Gc=2.7, l0=0.1,
        pf_model='AT2', energy_split='isotropic',
        degradation_type='cubic',
    )
    fem = FEMOperators(_Mesh(), mat)
    with pytest.raises(NotImplementedError, match="degradation_type='standard'"):
        PhaseFieldDamageSolver(fem, use_multigrid=False)
