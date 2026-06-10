"""
Tests for the standalone J2 (von Mises) return-mapping integrator
(issue #262 epic / #242 J2 sub-issue).

These cover the *standalone* kernel only — there is no PF coupling yet.

  1. test_j2_elastic_below_yield        — strain rate that doesn't reach
     yield must agree with pure Hooke (sigma = C : eps).
  2. test_j2_pure_tension_yield_then_plastic — 1D pure tension, no
     hardening: VM stress saturates exactly at sigma_y0.
  3. test_j2_linear_isotropic_hardening — VM stress = sigma_y0 + H * eps_p_eq
     to machine precision after a uniaxial pull.
  4. test_j2_unloading_elastic          — load past yield, then unload:
     unloading slope equals the elastic slope (no further plastic flow).
  5. test_j2_plane_strain_consistency   — strain → stress → infer strain
     via the elastic compliance is consistent with the supplied strain
     within elastic-only steps; checks plane-strain vs plane-stress
     stress states differ by the right hydrostatic offset.
"""

from __future__ import annotations

import math

import pytest
import torch

from phast.material import Material
from phast.plasticity import J2Plasticity, J2State
from phast.plasticity.j2_vonmises import (
    _stress_dev_norm,
    _stress_deviator_voigt6,
)


SQRT_3_2 = math.sqrt(1.5)


def _make_kernel(plane_stress: bool = False, H: float = 0.0,
                 hardening_type: str = 'linear_iso', **overrides) -> J2Plasticity:
    params = dict(
        E=210000.0, nu=0.3,
        plasticity_model='j2_isotropic',
        yield_stress=250.0,
        hardening_modulus=H,
        hardening_type=hardening_type,
        plane_stress=plane_stress,
    )
    params.update(overrides)
    mat = Material(**params)
    return J2Plasticity(mat)


# ---------------------------------------------------------------------------
# 1. Elastic-only step matches Hooke
# ---------------------------------------------------------------------------


def test_j2_elastic_below_yield():
    """A small strain step (well below yield) must give the elastic
    answer: sigma_{n+1} = C : (eps_{n+1} - eps_n) when stress_n = 0
    and plastic_strain_n = 0.
    """
    k = _make_kernel(plane_stress=False)
    state = J2State.zeros((1,), dtype=torch.float64)
    strain_n = torch.zeros((1, 6), dtype=torch.float64)
    strain_np1 = torch.zeros((1, 6), dtype=torch.float64)
    # Apply ε_xx = 1e-4: well below yield (yield_strain_3D ≈ 250/210e3 = 1.2e-3).
    strain_np1[..., 0] = 1e-4
    sigma, ep, eq = k.step(
        strain_n, strain_np1, state.stress, state.plastic_strain, state.eps_p_eq
    )
    # Elastic answer in plane strain with ε_yy = ε_zz = 0:
    #   σ_xx = (λ + 2μ) ε_xx
    #   σ_yy = σ_zz = λ ε_xx
    sigma_xx_expected = (k.lam + 2.0 * k.mu) * 1e-4
    sigma_yy_expected = k.lam * 1e-4
    assert sigma[0, 0].item() == pytest.approx(sigma_xx_expected, rel=1e-13)
    assert sigma[0, 1].item() == pytest.approx(sigma_yy_expected, rel=1e-13)
    assert sigma[0, 2].item() == pytest.approx(sigma_yy_expected, rel=1e-13)
    # No plastic flow.
    assert torch.all(ep == 0).item()
    assert eq[0].item() == 0.0


# ---------------------------------------------------------------------------
# 2. Pure tension yield-then-plastic (no hardening)
# ---------------------------------------------------------------------------


def test_j2_pure_tension_yield_then_plastic():
    """Apply a uniaxial-stress (plane stress) tensile load with H=0.
    The von Mises stress must saturate at exactly sigma_y0 once
    yielded, regardless of further strain.
    """
    k = _make_kernel(plane_stress=True, H=0.0)
    state = J2State.zeros((1,), dtype=torch.float64)
    strain = torch.zeros((1, 6), dtype=torch.float64)
    vm_history = []
    for i in range(20):
        new_strain = strain.clone()
        new_strain[..., 0] = (i + 1) * 2e-4  # ε_xx ramps to 4e-3
        sigma, ep, eq = k.step(
            strain, new_strain, state.stress, state.plastic_strain, state.eps_p_eq
        )
        s = _stress_deviator_voigt6(sigma)
        n = _stress_dev_norm(s)
        vm = SQRT_3_2 * n
        vm_history.append(vm.item())
        state = J2State(sigma, ep, eq)
        strain = new_strain
    # Saturated VM equals exactly sigma_y0 (within tight tol).
    assert vm_history[-1] == pytest.approx(250.0, abs=1e-6)
    # Last few steps all sit at 250 MPa.
    for vm in vm_history[-5:]:
        assert vm == pytest.approx(250.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Linear isotropic hardening: VM stress matches analytical
# ---------------------------------------------------------------------------


def test_j2_linear_isotropic_hardening():
    """With linear isotropic hardening H>0:
        VM stress = sigma_y0 + H * eps_p_eq
    must hold to machine precision at every plastic step.
    """
    H = 5000.0
    sigma_y0 = 250.0
    k = _make_kernel(plane_stress=True, H=H, hardening_type='linear_iso')
    state = J2State.zeros((1,), dtype=torch.float64)
    strain = torch.zeros((1, 6), dtype=torch.float64)
    n_plastic_checked = 0
    for i in range(30):
        new_strain = strain.clone()
        new_strain[..., 0] = (i + 1) * 1e-4  # finer increments → more plastic data
        sigma, ep, eq = k.step(
            strain, new_strain, state.stress, state.plastic_strain, state.eps_p_eq
        )
        s = _stress_deviator_voigt6(sigma)
        n = _stress_dev_norm(s)
        vm = SQRT_3_2 * n
        if eq[0].item() > 1e-12:
            expected = sigma_y0 + H * eq[0].item()
            assert vm.item() == pytest.approx(expected, rel=1e-9), (
                f"step {i}: VM={vm.item()} expected sigma_y0 + H*eps_p_eq = "
                f"{expected} (eps_p_eq={eq[0].item()})"
            )
            n_plastic_checked += 1
        state = J2State(sigma, ep, eq)
        strain = new_strain
    # Make sure we actually saw plastic steps
    assert n_plastic_checked >= 5, (
        f"Expected at least 5 plastic steps, only saw {n_plastic_checked}"
    )


# ---------------------------------------------------------------------------
# 4. Unloading is purely elastic
# ---------------------------------------------------------------------------


def test_j2_unloading_elastic():
    """Load past yield, then unload by reversing the strain. The
    unloading branch must follow the elastic slope (the plastic
    multiplier must be zero on every unload step).
    """
    k = _make_kernel(plane_stress=False, H=2000.0)
    state = J2State.zeros((1,), dtype=torch.float64)
    strain = torch.zeros((1, 6), dtype=torch.float64)
    # Loading phase: ramp ε_xx to 3e-3 in 10 steps. The 3D plane-strain
    # yield strain is ~1.5e-3, so we'll yield around step 5.
    for i in range(10):
        new_strain = strain.clone()
        new_strain[..., 0] = (i + 1) * 3e-4
        sigma, ep, eq = k.step(
            strain, new_strain, state.stress, state.plastic_strain, state.eps_p_eq
        )
        state = J2State(sigma, ep, eq)
        strain = new_strain
    # We should be plastic now.
    eps_p_eq_after_load = state.eps_p_eq[0].item()
    assert eps_p_eq_after_load > 0.0
    # Unloading phase: reverse the strain by Δε_xx = -3e-4 per step.
    sigma_before = state.stress.clone()
    eps_p_eq_before = state.eps_p_eq.clone()
    new_strain = strain.clone()
    new_strain[..., 0] = strain[..., 0] - 3e-4
    sigma_after, ep_after, eq_after = k.step(
        strain, new_strain, state.stress, state.plastic_strain, state.eps_p_eq
    )
    # No additional plastic flow.
    assert eq_after[0].item() == pytest.approx(eps_p_eq_before[0].item(), abs=1e-12)
    # Stress change matches pure elastic prediction.
    d_eps = new_strain - strain
    d_sigma_elastic = torch.einsum('ij,...j->...i', k.C_voigt6, d_eps)
    d_sigma_actual = sigma_after - sigma_before
    assert torch.allclose(d_sigma_actual, d_sigma_elastic, atol=1e-10), (
        f"Unloading is not pure elastic: actual Δσ={d_sigma_actual} "
        f"elastic Δσ={d_sigma_elastic}"
    )


# ---------------------------------------------------------------------------
# 5. Plane strain vs plane stress consistency
# ---------------------------------------------------------------------------


def test_j2_plane_strain_consistency():
    """Round-trip strain → stress → strain via the elastic compliance
    must match the input under elastic-only loading. Also check that
    plane stress yields σ_zz≈0 and that plane strain develops a
    nonzero σ_zz (the two paths agree on σ_xx in pure elasticity only
    when ν=0, so we only assert their σ_zz behaviour and the elastic
    consistency).
    """
    # Plane strain elastic round-trip
    k_ps = _make_kernel(plane_stress=False)
    strain_n = torch.zeros((1, 6), dtype=torch.float64)
    strain_np1 = torch.zeros((1, 6), dtype=torch.float64)
    strain_np1[..., 0] = 5e-4  # well below yield
    strain_np1[..., 1] = 2e-4
    state = J2State.zeros((1,), dtype=torch.float64)
    sigma, ep, eq = k_ps.step(
        strain_n, strain_np1, state.stress, state.plastic_strain, state.eps_p_eq
    )
    assert torch.all(ep == 0).item()
    assert eq[0].item() == 0.0
    # Round-trip: strain_recovered = C^-1 sigma
    C = k_ps.C_voigt6
    C_inv = torch.linalg.inv(C)
    strain_recovered = torch.einsum('ij,...j->...i', C_inv, sigma)
    assert torch.allclose(strain_recovered, strain_np1, atol=1e-14), (
        f"Strain round-trip failed: {strain_recovered} vs {strain_np1}"
    )
    # Plane strain σ_zz must equal ν*(σ_xx + σ_yy)
    sxx, syy, szz = sigma[0, 0].item(), sigma[0, 1].item(), sigma[0, 2].item()
    nu = k_ps.nu
    assert szz == pytest.approx(nu * (sxx + syy), rel=1e-12)
    # Plane stress: σ_zz must be ~0
    k_pst = _make_kernel(plane_stress=True)
    state2 = J2State.zeros((1,), dtype=torch.float64)
    sigma2, ep2, eq2 = k_pst.step(
        strain_n, strain_np1, state2.stress, state2.plastic_strain, state2.eps_p_eq
    )
    assert abs(sigma2[0, 2].item()) < 1e-9, (
        f"Plane stress σ_zz should be ~0, got {sigma2[0, 2].item()}"
    )
    # And the two branches must give *different* σ_xx (sanity check
    # they're not the same code path).
    assert abs(sigma[0, 0].item() - sigma2[0, 0].item()) > 1.0


def _fd_tangent(k: J2Plasticity, strain_n, strain_np1, state, h=1.0e-7):
    C = torch.zeros((1, 6, 6), dtype=torch.float64)
    for col in range(6):
        direction = torch.zeros_like(strain_np1)
        direction[..., col] = 1.0
        sig_p, _, _ = k.step(
            strain_n, strain_np1 + h * direction,
            state.stress, state.plastic_strain, state.eps_p_eq)
        sig_m, _, _ = k.step(
            strain_n, strain_np1 - h * direction,
            state.stress, state.plastic_strain, state.eps_p_eq)
        C[..., :, col] = (sig_p - sig_m) / (2.0 * h)
    return C


def test_j2_algorithmic_tangent_elastic_matches_hooke():
    k = _make_kernel(plane_stress=False, H=5000.0)
    state = J2State.zeros((1,), dtype=torch.float64)
    strain_n = torch.zeros((1, 6), dtype=torch.float64)
    strain_np1 = torch.zeros((1, 6), dtype=torch.float64)
    strain_np1[..., 0] = 1.0e-4
    strain_np1[..., 1] = -2.0e-5

    C_alg = k.algorithmic_tangent(
        strain_n, strain_np1, state.stress,
        state.plastic_strain, state.eps_p_eq)

    assert torch.allclose(C_alg[0], k.C_voigt6, atol=1.0e-12)


def test_j2_algorithmic_tangent_plastic_matches_finite_difference():
    k = _make_kernel(plane_stress=False, H=5000.0)
    state = J2State.zeros((1,), dtype=torch.float64)
    strain_n = torch.zeros((1, 6), dtype=torch.float64)
    strain_np1 = torch.tensor(
        [[3.5e-3, -7.0e-4, 0.0, 5.0e-4, 0.0, 0.0]],
        dtype=torch.float64,
    )

    C_alg = k.algorithmic_tangent(
        strain_n, strain_np1, state.stress,
        state.plastic_strain, state.eps_p_eq)
    C_fd = _fd_tangent(k, strain_n, strain_np1, state)

    assert torch.allclose(C_alg, C_fd, rtol=5.0e-4, atol=2.0e-3)


def test_j2_algorithmic_tangent_voce_matches_finite_difference():
    k = _make_kernel(
        plane_stress=False,
        H=1000.0,
        hardening_type='voce',
        voce_q_inf=80.0,
        voce_b=12.0,
    )
    state = J2State.zeros((1,), dtype=torch.float64)
    strain_n = torch.zeros((1, 6), dtype=torch.float64)
    strain_np1 = torch.tensor(
        [[3.5e-3, -7.0e-4, 0.0, 5.0e-4, 0.0, 0.0]],
        dtype=torch.float64,
    )

    C_alg = k.algorithmic_tangent(
        strain_n, strain_np1, state.stress,
        state.plastic_strain, state.eps_p_eq)
    C_fd = _fd_tangent(k, strain_n, strain_np1, state)

    assert torch.allclose(C_alg, C_fd, rtol=8.0e-4, atol=3.0e-3)


def test_j2_algorithmic_tangent_plane_stress_matches_finite_difference():
    k = _make_kernel(plane_stress=True, H=5000.0)
    state = J2State.zeros((1,), dtype=torch.float64)
    strain_n = torch.zeros((1, 6), dtype=torch.float64)
    strain_np1 = torch.tensor(
        [[3.5e-3, -7.0e-4, 1.2e-3, 5.0e-4, 1.0e-4, -2.0e-4]],
        dtype=torch.float64,
    )

    C_alg = k.algorithmic_tangent(
        strain_n, strain_np1, state.stress,
        state.plastic_strain, state.eps_p_eq)
    C_fd = _fd_tangent(k, strain_n, strain_np1, state)

    assert torch.allclose(C_alg, C_fd, rtol=7.0e-4, atol=3.0e-3)


def test_j2_step_with_tangent_plane_stress_matches_finite_difference():
    k = _make_kernel(plane_stress=True, H=5000.0)
    state = J2State.zeros((1,), dtype=torch.float64)
    strain_n = torch.zeros((1, 6), dtype=torch.float64)
    strain_np1 = torch.tensor(
        [[3.5e-3, -7.0e-4, 0.0, 5.0e-4, 0.0, 0.0]],
        dtype=torch.float64,
    )

    _, _, _, C_alg = k.step_with_tangent(
        strain_n, strain_np1, state.stress,
        state.plastic_strain, state.eps_p_eq)
    C_fd = _fd_tangent(k, strain_n, strain_np1, state)

    assert torch.allclose(C_alg, C_fd, rtol=5.0e-4, atol=2.0e-3)


def test_j2_algorithmic_tangent_perfect_plasticity_is_finite():
    k = _make_kernel(plane_stress=False, H=0.0)
    state = J2State.zeros((1,), dtype=torch.float64)
    strain_n = torch.zeros((1, 6), dtype=torch.float64)
    strain_np1 = torch.tensor(
        [[4.0e-3, -8.0e-4, 0.0, 4.0e-4, 0.0, 0.0]],
        dtype=torch.float64,
    )

    C_alg = k.algorithmic_tangent(
        strain_n, strain_np1, state.stress,
        state.plastic_strain, state.eps_p_eq)

    assert torch.isfinite(C_alg).all()
    assert torch.linalg.norm(C_alg).item() > 0.0


def test_j2_algorithmic_tangent_batched_elastic_and_plastic():
    k = _make_kernel(plane_stress=False, H=5000.0)
    state = J2State.zeros((2,), dtype=torch.float64)
    strain_n = torch.zeros((2, 6), dtype=torch.float64)
    strain_np1 = torch.zeros((2, 6), dtype=torch.float64)
    strain_np1[0, 0] = 1.0e-4
    strain_np1[1, 0] = 4.0e-3
    strain_np1[1, 1] = -8.0e-4

    C_alg = k.algorithmic_tangent(
        strain_n, strain_np1, state.stress,
        state.plastic_strain, state.eps_p_eq)

    assert torch.allclose(C_alg[0], k.C_voigt6, atol=1.0e-12)
    assert not torch.allclose(C_alg[1], k.C_voigt6)


def test_j2_kinematic_hardening_is_still_rejected():
    with pytest.raises(NotImplementedError, match="back-stress"):
        J2Plasticity(Material(
            E=210000.0, nu=0.3,
            plasticity_model='j2_kinematic',
            yield_stress=250.0,
            hardening_modulus=0.0,
            hardening_type='linear_iso',
            plane_stress=False,
        ))
