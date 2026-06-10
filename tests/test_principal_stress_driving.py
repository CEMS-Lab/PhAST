"""Tests for the principal-stress crack-driving force (issue #248).

The driving scalar fed into H is selected by ``material.driving_force``:

* ``'strain_energy'`` — Ψ⁺ from the configured ``energy_split`` (legacy).
* ``'principal_stress'`` — D = ⟨σ₁⟩²/(2E), Wu (2020)-style.

Coverage:

1. ``test_default_is_pass_through_to_psi_plus`` — with the default knob,
   ``compute_driving_force`` returns the **exact same tensor** as
   ``compute_psi_plus`` (bit-identical legacy behaviour).
2. ``test_principal_stress_non_negative`` — the principal-stress branch
   is non-negative for arbitrary strain inputs.
3. ``test_zero_in_pure_compression`` — pure-compression strain (negative
   ε_xx, ε_yy with σ₁ < 0) yields exactly zero driving force.
4. ``test_nonzero_in_pure_tension`` — pure-tension strain yields positive
   driving force.
5. ``test_1d_bar_tension_analytic`` — for a uniaxial tension state with
   σ_xx = E·ε > 0 (and σ_yy = σ_xy = 0 in plane stress with ν=0), the
   principal-stress driving force matches the closed form
   D = E·ε² / 2.
6. ``test_invalid_mode_raises`` — typo in ``driving_force`` raises a
   clear ``ValueError`` rather than silently falling through.
"""

from __future__ import annotations

import math

import pytest
import torch

from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mesh import FEMMesh


# ---------------------------------------------------------------------------
# Minimal single-element mesh fixture (one T3, area=0.5).
# ---------------------------------------------------------------------------

@pytest.fixture
def fem_factory():
    """Build a FEMOperators on a one-T3 mesh with the requested driving_force."""
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)

    def _build(driving_force='strain_energy', E=210000.0, nu=0.0,
               energy_split='isotropic'):
        mesh = FEMMesh.from_tensors(nodes, elements, device='cpu',
                                    dtype=torch.float64)
        mat = Material(
            E=E, nu=nu, Gc=2.7, l0=0.05, rho=7.8e-9,
            energy_split=energy_split, pf_model='AT2',
            eta_residual=1e-7, plane_stress=True,
            driving_force=driving_force,
        )
        return FEMOperators(mesh, mat)

    return _build


def _strain_from_disp(fem, u: torch.Tensor) -> tuple:
    return fem.compute_strain(u)


# ---------------------------------------------------------------------------
# 1. Default knob is a literal pass-through to compute_psi_plus.
# ---------------------------------------------------------------------------

def test_default_is_pass_through_to_psi_plus(fem_factory):
    fem = fem_factory(driving_force='strain_energy', energy_split='spectral',
                      nu=0.3)
    # Mixed tension/shear strain — exercises the spectral branch.
    u = torch.tensor([[0.0, 0.0],
                      [1.0e-3, 5.0e-4],
                      [2.0e-4, 8.0e-4]], dtype=torch.float64)
    psi_ref = fem.compute_psi_plus(u)
    D = fem.compute_driving_force(u)
    # Bit-identical (same call path).
    assert torch.equal(D, psi_ref), (
        "default driving_force=strain_energy must be a literal "
        "pass-through to compute_psi_plus")


# ---------------------------------------------------------------------------
# 2. Principal-stress branch is non-negative.
# ---------------------------------------------------------------------------

def test_principal_stress_non_negative(fem_factory):
    fem = fem_factory(driving_force='principal_stress', nu=0.3)
    torch.manual_seed(0)
    # Random strain at the three nodes — produces a single element strain.
    u = 1.0e-3 * torch.randn(3, 2, dtype=torch.float64)
    D = fem.compute_driving_force(u)
    assert (D >= 0).all(), f"principal-stress driving force has negatives: {D}"


# ---------------------------------------------------------------------------
# 3. Pure compression -> exactly zero driving force.
# ---------------------------------------------------------------------------

def test_zero_in_pure_compression(fem_factory):
    fem = fem_factory(driving_force='principal_stress', nu=0.0)
    # Uniform compressive strain ε_xx = ε_yy = -ε, ε_xy = 0 ->
    # σ_xx = σ_yy = E·(-ε), σ_xy = 0, so σ₁ = E·(-ε) < 0 -> ⟨σ₁⟩ = 0.
    eps = 1.0e-3
    # Build u that produces uniform compression on the single T3:
    # ε_xx = du_x/dx = -eps (linear in x), ε_yy = du_y/dy = -eps.
    u = torch.tensor([[0.0,    0.0],
                      [-eps,    0.0],
                      [0.0,   -eps]], dtype=torch.float64)
    D = fem.compute_driving_force(u)
    assert torch.allclose(D, torch.zeros_like(D), atol=0.0), (
        f"pure-compression driving force not zero: {D}")


# ---------------------------------------------------------------------------
# 4. Pure tension -> positive driving force.
# ---------------------------------------------------------------------------

def test_nonzero_in_pure_tension(fem_factory):
    fem = fem_factory(driving_force='principal_stress', nu=0.0)
    eps = 1.0e-3
    # Uniform biaxial tension.
    u = torch.tensor([[0.0,   0.0],
                      [eps,   0.0],
                      [0.0,   eps]], dtype=torch.float64)
    D = fem.compute_driving_force(u)
    assert (D > 0).all(), f"pure-tension driving force not positive: {D}"


# ---------------------------------------------------------------------------
# 5. 1D bar tension closed form: D = E·ε² / 2.
# ---------------------------------------------------------------------------

def test_1d_bar_tension_analytic(fem_factory):
    """Plane-stress, ν=0 uniaxial tension: σ_xx = E·ε, σ_yy = σ_xy = 0.

    Then σ₁ = max(σ_xx, σ_yy) = E·ε > 0, and
        D = ⟨σ₁⟩² / (2E) = (E·ε)² / (2E) = E·ε² / 2.
    """
    E = 210_000.0
    fem = fem_factory(driving_force='principal_stress', E=E, nu=0.0)
    eps = 2.5e-3
    # Uniaxial tension along x: u_x = eps·x, u_y = 0.
    u = torch.tensor([[0.0,   0.0],
                      [eps,   0.0],
                      [0.0,   0.0]], dtype=torch.float64)
    D = fem.compute_driving_force(u)
    expected = 0.5 * E * eps ** 2
    assert torch.allclose(D, torch.full_like(D, expected), rtol=1e-10), (
        f"D={D.item():.6e} expected {expected:.6e} (E·ε²/2)")


def test_principal_stress_uses_elementwise_E_with_diff_field(fem_factory):
    """The principal-stress driver must scale with local E, not bulk E."""
    fem = fem_factory(driving_force='principal_stress', E=100.0, nu=0.0)
    fem.diff_E_field = torch.tensor([100.0, 400.0], dtype=torch.float64)

    eps = torch.full((2,), 2.0e-3, dtype=torch.float64)
    zero = torch.zeros_like(eps)
    D = fem.compute_driving_force(None, strain=(eps, zero, zero))

    expected = 0.5 * fem.diff_E_field * eps ** 2
    torch.testing.assert_close(D, expected, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# 6. Invalid mode -> ValueError.
# ---------------------------------------------------------------------------

def test_invalid_mode_raises(fem_factory):
    fem = fem_factory(driving_force='strain_energy')
    # Bypass dataclass validation by mutating the live attribute.
    fem.material.driving_force = 'not_a_real_mode'
    u = torch.zeros(3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="Unknown driving_force"):
        fem.compute_driving_force(u)
