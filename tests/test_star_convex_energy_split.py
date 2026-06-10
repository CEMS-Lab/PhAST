"""Unit tests for the ``energy_split='star_convex'`` preset (Kumar et al. 2020).

W4 audit Tier-3, Gap 1: ``_psi_plus_star_convex`` and
``compute_stress_star_convex`` (``fem_operators.py:835`` and ``:369``)
are referenced by ``Material``'s ``star_convex`` literal but had no
unit tests. These tests pin the contract on a single-element MockMesh:

  - tension state (tr ε >= 0): ψ⁺ = 0.5 ε:C:ε  (full elastic energy)
  - compression state (tr ε < 0): ψ⁺ = μ dev:dev  (deviatoric only)
  - at d=0 the total stress equals the full elastic stress σ = λ tr I + 2μ ε
  - at d=1 with tension: stress collapses to the residual stiffness
    g(1) ≈ eta_residual (compression-only lockout)
  - at d=1 with compression: volumetric stress fully preserved
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from phast.fem_operators import FEMOperators  # noqa: E402
from phast.material import Material  # noqa: E402


class _MockMesh:
    """Single-triangle mesh — minimum surface area for unit tests."""

    def __init__(self, n_nodes=3, device='cpu', dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.n_nodes = n_nodes
        self.elements = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
        self.n_elems = 1
        self.areas = torch.tensor([0.5], dtype=dtype, device=device)
        self.grad_phi = torch.zeros(1, 3, 2, dtype=dtype, device=device)
        self.M_scalar = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=device)
        self._elem_flat = self.elements.flatten()
        self.h_min = 1.0


@pytest.fixture
def fem_star_convex():
    mesh = _MockMesh()
    mat = Material(
        E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
        rho=7.8e-9,
        pf_model='AT2', energy_split='star_convex',
        plane_stress=False, eta_residual=1e-7,
    )
    return FEMOperators(mesh, mat)


def test_psi_plus_star_convex_tension_full_energy(fem_star_convex):
    """Tension (tr ε > 0): ψ⁺ must equal the full isotropic 0.5 ε:C:ε."""
    eps_xx = torch.tensor([0.01], dtype=torch.float64)
    eps_yy = torch.tensor([0.005], dtype=torch.float64)  # tr = 0.015 > 0
    gam_xy = torch.tensor([0.002], dtype=torch.float64)
    strain = (eps_xx, eps_yy, gam_xy)

    psi_sc = fem_star_convex._psi_plus_star_convex(strain)
    psi_iso = fem_star_convex._psi_plus_isotropic(strain)
    torch.testing.assert_close(psi_sc, psi_iso)
    assert psi_sc.item() > 0.0


def test_psi_plus_star_convex_compression_deviatoric_only(fem_star_convex):
    """Compression (tr ε < 0): ψ⁺ must collapse to μ * dev:dev (no volumetric)."""
    eps_xx = torch.tensor([-0.01], dtype=torch.float64)
    eps_yy = torch.tensor([-0.005], dtype=torch.float64)  # tr = -0.015 < 0
    gam_xy = torch.tensor([0.002], dtype=torch.float64)
    strain = (eps_xx, eps_yy, gam_xy)

    psi_sc = fem_star_convex._psi_plus_star_convex(strain)

    # Hand-compute deviatoric-only μ dev:dev (3D dev with eps_zz=0 plane strain)
    mu = fem_star_convex.material.mu
    tr = eps_xx + eps_yy  # plane strain: 3D trace = 2D trace
    exy = gam_xy / 2.0
    dev_xx = eps_xx - tr / 3.0
    dev_yy = eps_yy - tr / 3.0
    dev_zz = -tr / 3.0
    dev_dot = dev_xx ** 2 + dev_yy ** 2 + dev_zz ** 2 + 2.0 * exy ** 2
    expected = mu * dev_dot

    torch.testing.assert_close(psi_sc, expected)
    # Sanity: compression psi must be strictly less than full-energy isotropic
    psi_iso = fem_star_convex._psi_plus_isotropic(strain)
    assert psi_sc.item() < psi_iso.item()


def test_stress_star_convex_d_zero_recovers_full_elastic(fem_star_convex):
    """At d=0 (g(0)=1), stress must equal full elastic σ for any strain state."""
    mat = fem_star_convex.material
    eps_xx = torch.tensor([0.01], dtype=torch.float64)
    eps_yy = torch.tensor([0.005], dtype=torch.float64)  # tension
    gam_xy = torch.tensor([0.002], dtype=torch.float64)
    g_d = torch.tensor([1.0], dtype=torch.float64)  # un-degraded

    sxx, syy, sxy = fem_star_convex.compute_stress_star_convex(
        eps_xx, eps_yy, gam_xy, g_d)

    # Hand-compute full plane-strain σ = λ tr I + 2μ ε
    lam = mat.lam
    mu = mat.mu
    tr = eps_xx + eps_yy
    sxx_ref = lam * tr + 2.0 * mu * eps_xx
    syy_ref = lam * tr + 2.0 * mu * eps_yy
    sxy_ref = mu * gam_xy  # 2μ * eps_xy = μ * gam_xy

    torch.testing.assert_close(sxx, sxx_ref)
    torch.testing.assert_close(syy, syy_ref)
    torch.testing.assert_close(sxy, sxy_ref)


def test_stress_star_convex_d_one_tension_collapses_to_eta(fem_star_convex):
    """At d=1 in tension, stress must collapse to ~eta_residual * full elastic.

    g(1) = eta_residual + (1 - eta_residual) * 0 = eta_residual ≈ 1e-7,
    so |σ_degraded| ~ eta_residual * |σ_full|, NOT exactly zero.
    """
    mat = fem_star_convex.material
    eta = mat.eta_residual
    eps_xx = torch.tensor([0.01], dtype=torch.float64)
    eps_yy = torch.tensor([0.005], dtype=torch.float64)
    gam_xy = torch.tensor([0.002], dtype=torch.float64)
    d = torch.tensor([1.0], dtype=torch.float64)
    g_d = mat.degradation(d)

    sxx, syy, sxy = fem_star_convex.compute_stress_star_convex(
        eps_xx, eps_yy, gam_xy, g_d)

    # Reference: full elastic stress (g=1)
    g_full = torch.tensor([1.0], dtype=torch.float64)
    sxx0, syy0, sxy0 = fem_star_convex.compute_stress_star_convex(
        eps_xx, eps_yy, gam_xy, g_full)

    # Each component should be ~eta * full, well below 1% of full magnitude.
    s_full_max = max(abs(sxx0.item()), abs(syy0.item()), abs(sxy0.item()))
    rel_tol = 100.0 * eta  # generous: 100x eta ~ 1e-5 floor
    assert abs(sxx.item()) <= rel_tol * s_full_max + 1e-12
    assert abs(syy.item()) <= rel_tol * s_full_max + 1e-12
    assert abs(sxy.item()) <= rel_tol * s_full_max + 1e-12


def test_stress_star_convex_d_one_compression_preserves_volumetric(fem_star_convex):
    """At d=1 in compression, the volumetric pressure (kappa * tr) must be preserved.

    This is the defining feature of star-convex: compression remains stiff
    even at full damage. The deviatoric component is degraded but the
    volumetric kappa*tr term is intact.
    """
    mat = fem_star_convex.material
    eps_xx = torch.tensor([-0.01], dtype=torch.float64)
    eps_yy = torch.tensor([-0.01], dtype=torch.float64)  # tr = -0.02 < 0
    gam_xy = torch.tensor([0.0], dtype=torch.float64)
    d = torch.tensor([1.0], dtype=torch.float64)
    g_d = mat.degradation(d)

    sxx, syy, sxy = fem_star_convex.compute_stress_star_convex(
        eps_xx, eps_yy, gam_xy, g_d)

    # Volumetric pressure under plane strain: tr_3D = tr_2D = -0.02
    # Deviatoric is zero (eps_xx == eps_yy and gam_xy=0 -> dev_xx = dev_yy = 0,
    # dev_zz = -tr/3); μ * dev:dev contributes only via dev_zz^2 to ψ but
    # to the 2D Cauchy stress only via dev_xx and dev_yy components, both 0.
    # Hence σ_xx = σ_yy = kappa * tr  (intact, NOT degraded).
    kappa = mat.kappa
    tr = eps_xx + eps_yy  # plane strain: 3D trace = 2D trace
    expected_normal = kappa * tr
    torch.testing.assert_close(sxx, expected_normal)
    torch.testing.assert_close(syy, expected_normal)
    # Shear should also vanish (symmetric strain, no shear)
    assert abs(sxy.item()) < 1e-12

    # Magnitude: |sxx| should be ≫ eta * full_elastic — the volumetric is intact.
    full_elastic_mag = abs((mat.lam * tr + 2.0 * mat.mu * eps_xx).item())
    assert abs(sxx.item()) > 0.1 * full_elastic_mag, (
        "Compression branch failed to preserve volumetric stiffness at d=1"
    )


def test_psi_plus_star_convex_plane_stress_uses_3d_trace_in_compression():
    mesh = _MockMesh()
    mat = Material(
        E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
        rho=7.8e-9, pf_model='AT2', energy_split='star_convex',
        plane_stress=True,
    )
    fem = FEMOperators(mesh, mat)

    c = 1.0e-3
    eps_xx = torch.tensor([-c], dtype=torch.float64)
    eps_yy = torch.tensor([-c], dtype=torch.float64)
    gam_xy = torch.tensor([0.0], dtype=torch.float64)
    psi = fem._psi_plus_star_convex((eps_xx, eps_yy, gam_xy))

    tr_2d = eps_xx + eps_yy
    ezz = -mat.nu / (1.0 - mat.nu) * tr_2d
    tr = tr_2d + ezz
    dev_xx = eps_xx - tr / 3.0
    dev_yy = eps_yy - tr / 3.0
    dev_zz = ezz - tr / 3.0
    expected = mat.mu * (dev_xx ** 2 + dev_yy ** 2 + dev_zz ** 2)
    torch.testing.assert_close(psi, expected)
