"""
Unit tests for the stress-spectral energy split (Miehe et al. 2010 §3.2).

Opt-in via ``energy_split='spectral_stress'`` (issue #213). Covers:

  1. At ``d=0`` total stress matches un-decomposed σ = λ·tr(ε)·I + 2μ·ε.
  2. Pure tension: σ⁺ ≈ full stress; σ⁻ ≈ 0; with d=1 total stress ≈ 0.
  3. Pure compression: σ⁻ ≈ full stress; σ⁺ ≈ 0; with d=1 total stress
     unchanged from undegraded (compression branch preserved).
  4. Strain-spectral vs stress-spectral DIFFER on a mixed-sign principal
     state (key empirical claim — Yu et al. 2021).

All tests run on CPU float64 with a tiny single-element-equivalent
strain vector (no mesh needed for the algebraic kernel).
"""

import math

import pytest
import torch


# ---------------------------------------------------------------------- #
# Lightweight FEMOperators stub so we can call the algebraic kernels
# without instantiating a mesh. The kernels only use:
#   self.material  (for E, nu, lam, mu)
#   self._spectral_eps  (regularisation floor)
# ---------------------------------------------------------------------- #


class _FEMStub:
    def __init__(self, material, dtype=torch.float64):
        self.material = material
        self.dtype = dtype
        self._spectral_eps = 1e-12

    # Bind the real implementations
    from phast.fem_operators import FEMOperators
    compute_stress_spectral_stress = FEMOperators.compute_stress_spectral_stress
    compute_stress_spectral_algebraic = FEMOperators.compute_stress_spectral_algebraic
    _psi_plus_spectral_stress = FEMOperators._psi_plus_spectral_stress
    _psi_plus_spectral = FEMOperators._psi_plus_spectral
    # The spectral kernels call this classmethod when plane_stress=True
    # to emit a one-shot warning. Re-bind from FEMOperators so the stub
    # stays in lock-step with the real implementation.
    _maybe_warn_plane_stress_spectral = FEMOperators._maybe_warn_plane_stress_spectral
    _plane_stress_spectral_warning_emitted = False


@pytest.fixture
def material_pstress():
    """Plane-stress material so the Hooke compliance form is exact."""
    from phast.material import Material
    return Material(
        E=210e3, nu=0.30, Gc=2.7, l0=0.05, rho=7.85e-9,
        energy_split='spectral_stress', plane_stress=True, pf_model='AT2',
    )


@pytest.fixture
def fem_stub(material_pstress):
    return _FEMStub(material_pstress)


# ---------------------------------------------------------------------- #
# Test 1 — at d=0, recover full elastic stress
# ---------------------------------------------------------------------- #
def test_stress_spectral_stress_d0_matches_linear_elastic(fem_stub):
    """At d=0, σ = g(0)·σ⁺ + σ⁻ = σ⁺ + σ⁻ = σ_full (since g(0)=1)."""
    fem = fem_stub
    mat = fem.material
    lam, mu = mat.lam, mat.mu

    eps_xx = torch.tensor([7e-4, -3e-4, 1e-4], dtype=torch.float64)
    eps_yy = torch.tensor([2e-4, -1e-4, -5e-4], dtype=torch.float64)
    gam_xy = torch.tensor([3e-4, 5e-4, -2e-4], dtype=torch.float64)
    g_d = torch.ones_like(eps_xx)  # d=0 ⇒ g(d)=1

    sxx, syy, sxy = fem.compute_stress_spectral_stress(eps_xx, eps_yy, gam_xy, g_d)

    # Reference: σ = λ·tr·I + 2μ·ε
    tr = eps_xx + eps_yy
    sxx_ref = lam * tr + 2.0 * mu * eps_xx
    syy_ref = lam * tr + 2.0 * mu * eps_yy
    sxy_ref = mu * gam_xy  # 2μ·ε_xy = 2μ·(γ/2) = μ·γ

    assert torch.allclose(sxx, sxx_ref, atol=1e-9)
    assert torch.allclose(syy, syy_ref, atol=1e-9)
    assert torch.allclose(sxy, sxy_ref, atol=1e-9)


# ---------------------------------------------------------------------- #
# Test 2 — pure tension: σ⁺ ≈ full, σ⁻ ≈ 0; d=1 ⇒ total ≈ 0
# ---------------------------------------------------------------------- #
def test_stress_spectral_stress_pure_tension(fem_stub):
    """Pure tension state with both principal stresses positive.

    Use uniaxial extension under plane stress: ε_xx = e0, ε_yy = -ν·e0,
    γ_xy = 0 ⇒ σ_yy ≈ 0, σ_xx ≈ E·e0. Both σ_1 = σ_xx > 0, σ_2 = σ_yy ~ 0.
    """
    fem = fem_stub
    mat = fem.material
    e0 = 1e-3
    eps_xx = torch.tensor([e0], dtype=torch.float64)
    eps_yy = torch.tensor([-mat.nu * e0], dtype=torch.float64)
    gam_xy = torch.tensor([0.0], dtype=torch.float64)

    # d=0: full stress. σ_xx ≈ E·e0, σ_yy ≈ 0.
    g_d0 = torch.ones_like(eps_xx)
    sxx0, syy0, sxy0 = fem.compute_stress_spectral_stress(eps_xx, eps_yy, gam_xy, g_d0)
    assert sxx0.item() == pytest.approx(mat.E * e0, rel=1e-6)
    assert abs(syy0.item()) < 1e-6 * mat.E * e0
    assert abs(sxy0.item()) < 1e-12

    # d=1: g(d)≈eta; total stress ≈ 0 (pure tension is fully degraded).
    g_d1 = torch.full_like(eps_xx, mat.eta_residual)
    sxx1, syy1, sxy1 = fem.compute_stress_spectral_stress(eps_xx, eps_yy, gam_xy, g_d1)
    # Tolerance: σ_xx0 * eta_residual.
    bound = mat.E * e0 * mat.eta_residual * 10
    assert abs(sxx1.item()) < bound
    assert abs(syy1.item()) < bound


# ---------------------------------------------------------------------- #
# Test 3 — pure compression: σ⁻ ≈ full, σ⁺ ≈ 0; d=1 ⇒ total ≈ undegraded
# ---------------------------------------------------------------------- #
def test_stress_spectral_stress_pure_compression(fem_stub):
    """Pure compression: ε_xx = -e0, ε_yy = ν·e0 ⇒ σ_xx ≈ -E·e0, σ_yy ≈ 0.

    σ_1 = max ≈ 0, σ_2 = min ≈ -E·e0. Both σ_i⁺ = 0 ⇒ σ⁺ = 0;
    σ⁻ = full undegraded stress; total stress with d=1 still equals σ⁻.
    """
    fem = fem_stub
    mat = fem.material
    e0 = 1e-3
    eps_xx = torch.tensor([-e0], dtype=torch.float64)
    eps_yy = torch.tensor([mat.nu * e0], dtype=torch.float64)
    gam_xy = torch.tensor([0.0], dtype=torch.float64)

    # Reference (linear elastic).
    lam, mu = mat.lam, mat.mu
    tr = eps_xx + eps_yy
    sxx_full = lam * tr + 2.0 * mu * eps_xx
    syy_full = lam * tr + 2.0 * mu * eps_yy

    # d=1: should still equal the full undegraded stress (compression preserved).
    g_d1 = torch.full_like(eps_xx, mat.eta_residual)
    sxx1, syy1, sxy1 = fem.compute_stress_spectral_stress(eps_xx, eps_yy, gam_xy, g_d1)
    assert sxx1.item() == pytest.approx(sxx_full.item(), rel=1e-4)
    assert abs(syy1.item() - syy_full.item()) < 1e-3 * abs(sxx_full.item())


# ---------------------------------------------------------------------- #
# Test 4 — strain-spectral vs stress-spectral DIFFER on mixed-sign state
# ---------------------------------------------------------------------- #
def test_strain_vs_stress_spectral_differ_mixed_sign(fem_stub):
    """Construct a strain state where the two splits disagree.

    Pick ε_xx = +e0, ε_yy = -1.5·e0, γ_xy = 0, ν = 0.3. Then:
      Principal strains: e_1 = e0,    e_2 = -1.5 e0.
      Trace = -0.5 e0 (negative).
      Principal stresses (plane stress, λ_ps = E·ν/(1-ν²)):
        σ_1 = λ_ps·tr + 2μ·e_1 = λ_ps·(-0.5 e0) + 2μ·e0     ← positive
        σ_2 = λ_ps·tr + 2μ·e_2 = λ_ps·(-0.5 e0) + 2μ·(-1.5 e0)  ← negative

    Strain-spectral degrades: λ·tr⁺ + 2μ·e_1⁺ → tr⁺ = 0, only e_1⁺ = e0
      contributes ⇒ ε_xx_plus comes entirely from e_1·n_1⊗n_1, very
      different geometry from σ⁺ which lumps the full λ·tr piece into σ_1.
    Numerically the two stress fields differ by O(10%) on this state.
    """
    fem = fem_stub
    e0 = 1e-3
    eps_xx = torch.tensor([e0], dtype=torch.float64)
    eps_yy = torch.tensor([-1.5 * e0], dtype=torch.float64)
    gam_xy = torch.tensor([0.0], dtype=torch.float64)
    g_d = torch.full_like(eps_xx, 0.25)  # mid-damage so both branches active

    s_strain = fem.compute_stress_spectral_algebraic(eps_xx, eps_yy, gam_xy, g_d)
    s_stress = fem.compute_stress_spectral_stress(eps_xx, eps_yy, gam_xy, g_d)

    # Compare component-wise relative difference.
    diffs = []
    for a, b in zip(s_strain, s_stress):
        denom = max(abs(a.item()), abs(b.item()), 1e-30)
        diffs.append(abs(a.item() - b.item()) / denom)

    max_rel = max(diffs)
    # Expect a real, non-trivial difference (Yu 2021 reports 10-15% on
    # full forward simulations; on a single mixed-sign element the
    # *stress component* difference is typically O(1-30%)).
    assert max_rel > 1e-3, (
        f"strain-spectral and stress-spectral produced near-identical "
        f"stresses on a mixed-sign state (max_rel={max_rel:.3e}); "
        f"the two splits should differ. Components: "
        f"strain={tuple(x.item() for x in s_strain)}, "
        f"stress={tuple(x.item() for x in s_stress)}"
    )

    # Sanity: psi+ scalars also differ (they use the compliance form).
    strain = (eps_xx, eps_yy, gam_xy)
    psi_strain = fem._psi_plus_spectral(strain)
    psi_stress = fem._psi_plus_spectral_stress(strain)
    assert (psi_strain - psi_stress).abs().max().item() > 1e-12


# ---------------------------------------------------------------------- #
# Bonus: forward-parity table — print stresses side-by-side at the
# same strain. Exercises both kernels end-to-end and prints a small
# table for the PR body. Always passes.
# ---------------------------------------------------------------------- #
def test_forward_parity_table(fem_stub, capsys):
    fem = fem_stub
    e0 = 1e-3
    states = {
        'pure_tension':  (e0, -fem.material.nu * e0, 0.0),
        'pure_compress': (-e0, fem.material.nu * e0, 0.0),
        'mixed_sign':    (e0, -1.5 * e0, 0.0),
        'shear':         (0.0, 0.0, e0),
        'biaxial_tens':  (e0, e0, 0.0),
    }
    g_d = torch.tensor([0.25], dtype=torch.float64)
    print("\nstate           | strain_spectral (sxx, syy, sxy)  | stress_spectral (sxx, syy, sxy)")
    print("-" * 100)
    for name, (a, b, c) in states.items():
        ex = torch.tensor([a], dtype=torch.float64)
        ey = torch.tensor([b], dtype=torch.float64)
        gx = torch.tensor([c], dtype=torch.float64)
        sa = fem.compute_stress_spectral_algebraic(ex, ey, gx, g_d)
        sb = fem.compute_stress_spectral_stress(ex, ey, gx, g_d)
        print(f"{name:15s} | "
              f"({sa[0].item():+.3e}, {sa[1].item():+.3e}, {sa[2].item():+.3e}) | "
              f"({sb[0].item():+.3e}, {sb[1].item():+.3e}, {sb[2].item():+.3e})")
    out = capsys.readouterr().out
    assert 'strain_spectral' in out
