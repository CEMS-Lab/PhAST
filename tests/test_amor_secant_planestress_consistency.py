"""Regression test for issue #222 (partial fix in fem_operators.py).

The Amor secant_matvec in ``FEMOperators`` previously used the 2D trace
``eps_xx + eps_yy`` while the residual ``compute_stress_amor`` used the
full 3D trace (with plane-stress reconstruction
``eps_zz = -nu/(1-nu)*(eps_xx+eps_yy)``). This made the volumetric
coefficient of the matvec off by ``(1-2nu)/(1-nu)`` under plane stress.
Plane-strain was unaffected (eps_zz = 0).

This file locks in:

1. Plane-stress secant_matvec now matches the autograd VJP of
   ``compute_stress_amor``-driven ``internal_force`` to ~1e-9.
2. Plane-strain secant_matvec is bit-identical to the pre-fix path
   (sanity check that the new branch is dormant when eps_zz = 0).
3. ``freeze_secant_state`` and ``secant_matvec`` produce the same
   trace value at the same state — no internal divergence.
"""

import torch

from phast.material import Material
from phast.mesh import FEMMesh
from phast.fem_operators import FEMOperators


def _two_tri_unit_square():
    nodes = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    elems = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    mesh = FEMMesh.from_tensors(nodes, elems, device='cpu')
    return mesh


def _amor_jvp_relative_error(plane_stress: bool) -> float:
    """Compute J @ v two ways: autograd VJP of internal_force vs
    secant_matvec. Picks a non-trivial uniform tensile state with tr > 0
    so the volumetric branch is exercised.
    """
    mesh = _two_tri_unit_square()
    mat = Material(
        E=1.0, nu=0.3, Gc=1.0, l0=0.1, rho=1.0,
        energy_split='amor', pf_model='AT2', plane_stress=plane_stress,
    )
    fem = FEMOperators(mesh, mat)

    # Biaxial tension at d=0.3 — ensures tr_3d > 0 for both modes.
    coords = mesh.nodes
    e0 = 1e-2
    u = torch.zeros_like(coords, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        u[:, 0] = e0 * coords[:, 0]
        u[:, 1] = 0.5 * e0 * coords[:, 1]
    u.requires_grad_(True)
    d = torch.full((mesh.n_nodes,), 0.3, dtype=torch.float64)

    torch.manual_seed(0)
    v = (torch.randn_like(u) * 1e-4).detach()

    f = fem.internal_force(u, d)
    Jv_auto = torch.autograd.grad(
        f, u, grad_outputs=v, retain_graph=False, create_graph=False)[0]

    state = fem.freeze_secant_state(u.detach(), d)
    Jv_secant = fem.secant_matvec(v, state)

    num = (Jv_auto - Jv_secant).norm().item()
    den = max(Jv_auto.norm().item(), 1e-30)
    return num / den


def test_amor_secant_matvec_plane_stress_matches_autograd():
    """Plane-stress matvec must agree with the autograd VJP of the
    residual to ~1e-9. Pre-fix this would fail by the
    (1-2nu)/(1-nu) factor on the volumetric component."""
    rel = _amor_jvp_relative_error(plane_stress=True)
    assert rel < 1e-9, (
        f"plane-stress amor secant_matvec disagrees with autograd VJP: "
        f"got rel={rel:.3e}. Likely cause: 2D vs 3D trace inconsistency "
        f"(issue #222)."
    )


def test_amor_secant_matvec_plane_strain_matches_autograd():
    """Plane-strain path: tr_3d == tr_2d (eps_zz=0), so the new
    branch must be a no-op and the matvec stays bit-identical to the
    pre-fix behaviour. Mirrors the existing
    ``test_secant_matvec_amor_matches_autograd_away_from_trace_zero``
    in tests/test_fem_math.py."""
    rel = _amor_jvp_relative_error(plane_stress=False)
    assert rel < 1e-9, f"plane-strain amor diverged from autograd: {rel:.3e}"


def test_amor_secant_freeze_and_matvec_use_same_trace():
    """Lock-in: ``freeze_secant_state`` and ``secant_matvec`` must build
    the **same** trace value at the same state. Catches a future drift
    where one site is updated but the other is not."""
    mesh = _two_tri_unit_square()
    mat = Material(
        E=210e3, nu=0.3, Gc=2.7, l0=0.05, rho=7.85e-9,
        energy_split='amor', pf_model='AT2', plane_stress=True,
    )
    fem = FEMOperators(mesh, mat)

    coords = mesh.nodes
    u = torch.zeros_like(coords, dtype=torch.float64)
    u[:, 0] = 1e-3 * coords[:, 0]
    u[:, 1] = 5e-4 * coords[:, 1]

    eps_xx, eps_yy, _ = fem.compute_strain(u)
    nu = mat.nu
    tr_2d = eps_xx + eps_yy
    tr_3d_expected = tr_2d + (-nu / (1.0 - nu) * tr_2d)

    # Verify freeze trace sign matches what compute_stress_amor would yield.
    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    state = fem.freeze_secant_state(u, d)
    sign_3d_expected = (tr_3d_expected >= 0).to(torch.float64)
    assert torch.equal(state['trace_pos'], sign_3d_expected), (
        "freeze_secant_state used the wrong trace for plane stress."
    )

    # And verify matvec on p=u reproduces the residual (linear consistency
    # check: at d=0 in tension, the secant path should equal the residual).
    f_residual = fem.internal_force(u, d)
    f_matvec = fem.secant_matvec(u, state)
    rel = (f_residual - f_matvec).norm().item() / (f_residual.norm().item() + 1e-30)
    assert rel < 1e-12, (
        f"At d=0 (g=1), residual and secant matvec must agree exactly "
        f"for amor (linear-in-u in tension). Got rel={rel:.3e} — "
        f"trace mismatch between freeze and matvec sites?"
    )


def test_amor_secant_planestrain_bit_identical_to_legacy():
    """Plane-strain: tr_3d == tr_2d. The fix must be bit-identical to a
    hand-rolled legacy 2D-trace matvec for plane strain (regression
    against accidentally changing plane-strain numerics)."""
    mesh = _two_tri_unit_square()
    mat = Material(
        E=210e3, nu=0.3, Gc=2.7, l0=0.05, rho=7.85e-9,
        energy_split='amor', pf_model='AT2', plane_stress=False,
    )
    fem = FEMOperators(mesh, mat)

    coords = mesh.nodes
    u = torch.zeros_like(coords, dtype=torch.float64)
    u[:, 0] = 1e-3 * coords[:, 0]
    u[:, 1] = 5e-4 * coords[:, 1]
    d = torch.full((mesh.n_nodes,), 0.2, dtype=torch.float64)

    torch.manual_seed(1)
    p = (torch.randn_like(u) * 1e-4).detach()

    state = fem.freeze_secant_state(u, d)
    Ap_new = fem.secant_matvec(p, state)

    # Legacy (pre-fix) path: 2D trace explicitly.
    _, mu, kappa = fem._resolve_lame()
    eps_xx, eps_yy, gam_xy = fem.compute_strain(p)
    tr = eps_xx + eps_yy  # legacy 2D trace
    g_d = state['g_d']
    tr_pos = state['trace_pos']
    tr_plus = tr * tr_pos
    tr_minus = tr * (1.0 - tr_pos)
    dev_xx = eps_xx - tr / 3.0
    dev_yy = eps_yy - tr / 3.0
    sxx = g_d * (kappa * tr_plus + 2 * mu * dev_xx) + kappa * tr_minus
    syy = g_d * (kappa * tr_plus + 2 * mu * dev_yy) + kappa * tr_minus
    sxy = g_d * mu * gam_xy
    Ap_legacy = fem._assemble_force(sxx, syy, sxy)

    diff = (Ap_new - Ap_legacy).abs().max().item()
    assert diff == 0.0, (
        f"Plane-strain matvec changed by {diff:.3e} — must be bit-identical "
        f"to pre-fix path (eps_zz = 0)."
    )
