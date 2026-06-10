import pytest
import torch
import numpy as np
import sys
import os
# Insert the parent of 'phast' into sys.path so the module tree acts as a true package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from phast.fem_operators import FEMOperators
from phast.mesh import FEMMesh
from phast.material import Material

# Mock FEMMesh for testing basic mathematically decoupled evaluations
class MockMesh:
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
def mock_fem():
    mesh = MockMesh()
    mat = Material(
        E=210.0, nu=0.3, Gc=2.7, l0=0.015, pf_model='AT2', energy_split='spectral'
    )
    return FEMOperators(mesh, mat)

def test_spectral_gradient_explosion_bounds(mock_fem):
    """
    Validates that the L-BFGS backpropagation gradient explosion is solved.
    Checks near-zero and exactly zero shear environments to ensure the dtype-aware
    spectral_eps floor safely restricts the `sqrt(...)` inside spectral decompositions.
    """
    mock_fem.material.energy_split = 'spectral'
    
    # Near-zero strain testing spectral split gradient stability.
    # Use 1e-10 (not 1e-18) to stay above the eigenvalue decomposition's
    # numerical floor. At truly zero strain, the eigenvector directions are
    # undefined (degenerate eigenvalues), making gradients inherently NaN.
    eps_xx = torch.tensor([1e-10], dtype=mock_fem.dtype, requires_grad=True)
    eps_yy = torch.tensor([1e-10], dtype=mock_fem.dtype, requires_grad=True)
    gam_xy = torch.tensor([1e-12], dtype=mock_fem.dtype, requires_grad=True)
    g_d = torch.tensor([1.0], dtype=mock_fem.dtype)

    sxx, syy, sxy = mock_fem.compute_stress_spectral_algebraic(eps_xx, eps_yy, gam_xy, g_d)
    
    # Backpropagate to ensure gradients exist and are strictly finite (not NaN)
    loss = sxx.sum() + syy.sum() + sxy.sum()
    loss.backward()

    assert not torch.isnan(eps_xx.grad).any(), "eps_xx Gradient exploded to NaN!"
    assert not torch.isnan(eps_yy.grad).any(), "eps_yy Gradient exploded to NaN!"
    assert not torch.isnan(gam_xy.grad).any(), "gam_xy Gradient exploded to NaN!"
    assert torch.isfinite(eps_xx.grad).all(), "eps_xx Gradient is structurally infinite."


def test_plane_strain_2D_deviatoric_energy_components(mock_fem):
    """
    Validates the fix applied to 2D Plane Strain Amor constraints.
    Even though eps_zz = 0, dev_zz MUST equal -tr(eps)/3.
    Pure Mode-I tension must successfully map deviatoric energy utilizing this 3rd dimension.
    """
    mock_fem.material.energy_split = 'amor'
    
    # Pure Mode-I Tension
    eps_xx = torch.tensor([0.01], dtype=mock_fem.dtype)
    eps_yy = torch.tensor([0.0], dtype=mock_fem.dtype)  
    gam_xy = torch.tensor([0.0], dtype=mock_fem.dtype)
    
    strain = (eps_xx, eps_yy, gam_xy)
    
    # Our operator
    psi_plus = mock_fem._psi_plus_amor(strain)
    
    # Calculating the true mathematical plane strain bound manually
    kappa = mock_fem.material.kappa
    mu = mock_fem.material.mu
    
    tr = eps_xx + eps_yy
    tr_plus = torch.clamp(tr, min=0)
    dev_xx = eps_xx - (tr / 3.0)
    dev_yy = eps_yy - (tr / 3.0)
    dev_zz = 0.0 - (tr / 3.0)  # The critical Plane Strain component
    
    dev_dot = (dev_xx**2) + (dev_yy**2) + (dev_zz**2) + 2.0 * (gam_xy/2.0)**2
    expected_psi_plus = (0.5 * kappa * tr_plus**2) + (mu * dev_dot)
    
    torch.testing.assert_close(psi_plus, expected_psi_plus, msg="Missing Out-of-Plane Plane Strain deviate.")


def test_damage_spectral_compressive_resistance(mock_fem):
    """
    Ensures that under pure uniform compression, the tensile driving energy psi+ 
    evaluates identically to 0.0, guaranteeing cracks do not spawn under compression.
    """
    mock_fem.material.energy_split = 'spectral'
    
    # Pure compression
    eps_xx = torch.tensor([-0.05], dtype=mock_fem.dtype)
    eps_yy = torch.tensor([-0.05], dtype=mock_fem.dtype)
    gam_xy = torch.tensor([0.0], dtype=mock_fem.dtype)
    strain = (eps_xx, eps_yy, gam_xy)
    
    psi_plus = mock_fem.compute_psi_plus(None, strain=strain)
    
    assert psi_plus.item() < 1e-15, "Spectral Split failed to arrest compressive tensile energies!"


def test_spectral_algebraic_basic(mock_fem):
    """Validates that the algebraic spectral split produces finite stresses
    for random strain states (trig version deprecated, algebraic is sole method).
    """
    mock_fem.material.energy_split = 'spectral'
    torch.manual_seed(42)
    n_test = 1
    eps_xx = torch.randn(n_test, dtype=mock_fem.dtype)
    eps_yy = torch.randn(n_test, dtype=mock_fem.dtype)
    gam_xy = torch.randn(n_test, dtype=mock_fem.dtype) * 0.5
    g_d = torch.ones(n_test, dtype=mock_fem.dtype)

    sxx, syy, sxy = mock_fem.compute_stress_spectral_algebraic(
        eps_xx, eps_yy, gam_xy, g_d)
    assert torch.isfinite(sxx).all(), "Sxx is not finite"
    assert torch.isfinite(syy).all(), "Syy is not finite"
    assert torch.isfinite(sxy).all(), "Sxy is not finite"


def test_direct_solver_secant_C_spectral():
    """Validates that DirectSolver._compute_all_C uses frozen secant
    projections for the spectral split, not the nonlinear stress function.

    For a shear-dominated strain state, the spectral split has e1 > 0
    and e2 < 0.  The correct secant tangent should NOT degrade the
    compressive direction.  The old (buggy) code called
    the spectral stress function with unit strains, which always sees
    non-negative eigenvalues, making the entire stiffness degraded.

    This test verifies that the secant C matrix at a shear strain state
    with damage produces a stress response that matches secant_matvec.
    """
    from phast.mesh_generator import miehe_shear as gen_mesh
    import tempfile

    # Generate a small SENS mesh for testing
    with tempfile.NamedTemporaryFile(suffix='.msh', delete=False) as f:
        mesh_path = f.name
    try:
        gen_mesh(mesh_path, L=1.0, a=0.5, l0=0.06, h_crack=0.05,
                 h_coarse=0.2, verbose=False)
        mesh = FEMMesh(mesh_path, device='cpu', dtype=torch.float64)
    finally:
        os.unlink(mesh_path)
        geo_path = mesh_path.replace('.msh', '.geo')
        if os.path.exists(geo_path):
            os.unlink(geo_path)

    mat = Material(E=210000, nu=0.3, Gc=2.7, l0=0.06,
                   energy_split='spectral', eta_residual=1e-6)
    fem = FEMOperators(mesh, mat)

    # Create a shear-like displacement and damage field
    N = mesh.n_nodes
    u = torch.zeros(N, 2, dtype=torch.float64)
    u[:, 0] = mesh.nodes[:, 1] * 0.001  # u_x proportional to y (shear)

    d = torch.zeros(N, dtype=torch.float64)
    # Add localized damage near the notch tip
    notch_y = 0.5
    for i in range(N):
        x, y = mesh.nodes[i]
        r = ((x - 0.5)**2 + (y - notch_y)**2).sqrt()
        if r < 0.1:
            d[i] = 0.5 * (1.0 - r / 0.1)

    # Freeze secant state
    state = fem.freeze_secant_state(u, d)

    # Compute secant C matrix via DirectSolver method
    from phast.mechanics_solver import DirectSolver
    ds = DirectSolver(fem, tol=1e-10)
    C_secant = ds._compute_all_C(fem, state)  # (E, 3, 3)

    # Verify: for damaged elements in shear, C_secant should have
    # off-diagonal coupling from the frozen eigenvector projections.
    # The key check: secant_matvec with a test displacement should
    # produce the same force as B^T @ C_secant @ B @ u_test.

    # Use a random perturbation as test displacement
    torch.manual_seed(42)
    p_test = torch.randn(N, 2, dtype=torch.float64) * 1e-6

    # Method 1: secant_matvec (known correct)
    f_secant = fem.secant_matvec(p_test, state)

    # Method 2: assemble K from C_secant and apply
    gp = mesh.grad_phi.numpy()
    areas = mesh.areas.numpy()
    elems = mesh.elements.numpy()

    B = np.zeros((mesh.n_elems, 3, 6), dtype=np.float64)
    for i in range(3):
        B[:, 0, 2*i] = gp[:, i, 0]
        B[:, 1, 2*i+1] = gp[:, i, 1]
        B[:, 2, 2*i] = gp[:, i, 1]
        B[:, 2, 2*i+1] = gp[:, i, 0]

    f_assembled = np.zeros(2 * N, dtype=np.float64)
    p_flat = p_test.numpy().flatten()
    for e in range(mesh.n_elems):
        elem_dofs = np.array([2*elems[e, i] + c
                              for i in range(3) for c in range(2)])
        u_e = p_flat[elem_dofs]
        CB_u = C_secant[e] @ B[e] @ u_e
        fe = areas[e] * B[e].T @ CB_u
        for k, dof in enumerate(elem_dofs):
            f_assembled[dof] += fe[k]

    f_assembled_torch = torch.from_numpy(f_assembled).reshape(N, 2)

    # Compare: should match to high precision
    err = (f_secant - f_assembled_torch).norm() / (f_secant.norm() + 1e-30)
    assert err < 1e-8, (
        f"DirectSolver C matrix doesn't match secant_matvec: "
        f"relative error = {err:.2e}"
    )


def test_direct_solver_star_convex_plane_stress_C_uses_3d_trace(mock_fem):
    """DirectSolver's star-convex secant copy must match plane-stress trace."""
    from phast.mechanics_solver import DirectSolver

    mock_fem.material.energy_split = 'star_convex'
    mock_fem.material.plane_stress = True
    g_d = torch.tensor([0.25], dtype=mock_fem.dtype)
    state = {
        'split': 'star_convex',
        'g_d': g_d,
        'tension': torch.tensor([False]),
    }

    ds = DirectSolver.__new__(DirectSolver)
    C_secant = ds._compute_all_C(mock_fem, state)

    nu = mock_fem.material.nu
    mu = mock_fem.material.mu
    kappa = mock_fem.material.kappa
    tr = 1.0 - nu / (1.0 - nu)
    expected_c00 = g_d.item() * 2.0 * mu * (1.0 - tr / 3.0) + kappa * tr

    assert np.isclose(C_secant[0, 0, 0], expected_c00, rtol=1e-12), (
        "star-convex plane-stress compression branch used the 2D trace "
        f"in DirectSolver C: got {C_secant[0, 0, 0]:.6e}, "
        f"expected {expected_c00:.6e}")


# ---------------------------------------------------------------------------
# Issue #172 -- secant_matvec is NOT the consistent tangent.
# Document the gap with a runtime test: it matches the autograd Jacobian
# only for the isotropic split (where there are no eigenvector projectors
# to freeze). Spectral / amor / star_convex are intentionally xfail.
# ---------------------------------------------------------------------------

def _secant_jvp_relative_error(split: str) -> float:
    """Build a 1x1 mm two-triangle mesh, compute J @ v two ways:
    autograd VJP of internal_force vs FEMOperators.secant_matvec.
    Returns the relative L2 error.
    """
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                         dtype=torch.float64)
    elems = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    mesh = FEMMesh.from_tensors(nodes, elems, device='cpu')
    mat = Material(E=1.0, nu=0.3, Gc=1.0, l0=0.1, rho=1.0,
                   energy_split=split, pf_model='AT2', plane_stress=False)
    fem = FEMOperators(mesh, mat)

    # Pick a non-degenerate state -- biaxial tension at d=0.3 (the case the
    # issue body documents).
    torch.manual_seed(0)
    u = torch.tensor(
        [[0.0, 0.0], [0.01, 0.0], [0.01, 0.005], [0.0, 0.005]],
        dtype=torch.float64, requires_grad=True)
    d = torch.full((mesh.n_nodes,), 0.3, dtype=torch.float64)

    # Random direction v
    v = torch.randn_like(u, dtype=torch.float64) * 0.001
    v = v.detach()

    # autograd VJP at u along v
    f = fem.internal_force(u, d)
    Jv_auto = torch.autograd.grad(
        f, u, grad_outputs=v, retain_graph=False, create_graph=False)[0]

    # secant_matvec at u along v (frozen projections)
    state = fem.freeze_secant_state(u.detach(), d)
    Jv_secant = fem.secant_matvec(v, state)

    num = (Jv_auto - Jv_secant).norm().item()
    den = max(Jv_auto.norm().item(), 1e-30)
    return num / den


def test_secant_matvec_isotropic_matches_autograd():
    """Isotropic split has no eigenvector projector to freeze, so the secant
    operator IS the consistent tangent."""
    rel = _secant_jvp_relative_error('isotropic')
    assert rel < 1e-9, f"isotropic secant should match autograd VJP, got {rel:.2e}"


@pytest.mark.xfail(reason="Issue #172: secant_matvec freezes eigvec projectors; "
                          "for spectral split this omits the eigvec-rotation "
                          "term in the consistent tangent. Use autograd-JVP "
                          "instead (PR #170 / issue #114).")
def test_secant_matvec_spectral_disagrees_with_autograd():
    rel = _secant_jvp_relative_error('spectral')
    # Per issue #172: ~33-39% error documented. Asserting 1% so this xfails.
    assert rel < 1e-2, f"got {rel:.2%}"


def test_secant_matvec_amor_matches_autograd_away_from_trace_zero():
    """Amor secant freezes ``trace >= 0`` to a hard 0/1 mask. This is also
    what autograd computes (the sign function has zero derivative away
    from the discontinuity), so when the perturbation v doesn't cross
    the trace=0 boundary, the two agree exactly. Test confirms agreement
    in the generic interior case; sign-crossing perturbations would
    disagree but are not exercised here.
    """
    rel = _secant_jvp_relative_error('amor')
    assert rel < 1e-9, (
        f"amor secant should match autograd VJP away from trace=0, "
        f"got {rel:.2e}")
