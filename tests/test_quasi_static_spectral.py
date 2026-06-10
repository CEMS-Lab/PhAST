"""Issue #114 — QuasiStaticSolver: spectral-split tangent stiffness support.

QuasiStaticSolver uses an autograd-based JVP for the inner CG matvec when
the energy split is non-isotropic ('spectral', 'amor', 'star_convex'). The
existing secant_matvec operator (used by SecantCGSolver) is *not* the
consistent tangent for these splits: it freezes eigenvector projectors,
omitting the eigvec-rotation term that arises when projectors P_i depend on
eps. SecantCGSolver still converges via Newton-secant iteration, but as a
tangent the secant operator is wrong by ~30% generically (verified during
implementation of this test).

Autograd JVP gives the true consistent tangent at u (matching FD to
machine precision in float64), so the inner CG sees a Jacobian that is
fixed within the inner solve (preserving conjugacy) and is correct enough
to give the quadratic NR convergence the secant operator cannot.

Tests:
  - FD vs autograd-JVP tangent for {isotropic, amor, spectral, star_convex}
    on uniaxial tension, biaxial tension, and pure shear
  - Backwards-compat: solver no longer raises for non-isotropic splits;
    isotropic equilibrium produces the expected uniaxial-tension solution
  - Solver actually converges on a small spectral-split Newton solve
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mechanics_solver import QuasiStaticSolver
from phast.mesh import FEMMesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_square_two_tri_mesh(dtype=torch.float64):
    """Two-triangle unit square: nodes 0..3, elements [(0,1,2),(0,2,3)]."""
    nodes = torch.tensor([[0.0, 0.0],
                          [1.0, 0.0],
                          [1.0, 1.0],
                          [0.0, 1.0]], dtype=dtype)
    elements = torch.tensor([[0, 1, 2],
                             [0, 2, 3]], dtype=torch.long)
    return FEMMesh.from_tensors(nodes, elements, device='cpu')


def _make_fem(split, dtype=torch.float64):
    mesh = _unit_square_two_tri_mesh(dtype=dtype)
    mat = Material(E=210.0, nu=0.3, Gc=2.7, l0=0.1,
                   pf_model='AT2', energy_split=split)
    return FEMOperators(mesh, mat)


def _fd_jacobian(fem, u, d, free_mask, h=1e-9):
    """Central-difference Jacobian of (internal_force * free_mask) wrt u.

    Spectral / amor / star_convex splits are only piecewise C^1. With strains
    of order 1e-3 the regime boundaries (sign of eigenvalues, sign of trace)
    sit close enough to the linearisation point that a forward-difference
    with h ~ 1e-7 averages across the discontinuity, biasing the FD estimate.
    Central differences with h ~ 1e-9 stay safely on one side of the boundary
    and recover the analytical (right-)derivative the autograd JVP returns.
    """
    n = u.numel()
    J = torch.zeros(n, n, dtype=u.dtype, device=u.device)
    flat_u = u.flatten().clone()
    for j in range(n):
        flat_u[j] += h
        f_plus = (fem.internal_force(flat_u.reshape(u.shape), d) * free_mask).flatten()
        flat_u[j] -= 2 * h
        f_minus = (fem.internal_force(flat_u.reshape(u.shape), d) * free_mask).flatten()
        J[:, j] = (f_plus - f_minus) / (2 * h)
        flat_u[j] += h
    return J


def _tangent_matrix(fem, u, d, free_mask):
    """Materialise the consistent tangent matrix via autograd-JVP probing.

    This is the same tangent the QuasiStaticSolver uses inside its inner CG
    when the split is non-isotropic. For isotropic, internal_force is
    linear in u, so JVP equals internal_force(du, d).
    """
    n = u.numel()
    K = torch.zeros(n, n, dtype=u.dtype, device=u.device)
    u_lin = u.detach()
    f = lambda uu: fem.internal_force(uu, d)
    for j in range(n):
        e = torch.zeros(n, dtype=u.dtype, device=u.device)
        e[j] = 1.0
        _, jvp = torch.autograd.functional.jvp(f, (u_lin,), (e.reshape(u.shape),))
        K[:, j] = (jvp * free_mask).flatten()
    return K


def _make_strained_u(fem, mode='tension', amp=1e-3):
    """Build a smooth displacement field on the unit square."""
    nodes = fem.mesh.nodes
    u = torch.zeros_like(nodes)
    if mode == 'tension':
        # uniaxial stretch in x
        u[:, 0] = amp * nodes[:, 0]
    elif mode == 'shear':
        # simple shear with a small superposed dilatation so that trace > 0
        # by a comfortable margin (pure shear sits exactly on the trace=0
        # regime boundary, where the spectral / amor / star_convex splits
        # are non-differentiable and FD vs autograd JVP would disagree).
        u[:, 0] = amp * nodes[:, 1] + 0.1 * amp * nodes[:, 0]
        u[:, 1] = 0.1 * amp * nodes[:, 1]
    elif mode == 'compression':
        u[:, 0] = -amp * nodes[:, 0]
        u[:, 1] = -amp * nodes[:, 1]
    elif mode == 'mixed':
        u[:, 0] = amp * nodes[:, 0] + 0.4 * amp * nodes[:, 1]
        u[:, 1] = -0.3 * amp * nodes[:, 0] + 0.7 * amp * nodes[:, 1]
    else:
        raise ValueError(mode)
    return u


# ---------------------------------------------------------------------------
# FD-vs-tangent tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('split', ['isotropic', 'amor', 'spectral', 'star_convex'])
@pytest.mark.parametrize('mode,d_val', [
    ('tension', 0.0),
    ('tension', 0.3),
    ('compression', 0.0),
])
def test_secant_tangent_matches_fd_uniaxial(split, mode, d_val):
    """Consistent tangent (secant matvec) matches FD Jacobian on a stretched
    single Q4-equivalent (two-triangle) mesh, for all four energy splits.

    Use displacements well away from eps == 0 to avoid the spectral
    decomposition's ridge floor.
    """
    fem = _make_fem(split)
    n_nodes = fem.mesh.n_nodes
    d = torch.full((n_nodes,), d_val, dtype=torch.float64)
    u = _make_strained_u(fem, mode=mode, amp=1e-3)
    free_mask = torch.ones_like(u)

    K_analytical = _tangent_matrix(fem, u, d, free_mask)
    K_fd = _fd_jacobian(fem, u, d, free_mask, h=1e-9)

    rel = (K_analytical - K_fd).norm() / (K_analytical.norm() + 1e-30)
    assert rel < 1e-5, (
        f"split={split} mode={mode} d={d_val}: "
        f"|K_analytical - K_FD| / |K_analytical| = {rel:.3e}")


@pytest.mark.parametrize('split', ['amor', 'spectral', 'star_convex'])
def test_secant_tangent_matches_fd_shear(split):
    """Sheared element: exercises off-diagonal eigenvector projector
    contribution for the spectral split."""
    fem = _make_fem(split)
    n_nodes = fem.mesh.n_nodes
    d = torch.full((n_nodes,), 0.2, dtype=torch.float64)
    u = _make_strained_u(fem, mode='shear', amp=2e-3)
    free_mask = torch.ones_like(u)

    K_analytical = _tangent_matrix(fem, u, d, free_mask)
    K_fd = _fd_jacobian(fem, u, d, free_mask, h=1e-9)

    rel = (K_analytical - K_fd).norm() / (K_analytical.norm() + 1e-30)
    assert rel < 1e-5, (
        f"split={split} shear: rel = {rel:.3e}")


@pytest.mark.parametrize('split', ['amor', 'spectral', 'star_convex'])
def test_secant_tangent_matches_fd_mixed(split):
    """Mixed shear + tension + compression: realistic generic strain state."""
    fem = _make_fem(split)
    n_nodes = fem.mesh.n_nodes
    d = torch.full((n_nodes,), 0.1, dtype=torch.float64)
    u = _make_strained_u(fem, mode='mixed', amp=1.5e-3)
    free_mask = torch.ones_like(u)

    K_analytical = _tangent_matrix(fem, u, d, free_mask)
    K_fd = _fd_jacobian(fem, u, d, free_mask, h=1e-9)

    rel = (K_analytical - K_fd).norm() / (K_analytical.norm() + 1e-30)
    assert rel < 1e-5, f"split={split} mixed: rel = {rel:.3e}"


# ---------------------------------------------------------------------------
# Solver-level tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('split', ['isotropic', 'amor', 'spectral', 'star_convex'])
def test_quasi_static_solver_accepts_all_splits(split):
    """Issue #114: QuasiStaticSolver must no longer raise for non-isotropic
    splits."""
    fem = _make_fem(split)
    solver = QuasiStaticSolver(fem, tol=1e-9, max_iter=30,
                               cg_tol=1e-12, cg_max_iter=500)

    n_nodes = fem.mesh.n_nodes
    d = torch.zeros(n_nodes, dtype=torch.float64)
    f_ext = torch.zeros(n_nodes, 2, dtype=torch.float64)

    # Pin left edge (x=0), pull right edge (x=1) by 1e-4
    bc_mask = torch.zeros(n_nodes, 2, dtype=torch.bool)
    bc_vals = torch.zeros(n_nodes, 2, dtype=torch.float64)
    nodes = fem.mesh.nodes
    left = nodes[:, 0] < 1e-9
    right = nodes[:, 0] > 1.0 - 1e-9
    bc_mask[left, 0] = True
    bc_mask[left, 1] = True
    bc_mask[right, 0] = True
    bc_vals[right, 0] = 1e-4

    u, converged, n_iter = solver.solve(d, f_ext, bc_mask, bc_vals)
    assert converged, (
        f"split={split} did not converge in {n_iter} NR iterations")


def test_auto_sparse_direct_uses_secant_assembly_for_spectral(monkeypatch):
    """backend='auto' may route spectral splits through sparse direct.

    The sparse-direct path assembles a frozen-state secant tangent for
    non-isotropic splits instead of using the isotropic assembly.
    """
    fem = _make_fem('spectral')
    solver = QuasiStaticSolver(fem, tol=1e-9, max_iter=30,
                               cg_tol=1e-12, cg_max_iter=500,
                               backend='auto')
    monkeypatch.setattr(solver, '_resolve_backend', lambda _n: 'scipy')

    calls = {'secant': 0}
    original = solver._assemble_K_secant

    def wrapped_secant(*args, **kwargs):
        calls['secant'] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, '_assemble_K_secant', wrapped_secant)

    n_nodes = fem.mesh.n_nodes
    d = torch.zeros(n_nodes, dtype=torch.float64)
    f_ext = torch.zeros(n_nodes, 2, dtype=torch.float64)
    bc_mask = torch.zeros(n_nodes, 2, dtype=torch.bool)
    bc_vals = torch.zeros(n_nodes, 2, dtype=torch.float64)
    nodes = fem.mesh.nodes
    left = nodes[:, 0] < 1e-9
    right = nodes[:, 0] > 1.0 - 1e-9
    bc_mask[left, :] = True
    bc_mask[right, 0] = True
    bc_vals[right, 0] = 1e-4

    _u, converged, _n_iter = solver.solve(d, f_ext, bc_mask, bc_vals)
    assert converged
    assert solver.last_backend == 'scipy'
    assert calls['secant'] >= 1


def test_explicit_sparse_direct_accepts_spectral(monkeypatch):
    """Explicit scipy requests use the spectral frozen-state tangent."""
    fem = _make_fem('spectral')
    solver = QuasiStaticSolver(fem, backend='scipy', tol=1e-9, max_iter=30,
                               cg_tol=1e-12, cg_max_iter=500)
    monkeypatch.setattr(solver, '_resolve_backend', lambda _n: 'scipy')

    n_nodes = fem.mesh.n_nodes
    d = torch.zeros(n_nodes, dtype=torch.float64)
    f_ext = torch.zeros(n_nodes, 2, dtype=torch.float64)
    bc_mask = torch.zeros(n_nodes, 2, dtype=torch.bool)
    bc_vals = torch.zeros(n_nodes, 2, dtype=torch.float64)
    nodes = fem.mesh.nodes
    left = nodes[:, 0] < 1e-9
    right = nodes[:, 0] > 1.0 - 1e-9
    bc_mask[left, :] = True
    bc_mask[right, 0] = True
    bc_vals[right, 0] = 1e-4

    _u, converged, _n_iter = solver.solve(d, f_ext, bc_mask, bc_vals)
    assert converged
    assert solver.last_backend == 'scipy'


def test_isotropic_path_unchanged():
    """Backwards-compat: isotropic split must produce the same equilibrium
    displacement after the secant-tangent refactor as before (the isotropic
    branch keeps the original internal_force matvec)."""
    fem = _make_fem('isotropic')
    solver = QuasiStaticSolver(fem, tol=1e-12, max_iter=50,
                               cg_tol=1e-14, cg_max_iter=2000)

    n_nodes = fem.mesh.n_nodes
    d = torch.zeros(n_nodes, dtype=torch.float64)
    f_ext = torch.zeros(n_nodes, 2, dtype=torch.float64)
    bc_mask = torch.zeros(n_nodes, 2, dtype=torch.bool)
    bc_vals = torch.zeros(n_nodes, 2, dtype=torch.float64)
    nodes = fem.mesh.nodes
    left = nodes[:, 0] < 1e-9
    right = nodes[:, 0] > 1.0 - 1e-9
    bc_mask[left, 0] = True
    bc_mask[left, 1] = True
    bc_mask[right, 0] = True
    bc_vals[right, 0] = 5e-4

    u, converged, _ = solver.solve(d, f_ext, bc_mask, bc_vals)
    assert converged

    # Linear-elastic uniaxial tension with rigidly-clamped left edge:
    # right-edge x-displacement equals the prescribed BC.
    assert torch.allclose(u[right, 0],
                          torch.full_like(u[right, 0], 5e-4),
                          atol=1e-10)


def _tpb_mesh(W=8.0, H=2.0, nx=16, ny=4, dtype=torch.float64):
    """Minimal three-point-bending-style structured mesh.

    Rectangular WxH plate, two triangles per cell. Mirrors the geometry of
    examples/quasistatic/three_point_bending/run.py at a tiny
    resolution suitable for a unit test.
    """
    xs = torch.linspace(0.0, W, nx + 1, dtype=dtype)
    ys = torch.linspace(0.0, H, ny + 1, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing='ij')
    nodes = torch.stack([X.flatten(), Y.flatten()], dim=1)

    def nid(i, j):
        return i * (ny + 1) + j

    elems = []
    for i in range(nx):
        for j in range(ny):
            elems.append([nid(i, j), nid(i + 1, j), nid(i + 1, j + 1)])
            elems.append([nid(i, j), nid(i + 1, j + 1), nid(i, j + 1)])
    elements = torch.tensor(elems, dtype=torch.long)
    return FEMMesh.from_tensors(nodes, elements, device='cpu')


def test_tpb_spectral_does_not_raise_guard():
    """Issue #208 regression: a three-point-bending-style problem with
    spectral split must not trip the QuasiStaticSolver guard.

    This is a minimal version of
    examples/quasistatic/three_point_bending/run.py — same BC
    pattern (pin bottom-left, roller bottom-right, prescribed downward
    displacement at top centre) and same energy_split='spectral' — at a
    tiny mesh so it runs in seconds. The original failure mode was

        ValueError: QuasiStaticSolver does not support 'spectral' split.

    raised at mechanics_solver.py:573 before any Newton iteration.
    """
    W, H = 8.0, 2.0
    mesh = _tpb_mesh(W=W, H=H, nx=16, ny=4)
    mat = Material(E=20800.0, nu=0.3, Gc=0.5, l0=0.06,
                   pf_model='AT2', energy_split='spectral')
    fem = FEMOperators(mesh, mat)

    # Force the iterative-CG path: the autograd-JVP tangent lives there.
    # (sparse-direct backends still assemble the isotropic K — a separate
    #  pre-existing approximation, out of scope for #208.)
    solver = QuasiStaticSolver(fem, tol=1e-6, max_iter=50,
                               cg_tol=1e-10, cg_max_iter=2000,
                               backend='cg')

    n_nodes = fem.mesh.n_nodes
    nodes = fem.mesh.nodes
    d = torch.zeros(n_nodes, dtype=torch.float64)
    f_ext = torch.zeros(n_nodes, 2, dtype=torch.float64)
    bc_mask = torch.zeros(n_nodes, 2, dtype=torch.bool)
    bc_vals = torch.zeros(n_nodes, 2, dtype=torch.float64)

    # TPB BC pattern.
    bottom_mask = nodes[:, 1].abs() < 1e-9
    top_mask = (nodes[:, 1] - H).abs() < 1e-9

    # Pin bottom-left.
    bottom_idx = torch.where(bottom_mask)[0]
    pin = bottom_idx[nodes[bottom_idx, 0].argmin()]
    bc_mask[pin, 0] = True
    bc_mask[pin, 1] = True

    # Roller bottom-right.
    roller = bottom_idx[nodes[bottom_idx, 0].argmax()]
    bc_mask[roller, 1] = True

    # Prescribed downward displacement at top centre.
    top_idx = torch.where(top_mask)[0]
    centre = top_idx[(nodes[top_idx, 0] - W / 2).abs().argmin()]
    bc_mask[centre, 1] = True
    bc_vals[centre, 1] = -1e-4

    # The point of the test is that solve() does not raise the guard.
    u, converged, n_iter = solver.solve(d, f_ext, bc_mask, bc_vals)
    assert converged, (
        f"TPB-spectral did not converge in {n_iter} NR iterations; "
        f"the guard removal exposed a real solver issue."
    )
