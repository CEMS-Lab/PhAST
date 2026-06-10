"""H-refinement study for QuasiStaticSolver rotation-free rigid-connector MPC.

Issue #260 — implicit-solver finish. PR #207 / PR #215 / PR #182 each
ported the master-slave T-matrix MPC into a different solver path
(ExplicitDynamics, SecantCGSolver, DirectSolver). QuasiStaticSolver was
the remaining gap. This test parallels
``tests/test_direct_solver_cantilever_mpc.py`` (PR #294) and
``tests/test_qs_legacy_cantilever_mpc.py`` (PR #224) routing the same
cantilever problem through ``QuasiStaticSolver._solve_mpc`` for both
``inner_solver='cg'`` and ``inner_solver='direct'``.

Setup mirrors PR #224 / PR #294 verbatim:
  - Cantilever, L=10, h=1, plane stress, E=1, nu=0.
  - T3 structured grid (80x8, 160x16, 320x32) → h-doubling.
  - Two variants: 'isolated' master at (L, 0) and 'connected' master
    at the mid-height tip-face grid node.
  - rotation_free=True, master u_x pinned to 0, master u_y prescribed
    to 0.05; tip face is the slave set.
  - Analytical: K = 3 E I / L^3, I = h^3 / 12.

PASS gate: max rel_err over both variants on the finest (320x32) mesh
must be ≤ 2% — same threshold used for DirectSolver / SecantCG.

Two parametrized inner_solver values ('cg', 'direct') × 2 variants × 3
meshes = 12 cases; +1 summary test = 13 total. The task spec says ~7;
the four-column comparison demands both inner_solver values be tested,
so we report both and ASSERT only the CG path (default).
"""
import numpy as np
import pytest
import torch


def _build_grid(L, h, nx, ny):
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(-h / 2, h / 2, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = a + 1
            c = a + (nx + 1)
            d = c + 1
            elems.append([a, b, d])
            elems.append([a, d, c])
    return nodes, np.array(elems, dtype=np.int64)


def _build_cantilever_isolated(L, h, nx, ny):
    from phast.mesh import FEMMesh
    nodes, elems = _build_grid(L, h, nx, ny)
    n_grid = nodes.shape[0]
    nodes = np.concatenate([nodes, np.array([[L, 0.0]])], axis=0)
    master_node = n_grid
    nodes_t = torch.tensor(nodes, dtype=torch.float64)
    elems_t = torch.tensor(elems, dtype=torch.long)
    left_idx = np.where(np.isclose(nodes[:, 0], 0.0))[0]
    right_grid = np.where(np.isclose(nodes[:n_grid, 0], L))[0]
    node_sets = {
        'left': torch.tensor(left_idx, dtype=torch.long),
        'tip_face': torch.tensor(right_grid, dtype=torch.long),
    }
    mesh = FEMMesh.from_tensors(nodes_t, elems_t, node_sets,
                                device='cpu', dtype=torch.float64)
    return mesh, master_node, node_sets


def _build_cantilever_connected(L, h, nx, ny):
    from phast.mesh import FEMMesh
    nodes, elems = _build_grid(L, h, nx, ny)
    nodes_t = torch.tensor(nodes, dtype=torch.float64)
    elems_t = torch.tensor(elems, dtype=torch.long)
    left_idx = np.where(np.isclose(nodes[:, 0], 0.0))[0]
    right_idx = np.where(np.isclose(nodes[:, 0], L))[0]
    master_node = int(ny // 2 * (nx + 1) + nx)
    node_sets = {
        'left': torch.tensor(left_idx, dtype=torch.long),
        'tip_face': torch.tensor(right_idx, dtype=torch.long),
    }
    mesh = FEMMesh.from_tensors(nodes_t, elems_t, node_sets,
                                device='cpu', dtype=torch.float64)
    return mesh, master_node, node_sets


def _solve_qs(fem, n, bcs, rcs, inner_solver):
    from phast.mechanics_solver import QuasiStaticSolver
    solver = QuasiStaticSolver(
        fem, tol=1e-10, max_iter=3, cg_tol=1e-12, cg_max_iter=20000,
        inner_solver=inner_solver,
    )
    n_nodes = fem.mesh.n_nodes
    d = torch.zeros(n_nodes, dtype=torch.float64)
    f_ext = torch.zeros(n_nodes, 2, dtype=torch.float64)
    bc_mask, bc_vals = bcs.get_masks_and_values()
    u, conv, _ = solver.solve(d, f_ext, bc_mask, bc_vals,
                              rigid_connectors=rcs)
    assert conv, "QuasiStaticSolver.solve() did not converge"
    return u


def _run_one(variant, nx, ny, inner_solver, L=10.0, h=1.0,
             E=1.0, nu=0.0, u_prescribed=0.05):
    from phast.material import Material
    from phast.fem_operators import FEMOperators
    from phast.boundary_conditions import BoundaryConditions

    if variant == 'isolated':
        mesh, master_node, node_sets = _build_cantilever_isolated(
            L, h, nx, ny)
    elif variant == 'connected':
        mesh, master_node, node_sets = _build_cantilever_connected(
            L, h, nx, ny)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    n = mesh.n_nodes
    mat = Material(E=E, nu=nu, Gc=1.0, l0=1.0,
                   energy_split='isotropic', plane_stress=True)
    fem = FEMOperators(mesh, mat, ctx=None)
    bcs = BoundaryConditions(n, device='cpu', dtype=torch.float64)
    bcs.fix(node_sets['left'], component=0)
    bcs.fix(node_sets['left'], component=1)
    bcs.add_rigid_connector(
        master_node=master_node,
        slave_indices=node_sets['tip_face'],
        locked_components=[0, 1],
        prescribe={0: 0.0, 1: u_prescribed},
        rotation_free=True,
    )
    rcs = bcs.get_active_rigid_connectors()
    u_sol = _solve_qs(fem, n, bcs, rcs, inner_solver=inner_solver)

    f_int = fem.internal_force(u_sol, torch.zeros(n, dtype=torch.float64))
    slave_idx = node_sets['tip_face'].numpy()
    ids = np.unique(np.concatenate([np.array([master_node]), slave_idx]))
    R_y = float(f_int[ids, 1].sum().item())
    K = R_y / u_prescribed
    K_analytical = 3.0 * E * (h ** 3 / 12.0) / L ** 3
    rel = abs(K - K_analytical) / K_analytical
    return K, K_analytical, rel


_RESULTS = {}  # (inner_solver, variant, nx, ny) -> rel_err


REFINEMENTS = [
    ('coarse',  80,  8),
    ('medium', 160, 16),
    ('fine',   320, 32),
]
VARIANTS = ['isolated', 'connected']
INNER_SOLVERS = ['cg', 'direct']
CASES = [(s, ref, var)
         for s in INNER_SOLVERS
         for ref in REFINEMENTS
         for var in VARIANTS]


@pytest.mark.parametrize(
    'inner_solver,refinement,variant',
    CASES,
    ids=[f"{s}-{r[0]}-{v}" for s, r, v in CASES],
)
def test_quasistatic_cantilever_mpc_refinement(
        inner_solver, refinement, variant):
    """Per-case: record rel_err into shared cache; assert non-divergent."""
    label, nx, ny = refinement
    K, K_an, rel = _run_one(variant, nx, ny, inner_solver=inner_solver)
    _RESULTS[(inner_solver, variant, nx, ny)] = rel
    print(f"\n[#260 / quasi-static / {inner_solver:>6} / "
          f"{label} ({nx}x{ny}) / {variant}] "
          f"K={K:.6e}, K_analytical={K_an:.6e}, rel_err={rel:.3%}")
    assert rel < 5.0, f"K diverged: rel_err={rel:.2%}"


def test_quasistatic_cantilever_mpc_summary():
    """Print 4-column convergence table; PASS if finest CG max rel_err ≤ 2%."""
    keys = [(s, v, nx, ny)
            for s in INNER_SOLVERS
            for (lbl, nx, ny) in REFINEMENTS
            for v in VARIANTS]
    missing = [k for k in keys if k not in _RESULTS]
    if missing:
        pytest.skip(f"refinement cases not all run yet: missing={missing}")

    print("\n" + "=" * 84)
    print("Issue #260 — QuasiStaticSolver MPC cantilever refinement")
    print("L=10, h=1, E=1, nu=0, plane stress, T3 elements, "
          "rotation_free=True")
    print("=" * 84)
    print(f"{'refinement':<12}{'mesh':<12}"
          f"{'CG isolated':>14}{'CG connected':>14}"
          f"{'Direct iso':>14}{'Direct conn':>14}")
    print("-" * 84)
    finest_max = 0.0
    for (label, nx, ny) in REFINEMENTS:
        rci = _RESULTS[('cg', 'isolated', nx, ny)]
        rcc = _RESULTS[('cg', 'connected', nx, ny)]
        rdi = _RESULTS[('direct', 'isolated', nx, ny)]
        rdc = _RESULTS[('direct', 'connected', nx, ny)]
        print(f"{label:<12}{f'{nx}x{ny}':<12}"
              f"{rci:>13.3%} {rcc:>13.3%} "
              f"{rdi:>13.3%} {rdc:>13.3%}")
        if (label, nx, ny) == REFINEMENTS[-1]:
            finest_max = max(rci, rcc, rdi, rdc)
    print("=" * 84)
    print(f"finest-mesh max rel_err = {finest_max:.3%} (threshold 2%)")
    print("=" * 84)
    assert finest_max <= 0.02, (
        f"Finest mesh max rel_err = {finest_max:.3%} > 2% — "
        f"QuasiStaticSolver MPC verdict cannot be locked in.")
