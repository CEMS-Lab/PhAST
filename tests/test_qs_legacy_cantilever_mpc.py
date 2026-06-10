"""Refinement-study probe for issue #223 — QS-legacy (SecantCGSolver) MPC T-block.

Issue #223: post-PR #207 the QS notched-holed-plate benchmark hits the
correct first-peak load (4% off, PASS) but the displacement at first peak
is 53% off (sim 0.154 mm vs reference 0.330 mm — sim is roughly 2x stiffer
up to peak). The hypothesis under test here is that
``SecantCGSolver._solve_impl_mpc``'s reduced T-matrix matvec
(``_matvec_red``) double-counts the kinematic stiffness when the master
of a rigid_connector is an *isolated* mesh node (no element connectivity)
— the exact configuration used by the notched-holed plate's
``upper_pin_centre`` / ``lower_pin_centre`` Physical Points.

PR #224 originally tested a single 80x8 T3 mesh and reported rel_err =
4.83% under a 5% threshold. Advisor flagged that as too soft to lock in
the "T-block fine" verdict — most of the 4.83% could be FE shear-locking
on a coarse linear-T3 cantilever rather than real T-block error. This
refinement study (h-doubling at three levels) empirically separates the
two contributions:

- the asymptote as h -> 0 = real T-block error
- the difference at coarse h = FE shear-locking offset

Discriminator (read off the finest mesh):
- max rel_err over both variants <= 1.5% -> T-BLOCK UNAMBIGUOUSLY FINE
- 1.5% < max rel_err <= 2.0%             -> borderline; T-block likely fine
- > 2.0% but trending down with h        -> FE locking dominant; T-block fine
- plateau / growth across refinement     -> SMOKING GUN; real T-block bug

The test is marked PASS only if max rel_err on the finest mesh is
<= 2%. Print the convergence table for the verdict.

Run manually (NOT in CI):

    pytest tests/test_qs_legacy_cantilever_mpc.py -x -s
"""

import numpy as np
import pytest
import torch


def _build_grid(L, h, nx, ny):
    """Structured T3 grid over [0, L] x [-h/2, h/2]."""
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
    """Grid + an *isolated* master node at (L, 0); no element refers to it."""
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
        'master': torch.tensor([master_node], dtype=torch.long),
    }
    mesh = FEMMesh.from_tensors(nodes_t, elems_t, node_sets,
                                device='cpu', dtype=torch.float64)
    return mesh, master_node, node_sets


def _build_cantilever_connected(L, h, nx, ny):
    """Grid only; master = mid-height tip-face grid node (has element refs)."""
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


def _solve_secant_cg(fem, mesh, bcs, n, *, rcs):
    from phast.mechanics_solver import SecantCGSolver

    solver = SecantCGSolver(fem, tol=1e-9, max_iter=8000, max_newton=2,
                            check_every=25, use_multigrid=False)
    u0 = torch.zeros(n, 2, dtype=torch.float64)
    d = torch.zeros(n, dtype=torch.float64)
    bc_mask, bc_vals = bcs.get_masks_and_values()
    f_ext = torch.zeros(n, 2, dtype=torch.float64)
    return solver.solve(u0, d, bc_mask, bc_vals, f_ext=f_ext,
                        rigid_connectors=rcs)


def _run_one(variant, nx, ny, L=10.0, h=1.0, E=1.0, nu=0.0,
             u_prescribed=0.05):
    """Run one (variant, mesh) case, return (K_sim, K_analytical, rel_err)."""
    from phast.material import Material
    from phast.fem_operators import FEMOperators
    from phast.boundary_conditions import BoundaryConditions

    if variant == 'isolated':
        mesh, master_node, node_sets = _build_cantilever_isolated(L, h, nx, ny)
    elif variant == 'connected':
        mesh, master_node, node_sets = _build_cantilever_connected(L, h, nx, ny)
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
    u_sol = _solve_secant_cg(fem, mesh, bcs, n, rcs=rcs)

    f_int = fem.internal_force(u_sol, torch.zeros(n, dtype=torch.float64))
    slave_idx = node_sets['tip_face'].numpy()
    ids = np.unique(np.concatenate([np.array([master_node]), slave_idx]))
    R_y = float(f_int[ids, 1].sum().item())
    K = R_y / u_prescribed
    K_analytical = 3.0 * E * (h ** 3 / 12.0) / L ** 3
    rel = abs(K - K_analytical) / K_analytical
    return K, K_analytical, rel


# --- shared cache so the parametrized cases produce one summary table -------

_RESULTS = {}  # (variant, nx, ny) -> rel_err


REFINEMENTS = [
    ('coarse',  80,  8),
    ('medium', 160, 16),
    ('fine',   320, 32),
]
VARIANTS = ['isolated', 'connected']
CASES = [(ref, var) for ref in REFINEMENTS for var in VARIANTS]


@pytest.mark.parametrize(
    'refinement,variant',
    CASES,
    ids=[f"{r[0]}-{v}" for r, v in CASES],
)
def test_qs_legacy_cantilever_mpc_refinement(refinement, variant):
    """Per-case run: record rel_err into shared cache; assert non-divergent."""
    label, nx, ny = refinement
    K, K_an, rel = _run_one(variant, nx, ny)
    _RESULTS[(variant, nx, ny)] = rel
    print(f"\n[#223 probe / qs_legacy / {label} ({nx}x{ny}) / {variant}] "
          f"K={K:.6e}, K_analytical={K_an:.6e}, rel_err={rel:.3%}")
    # guard against divergence; the verdict gate is in the summary test below
    assert rel < 5.0, f"K diverged: rel_err={rel:.2%}"


def test_qs_legacy_cantilever_mpc_summary():
    """Print the 6-row convergence table and gate on finest-mesh rel_err.

    Must run after the parametrized cases above (pytest collects in file
    order, so this is fine for `pytest -x -s`). PASS gate:
        max(rel_err) over both variants on the finest (320x32) mesh
            must be <= 2%.
    """
    missing = [k for k in [(v, nx, ny) for (lbl, nx, ny) in REFINEMENTS
                            for v in VARIANTS]
               if k not in _RESULTS]
    if missing:
        pytest.skip(f"refinement cases not all run yet: missing={missing}")

    print("\n" + "=" * 72)
    print("Issue #223 — QS-legacy MPC cantilever refinement study")
    print("L=10, h=1, E=1, nu=0, plane stress, T3 elements, "
          "rotation_free=True")
    print("=" * 72)
    print(f"{'refinement':<12}{'mesh':<12}{'isolated':>14}"
          f"{'connected':>14}{'max':>10}")
    print("-" * 72)
    finest_max = 0.0
    rows = []
    for (label, nx, ny) in REFINEMENTS:
        ri = _RESULTS[('isolated', nx, ny)]
        rc = _RESULTS[('connected', nx, ny)]
        rmax = max(ri, rc)
        rows.append((label, nx, ny, ri, rc, rmax))
        print(f"{label:<12}{f'{nx}x{ny}':<12}"
              f"{ri:>13.3%} {rc:>13.3%} {rmax:>9.3%}")
    print("=" * 72)

    # finest = last refinement
    _, fnx, fny, fri, frc, fmax = rows[-1]
    finest_max = fmax

    # Verdict
    if finest_max <= 0.015:
        verdict = "T-BLOCK UNAMBIGUOUSLY FINE (finest <= 1.5%)"
    elif finest_max <= 0.02:
        verdict = "BORDERLINE — T-block likely fine (finest <= 2%)"
    else:
        # Check trend across refinements
        ri0, ri1, ri2 = (_RESULTS[('isolated', nx, ny)]
                          for (_, nx, ny) in REFINEMENTS)
        rc0, rc1, rc2 = (_RESULTS[('connected', nx, ny)]
                          for (_, nx, ny) in REFINEMENTS)
        decreasing = (ri0 >= ri1 >= ri2) and (rc0 >= rc1 >= rc2)
        if decreasing:
            verdict = ("FE LOCKING DOMINANT — values still falling, "
                       "T-block likely fine but underrefined here")
        else:
            verdict = ("SMOKING GUN — rel_err plateau or growth under "
                       "h-refinement; real T-block bug suspected")
    print(f"VERDICT: {verdict}")
    print("=" * 72)

    assert finest_max <= 0.02, (
        f"Finest mesh ({fnx}x{fny}) max rel_err = {finest_max:.3%} > 2% — "
        f"T-block fine verdict cannot be locked in. Verdict: {verdict}"
    )
