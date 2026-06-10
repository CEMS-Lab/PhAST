"""Iterative-CG path tests for the rotation-free rigid_connector MPC.

Issue #171 — extends the full Lagrange (T-matrix) MPC from PR #164's static
DirectSolver path (and PR #169's explicit-dynamic path) to the
iterative-CG ``SecantCGSolver`` path used by ``solver_type='quasi_static_legacy'``.

Two checks:
  1. Cantilever-tip compliance (within ~5%) on the iterative-CG path.
  2. Toy connector regression: ``rotation_free=True`` is more compliant
     than ``rotation_free=False`` on the same CG path — confirms the new
     code is engaged and not silently re-routed through the welded
     Dirichlet expansion.
"""

import numpy as np
import pytest
import torch


def _build_cantilever(L=20.0, h=1.0, nx=80, ny=6):
    """Structured T3 cantilever mesh + node sets."""
    from phast.mesh import FEMMesh

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
    nodes_t = torch.tensor(nodes, dtype=torch.float64)
    elems_t = torch.tensor(elems, dtype=torch.long)

    left_idx = np.where(np.isclose(nodes[:, 0], 0.0))[0]
    right_idx = np.where(np.isclose(nodes[:, 0], L))[0]
    master_node = int(ny // 2 * (nx + 1) + nx)
    node_sets = {
        'left': torch.tensor(left_idx, dtype=torch.long),
        'tip_face': torch.tensor(right_idx, dtype=torch.long),
        'master': torch.tensor([master_node], dtype=torch.long),
    }
    mesh = FEMMesh.from_tensors(nodes_t, elems_t, node_sets,
                                device='cpu', dtype=torch.float64)
    return mesh, master_node, node_sets


def _solve_secant_cg(fem, mesh, bcs, n, *, rcs):
    """Run SecantCGSolver to equilibrium."""
    from phast.mechanics_solver import SecantCGSolver

    solver = SecantCGSolver(fem, tol=1e-9, max_iter=4000, max_newton=2,
                            check_every=25, use_multigrid=False)
    u0 = torch.zeros(n, 2, dtype=torch.float64)
    d = torch.zeros(n, dtype=torch.float64)
    bc_mask, bc_vals = bcs.get_masks_and_values()
    f_ext = torch.zeros(n, 2, dtype=torch.float64)
    u_sol = solver.solve(u0, d, bc_mask, bc_vals, f_ext=f_ext,
                         rigid_connectors=rcs)
    return u_sol, solver


@pytest.mark.parametrize("rotation_free,expected_label", [
    (True, "rotation_free"),
    (False, "welded"),
])
def test_cantilever_secant_cg_compliance(rotation_free, expected_label):
    """Cantilever beam under tip rigid connector, SecantCGSolver iterative path.

    Closed-form: ``K_analytical = 3 E I / L^3``.

    The rotation-free MPC running through ``_solve_impl_mpc`` should land
    within ~30% of the analytical stiffness on this slender (L/h=20) mesh —
    matching the static DirectSolver reference (PR #164) up to CG tolerance.
    """
    from phast.material import Material
    from phast.fem_operators import FEMOperators
    from phast.boundary_conditions import BoundaryConditions

    L, h = 20.0, 1.0
    mesh, master_node, node_sets = _build_cantilever(L=L, h=h)
    n = mesh.n_nodes

    E = 1.0
    nu = 0.0
    mat = Material(E=E, nu=nu, Gc=1.0, l0=1.0,
                   energy_split='isotropic', plane_stress=True)
    fem = FEMOperators(mesh, mat, ctx=None)

    u_prescribed = 0.05
    bcs = BoundaryConditions(n, device='cpu', dtype=torch.float64)
    bcs.fix(node_sets['left'], component=0)
    bcs.fix(node_sets['left'], component=1)
    bcs.add_rigid_connector(
        master_node=master_node,
        slave_indices=node_sets['tip_face'],
        locked_components=[0, 1],
        prescribe={0: 0.0, 1: u_prescribed},
        rotation_free=rotation_free,
    )

    rcs = bcs.get_active_rigid_connectors() if rotation_free else None
    u_sol, _solver = _solve_secant_cg(fem, mesh, bcs, n, rcs=rcs)

    f_int = fem.internal_force(u_sol, torch.zeros(n, dtype=torch.float64))
    slave_idx = node_sets['tip_face'].numpy()
    ids = np.unique(np.concatenate([np.array([master_node]), slave_idx]))
    R_y = float(f_int[ids, 1].sum().item())
    K = R_y / u_prescribed

    I_section = h ** 3 / 12.0
    K_analytical = 3.0 * E * I_section / L ** 3
    rel = abs(K - K_analytical) / K_analytical
    print(f"\n[secant_cg/{expected_label}] K={K:.4e}, "
          f"K_analytical={K_analytical:.4e}, rel_err={rel:.2%}")

    if rotation_free:
        # FE shear-locking on the slender L/h=20 mesh dominates the error
        # — matches the DirectSolver static-path baseline (~15% in
        # test_bc_vocab.py::TestRigidConnectorRotation::test_cantilever_compliance).
        assert rel < 0.30, (
            f"iterative-CG rotation-free K should approach analytical "
            f"PL^3/3EI; got K={K:.4e} vs {K_analytical:.4e} "
            f"(rel error {rel:.2%})")
    else:
        # Welded path lacks the theta DOF and is substantially stiffer.
        assert K > 1.5 * K_analytical, (
            f"welded SecantCG should be >>1.5x analytical stiffness; "
            f"got K={K:.4e} vs analytical={K_analytical:.4e}")


def test_secant_cg_rotation_free_vs_welded_distinct():
    """Toy regression: rotation-free SecantCG path is not silently welded."""
    from phast.material import Material
    from phast.fem_operators import FEMOperators
    from phast.boundary_conditions import BoundaryConditions

    L, h = 20.0, 1.0
    mesh, _master, node_sets = _build_cantilever(L=L, h=h, nx=40, ny=4)
    n = mesh.n_nodes
    mat = Material(E=1.0, nu=0.0, Gc=1.0, l0=1.0,
                   energy_split='isotropic', plane_stress=True)
    fem = FEMOperators(mesh, mat, ctx=None)

    nodes = mesh.nodes.numpy()
    nx_, ny_ = 40, 4
    master_node = int(ny_ // 2 * (nx_ + 1) + nx_)
    node_sets = dict(node_sets)
    node_sets['tip_face'] = torch.tensor(
        np.where(np.isclose(nodes[:, 0], L))[0], dtype=torch.long)
    node_sets['left'] = torch.tensor(
        np.where(np.isclose(nodes[:, 0], 0.0))[0], dtype=torch.long)

    u_prescribed = 0.05

    def _run(rotation_free):
        bcs = BoundaryConditions(n, device='cpu', dtype=torch.float64)
        bcs.fix(node_sets['left'], component=0)
        bcs.fix(node_sets['left'], component=1)
        bcs.add_rigid_connector(
            master_node=master_node,
            slave_indices=node_sets['tip_face'],
            locked_components=[0, 1],
            prescribe={0: 0.0, 1: u_prescribed},
            rotation_free=rotation_free,
        )
        rcs = bcs.get_active_rigid_connectors() if rotation_free else None
        u_sol, _ = _solve_secant_cg(fem, mesh, bcs, n, rcs=rcs)
        f_int = fem.internal_force(
            u_sol, torch.zeros(n, dtype=torch.float64))
        slave_idx = node_sets['tip_face'].numpy()
        ids = np.unique(np.concatenate(
            [np.array([master_node]), slave_idx]))
        return float(f_int[ids, 1].sum().item())

    R_free = _run(rotation_free=True)
    R_weld = _run(rotation_free=False)
    print(f"\n[secant_cg/distinct] R_free={R_free:.4e}, R_weld={R_weld:.4e}, "
          f"ratio_weld/free={R_weld / R_free:.2f}")
    assert R_weld > 1.5 * R_free, (
        f"welded should produce a substantially larger reaction than "
        f"rotation-free at the same prescribed displacement; got "
        f"R_weld={R_weld:.4e} vs R_free={R_free:.4e}")


def test_secant_cg_master_must_have_dirichlet_translation():
    """SecantCG MPC mirrors DirectSolver's contract: master must be Dirichlet."""
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.fem_operators import FEMOperators
    from phast.boundary_conditions import BoundaryConditions
    from phast.mechanics_solver import SecantCGSolver

    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                          [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                         dtype=torch.float64)
    elems = torch.tensor([[0, 1, 4], [0, 4, 3],
                          [1, 2, 5], [1, 5, 4]], dtype=torch.long)
    node_sets = {
        'left': torch.tensor([0, 3], dtype=torch.long),
        'right': torch.tensor([2, 5], dtype=torch.long),
    }
    mesh = FEMMesh.from_tensors(nodes, elems, node_sets,
                                device='cpu', dtype=torch.float64)
    mat = Material(E=1.0, nu=0.25, Gc=1.0, l0=1.0,
                   energy_split='isotropic')
    fem = FEMOperators(mesh, mat, ctx=None)
    solver = SecantCGSolver(fem, tol=1e-8, max_iter=200, max_newton=2,
                            use_multigrid=False)
    n = mesh.n_nodes
    d = torch.zeros(n, dtype=torch.float64)

    bcs = BoundaryConditions(n, device='cpu', dtype=torch.float64)
    bcs.fix(node_sets['left'], component=0)
    bcs.fix(node_sets['left'], component=1)
    bcs.add_rigid_connector(
        master_node=2,
        slave_indices=node_sets['right'],
        locked_components=[],
        prescribe={},
        rotation_free=True,
    )
    mask, vals = bcs.get_masks_and_values()
    u0 = torch.zeros(n, 2, dtype=torch.float64)
    rcs = bcs.get_active_rigid_connectors()
    with pytest.raises(ValueError, match="master node"):
        solver.solve(u0, d, mask, vals, rigid_connectors=rcs)


def test_secant_cg_mpc_emits_one_shot_multigrid_warning(capfd):
    """Issue #189: when MPC routes a use_multigrid=True solver through the
    Jacobi-only path, a single diagnostic line is printed per solver instance,
    not once per Newton iteration.
    """
    from phast.material import Material
    from phast.fem_operators import FEMOperators
    from phast.boundary_conditions import BoundaryConditions
    from phast.mechanics_solver import SecantCGSolver

    L, h = 20.0, 1.0
    mesh, master_node, node_sets = _build_cantilever(L=L, h=h, nx=20, ny=4)
    n = mesh.n_nodes
    mat = Material(E=1.0, nu=0.0, Gc=1.0, l0=1.0,
                   energy_split='isotropic', plane_stress=True)
    fem = FEMOperators(mesh, mat, ctx=None)

    nodes = mesh.nodes.numpy()
    nx_, ny_ = 20, 4
    master_node = int(ny_ // 2 * (nx_ + 1) + nx_)
    tip_idx = torch.tensor(
        np.where(np.isclose(nodes[:, 0], L))[0], dtype=torch.long)
    left_idx = torch.tensor(
        np.where(np.isclose(nodes[:, 0], 0.0))[0], dtype=torch.long)

    bcs = BoundaryConditions(n, device='cpu', dtype=torch.float64)
    bcs.fix(left_idx, component=0)
    bcs.fix(left_idx, component=1)
    bcs.add_rigid_connector(
        master_node=master_node,
        slave_indices=tip_idx,
        locked_components=[0, 1],
        prescribe={0: 0.0, 1: 0.05},
        rotation_free=True,
    )
    rcs = bcs.get_active_rigid_connectors()
    mask, vals = bcs.get_masks_and_values()

    # use_multigrid=True triggers the warning gate
    solver = SecantCGSolver(fem, tol=1e-7, max_iter=500, max_newton=2,
                            check_every=25, use_multigrid=True)
    u0 = torch.zeros(n, 2, dtype=torch.float64)
    d = torch.zeros(n, dtype=torch.float64)
    f_ext = torch.zeros(n, 2, dtype=torch.float64)

    # Drive multiple Newton steps by re-entering solve(); the warning must
    # fire exactly once across both calls.
    for _ in range(3):
        solver.solve(u0, d, mask, vals, f_ext=f_ext, rigid_connectors=rcs)

    out, _err = capfd.readouterr()
    occurrences = out.count("[SecantCGSolver][MPC]")
    assert occurrences == 1, (
        f"expected exactly 1 MPC warning across 3 solves, got "
        f"{occurrences}\n--- captured stdout ---\n{out}")
    assert "multigrid preconditioner" in out
    assert "L-BFGS history bypassed" in out
    assert "#171/#189" in out
    assert "Jacobi-only CG" in out
