"""Explicit-dynamic path tests for the rotation-free rigid_connector MPC.

Issue #165 — extends the full Lagrange (T-matrix) MPC from PR #164's static
DirectSolver path to the explicit velocity-Verlet path.

Two checks:
  1. Cantilever-tip compliance.  A clamped beam loaded at the tip via a
     prescribed master displacement should expose a tip stiffness close to
     the Euler–Bernoulli closed form 3 E I / L^3 once the dynamic transient
     has decayed.  The welded fallback (rotation_free=False) lacks the
     theta DOF and is substantially stiffer.
  2. Toy regression.  A two-slave connector run with rotation_free=True
     produces a different (more compliant) response than rotation_free=False
     — confirms the new code path is actually engaged and not silently
     re-routed through the welded Dirichlet expansion.
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


def _settle_explicit(fem, mesh, bcs, n, *, n_steps, dt_safety=0.9,
                     damping_ratio_max=0.4, rcs=None):
    """Velocity-Verlet to (approximate) static equilibrium with Rayleigh damping."""
    from phast.mechanics_solver import ExplicitDynamics

    explicit = ExplicitDynamics(fem, dt=None, dt_safety=dt_safety,
                                damping_ratio_max=damping_ratio_max)
    u = torch.zeros(n, 2, dtype=torch.float64)
    v = torch.zeros_like(u)
    a = torch.zeros_like(u)
    d = torch.zeros(n, dtype=torch.float64)
    f_ext = torch.zeros_like(u)
    bc_mask, bc_vals = bcs.get_masks_and_values()
    # Smooth ramp on the prescribed Dirichlet values to avoid exciting
    # high-frequency modes (which would not damp out cleanly in the budget).
    n_ramp = max(int(n_steps * 0.4), 100)
    bc_vals_full = bc_vals.clone()
    for k in range(n_steps):
        s = min(1.0, (k + 1) / n_ramp)
        # smooth-step ramp
        s = s * s * (3.0 - 2.0 * s)
        bc_vals_k = bc_vals_full * s
        u, v, a = explicit.step(u, v, a, d, f_ext, bc_mask, bc_vals_k,
                                rigid_connectors=rcs)
    return u, v, a, explicit


@pytest.mark.parametrize("rotation_free,expected_label", [
    (True, "rotation_free"),
    (False, "welded"),
])
def test_cantilever_explicit_compliance(rotation_free, expected_label):
    """Cantilever beam under tip rigid connector, explicit-dynamic settle.

    With rotation_free=True (issue #165's new path) the tip rigid connector
    should reproduce the Euler–Bernoulli compliance 3 E I / L^3 within
    ~30%.  With rotation_free=False the welded Dirichlet over-constrains
    the tip rotation and yields a substantially stiffer response.
    """
    from phast.material import Material
    from phast.fem_operators import FEMOperators
    from phast.boundary_conditions import BoundaryConditions

    L, h = 20.0, 1.0
    mesh, master_node, node_sets = _build_cantilever(L=L, h=h)
    n = mesh.n_nodes

    E = 1.0
    nu = 0.0
    mat = Material(E=E, nu=nu, Gc=1.0, l0=1.0, rho=1.0,
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
    # Long enough horizon for the ramped excitation + connector theta mode
    # to decay; the rotation_free path adds a low-frequency rotational mode
    # that the Kelvin-Voigt damping reaches via the slave forces but it
    # takes many CFL steps to settle.
    n_steps = 30000
    u_sol, _, _, _ = _settle_explicit(
        fem, mesh, bcs, n, n_steps=n_steps,
        damping_ratio_max=0.4, rcs=rcs)

    # Reaction at master = sum of internal force over the rigid set, comp 1.
    f_int = fem.internal_force(u_sol, torch.zeros(n, dtype=torch.float64))
    slave_idx = node_sets['tip_face'].numpy()
    ids = np.unique(np.concatenate([np.array([master_node]), slave_idx]))
    R_y = float(f_int[ids, 1].sum().item())
    K = R_y / u_prescribed

    I_section = h ** 3 / 12.0
    K_analytical = 3.0 * E * I_section / L ** 3
    rel = abs(K - K_analytical) / K_analytical
    print(f"\n[explicit/{expected_label}] K={K:.4e}, "
          f"K_analytical={K_analytical:.4e}, rel_err={rel:.2%}")

    if rotation_free:
        # Reference: static DirectSolver path lands within ~30% on this
        # mesh (FE shear-locking + slender L/h=20). The explicit-dynamic
        # settle adds dynamic-residual noise on top, so we widen the
        # tolerance to <30%.
        assert rel < 0.3, (
            f"explicit-dynamic rotation-free K should approach analytical "
            f"PL^3/3EI; got K={K:.4e} vs {K_analytical:.4e} "
            f"(rel error {rel:.2%})")
    else:
        # Welded fallback should be substantially stiffer than analytical.
        assert K > 1.5 * K_analytical, (
            f"welded explicit should be >>1.5x analytical stiffness; "
            f"got K={K:.4e} vs analytical={K_analytical:.4e}")


def test_explicit_rotation_free_vs_welded_distinct():
    """Toy 2-slave connector: rotation-free path is not silently welded.

    Same geometry, same prescribed master, both modes settled to (approximate)
    equilibrium.  Their tip-stiffness numbers must differ by a non-trivial
    factor — confirming the new MPC code in ExplicitDynamics is actually
    invoked and is not collapsed to the welded Dirichlet expansion.
    """
    from phast.material import Material
    from phast.fem_operators import FEMOperators
    from phast.boundary_conditions import BoundaryConditions

    L, h = 20.0, 1.0
    mesh, master_node, node_sets = _build_cantilever(L=L, h=h, nx=40, ny=4)
    n = mesh.n_nodes
    mat = Material(E=1.0, nu=0.0, Gc=1.0, l0=1.0, rho=1.0,
                   energy_split='isotropic', plane_stress=True)
    fem = FEMOperators(mesh, mat, ctx=None)

    # Reuse master at the centre row of the right face from the helper.
    # Re-detect for the smaller mesh:
    nodes = mesh.nodes.numpy()
    nx_, ny_ = 40, 4
    master_node = int(ny_ // 2 * (nx_ + 1) + nx_)
    right_idx = np.where(np.isclose(nodes[:, 0], L))[0]
    node_sets = dict(node_sets)
    node_sets['tip_face'] = torch.tensor(right_idx, dtype=torch.long)
    node_sets['left'] = torch.tensor(
        np.where(np.isclose(nodes[:, 0], 0.0))[0], dtype=torch.long)

    u_prescribed = 0.05
    n_steps = 4000

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
        u_sol, _, _, _ = _settle_explicit(
            fem, mesh, bcs, n, n_steps=n_steps,
            damping_ratio_max=0.4, rcs=rcs)
        f_int = fem.internal_force(
            u_sol, torch.zeros(n, dtype=torch.float64))
        slave_idx = node_sets['tip_face'].numpy()
        ids = np.unique(np.concatenate([np.array([master_node]), slave_idx]))
        return float(f_int[ids, 1].sum().item())

    R_free = _run(rotation_free=True)
    R_weld = _run(rotation_free=False)
    print(f"\n[explicit/distinct] R_free={R_free:.4e}, R_weld={R_weld:.4e}, "
          f"ratio_weld/free={R_weld / R_free:.2f}")
    assert R_weld > 1.5 * R_free, (
        f"welded should produce a substantially larger reaction than "
        f"rotation-free at the same prescribed displacement; got "
        f"R_weld={R_weld:.4e} vs R_free={R_free:.4e}")


def test_explicit_master_must_have_dirichlet_translation():
    """Explicit MPC mirrors the static contract: master must be Dirichlet."""
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.fem_operators import FEMOperators
    from phast.boundary_conditions import BoundaryConditions
    from phast.mechanics_solver import ExplicitDynamics

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
    mat = Material(E=1.0, nu=0.25, Gc=1.0, l0=1.0, rho=1.0,
                   energy_split='isotropic')
    fem = FEMOperators(mesh, mat, ctx=None)
    n = mesh.n_nodes

    bcs = BoundaryConditions(n, device='cpu', dtype=torch.float64)
    bcs.fix(node_sets['left'], component=0)
    bcs.fix(node_sets['left'], component=1)
    bcs.add_rigid_connector(
        master_node=2,
        slave_indices=node_sets['right'],
        locked_components=[],  # deliberately no Dirichlet on master
        prescribe={},
        rotation_free=True,
    )
    explicit = ExplicitDynamics(fem)
    u = torch.zeros(n, 2, dtype=torch.float64)
    v = torch.zeros_like(u); a = torch.zeros_like(u)
    d = torch.zeros(n, dtype=torch.float64)
    f_ext = torch.zeros_like(u)
    mask, vals = bcs.get_masks_and_values()
    rcs = bcs.get_active_rigid_connectors()
    with pytest.raises(ValueError, match="master node"):
        explicit.step(u, v, a, d, f_ext, mask, vals, rigid_connectors=rcs)
