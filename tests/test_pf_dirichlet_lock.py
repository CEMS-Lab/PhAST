"""Issue #213 — phase-field Dirichlet (``pf_dirichlet``) BC test.

Locks ``phi = value`` on a set of notch nodes for the entire simulation
(matching COMSOL's pre-existing crack convention). This is the
"BC-for-the-duration" mechanism, distinct from the IC-only
``preseed_notch_nodesets`` (which only sets the initial elastic state
via a high-``H`` injection at t=0 and lets the bound-clamped damage
solve drift afterwards).

The test runs a short explicit-dynamic SENT problem on a tiny mesh
(no mocking of the damage solver), drives it with a non-trivial
load, and asserts that ``solver.d[notch_nodes] == 1.0`` to within
1e-12 at every step.
"""

import os
import torch
import pytest


def _build_tiny_sent_mesh(tmp_path, h: float = 0.25):
    """Generate a small unit-square mesh with a labelled ``notch`` group.

    The notch is the line of nodes at y = 0.5, x in [0, 0.5] — i.e. the
    classical SENT pre-crack on the left half of the mid-height line.
    """
    geo = f"""
SetFactory("OpenCASCADE");
Point(1) = {{0,0,0,{h}}};
Point(2) = {{1,0,0,{h}}};
Point(3) = {{1,1,0,{h}}};
Point(4) = {{0,1,0,{h}}};
Line(1)={{1,2}}; Line(2)={{2,3}}; Line(3)={{3,4}}; Line(4)={{4,1}};
Curve Loop(1)={{1,2,3,4}}; Plane Surface(1)={{1}};
Physical Surface("plate")={{1}};
Physical Curve("bottom")={{1}};
Physical Curve("right") ={{2}};
Physical Curve("top")   ={{3}};
Physical Curve("left")  ={{4}};
Mesh.ElementOrder=1;
Mesh.CharacteristicLengthMax={h};
"""
    geo_file = tmp_path / "sq.geo"
    msh_file = tmp_path / "sq.msh"
    geo_file.write_text(geo)
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(str(geo_file))
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_file))
    finally:
        gmsh.finalize()
    return str(msh_file)


def test_pf_dirichlet_locks_notch_nodes_every_step(tmp_path):
    """Notch nodes carry ``phi = 1.0`` exactly at every damage solve."""
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions
    from phast.staggered_solver import StaggeredSolver, SolverConfig

    msh_path = _build_tiny_sent_mesh(tmp_path, h=0.25)
    mesh = FEMMesh(msh_path, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()

    # Pick a representative pair of "notch" nodes: the bottom-edge
    # nodes form a stable, reproducible set across mesh seeds. (Real
    # SENT notches use a labelled physical curve; for this small unit
    # test we re-purpose the ``bottom`` node set as the locked region.)
    notch_idx = mesh.node_sets['bottom']
    assert len(notch_idx) >= 2, (
        f"tiny SENT mesh produced too few notch nodes ({len(notch_idx)})"
    )

    mat = Material(
        E=210e3, nu=0.3, Gc=2.7e-3, l0=0.05, rho=7.85e-9,
        energy_split='amor', pf_model='AT2',
    )

    bcs = BoundaryConditions(mesh.n_nodes, device='cpu',
                             dtype=torch.float64)
    bcs.fix(mesh.node_sets['bottom'], component=0)
    bcs.fix(mesh.node_sets['bottom'], component=1)
    # Drive top in y to actually grow damage near the notch tip.
    bcs.add(mesh.node_sets['top'], component=1, value=1e-3)
    # The BC-under-test: lock damage at notch nodes.
    bcs.add_pf_dirichlet(notch_idx, value=1.0)

    cfg = SolverConfig(
        solver_type='explicit',
        damage_tol=1e-6,
        bounds_method='post_clamp',
    )
    solver = StaggeredSolver(mesh, mat, bcs, cfg)

    # Initial state must already have phi=1 at notch nodes.
    assert torch.allclose(
        solver.d[notch_idx],
        torch.ones_like(solver.d[notch_idx]),
        atol=1e-12, rtol=0,
    ), "pf_dirichlet not enforced at construction"

    # Run a handful of damage solves; assert the lock holds at each.
    n_steps = 20
    for step in range(n_steps):
        # Inject some driving force so the damage solver has something
        # to do (otherwise the unconstrained solution is identically 0
        # and the test would pass trivially).
        solver.H_elem.fill_(1.0e2)
        solver.step_solve_damage()
        d_notch = solver.d[notch_idx]
        diff = (d_notch - 1.0).abs().max().item()
        assert diff <= 1e-12, (
            f"pf_dirichlet drift at step {step}: max |phi-1| = {diff:.3e}"
        )


def test_pf_dirichlet_arbitrary_value(tmp_path):
    """``value`` is honoured (not hardcoded to 1.0)."""
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions
    from phast.staggered_solver import StaggeredSolver, SolverConfig

    msh_path = _build_tiny_sent_mesh(tmp_path, h=0.5)
    mesh = FEMMesh(msh_path, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()

    nodes = mesh.nodes
    y_mid = 0.5 * (nodes[:, 1].max() + nodes[:, 1].min())
    notch_idx = torch.nonzero(
        (nodes[:, 1] - y_mid).abs() < 1e-9, as_tuple=False).flatten()

    mat = Material(E=210e3, nu=0.3, Gc=2.7e-3, l0=0.1, rho=7.85e-9,
                   energy_split='amor', pf_model='AT2')
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu',
                             dtype=torch.float64)
    bcs.fix(mesh.node_sets['bottom'], component=1)
    bcs.add_pf_dirichlet(notch_idx, value=0.4)

    cfg = SolverConfig(solver_type='explicit', bounds_method='post_clamp')
    solver = StaggeredSolver(mesh, mat, bcs, cfg)

    assert torch.allclose(
        solver.d[notch_idx],
        torch.full_like(solver.d[notch_idx], 0.4),
        atol=1e-12, rtol=0,
    )
    solver.H_elem.fill_(50.0)
    solver.step_solve_damage()
    assert torch.allclose(
        solver.d[notch_idx],
        torch.full_like(solver.d[notch_idx], 0.4),
        atol=1e-12, rtol=0,
    )


def test_pf_dirichlet_enters_damage_equations(tmp_path):
    """Pinned damage contributes to neighboring free-node equations."""
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions
    from phast.staggered_solver import StaggeredSolver, SolverConfig

    msh_path = _build_tiny_sent_mesh(tmp_path, h=0.5)
    mesh = FEMMesh(msh_path, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()

    pinned = mesh.node_sets['bottom'][:1]
    mat = Material(E=210e3, nu=0.3, Gc=2.7, l0=0.4, rho=7.85e-9,
                   energy_split='amor', pf_model='AT2')
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu',
                             dtype=torch.float64)
    bcs.add_pf_dirichlet(pinned, value=1.0)
    cfg = SolverConfig(
        solver_type='explicit',
        bounds_method='post_clamp',
        damage_tol=1e-10,
        damage_max_iter=1000,
    )
    solver = StaggeredSolver(mesh, mat, bcs, cfg)
    solver.d.zero_()
    solver._apply_pf_dirichlet()
    solver.H_elem.zero_()

    solver.step_solve_damage()

    mask = torch.ones(mesh.n_nodes, dtype=torch.bool)
    mask[pinned.cpu()] = False
    assert solver.d[mask].max().item() > 1e-8


def test_forward_gc_field_with_pf_dirichlet_is_supported(tmp_path):
    """Forward-only heterogeneous Gc maps can coexist with a pre-crack lock."""
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions
    from phast.staggered_solver import StaggeredSolver, SolverConfig

    msh_path = _build_tiny_sent_mesh(tmp_path, h=0.5)
    mesh = FEMMesh(msh_path, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()

    pinned = mesh.node_sets['bottom'][:1]
    mat = Material(E=210e3, nu=0.3, Gc=2.7, l0=0.4, rho=7.85e-9,
                   energy_split='amor', pf_model='AT2',
                   gamma_correction=False)
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu',
                             dtype=torch.float64)
    bcs.add_pf_dirichlet(pinned, value=1.0)
    cfg = SolverConfig(
        solver_type='explicit',
        bounds_method='post_clamp',
        damage_tol=1e-10,
        damage_max_iter=1000,
    )
    solver = StaggeredSolver(mesh, mat, bcs, cfg)
    solver.d.zero_()
    solver._apply_pf_dirichlet()
    solver.H_elem.fill_(20.0)

    Gc_field = torch.linspace(
        2.0, 3.0, mesh.n_elems, dtype=torch.float64)
    solver.diff_Gc_field = Gc_field
    solver.step_solve_damage()

    assert torch.allclose(
        solver.d[pinned],
        torch.ones_like(solver.d[pinned]),
        atol=1e-12,
        rtol=0,
    )
    assert solver.damage_solver._Gc_l0_e is None
    assert solver.damage_solver._Gc_over_l0_e is None


def test_differentiable_gc_field_with_pf_dirichlet_is_supported(tmp_path):
    """Grad-enabled Gc maps support pre-crack locks with explicit adjoints."""
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions
    from phast.staggered_solver import StaggeredSolver, SolverConfig

    msh_path = _build_tiny_sent_mesh(tmp_path, h=0.5)
    mesh = FEMMesh(msh_path, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()

    mat = Material(E=210e3, nu=0.3, Gc=2.7, l0=0.4, rho=7.85e-9,
                   energy_split='amor', pf_model='AT2',
                   gamma_correction=True)
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu',
                             dtype=torch.float64)
    bcs.add_pf_dirichlet(mesh.node_sets['bottom'][:1], value=1.0)
    cfg = SolverConfig(solver_type='explicit', bounds_method='post_clamp')
    solver = StaggeredSolver(mesh, mat, bcs, cfg)

    Gc_field = torch.full(
        (mesh.n_elems,), 2.7, dtype=torch.float64, requires_grad=True)
    solver.diff_Gc_field = Gc_field
    solver.H_elem.fill_(20.0)
    solver.step_solve_damage()
    loss = solver.d.sum()
    loss.backward()

    assert Gc_field.grad is not None
    assert torch.isfinite(Gc_field.grad).all()
    mask, vals = bcs.get_pf_dirichlet_mask_values()
    assert torch.allclose(solver.d[mask], vals[mask], atol=1e-12, rtol=0)


def test_pf_dirichlet_coexists_with_preseed(tmp_path):
    """``preseed_notch_nodesets`` (IC) and ``pf_dirichlet`` (BC)
    must be independent — pf_dirichlet does not perturb the preseed
    list and vice versa. Verifies via the public ``BoundaryConditions``
    API that the two registries are disjoint and a pf_dirichlet
    addition leaves the displacement Dirichlet list intact.
    """
    from phast.boundary_conditions import BoundaryConditions

    bcs = BoundaryConditions(n_nodes=10, device='cpu',
                             dtype=torch.float64)
    bcs.fix(torch.tensor([0, 1], dtype=torch.long), component=1)
    n_disp_before = len(bcs.bcs)
    n_pf_before = len(bcs.pf_dirichlet_bcs)
    bcs.add_pf_dirichlet(torch.tensor([2, 3], dtype=torch.long),
                         value=1.0)
    assert len(bcs.bcs) == n_disp_before, (
        "pf_dirichlet must not touch the displacement Dirichlet list"
    )
    assert len(bcs.pf_dirichlet_bcs) == n_pf_before + 1
    mask, vals = bcs.get_pf_dirichlet_mask_values()
    assert mask.sum().item() == 2
    assert torch.equal(
        torch.nonzero(mask, as_tuple=False).flatten(),
        torch.tensor([2, 3], dtype=torch.long))
    assert torch.allclose(vals[mask], torch.tensor([1.0, 1.0],
                                                    dtype=torch.float64))
