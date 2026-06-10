import os
import tempfile
import types

import pytest
import torch


def _build_triangle_mesh():
    gmsh = pytest.importorskip("gmsh")
    from phast.mesh import FEMMesh

    geo = """
Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {0.5, 0.866025, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};
Curve Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};
Physical Surface("plate") = {1};
Physical Curve("bottom") = {1};
Physical Curve("right") = {2};
Physical Curve("left") = {3};
Mesh.ElementOrder = 1;
"""
    tmp = tempfile.mkdtemp()
    geo_path = os.path.join(tmp, "tri.geo")
    msh_path = os.path.join(tmp, "tri.msh")
    with open(geo_path, "w") as fh:
        fh.write(geo)

    if not gmsh.isInitialized():
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(geo_path)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(msh_path)
    finally:
        gmsh.finalize()

    mesh = FEMMesh(msh_path, device="cpu", dtype=torch.float64)
    mesh.identify_boundaries()
    return mesh


def test_riks_arc_length_mechanics_enforces_augmented_constraint():
    from phast.boundary_conditions import BoundaryConditions
    from phast.material import Material
    from phast.staggered_solver import SolverConfig, StaggeredSolver

    mesh = _build_triangle_mesh()
    mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
                   eta_residual=1e-6, energy_split="isotropic")
    bcs = BoundaryConditions(mesh.n_nodes, device="cpu", dtype=torch.float64)
    bcs.fix(mesh.node_sets["bottom"], 0)
    bcs.fix(mesh.node_sets["bottom"], 1)
    top = torch.argmax(mesh.nodes[:, 1]).reshape(1)
    bcs.add(top, 1, 1.0)

    cfg = SolverConfig(
        solver_type="quasi_static",
        static_tol=1e-10,
        static_max_iter=25,
        stagger_tol=1e-8,
        max_stagger=4,
        use_multigrid=False,
        backend="scipy",
        enable_damage=False,
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)
    solver.f_ext[:, 1] = 1.0e-9
    f_ext_ref = solver.f_ext.clone()
    u_prev = solver.u.clone()
    ds = 1.0e-4
    alpha = 1.0e-6

    lam, converged, _ = solver.step_mechanics_arc_length(
        lambda_prev=0.0,
        lambda_init=ds,
        ds=ds,
        alpha=alpha,
        u_prev=u_prev,
    )

    assert converged
    assert lam > 0.0
    delta_u = solver.u - u_prev
    constraint = ((delta_u * delta_u).sum().item() / delta_u.numel()
                  + (alpha * lam) ** 2)
    assert constraint == pytest.approx(ds * ds, rel=1e-6, abs=1e-12)
    assert abs(getattr(solver.mechanics, "last_arc_length_residual")) < 1e-8
    assert torch.allclose(
        getattr(solver, "_last_arc_f_ext_active"), lam * f_ext_ref)


def test_riks_arc_length_stagger_nonconvergence_raises_by_default():
    from phast.boundary_conditions import BoundaryConditions
    from phast.material import Material
    from phast.staggered_solver import SolverConfig, StaggeredSolver

    mesh = _build_triangle_mesh()
    mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
                   eta_residual=1e-6, energy_split="isotropic")
    bcs = BoundaryConditions(mesh.n_nodes, device="cpu", dtype=torch.float64)
    cfg = SolverConfig(
        solver_type="quasi_static",
        max_stagger=2,
        stagger_tol=0.0,
        fail_on_stagger_nonconvergence=True,
        enable_damage=True,
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)

    def _step_mechanics_arc_length(self, **kwargs):
        self.u = self.u + 1.0e-3
        self.bcs.load_factor = float(kwargs["lambda_init"])
        return float(kwargs["lambda_init"]), True, 1

    def _step_compute_driving_force(self, strain=None):
        return torch.zeros_like(self.H_nodal)

    def _step_solve_damage(self, d_prev_step=None):
        self.d = self.d + 1.0e-3

    solver.step_mechanics_arc_length = types.MethodType(
        _step_mechanics_arc_length, solver)
    solver.step_compute_driving_force = types.MethodType(
        _step_compute_driving_force, solver)
    solver.step_solve_damage = types.MethodType(_step_solve_damage, solver)

    with pytest.raises(RuntimeError, match="Arc-length stagger loop did not converge"):
        solver.step_full_arc_length(
            lambda_prev=0.0,
            lambda_init=1.0e-3,
            ds=1.0e-3,
            alpha=1.0,
            u_prev=solver.u.clone(),
        )


def test_riks_arc_length_stagger_nonconvergence_warns_in_diagnostic_mode():
    from phast.boundary_conditions import BoundaryConditions
    from phast.material import Material
    from phast.staggered_solver import SolverConfig, StaggeredSolver

    mesh = _build_triangle_mesh()
    mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
                   eta_residual=1e-6, energy_split="isotropic")
    bcs = BoundaryConditions(mesh.n_nodes, device="cpu", dtype=torch.float64)
    cfg = SolverConfig(
        solver_type="quasi_static",
        max_stagger=2,
        stagger_tol=0.0,
        fail_on_stagger_nonconvergence=False,
        enable_damage=True,
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)

    def _step_mechanics_arc_length(self, **kwargs):
        self.u = self.u + 1.0e-3
        self.bcs.load_factor = float(kwargs["lambda_init"])
        return float(kwargs["lambda_init"]), True, 1

    def _step_compute_driving_force(self, strain=None):
        return torch.zeros_like(self.H_nodal)

    def _step_solve_damage(self, d_prev_step=None):
        self.d = self.d + 1.0e-3

    solver.step_mechanics_arc_length = types.MethodType(
        _step_mechanics_arc_length, solver)
    solver.step_compute_driving_force = types.MethodType(
        _step_compute_driving_force, solver)
    solver.step_solve_damage = types.MethodType(_step_solve_damage, solver)

    with pytest.warns(RuntimeWarning, match="Arc-length stagger loop did not converge"):
        solver.step_full_arc_length(
            lambda_prev=0.0,
            lambda_init=1.0e-3,
            ds=1.0e-3,
            alpha=1.0,
            u_prev=solver.u.clone(),
        )
