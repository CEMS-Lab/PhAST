"""
Regression test for issue #196 -- StaggeredSolver was constructing
QuasiStaticSolver with a hard-coded ``backend='auto'`` even when the
caller had set a different backend in SolverConfig (or the YAML
SolverSettings.backend field).

The fix wires SolverConfig.backend (default 'auto') through
``_build_mechanics_solver`` to ``QuasiStaticSolver(backend=...)``.

This test asserts the inner solver receives the value we passed in,
for both an explicit non-default ('scipy') and the default ('auto').
"""

import os
import tempfile
import types

import pytest
import torch


def _build_mesh():
    """Single equilateral T3 mesh -- mirrors the equilateral_mesh
    fixture in test_physics.py without importing pytest fixtures."""
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
    d = tempfile.mkdtemp()
    geo_path = os.path.join(d, 'tri.geo')
    msh_path = os.path.join(d, 'tri.msh')
    with open(geo_path, 'w') as f:
        f.write(geo)

    import gmsh
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

    mesh = FEMMesh(msh_path, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()
    return mesh


@pytest.mark.parametrize("backend", ["auto", "scipy"])
def test_staggered_propagates_backend_to_quasistatic(backend):
    """SolverConfig.backend must reach QuasiStaticSolver(backend=...)."""
    from phast.staggered_solver import StaggeredSolver, SolverConfig
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions

    mesh = _build_mesh()
    mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
                   eta_residual=1e-6)
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu')
    bcs.fix(torch.tensor([0]), 0)
    bcs.fix(torch.tensor([0]), 1)

    cfg = SolverConfig(solver_type='quasi_static', backend=backend)
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)

    inner = solver._build_mechanics_solver()
    assert hasattr(inner, 'backend'), (
        "QuasiStaticSolver should expose a 'backend' attribute "
        "(see mechanics_solver.py:606)")
    assert inner.backend == backend, (
        f"Expected QuasiStaticSolver to receive backend={backend!r}, "
        f"got {inner.backend!r} -- regression of #196 "
        f"(hard-coded 'auto' in _build_mechanics_solver).")


def test_solver_config_backend_default_is_auto():
    """Default value preserves prior behaviour."""
    from phast.staggered_solver import SolverConfig
    cfg = SolverConfig()
    assert cfg.backend == 'auto'


def test_quasistatic_nonconvergence_raises_by_default():
    """A non-converged mechanics subsolve must not be accepted silently."""
    from phast.staggered_solver import StaggeredSolver, SolverConfig
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions

    class NonconvergedMechanics:
        def solve(self, d, f_ext, bc_mask, bc_vals, u_init=None):
            return u_init.clone(), False, 7

    mesh = _build_mesh()
    mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
                   eta_residual=1e-6)
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu')
    cfg = SolverConfig(solver_type='quasi_static')
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)
    solver.mechanics = NonconvergedMechanics()

    with pytest.raises(RuntimeError, match='did not converge'):
        solver.step_mechanics()
    assert solver._last_mechanics_converged is False
    assert solver._last_mechanics_iter == 7


def test_monolithic_nonconvergence_raises_by_default():
    """A failed monolithic coupled solve must not be accepted silently."""
    from phast.staggered_solver import StaggeredSolver, SolverConfig
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions

    class NonconvergedMonolithic:
        def solve(self, u, d, bc_mask, bc_vals, f_ext=None, d_prev=None):
            return u.clone(), d.clone(), False, 5

    mesh = _build_mesh()
    mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
                   eta_residual=1e-6)
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu')
    cfg = SolverConfig(solver_type='monolithic')
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)
    solver.mechanics = NonconvergedMonolithic()

    with pytest.raises(RuntimeError, match='Monolithic mechanics/damage solve'):
        solver.step_full()
    assert solver._last_mechanics_converged is False
    assert solver._last_mechanics_iter == 5


def test_quasistatic_stagger_nonconvergence_raises_by_default():
    """A nonconverged implicit stagger step must be rejected, not warned."""
    from phast.staggered_solver import StaggeredSolver, SolverConfig
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions

    mesh = _build_mesh()
    mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
                   eta_residual=1e-6)
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu')
    cfg = SolverConfig(
        solver_type='quasi_static',
        max_stagger=8,
        stagger_tol=0.0,
        fail_on_stagger_nonconvergence=True,
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)

    def _step_mechanics(self):
        self.u = self.u + 1.0e-3

    def _step_compute_driving_force(self, strain=None):
        return torch.zeros_like(self.H_nodal)

    def _step_solve_damage(self, d_prev_step=None):
        self.d = self.d + 1.0e-3

    solver.step_mechanics = types.MethodType(_step_mechanics, solver)
    solver.step_compute_driving_force = types.MethodType(
        _step_compute_driving_force, solver)
    solver.step_solve_damage = types.MethodType(_step_solve_damage, solver)

    with pytest.raises(RuntimeError, match='Stagger loop did not converge'):
        solver.step_full()


def test_quasistatic_stagger_nonconvergence_warns_in_diagnostic_mode():
    """Diagnostic mode preserves the historical warning-only behaviour."""
    from phast.staggered_solver import StaggeredSolver, SolverConfig
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions

    mesh = _build_mesh()
    mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
                   eta_residual=1e-6)
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu')
    cfg = SolverConfig(
        solver_type='quasi_static',
        max_stagger=8,
        stagger_tol=0.0,
        fail_on_stagger_nonconvergence=False,
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)

    def _step_mechanics(self):
        self.u = self.u + 1.0e-3

    def _step_compute_driving_force(self, strain=None):
        return torch.zeros_like(self.H_nodal)

    def _step_solve_damage(self, d_prev_step=None):
        self.d = self.d + 1.0e-3

    solver.step_mechanics = types.MethodType(_step_mechanics, solver)
    solver.step_compute_driving_force = types.MethodType(
        _step_compute_driving_force, solver)
    solver.step_solve_damage = types.MethodType(_step_solve_damage, solver)

    with pytest.warns(RuntimeWarning, match='Stagger loop did not converge'):
        solver.step_full()


@pytest.mark.parametrize("solver_type", [
    "static", "quasi_static", "quasi_static_legacy", "lbfgs", "monolithic",
])
def test_implicit_solver_config_defaults_damage_to_jacobi(solver_type):
    """Implicit paths should not silently resolve None+use_multigrid into AMG."""
    from phast.staggered_solver import SolverConfig

    cfg = SolverConfig(solver_type=solver_type)
    assert cfg.preconditioner == 'jacobi'


def test_explicit_solver_config_preserves_legacy_preconditioner_resolution():
    """Explicit dynamics keeps the historical auto/multigrid resolution path."""
    from phast.staggered_solver import SolverConfig

    cfg = SolverConfig(solver_type='explicit')
    assert cfg.preconditioner is None


def test_staggered_allows_guarded_quasistatic_j2_plasticity():
    """The first supported PF-plasticity slice must wire sparse J2 mechanics."""
    from phast.staggered_solver import StaggeredSolver, SolverConfig
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions

    mesh = _build_mesh()
    mat = Material(
        E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
        eta_residual=1e-6,
        plasticity_model='j2_isotropic',
        yield_stress=250.0,
        hardening_modulus=1000.0,
        hardening_type='linear_iso',
    )
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu')
    bcs.fix(torch.tensor([0]), 0)
    bcs.fix(torch.tensor([0]), 1)
    cfg = SolverConfig(
        solver_type='quasi_static',
        backend='auto',
        static_max_iter=8,
        damage_max_iter=50,
        max_stagger=2,
    )

    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)

    assert solver.plasticity_operator is not None
    assert solver.ductile_coupling is not None
    assert solver.mechanics.plasticity_operator is solver.plasticity_operator


@pytest.mark.parametrize("solver_type", ["explicit", "static"])
def test_staggered_rejects_unsupported_plasticity_solver_types(solver_type):
    from phast.staggered_solver import StaggeredSolver, SolverConfig
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions

    mesh = _build_mesh()
    mat = Material(
        E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
        eta_residual=1e-6,
        plasticity_model='j2_isotropic',
        yield_stress=250.0,
        hardening_modulus=1000.0,
        hardening_type='linear_iso',
    )
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu')
    cfg = SolverConfig(solver_type=solver_type)

    with pytest.raises(NotImplementedError, match="quasi_static"):
        StaggeredSolver(mesh, mat, bcs, config=cfg)


def test_staggered_rejects_j2_cg_backend():
    from phast.staggered_solver import StaggeredSolver, SolverConfig
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions

    mesh = _build_mesh()
    mat = Material(
        E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
        eta_residual=1e-6,
        plasticity_model='j2_isotropic',
        yield_stress=250.0,
        hardening_modulus=1000.0,
        hardening_type='linear_iso',
    )
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu')
    cfg = SolverConfig(solver_type='quasi_static', backend='cg')

    with pytest.raises(NotImplementedError, match="backend='cg'"):
        StaggeredSolver(mesh, mat, bcs, config=cfg)


def test_staggered_quasistatic_j2_step_full_and_state_roundtrip():
    from phast.staggered_solver import StaggeredSolver, SolverConfig
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions
    from phast.mesh import FEMMesh

    nodes = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    mesh = FEMMesh.from_tensors(nodes, elements, device='cpu', dtype=torch.float64)
    mat = Material(
        E=210000.0, nu=0.3, Gc=2.7, l0=0.1,
        eta_residual=1e-7,
        energy_split='amor',
        pf_model='AT2',
        plasticity_model='j2_isotropic',
        yield_stress=250.0,
        hardening_modulus=5000.0,
        hardening_type='linear_iso',
    )
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu')
    bcs.fix(torch.tensor([0, 3]), 0)
    bcs.fix(torch.tensor([0, 3]), 1)
    bcs.add(torch.tensor([1, 2]), 0, 5.0e-3)
    cfg = SolverConfig(
        solver_type='quasi_static',
        backend='auto',
        static_tol=1.0e-6,
        static_max_iter=15,
        damage_tol=1.0e-8,
        damage_max_iter=100,
        bounds_method='post_clamp',
        preconditioner='jacobi',
        max_stagger=8,
        stagger_tol=1.0e-8,
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)

    psi = solver.step_full()

    assert torch.isfinite(psi).all()
    assert solver.plasticity_operator.state.eps_p_eq.max().item() > 0.0
    assert solver.H_elem.max().item() > 0.0
    assert solver.d.max().item() >= 0.0
    saved = solver.get_state()
    eps_saved = saved['plasticity_state']['eps_p_eq'].clone()
    solver.plasticity_operator.state.eps_p_eq.add_(1.0)
    solver.set_state(saved)
    assert torch.allclose(solver.plasticity_operator.state.eps_p_eq, eps_saved)
