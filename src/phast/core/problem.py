"""
Fluent API for setting up and running phase-field fracture simulations.

Usage::

    from phast import Problem

    solver = (Problem('Miehe SENT')
        .geometry('rectangular_sent', W=100, H=40, a=50, h_crack=0.5, h_coarse=4)
        .material('glass_borden', l0=0.5, energy_split='spectral')
        .fix('left', dof='x')
        .neumann('top', dof='y', value=1.0)
        .neumann('bottom', dof='y', value=-1.0)
        .loading(protocol='simple', t_total=80e-6)
        .solver(dt_safety=0.8, use_multigrid=True)
        .device('cpu')
        .run())

    # YAML round-trip
    prob = Problem('test').geometry(...)
    prob.save('my_problem.yaml')
    prob2 = Problem.from_yaml('my_problem.yaml')
"""

import math
import os
import time

from ..config.config import (
    ProblemConfig, GeometryConfig, MaterialConfig, BoundaryConditionEntry,
    LoadingConfig, SolverSettings, OutputConfig, DeviceConfig,
    load_config, save_config, resolve_config, _ensure_defaults,
    compute_load_factor,
)
from ..solvers.staggered_solver import StaggeredSolver


_DOF_MAP = {'x': [0], 'y': [1], 'xy': [0, 1], 'yx': [0, 1]}


def _parse_dof(dof):
    if isinstance(dof, int):
        return [dof]
    if isinstance(dof, str):
        key = dof.lower().replace(' ', '')
        if key in _DOF_MAP:
            return _DOF_MAP[key]
        raise ValueError(f"Unknown dof '{dof}'. Use 'x', 'y', 'xy', or 0/1.")
    return list(dof)


class Problem:
    """Fluent builder for phase-field fracture problems.

    Wraps :class:`ProblemConfig` and :func:`resolve_config` so that a
    complete simulation can be set up in a single chained call.
    """

    def __init__(self, name='Phase-Field Problem'):
        self._config = ProblemConfig(name=name)
        _ensure_defaults(self._config)

    # ------------------------------------------------------------------
    # Fluent setters (each returns self for chaining)
    # ------------------------------------------------------------------

    def geometry(self, type, **params):
        """Set the mesh generator and its parameters."""
        self._config.geometry = GeometryConfig(type=type, parameters=params)
        return self

    def material(self, preset='miehe_tension', **overrides):
        """Set material from a preset name with optional overrides."""
        self._config.material = MaterialConfig(preset=preset, overrides=overrides)
        return self

    def fix(self, nodes, dof='xy'):
        """Fix displacement on a named node set (homogeneous Dirichlet)."""
        for c in _parse_dof(dof):
            self._config.boundary_conditions.append(
                BoundaryConditionEntry(nodes=nodes, type='fix', component=c))
        return self

    def prescribe(self, nodes, dof, value=1.0):
        """Prescribe displacement on a named node set."""
        for c in _parse_dof(dof):
            self._config.boundary_conditions.append(
                BoundaryConditionEntry(nodes=nodes, type='prescribe',
                                       component=c, value=value))
        return self

    def neumann(self, nodes, dof, value=1.0):
        """Apply traction on a named node set."""
        for c in _parse_dof(dof):
            self._config.boundary_conditions.append(
                BoundaryConditionEntry(nodes=nodes, type='neumann',
                                       component=c, value=value))
        return self

    def loading(self, protocol='simple', t_total=0.0, num_steps=100,
                dt=1e-5, prestrain_displacement=0.0):
        """Set the loading protocol."""
        self._config.loading = LoadingConfig(
            protocol=protocol, t_total=t_total, num_steps=num_steps,
            dt=dt, prestrain_displacement=prestrain_displacement)
        return self

    def solver(self, solver_type='explicit', **kwargs):
        """Set solver settings."""
        self._config.solver = SolverSettings(solver_type=solver_type, **kwargs)
        return self

    def output(self, **kwargs):
        """Set output options (h5, gif, plots, fast, print_every, ...)."""
        self._config.output = OutputConfig(**kwargs)
        return self

    def device(self, device):
        """Set compute device ('cpu', 'cuda', 'mps')."""
        self._config.device = DeviceConfig(device=device)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, output_dir=None, verbose=True):
        """Resolve the config and run the simulation.

        Returns the :class:`StaggeredSolver` instance after completion.
        """
        cfg = self._config
        _ensure_defaults(cfg)
        objs = resolve_config(cfg)

        mesh = objs['mesh']
        mat = objs['material']
        bcs = objs['bcs']
        solver_config = objs['solver_config']
        ctx = objs['ctx']

        solver = StaggeredSolver(mesh, mat, bcs, config=solver_config, ctx=ctx)

        neumann_forces = bcs.get_neumann_forces(mesh)
        if neumann_forces is not None:
            solver.f_ext = neumann_forces

        n_steps = cfg.loading.num_steps
        if (cfg.solver.solver_type == 'explicit'
                and n_steps == 0 and cfg.loading.t_total > 0):
            n_steps = int(math.ceil(cfg.loading.t_total / solver.dt))
        solver.config.num_steps = n_steps

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print(f"Problem: {cfg.name}")
            print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elems} elements")
            print(f"  Steps: {n_steps}, dt={solver.dt:.4e}")

        if cfg.loading.protocol == 'two_step_prestrain':
            solver.pre_strain()

        t0 = time.time()
        for step in range(n_steps):
            if solver_config.solver_type != 'explicit' and objs['loading']:
                bcs.load_factor = objs['loading'][step]
            elif solver_config.solver_type == 'explicit':
                bcs.load_factor = compute_load_factor(step, solver.dt, cfg.loading)
            solver.step_full()
            solver.steps = step + 1

            if verbose and step % cfg.output.print_every == 0:
                max_d = solver.d.max().item()
                t_sim = step * solver.dt
                print(f"  {step:5d} | t={t_sim*1e6:7.2f}us | max(d)={max_d:.4f}")

        wall = time.time() - t0
        if verbose:
            print(f"  Done: {wall:.1f}s ({wall/max(n_steps,1)*1000:.1f} ms/step)")

        if output_dir:
            self._save_outputs(solver, mesh, output_dir, n_steps)

        return solver

    def _save_outputs(self, solver, mesh, output_dir, n_steps):
        import matplotlib
        matplotlib.use('Agg')
        from ..utils.visualization import plot_field, plot_initial_conditions
        from ..utils.io_utils import save_run_metadata
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_field(mesh, solver.d, title=f'Final Damage (step {n_steps})',
                   cmap='inferno', vmin=0, vmax=1, ax=ax)
        fig.savefig(os.path.join(output_dir, 'damage_final.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        save_config(self._config, os.path.join(output_dir, 'config.yaml'))

        save_run_metadata(
            output_dir,
            problem_name=self._config.name,
            device=str(solver.device),
            material=solver.material,
            mesh=mesh,
            solver_config={
                'solver_type': solver.config.solver_type,
                'num_steps': n_steps,
                'dt': solver.dt if solver.dt else 0,
            },
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path):
        """Save the problem definition to a YAML file."""
        save_config(self._config, path)

    @classmethod
    def from_yaml(cls, path):
        """Load a Problem from a YAML config file."""
        cfg = load_config(path)
        p = cls.__new__(cls)
        p._config = cfg
        return p

    @property
    def config(self):
        """Access the underlying ProblemConfig."""
        return self._config

    def __repr__(self):
        g = self._config.geometry
        m = self._config.material
        return (f"Problem('{self._config.name}', "
                f"geometry='{g.type}', material='{m.preset}')")
