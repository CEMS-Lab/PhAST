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
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from ..config.config import (
    ProblemConfig, GeometryConfig, MaterialConfig, BoundaryConditionEntry,
    InitialConditionsConfig, LoadingConfig, SolverSettings, OutputConfig, DeviceConfig,
    load_config, save_config, resolve_config, _ensure_defaults,
    compute_load_factor,
)
from ..solid_mechanics_runner import run_solid_mechanics_config
from ..solvers.staggered_solver import StaggeredSolver
from ..workflow.execution import (
    WorkflowExecutionError,
    _absolutize_legacy_mesh_path,
    _quasistatic_fracture_legacy_yaml,
    _schema_v2_solid_mechanics_legacy_yaml,
)


_DOF_MAP = {'x': [0], 'y': [1], 'xy': [0, 1], 'yx': [0, 1]}
_BC_KIND_ALIASES = {
    'fixed': 'fix',
    'dirichlet': 'prescribe',
    'displacement': 'prescribe',
    'prescribed_displacement': 'prescribe',
    'phase_field_dirichlet': 'pf_dirichlet',
}
_WORKFLOW_SOLVER_PARAMETERS = {'device', 'example'}


def _parse_dof(dof):
    if isinstance(dof, int):
        return [dof]
    if isinstance(dof, str):
        key = dof.lower().replace(' ', '')
        if key in _DOF_MAP:
            return _DOF_MAP[key]
        raise ValueError(f"Unknown dof '{dof}'. Use 'x', 'y', 'xy', or 0/1.")
    return list(dof)


def _normalized_bc_kind(kind):
    key = str(kind).lower().strip()
    return _BC_KIND_ALIASES.get(key, key)


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

    def mesh(self, path):
        """Use an imported mesh file instead of a built-in generator."""
        self._config.geometry = GeometryConfig(mesh_path=str(path))
        return self

    def region(self, name, kind='region', **selector):
        """Declare a workflow region used by materials, BCs, and outputs."""
        regions = getattr(self._config, '_workflow_regions', [])
        regions.append({'name': name, 'kind': kind, 'selector': dict(selector)})
        self._config._workflow_regions = regions
        return self

    def material(self, preset='miehe_tension', region=None, **overrides):
        """Set material from a preset name with optional overrides."""
        self._config.material = MaterialConfig(preset=preset, overrides=overrides)
        if region is not None:
            self._config.material._workflow_region = region
        return self

    def initial_condition(self, field='damage', *, region=None, nodes=None,
                          value=1.0, **parameters):
        """Add an initial condition supported by the existing config path."""
        if field != 'damage':
            raise ValueError(
                "Only damage initial conditions are supported by Problem.initial_condition()"
            )
        if region is None and nodes is None:
            raise ValueError("Problem.initial_condition() requires region or nodes")
        if self._config.initial_conditions is None:
            self._config.initial_conditions = InitialConditionsConfig()
        if self._config.initial_conditions.preseed_damage is None:
            self._config.initial_conditions.preseed_damage = []
        entry = dict(parameters)
        if nodes is not None:
            entry['nodes'] = nodes
        else:
            entry['nodes'] = region
        entry['value'] = value
        self._config.initial_conditions.preseed_damage.append(entry)
        return self

    def boundary_condition(self, kind, *, region, dof=None, value=None, name=None,
                           **parameters):
        """Add a boundary condition using workflow-domain naming."""
        kind = _normalized_bc_kind(kind)
        if dof is None and kind in {'fix', 'prescribe', 'neumann', 'traction'}:
            raise ValueError(
                f"Boundary condition {kind!r} requires dof='x', 'y', or 'xy'."
            )
        components = _parse_dof(dof) if dof is not None else [None]
        for component in components:
            entry = BoundaryConditionEntry(
                nodes=region,
                type=kind,
                component=component,
                value=value,
                **parameters,
            )
            if name is not None:
                entry._workflow_name = name
            self._config.boundary_conditions.append(entry)
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
                dt=1e-5, prestrain_displacement=0.0,
                coupled_prestrain=False):
        """Set the loading protocol."""
        self._config.loading = LoadingConfig(
            protocol=protocol, t_total=t_total, num_steps=num_steps,
            dt=dt, prestrain_displacement=prestrain_displacement,
            coupled_prestrain=coupled_prestrain)
        return self

    def analysis_step(self, name, *, kind, controls=None,
                      active_boundary_conditions=None, **control_kwargs):
        """Set the primary analysis step using workflow-domain naming."""
        data = dict(controls or {})
        data.update(control_kwargs)
        loading_fields = set(LoadingConfig.__dataclass_fields__)
        loading_data = {key: value for key, value in data.items() if key in loading_fields}
        loading_data.setdefault('protocol', 'simple')
        self._config.loading = LoadingConfig(**loading_data)
        self._config.loading._workflow_step_name = name
        self._config.loading._workflow_controls = data
        self._config.loading._workflow_active_boundary_conditions = tuple(
            active_boundary_conditions or ()
        )
        self._config.solver = SolverSettings(solver_type=kind)
        return self

    def solver(self, solver_type='explicit', **kwargs):
        """Set solver settings."""
        solver_fields = set(SolverSettings.__dataclass_fields__)
        solver_kwargs = {
            key: value for key, value in kwargs.items() if key in solver_fields
        }
        workflow_kwargs = {
            key: value for key, value in kwargs.items() if key not in solver_fields
        }
        unknown = sorted(set(workflow_kwargs) - _WORKFLOW_SOLVER_PARAMETERS)
        if unknown:
            names = ', '.join(unknown)
            raise ValueError(
                f"Unknown solver setting(s): {names}. "
                "Use SolverSettings fields or supported workflow parameters "
                "'device' and 'example'."
            )
        self._config.solver = SolverSettings(solver_type=solver_type, **solver_kwargs)
        if workflow_kwargs:
            self._config.solver._workflow_parameters = workflow_kwargs
        return self

    def output(self, **kwargs):
        """Set output options (h5, gif, plots, fast, print_every, ...)."""
        self._config.output = OutputConfig(**kwargs)
        return self

    def outputs(self, fields=None, histories=None, **kwargs):
        """Set output options using the public workflow noun."""
        if fields is not None and self._config.solver.solver_type != 'solid_mechanics':
            kwargs.setdefault('trajectory', True)
            kwargs.setdefault('h5', True)
        history_specs = list(histories or kwargs.pop('history', []) or [])
        if history_specs:
            first = history_specs[0]
            if isinstance(first, dict):
                if first.get('region') is not None:
                    kwargs.setdefault('reaction_node_set', first['region'])
                component = first.get('component', first.get('dof'))
                if component is not None:
                    kwargs.setdefault('reaction_component', _parse_dof(component)[0])
        output = self.output(**kwargs)
        self._config.output._workflow_fields = list(fields or [])
        self._config.output._workflow_history = history_specs
        return output

    def device(self, device):
        """Set compute device ('cpu', 'cuda', 'mps')."""
        self._config.device = DeviceConfig(device=device)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, output_dir=None, verbose=True, return_result=False):
        """Resolve the config and run the simulation.

        Returns the :class:`StaggeredSolver` instance after completion by
        default. Set ``return_result=True`` with ``output_dir`` to return a
        read-only :class:`phast.Result` for the written run directory.
        """
        cfg = self._config
        _ensure_defaults(cfg)
        if cfg.solver.solver_type == 'solid_mechanics':
            return self._run_solid_mechanics(
                output_dir=output_dir,
                return_result=return_result,
            )
        if self._uses_workflow_fracture_runner():
            return self._run_workflow_fracture(
                output_dir=output_dir,
                return_result=return_result,
            )
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
            solver.pre_strain(coupled=bool(cfg.loading.coupled_prestrain))

        t0 = time.time()
        for step in range(n_steps):
            if solver_config.solver_type != 'explicit' and objs['loading']:
                bcs.load_factor = objs['loading'][step]
            elif solver_config.solver_type == 'explicit':
                bcs.load_factor = compute_load_factor(step, solver.dt, cfg.loading)
            solver.step_full()

            if verbose and step % cfg.output.print_every == 0:
                max_d = solver.d.max().item()
                t_sim = step * solver.dt
                print(f"  {step:5d} | t={t_sim*1e6:7.2f}us | max(d)={max_d:.4f}")
        solver.steps = int(n_steps)

        wall = time.time() - t0
        if verbose:
            print(f"  Done: {wall:.1f}s ({wall/max(n_steps,1)*1000:.1f} ms/step)")
            solver.print_route_report()

        if output_dir:
            self._save_outputs(solver, mesh, output_dir, n_steps)

        if return_result:
            if not output_dir:
                raise ValueError("Problem.run(return_result=True) requires output_dir")
            from ..result import load_result

            return load_result(output_dir)
        return solver

    def _run_solid_mechanics(self, *, output_dir=None, return_result=False):
        spec = self.to_spec()
        try:
            payload = _schema_v2_solid_mechanics_legacy_yaml(spec)
        except WorkflowExecutionError as exc:
            raise ValueError(
                "Problem.run() supports solid_mechanics only for promoted "
                "examples declared with solver(..., example='solid_mechanics.<name>')."
            ) from exc
        with tempfile.TemporaryDirectory(prefix='phast-fluent-solid-') as tmp:
            config_path = Path(tmp) / 'config.yaml'
            import yaml

            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding='utf-8',
            )
            effective_output_dir = output_dir
            if effective_output_dir is None:
                configured = payload.get('output', {}).get('directory')
                effective_output_dir = configured or 'outputs'
                effective_output_dir = Path(effective_output_dir)
                if not effective_output_dir.is_absolute():
                    effective_output_dir = Path.cwd() / effective_output_dir
            rc = run_solid_mechanics_config(
                config_path,
                output_dir=effective_output_dir,
                validate_only=False,
            )
        if return_result:
            from ..result import load_result

            return load_result(effective_output_dir)
        return rc

    def _uses_workflow_fracture_runner(self):
        cfg = self._config
        if cfg.solver.solver_type != 'quasi_static':
            return False
        if getattr(cfg, '_workflow_regions', None):
            return True
        if getattr(cfg.material, '_workflow_region', None):
            return True
        if getattr(cfg.loading, '_workflow_active_boundary_conditions', None):
            return True
        if getattr(cfg.output, '_workflow_history', None):
            return True
        if getattr(cfg.output, '_workflow_fields', None):
            return True
        return False

    def _run_workflow_fracture(self, *, output_dir=None, return_result=False):
        if output_dir is None:
            raise ValueError(
                "Contract-style Problem.run() for quasi_static fracture requires "
                "output_dir so existing run_config artifacts can be inspected."
            )
        spec = self.to_spec()
        from ..workflow.validation import validate_problem_spec

        issues = validate_problem_spec(spec)
        if issues:
            detail = '; '.join(issue.message for issue in issues)
            raise ValueError(f"Invalid workflow ProblemSpec: {detail}")
        try:
            payload = _quasistatic_fracture_legacy_yaml(spec)
        except WorkflowExecutionError as exc:
            raise ValueError(
                "Problem.run() supports contract-style quasi_static fracture "
                "only when the fluent workflow lowers cleanly to v1 run_config."
            ) from exc
        _absolutize_legacy_mesh_path(payload, Path.cwd())
        with tempfile.TemporaryDirectory(prefix='phast-fluent-fracture-') as tmp:
            config_path = Path(tmp) / 'config.yaml'
            import yaml

            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding='utf-8',
            )
            cmd = [
                sys.executable,
                '-m',
                'phast',
                'run',
                str(config_path),
                '--output_dir',
                os.fspath(output_dir),
            ]
            completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise ValueError(
                "Existing run_config runner failed while executing the "
                f"contract-style quasi_static fracture workflow "
                f"(exit code {completed.returncode})."
            )
        if return_result:
            from ..result import load_result

            return load_result(output_dir)
        return int(completed.returncode)

    def to_spec(self):
        """Return this problem as the internal workflow ProblemSpec."""
        from ..workflow import problem_spec_from_problem

        return problem_spec_from_problem(self)

    def validate_setup(self):
        """Validate workflow setup that can be checked before solver creation."""
        spec = self.to_spec()
        if spec.mesh is None or spec.mesh.path is None:
            return {"mesh": None, "regions": {}}
        from ..region_resolution import resolve_regions

        return resolve_regions(spec.mesh.path, spec.regions)

    def preview(self, *, output):
        """Write a static, non-interactive setup preview artifact."""
        from ..setup_preview import write_setup_preview

        return write_setup_preview(self.to_spec(), output)

    def plot_setup(self, *, output):
        """Alias for :meth:`preview` for GUI-style setup confirmation."""
        return self.preview(output=output)

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

    def _has_workflow_metadata(self):
        cfg = self._config
        return any(
            (
                bool(getattr(cfg, '_workflow_regions', None)),
                bool(getattr(cfg.material, '_workflow_region', None)),
                bool(getattr(cfg.loading, '_workflow_step_name', None)),
                bool(getattr(cfg.loading, '_workflow_controls', None)),
                bool(getattr(cfg.loading, '_workflow_active_boundary_conditions', None)),
                bool(getattr(cfg.solver, '_workflow_parameters', None)),
                bool(getattr(cfg.output, '_workflow_fields', None)),
                bool(getattr(cfg.output, '_workflow_history', None)),
                any(
                    bool(getattr(entry, '_workflow_name', None))
                    for entry in (cfg.boundary_conditions or [])
                ),
            )
        )

    def save(self, path):
        """Save the problem definition to a YAML file."""
        if self._has_workflow_metadata():
            import yaml
            from ..workflow import problem_spec_to_schema_v2_dict

            payload = problem_spec_to_schema_v2_dict(self.to_spec())
            os.makedirs(os.path.dirname(os.fspath(path)) or '.', exist_ok=True)
            with open(path, 'w') as f:
                yaml.safe_dump(payload, f, sort_keys=False)
            return
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
