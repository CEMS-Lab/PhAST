"""Human-readable YAML capability summary for ``phast`` configs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .config_validation import format_errors, validate_config_file
from ..utils.units import parse_quantity


IMPLICIT_SOLVERS = {'quasi_static', 'quasi_static_legacy', 'static', 'lbfgs'}


def _set_fields(obj: Any) -> list[str]:
    if not is_dataclass(obj):
        return []
    out = []
    for fld in fields(obj):
        value = getattr(obj, fld.name)
        if value not in (None, '', [], {}, False):
            out.append(fld.name)
    return out


def _material_value(material: Any, key: str, default: str = 'default') -> Any:
    value = getattr(material, key, None)
    if value is not None:
        return value
    overrides = getattr(material, 'overrides', {}) or {}
    return overrides.get(key, default)


def _fmt_bool(flag: bool) -> str:
    return 'yes' if flag else 'no'


def _fmt_acceptance_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    if isinstance(value, dict):
        keys = ', '.join(str(key) for key in sorted(value))
        return f"{len(value)} fields ({keys})" if keys else "empty mapping"
    if isinstance(value, list):
        return f"{len(value)} entries"
    return type(value).__name__


def _as_length_mm(value: Any) -> float | None:
    try:
        return parse_quantity(value, 'length')
    except (TypeError, ValueError):
        return None


def _declared_mesh_sizes_mm(geometry: Any) -> list[float]:
    """Collect declared mesh-size values without resolving/generating a mesh."""
    sizes: list[float] = []
    params = getattr(geometry, 'parameters', {}) or {}
    for key in ('h', 'he', 'lc', 'h_crack', 'h_coarse', 'element_size',
                'mesh_size'):
        if key in params:
            size = _as_length_mm(params[key])
            if size is not None and size > 0:
                sizes.append(size)

    mesh = getattr(geometry, 'mesh', None) or {}
    element_size = mesh.get('element_size') if isinstance(mesh, dict) else None
    if isinstance(element_size, dict):
        default = _as_length_mm(element_size.get('default'))
        if default is not None and default > 0:
            sizes.append(default)
        for rule in element_size.get('refined', []) or []:
            if isinstance(rule, dict):
                size = _as_length_mm(rule.get('size'))
                if size is not None and size > 0:
                    sizes.append(size)
    else:
        size = _as_length_mm(element_size)
        if size is not None and size > 0:
            sizes.append(size)

    return sizes


def _mesh_path_candidates(config_path: str, mesh_path: str) -> list[Path]:
    path = Path(mesh_path).expanduser()
    if path.is_absolute():
        return [path]
    return [Path(config_path).resolve().parent / path, path]


def _mesh_path_exists(config_path: str, mesh_path: str) -> bool:
    return any(path.exists() for path in _mesh_path_candidates(config_path, mesh_path))


def _yaml_declared_node_sets(cfg: Any) -> set[str]:
    """Collect node-set names visible in YAML without resolving a mesh."""
    names: set[str] = set()
    geom = cfg.geometry
    if geom.named_groups:
        names.update(str(name) for name in geom.named_groups)
    if geom.primitives:
        for name in geom.primitives:
            names.add(str(name))
            names.add(f"{name}.boundary")
            names.add(f"{name}.centre")

    for bc in cfg.boundary_conditions or []:
        if bc.nodes:
            names.add(str(bc.nodes))
        if bc.master:
            names.add(str(bc.master))

    initial = cfg.initial_conditions
    if initial and initial.preseed_notch_nodesets:
        names.update(str(name) for name in initial.preseed_notch_nodesets)
    return names


def build_explanation(config_path: str) -> tuple[str, int]:
    """Return ``(report, exit_code)`` for a YAML config path.

    The command intentionally stops at ``load_config``. It does not call
    ``resolve_config`` because that may generate/load meshes and allocate
    solver objects, which is too heavy for a dry-run explanation command.
    """
    raw, errors = validate_config_file(config_path)
    if errors:
        return format_errors(errors, config_path), 2

    cfg = load_config(config_path)
    geom = cfg.geometry
    mat = cfg.material
    loading = cfg.loading
    solver = cfg.solver
    output = cfg.output
    device = cfg.device
    initial = cfg.initial_conditions
    declared_node_sets = _yaml_declared_node_sets(cfg)

    lines: list[str] = []
    warnings: list[str] = []

    solver_type = solver.solver_type
    pf_model = _material_value(mat, 'pf_model', 'AT2')
    energy_split = _material_value(mat, 'energy_split', 'spectral')
    raw_solver = raw.get('solver') if isinstance(raw, dict) else {}
    raw_solver = raw_solver if isinstance(raw_solver, dict) else {}

    lines.append(f"Config: {config_path}")
    lines.append(f"Schema version: {cfg.schema_version}")
    lines.append(f"Problem: {cfg.name}")
    if cfg.reference:
        lines.append(f"Reference: {cfg.reference}")
    if cfg.example:
        lines.append(f"Provenance example: {cfg.example}")
    if cfg.acceptance:
        lines.append("Acceptance metadata: yes")

    lines.append("")
    lines.append("Geometry")
    if geom.mesh_path:
        lines.append(f"  source: external mesh ({geom.mesh_path})")
        if not _mesh_path_exists(config_path, geom.mesh_path):
            warnings.append(
                "geometry.mesh_path does not exist; checked relative to the "
                "config file and current working directory. Fix this before "
                "launching external-mesh runs."
            )
    elif geom.primitives:
        primitive_count = len(geom.primitives)
        group_count = len(geom.named_groups or {})
        lines.append(f"  source: declarative geometry DSL ({primitive_count} primitives)")
        lines.append(f"  units: {geom.units}; named groups: {group_count}")
        if geom.domain:
            lines.append("  boolean domain recipe: yes")
        if geom.mesh:
            lines.append("  mesh refinement recipe: yes")
    else:
        lines.append(f"  source: built-in generator ({geom.type})")
        if geom.parameters:
            params = ', '.join(f"{k}={v}" for k, v in sorted(geom.parameters.items()))
            lines.append(f"  parameters: {params}")

    lines.append("")
    lines.append("Material And Fracture Model")
    lines.append(f"  preset: {mat.preset or 'inline properties'}")
    lines.append(f"  phase-field model: {pf_model}")
    lines.append(f"  energy split: {energy_split}")
    lines.append(f"  E={_material_value(mat, 'E')}, nu={_material_value(mat, 'nu')}, "
                 f"Gc={_material_value(mat, 'Gc')}, l0={_material_value(mat, 'l0')}")
    if pf_model == 'AT1':
        lines.append("  AT1 threshold fields: "
                     f"sigma_ts={_material_value(mat, 'sigma_ts')}, "
                     f"cubic_s={_material_value(mat, 'cubic_s')}")
    if pf_model not in {'AT1', 'AT2'}:
        warnings.append(f"Unsupported phase-field model for production runs: {pf_model!r}.")
    if 'schema_version' not in raw:
        warnings.append(
            "No top-level schema_version is set; add schema_version: 1 "
            "to make future migrations explicit."
        )

    l0_mm = _as_length_mm(_material_value(mat, 'l0', None))
    mesh_sizes = _declared_mesh_sizes_mm(geom)
    if l0_mm is not None and l0_mm > 0 and mesh_sizes:
        h_min = min(mesh_sizes)
        ratio = h_min / l0_mm
        lines.append(f"  finest declared h/l0: {ratio:.3g} ({h_min:g} / {l0_mm:g})")
        if ratio > 0.5:
            warnings.append(
                f"Finest declared mesh size h={h_min:g} mm gives h/l0={ratio:.3g}; "
                "phase-field fracture validation usually needs h <= l0/2 "
                "(often l0/4 near the crack path)."
            )

    lines.append("")
    lines.append("Solver")
    lines.append(f"  type: {solver_type}")
    lines.append(f"  backend: {getattr(solver, 'backend', 'auto')}")
    lines.append(f"  preconditioner: {solver.preconditioner or 'default'}")
    lines.append(f"  damage enabled: {_fmt_bool(solver.enable_damage)}")
    lines.append(f"  bounds method: {solver.bounds_method}")
    lines.append(f"  history update: {solver.H_update_method}")
    lines.append(f"  stagger criterion: {solver.stagger_criterion} "
                 f"(tol={solver.stagger_tol}, max={solver.max_stagger})")
    if pf_model == 'AT1' and solver.bounds_method != 'projected_cg':
        warnings.append(
            "AT1 damage requires projected bound enforcement; runtime solver "
            "construction will switch bounds_method to projected_cg."
        )
    if solver_type in IMPLICIT_SOLVERS and solver.preconditioner in (None, 'auto'):
        lines.append("  implicit damage linear solve: defaults to Jacobi unless overridden")
    if solver_type in IMPLICIT_SOLVERS and solver.preconditioner in {'amg', 'amgx', 'gmg'}:
        warnings.append(
            f"Implicit/quasi-static fracture is configured with "
            f"preconditioner={solver.preconditioner!r}; public validation "
            "validation should use Jacobi unless this run is specifically "
            "testing the multigrid preconditioner."
        )
    if solver_type == 'monolithic':
        warnings.append(
            "Monolithic phase-field solve is experimental; use staggered "
            "quasi_static/static/lbfgs for public implicit validation."
        )
    if (solver_type == 'explicit'
            and raw_solver.get('use_multigrid') is True):
        warnings.append(
            "solver.use_multigrid is ignored by the explicit dynamics path."
        )

    lines.append("")
    lines.append("Loading")
    lines.append(f"  protocol: {loading.protocol}")
    if loading.protocol == 'two_step_prestrain':
        lines.append(
            "  pre-strain: "
            f"DeltaU={loading.prestrain_displacement}, "
            f"coupled={_fmt_bool(loading.coupled_prestrain)}"
        )
    lines.append(f"  steps: {loading.num_steps}")
    if solver_type == 'explicit':
        lines.append(f"  explicit time: dt={loading.dt}, t_total={loading.t_total}, "
                     f"ramp={loading.ramp_type}, t_ramp={loading.t_ramp}")
    else:
        lines.append(f"  quasi-static displacement target: {loading.disp_max}")
        if loading.cyclic_phases:
            lines.append(f"  cyclic schedule: {loading.cyclic_phases}")

    lines.append("")
    lines.append("Boundary Conditions")
    if cfg.boundary_conditions:
        for i, bc in enumerate(cfg.boundary_conditions, start=1):
            target = bc.nodes or bc.master or '<unset>'
            detail = f"{bc.type} on {target}"
            if bc.type in {'fix', 'prescribe', 'neumann', 'traction'}:
                detail += f" component={bc.component} value={bc.value}"
            if bc.type == 'symmetry':
                detail += f" axis={bc.axis}"
            if bc.type == 'rigid_connector':
                detail += f" master={bc.master} dofs={bc.dofs}"
            lines.append(f"  {i}. {detail}")
    else:
        warnings.append("No boundary_conditions entries are present.")
        lines.append("  none")

    if initial and _set_fields(initial):
        lines.append("")
        lines.append("Initial Conditions")
        if initial.preseed_notch_nodesets:
            lines.append("  preseed notch node sets: "
                         + ', '.join(initial.preseed_notch_nodesets))
        if initial.preseed_damage:
            lines.append(f"  preseed damage entries: {len(initial.preseed_damage)}")

    if cfg.acceptance:
        lines.append("")
        lines.append("Acceptance")
        for key, value in sorted(cfg.acceptance.items()):
            lines.append(f"  {key}: {_fmt_acceptance_value(value)}")

    lines.append("")
    lines.append("Outputs")
    enabled = []
    if output.h5:
        enabled.append(
            f"{output.trajectory_format} trajectory snapshots every "
            f"{output.h5_every} steps")
    if output.vtu:
        enabled.append(f"{output.viz_format} every {output.vtu_every}")
    if output.gif:
        enabled.append(f"gif ({output.gif_fields}, {output.gif_frames} frames)")
    if output.plots:
        enabled.append("plots")
    if output.profile:
        enabled.append("profile")
    lines.append(f"  output_dir: {output.output_dir or '<auto>'}")
    lines.append("  enabled: " + (', '.join(enabled) if enabled else 'console only'))
    if output.reaction_node_set:
        lines.append(f"  reaction logging: node_set={output.reaction_node_set}, "
                     f"component={output.reaction_component}")
        if (declared_node_sets
                and output.reaction_node_set not in declared_node_sets):
            sample = ', '.join(sorted(declared_node_sets)[:8])
            warnings.append(
                f"output.reaction_node_set={output.reaction_node_set!r} is "
                "not declared or referenced elsewhere in the YAML; check the "
                f"node-set name before running. Known YAML names: {sample}."
            )
    elif solver_type != 'explicit':
        warnings.append(
            "Implicit/quasi-static config has no output.reaction_node_set; "
            "load-displacement CSV will not be written."
        )

    lines.append("")
    lines.append("Device")
    lines.append(f"  device: {device.device or 'auto'}")
    lines.append(f"  torch.compile: {_fmt_bool(device.compile)}")

    lines.append("")
    lines.append("Warnings")
    if warnings:
        for warning in warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("  none")

    return '\n'.join(lines), 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phast explain-config",
        description="Explain a phast YAML config without running it."
    )
    parser.add_argument('config', help='Path to YAML config')
    args = parser.parse_args(argv)

    report, code = build_explanation(args.config)
    stream = sys.stderr if code else sys.stdout
    print(report, file=stream)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
