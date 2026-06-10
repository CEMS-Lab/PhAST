"""
Schema validation for phast YAML configs.

Walks the ``ProblemConfig`` dataclass tree, checks each field against the
declared type / enum / numeric range, and produces line-numbered error
messages instead of opaque KeyError / AttributeError / TypeError tracebacks
from deep inside the loader.

Public API:

* :class:`ConfigValidationError` — list-of-errors carrier; ``str(err)``
  formats the multi-line block shown to users.
* :func:`validate_config_file(path)` — parse YAML, validate, return
  ``(raw_dict, errors)``.
* :func:`format_errors(errors, source_path)` — pretty-print error list.

Issue #150, epic #136 phase 3.4.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, List, Optional, Tuple, Union, get_args, get_origin, get_type_hints

import meshio
import yaml

from . import config as _cfg
from ..utils.units import (
    MATERIAL_OVERRIDE_KINDS,
    LOADING_QUANTITY_KINDS,
    BOUNDARY_VALUE_QUANTITY_KINDS,
    BOUNDARY_TIME_QUANTITY_KINDS,
    parse_quantity,
)

# Inline material fields and loading fields that accept unit-suffixed
# string values (e.g. ``E: "32 GPa"``, ``t_total: "80 us"``). The
# loader normalises these via ``units.parse_quantity`` at resolve time.
_MATERIAL_UNIT_KEYS = set(MATERIAL_OVERRIDE_KINDS)
_LOADING_UNIT_KEYS = set(LOADING_QUANTITY_KINDS)
_BOUNDARY_TIME_UNIT_KEYS = set(BOUNDARY_TIME_QUANTITY_KINDS)


# ---------------------------------------------------------------------------
# Side specs: enums + numeric ranges that aren't expressible by Python types.
# Keys are dotted YAML paths (e.g. ``solver.dt_safety``).
# ---------------------------------------------------------------------------

ENUMS = {
    'geometry.type': sorted([
        'miehe_tension', 'miehe_shear', 'square_plate', 'three_point_bending',
        'l_shaped_panel', 'plate_with_holes', 'bazant_gap_test',
        'rectangular_sent', 'rectangular_sent_comsol_structured',
        'rectangular_sent_liu_structured',
        'kalthoff_winkler', 'crack_coalescence',
        'brazilian_disc', 'glass_impact_vnotch', 'perforated_sent',
    ]),
    'loading.protocol': ['simple', 'two_step_prestrain', 'cyclic'],
    'loading.ramp_type': [
        'constant', 'linear', 'smooth', 'smooth_step', 'velocity_impact',
    ],
    'solver.solver_type': [
        'explicit', 'quasi_static', 'quasi_static_legacy', 'static', 'lbfgs',
        'monolithic',
    ],
    'solver.time_integrator': [
        'central_difference', 'verlet', 'newmark',
        'generalized_alpha', 'gen_alpha',
    ],
    'solver.stagger_criterion': [
        'relative', 'absolute', 'linf', 'residual', 'am_energy',
    ],
    # Norm used inside the relative stagger convergence check. Default
    # ``l2`` matches the historical behaviour; ``linf`` mirrors the
    # max-norm criterion adopted by Bleyer & Roux-Langlois (2017) and
    # several anisotropic-PF papers since (issue #244).
    'solver.stagger_norm': ['l2', 'linf'],
    'solver.bounds_method': ['post_clamp', 'projected_cg'],
    # Issue #360 — H-update operator dispatcher. ``hard_max`` (default)
    # is byte-identical to ``torch.maximum``; the other methods are
    # opt-in differentiable alternatives.
    'solver.H_update_method': [
        'hard_max', 'softmax', 'smooth_max', 'log_smooth', 'custom_subgrad',
    ],
    'solver.preconditioner': [
        None, 'auto', 'amg', 'amgx', 'gmg', 'jacobi', 'none',
    ],
    'device.device': [None, 'cpu', 'cuda', 'mps'],
    'boundary_conditions[].type': [
        'fix', 'prescribe', 'neumann',
        # New BC vocabulary (PR #155, #154/#171/#182, #181):
        'traction', 'symmetry', 'rigid_connector',
        # Phase-field Dirichlet (issue #213) — locks scalar damage
        # ``phi = value`` on listed nodes for the whole simulation
        # (matches COMSOL pre-crack convention).
        'pf_dirichlet',
    ],
    'boundary_conditions[].component': [0, 1],
    # Per-BC traction ramp_type (independent of loading.ramp_type).
    'boundary_conditions[].ramp_type': [
        'constant', 'linear', 'smooth_step', 'cosine',
    ],
    # Symmetry axis: 'x' or 'y'.
    'boundary_conditions[].axis': ['x', 'y'],
}

# Enums that apply inside the free-form ``material.overrides`` dict
# (we still validate when the user supplies them).
OVERRIDE_ENUMS = {
    'energy_split': [
        'spectral', 'spectral_stress', 'amor', 'volumetric_deviatoric',
        'star_convex', 'isotropic',
    ],
    'pf_model': ['AT1', 'AT2', 'PFCZM', 'allencahn'],
    'driving_force': ['strain_energy', 'principal_stress'],
    'pfczm_softening': ['linear', 'exponential'],
}

# (lo, hi) inclusive ranges on dotted paths. None = unbounded.
RANGES = {
    'solver.dt_safety': (0.0, 1.0),
    'solver.damage_every': (1, None),
    'solver.max_stagger': (1, None),
    'solver.anderson_depth': (0, None),
    'solver.damage_max_iter': (1, None),
    'solver.static_max_iter': (1, None),
    'solver.damping_ratio_max': (0.0, None),
    'solver.eta_residual': (0.0, None),
    'solver.H_cap_factor': (0.0, None),
    'solver.stagger_tol': (0.0, None),
    'solver.damage_tol': (0.0, None),
    'solver.static_tol': (0.0, None),
    'loading.num_steps': (0, None),
    'loading.t_total': (0.0, None),
    'loading.dt': (0.0, None),
    'loading.t_ramp': (0.0, None),
    'output.print_every': (1, None),
    'output.h5_every': (1, None),
    'output.vtu_every': (1, None),
    'output.gif_frames': (1, None),
    'output.animation_raster_width': (32, None),
    'output.reaction_component': (0, 1),
}

# Override-dict numeric ranges (any negative value is wrong).
OVERRIDE_RANGES = {
    'l0': (0.0, None),
    'Gc': (0.0, None),
    'E': (0.0, None),
    'nu': (-1.0, 0.5),
    'rho': (0.0, None),
    'eta_residual': (0.0, None),
    'sigma_ts': (0.0, None),
    'pfczm_p': (2, None),
}

# Mutually-exclusive groups — exactly one or zero of these may be set.
# Keys: identifier; value: (parent_path, [field_names], "explanation").
MUTEX = [
    (
        'geometry',
        ['type', 'mesh_path'],
        "geometry: set either 'type' (use built-in generator) or "
        "'mesh_path' (load external mesh), not both.",
    ),
]


# ---------------------------------------------------------------------------
# Line-number-aware YAML loader
# ---------------------------------------------------------------------------

class _LineLoader(yaml.SafeLoader):
    """SafeLoader that records the source line on each mapping/sequence node."""


def _compose_node(self, parent, index):
    line = self.line
    node = yaml.SafeLoader.compose_node(self, parent, index)
    node.__line__ = line + 1
    return node


_LineLoader.compose_node = _compose_node


def _build_line_map(node, prefix: str = '', out: dict = None) -> dict:
    """Walk a yaml node tree and return ``{dotted_path: line_no}``."""
    if out is None:
        out = {}
    if isinstance(node, yaml.MappingNode):
        for k_node, v_node in node.value:
            key = k_node.value
            path = f"{prefix}.{key}" if prefix else key
            out[path] = getattr(k_node, '__line__', getattr(v_node, '__line__', 0))
            _build_line_map(v_node, path, out)
    elif isinstance(node, yaml.SequenceNode):
        for i, item in enumerate(node.value):
            path = f"{prefix}[{i}]"
            out[path] = getattr(item, '__line__', 0)
            _build_line_map(item, path, out)
    return out


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    path: str
    message: str
    line_no: int = 0
    allowed_values: Optional[list] = None
    suggestion: Optional[str] = None

    def format(self) -> str:
        head = f"  {self.path}: {self.message}"
        lines = [head]
        if self.allowed_values is not None:
            allowed = ', '.join(repr(v) if not isinstance(v, str) else v
                                for v in self.allowed_values)
            lines.append(f"    Allowed values: {allowed}")
        if self.suggestion:
            lines.append(f"    {self.suggestion}")
        return '\n'.join(lines)


@dataclass
class ValidationWarning:
    path: str
    message: str
    line_no: int = 0
    suggestion: Optional[str] = None

    def format(self) -> str:
        head = f"  {self.path}: {self.message}"
        lines = [head]
        if self.suggestion:
            lines.append(f"    {self.suggestion}")
        return '\n'.join(lines)


class ConfigValidationError(Exception):
    """Raised when a config fails validation. Aggregates all errors."""

    def __init__(self, errors: List[ValidationError], source_path: str = ''):
        self.errors = errors
        self.source_path = source_path
        super().__init__(format_errors(errors, source_path))


def format_errors(errors: List[ValidationError], source_path: str = '') -> str:
    if not errors:
        return ''
    src = source_path or '<config>'
    # Group adjacent errors by line for a tidier block.
    lines = []
    seen_lines = set()
    for err in errors:
        if err.line_no and err.line_no not in seen_lines:
            lines.append(f"Error in {src} line {err.line_no}:")
            seen_lines.add(err.line_no)
        elif not err.line_no and ('top' not in seen_lines):
            lines.append(f"Error in {src}:")
            seen_lines.add('top')
        lines.append(err.format())
    lines.append(f"  See: configs/REFERENCE.yaml for the full schema.")
    return '\n'.join(lines)


def format_warnings(warnings: List[ValidationWarning],
                    source_path: str = '') -> str:
    if not warnings:
        return ''
    src = source_path or '<config>'
    lines = []
    seen_lines = set()
    for warning in warnings:
        if warning.line_no and warning.line_no not in seen_lines:
            lines.append(f"Warning in {src} line {warning.line_no}:")
            seen_lines.add(warning.line_no)
        elif not warning.line_no and ('top' not in seen_lines):
            lines.append(f"Warning in {src}:")
            seen_lines.add('top')
        lines.append(warning.format())
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------

# Per-section keys that are accepted but ignored (legacy / typo-tolerated).
# These appear in shipped configs that we don't want to break; add new ones
# sparingly. New code should use the canonical location instead.
_DEPRECATED_KEYS = {
    'solver': {'device'},  # typo for top-level device.device, several configs
}


_DATACLASS_BY_SECTION = {
    'geometry': _cfg.GeometryConfig,
    'material': _cfg.MaterialConfig,
    'loading': _cfg.LoadingConfig,
    'solver': _cfg.SolverSettings,
    'output': _cfg.OutputConfig,
    'device': _cfg.DeviceConfig,
    'initial_conditions': _cfg.InitialConditionsConfig,
}


def _strip_optional(tp):
    """Return inner type if ``tp`` is ``Optional[X]``."""
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _is_instance(value, tp) -> bool:
    """Forgiving isinstance respecting common YAML coercions (int <-> float)."""
    tp = _strip_optional(tp)
    origin = get_origin(tp)
    if origin is list or tp is list:
        return isinstance(value, list)
    if origin is dict or tp is dict:
        return isinstance(value, dict)
    if tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        # PyYAML safe_load returns '1e-6' as str (YAML 1.1); the loader
        # coerces these in _dict_to_dataclass — accept here.
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False
        return False
    if tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return True
        if isinstance(value, str):
            try:
                int(value)
                return True
            except ValueError:
                return False
        return False
    if tp is bool:
        return isinstance(value, bool)
    if tp is str:
        return isinstance(value, str)
    if tp is Any:
        return True
    try:
        return isinstance(value, tp)
    except TypeError:
        return True


def _type_name(tp) -> str:
    tp = _strip_optional(tp)
    return getattr(tp, '__name__', str(tp))


def _as_number(value):
    """Return ``value`` as float if numeric / numeric-string, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_length_mm(value):
    """Return ``value`` as millimetres if numeric / length-string, else None."""
    if value is None:
        return None
    try:
        return parse_quantity(value, 'length')
    except (TypeError, ValueError):
        return _as_number(value)


def _did_you_mean(key: str, options) -> Optional[str]:
    matches = difflib.get_close_matches(key, list(options), n=1, cutoff=0.6)
    if matches:
        return f"Did you mean '{matches[0]}'?"
    return None


# ---------------------------------------------------------------------------
# Section validators
# ---------------------------------------------------------------------------

def _validate_section(
    section_name: str,
    section_dict: Any,
    cls,
    line_map: dict,
    errors: List[ValidationError],
):
    """Validate one nested section against a dataclass schema."""
    if section_dict is None:
        return
    if not isinstance(section_dict, dict):
        errors.append(ValidationError(
            path=section_name,
            message=f"expected mapping, got {type(section_dict).__name__}",
            line_no=line_map.get(section_name, 0),
        ))
        return

    hints = get_type_hints(cls)
    known = set(hints.keys())

    for key, value in section_dict.items():
        path = f"{section_name}.{key}"
        line = line_map.get(path, line_map.get(section_name, 0))

        if key not in known:
            if key in _DEPRECATED_KEYS.get(section_name, set()):
                continue  # silently accepted, legacy field
            errors.append(ValidationError(
                path=path,
                message=f"unknown field",
                line_no=line,
                suggestion=_did_you_mean(key, known),
            ))
            continue

        # Type check.
        #
        # Unit-suffix strings (e.g. ``E: "32 GPa"``) are accepted on
        # inline-material fields and unit-bearing loading fields; the
        # loader normalises them to floats via ``parse_quantity``. We
        # therefore accept ``str`` for these fields at validation time
        # (the actual unit suffix is rechecked when ``parse_quantity``
        # runs at config resolution).
        _unit_string_ok = (
            (section_name == 'material' and key in _MATERIAL_UNIT_KEYS)
            or (section_name == 'loading' and key in _LOADING_UNIT_KEYS)
        )
        if (value is not None
                and not _is_instance(value, hints[key])
                and not (_unit_string_ok and isinstance(value, str))):
            errors.append(ValidationError(
                path=path,
                message=(f"expected {_type_name(hints[key])}, got "
                         f"{type(value).__name__}"),
                line_no=line,
            ))
            continue

        # Enum check
        if path in ENUMS and value not in ENUMS[path]:
            suggestion = None
            if isinstance(value, str):
                str_options = [v for v in ENUMS[path] if isinstance(v, str)]
                suggestion = _did_you_mean(value, str_options)
            errors.append(ValidationError(
                path=path,
                message=f"invalid value {value!r}",
                line_no=line,
                allowed_values=ENUMS[path],
                suggestion=suggestion,
            ))

        # Range check
        if path in RANGES:
            num = _as_number(value)
            if num is not None:
                lo, hi = RANGES[path]
                if (lo is not None and num < lo) or (hi is not None and num > hi):
                    bound = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
                    errors.append(ValidationError(
                        path=path,
                        message=f"value {value} out of range; expected {bound}",
                        line_no=line,
                    ))


def _validate_overrides(overrides: dict, line_map: dict,
                        errors: List[ValidationError]):
    """Validate ``material.overrides`` (free-form dict, side specs)."""
    if not isinstance(overrides, dict):
        return
    for key, value in overrides.items():
        path = f"material.overrides.{key}"
        line = line_map.get(path, line_map.get('material.overrides', 0))
        if key in OVERRIDE_ENUMS and value not in OVERRIDE_ENUMS[key]:
            errors.append(ValidationError(
                path=path,
                message=f"invalid value {value!r}",
                line_no=line,
                allowed_values=OVERRIDE_ENUMS[key],
            ))
        if key in OVERRIDE_RANGES:
            num = _as_number(value)
            if num is not None:
                lo, hi = OVERRIDE_RANGES[key]
                if (lo is not None and num < lo) or (hi is not None and num > hi):
                    bound = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
                    errors.append(ValidationError(
                        path=path,
                        message=f"value {value} out of range; expected {bound}",
                        line_no=line,
                    ))


def _validate_boundary_conditions(bc_list, line_map: dict,
                                  errors: List[ValidationError]):
    if bc_list is None:
        return
    if not isinstance(bc_list, list):
        errors.append(ValidationError(
            path='boundary_conditions',
            message=f"expected list, got {type(bc_list).__name__}",
            line_no=line_map.get('boundary_conditions', 0),
        ))
        return

    hints = get_type_hints(_cfg.BoundaryConditionEntry)
    known = set(hints.keys())

    for i, entry in enumerate(bc_list):
        if not isinstance(entry, dict):
            errors.append(ValidationError(
                path=f'boundary_conditions[{i}]',
                message=f"expected mapping, got {type(entry).__name__}",
                line_no=line_map.get(f'boundary_conditions[{i}]', 0),
            ))
            continue
        for key, value in entry.items():
            path = f"boundary_conditions[{i}].{key}"
            line = line_map.get(path, line_map.get(f'boundary_conditions[{i}]', 0))
            if key not in known:
                errors.append(ValidationError(
                    path=path,
                    message="unknown field",
                    line_no=line,
                    suggestion=_did_you_mean(key, known),
                ))
                continue
            bc_type = entry.get('type', 'fix')
            _bc_unit_string_ok = (
                key in _BOUNDARY_TIME_UNIT_KEYS
                or (
                    key == 'value'
                    and bc_type in BOUNDARY_VALUE_QUANTITY_KINDS
                )
            )
            if value is not None and not _is_instance(value, hints[key]):
                if _bc_unit_string_ok and isinstance(value, str):
                    continue
                errors.append(ValidationError(
                    path=path,
                    message=(f"expected {_type_name(hints[key])}, got "
                             f"{type(value).__name__}"),
                    line_no=line,
                ))
                continue
            enum_key = f"boundary_conditions[].{key}"
            if enum_key in ENUMS and value not in ENUMS[enum_key]:
                suggestion = None
                if isinstance(value, str):
                    str_options = [v for v in ENUMS[enum_key]
                                   if isinstance(v, str)]
                    suggestion = _did_you_mean(value, str_options)
                errors.append(ValidationError(
                    path=path,
                    message=f"invalid value {value!r}",
                    line_no=line,
                    allowed_values=ENUMS[enum_key],
                    suggestion=suggestion,
                ))


def _validate_mutex(raw: dict, line_map: dict,
                    errors: List[ValidationError]):
    for parent, names, msg in MUTEX:
        section = raw.get(parent)
        if not isinstance(section, dict):
            continue
        present = [n for n in names if section.get(n) not in (None, '', {})]
        if len(present) > 1:
            errors.append(ValidationError(
                path=f"{parent}.{'+'.join(present)}",
                message=f"mutually exclusive fields both set: {present}",
                line_no=line_map.get(f"{parent}.{present[1]}",
                                     line_map.get(parent, 0)),
                suggestion=msg,
            ))


def _mesh_path_candidates(config_path: str, mesh_path: str) -> List[str]:
    expanded = os.path.expanduser(mesh_path)
    if os.path.isabs(expanded):
        return [expanded]
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return [
        os.path.abspath(os.path.join(config_dir, expanded)),
        os.path.abspath(expanded),
    ]


def _existing_mesh_path(config_path: str, mesh_path: str) -> Optional[str]:
    for candidate in _mesh_path_candidates(config_path, mesh_path):
        if os.path.exists(candidate):
            return candidate
    return None


def _node_sets_from_mesh_file(mesh_path: str) -> set:
    """Read an external mesh and return named point/line/vertex sets."""
    if not os.path.exists(mesh_path):
        raise ValueError(f"Mesh file not found at: {mesh_path}")

    raw = meshio.read(mesh_path)
    names = set()

    if getattr(raw, 'point_sets', None):
        names.update(str(name) for name in raw.point_sets)

    if getattr(raw, 'cell_sets', None):
        for name, blocks in raw.cell_sets.items():
            if str(name).startswith('gmsh:'):
                continue
            for i, cb in enumerate(raw.cells):
                if cb.type not in ('line', 'vertex') or i >= len(blocks):
                    continue
                block = blocks[i]
                try:
                    if len(block) > 0:
                        names.add(str(name))
                except TypeError:
                    continue

    if getattr(raw, 'field_data', None):
        phys_data = raw.cell_data.get('gmsh:physical', [])
        if phys_data:
            line_or_vertex_ids = set()
            for i, cb in enumerate(raw.cells):
                if cb.type in ('line', 'vertex') and i < len(phys_data):
                    line_or_vertex_ids.update(int(v) for v in phys_data[i])
            for name, info in raw.field_data.items():
                phys_id, dim = int(info[0]), int(info[1])
                if dim <= 1 and phys_id in line_or_vertex_ids:
                    names.add(str(name))

    return names


def _referenced_node_sets(raw: dict) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []

    for i, entry in enumerate(raw.get('boundary_conditions') or []):
        if not isinstance(entry, dict):
            continue
        nodes = entry.get('nodes')
        if nodes:
            refs.append((f'boundary_conditions[{i}].nodes', str(nodes)))
        master = entry.get('master')
        if master:
            refs.append((f'boundary_conditions[{i}].master', str(master)))

    initial = raw.get('initial_conditions') or {}
    if isinstance(initial, dict):
        for i, name in enumerate(initial.get('preseed_notch_nodesets') or []):
            refs.append((f'initial_conditions.preseed_notch_nodesets[{i}]',
                         str(name)))
        for i, spec in enumerate(initial.get('preseed_damage') or []):
            if not isinstance(spec, dict):
                continue
            name = spec.get('nodes')
            if name:
                refs.append((f'initial_conditions.preseed_damage[{i}].nodes',
                             str(name)))

    output = raw.get('output') or {}
    if isinstance(output, dict) and output.get('reaction_node_set'):
        refs.append(('output.reaction_node_set', str(output['reaction_node_set'])))

    return refs


def _validate_external_mesh_node_sets(raw: dict, line_map: dict,
                                      mesh_path: str,
                                      errors: List[ValidationError]) -> None:
    try:
        available = _node_sets_from_mesh_file(mesh_path)
    except SystemExit as exc:
        errors.append(ValidationError(
            path='geometry.mesh_path',
            message=(
                "could not read mesh file for validation: mesh reader exited "
                f"with status {exc.code}"
            ),
            line_no=line_map.get('geometry.mesh_path',
                                 line_map.get('geometry', 0)),
            suggestion=(
                "Check the mesh format with meshio or regenerate the mesh "
                "before submitting the job."
            ),
        ))
        return
    except Exception as exc:
        errors.append(ValidationError(
            path='geometry.mesh_path',
            message=f"could not read mesh file for validation: {exc}",
            line_no=line_map.get('geometry.mesh_path',
                                 line_map.get('geometry', 0)),
            suggestion=(
                "Check the mesh format with meshio or regenerate the mesh "
                "before submitting the job."
            ),
        ))
        return

    if not available:
        # FEMMesh can auto-detect simple domain boundaries at runtime when no
        # named sets exist. Without loading/precomputing the full FEM mesh here
        # we cannot prove those names, so leave this as a runtime concern.
        return

    for path, name in _referenced_node_sets(raw):
        if name in available:
            continue
        sample = ', '.join(sorted(available)[:12])
        errors.append(ValidationError(
            path=path,
            message=f"node set {name!r} is not present in external mesh",
            line_no=line_map.get(path, 0),
            suggestion=_did_you_mean(name, available)
                       or f"Available node sets: {sample}",
        ))


def _validate_file_context(raw: dict, line_map: dict, yaml_path: str,
                           errors: List[ValidationError]) -> None:
    """Validate fields that need the config file's directory for context."""
    geom = raw.get('geometry')
    if not isinstance(geom, dict):
        return

    mesh_path = geom.get('mesh_path')
    if not mesh_path:
        return
    if not isinstance(mesh_path, str):
        return

    existing = _existing_mesh_path(yaml_path, mesh_path)
    if existing is None:
        candidates = _mesh_path_candidates(yaml_path, mesh_path)
        checked = ', '.join(candidates)
        errors.append(ValidationError(
            path='geometry.mesh_path',
            message=f"mesh file does not exist; checked {checked}",
            line_no=line_map.get('geometry.mesh_path',
                                 line_map.get('geometry', 0)),
            suggestion=(
                "Use a path relative to the YAML file, an absolute path, "
                "or run the mesh generator before submitting the job."
            ),
        ))
        return

    _validate_external_mesh_node_sets(raw, line_map, existing, errors)


def _validate_cross_field_compatibility(raw: dict, line_map: dict,
                                        errors: List[ValidationError]) -> None:
    """Reject known unsupported combinations across YAML sections."""
    solver = raw.get('solver') or {}
    if not isinstance(solver, dict):
        return

    solver_type = solver.get('solver_type', 'explicit')
    time_integrator = solver.get('time_integrator')
    if solver_type != 'explicit' and time_integrator is not None:
        errors.append(ValidationError(
            path='solver.time_integrator',
            message=(
                "time_integrator is only used by solver_type='explicit'; "
                f"got solver_type={solver_type!r}"
            ),
            line_no=line_map.get('solver.time_integrator',
                                 line_map.get('solver', 0)),
            suggestion=(
                "Remove solver.time_integrator for quasi-static/static runs, "
                "or set solver.solver_type: explicit for dynamic integration."
            ),
        ))

    has_rigid_connector = False
    first_rigid_path = None
    for i, entry in enumerate(raw.get('boundary_conditions') or []):
        if isinstance(entry, dict) and entry.get('type') == 'rigid_connector':
            has_rigid_connector = True
            first_rigid_path = f'boundary_conditions[{i}].type'
            break

    if not has_rigid_connector:
        return

    if solver_type == 'explicit' and solver.get('fresh_d_in_corrector') is True:
        errors.append(ValidationError(
            path='solver.fresh_d_in_corrector',
            message=(
                "fresh_d_in_corrector is not supported with rigid_connector "
                "MPCs in the explicit dynamics path"
            ),
            line_no=line_map.get('solver.fresh_d_in_corrector',
                                 line_map.get('solver', 0)),
            suggestion=(
                "Set fresh_d_in_corrector: false or remove rigid_connector "
                f"MPCs (first rigid_connector at {first_rigid_path})."
            ),
        ))

    if (solver_type == 'explicit'
            and time_integrator in {'generalized_alpha', 'gen_alpha'}):
        errors.append(ValidationError(
            path='solver.time_integrator',
            message=(
                "generalized_alpha does not yet support rigid_connector MPCs"
            ),
            line_no=line_map.get('solver.time_integrator',
                                 line_map.get('solver', 0)),
            suggestion=(
                "Use time_integrator: central_difference for rigid_connector "
                "runs, or remove rigid_connector MPCs."
            ),
        ))


def _raw_material_l0_mm(raw: dict) -> Optional[float]:
    mat = raw.get('material') or {}
    if not isinstance(mat, dict):
        return None
    if 'l0' in mat:
        return _as_length_mm(mat.get('l0'))
    overrides = mat.get('overrides')
    if isinstance(overrides, dict) and 'l0' in overrides:
        return _as_length_mm(overrides.get('l0'))
    return None


def _declared_mesh_sizes_mm(raw: dict) -> List[float]:
    geom = raw.get('geometry') or {}
    if not isinstance(geom, dict):
        return []

    sizes: List[float] = []

    params = geom.get('parameters')
    if isinstance(params, dict):
        for key in ('h_crack', 'h_min', 'h', 'element_size'):
            value = _as_length_mm(params.get(key))
            if value is not None and value > 0:
                sizes.append(value)

    mesh = geom.get('mesh')
    if isinstance(mesh, dict):
        element_size = mesh.get('element_size')
        if isinstance(element_size, dict):
            default = _as_length_mm(element_size.get('default'))
            if default is not None and default > 0:
                sizes.append(default)
            refined = element_size.get('refined') or []
            if isinstance(refined, list):
                for entry in refined:
                    if isinstance(entry, dict):
                        size = _as_length_mm(entry.get('size'))
                        if size is not None and size > 0:
                            sizes.append(size)
        else:
            size = _as_length_mm(element_size)
            if size is not None and size > 0:
                sizes.append(size)

    return sizes


def validate_config_warnings(raw: dict, line_map: Optional[dict] = None
                             ) -> List[ValidationWarning]:
    """Return non-fatal advisories for physically risky but legal configs."""
    warnings: List[ValidationWarning] = []
    line_map = line_map or {}

    if raw is None or not isinstance(raw, dict):
        return warnings

    if 'schema_version' not in raw:
        warnings.append(ValidationWarning(
            path='schema_version',
            message='missing schema_version; config will load as schema v1',
            line_no=line_map.get('<root>', 0),
            suggestion='Add top-level schema_version: 1 for reproducibility.',
        ))

    l0_mm = _raw_material_l0_mm(raw)
    mesh_sizes = _declared_mesh_sizes_mm(raw)
    if l0_mm is not None and l0_mm > 0 and mesh_sizes:
        h_min = min(mesh_sizes)
        ratio = h_min / l0_mm
        if ratio > 0.5:
            warnings.append(ValidationWarning(
                path='geometry',
                message=(
                    f"finest declared mesh size h={h_min:g} mm gives "
                    f"h/l0={ratio:.3g}"
                ),
                line_no=line_map.get('geometry.mesh',
                                     line_map.get('geometry.parameters',
                                                  line_map.get('geometry', 0))),
                suggestion=(
                    "Phase-field fracture validation usually needs h <= l0/2 "
                    "near the crack path; h <= l0/4 is preferred for "
                    "quantitative convergence."
                ),
            ))

    return warnings


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

# Allowed top-level keys (matches loader behaviour incl. flat scalars).
_TOP_KEYS = set(_DATACLASS_BY_SECTION) | {
    'schema_version', 'name', 'reference', 'problem', 'boundary_conditions',
    'example', 'inversion', 'acceptance',
}


def validate_config(raw: dict, line_map: Optional[dict] = None
                    ) -> List[ValidationError]:
    """Validate a parsed YAML dict against the ProblemConfig schema.

    Parameters
    ----------
    raw : dict
        Parsed YAML document.
    line_map : dict, optional
        ``{dotted_path: line_no}`` from :func:`_build_line_map`.

    Returns
    -------
    list[ValidationError]
    """
    errors: List[ValidationError] = []
    line_map = line_map or {}

    if raw is None:
        return errors
    if not isinstance(raw, dict):
        errors.append(ValidationError(
            path='<root>',
            message=f"top-level YAML must be a mapping, got "
                    f"{type(raw).__name__}",
        ))
        return errors

    manifest_type = raw.get('manifest_type')
    if manifest_type:
        errors.append(ValidationError(
            path='manifest_type',
            message=(
                f"{manifest_type!r} is an orchestration manifest, not a "
                "single PhAST problem config"
            ),
            line_no=line_map.get('manifest_type', 0),
            suggestion=(
                "Run the commands listed under cases/command/module, or use a "
                "canonical problem config under configs/benchmarks/dynamic/ "
                "or configs/benchmarks/quasistatic/ with python -m phast run."
            ),
        ))
        return errors

    # Unknown top-level keys
    for key in raw:
        if key not in _TOP_KEYS:
            errors.append(ValidationError(
                path=key,
                message="unknown top-level key",
                line_no=line_map.get(key, 0),
                suggestion=_did_you_mean(key, _TOP_KEYS),
            ))

    if 'schema_version' in raw:
        value = raw['schema_version']
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(ValidationError(
                path='schema_version',
                message=f"expected int, got {type(value).__name__}",
                line_no=line_map.get('schema_version', 0),
            ))
        elif value < 1:
            errors.append(ValidationError(
                path='schema_version',
                message=f"value {value} out of range; expected >= 1",
                line_no=line_map.get('schema_version', 0),
            ))

    # Each known section
    for section, cls in _DATACLASS_BY_SECTION.items():
        if section in raw:
            _validate_section(section, raw[section], cls, line_map, errors)

    # material.overrides (separate side spec)
    mat = raw.get('material') or {}
    if isinstance(mat, dict) and 'overrides' in mat:
        _validate_overrides(mat['overrides'], line_map, errors)

    # boundary_conditions
    if 'boundary_conditions' in raw:
        _validate_boundary_conditions(raw['boundary_conditions'], line_map,
                                      errors)

    # mutually exclusive fields
    _validate_mutex(raw, line_map, errors)
    _validate_cross_field_compatibility(raw, line_map, errors)

    return errors


def validate_config_file(yaml_path: str
                         ) -> Tuple[dict, List[ValidationError]]:
    """Load, parse with line tracking, validate. Returns (raw, errors)."""
    with open(yaml_path, 'r') as f:
        text = f.read()
    raw = yaml.safe_load(text) or {}
    # Compose with line tracking
    loader = _LineLoader(text)
    try:
        node = loader.get_single_node()
    finally:
        loader.dispose()
    line_map = _build_line_map(node) if node is not None else {}
    errors = validate_config(raw, line_map)
    _validate_file_context(raw, line_map, yaml_path, errors)
    return raw, errors


def validate_config_file_with_warnings(
        yaml_path: str
        ) -> Tuple[dict, List[ValidationError], List[ValidationWarning]]:
    """Load, validate, and return non-fatal setup advisories."""
    with open(yaml_path, 'r') as f:
        text = f.read()
    raw = yaml.safe_load(text) or {}
    loader = _LineLoader(text)
    try:
        node = loader.get_single_node()
    finally:
        loader.dispose()
    line_map = _build_line_map(node) if node is not None else {}
    errors = validate_config(raw, line_map)
    _validate_file_context(raw, line_map, yaml_path, errors)
    warnings = validate_config_warnings(raw, line_map)
    return raw, errors, warnings


def assert_valid(yaml_path: str) -> None:
    """Validate a YAML config file; raise ConfigValidationError on failure."""
    _, errs = validate_config_file(yaml_path)
    if errs:
        raise ConfigValidationError(errs, source_path=yaml_path)
