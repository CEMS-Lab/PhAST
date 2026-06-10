"""Auto-generate ``configs/REFERENCE.yaml`` from the dataclass schema.

Single-source-of-truth machinery for the user-facing reference config.
Walks the ``ProblemConfig`` dataclass tree (``phast/config.py``)
plus the ``ENUMS`` / ``RANGES`` / ``OVERRIDE_ENUMS`` / ``OVERRIDE_RANGES``
/ ``MUTEX`` side dicts (``phast/config_validation.py``) and
fills in slot markers (``# {{schema:<section>}}``) inside
``configs/REFERENCE.template.yaml``.

Run modes
---------

* dry-run (default): print the generated file to stdout and a unified
  diff against the checked-in ``configs/REFERENCE.yaml``.
* ``--write``: overwrite ``configs/REFERENCE.yaml`` in place.
* ``--check``: exit non-zero if the generated content differs from the
  checked-in file (used by CI via
  ``tests/test_reference_yaml_consistency.py``).

Issue #148 (epic #136 phase 3.2).
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import os
import sys
import typing
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Optional, get_args, get_origin

import phast.config as _cfg
import phast.config_validation as _val


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(_REPO_ROOT, 'configs', 'REFERENCE.template.yaml')
OUTPUT_PATH = os.path.join(_REPO_ROOT, 'configs', 'REFERENCE.yaml')


# Sections for which the generator emits a full schema-derived listing
# of every field on the corresponding dataclass.  Order is the YAML
# section name -> (dataclass, base indent for the emitted block).
_SCHEMA_SECTIONS = {
    'loading': (_cfg.LoadingConfig, '  '),
    'solver': (_cfg.SolverSettings, '  '),
    'output': (_cfg.OutputConfig, '  '),
    'device': (_cfg.DeviceConfig, '  '),
    'initial_conditions': (_cfg.InitialConditionsConfig, '  '),
}


# ---------------------------------------------------------------------------
# Type / value formatting helpers
# ---------------------------------------------------------------------------

def _strip_optional(tp):
    """Return inner type if ``tp`` is ``Optional[X]``, else ``tp``."""
    if get_origin(tp) is typing.Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _type_label(tp) -> str:
    """Human-readable type label for a field annotation."""
    inner = _strip_optional(tp)
    origin = get_origin(inner)
    if origin in (list, typing.List):
        args = get_args(inner)
        if args:
            return f"list[{_type_label(args[0])}]"
        return "list"
    if origin in (dict, typing.Dict):
        return "dict"
    if inner is type(None):
        return "null"
    name = getattr(inner, '__name__', None) or str(inner)
    optional = (get_origin(tp) is typing.Union
                and type(None) in get_args(tp))
    return f"{name} or null" if optional else name


def _format_default(value: Any) -> str:
    """YAML-friendly literal for a default value."""
    if value is MISSING:
        return '<required>'
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        if value == '':
            return '""'
        # Quote if it could be confused with another YAML scalar.
        if any(c in value for c in ' :,#') or value.lower() in (
                'true', 'false', 'null', 'yes', 'no'):
            return f'"{value}"'
        return value
    if isinstance(value, float):
        # Compact scientific for very small / very large magnitudes.
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            s = f"{value:.6g}"
            # Normalise to lowercase 'e' and drop '+0' padding.
            s = s.replace('E', 'e')
            return s
        # Integer-valued floats keep a trailing ``.0`` so YAML keeps the
        # type on round-trip.
        if value == int(value) and abs(value) < 1e16:
            return f"{value:.1f}"
        return repr(value)
    if isinstance(value, list):
        if not value:
            return '[]'
        return '[' + ', '.join(_format_default(v) for v in value) + ']'
    if isinstance(value, dict):
        if not value:
            return '{}'
        return repr(value)
    return repr(value)


def _field_default(fld: dataclasses.Field) -> Any:
    if fld.default is not MISSING:
        return fld.default
    if fld.default_factory is not MISSING:  # type: ignore[misc]
        try:
            return fld.default_factory()  # type: ignore[misc]
        except Exception:  # pragma: no cover — defensive
            return MISSING
    return MISSING


def _enum_for(path: str) -> Optional[list]:
    return _val.ENUMS.get(path)


def _range_for(path: str) -> Optional[tuple]:
    return _val.RANGES.get(path)


def _mutex_for(section: str, field_name: str) -> Optional[list]:
    """Return the *other* fields in any MUTEX group containing this one."""
    out = []
    for parent, names, _msg in _val.MUTEX:
        if parent == section and field_name in names:
            others = [n for n in names if n != field_name]
            out.extend(others)
    return out or None


# ---------------------------------------------------------------------------
# Schema-block emitters
# ---------------------------------------------------------------------------

def _format_meta(field_name: str, fld: dataclasses.Field, *,
                 section: str) -> str:
    """Build the ``# type[, enum: ...][, range: ...][, mutex with: X]``
    trailing comment for a single field."""
    bits = [_type_label(fld.type)]
    enum = _enum_for(f"{section}.{field_name}")
    if enum is not None:
        rendered = '|'.join('null' if v is None else str(v) for v in enum)
        bits.append(f"enum: {rendered}")
    rng = _range_for(f"{section}.{field_name}")
    if rng is not None:
        lo, hi = rng
        lo_s = '-inf' if lo is None else str(lo)
        hi_s = '+inf' if hi is None else str(hi)
        bits.append(f"range: [{lo_s}, {hi_s}]")
    mutex = _mutex_for(section, field_name)
    if mutex:
        bits.append(f"mutually exclusive with: {', '.join(mutex)}")
    return ', '.join(bits)


def _emit_dataclass_block(
    dc, *, section: str, indent: str = '  ', skip_fields=()
) -> list[str]:
    """Emit one commented-out YAML line per dataclass field."""
    lines: list[str] = []
    for fld in fields(dc):
        if fld.name in skip_fields:
            continue
        default = _field_default(fld)
        rendered_default = _format_default(default)
        meta = _format_meta(fld.name, fld, section=section)
        # Pad the key:value portion to a fixed column for readability.
        key_value = f"{fld.name}: {rendered_default}"
        # All fields are emitted commented-out; users uncomment what they
        # need.  Active example values live in the template, never here.
        lines.append(f"{indent}# {key_value}  # {meta}")
    return lines


def _emit_overrides_block(indent: str = '#   ') -> list[str]:
    """Emit allowed values for ``material.overrides`` keys (string enums
    + numeric ranges)."""
    lines: list[str] = []
    for key, allowed in _val.OVERRIDE_ENUMS.items():
        rendered = '|'.join(str(v) for v in allowed)
        lines.append(f"{indent}{key}  # enum: {rendered}")
    for key, rng in _val.OVERRIDE_RANGES.items():
        lo, hi = rng
        lo_s = '-inf' if lo is None else str(lo)
        hi_s = '+inf' if hi is None else str(hi)
        lines.append(f"{indent}{key}  # float, range: [{lo_s}, {hi_s}]")
    return lines


def _emit_geometry_type_table(indent: str = '#   ') -> list[str]:
    """Emit the list of allowed ``geometry.type`` strings (registry-driven)."""
    allowed = _enum_for('geometry.type') or []
    return [f"{indent}{name}" for name in allowed]


def _emit_material_inline_block(indent: str = '#   ') -> list[str]:
    """Inline material fields from MaterialConfig (skip preset / overrides)."""
    return _emit_dataclass_block(
        _cfg.MaterialConfig,
        section='material',
        indent=indent.rstrip('#').rstrip() + ' ' * (len(indent) -
                                                    len(indent.lstrip('#'))),
        skip_fields=('preset', 'overrides'),
    )


def _emit_bc_block(indent: str = '#   ') -> list[str]:
    """Per-entry schema for ``boundary_conditions``."""
    lines: list[str] = []
    for fld in fields(_cfg.BoundaryConditionEntry):
        default = _field_default(fld)
        rendered = _format_default(default)
        bits = [_type_label(fld.type)]
        enum_key = f"boundary_conditions[].{fld.name}"
        if enum_key in _val.ENUMS:
            allowed = _val.ENUMS[enum_key]
            bits.append('enum: ' + '|'.join(str(v) for v in allowed))
        lines.append(f"{indent}{fld.name}: {rendered}  # {', '.join(bits)}")
    return lines


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _slot_block(slot: str) -> list[str]:
    """Resolve one ``{{schema:<slot>}}`` marker to its rendered lines."""
    if slot in _SCHEMA_SECTIONS:
        dc, indent = _SCHEMA_SECTIONS[slot]
        return _emit_dataclass_block(dc, section=slot, indent=indent)
    if slot == 'material':
        # Inline material fields, presented as a plain list (the template
        # places this inside a `# ` prose block, so we prefix `# ` to make
        # each line a comment line rather than YAML).
        out = []
        for fld in fields(_cfg.MaterialConfig):
            if fld.name in ('preset', 'overrides'):
                continue
            default = _field_default(fld)
            rendered = _format_default(default)
            meta = _format_meta(fld.name, fld, section='material')
            out.append(f"#   {fld.name}: {rendered}  # {meta}")
        return out
    if slot == 'material.overrides':
        return _emit_overrides_block(indent='#   ')
    if slot == 'geometry.type_table':
        return _emit_geometry_type_table(indent='#   ')
    if slot == 'boundary_conditions':
        return _emit_bc_block(indent='#   ')
    raise KeyError(f"unknown schema slot: {slot!r}")


def render(template_text: str) -> str:
    """Return ``template_text`` with all ``{{schema:...}}`` slots resolved."""
    out_lines: list[str] = []
    for raw_line in template_text.splitlines():
        stripped = raw_line.strip()
        # Slot markers must occupy the whole (possibly commented) line.
        # Recognise either ``# {{schema:foo}}`` (template prose region)
        # or ``  # {{schema:foo}}`` (in-section schema block).
        if '{{schema:' in stripped and stripped.endswith('}}'):
            slot = stripped.split('{{schema:', 1)[1].rstrip('}').strip()
            out_lines.extend(_slot_block(slot))
            continue
        out_lines.append(raw_line)
    text = '\n'.join(out_lines)
    if not text.endswith('\n'):
        text += '\n'
    return text


def generate() -> str:
    """Read template, return the fully rendered REFERENCE.yaml content."""
    with open(TEMPLATE_PATH, 'r') as f:
        template = f.read()
    return render(template)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _diff(old: str, new: str, *, label: str = 'configs/REFERENCE.yaml') -> str:
    return ''.join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"{label} (checked-in)",
        tofile=f"{label} (regenerated)",
    ))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--write', action='store_true',
                      help='Overwrite configs/REFERENCE.yaml in place.')
    mode.add_argument('--check', action='store_true',
                      help='Exit non-zero if the generated content '
                           'differs from configs/REFERENCE.yaml.')
    args = p.parse_args(argv)

    new = generate()
    try:
        with open(OUTPUT_PATH, 'r') as f:
            old = f.read()
    except FileNotFoundError:
        old = ''

    if args.write:
        with open(OUTPUT_PATH, 'w') as f:
            f.write(new)
        if old == new:
            print(f"[generate_reference_yaml] {OUTPUT_PATH} unchanged.")
        else:
            print(f"[generate_reference_yaml] wrote {OUTPUT_PATH} "
                  f"({len(new.splitlines())} lines).")
        return 0

    if args.check:
        if old == new:
            return 0
        sys.stderr.write(
            "configs/REFERENCE.yaml is out of date with the dataclass "
            "schema.\nRerun: python scripts/generate_reference_yaml.py "
            "--write\n\n"
        )
        sys.stderr.write(_diff(old, new))
        return 1

    # Default: dry-run -- show the generated content + diff vs current.
    sys.stdout.write(new)
    if old != new:
        sys.stderr.write('\n--- diff vs configs/REFERENCE.yaml ---\n')
        sys.stderr.write(_diff(old, new))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
