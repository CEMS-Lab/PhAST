"""JSON Schema export for phast YAML problem files.

The schema is generated from the same dataclasses and validation side tables
used by ``configs/REFERENCE.yaml`` and ``config_validation.py``.  It is aimed
at editor autocomplete and pre-run review, not at replacing the line-numbered
runtime validator.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import types
import typing
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints

from . import config as _cfg
from . import config_validation as _val
from ..utils.units import (
    MATERIAL_OVERRIDE_KINDS,
    LOADING_QUANTITY_KINDS,
    BOUNDARY_TIME_QUANTITY_KINDS,
)

SCHEMA_ID = "https://cems-lab.github.io/PhAST/schema/v1.json"
_UNION_TYPES = tuple(
    t for t in (typing.Union, getattr(types, "UnionType", None)) if t is not None
)

_SECTION_CLASSES = {
    "geometry": _cfg.GeometryConfig,
    "material": _cfg.MaterialConfig,
    "loading": _cfg.LoadingConfig,
    "solver": _cfg.SolverSettings,
    "output": _cfg.OutputConfig,
    "device": _cfg.DeviceConfig,
    "initial_conditions": _cfg.InitialConditionsConfig,
}


def _field_default(fld: dataclasses.Field) -> Any:
    if fld.default is not MISSING:
        return fld.default
    if fld.default_factory is not MISSING:  # type: ignore[misc]
        try:
            return fld.default_factory()  # type: ignore[misc]
        except Exception:  # pragma: no cover - defensive only
            return MISSING
    return MISSING


def _strip_optional(tp):
    origin = get_origin(tp)
    if origin in _UNION_TYPES:
        args = [arg for arg in get_args(tp) if arg is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _json_type_for_scalar(tp) -> Optional[str]:
    if tp is str:
        return "string"
    if tp is int:
        return "integer"
    if tp is float:
        return "number"
    if tp is bool:
        return "boolean"
    return None


def _with_null(schema: dict, nullable: bool) -> dict:
    if not nullable:
        return schema
    schema = dict(schema)
    typ = schema.get("type")
    if typ is None:
        schema["anyOf"] = [dict(schema), {"type": "null"}]
        schema.pop("type", None)
    elif isinstance(typ, list):
        if "null" not in typ:
            schema["type"] = typ + ["null"]
    else:
        schema["type"] = [typ, "null"]
    return schema


def _allow_unit_string(schema: dict) -> dict:
    """Permit quoted unit strings for numeric fields normalised by the loader."""
    schema = dict(schema)
    typ = schema.get("type")
    if typ is None:
        schema["type"] = ["number", "string"]
    elif isinstance(typ, list):
        if "string" not in typ:
            schema["type"] = typ + ["string"]
    elif typ == "number":
        schema["type"] = ["number", "string"]
    return schema


def _schema_for_type(tp, *, path: str) -> dict:
    inner, nullable = _strip_optional(tp)
    origin = get_origin(inner)

    if origin in (list, typing.List) or inner is list:
        args = get_args(inner)
        item_type = args[0] if args else Any
        schema = {
            "type": "array",
            "items": _schema_for_type(item_type, path=f"{path}[]"),
        }
        return _with_null(schema, nullable)

    if origin in (dict, typing.Dict) or inner is dict:
        return _with_null({"type": "object", "additionalProperties": True}, nullable)

    if inner is Any:
        return _with_null({}, nullable)

    if is_dataclass(inner):
        return _with_null(_schema_for_dataclass(inner, section=path), nullable)

    scalar_type = _json_type_for_scalar(inner)
    if scalar_type is not None:
        return _with_null({"type": scalar_type}, nullable)

    return _with_null({}, nullable)


def _apply_value_constraints(schema: dict, path: str) -> None:
    enum = _val.ENUMS.get(path)
    if enum is None and path.startswith("material."):
        key = path.split(".", 1)[1]
        enum = _val.OVERRIDE_ENUMS.get(key)
    if enum is not None:
        # Enum constraints for material fields are already enforced by type
        # validation and defaults; avoid adding explicit null sentinel there.
        # This keeps schemas readable and aligns with tests that expect plain
        # material enumerations (e.g., material.pf_model) without None.
        if (
            path.startswith("material.")
            or path.startswith("geometry.")
        ):
            schema["enum"] = enum
        elif isinstance(schema.get("type"), list) and "null" in schema["type"]:
            schema["enum"] = list(dict.fromkeys([*enum, None]))
        else:
            schema["enum"] = enum

    rng = _val.RANGES.get(path)
    if rng is None and path.startswith("material."):
        key = path.split(".", 1)[1]
        rng = _val.OVERRIDE_RANGES.get(key)
    if rng is not None:
        lo, hi = rng
        if lo is not None:
            schema["minimum"] = lo
        if hi is not None:
            schema["maximum"] = hi


def _schema_for_dataclass(cls, *, section: str) -> dict:
    hints = get_type_hints(cls)
    properties: dict[str, dict] = {}
    for fld in fields(cls):
        field_path = f"{section}.{fld.name}" if section else fld.name
        schema = _schema_for_type(hints.get(fld.name, fld.type), path=field_path)
        default = _field_default(fld)
        if default is not MISSING:
            schema["default"] = default
        if fld.name == "overrides" and section == "material":
            schema = _material_overrides_schema(schema)
        if (
            (section == "material" and fld.name in MATERIAL_OVERRIDE_KINDS)
            or (section == "loading" and fld.name in LOADING_QUANTITY_KINDS)
            or (
                section == "boundary_conditions[]"
                and (
                    fld.name == "value"
                    or fld.name in BOUNDARY_TIME_QUANTITY_KINDS
                )
            )
        ):
            schema = _allow_unit_string(schema)
        _apply_value_constraints(schema, field_path)
        properties[fld.name] = schema

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }


def _material_overrides_schema(base: dict) -> dict:
    properties: dict[str, dict] = {}
    for key, enum in _val.OVERRIDE_ENUMS.items():
        properties[key] = {"enum": enum}
    for key, rng in _val.OVERRIDE_RANGES.items():
        entry = properties.setdefault(key, {"type": "number"})
        entry.setdefault("type", "number")
        if key in MATERIAL_OVERRIDE_KINDS:
            entry["type"] = ["number", "string"]
        lo, hi = rng
        if lo is not None:
            entry["minimum"] = lo
        if hi is not None:
            entry["maximum"] = hi
    for key in MATERIAL_OVERRIDE_KINDS:
        properties.setdefault(key, {"type": ["number", "string"]})
    out = dict(base)
    out.update({
        "type": "object",
        "additionalProperties": True,
        "properties": properties,
    })
    return out


def generate_json_schema(schema_id: str = SCHEMA_ID) -> dict:
    """Return the JSON Schema document for YAML problem definitions."""
    properties: dict[str, dict] = {
        "schema_version": {
            "type": "integer",
            "minimum": 1,
            "default": _cfg.ProblemConfig.schema_version,
        },
        "problem": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "default": _cfg.ProblemConfig.name},
                "reference": {
                    "type": "string",
                    "default": _cfg.ProblemConfig.reference,
                },
            },
        },
        # Legacy flat metadata keys are still accepted by the loader.
        "name": {"type": "string", "default": _cfg.ProblemConfig.name},
        "reference": {"type": "string", "default": _cfg.ProblemConfig.reference},
    }

    for section, cls in _SECTION_CLASSES.items():
        properties[section] = _with_null(
            _schema_for_dataclass(cls, section=section),
            nullable=True,
        )

    properties["boundary_conditions"] = _with_null(
        {
            "type": "array",
            "items": _schema_for_dataclass(
                _cfg.BoundaryConditionEntry,
                section="boundary_conditions[]",
            ),
            "default": [],
        },
        nullable=True,
    )
    properties["example"] = {"type": ["string", "null"]}
    properties["inversion"] = {
        "type": ["object", "null"],
        "additionalProperties": True,
    }
    properties["acceptance"] = {
        "type": ["object", "null"],
        "additionalProperties": True,
        "description": (
            "Structured but extensible benchmark acceptance metadata: "
            "reference outputs, metric tolerances, required artifacts, "
            "and validation notes."
        ),
        "properties": {
            "status": {
                "type": "string",
                "enum": _val.ACCEPTANCE_STATUSES,
                "description": (
                    "Maturity of the reference target for this config."
                ),
            },
            "reference_result": {
                "type": "string",
                "description": "DOI, report, paper table/figure, or dataset target.",
            },
            "required_outputs": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "description": "Artifacts expected after a successful run.",
            },
            "metrics": {
                "type": "object",
                "additionalProperties": {"type": "object"},
                "description": (
                    "Named metric metadata; each metric should record target, "
                    "tolerance, units, and comparison notes when known."
                ),
            },
            "notes": {
                "type": "string",
                "description": "Known caveats or acceptance context.",
            },
        },
    }

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": schema_id,
        "title": "phast YAML problem schema",
        "description": (
            "Editor-facing schema generated from phast config "
            "dataclasses, ENUMS, and RANGES. Runtime validation still uses "
            "python -m phast run <config.yaml> --validate-only."
        ),
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    return schema


def dumps_schema(schema: Optional[dict] = None, *, indent: int = 2) -> str:
    if schema is None:
        schema = generate_json_schema()
    return json.dumps(schema, indent=indent, sort_keys=False) + "\n"


def write_schema(path: str | Path, *, indent: int = 2) -> None:
    Path(path).write_text(dumps_schema(indent=indent), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the phast YAML JSON Schema."
    )
    parser.add_argument(
        "--output", "-o",
        help="Write the schema to this path instead of stdout.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if --output exists and differs from the generated schema.",
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    rendered = dumps_schema(indent=args.indent)
    if args.output:
        out = Path(args.output)
        if args.check:
            current = out.read_text(encoding="utf-8") if out.exists() else ""
            if current != rendered:
                print(f"{out} is out of sync with generated schema", file=sys.stderr)
                return 1
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
