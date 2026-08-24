"""YAML runner dispatch for promoted solid-mechanics examples."""
from __future__ import annotations

import importlib
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


_SUPPORTED_EXAMPLES = {
    "solid_mechanics.linear_plate": "phast.solid_mechanics_runners.linear_plate",
    "solid_mechanics.neohookean_plate": "phast.solid_mechanics_runners.neohookean_plate",
    "solid_mechanics.j2_bar": "phast.solid_mechanics_runners.j2_bar",
}


def _validate_solid_mechanics_config(
    raw: dict[str, Any], example_id: str,
) -> list[str]:
    """Return concise schema and semantic errors for a public solid example."""
    errors: list[str] = []

    if raw.get("schema_version") != 1:
        errors.append("schema_version must be the integer 1")

    def section(name: str) -> dict[str, Any]:
        value = raw.get(name)
        if not isinstance(value, dict):
            errors.append(f"{name} must be a mapping")
            return {}
        return value

    mesh = section("mesh")
    material = section("material")
    loading = section("loading")

    def number(
        values: dict[str, Any], section_name: str, key: str, *,
        positive: bool = False, nonnegative: bool = False,
    ) -> float | None:
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            errors.append(f"{section_name}.{key} must be a number")
            return None
        try:
            converted = float(value)
        except ValueError:
            errors.append(f"{section_name}.{key} must be a number")
            return None
        if not math.isfinite(converted):
            errors.append(f"{section_name}.{key} must be finite")
            return None
        if positive and converted <= 0.0:
            errors.append(f"{section_name}.{key} must be greater than zero")
        if nonnegative and converted < 0.0:
            errors.append(f"{section_name}.{key} must be nonnegative")
        return converted

    def positive_integer(
        values: dict[str, Any], section_name: str, key: str,
    ) -> int | None:
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{section_name}.{key} must be an integer")
            return None
        if value <= 0:
            errors.append(f"{section_name}.{key} must be greater than zero")
        return value

    positive_integer(mesh, "mesh", "nx")
    positive_integer(mesh, "mesh", "ny")
    number(mesh, "mesh", "length", positive=True)
    number(mesh, "mesh", "height", positive=True)
    number(material, "material", "E", positive=True)
    nu = number(material, "material", "nu")
    if nu is not None and not (-1.0 < nu < 0.5):
        errors.append("material.nu must satisfy -1 < nu < 0.5")

    if example_id == "solid_mechanics.linear_plate":
        number(loading, "loading", "tip_force_y")
    elif example_id == "solid_mechanics.neohookean_plate":
        positive_integer(loading, "loading", "load_steps")
        number(
            loading, "loading", "target_linear_tip_displacement_fraction",
            positive=True,
        )
        number(loading, "loading", "load_scale", positive=True)
    elif example_id == "solid_mechanics.j2_bar":
        number(mesh, "mesh", "waist_depth", nonnegative=True)
        width = number(mesh, "mesh", "waist_width_fraction", positive=True)
        if width is not None and width >= 1.0:
            errors.append("mesh.waist_width_fraction must be less than one")
        number(material, "material", "sigma_y0", positive=True)
        number(material, "material", "hardening_modulus", nonnegative=True)
        positive_integer(loading, "loading", "n_steps")
        number(loading, "loading", "max_strain_xx", positive=True)
        solver = section("solver")
        number(solver, "solver", "tol", positive=True)
        number(solver, "solver", "tol_rel", positive=True)
        positive_integer(solver, "solver", "max_iter")
        backend = solver.get("backend")
        if not isinstance(backend, str) or not backend.strip():
            errors.append("solver.backend must be a non-empty string")

    output = raw.get("output")
    if output is not None:
        if not isinstance(output, dict):
            errors.append("output must be a mapping")
        elif "directory" in output and not isinstance(output["directory"], str):
            errors.append("output.directory must be a string")

    return errors


def solid_example_id(raw: dict[str, Any]) -> str | None:
    """Return the promoted solid-mechanics example id, if this is one."""
    if not isinstance(raw, dict):
        return None
    example = raw.get("example")
    if isinstance(example, str) and example in _SUPPORTED_EXAMPLES:
        return example

    problem = raw.get("problem")
    if isinstance(problem, dict):
        problem_type = problem.get("type")
        if isinstance(problem_type, str):
            if problem_type in _SUPPORTED_EXAMPLES:
                return problem_type
            candidate = f"solid_mechanics.{problem_type}"
            if candidate in _SUPPORTED_EXAMPLES:
                return candidate
    return None


def is_solid_mechanics_config(config_path: str | os.PathLike) -> bool:
    """True if the YAML config targets a promoted solid-mechanics runner."""
    try:
        raw = yaml.safe_load(Path(config_path).read_text()) or {}
    except OSError:
        return False
    return solid_example_id(raw) is not None


@contextmanager
def _temporary_env(overrides: dict[str, str | None]):
    old = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_run_module(example_id: str):
    return importlib.import_module(_SUPPORTED_EXAMPLES[example_id])


def run_solid_mechanics_config(
    config_path: str | os.PathLike,
    *,
    output_dir: str | os.PathLike | None = None,
    validate_only: bool = False,
) -> int:
    """Run a promoted solid-mechanics YAML config through the common CLI."""
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("solid-mechanics configuration must be a YAML mapping")
    example_id = solid_example_id(raw)
    if example_id is None:
        raise ValueError(f"Not a supported solid-mechanics config: {config_path}")

    errors = _validate_solid_mechanics_config(raw, example_id)
    if errors:
        details = "\n".join(f"- {message}" for message in errors)
        raise ValueError(f"invalid solid-mechanics configuration:\n{details}")

    if validate_only:
        print(f"OK: {config_path} passes solid-mechanics schema validation.")
        return 0

    command = f"python -m phast run {config_path}"
    if output_dir is not None:
        command += f" --output_dir {output_dir}"
    print(f"Solid mechanics YAML runner: {example_id}")
    module = _load_run_module(example_id)
    env = {
        "PHAST_SOLID_MECH_COMMAND": command,
        "PHAST_SOLID_MECH_OUTPUT_DIR": (
            os.fspath(output_dir) if output_dir is not None else None
        ),
    }
    with _temporary_env(env):
        module.run(path)
    return 0
