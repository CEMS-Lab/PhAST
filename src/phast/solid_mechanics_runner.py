"""YAML runner dispatch for promoted solid-mechanics examples."""
from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


_SUPPORTED_EXAMPLES = {
    "solid_mechanics.linear_plate": "phast.solid_mechanics_examples.linear_plate",
    "solid_mechanics.neohookean_plate": "phast.solid_mechanics_examples.neohookean_plate",
    "solid_mechanics.j2_bar": "phast.solid_mechanics_examples.j2_bar",
}


def solid_example_id(raw: dict[str, Any]) -> str | None:
    """Return the promoted solid-mechanics example id, if this is one."""
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
    example_id = solid_example_id(raw)
    if example_id is None:
        raise ValueError(f"Not a supported solid-mechanics config: {config_path}")

    if validate_only:
        print(f"OK: {config_path} is a supported solid mechanics YAML config.")
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
