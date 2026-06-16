"""Curated YAML runner dispatch for beta plasticity/interface validations."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable

import yaml


_SUPPORTED_VALIDATIONS = {
    "diffuse_interphase": (
        "examples.plasticity_interface.run_diffuse_interphase_validation",
        "run_validation",
    ),
    "j2_validation": (
        "examples.plasticity_interface.run_j2_validation",
        "run_validation",
    ),
    "pfczm_uniaxial_strength": (
        "examples.plasticity_interface.run_pfczm_uniaxial_strength_validation",
        "run_validation",
    ),
    "structural_dcb_cohesive": (
        "examples.plasticity_interface.run_structural_dcb_cohesive_benchmark",
        "run_benchmark",
    ),
    "structural_dcb_refinement": (
        "examples.plasticity_interface.run_structural_dcb_refinement_study",
        "run_study",
    ),
}


def is_plasticity_interface_contract(config_path: str | os.PathLike) -> bool:
    """True if the YAML file is the curated plasticity/interface contract."""
    try:
        raw = yaml.safe_load(Path(config_path).read_text()) or {}
    except OSError:
        return False
    if raw.get("manifest_type") != "reproducibility_contract":
        return False
    base_paths = raw.get("base_paths") or {}
    examples_path = str(base_paths.get("examples", ""))
    runs = raw.get("runs") or []
    return (
        "plasticity_interface" in examples_path
        and any(str(run.get("id", "")) in _SUPPORTED_VALIDATIONS for run in runs)
    )


def _contract_runs(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(run["id"]): dict(run or {}) for run in raw.get("runs") or []}


def _default_output_dir(run: dict[str, Any]) -> Path | None:
    launcher = list(run.get("launcher") or [])
    try:
        index = launcher.index("--output-dir")
    except ValueError:
        return None
    try:
        return Path(str(launcher[index + 1]))
    except IndexError:
        return None


def _load_callable(validation_id: str) -> Callable[..., dict[str, Any]]:
    try:
        module_name, func_name = _SUPPORTED_VALIDATIONS[validation_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_SUPPORTED_VALIDATIONS))
        raise ValueError(
            f"Plasticity/interface validation {validation_id!r} is not wired "
            f"for curated YAML execution. Supported: {supported}"
        ) from exc
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def run_plasticity_interface_contract(
    config_path: str | os.PathLike,
    *,
    validation_id: str | None = None,
    output_dir: str | os.PathLike | None = None,
    validate_only: bool = False,
) -> int:
    """Run an allowlisted beta validation from the curated manifest."""
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text()) or {}
    if raw.get("manifest_type") != "reproducibility_contract":
        raise ValueError(f"Not a plasticity/interface validation contract: {config_path}")

    runs = _contract_runs(raw)
    selected = validation_id or next(iter(runs), None)
    if selected is None or selected not in runs:
        available = ", ".join(sorted(runs))
        raise ValueError(
            f"Unknown plasticity/interface validation {selected!r}. "
            f"Available: {available}"
        )
    _load_callable(selected)

    if validate_only:
        print(
            f"OK: {config_path} is a supported plasticity/interface "
            f"validation contract ({selected})."
        )
        return 0

    out = Path(output_dir) if output_dir is not None else _default_output_dir(runs[selected])
    if out is None:
        raise ValueError(
            f"Validation {selected!r} does not declare an output directory and "
            "--output_dir was not provided."
        )
    print(f"Plasticity/interface YAML runner: {selected}")
    run_validation = _load_callable(selected)
    run_validation(out)
    return 0
