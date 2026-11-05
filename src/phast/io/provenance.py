"""Run provenance and lockfile helpers.

``run_metadata.json`` is a compact post-processing contract.  The lockfile
written here is the reproducibility contract: exact resolved YAML, CLI
overrides, selected runtime objects, git state, and dependency versions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        arr = value.detach().cpu()
        if arr.ndim == 0:
            return arr.item()
        return f"<tensor shape={tuple(arr.shape)} dtype={arr.dtype}>"
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_file(path: Union[str, os.PathLike]) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _run_git(args: list[str], cwd: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_state(repo_dir: Optional[str] = None) -> dict:
    repo_dir = repo_dir or os.path.dirname(os.path.abspath(__file__))
    full = _run_git(["rev-parse", "HEAD"], repo_dir)
    short = _run_git(["rev-parse", "--short", "HEAD"], repo_dir)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    status = _run_git(["status", "--porcelain"], repo_dir)
    out = {
        "commit": full,
        "commit_short": short,
        "branch": branch,
        "dirty": bool(status),
    }
    if status:
        out["dirty_paths"] = [line[3:] for line in status.splitlines() if len(line) > 3]
    return out


def dependency_versions() -> dict:
    packages = [
        "phast",
        "torch",
        "numpy",
        "scipy",
        "pyyaml",
        "h5py",
        "meshio",
        "gmsh",
        "matplotlib",
        "zarr",
        "pyamg",
    ]
    versions: dict[str, Optional[str]] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _object_fields(obj: Any, names: list[str]) -> dict:
    return {
        name: _jsonable(getattr(obj, name))
        for name in names
        if hasattr(obj, name)
    }


def _mesh_summary(mesh: Any) -> dict:
    summary = _object_fields(
        mesh,
        ["mesh_path", "n_nodes", "n_elems", "h_min"],
    )
    if hasattr(mesh, "node_sets"):
        summary["node_sets"] = sorted(str(k) for k in getattr(mesh, "node_sets", {}))
    if hasattr(mesh, "elem_h"):
        try:
            elem_h = getattr(mesh, "elem_h")
            summary["h_max"] = float(elem_h.max())
            summary["h_mean"] = float(elem_h.mean())
        except Exception:
            pass
    return summary


def _material_summary(material: Any) -> dict:
    return _object_fields(
        material,
        [
            "E",
            "nu",
            "Gc",
            "l0",
            "rho",
            "energy_split",
            "pf_model",
            "eta_residual",
            "plane_stress",
            "kinematics",
        ],
    )


def _solver_summary(solver_config: Any) -> dict:
    if dataclasses.is_dataclass(solver_config):
        return _jsonable(solver_config)
    if isinstance(solver_config, dict):
        return _jsonable(solver_config)
    names = [
        "solver_type",
        "time_integrator",
        "rho_inf",
        "dt_safety",
        "stagger_tol",
        "max_stagger",
        "stagger_criterion",
        "stagger_norm",
        "preconditioner",
        "damage_tol",
        "static_tol",
        "bounds_method",
        "damage_every",
        "damage_max_iter",
        "static_max_iter",
        "H_update_method",
        "enable_damage",
        "backend",
    ]
    return _object_fields(solver_config, names)


def build_run_lockfile(
    *,
    config,
    config_path: str,
    output_dir: str,
    args: Any = None,
    mesh: Any = None,
    material: Any = None,
    solver_config: Any = None,
    ctx: Any = None,
) -> dict:
    """Build the run lockfile document.

    The ``config`` argument should be the post-CLI-override
    ``ProblemConfig`` so the lockfile matches ``config.yaml``.
    """
    abs_config = os.path.abspath(config_path)
    lock = {
        "lockfile_schema": "phast.run_lockfile.v1",
        "created_at": datetime.now().isoformat(),
        "input": {
            "config_file": abs_config,
            "config_sha256": _sha256_file(abs_config),
            "output_dir": os.path.abspath(output_dir),
        },
        "resolved_config": _jsonable(config),
        "runtime": {
            "hostname": socket.gethostname(),
            "cwd": os.getcwd(),
            "argv": list(sys.argv),
            "cli_args": _jsonable(vars(args)) if args is not None else {},
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
            },
            "git": git_state(),
            "dependencies": dependency_versions(),
        },
        "resolved_objects": {
            "mesh": _mesh_summary(mesh) if mesh is not None else {},
            "material": _material_summary(material) if material is not None else {},
            "solver": _solver_summary(solver_config) if solver_config is not None else {},
        },
    }
    if ctx is not None:
        lock["resolved_objects"]["device"] = {
            "device": str(getattr(ctx, "device", "")),
            "dtype": str(getattr(ctx, "dtype", "")),
        }
    return lock


def write_run_lockfile(path: Union[str, os.PathLike], **kwargs) -> str:
    path = str(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(build_run_lockfile(**kwargs), fh, indent=2, default=str)
        fh.write("\n")
    return path
