"""Shared helpers for public solid-mechanics examples."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_config_arg(description: str) -> Path | None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the example YAML config. Defaults to config.yaml beside run.py.",
    )
    return parser.parse_args().config


def load_config(path: str | Path | None, defaults: dict[str, Any]) -> dict[str, Any]:
    """Load a small YAML config if PyYAML is available; fall back to defaults."""
    if path is None:
        path = Path(__file__).resolve().parent / "config.yaml"
    path = Path(path)
    cfg = dict(defaults)
    if not path.exists():
        return cfg
    try:
        import yaml
    except Exception:
        return cfg
    data = yaml.safe_load(path.read_text()) or {}
    return _deep_update(cfg, data)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def prepare_output_dir(example_file: str | Path, cfg: dict[str, Any]) -> Path:
    env_out = os.environ.get("PHAST_SOLID_MECH_OUTPUT_DIR")
    if env_out:
        path = Path(env_out)
        path.mkdir(parents=True, exist_ok=True)
        return path
    out = cfg.get("output", {}).get("directory", "outputs")
    path = Path(out)
    if not path.is_absolute():
        path = Path(example_file).resolve().parent / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_commit(repo_root: Path | None = None) -> str:
    cwd = repo_root or Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def write_manifest(
    out_dir: Path,
    *,
    example: str,
    command: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    files: list[str],
    started_at: float,
) -> None:
    command = os.environ.get("PHAST_SOLID_MECH_COMMAND", command)
    manifest = {
        "schema_version": 1,
        "example": example,
        "command": command,
        "runtime_seconds": round(time.perf_counter() - started_at, 6),
        "git_commit": git_commit(),
        "config": config,
        "metrics": metrics,
        "files": files,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def copy_thumbnail(out_dir: Path, source_name: str = "response.png") -> None:
    src = out_dir / source_name
    if src.exists():
        shutil.copyfile(src, out_dir / "thumbnail.png")


def write_diagnostic_setup_preview(
    out_dir: Path,
    *,
    title: str,
    config: dict[str, Any],
) -> None:
    """Write the standard setup preview for numerical-method diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=160)
    ax.axis("off")
    lines = [
        title,
        "",
        "Numerical-method diagnostic; not a mesh-level FEA solve.",
    ]
    for key, value in config.items():
        if key in {"schema_version", "output"}:
            continue
        lines.append(f"{key}: {value}")
    ax.text(
        0.04,
        0.94,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "initial_conditions.png")
    plt.close(fig)
