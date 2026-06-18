"""Utilities for retained promoted plasticity/interface result bundles."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def write_csv_rows(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_zarr_trajectory(
    output_dir: Path,
    *,
    nodes: np.ndarray,
    elements: np.ndarray,
    snapshots: list[tuple[int, Mapping[str, np.ndarray]]],
    metadata: Mapping[str, object],
) -> Path:
    """Write a compact PhAST-compatible Zarr trajectory store."""
    import zarr

    zarr_path = output_dir / "training_data.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["format"] = "phast.trajectory.zarr"
    root.attrs["writer"] = "examples.plasticity_interface_beta._promoted_result_utils"

    sim = root.create_group("simulation_data")
    mesh = sim.create_group("mesh")
    mesh.create_dataset("node_coordinates", data=np.asarray(nodes, dtype=np.float64))
    mesh.create_dataset("element_connectivity", data=np.asarray(elements, dtype=np.int64))
    meta = sim.create_group("metadata")
    for key, value in metadata.items():
        meta.attrs[key] = value

    steps = sim.create_group("steps")
    for step, arrays in snapshots:
        group = steps.create_group(f"step_{int(step):04d}")
        for name, array in arrays.items():
            group.create_dataset(name, data=np.asarray(array))
    return zarr_path


def merge_run_manifest_artifacts(output_dir: Path, required_names: list[str]) -> None:
    """Ensure run_manifest.json lists every retained promoted artifact."""
    path = output_dir / "run_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema": "phast_run_manifest_v1"}
    existing = list(manifest.get("artifacts") or [])
    for name in required_names:
        if name not in existing:
            existing.append(name)
    manifest["artifacts"] = existing
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def ensure_json_file(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
