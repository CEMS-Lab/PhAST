"""Utilities for retained promoted plasticity/interface result bundles."""
from __future__ import annotations

import csv
import json
from contextlib import nullcontext
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


def write_trajectory(
    output_dir: Path,
    *,
    nodes: np.ndarray,
    elements: np.ndarray,
    snapshots: list[tuple[int, Mapping[str, np.ndarray]]],
    metadata: Mapping[str, object],
    trajectory_format: str = "h5",
) -> Path:
    """Write HDF5 by default, or Zarr when explicitly requested.

    Both formats retain the same per-step field layout. The HDF5 handle is
    closed before returning so the completed file can be copied or synced.
    """
    if trajectory_format not in ("h5", "zarr"):
        raise ValueError("trajectory_format must be h5 or zarr")
    path = output_dir / f"training_data.{trajectory_format}"
    if trajectory_format == "h5":
        import h5py

        store = h5py.File(path, "w")
    else:
        import zarr

        store = nullcontext(zarr.open_group(str(path), mode="w"))

    with store as root:
        root.attrs["format"] = f"phast.trajectory.{trajectory_format}"
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
                data = np.asarray(array)
                compression = (
                    {"compression": "gzip", "compression_opts": 4}
                    if trajectory_format == "h5" and data.ndim > 0 else {}
                )
                group.create_dataset(name, data=data, **compression)
    return path


def write_zarr_trajectory(
    output_dir: Path,
    *,
    nodes: np.ndarray,
    elements: np.ndarray,
    snapshots: list[tuple[int, Mapping[str, np.ndarray]]],
    metadata: Mapping[str, object],
) -> Path:
    """Compatibility entry point for callers explicitly requesting Zarr."""
    return write_trajectory(
        output_dir, nodes=nodes, elements=elements, snapshots=snapshots,
        metadata=metadata, trajectory_format="zarr",
    )


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
