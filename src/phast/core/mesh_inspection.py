"""Read-only mesh discovery helpers for workflow setup."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def inspect_mesh(path: str | Path) -> dict[str, Any]:
    """Return meshio-backed metadata for a mesh file without constructing FEM state."""
    mesh_path = Path(path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file does not exist: {mesh_path}")
    try:
        import meshio
    except ImportError as exc:
        raise ImportError("phast.inspect_mesh() requires the meshio package") from exc

    mesh = meshio.read(mesh_path)
    return {
        "path": str(mesh_path),
        "n_points": int(len(mesh.points)),
        "cells": _cell_summaries(mesh),
        "named_groups": _named_group_summaries(mesh),
        "point_sets": _set_summaries(getattr(mesh, "point_sets", {}) or {}),
        "cell_sets": _cell_set_summaries(mesh),
    }


def _cell_summaries(mesh) -> list[dict[str, int | str]]:
    summaries = []
    for block in mesh.cells:
        data = np.asarray(block.data)
        nodes_per_cell = int(data.shape[1]) if data.ndim == 2 else 0
        summaries.append(
            {
                "type": str(block.type),
                "count": int(data.shape[0]),
                "nodes_per_cell": nodes_per_cell,
            }
        )
    return summaries


def _named_group_summaries(mesh) -> dict[str, dict[str, Any]]:
    field_data = getattr(mesh, "field_data", {}) or {}
    physical_by_type = (
        getattr(mesh, "cell_data_dict", {}) or {}
    ).get("gmsh:physical", {})
    groups: dict[str, dict[str, Any]] = {}
    for name in sorted(field_data):
        raw = np.asarray(field_data[name]).reshape(-1)
        if raw.size == 0:
            continue
        group_id = int(raw[0])
        dimension = int(raw[1]) if raw.size > 1 else None
        cell_counts: dict[str, int] = {}
        for cell_type, physical_tags in physical_by_type.items():
            tags = np.asarray(physical_tags)
            count = int(np.count_nonzero(tags == group_id))
            if count:
                cell_counts[str(cell_type)] = count
        groups[str(name)] = {
            "id": group_id,
            "dimension": dimension,
            "cell_counts": cell_counts,
        }
    return groups


def _set_summaries(sets: dict[str, Any]) -> dict[str, int]:
    return {
        str(name): int(np.asarray(indices).size)
        for name, indices in sorted(sets.items())
    }


def _cell_set_summaries(mesh) -> dict[str, dict[str, int]]:
    cell_sets = getattr(mesh, "cell_sets", {}) or {}
    if not cell_sets:
        return {}
    block_types = [str(block.type) for block in mesh.cells]
    summaries: dict[str, dict[str, int]] = {}
    for name, blocks in sorted(cell_sets.items()):
        per_type: dict[str, int] = {}
        for index, indices in enumerate(blocks):
            count = int(np.asarray(indices).size)
            if count and index < len(block_types):
                per_type[block_types[index]] = count
        summaries[str(name)] = per_type
    return summaries
