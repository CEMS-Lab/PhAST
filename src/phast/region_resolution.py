"""Read-only mesh-to-workflow-region resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .mesh_inspection import inspect_mesh
from .workflow.specs import RegionSpec


class RegionResolutionError(ValueError):
    """Raised when workflow regions cannot be resolved against a mesh."""


_SELECTOR_KEYS = (
    "from_mesh",
    "mesh_group",
    "physical_group",
    "node_set",
    "element_set",
)


def resolve_regions(
    mesh_path: str | Path,
    regions: Iterable[RegionSpec],
) -> dict[str, Any]:
    """Resolve workflow regions against meshio-readable mesh groups.

    This function validates the mapping contract only. It does not construct
    FEM tensors, material assignments, or solver boundary-condition objects.
    """
    summary = inspect_mesh(mesh_path)
    resolved: dict[str, dict[str, Any]] = {}
    external_to_internal: dict[tuple[str, str], str] = {}
    for region in regions:
        selector_key, external_name = _region_selector(region)
        if selector_key is None or external_name is None:
            continue
        duplicate_key = (selector_key, external_name)
        if duplicate_key in external_to_internal:
            raise RegionResolutionError(
                f"Mesh group {external_name!r} maps to multiple regions: "
                f"{external_to_internal[duplicate_key]!r}, {region.name!r}"
            )
        external_to_internal[duplicate_key] = region.name
        resolved[region.name] = _resolve_one(summary, selector_key, external_name)
    return {"mesh": summary["path"], "regions": resolved}


def _region_selector(region: RegionSpec) -> tuple[str | None, str | None]:
    for key in _SELECTOR_KEYS:
        value = region.selector.get(key)
        if value not in (None, ""):
            return key, str(value)
    return None, None


def _resolve_one(
    summary: dict[str, Any],
    selector_key: str,
    external_name: str,
) -> dict[str, Any]:
    named_groups = summary.get("named_groups", {})
    point_sets = summary.get("point_sets", {})
    cell_sets = summary.get("cell_sets", {})

    if selector_key in {"from_mesh", "mesh_group", "physical_group"}:
        if external_name in named_groups:
            group = dict(named_groups[external_name])
            return {
                "source": "named_group",
                "external_name": external_name,
                "dimension": group.get("dimension"),
                "cell_counts": dict(group.get("cell_counts") or {}),
            }
        if selector_key == "physical_group":
            raise _missing(summary, external_name)
    if selector_key in {"from_mesh", "mesh_group", "node_set"}:
        if external_name in point_sets:
            return {
                "source": "point_set",
                "external_name": external_name,
                "count": int(point_sets[external_name]),
            }
        if selector_key == "node_set":
            raise _missing(summary, external_name)
    if selector_key in {"from_mesh", "mesh_group", "element_set"}:
        if external_name in cell_sets:
            return {
                "source": "cell_set",
                "external_name": external_name,
                "cell_counts": dict(cell_sets[external_name]),
            }
        if selector_key == "element_set":
            raise _missing(summary, external_name)
    raise _missing(summary, external_name)


def _missing(summary: dict[str, Any], external_name: str) -> RegionResolutionError:
    available = sorted(
        set(summary.get("named_groups", {}))
        | set(summary.get("point_sets", {}))
        | set(summary.get("cell_sets", {}))
    )
    return RegionResolutionError(
        f"Region references mesh group {external_name!r}, but it is missing. "
        f"Available mesh groups: {available}"
    )
