"""Mesh-integration layer for cohesive zone elements (issue #261, epic #259).

Given nodal coordinates, T3/Q4 bulk connectivity, and a set of interior edges
defining the cohesive interface, produce a NEW mesh with nodes doubled along
the interface, plus the list of :class:`CohesiveElement` records connecting
the original (top) and duplicated (bottom) sides.

API
---
- :func:`insert_cohesive_layer` — pure-Python node-doubling driver. Operates on
  numpy arrays (``nodes``, ``elements``) so it can be exercised without an
  ``FEMMesh`` instance. Callers that already have an ``FEMMesh`` should pass
  its tensor data as arrays and rebuild with :meth:`FEMMesh.from_tensors`.

Side-selection convention
-------------------------
Marked edges are canonicalized by node id, so input orientation does not
change which bulk side is duplicated. For a canonical edge with endpoints
``p0``, ``p1`` and unit tangent ``t``, the edge normal is taken as
``n = (-t_y, t_x)`` (90 deg CCW rotation of the tangent — same convention as
:func:`build_cohesive_strip`). The "bottom" side (the side whose nodes are
duplicated) is the one whose element centroids lie on the *negative* side of
the normal averaged over all interface edges incident to the node. The "top"
side keeps the original node IDs.

This deliberately does NOT mutate the input arrays / mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, List, Sequence, Tuple

import numpy as np

from .cohesive_elements import CohesiveElement


@dataclass(frozen=True)
class CohesiveInsertionResult:
    """Metadata-preserving cohesive insertion result.

    ``node_sets`` keeps each input set valid on the doubled mesh: when a set
    contains an interface node, the corresponding duplicate node is included
    as well. Side-specific ``<name>_top`` / ``<name>_bottom`` sets are added
    for interface-bearing sets so callers can apply BCs to one side of the
    inserted cohesive layer. Element sets and element data are copied because
    the bulk element count and ordering are unchanged.
    """

    nodes: np.ndarray
    elements: np.ndarray
    cohesives: List[CohesiveElement]
    node_sets: dict[str, np.ndarray]
    element_sets: dict[str, np.ndarray]
    element_data: dict[str, np.ndarray]
    duplicate_node_map: dict[int, int]
    element_side: np.ndarray


@dataclass(frozen=True)
class MeshIOCohesiveInsertionResult:
    """Cohesive insertion result for an imported ``meshio.Mesh``."""

    mesh: object
    insertion: CohesiveInsertionResult
    # Legacy name kept for callers/tests from the original T3-only helper.
    # For Q4 meshes this is the selected bulk cell-block index.
    triangle_block_index: int
    interface_edges: list[tuple[int, int]]
    cell_type: str = "triangle"


def _edge_normal(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    edge = p1 - p0
    n = np.array([-edge[1], edge[0]])
    norm = np.linalg.norm(n)
    return n / norm if norm > 0.0 else n


def _classify_sides(
    nodes: np.ndarray,
    elements: np.ndarray,
    interface_edges: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Return a per-element side label in {-1, +1, 0}.

    ``+1`` = top (keeps original node IDs); ``-1`` = bottom (gets duplicated
    node IDs); ``0`` = element does not touch the interface (kept as-is).
    """
    interface_node_set = {int(n) for e in interface_edges for n in e}
    # average normal at each interface node (CCW-of-tangent convention)
    avg_normal = {n: np.zeros(2) for n in interface_node_set}
    midpoint = {n: np.zeros(2) for n in interface_node_set}
    counts = {n: 0 for n in interface_node_set}
    for n0, n1 in interface_edges:
        nrm = _edge_normal(nodes[n0], nodes[n1])
        mid = 0.5 * (nodes[n0] + nodes[n1])
        for n in (n0, n1):
            avg_normal[int(n)] += nrm
            midpoint[int(n)] += mid
            counts[int(n)] += 1
    for n in avg_normal:
        if counts[n] > 0:
            avg_normal[n] /= counts[n]
            midpoint[n] /= counts[n]

    side = np.zeros(elements.shape[0], dtype=int)
    for ei, elem in enumerate(elements):
        touched = [int(n) for n in elem if int(n) in interface_node_set]
        if not touched:
            continue
        centroid = nodes[elem].mean(axis=0)
        # vote across touched nodes' local frames
        s = 0.0
        for n in touched:
            s += float(np.dot(centroid - midpoint[n], avg_normal[n]))
        side[ei] = 1 if s >= 0.0 else -1
    return side


def _validate_inputs(
    nodes: np.ndarray,
    elements: np.ndarray,
    edges: Sequence[Tuple[int, int]],
) -> None:
    if nodes.ndim != 2 or nodes.shape[1] < 2:
        raise ValueError(
            f"nodes must have shape (N, >=2), got {tuple(nodes.shape)}")
    if elements.ndim != 2 or elements.shape[1] not in (3, 4):
        raise ValueError(
            "insert_cohesive_layer currently supports T3 triangle or Q4 quad "
            f"connectivity only, got shape {tuple(elements.shape)}")
    n_nodes = int(nodes.shape[0])
    seen = set()
    for raw_a, raw_b in edges:
        a, b = int(raw_a), int(raw_b)
        if a == b:
            raise ValueError(f"zero-length cohesive edge ({a}, {b})")
        if a < 0 or b < 0 or a >= n_nodes or b >= n_nodes:
            raise ValueError(
                f"cohesive edge ({a}, {b}) references nodes outside "
                f"[0, {n_nodes})")
        if float(np.linalg.norm(nodes[b] - nodes[a])) <= 0.0:
            raise ValueError(
                f"zero-length/coincident cohesive edge ({a}, {b})")
        key = (min(a, b), max(a, b))
        if key in seen:
            raise ValueError(f"duplicate cohesive edge {key}")
        seen.add(key)


def _canonical_edges(
        interface_edges: Sequence[Tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in interface_edges
    ]


def _orient_interface_edges(
    nodes: np.ndarray,
    interface_edges: Sequence[Tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return duplicate-free edges oriented consistently by geometry.

    Node-id canonicalization is useful for duplicate detection, but using it as
    the physical tangent can flip normals along an interface polyline. Each
    connected chain is therefore walked from its geometrically lower endpoint
    along the dominant coordinate axis; isolated edges reduce to the same
    convention as the previous min-id canonicalization for horizontal cases.
    """
    canonical = _canonical_edges(interface_edges)
    if not canonical:
        return []

    adjacency: dict[int, set[int]] = {}
    for a, b in canonical:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    def coord_key(n: int, axis: int) -> tuple[float, float, int]:
        other = 1 - axis
        return (float(nodes[n, axis]), float(nodes[n, other]), int(n))

    remaining = {tuple(edge) for edge in canonical}
    oriented: list[tuple[int, int]] = []
    while remaining:
        stack = [next(iter(remaining))[0]]
        component_nodes: set[int] = set()
        while stack:
            n = stack.pop()
            if n in component_nodes:
                continue
            component_nodes.add(n)
            stack.extend(adjacency.get(n, ()))

        coords = nodes[sorted(component_nodes), :2]
        span = np.ptp(coords, axis=0)
        axis = int(np.argmax(span))
        endpoints = [
            n for n in component_nodes
            if len(adjacency.get(n, ())) == 1
        ]
        if endpoints:
            start = min(endpoints, key=lambda n: coord_key(n, axis))
        else:
            start = min(component_nodes, key=lambda n: coord_key(n, axis))

        prev: int | None = None
        current = start
        visited_nodes: set[int] = set()
        while True:
            visited_nodes.add(current)
            candidates = [
                n for n in adjacency.get(current, ())
                if n != prev and tuple(sorted((current, n))) in remaining
            ]
            if not candidates:
                break
            nxt = min(candidates, key=lambda n: coord_key(n, axis))
            key = tuple(sorted((current, nxt)))
            remaining.remove(key)
            oriented.append((current, nxt))
            prev, current = current, nxt

        # Branches/cycles can leave edges unwalked from the first start node.
        for edge in list(remaining):
            if edge[0] in component_nodes or edge[1] in component_nodes:
                a, b = edge
                if coord_key(b, axis) < coord_key(a, axis):
                    a, b = b, a
                remaining.remove(edge)
                oriented.append((a, b))

    return oriented


def _insert_core(
    nodes: np.ndarray,
    elements: np.ndarray,
    interface_edges: Sequence[Tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, List[CohesiveElement], dict[int, int], np.ndarray]:
    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape[1] > 2:
        nodes_2d = nodes[:, :2]
    else:
        nodes_2d = nodes
    elements = np.asarray(elements, dtype=int)
    _validate_inputs(nodes_2d, elements, interface_edges)
    edges = _orient_interface_edges(nodes_2d, interface_edges)
    if not edges:
        return nodes.copy(), elements.copy(), [], {}, np.zeros(elements.shape[0], dtype=int)

    interface_nodes = sorted({n for e in edges for n in e})
    n0_orig = int(nodes.shape[0])
    duplicate_id = {orig: n0_orig + i for i, orig in enumerate(interface_nodes)}

    side = _classify_sides(nodes_2d, elements, edges)
    new_elements = elements.copy()
    for ei in range(elements.shape[0]):
        if side[ei] != -1:
            continue
        for k in range(elements.shape[1]):
            n = int(elements[ei, k])
            if n in duplicate_id:
                new_elements[ei, k] = duplicate_id[n]

    duplicates = nodes[interface_nodes].copy()
    new_nodes = np.vstack([nodes, duplicates])

    cohesives: List[CohesiveElement] = []
    for n0, n1 in edges:
        p0, p1 = nodes_2d[n0], nodes_2d[n1]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        tangent = edge / length
        normal = np.array([-tangent[1], tangent[0]])
        cohesives.append(
            CohesiveElement(
                nodes_top=(n0, n1),
                nodes_bottom=(duplicate_id[n0], duplicate_id[n1]),
                normal=normal,
                tangent=tangent,
                length=length,
            )
        )
    return new_nodes, new_elements, cohesives, duplicate_id, side


def insert_cohesive_layer(
    nodes: np.ndarray,
    elements: np.ndarray,
    interface_edges: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, List[CohesiveElement]]:
    """Double nodes along ``interface_edges`` and return a new (nodes, elements, cohesives).

    Parameters
    ----------
    nodes : (N, 2) float array — original node coordinates.
    elements : (E, 3) or (E, 4) int array — T3 triangle or Q4 connectivity.
    interface_edges : iterable of ``(n0, n1)`` pairs — interior edges marking
        the cohesive interface. Edge orientation is canonicalized by node id.

    Returns
    -------
    new_nodes : (N + n_iface, 2) array — original nodes followed by duplicates.
    new_elements : array — same shape as ``elements``; bottom-side elements
        have their interface-node references rewritten to the duplicate IDs.
    cohesives : list of :class:`CohesiveElement` records connecting top to
        bottom (one per interface edge).
    """
    new_nodes, new_elements, cohesives, _dup, _side = _insert_core(
        nodes, elements, interface_edges)
    return new_nodes, new_elements, cohesives


def insert_cohesive_layer_with_metadata(
    nodes: np.ndarray,
    elements: np.ndarray,
    interface_edges: Sequence[Tuple[int, int]],
    *,
    node_sets: Mapping[str, Sequence[int]] | None = None,
    element_sets: Mapping[str, Sequence[int]] | None = None,
    element_data: Mapping[str, Sequence] | None = None,
) -> CohesiveInsertionResult:
    """Insert a cohesive layer while preserving external-mesh metadata.

    This is the safer entry point for imported meshes. It preserves node sets,
    element sets, and per-element material/region arrays, and adds side-specific
    node sets for interface-bearing groups. The bulk element ordering is not
    changed, so material-region arrays remain index-aligned.
    """
    new_nodes, new_elements, cohesives, duplicate_id, side = _insert_core(
        nodes, elements, interface_edges)
    n_elem = int(np.asarray(elements).shape[0])

    out_node_sets: dict[str, np.ndarray] = {}
    if node_sets:
        for name, values in node_sets.items():
            arr = np.unique(np.asarray(values, dtype=int))
            if arr.size and (arr.min() < 0 or arr.max() >= np.asarray(nodes).shape[0]):
                raise ValueError(f"node set {name!r} contains out-of-range node ids")
            dup = np.asarray(
                [duplicate_id[int(n)] for n in arr if int(n) in duplicate_id],
                dtype=int,
            )
            out_node_sets[name] = np.unique(
                np.concatenate([arr, dup]) if dup.size else arr).astype(int)
            if dup.size:
                iface = np.asarray(
                    [int(n) for n in arr if int(n) in duplicate_id], dtype=int)
                top_name = f"{name}_top"
                bottom_name = f"{name}_bottom"
                if top_name in node_sets or bottom_name in node_sets:
                    top_name = f"cohesive_generated_{name}_top"
                    bottom_name = f"cohesive_generated_{name}_bottom"
                out_node_sets[top_name] = iface
                out_node_sets[bottom_name] = dup

    if duplicate_id:
        top = np.asarray(sorted(duplicate_id), dtype=int)
        bottom = np.asarray([duplicate_id[int(n)] for n in top], dtype=int)
        out_node_sets.setdefault("cohesive_interface_top", top)
        out_node_sets.setdefault("cohesive_interface_bottom", bottom)
        out_node_sets.setdefault(
            "cohesive_interface",
            np.unique(np.concatenate([top, bottom])).astype(int))

    out_element_sets: dict[str, np.ndarray] = {}
    if element_sets:
        for name, values in element_sets.items():
            arr = np.unique(np.asarray(values, dtype=int))
            if arr.size and (arr.min() < 0 or arr.max() >= n_elem):
                raise ValueError(
                    f"element set {name!r} contains out-of-range element ids")
            out_element_sets[name] = arr

    out_element_data: dict[str, np.ndarray] = {}
    if element_data:
        for name, values in element_data.items():
            arr = np.asarray(values).copy()
            if arr.shape[0] != n_elem:
                raise ValueError(
                    f"element_data {name!r} must have first dimension {n_elem}, "
                    f"got {arr.shape[0]}")
            out_element_data[name] = arr

    return CohesiveInsertionResult(
        nodes=new_nodes,
        elements=new_elements,
        cohesives=cohesives,
        node_sets=out_node_sets,
        element_sets=out_element_sets,
        element_data=out_element_data,
        duplicate_node_map=duplicate_id,
        element_side=side.copy(),
    )


def _cell_set_indices(cell_sets_for_name, block_index: int,
                      block_size: int) -> np.ndarray:
    if block_index >= len(cell_sets_for_name):
        return np.empty(0, dtype=int)
    raw = np.asarray(cell_sets_for_name[block_index])
    if raw.size == 0:
        return np.empty(0, dtype=int)
    if raw.dtype == np.bool_:
        if raw.shape[0] != block_size:
            raise ValueError(
                "boolean cell-set mask length does not match cell block size")
        return np.nonzero(raw)[0].astype(int)
    return raw.astype(int).reshape(-1)


def _find_bulk_block(mesh, cell_block_index: int | None) -> int:
    bulk_indices = [
        i for i, block in enumerate(mesh.cells)
        if block.type in ("triangle", "triangle3", "quad", "quad4")
    ]
    if not bulk_indices:
        raise ValueError("meshio mesh contains no T3 triangle or Q4 quad cell block")
    if cell_block_index is not None:
        if cell_block_index not in bulk_indices:
            raise ValueError(
                f"cell_block_index={cell_block_index} is not a T3 triangle "
                "or Q4 quad cell block")
        return cell_block_index
    if len(bulk_indices) > 1:
        raise ValueError(
            "meshio mesh has multiple T3/Q4 bulk blocks; pass "
            "cell_block_index explicitly")
    return bulk_indices[0]


def _extract_interface_edges_from_meshio(mesh, interface_set: str) -> list[tuple[int, int]]:
    if not getattr(mesh, "cell_sets", None) or interface_set not in mesh.cell_sets:
        raise ValueError(f"meshio cell set {interface_set!r} not found")
    blocks_for_set = mesh.cell_sets[interface_set]
    edges: list[tuple[int, int]] = []
    for bi, block in enumerate(mesh.cells):
        if block.type not in ("line", "line2"):
            continue
        selected = _cell_set_indices(blocks_for_set, bi, len(block.data))
        for idx in selected:
            n0, n1 = block.data[int(idx)]
            edges.append((int(n0), int(n1)))
    if not edges:
        raise ValueError(
            f"meshio cell set {interface_set!r} contains no line edges")
    return edges


def _extend_point_arrays(point_arrays: Mapping[str, Sequence] | None,
                         duplicate_node_map: Mapping[int, int],
                         original_n_nodes: int) -> dict[str, np.ndarray]:
    if not point_arrays:
        return {}
    interface_nodes = np.asarray(sorted(duplicate_node_map), dtype=int)
    out: dict[str, np.ndarray] = {}
    for name, values in point_arrays.items():
        arr = np.asarray(values)
        if arr.shape[0] != original_n_nodes:
            raise ValueError(
                f"meshio point_data {name!r} must have first dimension "
                f"{original_n_nodes}, got {arr.shape[0]}")
        if arr.shape[0] == 0 or interface_nodes.size == 0:
            out[name] = arr.copy()
        else:
            out[name] = np.concatenate([arr, arr[interface_nodes]], axis=0)
    return out


def insert_cohesive_layer_meshio(
    mesh,
    *,
    interface_edges: Sequence[Tuple[int, int]] | None = None,
    interface_set: str | None = None,
    triangle_block_index: int | None = None,
    cell_block_index: int | None = None,
) -> MeshIOCohesiveInsertionResult:
    """Insert a cohesive layer in a ``meshio.Mesh`` while preserving metadata.

    Parameters
    ----------
    mesh : meshio.Mesh
        Imported mesh with one T3 triangle or Q4 quad block unless
        ``cell_block_index`` is supplied. Point sets, point data, selected
        bulk cell sets, and selected bulk cell data are propagated to the
        doubled-node mesh.
    interface_edges : sequence of node pairs, optional
        Explicit interface edges. Mutually exclusive with ``interface_set``.
    interface_set : str, optional
        Name of a meshio line-cell set whose selected line cells mark the
        cohesive interface.
    triangle_block_index : int, optional
        Backward-compatible alias for ``cell_block_index``.
    cell_block_index : int, optional
        Required when the mesh has multiple T3/Q4 bulk blocks.
    """
    if triangle_block_index is not None and cell_block_index is not None:
        if int(triangle_block_index) != int(cell_block_index):
            raise ValueError(
                "triangle_block_index and cell_block_index disagree; pass "
                "only cell_block_index for new code.")
    if cell_block_index is None:
        cell_block_index = triangle_block_index

    if interface_edges is not None and interface_set is not None:
        raise ValueError("Pass either interface_edges or interface_set, not both")
    if interface_edges is None:
        if interface_set is None:
            raise ValueError("Either interface_edges or interface_set is required")
        edges = _canonical_edges(
            _extract_interface_edges_from_meshio(mesh, interface_set))
    else:
        edges = _canonical_edges(interface_edges)

    tri_idx = _find_bulk_block(mesh, cell_block_index)
    tri_block = mesh.cells[tri_idx]
    element_sets: dict[str, np.ndarray] = {}
    for name, blocks in (mesh.cell_sets or {}).items():
        indices = _cell_set_indices(blocks, tri_idx, len(tri_block.data))
        if indices.size:
            element_sets[name] = indices
    element_data: dict[str, np.ndarray] = {}
    for name, blocks in (mesh.cell_data or {}).items():
        if tri_idx >= len(blocks):
            raise ValueError(
                f"meshio cell_data {name!r} has no block for bulk cell block "
                f"{tri_idx}")
        arr = np.asarray(blocks[tri_idx])
        if arr.shape[0] != len(tri_block.data):
            raise ValueError(
                f"meshio cell_data {name!r} bulk block must have first "
                f"dimension {len(tri_block.data)}, got {arr.shape[0]}")
        element_data[name] = arr.copy()
    insertion = insert_cohesive_layer_with_metadata(
        np.asarray(mesh.points),
        np.asarray(tri_block.data),
        edges,
        node_sets=getattr(mesh, "point_sets", None),
        element_sets=element_sets,
        element_data=element_data,
    )

    cells = []
    for bi, block in enumerate(mesh.cells):
        data = insertion.elements if bi == tri_idx else np.asarray(block.data).copy()
        cells.append((block.type, data))

    point_data = _extend_point_arrays(
        getattr(mesh, "point_data", None),
        insertion.duplicate_node_map,
        int(np.asarray(mesh.points).shape[0]))
    cell_data = {
        name: [
            insertion.element_data[name].copy()
            if bi == tri_idx and name in insertion.element_data
            else np.asarray(values).copy()
            for bi, values in enumerate(blocks)
        ]
        for name, blocks in (mesh.cell_data or {}).items()
    }
    cell_sets = {
        name: [np.asarray(values).copy() for values in blocks]
        for name, blocks in (mesh.cell_sets or {}).items()
    }

    try:
        import meshio
    except ImportError as exc:
        raise ImportError(
            "insert_cohesive_layer_meshio requires meshio") from exc

    new_mesh = meshio.Mesh(
        points=insertion.nodes,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=getattr(mesh, "field_data", None),
        point_sets=insertion.node_sets,
        cell_sets=cell_sets,
        gmsh_periodic=getattr(mesh, "gmsh_periodic", None),
        info=getattr(mesh, "info", None),
    )
    return MeshIOCohesiveInsertionResult(
        mesh=new_mesh,
        insertion=insertion,
        triangle_block_index=tri_idx,
        interface_edges=edges,
        cell_type=tri_block.type,
    )
