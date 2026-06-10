"""Initial-condition resolvers for damage preseeding.

Generalises ``initial_conditions.preseed_*`` to accept geometric regions
and named node-sets. Each spec resolves to a node mask plus a target
damage value per node; the run-loop converts these to a one-ring element
mask and seeds the history variable ``H_elem`` so the first damage solve
yields ``d ~= value`` on those nodes (Borden 2012 / Bleyer 2017
convention: pre-existing notches are real broken material, not just a
geometric slit).

Supported region types
----------------------
- ``line_segment`` — points within ``thickness`` of segment ``[from -> to]``.
  ``thickness`` defaults to the mean mesh edge length.
- ``rectangle`` — axis-aligned, ``origin`` is the lower-left corner,
  ``size`` is ``[width, height]``.
- ``circle`` — points within ``radius`` of ``center``.
- ``polygon`` — points inside the polygon defined by ``vertices``
  (closed ring, ray-cast point-in-polygon).
- ``nodes`` — passthrough of a named mesh node-set.

Each spec is a dict with either ``region: {type: ..., ...}`` or
``nodes: <name>``, plus a scalar ``value`` (default 1.0). When the same
node is matched by multiple specs, the resolver keeps the maximum
``value`` so later entries cannot lower an earlier preseed.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Geometry predicates (return a boolean node mask of shape (n_nodes,))
# ---------------------------------------------------------------------------

def _mask_line_segment(nodes: torch.Tensor, frm, to,
                       thickness: float) -> torch.Tensor:
    """Nodes within ``thickness`` of segment ``frm -> to``."""
    p0 = torch.as_tensor(frm, dtype=nodes.dtype, device=nodes.device)
    p1 = torch.as_tensor(to, dtype=nodes.dtype, device=nodes.device)
    seg = p1 - p0
    L2 = float((seg * seg).sum())
    if L2 == 0.0:
        # Degenerate segment: treat as a circle of radius=thickness.
        d = (nodes - p0).norm(dim=1)
        return d <= thickness
    rel = nodes - p0  # (N, 2)
    t = (rel @ seg) / L2
    t = t.clamp(0.0, 1.0)
    proj = p0 + t.unsqueeze(1) * seg
    d = (nodes - proj).norm(dim=1)
    return d <= thickness


def _mask_rectangle(nodes: torch.Tensor, origin, size) -> torch.Tensor:
    """Axis-aligned rectangle: origin is lower-left, size is [w, h]."""
    o = torch.as_tensor(origin, dtype=nodes.dtype, device=nodes.device)
    s = torch.as_tensor(size, dtype=nodes.dtype, device=nodes.device)
    upper = o + s
    return ((nodes[:, 0] >= o[0]) & (nodes[:, 0] <= upper[0]) &
            (nodes[:, 1] >= o[1]) & (nodes[:, 1] <= upper[1]))


def _mask_circle(nodes: torch.Tensor, center, radius: float) -> torch.Tensor:
    c = torch.as_tensor(center, dtype=nodes.dtype, device=nodes.device)
    return (nodes - c).norm(dim=1) <= float(radius)


def _mask_polygon(nodes: torch.Tensor, vertices) -> torch.Tensor:
    """Ray-cast point-in-polygon (inclusive of edges via tolerance).

    Vertices form a (possibly open) ring; the closing edge is implied.
    """
    V = torch.as_tensor(vertices, dtype=nodes.dtype, device=nodes.device)
    if V.shape[0] < 3:
        raise ValueError("polygon requires at least 3 vertices")
    n = V.shape[0]
    inside = torch.zeros(nodes.shape[0], dtype=torch.bool, device=nodes.device)
    x = nodes[:, 0]
    y = nodes[:, 1]
    j = n - 1
    for i in range(n):
        xi, yi = V[i, 0], V[i, 1]
        xj, yj = V[j, 0], V[j, 1]
        cond_y = ((yi > y) != (yj > y))
        # Avoid division by zero on horizontal edges; cond_y already excludes them.
        denom = (yj - yi)
        # Where cond_y is False, x_intersect is irrelevant; clamp denom safely.
        safe_denom = torch.where(denom == 0,
                                 torch.ones_like(denom),
                                 denom)
        x_intersect = (xj - xi) * (y - yi) / safe_denom + xi
        cross = cond_y & (x < x_intersect)
        inside ^= cross
        j = i
    return inside


# ---------------------------------------------------------------------------
# Spec resolution
# ---------------------------------------------------------------------------

def _mean_edge_length(mesh) -> float:
    """Mean edge length over the mesh (used as default segment thickness)."""
    nodes = mesh.nodes
    elems = mesh.elements
    # Triangle elements: edges are (0,1), (1,2), (2,0).
    e01 = (nodes[elems[:, 0]] - nodes[elems[:, 1]]).norm(dim=1)
    e12 = (nodes[elems[:, 1]] - nodes[elems[:, 2]]).norm(dim=1)
    e20 = (nodes[elems[:, 2]] - nodes[elems[:, 0]]).norm(dim=1)
    return float(torch.cat([e01, e12, e20]).mean())


def _resolve_one(mesh, spec: dict) -> Tuple[torch.Tensor, float]:
    """Resolve a single spec entry to (node_mask, value)."""
    if not isinstance(spec, dict):
        raise TypeError(f"preseed_damage entry must be a dict, got {type(spec)}")

    value = float(spec.get('value', 1.0))
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"preseed_damage value must be in [0, 1], got {value}")

    nodes = mesh.nodes
    n_nodes = mesh.n_nodes
    device = nodes.device

    if 'nodes' in spec:
        name = spec['nodes']
        if name not in mesh.node_sets:
            raise RuntimeError(
                f"preseed_damage: mesh has no node set '{name}'. "
                f"Available: {list(mesh.node_sets.keys())}")
        idx = mesh.node_sets[name]
        mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        mask[idx] = True
        return mask, value

    if 'region' not in spec:
        raise ValueError(
            f"preseed_damage entry must specify 'region' or 'nodes': {spec}")

    region = spec['region']
    if not isinstance(region, dict) or 'type' not in region:
        raise ValueError(f"preseed_damage region must be a dict with 'type': {region}")

    rtype = region['type']
    if rtype == 'line_segment':
        thickness = region.get('thickness', None)
        if thickness is None:
            thickness = _mean_edge_length(mesh)
        mask = _mask_line_segment(nodes, region['from'], region['to'],
                                  float(thickness))
    elif rtype == 'rectangle':
        mask = _mask_rectangle(nodes, region['origin'], region['size'])
    elif rtype == 'circle':
        mask = _mask_circle(nodes, region['center'], region['radius'])
    elif rtype == 'polygon':
        mask = _mask_polygon(nodes, region['vertices'])
    else:
        raise ValueError(
            f"Unknown preseed_damage region type '{rtype}'. "
            f"Supported: line_segment, rectangle, circle, polygon")

    return mask, value


def resolve_preseed_specs(mesh, specs: Iterable[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve a list of preseed specs to (node_mask, value_per_node).

    Returns
    -------
    node_mask : (n_nodes,) bool tensor
        True for any node touched by at least one spec.
    value_per_node : (n_nodes,) float tensor
        Per-node target damage value (max across overlapping specs);
        0 where node_mask is False.
    """
    n_nodes = mesh.n_nodes
    device = mesh.nodes.device
    dtype = mesh.nodes.dtype

    node_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    value_per_node = torch.zeros(n_nodes, dtype=dtype, device=device)

    for spec in specs:
        mask, value = _resolve_one(mesh, spec)
        node_mask |= mask
        # max so later, lower-value specs don't downgrade an earlier seed
        value_per_node = torch.where(
            mask,
            torch.maximum(value_per_node,
                          torch.full_like(value_per_node, value)),
            value_per_node)
    return node_mask, value_per_node


def normalise_legacy_preseed(notch_nodesets: List[str]) -> List[dict]:
    """Convert the legacy ``preseed_notch_nodesets: [a, b]`` form to the
    new spec list ``[{nodes: a, value: 1.0}, {nodes: b, value: 1.0}]``.
    """
    return [{'nodes': name, 'value': 1.0} for name in notch_nodesets]


def value_to_H_seed(value: float, Gc: float, l0: float,
                    saturation: float = 0.999) -> float:
    """Map a target damage value to an initial history variable H.

    AT2 zero-gradient equilibrium: d = 2H / (Gc/l0 + 2H), so
        H(d) = d * (Gc/l0) / (2 * (1 - d)).
    For d >= ``saturation`` (e.g. d=1) we return the legacy sentinel
    ``1e4 * Gc/l0`` which drives d to within 1e-4 of 1 in one solve and
    matches the pre-existing preseed_notch_nodesets behaviour.

    Note: in the staggered solver the Laplacian smoothing means partial
    values will diffuse with neighbours over the first damage step;
    value=1.0 is the well-tested case.
    """
    if value >= saturation:
        return 1.0e4 * Gc / l0
    if value <= 0.0:
        return 0.0
    return value * (Gc / l0) / (2.0 * (1.0 - value))
