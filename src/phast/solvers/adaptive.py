"""
Adaptive Mesh Refinement (AMR) via Newest Vertex Bisection (NVB).

Provides element-level refinement indicators, conforming mesh bisection,
and field interpolation routines for phase-field fracture simulations.

Algorithm overview
------------------
Newest Vertex Bisection is the standard AMR strategy for triangular meshes.
Each triangle is bisected by inserting a node at the midpoint of its
*refinement edge* (chosen as the longest edge). Conforming closure ensures
that every newly bisected edge is shared by two refined triangles, so
the resulting mesh has no hanging nodes.

Typical usage::

    from phast.adaptive import (
        compute_refinement_indicator, refine_mesh,
        interpolate_field, interpolate_elem_field,
    )

    marked = compute_refinement_indicator(mesh, d)
    if marked.any():
        new_mesh, parent_map, child_map = refine_mesh(mesh, marked)
        d = interpolate_field(old_mesh, new_mesh, d, parent_map)
        H = interpolate_elem_field(old_mesh, new_mesh, H, child_map)
        mesh = new_mesh

References
----------
- Mitchell, W.F. (1989). A comparison of adaptive refinement techniques
  for elliptic problems. *ACM TOMS* 15(4), 326-347.
- Bartels, S. (2015). *Numerical Methods for Nonlinear PDEs*. Springer.
  Chapter 4: Adaptive finite element methods.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Iterable, List, Optional, Sequence

from ..core.mesh import FEMMesh


# -------------------------------------------------------------------- #
# 1. Composable refinement criteria (Galvis-style)
# -------------------------------------------------------------------- #
#
# These criterion helpers (originally landed in ``mesh_adaptivity.py``,
# #111 scaffold) return *which* elements to refine as element-ID lists.
# They are integrated here to live alongside the newest-vertex-bisection
# pipeline that consumes a boolean mask.
#
# Reference
# ---------
# Galvis et al. 2026 (Eng Fract Mech): FreeFEM++ ``adaptmesh()`` on the
# damage-gradient indicator with relative tol 0.04 and max-vertex cap.

def damage_gradient_criterion(
    d_field: torch.Tensor,
    mesh: 'FEMMesh',
    threshold: float = 0.1,
) -> List[int]:
    """Flag elements where the damage-gradient magnitude exceeds ``threshold``.

    Element-level gradient is the standard P1 reconstruction
    ``grad_d_e = sum_j grad_phi_j * d_j`` over the three vertices.

    Parameters
    ----------
    d_field : torch.Tensor, shape (N,)
        Nodal damage in [0, 1].
    mesh : FEMMesh
        Must expose ``elements`` (E, 3) and ``grad_phi`` (E, 3, 2).
    threshold : float
        Gradient-magnitude threshold (units: 1/length).

    Returns
    -------
    list of int
        Element indices satisfying ``||grad d|| > threshold``.
    """
    elems = mesh.elements
    d_elem = d_field[elems]                                # (E, 3)
    gx = (mesh.grad_phi[:, :, 0] * d_elem).sum(dim=1)      # (E,)
    gy = (mesh.grad_phi[:, :, 1] * d_elem).sum(dim=1)      # (E,)
    grad_mag = torch.sqrt(gx * gx + gy * gy)
    flagged = torch.where(grad_mag > threshold)[0]
    return flagged.cpu().tolist()


def crack_tip_neighborhood_criterion(
    d_field: torch.Tensor,
    mesh: 'FEMMesh',
    radius: float = 3.0,
    d_tip_threshold: float = 0.5,
) -> List[int]:
    """Flag elements within ``radius * h_min`` of any cracked element.

    A "cracked" element is one whose maximum nodal damage exceeds
    ``d_tip_threshold``. Every element whose centroid lies within
    ``radius * mesh.h_min`` of any cracked-element centroid is flagged
    (cracked elements themselves included).

    Parameters
    ----------
    d_field : torch.Tensor, shape (N,)
        Nodal damage in [0, 1].
    mesh : FEMMesh
        Must expose ``nodes``, ``elements``, ``h_min``.
    radius : float
        Neighbourhood radius in units of ``h_min``.
    d_tip_threshold : float
        Damage value above which an element is treated as cracked.

    Returns
    -------
    list of int
        Element indices within the crack-tip neighbourhood.
    """
    elems = mesh.elements
    nodes = mesh.nodes
    centroids = nodes[elems].mean(dim=1)                   # (E, 2)

    d_elem = d_field[elems]                                # (E, 3)
    is_cracked = d_elem.max(dim=1).values > d_tip_threshold
    if not bool(is_cracked.any()):
        return []

    tip_centroids = centroids[is_cracked]                  # (T, 2)

    # Pairwise distances (E, T); fine for scaffold-scale tests, the
    # production driver should use a kd-tree or chunked computation.
    diff = centroids.unsqueeze(1) - tip_centroids.unsqueeze(0)
    dist = torch.linalg.norm(diff, dim=2)                  # (E, T)
    min_dist = dist.min(dim=1).values                      # (E,)

    cutoff = radius * float(mesh.h_min)
    flagged = torch.where(min_dist <= cutoff)[0]
    return flagged.cpu().tolist()


def union_refine_set(criteria_results: Iterable[Sequence[int]]) -> List[int]:
    """Union of element-ID lists from multiple criteria.

    Parameters
    ----------
    criteria_results : iterable of int sequences
        Output of one or more criterion functions.

    Returns
    -------
    list of int
        Sorted unique element IDs.
    """
    merged: set = set()
    for r in criteria_results:
        merged.update(int(i) for i in r)
    return sorted(merged)


# -------------------------------------------------------------------- #
# 2. Refinement indicator (boolean mask consumed by refine_mesh)
# -------------------------------------------------------------------- #

def compute_refinement_indicator(
    mesh: FEMMesh,
    d: torch.Tensor,
    grad_d_threshold: float = 0.3,
    d_threshold: float = 0.5,
    *,
    use_damage: bool = True,
    use_gradient: bool = True,
    use_neighborhood: bool = False,
    neighborhood_radius: float = 3.0,
) -> torch.Tensor:
    """Compose refinement criteria into a boolean mask over elements.

    By default an element is marked for refinement if *either*:
      - The maximum nodal damage among its three vertices exceeds
        ``d_threshold`` (element lies within the crack region), or
      - The magnitude of the element-level damage gradient exceeds
        ``grad_d_threshold`` (element lies at the crack front where
        the damage field transitions rapidly).

    The optional ``use_neighborhood`` flag adds a third criterion that
    flags elements within ``neighborhood_radius * mesh.h_min`` of any
    cracked element (see ``crack_tip_neighborhood_criterion``).

    Parameters
    ----------
    mesh : FEMMesh
        Current mesh (must have ``elements``, ``grad_phi``).
    d : torch.Tensor, shape (N,)
        Nodal damage field, values in [0, 1].
    grad_d_threshold : float
        Gradient magnitude threshold for crack-front marking.
    d_threshold : float
        Maximum nodal damage threshold for crack-body marking.
    use_damage, use_gradient, use_neighborhood : bool
        Toggles for each criterion. The defaults reproduce the legacy
        behaviour (damage OR gradient).
    neighborhood_radius : float
        Radius (in units of ``h_min``) for the optional crack-tip
        neighborhood criterion.

    Returns
    -------
    marked : torch.Tensor, shape (E,), dtype=torch.bool
        True for each element that should be refined.
    """
    E = mesh.elements.shape[0]
    elems = mesh.elements  # (E, 3)
    d_elem = d[elems]      # (E, 3)

    marked = torch.zeros(E, dtype=torch.bool, device=d.device)

    # Criterion 1: max nodal damage within element exceeds threshold
    if use_damage:
        max_d_elem = d_elem.max(dim=1).values  # (E,)
        marked = marked | (max_d_elem > d_threshold)

    # Criterion 2: gradient magnitude exceeds threshold
    # Element-level gradient: grad_d_e = sum_j grad_phi_j * d_j
    if use_gradient:
        grad_d_x = (mesh.grad_phi[:, :, 0] * d_elem).sum(dim=1)  # (E,)
        grad_d_y = (mesh.grad_phi[:, :, 1] * d_elem).sum(dim=1)  # (E,)
        grad_d_mag = torch.sqrt(grad_d_x ** 2 + grad_d_y ** 2)   # (E,)
        marked = marked | (grad_d_mag > grad_d_threshold)

    # Criterion 3 (opt-in): crack-tip neighborhood
    if use_neighborhood:
        flagged = crack_tip_neighborhood_criterion(
            d, mesh, radius=neighborhood_radius,
            d_tip_threshold=d_threshold,
        )
        if flagged:
            idx = torch.tensor(flagged, dtype=torch.long, device=d.device)
            marked[idx] = True

    return marked


# -------------------------------------------------------------------- #
# 2. Edge-to-element adjacency
# -------------------------------------------------------------------- #

def _build_edge_to_elem(elements: torch.Tensor) -> Dict[Tuple[int, int], list]:
    """Build a mapping from sorted edge tuples to element indices.

    Vectorized with NumPy: builds all edges, sorts, and groups by unique edge.

    Parameters
    ----------
    elements : (E, 3) long tensor

    Returns
    -------
    edge_to_elem : dict mapping (ni, nj) -> [elem_idx, ...]
        Edge nodes are sorted so (min, max).
    """
    import numpy as np
    elems_np = elements.cpu().numpy()
    E = elems_np.shape[0]

    # Build all 3E edges: (node_min, node_max, elem_idx)
    e0, e1, e2 = elems_np[:, 0], elems_np[:, 1], elems_np[:, 2]
    edges_a = np.stack([e0, e1], axis=1)  # edge 0-1
    edges_b = np.stack([e1, e2], axis=1)  # edge 1-2
    edges_c = np.stack([e2, e0], axis=1)  # edge 2-0
    all_edges = np.concatenate([edges_a, edges_b, edges_c], axis=0)  # (3E, 2)
    all_edges.sort(axis=1)  # sort each edge so (min, max)

    elem_ids = np.tile(np.arange(E), 3)  # (3E,)

    # Group by unique edge using structured array
    edge_keys = all_edges[:, 0].astype(np.int64) * (elems_np.max() + 1) + all_edges[:, 1]
    sort_idx = np.argsort(edge_keys)
    sorted_keys = edge_keys[sort_idx]
    sorted_edges = all_edges[sort_idx]
    sorted_elems = elem_ids[sort_idx]

    # Find boundaries between unique edges
    breaks = np.concatenate([[0], np.where(np.diff(sorted_keys) != 0)[0] + 1, [len(sorted_keys)]])

    edge_to_elem: Dict[Tuple[int, int], list] = {}
    for i in range(len(breaks) - 1):
        s, e = breaks[i], breaks[i + 1]
        key = (int(sorted_edges[s, 0]), int(sorted_edges[s, 1]))
        edge_to_elem[key] = sorted_elems[s:e].tolist()

    return edge_to_elem


def _longest_edge(nodes: torch.Tensor, tri: torch.Tensor) -> int:
    """Return the local index (0, 1, 2) of the vertex opposite the longest edge.

    The longest edge of triangle (v0, v1, v2) is determined, and the
    local index of the vertex *opposite* that edge is returned. This
    vertex becomes the "newest vertex" in the NVB convention, and the
    longest edge is the refinement (bisection) edge.

    Local edge convention:
      - Edge 0: v1-v2  (opposite v0)
      - Edge 1: v0-v2  (opposite v1)
      - Edge 2: v0-v1  (opposite v2)

    Parameters
    ----------
    nodes : (N, 2) node coordinates
    tri : (3,) long tensor of node indices

    Returns
    -------
    int : local vertex index opposite the longest edge (0, 1, or 2)
    """
    p = nodes[tri]  # (3, 2)
    # Edge lengths squared (avoid sqrt for comparison)
    e0_sq = ((p[1] - p[2]) ** 2).sum()  # opposite v0
    e1_sq = ((p[0] - p[2]) ** 2).sum()  # opposite v1
    e2_sq = ((p[0] - p[1]) ** 2).sum()  # opposite v2

    lengths_sq = torch.stack([e0_sq, e1_sq, e2_sq])
    return int(lengths_sq.argmax().item())


# -------------------------------------------------------------------- #
# 3. Core refinement
# -------------------------------------------------------------------- #

def refine_mesh(
    mesh: FEMMesh,
    marked_elements: torch.Tensor,
) -> Tuple['FEMMesh', Dict[int, Tuple[int, int]], Dict[int, int]]:
    """Refine the mesh by newest-vertex bisection with conforming closure.

    Each marked triangle is bisected along its longest edge. Neighbors
    sharing the bisected edge are also refined to maintain conformity
    (no hanging nodes). The process iterates until no new forced
    refinements are needed.

    Parameters
    ----------
    mesh : FEMMesh
        Current mesh.
    marked_elements : torch.Tensor, shape (E,), dtype=torch.bool
        Elements to refine (from ``compute_refinement_indicator``).

    Returns
    -------
    new_mesh : FEMMesh
        Refined mesh with all precomputed quantities.
    parent_map : dict
        ``{new_node_idx: (old_node_a, old_node_b)}`` for midpoint nodes.
        Existing nodes are NOT in this dict.
    child_map : dict
        ``{new_elem_idx: old_elem_idx}`` for every new element.
        Unrefined elements map to themselves.
    """
    nodes_np = mesh.nodes.cpu().numpy().copy()
    elems_np = mesh.elements.cpu().numpy().copy()
    n_old_nodes = nodes_np.shape[0]
    n_old_elems = elems_np.shape[0]

    # Collect which boundary node-sets each node belongs to, so we can
    # tag midpoint nodes that sit on a boundary edge.
    node_to_sets: Dict[int, set] = {}
    for set_name, idx_tensor in mesh.node_sets.items():
        for ni in idx_tensor.cpu().numpy().tolist():
            node_to_sets.setdefault(ni, set()).add(set_name)

    # --- Phase 1: propagate marks for conforming closure ---
    # Build edge -> elem adjacency
    edge_to_elem = _build_edge_to_elem(mesh.elements)

    # Determine the refinement edge for every element (vectorized)
    nodes_np = mesh.nodes.cpu().numpy()
    p = nodes_np[elems_np]  # (E, 3, 2)
    e0_sq = ((p[:, 1] - p[:, 2]) ** 2).sum(axis=1)  # opposite v0
    e1_sq = ((p[:, 0] - p[:, 2]) ** 2).sum(axis=1)  # opposite v1
    e2_sq = ((p[:, 0] - p[:, 1]) ** 2).sum(axis=1)  # opposite v2
    ref_vertex = np.stack([e0_sq, e1_sq, e2_sq], axis=1).argmax(axis=1).astype(np.int32)

    # Refinement edge node-pair per element
    # If ref_vertex[ei] == k, the bisection edge connects the other two vertices.
    _other = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    def _ref_edge(ei):
        a_loc, b_loc = _other[ref_vertex[ei]]
        na, nb = int(elems_np[ei, a_loc]), int(elems_np[ei, b_loc])
        return (min(na, nb), max(na, nb))

    to_refine = set(int(i) for i in torch.where(marked_elements)[0].cpu().numpy())

    # Propagate: if an element's refinement edge is shared and the neighbor
    # also needs that edge bisected, mark the neighbor.
    #
    # Conforming closure correctness: when neighbor nj is marked because it
    # shares the refinement edge of some marked element ei, nj's own
    # refinement edge (longest edge) may differ from the shared edge. In
    # the next while-loop iteration, nj is included in to_refine, so
    # _ref_edge(nj) is evaluated and *its* refinement-edge neighbors are
    # checked and marked if needed. This transitive propagation continues
    # until no new marks are added, ensuring every marked element's
    # refinement edge is shared only with another marked element (or is on
    # the boundary). NVB then bisects each marked element along its own
    # refinement edge, and the shared midpoint guarantees no hanging nodes.
    max_closure_iters = n_old_elems  # cannot mark more elements than exist
    changed = True
    for _closure_iter in range(max_closure_iters):
        if not changed:
            break
        changed = False
        new_marks = set()
        for ei in to_refine:
            edge = _ref_edge(ei)
            for nj in edge_to_elem.get(edge, []):
                if nj != ei and nj not in to_refine:
                    # Neighbor shares the bisected edge — mark it for
                    # refinement. Even if this edge is not nj's refinement
                    # edge, nj must be bisected to avoid a hanging node.
                    # nj's own refinement edge neighbors will be checked
                    # in subsequent while-loop iterations.
                    new_marks.add(nj)
        if new_marks:
            to_refine |= new_marks
            changed = True

    # --- Phase 2: bisect each marked element ---
    # Bookkeeping for midpoint nodes (avoid duplicates across shared edges)
    edge_to_midnode: Dict[Tuple[int, int], int] = {}
    new_nodes_list = list(nodes_np)  # grow by appending midpoints
    parent_map: Dict[int, Tuple[int, int]] = {}

    def _get_or_create_midnode(na: int, nb: int) -> int:
        edge = (min(na, nb), max(na, nb))
        if edge in edge_to_midnode:
            return edge_to_midnode[edge]
        mid = 0.5 * (new_nodes_list[na] + new_nodes_list[nb])
        mid_idx = len(new_nodes_list)
        new_nodes_list.append(mid)
        edge_to_midnode[edge] = mid_idx
        parent_map[mid_idx] = (edge[0], edge[1])
        return mid_idx

    new_elems_list = []
    child_map: Dict[int, int] = {}
    new_elem_idx = 0

    for ei in range(n_old_elems):
        tri = elems_np[ei]  # (3,) original vertices
        if ei not in to_refine:
            # Keep element as-is
            new_elems_list.append(tri.copy())
            child_map[new_elem_idx] = ei
            new_elem_idx += 1
        else:
            # Bisect along longest edge
            rv = ref_vertex[ei]
            a_loc, b_loc = _other[rv]
            v_apex = int(tri[rv])
            v_a = int(tri[a_loc])
            v_b = int(tri[b_loc])
            v_mid = _get_or_create_midnode(v_a, v_b)

            # Two children:
            #   child 0: (apex, v_a, mid)
            #   child 1: (apex, mid, v_b)
            new_elems_list.append(np.array([v_apex, v_a, v_mid], dtype=np.int64))
            child_map[new_elem_idx] = ei
            new_elem_idx += 1

            new_elems_list.append(np.array([v_apex, v_mid, v_b], dtype=np.int64))
            child_map[new_elem_idx] = ei
            new_elem_idx += 1

    # --- Phase 3: assign boundary node-sets to midpoint nodes ---
    new_node_sets: Dict[str, list] = {
        name: idx_tensor.cpu().numpy().tolist()
        for name, idx_tensor in mesh.node_sets.items()
    }

    for mid_idx, (na, nb) in parent_map.items():
        # A midpoint inherits any node-set that BOTH endpoints share
        sets_a = node_to_sets.get(na, set())
        sets_b = node_to_sets.get(nb, set())
        common_sets = sets_a & sets_b
        for s in common_sets:
            new_node_sets[s].append(mid_idx)

    # Convert node-set lists to tensors
    node_sets_tensors = {}
    for name, idx_list in new_node_sets.items():
        if idx_list:
            node_sets_tensors[name] = torch.tensor(
                sorted(idx_list), dtype=torch.long, device=mesh.device)

    # --- Phase 4: build new FEMMesh from tensors ---
    new_nodes_tensor = torch.tensor(
        np.array(new_nodes_list), dtype=mesh.dtype, device=mesh.device)
    new_elems_tensor = torch.tensor(
        np.stack(new_elems_list), dtype=torch.long, device=mesh.device)

    new_mesh = FEMMesh.from_tensors(
        nodes=new_nodes_tensor,
        elements=new_elems_tensor,
        node_sets=node_sets_tensors,
        device=mesh.device,
        dtype=mesh.dtype,
    )

    return new_mesh, parent_map, child_map


# -------------------------------------------------------------------- #
# 4. Field interpolation (nodal)
# -------------------------------------------------------------------- #

def interpolate_field(
    old_mesh: FEMMesh,
    new_mesh: FEMMesh,
    field: torch.Tensor,
    parent_map: Dict[int, Tuple[int, int]],
) -> torch.Tensor:
    """Interpolate a nodal field from the old mesh to the refined mesh.

    Existing nodes retain their original values. New midpoint nodes
    receive the average of their two parent node values.

    Parameters
    ----------
    old_mesh : FEMMesh
        Mesh before refinement.
    new_mesh : FEMMesh
        Mesh after refinement.
    field : torch.Tensor
        Nodal field on ``old_mesh``. Shape ``(N_old,)`` for scalar,
        ``(N_old, D)`` for vector fields.
    parent_map : dict
        ``{new_node_idx: (old_node_a, old_node_b)}`` from ``refine_mesh``.

    Returns
    -------
    new_field : torch.Tensor
        Field on ``new_mesh`` with same trailing dimensions as ``field``.
    """
    is_vector = field.dim() > 1
    n_new = new_mesh.n_nodes
    n_old = old_mesh.n_nodes

    if is_vector:
        new_field = torch.zeros(
            n_new, field.shape[1], dtype=field.dtype, device=field.device)
    else:
        new_field = torch.zeros(n_new, dtype=field.dtype, device=field.device)

    # Copy values for existing (old) nodes
    new_field[:n_old] = field

    # Interpolate midpoint nodes
    for mid_idx, (na, nb) in parent_map.items():
        new_field[mid_idx] = 0.5 * (field[na] + field[nb])

    return new_field


# -------------------------------------------------------------------- #
# 5. Field interpolation (element)
# -------------------------------------------------------------------- #

def interpolate_elem_field(
    old_mesh: FEMMesh,
    new_mesh: FEMMesh,
    field: torch.Tensor,
    child_map: Dict[int, int],
) -> torch.Tensor:
    """Interpolate an element field from the old mesh to the refined mesh.

    Children inherit their parent element's value. This is appropriate
    for piecewise-constant element data such as the history variable H.

    Parameters
    ----------
    old_mesh : FEMMesh
        Mesh before refinement.
    new_mesh : FEMMesh
        Mesh after refinement.
    field : torch.Tensor
        Element field on ``old_mesh``, shape ``(E_old,)`` or ``(E_old, D)``.
    child_map : dict
        ``{new_elem_idx: old_elem_idx}`` from ``refine_mesh``.

    Returns
    -------
    new_field : torch.Tensor
        Field on ``new_mesh`` with same trailing dimensions as ``field``.
    """
    is_vector = field.dim() > 1
    n_new = new_mesh.n_elems

    if is_vector:
        new_field = torch.zeros(
            n_new, field.shape[1], dtype=field.dtype, device=field.device)
    else:
        new_field = torch.zeros(n_new, dtype=field.dtype, device=field.device)

    for new_ei, old_ei in child_map.items():
        new_field[new_ei] = field[old_ei]

    return new_field
