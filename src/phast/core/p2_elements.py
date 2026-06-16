"""P2 (T6) quadratic triangle element primitives.

Standalone module scaffolding higher-order element support for issue #112.
Provides shape functions, Gauss quadrature, and single-element stiffness
assembly for biquadratic Lagrange triangles on the reference triangle with
vertices (0,0), (1,0), (0,1).

Node ordering (standard gmsh order=2):
    0: (0, 0)         -- vertex
    1: (1, 0)         -- vertex
    2: (0, 1)         -- vertex
    3: (1/2, 0)       -- midpoint of edge 0-1
    4: (1/2, 1/2)     -- midpoint of edge 1-2
    5: (0, 1/2)       -- midpoint of edge 2-0

NOT integrated into ``fem_operators.py`` / ``mechanics_solver.py`` /
``damage_solver.py`` -- this module pins the API only.
"""
from __future__ import annotations

from typing import Tuple

import torch

_DTYPE = torch.float64


# Reference-triangle node coordinates (6, 2) in (xi, eta).
P2_REF_NODES = torch.tensor(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
    ],
    dtype=_DTYPE,
)


def p2_shape_functions(xi: float, eta: float) -> torch.Tensor:
    """Biquadratic Lagrange shape functions; returns shape (6,)."""
    l1, l2, l3 = 1.0 - xi - eta, xi, eta
    return torch.tensor(
        [l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1),
         4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1],
        dtype=_DTYPE,
    )


def p2_shape_function_derivs(xi: float, eta: float) -> torch.Tensor:
    """Shape-function derivatives wrt (xi, eta); returns shape (6, 2)."""
    l1 = 1.0 - xi - eta
    dN_dxi = torch.tensor(
        [-(4 * l1 - 1), (4 * xi - 1), 0.0,
         4 * (l1 - xi), 4 * eta, -4 * eta],
        dtype=_DTYPE,
    )
    dN_deta = torch.tensor(
        [-(4 * l1 - 1), 0.0, (4 * eta - 1),
         -4 * xi, 4 * xi, 4 * (l1 - eta)],
        dtype=_DTYPE,
    )
    return torch.stack([dN_dxi, dN_deta], dim=1)


def p2_gauss_points(order: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gauss quadrature on the reference triangle.

    Parameters
    ----------
    order : int
        ``order=2`` returns the 3-point rule (exact for degree-2 polys).
        ``order=3`` returns the 6-point rule (exact for degree-4 polys).

    Returns
    -------
    points : torch.Tensor, shape (n, 2)
    weights : torch.Tensor, shape (n,)  -- sum to 1/2 (reference-triangle area).
    """
    if order == 2:
        pts = torch.tensor(
            [[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]], dtype=_DTYPE
        )
        wts = torch.tensor([1 / 6, 1 / 6, 1 / 6], dtype=_DTYPE)
        return pts, wts
    if order == 3:
        # Strang & Fix 6-point rule, exact for degree 4.
        a1, w1 = 0.445948490915965, 0.111690794839005
        a2, w2 = 0.091576213509771, 0.054975871827661
        pts = torch.tensor(
            [
                [a1, a1],
                [1 - 2 * a1, a1],
                [a1, 1 - 2 * a1],
                [a2, a2],
                [1 - 2 * a2, a2],
                [a2, 1 - 2 * a2],
            ],
            dtype=_DTYPE,
        )
        wts = torch.tensor([w1, w1, w1, w2, w2, w2], dtype=_DTYPE)
        return pts, wts
    raise ValueError(f"Unsupported order={order}; use 2 or 3.")


def _plane_strain_D(E: float, nu: float) -> torch.Tensor:
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * torch.tensor(
        [[1 - nu, nu, 0.0], [nu, 1 - nu, 0.0], [0.0, 0.0, 0.5 - nu]],
        dtype=_DTYPE,
    )


def _plane_stress_D(E: float, nu: float) -> torch.Tensor:
    c = E / (1 - nu * nu)
    return c * torch.tensor(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1 - nu)]],
        dtype=_DTYPE,
    )


def p2_element_stiffness(
    node_coords: torch.Tensor,
    E: float,
    nu: float,
    plane_strain: bool = True,
    order: int = 2,
) -> torch.Tensor:
    """Assemble single P2 element stiffness via B^T D B integration.

    Parameters
    ----------
    node_coords : torch.Tensor
        Shape (6, 2), physical (x, y) coordinates of the 6 P2 nodes.
    E, nu : float
        Young's modulus and Poisson's ratio.
    plane_strain : bool
        If False uses plane stress.
    order : int
        Gauss-rule order (default 2 = 3-point).

    Returns
    -------
    torch.Tensor
        Shape (12, 12); DOF order is [u0, v0, u1, v1, ..., u5, v5].
    """
    coords = node_coords.to(_DTYPE)
    if coords.shape != (6, 2):
        raise ValueError(f"node_coords must be (6, 2), got {tuple(coords.shape)}")
    D = _plane_strain_D(E, nu) if plane_strain else _plane_stress_D(E, nu)
    pts, wts = p2_gauss_points(order=order)
    K = torch.zeros((12, 12), dtype=_DTYPE)
    for q in range(pts.shape[0]):
        xi, eta = float(pts[q, 0]), float(pts[q, 1])
        dN_ref = p2_shape_function_derivs(xi, eta)         # (6, 2)
        J = dN_ref.T @ coords                              # (2, 2)
        detJ = torch.det(J)
        if detJ <= 0:
            raise ValueError("Non-positive Jacobian; check node ordering.")
        Jinv = torch.linalg.inv(J)
        # Chain rule: with J[a,b] = dx_b/dxi_a (= dN_ref.T @ coords),
        # Jinv[c,d] = dxi_d/dx_c, so dN/dx_j = dN/dxi_i * dxi_i/dx_j
        # = dN_ref[n,i] * Jinv[j,i] = (dN_ref @ Jinv.T)[n,j].
        dN_xy = dN_ref @ Jinv.T                            # (6, 2)
        B = torch.zeros((3, 12), dtype=_DTYPE)
        for i in range(6):
            B[0, 2 * i] = dN_xy[i, 0]
            B[1, 2 * i + 1] = dN_xy[i, 1]
            B[2, 2 * i] = dN_xy[i, 1]
            B[2, 2 * i + 1] = dN_xy[i, 0]
        K = K + (B.T @ D @ B) * (detJ * wts[q])
    return K


def _as_long_cpu(elements) -> torch.Tensor:
    elems = elements if torch.is_tensor(elements) else torch.as_tensor(elements)
    if elems.ndim != 2 or elems.shape[1] != 3:
        raise ValueError(f"P1 elements must have shape (n_elements, 3), got {tuple(elems.shape)}")
    return elems.detach().cpu().to(dtype=torch.long)


def _p2_connectivity_from_p1(elements, n_vertices: int) -> torch.Tensor:
    elems = _as_long_cpu(elements)
    edge_to_midpoint: dict[tuple[int, int], int] = {}
    rows: list[list[int]] = []
    next_node = int(n_vertices)

    for tri in elems.tolist():
        v0, v1, v2 = (int(tri[0]), int(tri[1]), int(tri[2]))
        mids: list[int] = []
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            edge = (a, b) if a < b else (b, a)
            mid = edge_to_midpoint.get(edge)
            if mid is None:
                mid = next_node
                edge_to_midpoint[edge] = mid
                next_node += 1
            mids.append(mid)
        rows.append([v0, v1, v2, mids[0], mids[1], mids[2]])

    return torch.tensor(rows, dtype=torch.long)


def p2_node_indices_from_p1_mesh(p1_mesh) -> torch.Tensor:
    """Return T6 connectivity for a P1 triangle mesh.

    The first three columns are the original P1 vertex indices. Columns 3-5
    are unique midpoint-node indices for local edges (0-1), (1-2), and (2-0),
    matching the standard Gmsh order-2 triangle convention used by the shape
    functions above. Shared P1 edges receive the same midpoint index in both
    neighboring T6 elements.

    This helper returns connectivity only. Use :func:`p2_mesh_from_p1` when
    the corresponding midpoint coordinates are needed.
    """
    if not hasattr(p1_mesh, "elements"):
        raise AttributeError("p1_mesh must expose `.elements` (n_e, 3)")
    elems = _as_long_cpu(p1_mesh.elements)
    if hasattr(p1_mesh, "nodes"):
        n_vertices = int(p1_mesh.nodes.shape[0])
    else:
        n_vertices = int(elems.max().item()) + 1
    out = _p2_connectivity_from_p1(elems, n_vertices)
    return out.to(device=p1_mesh.elements.device if torch.is_tensor(p1_mesh.elements) else None)


def p2_mesh_from_p1(
    nodes: torch.Tensor,
    elements: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert P1 triangle coordinates/connectivity into P2 T6 data.

    Parameters
    ----------
    nodes
        P1 node coordinates, shape ``(n_nodes, 2)``.
    elements
        P1 triangle connectivity, shape ``(n_elements, 3)``.

    Returns
    -------
    p2_nodes, p2_elements
        ``p2_nodes`` contains the original vertices followed by one midpoint
        node per unique undirected P1 edge. ``p2_elements`` has columns
        ``[v0, v1, v2, m01, m12, m20]``.
    """
    if not torch.is_tensor(nodes):
        nodes = torch.as_tensor(nodes, dtype=_DTYPE)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes must have shape (n_nodes, 2), got {tuple(nodes.shape)}")

    device = nodes.device
    dtype = nodes.dtype
    elems_cpu = _as_long_cpu(elements)
    p2_elements_cpu = _p2_connectivity_from_p1(elems_cpu, int(nodes.shape[0]))

    nodes_t = nodes.to(dtype=dtype, device=device)
    if p2_elements_cpu.numel() == 0:
        return nodes_t.clone(), p2_elements_cpu.to(device=device)

    n_midpoints = int(p2_elements_cpu[:, 3:].max().item()) - int(nodes.shape[0]) + 1
    if n_midpoints > 0:
        elems_dev = elems_cpu.to(device=device)
        edge_nodes = torch.stack(
            (
                elems_dev[:, [0, 1]],
                elems_dev[:, [1, 2]],
                elems_dev[:, [2, 0]],
            ),
            dim=1,
        ).reshape(-1, 2)
        midpoint_ids = (p2_elements_cpu[:, 3:].reshape(-1) - int(nodes.shape[0])).to(
            device=device)
        mids = torch.empty(n_midpoints, 2, dtype=dtype, device=device)
        mids[midpoint_ids] = 0.5 * (
            nodes_t[edge_nodes[:, 0]] + nodes_t[edge_nodes[:, 1]])
        p2_nodes = torch.cat([nodes_t, mids], dim=0)
    else:
        p2_nodes = nodes_t.clone()

    return p2_nodes, p2_elements_cpu.to(device=device)
