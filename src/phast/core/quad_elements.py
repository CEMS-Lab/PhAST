"""Quadrilateral element primitives for issue #5.

Provides verified shape functions, derivatives, tensor-product Gauss rules,
and single-element elastic stiffness assembly for Q4, Q8, and Q9 elements on
the reference square ``[-1, 1] x [-1, 1]``.

This is intentionally a primitive layer, like ``p2_elements.py``. The
production phase-field solvers are still P1-triangle based until mechanics,
damage, mass, preconditioning, and IO dispatch are all lifted together.
"""
from __future__ import annotations

from typing import Literal, Tuple

import torch

_DTYPE = torch.float64
QuadKind = Literal["Q4", "Q8", "Q9"]


class StructuredQ4Mesh:
    """Structured Q4 mesh container with boundary node sets."""

    def __init__(self, nodes: torch.Tensor, quads: torch.Tensor,
                 node_sets: dict[str, torch.Tensor], nx: int, ny: int):
        self.nodes = nodes
        self.quads = quads
        self.node_sets = node_sets
        self.nx = int(nx)
        self.ny = int(ny)

Q4_REF_NODES = torch.tensor(
    [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
    dtype=_DTYPE,
)

Q8_REF_NODES = torch.tensor(
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ],
    dtype=_DTYPE,
)

Q9_REF_NODES = torch.tensor(
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, 0.0],
    ],
    dtype=_DTYPE,
)


def quad_ref_nodes(kind: QuadKind) -> torch.Tensor:
    """Return reference-node coordinates for ``kind``."""
    kind = kind.upper()
    if kind == "Q4":
        return Q4_REF_NODES.clone()
    if kind == "Q8":
        return Q8_REF_NODES.clone()
    if kind == "Q9":
        return Q9_REF_NODES.clone()
    raise ValueError(f"Unsupported quadrilateral kind {kind!r}")


def _line_lagrange_2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quadratic 1D Lagrange basis at nodes [-1, 0, 1]."""
    L0 = 0.5 * x * (x - 1.0)
    L1 = 1.0 - x * x
    L2 = 0.5 * x * (x + 1.0)
    dL0 = x - 0.5
    dL1 = -2.0 * x
    dL2 = x + 0.5
    return torch.stack([L0, L1, L2]), torch.stack([dL0, dL1, dL2])


def q4_shape_functions(xi: float, eta: float) -> torch.Tensor:
    """Bilinear Q4 shape functions in counter-clockwise corner order."""
    x = torch.tensor(float(xi), dtype=_DTYPE)
    y = torch.tensor(float(eta), dtype=_DTYPE)
    return 0.25 * torch.stack([
        (1.0 - x) * (1.0 - y),
        (1.0 + x) * (1.0 - y),
        (1.0 + x) * (1.0 + y),
        (1.0 - x) * (1.0 + y),
    ])


def q4_shape_function_derivs(xi: float, eta: float) -> torch.Tensor:
    """Q4 derivatives wrt ``(xi, eta)``; returns ``(4, 2)``."""
    x = torch.tensor(float(xi), dtype=_DTYPE)
    y = torch.tensor(float(eta), dtype=_DTYPE)
    dxi = 0.25 * torch.stack([
        -(1.0 - y),
        (1.0 - y),
        (1.0 + y),
        -(1.0 + y),
    ])
    deta = 0.25 * torch.stack([
        -(1.0 - x),
        -(1.0 + x),
        (1.0 + x),
        (1.0 - x),
    ])
    return torch.stack([dxi, deta], dim=1)


def q8_shape_functions(xi: float, eta: float) -> torch.Tensor:
    """Serendipity Q8 shape functions in Gmsh-style corner/midside order."""
    x = torch.tensor(float(xi), dtype=_DTYPE)
    y = torch.tensor(float(eta), dtype=_DTYPE)
    return torch.stack([
        -0.25 * (1.0 - x) * (1.0 - y) * (1.0 + x + y),
        -0.25 * (1.0 + x) * (1.0 - y) * (1.0 - x + y),
        -0.25 * (1.0 + x) * (1.0 + y) * (1.0 - x - y),
        -0.25 * (1.0 - x) * (1.0 + y) * (1.0 + x - y),
        0.5 * (1.0 - x * x) * (1.0 - y),
        0.5 * (1.0 + x) * (1.0 - y * y),
        0.5 * (1.0 - x * x) * (1.0 + y),
        0.5 * (1.0 - x) * (1.0 - y * y),
    ])


def q8_shape_function_derivs(xi: float, eta: float) -> torch.Tensor:
    """Q8 derivatives wrt ``(xi, eta)``; returns ``(8, 2)``."""
    x = torch.tensor(float(xi), dtype=_DTYPE)
    y = torch.tensor(float(eta), dtype=_DTYPE)
    dxi = torch.stack([
        0.25 * (1.0 - y) * (2.0 * x + y),
        0.25 * (1.0 - y) * (2.0 * x - y),
        0.25 * (1.0 + y) * (2.0 * x + y),
        0.25 * (1.0 + y) * (2.0 * x - y),
        -x * (1.0 - y),
        0.5 * (1.0 - y * y),
        -x * (1.0 + y),
        -0.5 * (1.0 - y * y),
    ])
    deta = torch.stack([
        0.25 * (1.0 - x) * (x + 2.0 * y),
        0.25 * (1.0 + x) * (-x + 2.0 * y),
        0.25 * (1.0 + x) * (x + 2.0 * y),
        0.25 * (1.0 - x) * (-x + 2.0 * y),
        -0.5 * (1.0 - x * x),
        -y * (1.0 + x),
        0.5 * (1.0 - x * x),
        -y * (1.0 - x),
    ])
    return torch.stack([dxi, deta], dim=1)


def q9_shape_functions(xi: float, eta: float) -> torch.Tensor:
    """Tensor-product quadratic Q9 shape functions."""
    x = torch.tensor(float(xi), dtype=_DTYPE)
    y = torch.tensor(float(eta), dtype=_DTYPE)
    Lx, _ = _line_lagrange_2(x)
    Ly, _ = _line_lagrange_2(y)
    return torch.stack([
        Lx[0] * Ly[0],
        Lx[2] * Ly[0],
        Lx[2] * Ly[2],
        Lx[0] * Ly[2],
        Lx[1] * Ly[0],
        Lx[2] * Ly[1],
        Lx[1] * Ly[2],
        Lx[0] * Ly[1],
        Lx[1] * Ly[1],
    ])


def q9_shape_function_derivs(xi: float, eta: float) -> torch.Tensor:
    """Q9 derivatives wrt ``(xi, eta)``; returns ``(9, 2)``."""
    x = torch.tensor(float(xi), dtype=_DTYPE)
    y = torch.tensor(float(eta), dtype=_DTYPE)
    Lx, dLx = _line_lagrange_2(x)
    Ly, dLy = _line_lagrange_2(y)
    pairs = [
        (0, 0), (2, 0), (2, 2), (0, 2), (1, 0),
        (2, 1), (1, 2), (0, 1), (1, 1),
    ]
    dxi = torch.stack([dLx[i] * Ly[j] for i, j in pairs])
    deta = torch.stack([Lx[i] * dLy[j] for i, j in pairs])
    return torch.stack([dxi, deta], dim=1)


def quad_shape_functions(kind: QuadKind, xi: float, eta: float) -> torch.Tensor:
    """Dispatch shape functions for Q4/Q8/Q9."""
    kind = kind.upper()
    if kind == "Q4":
        return q4_shape_functions(xi, eta)
    if kind == "Q8":
        return q8_shape_functions(xi, eta)
    if kind == "Q9":
        return q9_shape_functions(xi, eta)
    raise ValueError(f"Unsupported quadrilateral kind {kind!r}")


def quad_shape_function_derivs(kind: QuadKind, xi: float,
                               eta: float) -> torch.Tensor:
    """Dispatch reference derivatives for Q4/Q8/Q9."""
    kind = kind.upper()
    if kind == "Q4":
        return q4_shape_function_derivs(xi, eta)
    if kind == "Q8":
        return q8_shape_function_derivs(xi, eta)
    if kind == "Q9":
        return q9_shape_function_derivs(xi, eta)
    raise ValueError(f"Unsupported quadrilateral kind {kind!r}")


def quad_gauss_points(order: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensor-product Gauss-Legendre quadrature on ``[-1, 1]^2``.

    ``order`` is the number of 1D points. Use 2 for Q4 and 3 for quadratic
    Q8/Q9 stiffness integration on affine or mildly distorted elements.
    """
    if order == 1:
        pts_1d = torch.tensor([0.0], dtype=_DTYPE)
        w_1d = torch.tensor([2.0], dtype=_DTYPE)
    elif order == 2:
        a = 1.0 / torch.sqrt(torch.tensor(3.0, dtype=_DTYPE))
        pts_1d = torch.tensor([-a, a], dtype=_DTYPE)
        w_1d = torch.ones(2, dtype=_DTYPE)
    elif order == 3:
        a = torch.sqrt(torch.tensor(3.0 / 5.0, dtype=_DTYPE))
        pts_1d = torch.tensor([-a, 0.0, a], dtype=_DTYPE)
        w_1d = torch.tensor([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0],
                            dtype=_DTYPE)
    else:
        raise ValueError(f"Unsupported order={order}; use 1, 2, or 3.")

    pts = []
    wts = []
    for j, eta in enumerate(pts_1d):
        for i, xi in enumerate(pts_1d):
            pts.append(torch.stack([xi, eta]))
            wts.append(w_1d[i] * w_1d[j])
    return torch.stack(pts), torch.stack(wts)


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


def quad_element_stiffness(
    kind: QuadKind,
    node_coords: torch.Tensor,
    E: float,
    nu: float,
    plane_strain: bool = True,
    order: int | None = None,
) -> torch.Tensor:
    """Assemble one Q4/Q8/Q9 elastic stiffness matrix with ``B.T @ D @ B``.

    Returns a ``(2*n_nodes, 2*n_nodes)`` matrix with interleaved DOF ordering
    ``[u0, v0, u1, v1, ...]``.
    """
    kind = kind.upper()
    coords = node_coords.to(_DTYPE)
    n = {"Q4": 4, "Q8": 8, "Q9": 9}.get(kind)
    if n is None:
        raise ValueError(f"Unsupported quadrilateral kind {kind!r}")
    if coords.shape != (n, 2):
        raise ValueError(
            f"node_coords for {kind} must be ({n}, 2), got {tuple(coords.shape)}")
    if order is None:
        order = 2 if kind == "Q4" else 3

    D = _plane_strain_D(E, nu) if plane_strain else _plane_stress_D(E, nu)
    pts, wts = quad_gauss_points(order=order)
    K = torch.zeros((2 * n, 2 * n), dtype=_DTYPE)
    for q in range(pts.shape[0]):
        xi, eta = float(pts[q, 0]), float(pts[q, 1])
        dN_ref = quad_shape_function_derivs(kind, xi, eta)
        J = dN_ref.T @ coords
        detJ = torch.det(J)
        if detJ <= 0:
            raise ValueError("Non-positive Jacobian; check node ordering.")
        Jinv = torch.linalg.inv(J)
        dN_xy = dN_ref @ Jinv.T
        B = torch.zeros((3, 2 * n), dtype=_DTYPE)
        for i in range(n):
            B[0, 2 * i] = dN_xy[i, 0]
            B[1, 2 * i + 1] = dN_xy[i, 1]
            B[2, 2 * i] = dN_xy[i, 1]
            B[2, 2 * i + 1] = dN_xy[i, 0]
        K = K + (B.T @ D @ B) * (detJ * wts[q])
    return K


def q4_quadrature_geometry(
    nodes: torch.Tensor,
    quads: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(N_q, gradN_q, wdetJ_q)`` for native Q4 integration.

    Shapes are ``N_q=(4,4)``, ``gradN_q=(E,4,4,2)``, and
    ``wdetJ_q=(E,4)`` for the standard 2x2 Gauss rule.
    """

    device = nodes.device
    qidx = quads.to(device=device, dtype=torch.long)
    coords = nodes.to(device=device, dtype=_DTYPE)[qidx]
    pts, wts = quad_gauss_points(order=2)
    wts = wts.to(device=device, dtype=_DTYPE)
    n_elems = coords.shape[0]
    N_all = []
    grad_all = torch.zeros((n_elems, 4, 4, 2), dtype=_DTYPE, device=device)
    wdet = torch.zeros((n_elems, 4), dtype=_DTYPE, device=device)
    for q in range(4):
        xi, eta = float(pts[q, 0]), float(pts[q, 1])
        N = q4_shape_functions(xi, eta).to(device=device, dtype=_DTYPE)
        dN_ref = q4_shape_function_derivs(xi, eta).to(
            device=device,
            dtype=_DTYPE,
        )
        J = torch.einsum("ai,eaj->eij", dN_ref, coords)
        detJ = torch.linalg.det(J)
        if torch.any(detJ <= 0.0):
            raise ValueError("Q4 element has non-positive Jacobian")
        invJ = torch.linalg.inv(J)
        grad_all[:, q] = torch.einsum("ai,eji->eaj", dN_ref, invJ)
        wdet[:, q] = wts[q] * detJ
        N_all.append(N)
    return torch.stack(N_all), grad_all, wdet


def q4_strain_at_gauss(
    nodes: torch.Tensor,
    quads: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Return Q4 strain ``[exx, eyy, gamma_xy]`` at 2x2 Gauss points."""

    _, grad, _ = q4_quadrature_geometry(nodes, quads)
    device = grad.device
    qidx = quads.to(device=device, dtype=torch.long)
    u_e = u.to(device=device, dtype=_DTYPE)[qidx]
    eps = torch.zeros((quads.shape[0], 4, 3), dtype=_DTYPE, device=device)
    eps[:, :, 0] = torch.einsum("eqa,ea->eq", grad[..., 0], u_e[..., 0])
    eps[:, :, 1] = torch.einsum("eqa,ea->eq", grad[..., 1], u_e[..., 1])
    eps[:, :, 2] = (
        torch.einsum("eqa,ea->eq", grad[..., 1], u_e[..., 0])
        + torch.einsum("eqa,ea->eq", grad[..., 0], u_e[..., 1])
    )
    return eps


def q4_internal_force(
    nodes: torch.Tensor,
    quads: torch.Tensor,
    u: torch.Tensor,
    *,
    E: float,
    nu: float,
    plane_strain: bool = True,
    d: torch.Tensor | None = None,
    eta_residual: float = 1.0e-7,
) -> torch.Tensor:
    """Assemble native Q4 linear-elastic internal force."""

    N, grad, wdet = q4_quadrature_geometry(nodes, quads)
    eps = q4_strain_at_gauss(nodes, quads, u)
    device = grad.device
    qidx = quads.to(device=device, dtype=torch.long)
    D = _plane_strain_D(E, nu) if plane_strain else _plane_stress_D(E, nu)
    D = D.to(device=device, dtype=_DTYPE)
    stress = torch.einsum("ij,eqj->eqi", D, eps)
    if d is not None:
        d_q = torch.einsum("qa,ea->eq", N, d.to(device=device, dtype=_DTYPE)[qidx])
        g = (1.0 - d_q) ** 2 + eta_residual
        stress = stress * g.unsqueeze(-1)

    f_q = torch.zeros((quads.shape[0], 4, 4, 2), dtype=_DTYPE, device=device)
    f_q[..., 0] = (
        grad[..., 0] * stress[..., 0].unsqueeze(-1)
        + grad[..., 1] * stress[..., 2].unsqueeze(-1)
    ) * wdet.unsqueeze(-1)
    f_q[..., 1] = (
        grad[..., 1] * stress[..., 1].unsqueeze(-1)
        + grad[..., 0] * stress[..., 2].unsqueeze(-1)
    ) * wdet.unsqueeze(-1)
    f_e = f_q.sum(dim=1)
    out = torch.zeros((nodes.shape[0], 2), dtype=_DTYPE, device=device)
    out.scatter_add_(
        0,
        qidx.reshape(-1, 1).expand(-1, 2),
        f_e.reshape(-1, 2),
    )
    return out


def q4_laplacian_matvec(
    nodes: torch.Tensor,
    quads: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Native Q4 scalar Laplacian action ``∫ gradN.gradx``."""

    _, grad, wdet = q4_quadrature_geometry(nodes, quads)
    device = grad.device
    qidx = quads.to(device=device, dtype=torch.long)
    x_e = x.to(device=device, dtype=_DTYPE)[qidx]
    gx = torch.einsum("eqa,ea->eq", grad[..., 0], x_e)
    gy = torch.einsum("eqa,ea->eq", grad[..., 1], x_e)
    r_e = (
        grad[..., 0] * gx.unsqueeze(-1)
        + grad[..., 1] * gy.unsqueeze(-1)
    ) * wdet.unsqueeze(-1)
    r_e = r_e.sum(dim=1)
    out = torch.zeros(nodes.shape[0], dtype=_DTYPE, device=device)
    out.scatter_add_(0, qidx.reshape(-1), r_e.reshape(-1))
    return out


def q4_mass_matvec(
    nodes: torch.Tensor,
    quads: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Native Q4 scalar consistent mass action ``∫ N_i N_j x_j``."""

    N, _, wdet = q4_quadrature_geometry(nodes, quads)
    device = wdet.device
    qidx = quads.to(device=device, dtype=torch.long)
    x_e = x.to(device=device, dtype=_DTYPE)[qidx]
    x_q = torch.einsum("qa,ea->eq", N, x_e)
    r_e = torch.einsum("qa,eq,eq->ea", N, x_q, wdet)
    out = torch.zeros(nodes.shape[0], dtype=_DTYPE, device=device)
    out.scatter_add_(0, qidx.reshape(-1), r_e.reshape(-1))
    return out


def structured_q4_mesh(width: float, height: float, nx: int, ny: int,
                       *, dtype: torch.dtype = _DTYPE) -> StructuredQ4Mesh:
    """Build a structured rectangular Q4 mesh in counter-clockwise ordering."""

    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive")
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    xs = torch.linspace(0.0, float(width), int(nx) + 1, dtype=dtype)
    ys = torch.linspace(0.0, float(height), int(ny) + 1, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    nodes = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    quads = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n3 = n0 + (nx + 1)
            n2 = n3 + 1
            quads.append((n0, n1, n2, n3))
    quads_t = torch.tensor(quads, dtype=torch.long)
    node_sets = {
        "left": torch.arange(0, (ny + 1) * (nx + 1), nx + 1, dtype=torch.long),
        "right": torch.arange(nx, (ny + 1) * (nx + 1), nx + 1, dtype=torch.long),
        "bottom": torch.arange(0, nx + 1, dtype=torch.long),
        "top": torch.arange(ny * (nx + 1), (ny + 1) * (nx + 1), dtype=torch.long),
    }
    return StructuredQ4Mesh(nodes=nodes, quads=quads_t,
                            node_sets=node_sets, nx=nx, ny=ny)


def q4_to_triangles(quads: torch.Tensor, *, diagonal: Literal["02", "13"] = "02"
                    ) -> torch.Tensor:
    """Convert Q4 connectivity to P1 triangles for existing triangle solvers."""

    q = quads.to(dtype=torch.long)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quads must have shape (n, 4), got {tuple(q.shape)}")
    if diagonal == "02":
        tri = torch.stack([
            torch.stack([q[:, 0], q[:, 1], q[:, 2]], dim=1),
            torch.stack([q[:, 0], q[:, 2], q[:, 3]], dim=1),
        ], dim=1)
    elif diagonal == "13":
        tri = torch.stack([
            torch.stack([q[:, 0], q[:, 1], q[:, 3]], dim=1),
            torch.stack([q[:, 1], q[:, 2], q[:, 3]], dim=1),
        ], dim=1)
    else:
        raise ValueError("diagonal must be '02' or '13'")
    return tri.reshape(-1, 3)


def q4_signed_areas(nodes: torch.Tensor, quads: torch.Tensor) -> torch.Tensor:
    """Signed polygon area for Q4 elements."""

    xy = nodes[quads.to(dtype=torch.long)]
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    return 0.5 * (
        (x * torch.roll(y, shifts=-1, dims=1)).sum(dim=1)
        - (y * torch.roll(x, shifts=-1, dims=1)).sum(dim=1)
    )


def q4_quality(nodes: torch.Tensor, quads: torch.Tensor) -> dict[str, float]:
    """Return basic Q4 quality metrics for production checks."""

    areas = q4_signed_areas(nodes, quads)
    xy = nodes[quads.to(dtype=torch.long)]
    edges = torch.roll(xy, shifts=-1, dims=1) - xy
    lengths = torch.linalg.norm(edges, dim=2)
    aspect = lengths.max(dim=1).values / lengths.clamp_min(1e-30).min(dim=1).values
    return {
        "n_quads": int(quads.shape[0]),
        "min_signed_area": float(areas.min().item()),
        "max_signed_area": float(areas.max().item()),
        "max_aspect_ratio": float(aspect.max().item()),
        "all_positive_orientation": bool(torch.all(areas > 0.0).item()),
    }
