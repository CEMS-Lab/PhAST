"""Eight-node hexahedral (H8) element primitives: reference nodes, 2x2x2 Gauss quadrature, strain-displacement matrices, structured blocks."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


_DTYPE = torch.float64


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


def h8_reference_nodes(
    *,
    dtype: torch.dtype = _DTYPE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return H8 corner signs in the local node ordering used by this module."""

    return torch.tensor(
        (
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ),
        dtype=dtype,
        device=device,
    )


@dataclass
class H8Mesh:
    """Linear hexahedral mesh with a structured-block construction helper."""

    nodes: torch.Tensor
    elements: torch.Tensor
    structured_shape: tuple[int, int, int] | None = None
    dimensions: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(f"nodes must have shape (N, 3), got {tuple(self.nodes.shape)}")
        if self.elements.ndim != 2 or self.elements.shape[1] != 8:
            raise ValueError(f"elements must have shape (E, 8), got {tuple(self.elements.shape)}")
        if not torch.is_floating_point(self.nodes):
            raise TypeError("H8 nodes must use a floating-point dtype")
        if self.elements.numel() == 0:
            raise ValueError("at least one H8 element is required")
        self.elements = self.elements.to(device=self.nodes.device, dtype=torch.long).contiguous()
        if self.elements.min().item() < 0 or self.elements.max().item() >= self.nodes.shape[0]:
            raise ValueError("H8 connectivity contains an out-of-range node index")
        self.nodes = self.nodes.contiguous()

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elems(self) -> int:
        return int(self.elements.shape[0])

    @property
    def n_dofs(self) -> int:
        return 3 * self.n_nodes

    @property
    def dtype(self) -> torch.dtype:
        return self.nodes.dtype

    @property
    def device(self) -> torch.device:
        return self.nodes.device

    @classmethod
    def structured_block(
        cls,
        *,
        length: float,
        height: float,
        thickness: float,
        nx: int,
        ny: int,
        nz: int,
        dtype: torch.dtype = _DTYPE,
        device: torch.device | str = "cpu",
    ) -> "H8Mesh":
        """Create an axis-aligned H8 block with positive reference orientation."""

        for name, value in (
            ("length", length),
            ("height", height),
            ("thickness", thickness),
        ):
            _positive(name, value)
        for name, value in (("nx", nx), ("ny", ny), ("nz", nz)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        device = torch.device(device)
        xs = torch.linspace(0.0, length, nx + 1, dtype=dtype, device=device)
        ys = torch.linspace(0.0, height, ny + 1, dtype=dtype, device=device)
        zs = torch.linspace(0.0, thickness, nz + 1, dtype=dtype, device=device)
        z_grid, y_grid, x_grid = torch.meshgrid(zs, ys, xs, indexing="ij")
        nodes = torch.stack((x_grid, y_grid, z_grid), dim=-1).reshape(-1, 3)

        kk, jj, ii = torch.meshgrid(
            torch.arange(nz, dtype=torch.long, device=device),
            torch.arange(ny, dtype=torch.long, device=device),
            torch.arange(nx, dtype=torch.long, device=device),
            indexing="ij",
        )
        stride_x = nx + 1
        stride_z = (ny + 1) * stride_x
        v0 = (kk * (ny + 1) + jj) * stride_x + ii
        v1 = v0 + 1
        v3 = v0 + stride_x
        v2 = v3 + 1
        v4 = v0 + stride_z
        v5 = v4 + 1
        v7 = v4 + stride_x
        v6 = v7 + 1
        elements = torch.stack((v0, v1, v2, v3, v4, v5, v6, v7), dim=-1).reshape(-1, 8)
        return cls(
            nodes=nodes,
            elements=elements,
            structured_shape=(nx, ny, nz),
            dimensions=(float(length), float(height), float(thickness)),
        )

    def nodes_on_plane(
        self,
        axis: int,
        coordinate: float,
        *,
        tolerance: float | None = None,
    ) -> torch.Tensor:
        """Return mesh nodes on a coordinate-aligned plane."""

        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
        if tolerance is None:
            scale = max(float(self.nodes.abs().max().item()), 1.0)
            tolerance = 1.0e-10 * scale
        return torch.nonzero(
            torch.abs(self.nodes[:, axis] - coordinate) <= tolerance,
            as_tuple=False,
        ).flatten()


def h8_gauss_points(
    *,
    order: int = 2,
    dtype: torch.dtype = _DTYPE,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tensor-product Gauss points and weights on [-1, 1]^3."""

    if order == 1:
        points_1d = torch.zeros(1, dtype=dtype, device=device)
        weights_1d = torch.full((1,), 2.0, dtype=dtype, device=device)
    elif order == 2:
        a = 1.0 / math.sqrt(3.0)
        points_1d = torch.tensor((-a, a), dtype=dtype, device=device)
        weights_1d = torch.ones(2, dtype=dtype, device=device)
    else:
        raise ValueError(f"unsupported H8 Gauss order {order!r}; use 1 or 2")
    zeta, eta, xi = torch.meshgrid(points_1d, points_1d, points_1d, indexing="ij")
    wz, wy, wx = torch.meshgrid(weights_1d, weights_1d, weights_1d, indexing="ij")
    points = torch.stack((xi, eta, zeta), dim=-1).reshape(-1, 3)
    weights = (wx * wy * wz).reshape(-1)
    return points, weights


def h8_shape_functions(
    xi: float | torch.Tensor,
    eta: float | torch.Tensor,
    zeta: float | torch.Tensor,
    *,
    dtype: torch.dtype = _DTYPE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Evaluate H8 shape functions in the local H8 ordering."""

    signs = h8_reference_nodes(dtype=dtype, device=device)
    xi_t = torch.as_tensor(xi, dtype=dtype, device=signs.device)
    eta_t = torch.as_tensor(eta, dtype=dtype, device=signs.device)
    zeta_t = torch.as_tensor(zeta, dtype=dtype, device=signs.device)
    return 0.125 * (
        (1.0 + signs[:, 0] * xi_t)
        * (1.0 + signs[:, 1] * eta_t)
        * (1.0 + signs[:, 2] * zeta_t)
    )


def h8_shape_function_derivs(
    xi: float | torch.Tensor,
    eta: float | torch.Tensor,
    zeta: float | torch.Tensor,
    *,
    dtype: torch.dtype = _DTYPE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return H8 local derivatives with columns (d/dxi, d/deta, d/dzeta)."""

    signs = h8_reference_nodes(dtype=dtype, device=device)
    xi_t = torch.as_tensor(xi, dtype=dtype, device=signs.device)
    eta_t = torch.as_tensor(eta, dtype=dtype, device=signs.device)
    zeta_t = torch.as_tensor(zeta, dtype=dtype, device=signs.device)
    d_xi = 0.125 * signs[:, 0] * (1.0 + signs[:, 1] * eta_t) * (1.0 + signs[:, 2] * zeta_t)
    d_eta = 0.125 * (1.0 + signs[:, 0] * xi_t) * signs[:, 1] * (1.0 + signs[:, 2] * zeta_t)
    d_zeta = 0.125 * (1.0 + signs[:, 0] * xi_t) * (1.0 + signs[:, 1] * eta_t) * signs[:, 2]
    return torch.stack((d_xi, d_eta, d_zeta), dim=-1)


def h8_quadrature_geometry(
    nodes: torch.Tensor,
    elements: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return H8 shape values, physical gradients and weighted determinants.

    Returned shapes are N_q=(8, 8), grad_N=(E, 8, 8, 3), and
    wdet_J=(E, 8) for the standard 2x2x2 quadrature rule. Callers may pass an
    element batch rather than the entire mesh; a benchmark-scale route must
    not retain all quadrature gradients or B matrices at once.
    """

    mesh = H8Mesh(nodes=nodes, elements=elements)
    coords = mesh.nodes[mesh.elements]
    points, weights = h8_gauss_points(dtype=mesh.dtype, device=mesh.device)
    n_q = int(points.shape[0])
    N_all: list[torch.Tensor] = []
    gradients = torch.zeros(
        (mesh.n_elems, n_q, 8, 3),
        dtype=mesh.dtype,
        device=mesh.device,
    )
    weighted_determinants = torch.zeros(
        (mesh.n_elems, n_q),
        dtype=mesh.dtype,
        device=mesh.device,
    )
    for q in range(n_q):
        xi, eta, zeta = points[q]
        N = h8_shape_functions(xi, eta, zeta, dtype=mesh.dtype, device=mesh.device)
        derivatives = h8_shape_function_derivs(
            xi, eta, zeta, dtype=mesh.dtype, device=mesh.device
        )
        jacobian = torch.einsum("ar,eaj->erj", derivatives, coords)
        determinant = torch.linalg.det(jacobian)
        if bool((determinant <= 0.0).any()):
            raise ValueError("H8 element has non-positive Jacobian")
        inverse_jacobian = torch.linalg.inv(jacobian)
        gradients[:, q] = torch.einsum("ar,ejr->eaj", derivatives, inverse_jacobian)
        weighted_determinants[:, q] = weights[q] * determinant
        N_all.append(N)
    return torch.stack(N_all), gradients, weighted_determinants


def h8_B_matrices(gradient_N: torch.Tensor) -> torch.Tensor:
    """Build engineering-strain H8 B matrices with shape (E, Q, 6, 24)."""

    if gradient_N.ndim != 4 or gradient_N.shape[2:] != (8, 3):
        raise ValueError(
            "gradient_N must have shape (E, Q, 8, 3), "
            f"got {tuple(gradient_N.shape)}")
    n_elem, n_q = gradient_N.shape[:2]
    B = torch.zeros(
        (n_elem, n_q, 6, 24),
        dtype=gradient_N.dtype,
        device=gradient_N.device,
    )
    for local_node in range(8):
        dof = 3 * local_node
        gx = gradient_N[:, :, local_node, 0]
        gy = gradient_N[:, :, local_node, 1]
        gz = gradient_N[:, :, local_node, 2]
        B[:, :, 0, dof] = gx
        B[:, :, 1, dof + 1] = gy
        B[:, :, 2, dof + 2] = gz
        B[:, :, 3, dof + 1] = gz
        B[:, :, 3, dof + 2] = gy
        B[:, :, 4, dof] = gz
        B[:, :, 4, dof + 2] = gx
        B[:, :, 5, dof] = gy
        B[:, :, 5, dof + 1] = gx
    return B


def isotropic_stiffness_3d(
    youngs_modulus: float,
    poisson_ratio: float,
    *,
    dtype: torch.dtype = _DTYPE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return engineering-strain 3D isotropic stiffness in Voigt-6 order."""

    _positive("youngs_modulus", youngs_modulus)
    if not math.isfinite(poisson_ratio) or not -0.99 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie between -0.99 and 0.5")
    shear = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lame = youngs_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    stiffness = torch.zeros((6, 6), dtype=dtype, device=device)
    stiffness[:3, :3] = lame
    stiffness[0, 0] += 2.0 * shear
    stiffness[1, 1] += 2.0 * shear
    stiffness[2, 2] += 2.0 * shear
    stiffness[3, 3] = shear
    stiffness[4, 4] = shear
    stiffness[5, 5] = shear
    return stiffness
