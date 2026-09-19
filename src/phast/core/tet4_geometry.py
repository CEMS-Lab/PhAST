"""Experimental TET4 geometry primitives.

This module is intentionally limited to geometry precomputation for linear
tetrahedra: node/connectivity validation, orientation repair, element volumes,
and constant shape-function gradients. It does not provide a 3D solver,
constitutive law, fracture model, boundary-condition API, material-region
assignment, or output writer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import torch


_SUPPORTED_NODE_DTYPES = {torch.float32, torch.float64}


def _require_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return value


def _require_finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return value


def _require_axis(axis: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, Integral):
        raise TypeError(f"axis must be an integer 0, 1, or 2, got {axis!r}")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
    return axis


def _require_integral_connectivity(elements: torch.Tensor) -> None:
    if elements.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("elements must use an integer dtype")


@dataclass(frozen=True)
class Tet4Geometry:
    """Linear tetrahedral geometry with precomputed TET4 quantities.

    Parameters are stored as torch tensors so downstream numerical kernels can
    choose their own device and dtype. The object is geometry-only: the
    gradients are the gradients of the four scalar shape functions on each
    tetrahedron, not mechanics ``B`` matrices.
    """

    nodes: torch.Tensor
    elements: torch.Tensor
    volumes: torch.Tensor
    shape_gradients: torch.Tensor
    structured_shape: tuple[int, int, int] | None = None
    dimensions: tuple[float, float, float] | None = None

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])

    @property
    def dtype(self) -> torch.dtype:
        return self.nodes.dtype

    @property
    def device(self) -> torch.device:
        return self.nodes.device

    @property
    def centroids(self) -> torch.Tensor:
        return self.nodes[self.elements].mean(dim=1)

    @classmethod
    def from_tensors(
        cls,
        nodes: torch.Tensor | Sequence[Sequence[float]],
        elements: torch.Tensor | Sequence[Sequence[int]],
        *,
        structured_shape: tuple[int, int, int] | None = None,
        dimensions: tuple[float, float, float] | None = None,
        volume_tolerance: float | None = None,
    ) -> "Tet4Geometry":
        """Build and validate TET4 geometry from node and connectivity arrays.

        Negative-orientation tetrahedra are repaired by swapping the last two
        local nodes. Degenerate tetrahedra are rejected after this orientation
        pass.
        """

        node_tensor = torch.as_tensor(nodes)
        if node_tensor.ndim != 2 or node_tensor.shape[1] != 3:
            raise ValueError(
                f"nodes must have shape (N, 3), got {tuple(node_tensor.shape)}"
            )
        if not torch.is_floating_point(node_tensor):
            raise TypeError("nodes must use a floating-point dtype")
        if node_tensor.dtype not in _SUPPORTED_NODE_DTYPES:
            raise TypeError(
                "nodes must use torch.float32 or torch.float64, "
                f"got {node_tensor.dtype}"
            )
        if not bool(torch.isfinite(node_tensor).all()):
            raise ValueError("nodes must contain only finite coordinates")
        node_tensor = node_tensor.contiguous()
        if volume_tolerance is not None:
            volume_tolerance = _require_finite_nonnegative(
                "volume_tolerance", volume_tolerance
            )

        element_tensor = torch.as_tensor(elements)
        _require_integral_connectivity(element_tensor)
        element_tensor = element_tensor.to(device=node_tensor.device, dtype=torch.long)
        if element_tensor.ndim != 2 or element_tensor.shape[1] != 4:
            raise ValueError(
                "elements must have shape (E, 4), "
                f"got {tuple(element_tensor.shape)}"
            )
        if element_tensor.numel() == 0:
            raise ValueError("at least one TET4 element is required")
        if element_tensor.min().item() < 0 or element_tensor.max().item() >= node_tensor.shape[0]:
            raise ValueError("TET4 connectivity contains an out-of-range node index")
        sorted_nodes = torch.sort(element_tensor, dim=1).values
        repeated_nodes = (sorted_nodes[:, 1:] == sorted_nodes[:, :-1]).any(dim=1)
        if bool(repeated_nodes.any()):
            first_repeated = int(torch.nonzero(repeated_nodes, as_tuple=False)[0].item())
            raise ValueError(
                "TET4 connectivity contains repeated node indices in element "
                f"{first_repeated}"
            )
        element_tensor = element_tensor.clone()

        coords = node_tensor[element_tensor]
        edge_matrix = torch.stack(
            (
                coords[:, 1] - coords[:, 0],
                coords[:, 2] - coords[:, 0],
                coords[:, 3] - coords[:, 0],
            ),
            dim=1,
        )
        signed_six_volume = torch.linalg.det(edge_matrix)
        if not bool(torch.isfinite(signed_six_volume).all()):
            raise ValueError("computed TET4 signed volumes must be finite")
        negative = signed_six_volume < 0.0
        if bool(negative.any()):
            repaired = element_tensor[negative].clone()
            repaired[:, 2], repaired[:, 3] = repaired[:, 3].clone(), repaired[:, 2].clone()
            element_tensor[negative] = repaired
            coords = node_tensor[element_tensor]
            edge_matrix = torch.stack(
                (
                    coords[:, 1] - coords[:, 0],
                    coords[:, 2] - coords[:, 0],
                    coords[:, 3] - coords[:, 0],
                ),
                dim=1,
            )
            signed_six_volume = torch.linalg.det(edge_matrix)
            if not bool(torch.isfinite(signed_six_volume).all()):
                raise ValueError("computed TET4 signed volumes must be finite")

        volumes = signed_six_volume / 6.0
        if not bool(torch.isfinite(volumes).all()):
            raise ValueError("computed TET4 volumes must be finite")
        if volume_tolerance is None:
            edge_lengths = torch.stack(
                (
                    torch.linalg.vector_norm(coords[:, 1] - coords[:, 0], dim=1),
                    torch.linalg.vector_norm(coords[:, 2] - coords[:, 0], dim=1),
                    torch.linalg.vector_norm(coords[:, 3] - coords[:, 0], dim=1),
                    torch.linalg.vector_norm(coords[:, 2] - coords[:, 1], dim=1),
                    torch.linalg.vector_norm(coords[:, 3] - coords[:, 1], dim=1),
                    torch.linalg.vector_norm(coords[:, 3] - coords[:, 2], dim=1),
                ),
                dim=1,
            )
            max_edge = edge_lengths.max(dim=1).values
            volume_tolerances = 64.0 * torch.finfo(node_tensor.dtype).eps * max_edge.pow(3)
        else:
            volume_tolerances = torch.full_like(volumes, float(volume_tolerance))
        degenerate = volumes <= volume_tolerances
        if bool(degenerate.any()):
            count = int(degenerate.sum().item())
            raise ValueError(f"mesh contains {count} degenerate or inverted tetrahedra")

        inverse_edges = torch.linalg.inv(edge_matrix)
        if not bool(torch.isfinite(inverse_edges).all()):
            raise ValueError("computed TET4 edge inverses must be finite")
        nonzero_gradients = inverse_edges.transpose(1, 2)
        shape_gradients = torch.empty(
            (element_tensor.shape[0], 4, 3),
            dtype=node_tensor.dtype,
            device=node_tensor.device,
        )
        shape_gradients[:, 1:, :] = nonzero_gradients
        shape_gradients[:, 0, :] = -nonzero_gradients.sum(dim=1)
        shape_gradients = shape_gradients.contiguous()
        if not bool(torch.isfinite(shape_gradients).all()):
            raise ValueError("computed TET4 shape gradients must be finite")

        return cls(
            nodes=node_tensor,
            elements=element_tensor.contiguous(),
            volumes=volumes.contiguous(),
            shape_gradients=shape_gradients,
            structured_shape=structured_shape,
            dimensions=dimensions,
        )

    def nodes_on_plane(
        self,
        axis: int,
        coordinate: float,
        *,
        tolerance: float | None = None,
    ) -> torch.Tensor:
        """Return nodes with coordinate ``axis`` on the requested plane.

        The default tolerance is based on the local extent of the selected
        coordinate and the spacing of representable floating-point values near
        ``coordinate``. It deliberately avoids scaling with the absolute
        coordinate magnitude of unrelated axes, so translated meshes retain the
        same plane membership when the translated coordinates are
        representable.
        """

        axis = _require_axis(axis)
        coordinate = float(coordinate)
        if not math.isfinite(coordinate):
            raise ValueError("coordinate must be finite")
        coordinate_tensor = torch.tensor(
            coordinate,
            dtype=self.nodes.dtype,
            device=self.nodes.device,
        )
        if not bool(torch.isfinite(coordinate_tensor)):
            raise ValueError("coordinate must be finite in the node dtype")
        if tolerance is None:
            axis_values = self.nodes[:, axis]
            local_extent = torch.abs(axis_values.max() - axis_values.min())
            finfo = torch.finfo(self.nodes.dtype)
            positive_infinity = torch.tensor(
                float("inf"),
                dtype=self.nodes.dtype,
                device=self.nodes.device,
            )
            negative_infinity = torch.tensor(
                float("-inf"),
                dtype=self.nodes.dtype,
                device=self.nodes.device,
            )
            next_up = torch.nextafter(coordinate_tensor, positive_infinity)
            next_down = torch.nextafter(coordinate_tensor, negative_infinity)
            representable_spacing = torch.maximum(
                torch.abs(next_up - coordinate_tensor),
                torch.abs(coordinate_tensor - next_down),
            )
            tolerance_tensor = torch.maximum(
                4.0 * finfo.eps * local_extent,
                4.0 * representable_spacing,
            )
            tolerance = float(
                torch.maximum(
                    tolerance_tensor,
                    torch.tensor(finfo.tiny, dtype=self.nodes.dtype, device=self.nodes.device),
                ).item()
            )
        tolerance = _require_finite_nonnegative("tolerance", tolerance)
        return torch.nonzero(
            torch.abs(self.nodes[:, axis] - coordinate_tensor) <= tolerance,
            as_tuple=False,
        ).flatten()


def structured_tet4_block(
    *,
    length: float,
    height: float,
    thickness: float,
    nx: int,
    ny: int,
    nz: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tet4Geometry:
    """Create a rectangular structured block split into six TET4s per cell.

    This helper is intended for tests and small experimental setup checks. It
    is not a meshing front-end and does not create physical groups, regions or
    boundary-condition objects.
    """

    length = _require_positive("length", length)
    height = _require_positive("height", height)
    thickness = _require_positive("thickness", thickness)
    nx = _require_positive_int("nx", nx)
    ny = _require_positive_int("ny", ny)
    nz = _require_positive_int("nz", nz)
    target_device = torch.device(device)

    def node_id(i: int, j: int, k: int) -> int:
        return (k * (ny + 1) + j) * (nx + 1) + i

    points: list[tuple[float, float, float]] = []
    for k in range(nz + 1):
        z = thickness * k / nz
        for j in range(ny + 1):
            y = height * j / ny
            for i in range(nx + 1):
                x = length * i / nx
                points.append((x, y, z))

    elements: list[list[int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v0 = node_id(i, j, k)
                v1 = node_id(i + 1, j, k)
                v2 = node_id(i + 1, j + 1, k)
                v3 = node_id(i, j + 1, k)
                v4 = node_id(i, j, k + 1)
                v5 = node_id(i + 1, j, k + 1)
                v6 = node_id(i + 1, j + 1, k + 1)
                v7 = node_id(i, j + 1, k + 1)
                elements.extend(
                    (
                        [v0, v1, v2, v6],
                        [v0, v2, v3, v6],
                        [v0, v3, v7, v6],
                        [v0, v7, v4, v6],
                        [v0, v4, v5, v6],
                        [v0, v5, v1, v6],
                    )
                )

    return Tet4Geometry.from_tensors(
        torch.tensor(points, dtype=dtype, device=target_device),
        torch.tensor(elements, dtype=torch.long, device=target_device),
        structured_shape=(nx, ny, nz),
        dimensions=(length, height, thickness),
    )
