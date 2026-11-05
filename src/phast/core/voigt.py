"""Voigt, Mandel and tensor conversions for 3D strain, stress and stiffness (engineering shear convention), and the periodic node map of a voxel cell."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import torch
from phast.core.hex8 import H8Mesh, h8_B_matrices, h8_quadrature_geometry


_DTYPE = torch.float64


def engineering_strain_to_tensor(strain: torch.Tensor) -> torch.Tensor:
    """Convert engineering strain ``(..., 6)`` to symmetric tensors.

    The engineering order is ``(xx, yy, zz, yz, xz, xy)`` and therefore uses
    half of each engineering shear in the off-diagonal tensor entries.
    """

    if strain.shape[-1] != 6:
        raise ValueError(f"strain must end in 6 entries, got {tuple(strain.shape)}")
    tensor = torch.zeros(
        (*strain.shape[:-1], 3, 3), dtype=strain.dtype, device=strain.device
    )
    tensor[..., 0, 0] = strain[..., 0]
    tensor[..., 1, 1] = strain[..., 1]
    tensor[..., 2, 2] = strain[..., 2]
    tensor[..., 1, 2] = tensor[..., 2, 1] = 0.5 * strain[..., 3]
    tensor[..., 0, 2] = tensor[..., 2, 0] = 0.5 * strain[..., 4]
    tensor[..., 0, 1] = tensor[..., 1, 0] = 0.5 * strain[..., 5]
    return tensor


def engineering_stress_to_tensor(stress: torch.Tensor) -> torch.Tensor:
    """Convert engineering-Voigt stress ``(..., 6)`` to symmetric tensors."""

    if stress.shape[-1] != 6:
        raise ValueError(f"stress must end in 6 entries, got {tuple(stress.shape)}")
    tensor = torch.zeros(
        (*stress.shape[:-1], 3, 3), dtype=stress.dtype, device=stress.device
    )
    tensor[..., 0, 0] = stress[..., 0]
    tensor[..., 1, 1] = stress[..., 1]
    tensor[..., 2, 2] = stress[..., 2]
    tensor[..., 1, 2] = tensor[..., 2, 1] = stress[..., 3]
    tensor[..., 0, 2] = tensor[..., 2, 0] = stress[..., 4]
    tensor[..., 0, 1] = tensor[..., 1, 0] = stress[..., 5]
    return tensor


def tensor_to_engineering_strain(tensor: torch.Tensor) -> torch.Tensor:
    """Convert symmetric strain tensors to engineering-Voigt vectors."""

    if tensor.shape[-2:] != (3, 3):
        raise ValueError(f"tensor must end in (3, 3), got {tuple(tensor.shape)}")
    return torch.stack(
        (
            tensor[..., 0, 0],
            tensor[..., 1, 1],
            tensor[..., 2, 2],
            2.0 * tensor[..., 1, 2],
            2.0 * tensor[..., 0, 2],
            2.0 * tensor[..., 0, 1],
        ),
        dim=-1,
    )


def tensor_to_engineering_stress(tensor: torch.Tensor) -> torch.Tensor:
    """Convert symmetric stress tensors to engineering-Voigt vectors."""

    if tensor.shape[-2:] != (3, 3):
        raise ValueError(f"tensor must end in (3, 3), got {tuple(tensor.shape)}")
    return torch.stack(
        (
            tensor[..., 0, 0],
            tensor[..., 1, 1],
            tensor[..., 2, 2],
            tensor[..., 1, 2],
            tensor[..., 0, 2],
            tensor[..., 0, 1],
        ),
        dim=-1,
    )


def mandel_scaling(
    *, dtype: torch.dtype = _DTYPE, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return ``S`` where stress_Mandel = S stress_engineering."""

    return torch.tensor(
        (1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0), math.sqrt(2.0)),
        dtype=dtype,
        device=device,
    )


def engineering_to_mandel_stiffness(stiffness: torch.Tensor) -> torch.Tensor:
    """Convert engineering stiffness to the work-conjugate Mandel convention."""

    if stiffness.shape[-2:] != (6, 6):
        raise ValueError(f"stiffness must end in (6, 6), got {tuple(stiffness.shape)}")
    scale = mandel_scaling(dtype=stiffness.dtype, device=stiffness.device)
    return scale.unsqueeze(-1) * stiffness * scale.unsqueeze(-2)


@dataclass(frozen=True)
class PeriodicNodeMap:
    """Matrix-free periodic reduction map for a structured H8 mesh."""

    reduced_node_of_full: torch.Tensor
    n_reduced_nodes: int
    pinned_reduced_node: int = 0

    def __post_init__(self) -> None:
        if self.reduced_node_of_full.ndim != 1:
            raise ValueError("reduced_node_of_full must be one-dimensional")
        if self.reduced_node_of_full.dtype != torch.long:
            raise TypeError("reduced_node_of_full must use torch.long")
        if self.n_reduced_nodes < 1:
            raise ValueError("n_reduced_nodes must be positive")
        if not 0 <= self.pinned_reduced_node < self.n_reduced_nodes:
            raise ValueError("pinned_reduced_node is out of range")
        if self.reduced_node_of_full.numel() < 1:
            raise ValueError("periodic map needs at least one full node")
        if self.reduced_node_of_full.min().item() < 0 or (
            self.reduced_node_of_full.max().item() >= self.n_reduced_nodes
        ):
            raise ValueError("periodic map contains an out-of-range reduced node")

    @classmethod
    def from_structured_h8(cls, mesh: H8Mesh) -> "PeriodicNodeMap":
        """Map canonical structured-block nodes to their periodic representative.

        The arithmetic below deliberately relies on the node and H8 ordering
        emitted by :meth:`H8Mesh.structured_block`.  Validate that convention
        here rather than silently applying a wrong periodic map to a manually
        ordered mesh carrying only matching ``structured_shape`` metadata.
        """

        if mesh.structured_shape is None or mesh.dimensions is None:
            raise ValueError("periodic map requires a structured H8 mesh")
        nx, ny, nz = mesh.structured_shape
        length, height, thickness = mesh.dimensions
        canonical = H8Mesh.structured_block(
            length=length,
            height=height,
            thickness=thickness,
            nx=nx,
            ny=ny,
            nz=nz,
            dtype=mesh.dtype,
            device=mesh.device,
        )
        if mesh.n_nodes != canonical.n_nodes or mesh.n_elems != canonical.n_elems:
            raise ValueError("structured H8 metadata does not match mesh cardinality")
        coordinate_scale = max(float(canonical.nodes.abs().max().item()), 1.0)
        coordinate_tolerance = max(
            100.0 * torch.finfo(mesh.dtype).eps * coordinate_scale,
            1.0e-12,
        )
        if not torch.allclose(
            mesh.nodes,
            canonical.nodes,
            rtol=0.0,
            atol=coordinate_tolerance,
        ) or not torch.equal(mesh.elements, canonical.elements):
            raise ValueError(
                "periodic map requires canonical H8Mesh.structured_block node and "
                "connectivity ordering"
            )
        node_ids = torch.arange(mesh.n_nodes, dtype=torch.long, device=mesh.device)
        stride_x = nx + 1
        stride_z = (ny + 1) * stride_x
        i = node_ids % stride_x
        j = (node_ids // stride_x) % (ny + 1)
        k = node_ids // stride_z
        reduced = (i % nx) + nx * ((j % ny) + ny * (k % nz))
        return cls(
            reduced_node_of_full=reduced.contiguous(),
            n_reduced_nodes=nx * ny * nz,
        )

    @property
    def n_full_nodes(self) -> int:
        return int(self.reduced_node_of_full.numel())

    @property
    def n_reduced_dofs(self) -> int:
        return 3 * self.n_reduced_nodes

    def expand_fluctuation(self, reduced: torch.Tensor) -> torch.Tensor:
        """Apply the Boolean expansion P from reduced to full nodal DOFs."""

        if reduced.shape == (self.n_reduced_nodes, 3):
            reduced = reduced.reshape(-1)
        if reduced.shape != (self.n_reduced_dofs,):
            raise ValueError(
                f"reduced fluctuation must have shape ({self.n_reduced_dofs},), "
                f"got {tuple(reduced.shape)}"
            )
        full = reduced.reshape(self.n_reduced_nodes, 3)[self.reduced_node_of_full]
        return full.reshape(-1)

    def restrict_force(self, full: torch.Tensor) -> torch.Tensor:
        """Apply P^T by summing full nodal force contributions by representative."""

        if full.shape == (self.n_full_nodes, 3):
            full = full.reshape(-1)
        if full.shape != (3 * self.n_full_nodes,):
            raise ValueError(
                f"full force must have shape ({3 * self.n_full_nodes},), "
                f"got {tuple(full.shape)}"
            )
        result = torch.zeros(
            (self.n_reduced_nodes, 3), dtype=full.dtype, device=full.device
        )
        indices = self.reduced_node_of_full.unsqueeze(-1).expand(-1, 3)
        result.scatter_add_(0, indices, full.reshape(self.n_full_nodes, 3))
        return result.reshape(-1)

    def fluctuation_pin_mask(self, *, dtype_device: torch.device | None = None) -> torch.Tensor:
        """Fix one periodic fluctuation vector to eliminate rigid translation."""

        device = self.reduced_node_of_full.device if dtype_device is None else dtype_device
        fixed = torch.zeros(self.n_reduced_dofs, dtype=torch.bool, device=device)
        begin = 3 * self.pinned_reduced_node
        fixed[begin : begin + 3] = True
        return fixed
