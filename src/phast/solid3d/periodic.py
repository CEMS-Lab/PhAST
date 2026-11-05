"""Periodic node map and reference voxel for periodic voxel cells.

Full nodes ``(i, j, k)`` with ``0 <= i <= nx`` (and likewise for ``j``, ``k``)
use the :meth:`phast.core.hex8.H8Mesh.structured_block` numbering
``i + (nx+1)*(j + (ny+1)*k)``.  The reduced (independent) node of a full node
maps every periodic axis index ``i`` to ``i % nx`` and keeps a non-periodic
axis index unchanged (range ``nx + 1``):

``reduced = i' + nx'*(j' + ny'*k')``,

where ``nx' = nx`` on a periodic axis and ``nx + 1`` otherwise.  A triply
periodic cell therefore has ``nx*ny*nz`` reduced nodes and a cell with
free ``z`` faces has ``nx*ny*(nz+1)``.

Every voxel of a :class:`~phast.solid3d.model.VoxelGrid` has
the same size, so one :class:`ReferenceVoxel` (2x2x2 Gauss rule, shape
values, gradients, weights, engineering-strain ``B`` matrices and the scalar
mass and stiffness matrices) serves every element.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from phast.core.voigt import PeriodicNodeMap
from phast.core.hex8 import H8Mesh, h8_B_matrices, h8_quadrature_geometry, h8_reference_nodes
from phast.solid3d.model import VoxelGrid, solver_dtype


_DTYPE = torch.float64


@dataclass(frozen=True, eq=False)
class PeriodicVoxelMap:
    """Boolean periodic reduction ``P`` of a structured voxel grid.

    ``reduced_shape`` is ``(nx', ny', nz')`` and ``n_reduced`` their product.
    ``reduced_of_full`` ``(N_full,)`` gives the reduced node of every full
    node.  ``element_nodes`` ``(E, 8)`` lists the reduced nodes of every voxel
    in the local H8 order of :func:`phast.core.hex8.h8_reference_nodes`,
    and ``element_dofs`` ``(E, 24)`` the matching vector dofs ``3*node + c``
    (node-major, as in the H8 ``B`` matrices).

    On a periodic axis with a single voxel two local nodes of one element map
    to the same reduced node; ``scatter_add_`` assembly stays exact.
    """

    grid: VoxelGrid
    reduced_shape: tuple[int, int, int]
    n_reduced: int
    reduced_of_full: torch.Tensor
    element_nodes: torch.Tensor
    element_dofs: torch.Tensor

    @classmethod
    def build(cls, grid: VoxelGrid, *, device: torch.device | str = "cpu") -> "PeriodicVoxelMap":
        """Build the map of ``grid`` on ``device``."""

        if not isinstance(grid, VoxelGrid):
            raise TypeError("grid must be a VoxelGrid")
        device = torch.device(device)
        nx, ny, nz = grid.shape
        px, py, pz = grid.periodic
        rx = nx if px else nx + 1
        ry = ny if py else ny + 1
        rz = nz if pz else nz + 1

        def reduce(i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
            ir = torch.remainder(i, nx) if px else i
            jr = torch.remainder(j, ny) if py else j
            kr = torch.remainder(k, nz) if pz else k
            return ir + rx * (jr + ry * kr)

        kk, jj, ii = torch.meshgrid(
            torch.arange(nz + 1, dtype=torch.long, device=device),
            torch.arange(ny + 1, dtype=torch.long, device=device),
            torch.arange(nx + 1, dtype=torch.long, device=device),
            indexing="ij",
        )
        reduced_of_full = reduce(ii, jj, kk).reshape(-1).contiguous()

        ke, je, ie = torch.meshgrid(
            torch.arange(nz, dtype=torch.long, device=device),
            torch.arange(ny, dtype=torch.long, device=device),
            torch.arange(nx, dtype=torch.long, device=device),
            indexing="ij",
        )
        offsets = ((h8_reference_nodes(dtype=_DTYPE, device=device) + 1.0) / 2.0).round().to(torch.long)
        element_nodes = reduce(
            ie.reshape(-1, 1) + offsets[:, 0],
            je.reshape(-1, 1) + offsets[:, 1],
            ke.reshape(-1, 1) + offsets[:, 2],
        ).contiguous()
        components = torch.arange(3, dtype=torch.long, device=device)
        element_dofs = (3 * element_nodes.unsqueeze(-1) + components).reshape(-1, 24).contiguous()
        return cls(
            grid=grid,
            reduced_shape=(rx, ry, rz),
            n_reduced=rx * ry * rz,
            reduced_of_full=reduced_of_full,
            element_nodes=element_nodes,
            element_dofs=element_dofs,
        )

    @property
    def n_full(self) -> int:
        """Number of full grid nodes ``(nx+1)(ny+1)(nz+1)``."""

        return int(self.reduced_of_full.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.element_nodes.shape[0])

    @property
    def device(self) -> torch.device:
        return self.reduced_of_full.device

    def expand(self, red: torch.Tensor) -> torch.Tensor:
        """Apply ``P``: reduced ``(n,)``, ``(n, 3)`` or flat ``(3n,)`` to full nodes.

        The output has the matching full shape ``(N,)``, ``(N, 3)`` or ``(3N,)``.
        """

        n = self.n_reduced
        if red.shape == (n,):
            return red[self.reduced_of_full]
        if red.shape == (n, 3):
            return red[self.reduced_of_full]
        if red.shape == (3 * n,):
            return red.reshape(n, 3)[self.reduced_of_full].reshape(-1)
        raise ValueError(
            f"reduced field must have shape ({n},), ({n}, 3) or ({3 * n},), got {tuple(red.shape)}"
        )

    def restrict(self, full: torch.Tensor) -> torch.Tensor:
        """Apply ``P^T`` (sum over periodic images) with ``scatter_add_``.

        Accepts full ``(N,)``, ``(N, 3)`` or flat ``(3N,)`` fields and returns
        the matching reduced shape.
        """

        n_full = self.n_full
        if full.shape == (n_full,):
            out = torch.zeros(self.n_reduced, dtype=full.dtype, device=full.device)
            out.scatter_add_(0, self.reduced_of_full, full)
            return out
        flat = full.shape == (3 * n_full,)
        if full.shape == (n_full, 3) or flat:
            values = full.reshape(n_full, 3)
            out = torch.zeros((self.n_reduced, 3), dtype=full.dtype, device=full.device)
            out.scatter_add_(0, self.reduced_of_full.unsqueeze(-1).expand(-1, 3), values)
            return out.reshape(-1) if flat else out
        raise ValueError(
            f"full field must have shape ({n_full},), ({n_full}, 3) or ({3 * n_full},), "
            f"got {tuple(full.shape)}"
        )

    def as_node_map(self, *, pinned_node: int = 0) -> PeriodicNodeMap:
        """Return the equivalent :class:`~phast.core.voigt.PeriodicNodeMap`."""

        return PeriodicNodeMap(
            reduced_node_of_full=self.reduced_of_full,
            n_reduced_nodes=self.n_reduced,
            pinned_reduced_node=int(pinned_node),
        )


@dataclass(frozen=True, eq=False)
class ReferenceVoxel:
    """Quadrature data of one voxel, shared by every element of a grid.

    ``N`` is ``(8 q, 8 a)``, ``grad_N`` ``(8 q, 8 a, 3)``, ``w`` ``(8,)`` the
    weighted Jacobian determinants, ``B`` ``(8, 6, 24)`` the engineering-strain
    matrices, ``M = sum_q w N N^T`` and ``L = sum_q w gradN gradN^T``.
    """

    h: tuple[float, float, float]
    N: torch.Tensor
    grad_N: torch.Tensor
    w: torch.Tensor
    B: torch.Tensor
    M: torch.Tensor
    L: torch.Tensor
    volume: float


def reference_voxel(
    h: Sequence[float],
    *,
    dtype: torch.dtype = _DTYPE,
    device: torch.device | str = "cpu",
) -> ReferenceVoxel:
    """Return the :class:`ReferenceVoxel` of a voxel with edge lengths ``h``."""

    dtype = solver_dtype(dtype)
    hx, hy, hz = (float(v) for v in h)
    mesh = H8Mesh.structured_block(
        length=hx, height=hy, thickness=hz, nx=1, ny=1, nz=1, dtype=dtype, device=device
    )
    N, grad, wdet = h8_quadrature_geometry(mesh.nodes, mesh.elements)
    B = h8_B_matrices(grad)[0]
    grad_N = grad[0]
    w = wdet[0]
    M = torch.einsum("q,qa,qb->ab", w, N, N)
    L = torch.einsum("q,qaj,qbj->ab", w, grad_N, grad_N)
    return ReferenceVoxel(
        h=(hx, hy, hz),
        N=N.contiguous(),
        grad_N=grad_N.contiguous(),
        w=w.contiguous(),
        B=B.contiguous(),
        M=0.5 * (M + M.T),
        L=0.5 * (L + L.T),
        volume=hx * hy * hz,
    )


def choose_pinned_node(pmap: PeriodicVoxelMap, phase_flat: Any, pore_id: int | None) -> int:
    """Return the first reduced node none of whose incident voxels has ``pore_id``.

    ``phase_flat`` holds one phase id per element (numpy or torch).  With
    ``pore_id=None`` node 0 is returned.  When every reduced node touches a
    pore voxel (possible only on very small or very porous cells) the first
    node with the fewest incident pore voxels is returned instead: the pin
    only removes the rigid translation and the pore phase keeps a positive
    stiffness, so any node is admissible.
    """

    if pore_id is None:
        return 0
    if isinstance(phase_flat, torch.Tensor):
        phase = phase_flat.to(device=pmap.device, dtype=torch.long).reshape(-1)
    else:
        phase = torch.as_tensor(np.asarray(phase_flat).astype(np.int64).reshape(-1), device=pmap.device)
    if phase.shape[0] != pmap.n_elements:
        raise ValueError(
            f"phase_flat must have {pmap.n_elements} entries, got {phase.shape[0]}"
        )
    pore_nodes = pmap.element_nodes[phase == int(pore_id)].reshape(-1)
    count = torch.zeros(pmap.n_reduced, dtype=torch.long, device=pmap.device)
    count.scatter_add_(0, pore_nodes, torch.ones_like(pore_nodes))
    # argmin returns the first minimiser, i.e. the first pore-free node if any.
    return int(torch.argmin(count))


__all__ = [
    "PeriodicVoxelMap",
    "ReferenceVoxel",
    "choose_pinned_node",
    "reference_voxel",
]
