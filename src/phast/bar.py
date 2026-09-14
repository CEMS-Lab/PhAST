"""Small-strain axial elasticity using PhAST's sparse autograd solver.

This independent 1D pathway has one displacement per node. It models a
uniform material and area, a prescribed left displacement and a right nodal
force. It does not route through the 2D fracture workflow.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .line_mesh import LineMesh
from .sparse_solve import SparseSolveAutograd


@dataclass(frozen=True)
class BarResult:
    """Nodal and element tensors, retaining first-order autograd support.

    ``reaction`` is the signed left-end reaction. ``free_residual`` contains
    equilibrium residuals at all other nodes. Stress and strain are constant
    within each line element. Units follow the supplied inputs.
    """

    displacement: torch.Tensor
    reaction: torch.Tensor
    free_residual: torch.Tensor
    strain: torch.Tensor
    stress: torch.Tensor
    axial_force: torch.Tensor


def _scalar(value: float | torch.Tensor, name: str, *, positive: bool = False) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.dtype != torch.float64:
            raise ValueError(f"{name} must be a CPU float64 tensor or a Python number")
        out = value
    else:
        out = torch.as_tensor(value, dtype=torch.float64)
    if out.ndim != 0 or not bool(torch.isfinite(out)):
        raise ValueError(f"{name} must be a finite scalar")
    if positive and not bool(out > 0):
        raise ValueError(f"{name} must be positive")
    return out


def solve_bar(
    mesh: LineMesh,
    young_modulus: float | torch.Tensor,
    area: float | torch.Tensor,
    end_force: float | torch.Tensor,
    *,
    left_displacement: float | torch.Tensor = 0.0,
) -> BarResult:
    """Assemble and solve a uniform axial bar in static equilibrium.

    Uses two-node linear elements and exact Dirichlet elimination, with
    PhAST's ``SparseSolveAutograd`` / SciPy SuperLU CPU backend. Scalars
    ``young_modulus`` and ``area`` must be positive. Positive ``end_force``
    pulls the right end in the positive coordinate direction. A nonzero
    prescribed left displacement adds a rigid translation.

    The supported scope is CPU float64, one load case per call and first-order
    derivatives on a fixed topology. Tensor inputs retain gradients through
    stiffness assembly, the sparse solve and the returned fields.
    """
    if not isinstance(mesh, LineMesh):
        raise TypeError("mesh must be a LineMesh")
    E = _scalar(young_modulus, "young_modulus", positive=True)
    A = _scalar(area, "area", positive=True)
    F = _scalar(end_force, "end_force")
    u0 = _scalar(left_displacement, "left_displacement")
    connectivity = mesh.elements
    coefficient = E * A / mesh.element_lengths
    i, j = connectivity.unbind(dim=1)
    rows = torch.stack((i, i, j, j), dim=1).reshape(-1)
    cols = torch.stack((i, j, i, j), dim=1).reshape(-1)
    indices = torch.stack((rows, cols))
    signs = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64)
    values = (coefficient[:, None] * signs).reshape(-1)
    n = mesh.nodes.numel()
    load = F * (torch.arange(n) == n - 1).to(torch.float64)
    keep = (rows > 0) & (cols > 0)
    free_u = SparseSolveAutograd.apply(indices[:, keep] - 1, values[keep], load[1:], n - 1)
    # K * 1 = 0: a constant prescribed translation does not change strain.
    u = torch.cat((torch.zeros(1, dtype=torch.float64), free_u)) + u0
    K = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    residual = torch.sparse.mm(K, u[:, None]).squeeze(1) - load
    strain = torch.diff(u) / mesh.element_lengths
    stress = E * strain
    return BarResult(u, residual[0], residual[1:], strain, stress, A * stress)
