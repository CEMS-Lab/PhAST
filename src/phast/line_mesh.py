"""One-dimensional meshes for axial line elements, independent of 2D FEMMesh."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import torch


@dataclass(frozen=True)
class LineMesh:
    """Ordered axial coordinates with adjacent two-node elements.

    Coordinates have shape ``(n_nodes,)`` and must increase strictly.
    Connectivity is derived from their order. CPU float64 is the reference
    representation for the accompanying :func:`phast.solve_bar` solver.
    Tensor coordinates retain their computational graph.
    """

    nodes: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, torch.Tensor):
            raise TypeError("nodes must be a torch.Tensor")
        if self.nodes.ndim != 1 or self.nodes.numel() < 2:
            raise ValueError("nodes must be a 1D tensor with at least two coordinates")
        if self.nodes.dtype != torch.float64 or self.nodes.device.type != "cpu":
            raise ValueError("LineMesh currently requires CPU float64 coordinates")
        if not bool(torch.isfinite(self.nodes).all()):
            raise ValueError("node coordinates must be finite")
        if not bool((torch.diff(self.nodes) > 0).all()):
            raise ValueError("node coordinates must increase strictly")

    @property
    def elements(self) -> torch.Tensor:
        """Connectivity of shape ``(n_elements, 2)``."""
        starts = torch.arange(self.nodes.numel() - 1, dtype=torch.long)
        return torch.stack((starts, starts + 1), dim=1)

    @property
    def element_lengths(self) -> torch.Tensor:
        """Positive lengths, retaining dependence on tensor coordinates."""
        return torch.diff(self.nodes)


def line_mesh(length: float, n_elements: int = 10, *, origin: float = 0.0) -> LineMesh:
    """Generate a uniform axial mesh without external meshing software.

    Length and origin use the caller's consistent unit system. For tensor
    coordinate sensitivities or a nonuniform mesh, construct ``LineMesh``
    directly from a CPU float64 coordinate tensor.
    """
    if isinstance(n_elements, bool) or not isinstance(n_elements, Integral):
        raise TypeError("n_elements must be an integer")
    if n_elements < 1:
        raise ValueError("n_elements must be at least one")
    if isinstance(length, torch.Tensor) or isinstance(origin, torch.Tensor):
        raise TypeError("Use LineMesh(nodes) for tensor-valued coordinates")
    if not math.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive")
    if not math.isfinite(origin):
        raise ValueError("origin must be finite")
    nodes = torch.linspace(origin, origin + length, n_elements + 1, dtype=torch.float64)
    return LineMesh(nodes)
