"""Cell types of layout 2.0: VTK identifiers, nodes per cell and cell edges.

Node order inside a cell is VTK order for every type. Edges follow the cell
boundary; for quadratic cells each geometric edge is split at its mid-side
node, so the graph connects neighbouring nodes only.
"""
from __future__ import annotations

import numpy as np

from .schema import load_schema

_EDGES = {
    "T3": [(0, 1), (1, 2), (2, 0)],
    "T6": [(0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0)],
    "Q4": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "Q8": [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0)],
    "Q9": [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0), (4, 8), (5, 8), (6, 8), (7, 8)],
    "TET4": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
    "TET10": [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 0), (0, 7), (7, 3), (1, 8), (8, 3), (2, 9), (9, 3)],
    "H8": [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)],
}
NODES_PER_CELL = {"T3": 3, "T6": 6, "Q4": 4, "Q8": 8, "Q9": 9, "TET4": 4, "TET10": 10, "H8": 8}


def cell_info(cell_type: str) -> dict:
    """Schema entry of a cell type (``vtk``, ``dim``, ``quadrature_points``) plus nodes per cell."""
    types = load_schema()["cell_types"]
    if cell_type not in types or cell_type not in NODES_PER_CELL:
        hint = ("; one-dimensional runs (line cells) are outside layout 2.x"
                if str(cell_type).upper().startswith(("LINE", "BAR", "EDGE", "L2", "L3")) else "")
        raise ValueError(f"unknown cell type {cell_type!r}; known: {', '.join(NODES_PER_CELL)}{hint}")
    return dict(types[cell_type], nodes=NODES_PER_CELL[cell_type])


def edge_index(blocks: dict[str, np.ndarray]) -> np.ndarray:
    """Graph edges (2, n_edges), both directions, sorted, from the edges of every cell of every block."""
    pairs = []
    for cell_type, cells in blocks.items():
        cells = np.asarray(cells, np.int64)
        for a, b in _EDGES[cell_type]:
            pairs.append(cells[:, [a, b]])
    if not pairs:
        return np.zeros((2, 0), np.int64)
    e = np.concatenate(pairs)
    e = np.concatenate([e, e[:, ::-1]])
    return np.unique(e, axis=0).T.copy()
