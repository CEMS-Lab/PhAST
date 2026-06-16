"""Basic cohesive zone topology helpers.

Implements:
- :class:`CohesiveElement` dataclass — connectivity + local frame for a single
  zero-thickness 4-noded interface element (2 top + 2 bottom nodes).
- :func:`build_cohesive_strip` — duplicates the nodes along a Physical Line and
  returns the side-data; does NOT mutate the input mesh.
- :func:`cohesive_traction` — legacy exponential traction-separation helper.

The solver-coupled bilinear cohesive law lives in ``operator.py``.

References
----------
Issue #261, epic #259, design doc commit ae39668 (PF-CZM contrast).
Park, Paulino, Roesler (2009) Mech. Mater. 41:1109 — full TSL suite is #350.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class CohesiveElement:
    """Side-data record for a single zero-thickness 4-noded interface element."""

    nodes_top: Tuple[int, int]
    nodes_bottom: Tuple[int, int]
    normal: np.ndarray
    tangent: np.ndarray
    length: float
    gauss_points: np.ndarray = field(
        default_factory=lambda: np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    )


def _edges_on_physical_line(mesh, line_id: int) -> List[Tuple[int, int]]:
    """Best-effort extractor: find edges (node-pair) tagged with ``line_id``.

    This helper accepts any object exposing either ``physical_lines`` (dict
    mapping id -> list of node-pair tuples) or a meshio-style ``cell_sets``.
    """

    if hasattr(mesh, "physical_lines"):
        return list(mesh.physical_lines[line_id])
    if hasattr(mesh, "cell_sets") and hasattr(mesh, "cells_dict"):
        lines = mesh.cells_dict.get("line", np.empty((0, 2), dtype=int))
        mask = mesh.cell_sets[line_id]
        return [tuple(lines[i]) for i in mask]
    raise TypeError(
        "build_cohesive_strip: mesh must expose `physical_lines` or "
        "meshio-style `cell_sets`/`cells_dict`."
    )


def build_cohesive_strip(mesh, line_id: int) -> Tuple[List[CohesiveElement], dict]:
    """Build interface elements along a Physical Line; return (elements, node_map).

    ``node_map`` maps original-node-index -> duplicated (bottom-side) node index.
    The mesh is **not** mutated; use ``insert_cohesive_layer`` when the
    doubled-node mesh connectivity is needed.
    """

    edges = _edges_on_physical_line(mesh, line_id)
    if not edges:
        return [], {}

    coords = np.asarray(mesh.points)[:, :2]
    interface_nodes = sorted({n for e in edges for n in e})
    next_id = int(coords.shape[0])
    node_map = {orig: next_id + i for i, orig in enumerate(interface_nodes)}

    elements: List[CohesiveElement] = []
    for n0, n1 in edges:
        p0, p1 = coords[n0], coords[n1]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length == 0.0:
            continue
        tangent = edge / length
        normal = np.array([-tangent[1], tangent[0]])
        elements.append(
            CohesiveElement(
                nodes_top=(int(n0), int(n1)),
                nodes_bottom=(node_map[n0], node_map[n1]),
                normal=normal,
                tangent=tangent,
                length=length,
            )
        )
    return elements, node_map


def cohesive_traction(
    jump: Sequence[float],
    max_jump: float,
    sigma_max: float,
    mode: str = "exponential",
) -> np.ndarray:
    """Legacy exponential cohesive TSL helper.

    Parameters
    ----------
    jump : (delta_n, delta_t) — normal and tangential displacement jumps.
    max_jump : characteristic opening ``delta_c`` controlling the decay scale.
    sigma_max : peak normal traction at ``delta_n = max_jump``.
    mode : only ``'exponential'`` (Xu-Needleman-style) is implemented here;
        the bilinear / PPR / Camanho suite is tracked under #350.

    Returns ``(t_n, t_t)``. Compression (``delta_n <= 0``) returns zero for
    this legacy helper; contact is handled by ``BilinearCohesiveLaw``.
    """

    if mode != "exponential":
        raise NotImplementedError(
            f"TSL mode {mode!r} is not implemented here; full suite is #350."
        )
    delta_n, delta_t = float(jump[0]), float(jump[1])
    if max_jump <= 0.0:
        raise ValueError("max_jump (delta_c) must be positive")
    if delta_n <= 0.0:
        return np.zeros(2)
    # Xu-Needleman exponential: t = sigma_max * (delta/delta_c) * exp(1 - delta/delta_c)
    r_n = delta_n / max_jump
    r_t = delta_t / max_jump
    t_n = sigma_max * r_n * np.exp(1.0 - r_n)
    t_t = sigma_max * r_t * np.exp(1.0 - abs(r_t))
    return np.array([t_n, t_t])
