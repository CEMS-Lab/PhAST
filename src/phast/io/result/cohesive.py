"""Connectivity of cohesive interface blocks (``/interfaces/<name>/connectivity``).

An interface cell lists the nodes of one side first and the nodes of the other side after them. The first side is the
side the solver calls ``top`` (``CohesiveElement.nodes_top``): the side to which the interface normal
``n = (-t_y, t_x)`` of the tangent ``t`` points, whose nodes keep their original ids when a cohesive layer is inserted
(``phast.cohesive_elements.insert_cohesive_layer``). The second side is ``bottom``. At rest the two sides coincide; a
drawing of the interface uses the first side, which follows its own displacement when the interface opens.

2D cell ``COH2`` (four entries): ``top0, top1, bottom1, bottom0``, a ring around the zero-thickness quadrilateral.
3D cell ``COH4`` (eight entries): the four nodes of the top face, then the four nodes of the bottom face in the same order.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def interface_connectivity(cohesives: Sequence) -> np.ndarray:
    """Rows of ``/interfaces/<name>/connectivity`` for 2D cohesive elements (``nodes_top`` and ``nodes_bottom``, two nodes each).

    The order is ``top0, top1, bottom1, bottom0``; ``cell_type`` of the block is ``COH2``.
    """
    rows = []
    for k, ce in enumerate(cohesives):
        top, bottom = tuple(ce.nodes_top), tuple(ce.nodes_bottom)
        if len(top) != 2 or len(bottom) != 2:
            raise ValueError(f"cohesive element {k}: interface_connectivity handles 2D elements (two nodes per side), "
                             f"got {len(top)} and {len(bottom)}")
        rows.append([top[0], top[1], bottom[1], bottom[0]])
    return np.asarray(rows, np.int64).reshape(len(rows), 4)
