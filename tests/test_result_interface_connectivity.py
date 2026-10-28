"""Row order of cohesive interface cells in ``/interfaces/<name>/connectivity``: the solver's ``top`` side first."""
from __future__ import annotations

import numpy as np
import pytest

from phast.io.result import interface_connectivity


class _Side:
    """Stand-in carrying only what the helper reads."""

    def __init__(self, top, bottom):
        self.nodes_top, self.nodes_bottom = top, bottom


def test_rows_list_top_then_bottom_reversed():
    rows = interface_connectivity([_Side((0, 1), (10, 11)), _Side((1, 2), (11, 12))])
    assert rows.dtype == np.int64 and rows.shape == (2, 4)
    assert rows.tolist() == [[0, 1, 11, 10], [1, 2, 12, 11]]          # top0, top1, bottom1, bottom0


def test_no_elements_gives_an_empty_table_of_four_columns():
    assert interface_connectivity([]).shape == (0, 4)


def test_three_dimensional_elements_are_refused():
    with pytest.raises(ValueError, match="2D elements"):
        interface_connectivity([_Side((0, 1, 2, 3), (4, 5, 6, 7))])


def test_first_side_is_the_side_the_normal_points_to_on_solver_elements():
    """Elements from ``insert_cohesive_layer``: the first side keeps the original node ids and lies on the normal's side."""
    from phast.cohesive_elements import insert_cohesive_layer
    nodes = np.array([[0, 0], [1, 0], [0, .5], [1, .5], [0, 1], [1, 1]], float)
    tris = np.array([[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4]])
    new_nodes, new_tris, cohesives = insert_cohesive_layer(nodes, tris, [(2, 3)])
    rows = interface_connectivity(cohesives)
    assert rows.shape == (1, 4) and set(rows[0, :2]) == {2, 3}                 # the first side keeps its original ids
    ce = cohesives[0]
    mid = new_nodes[list(ce.nodes_top)].mean(axis=0)
    side = lambda ids: np.mean([new_nodes[t].mean(axis=0) for t in new_tris if set(t) & set(ids)], axis=0)
    first = np.dot(ce.normal, side(rows[0, :2]) - mid)
    second = np.dot(ce.normal, side(rows[0, 2:]) - mid)
    assert first > 0.0 > second                                                # the normal points to the first side
    assert np.allclose(new_nodes[rows[0, 2:]][::-1], new_nodes[rows[0, :2]])   # zero thickness at rest
