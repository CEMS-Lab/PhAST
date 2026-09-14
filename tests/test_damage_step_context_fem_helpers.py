"""Tests for the architecture-neutral finite-element helpers on the context.

These helpers are shared by every learned-damage predictor, whatever its
architecture, so they are tested here rather than beside any one adapter.
"""
from __future__ import annotations

import pytest
import torch

from phast.learned_damage import DamageStepContext

DTYPE = torch.float64

# Unit square split into four triangles around a central node.
NODES = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], dtype=DTYPE)
ELEMENTS = torch.tensor(
    [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=torch.long)


def make_context(nodes=NODES, elements=ELEMENTS, *, history=1.0):
    n_nodes = nodes.shape[0]
    return DamageStepContext(
        step=1,
        time=0.0,
        load_factor=1.0,
        nodes=nodes,
        elements=elements,
        displacement=torch.zeros((n_nodes, 2), dtype=DTYPE),
        velocity=torch.zeros((n_nodes, 2), dtype=DTYPE),
        history_element=torch.full((elements.shape[0],), history, dtype=DTYPE),
        history_nodal=torch.zeros((n_nodes,), dtype=DTYPE),
        damage_previous=torch.zeros((n_nodes,), dtype=DTYPE),
        material={"Gc": 2.7, "l0": 0.4, "pf_model": "AT2"},
        phase_field_model="AT2",
        energy_split="spectral",
        device=torch.device("cpu"),
        dtype=DTYPE,
    )


def test_element_areas_sum_to_the_domain_area():
    assert float(make_context().element_areas().sum()) == pytest.approx(1.0)


def test_element_areas_are_orientation_independent():
    flipped = ELEMENTS[:, [0, 2, 1]]
    assert torch.allclose(
        make_context().element_areas(),
        make_context(elements=flipped).element_areas())


def test_assembly_conserves_the_integral():
    context = make_context()
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DTYPE)
    assembled = context.assemble_element_to_nodes(values)
    expected = float((values * context.element_areas()).sum())
    assert float(assembled.sum()) == pytest.approx(expected)


def test_assembly_of_a_unit_field_returns_the_lumped_mass():
    context = make_context()
    lumped = context.assemble_element_to_nodes(
        torch.ones(ELEMENTS.shape[0], dtype=DTYPE))
    # The central node belongs to every triangle, so it carries a third of
    # the whole domain area.
    assert float(lumped[4]) == pytest.approx(1.0 / 3.0)
    assert float(lumped.sum()) == pytest.approx(1.0)


def test_assembly_rejects_a_wrong_length():
    with pytest.raises(ValueError, match="one entry per element"):
        make_context().assemble_element_to_nodes(torch.zeros(2, dtype=DTYPE))


def test_boundary_mask_marks_the_perimeter_and_not_the_interior():
    assert make_context().boundary_node_mask().tolist() == [
        True, True, True, True, False]


def test_boundary_mask_detects_an_interior_void():
    # An annulus-like patch: a square ring of eight triangles around a hole.
    outer = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0],
         [2.0, 2.0], [1.0, 2.0], [0.0, 2.0], [0.0, 1.0]], dtype=DTYPE)
    inner = torch.tensor(
        [[0.75, 0.75], [1.25, 0.75], [1.25, 1.25], [0.75, 1.25]], dtype=DTYPE)
    nodes = torch.cat((outer, inner), dim=0)
    ring = torch.tensor(
        [[0, 1, 8], [1, 2, 9], [1, 9, 8], [2, 3, 9],
         [3, 4, 10], [3, 10, 9], [4, 5, 10], [5, 6, 11],
         [5, 11, 10], [6, 7, 11], [7, 0, 8], [7, 8, 11]], dtype=torch.long)
    mask = make_context(nodes, ring).boundary_node_mask()
    # Every node lies on either the outer perimeter or the hole.
    assert bool(mask.all())


def test_edge_lengths_match_the_geometry():
    context = make_context()
    edge_index = torch.tensor([[0, 0], [1, 4]], dtype=torch.long)
    lengths = context.edge_lengths(edge_index)
    assert float(lengths[0]) == pytest.approx(1.0)
    assert float(lengths[1]) == pytest.approx(0.5 * 2.0 ** 0.5)


def test_graph_edge_index_is_symmetric_and_duplicate_free():
    edge_index = make_context().graph_edge_index()
    pairs = {tuple(pair) for pair in edge_index.t().tolist()}
    assert len(pairs) == edge_index.shape[1]
    assert all((b, a) in pairs for a, b in pairs)


def test_canonical_node_features_have_six_columns():
    features = make_context().canonical_node_features()
    assert features.shape == (NODES.shape[0], 6)


def test_element_areas_reject_a_non_triangular_mesh():
    quads = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with pytest.raises(NotImplementedError, match="three-node triangles"):
        make_context(elements=quads).element_areas()
