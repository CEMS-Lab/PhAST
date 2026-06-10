"""Scaffold tests for cohesive_elements (issue #261).

Mesh-integration / staggered-coupling tests are deferred (see #259 epic).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from phast.cohesive_elements import (
    BilinearCohesiveLaw,
    CohesiveElement,
    CohesiveInterfaceOperator,
    CohesiveState,
    build_cohesive_strip,
    cohesive_traction,
)


def _two_element_mesh_with_horizontal_edge():
    # Four nodes; horizontal edge (0, 1) marked as cohesive.
    pts = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, -1.0]], dtype=float
    )
    mesh = SimpleNamespace(points=pts, physical_lines={42: [(0, 1)]})
    return mesh


def test_build_strip_doubles_nodes():
    mesh = _two_element_mesh_with_horizontal_edge()
    elems, node_map = build_cohesive_strip(mesh, line_id=42)
    assert len(elems) == 1
    assert len(node_map) == 2  # two original nodes duplicated
    e = elems[0]
    assert e.nodes_top == (0, 1)
    assert set(e.nodes_bottom) == set(node_map.values())
    assert e.length == pytest.approx(1.0)
    np.testing.assert_allclose(e.tangent, [1.0, 0.0])
    np.testing.assert_allclose(e.normal, [0.0, 1.0])
    assert all(n >= mesh.points.shape[0] for n in e.nodes_bottom)


def test_build_strip_empty_line_id():
    mesh = SimpleNamespace(points=np.zeros((1, 2)), physical_lines={7: []})
    elems, nmap = build_cohesive_strip(mesh, line_id=7)
    assert elems == [] and nmap == {}


def test_cohesive_traction_zero_jump():
    t = cohesive_traction(jump=(0.0, 0.0), max_jump=1e-3, sigma_max=10.0)
    np.testing.assert_allclose(t, [0.0, 0.0])


def test_cohesive_traction_compression_is_zero():
    t = cohesive_traction(jump=(-1e-4, 0.0), max_jump=1e-3, sigma_max=10.0)
    np.testing.assert_allclose(t, [0.0, 0.0])


def test_cohesive_traction_peak_at_delta_c():
    sigma_max, dc = 10.0, 1e-3
    t = cohesive_traction(jump=(dc, 0.0), max_jump=dc, sigma_max=sigma_max)
    assert t[0] == pytest.approx(sigma_max)


def test_cohesive_traction_exponential_decay():
    sigma_max, dc = 10.0, 1e-3
    t_peak = cohesive_traction(jump=(dc, 0.0), max_jump=dc, sigma_max=sigma_max)
    t_far = cohesive_traction(jump=(5.0 * dc, 0.0), max_jump=dc, sigma_max=sigma_max)
    assert t_far[0] < 0.2 * t_peak[0]
    assert t_far[0] > 0.0


def test_cohesive_traction_unknown_mode_raises():
    with pytest.raises(NotImplementedError):
        cohesive_traction(jump=(1e-4, 0.0), max_jump=1e-3, sigma_max=10.0, mode="ppr")


# --- Mesh-integration layer (issue #261 follow-up to #390) ----------------------

def _two_strip_mesh():
    r"""4-row x 1-col grid of triangles split by a horizontal interior edge.

    Nodes:
        2 - 3      y = +1
        |\ |
        | \|
        0 - 1      y = 0   <-- interface
        |\ |
        | \|
        4 - 5      y = -1
    """
    nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0],
         [0.0, 1.0], [1.0, 1.0],
         [0.0, -1.0], [1.0, -1.0]],
        dtype=float,
    )
    elements = np.array(
        [[0, 1, 3], [0, 3, 2],   # top strip
         [4, 1, 0], [4, 5, 1]],  # bottom strip
        dtype=int,
    )
    interface_edges = [(0, 1)]
    return nodes, elements, interface_edges


def _two_strip_quad_mesh():
    r"""Two Q4 elements split by a horizontal interior edge."""
    nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0],
         [0.0, 1.0], [1.0, 1.0],
         [0.0, -1.0], [1.0, -1.0]],
        dtype=float,
    )
    elements = np.array(
        [[0, 1, 3, 2],   # top quad
         [4, 5, 1, 0]],  # bottom quad
        dtype=int,
    )
    interface_edges = [(0, 1)]
    return nodes, elements, interface_edges


def test_insert_cohesive_layer_node_count():
    from phast.cohesive_elements import insert_cohesive_layer

    nodes, elements, edges = _two_strip_mesh()
    new_nodes, new_elements, cohesives = insert_cohesive_layer(
        nodes, elements, edges
    )
    n_line_nodes = len({n for e in edges for n in e})
    assert new_nodes.shape[0] == nodes.shape[0] + n_line_nodes
    assert new_elements.shape == elements.shape
    assert len(cohesives) == 1
    # exactly one side should have been rewritten — pick whichever, but it
    # must be one full strip (2 elements) referencing both duplicate IDs
    rewritten = (new_elements >= nodes.shape[0]).any(axis=1).sum()
    assert rewritten == 2


def test_insert_cohesive_layer_supports_q4_node_doubling():
    from phast.cohesive_elements import insert_cohesive_layer

    nodes, elements, edges = _two_strip_quad_mesh()
    new_nodes, new_elements, cohesives = insert_cohesive_layer(
        nodes, elements, edges)

    assert new_nodes.shape[0] == nodes.shape[0] + 2
    assert new_elements.shape == elements.shape
    assert len(cohesives) == 1
    assert cohesives[0].nodes_top == (0, 1)
    assert cohesives[0].nodes_bottom == (6, 7)
    rewritten = (new_elements >= nodes.shape[0]).any(axis=1)
    assert rewritten.sum() == 1
    assert set(new_elements[rewritten][0]).issuperset({6, 7})


def test_insert_cohesive_layer_edge_orientation_invariant():
    from phast.cohesive_elements import insert_cohesive_layer

    nodes, elements, edges = _two_strip_mesh()
    out_fwd = insert_cohesive_layer(nodes, elements, edges)
    out_rev = insert_cohesive_layer(nodes, elements, [(1, 0)])

    np.testing.assert_allclose(out_fwd[0], out_rev[0])
    np.testing.assert_array_equal(out_fwd[1], out_rev[1])
    assert out_fwd[2][0].nodes_top == out_rev[2][0].nodes_top
    assert out_fwd[2][0].nodes_bottom == out_rev[2][0].nodes_bottom
    np.testing.assert_allclose(out_fwd[2][0].normal, out_rev[2][0].normal)


def test_insert_cohesive_layer_q4_edge_orientation_invariant():
    from phast.cohesive_elements import insert_cohesive_layer

    nodes, elements, edges = _two_strip_quad_mesh()
    out_fwd = insert_cohesive_layer(nodes, elements, edges)
    out_rev = insert_cohesive_layer(nodes, elements, [(1, 0)])

    np.testing.assert_allclose(out_fwd[0], out_rev[0])
    np.testing.assert_array_equal(out_fwd[1], out_rev[1])
    assert out_fwd[2][0].nodes_top == out_rev[2][0].nodes_top
    assert out_fwd[2][0].nodes_bottom == out_rev[2][0].nodes_bottom


def test_insert_cohesive_layer_no_mutation_of_input():
    from phast.cohesive_elements import insert_cohesive_layer

    nodes, elements, edges = _two_strip_mesh()
    nodes_id = id(nodes)
    elem_id = id(elements)
    nodes_snapshot = nodes.copy()
    elem_snapshot = elements.copy()
    insert_cohesive_layer(nodes, elements, edges)
    assert id(nodes) == nodes_id and id(elements) == elem_id
    np.testing.assert_array_equal(nodes, nodes_snapshot)
    np.testing.assert_array_equal(elements, elem_snapshot)


def test_insert_cohesive_layer_zero_jump():
    """In zero-displacement state, traction across cohesive elements is zero."""
    from phast.cohesive_elements import (
        cohesive_traction,
        insert_cohesive_layer,
    )

    nodes, elements, edges = _two_strip_mesh()
    new_nodes, _, cohesives = insert_cohesive_layer(nodes, elements, edges)
    # zero displacement -> top and bottom coords coincide -> jump = 0
    for ce in cohesives:
        for nt, nb in zip(ce.nodes_top, ce.nodes_bottom):
            jump_vec = new_nodes[nb] - new_nodes[nt]
            assert np.linalg.norm(jump_vec) == 0.0
        # exponential TSL returns 0 on non-positive normal jump
        t = cohesive_traction(jump=(0.0, 0.0), max_jump=1e-3, sigma_max=1.0)
        assert np.allclose(t, 0.0)


def test_insert_cohesive_layer_with_metadata_preserves_external_mesh_sets():
    from phast.cohesive_elements import (
        CohesiveInsertionResult,
        insert_cohesive_layer_with_metadata,
    )

    nodes, elements, edges = _two_strip_mesh()
    result = insert_cohesive_layer_with_metadata(
        nodes,
        elements,
        edges,
        node_sets={
            "interface": np.array([0, 1]),
            "top_boundary": np.array([2, 3]),
            "mixed_boundary": np.array([0, 2]),
        },
        element_sets={
            "top_material": np.array([0, 1]),
            "bottom_material": np.array([2, 3]),
        },
        element_data={
            "material_id": np.array([10, 10, 20, 20]),
            "thickness": np.ones(4),
        },
    )

    assert isinstance(result, CohesiveInsertionResult)
    assert result.nodes.shape[0] == nodes.shape[0] + 2
    assert result.elements.shape == elements.shape
    assert len(result.cohesives) == 1
    assert result.duplicate_node_map == {0: 6, 1: 7}
    assert set(result.node_sets["interface"]) == {0, 1, 6, 7}
    assert set(result.node_sets["interface_top"]) == {0, 1}
    assert set(result.node_sets["interface_bottom"]) == {6, 7}
    assert set(result.node_sets["mixed_boundary"]) == {0, 2, 6}
    assert set(result.node_sets["mixed_boundary_top"]) == {0}
    assert set(result.node_sets["mixed_boundary_bottom"]) == {6}
    assert set(result.node_sets["cohesive_interface_top"]) == {0, 1}
    assert set(result.node_sets["cohesive_interface_bottom"]) == {6, 7}
    np.testing.assert_array_equal(result.element_sets["top_material"], [0, 1])
    np.testing.assert_array_equal(result.element_sets["bottom_material"], [2, 3])
    np.testing.assert_array_equal(result.element_data["material_id"], [10, 10, 20, 20])
    np.testing.assert_array_equal(result.element_data["thickness"], np.ones(4))
    assert np.count_nonzero(result.element_side == -1) == 2


def test_insert_cohesive_layer_q4_with_metadata_preserves_sets():
    from phast.cohesive_elements import (
        CohesiveInsertionResult,
        insert_cohesive_layer_with_metadata,
    )

    nodes, elements, edges = _two_strip_quad_mesh()
    result = insert_cohesive_layer_with_metadata(
        nodes,
        elements,
        edges,
        node_sets={"interface": [0, 1], "top": [2, 3]},
        element_sets={"top_material": [0], "bottom_material": [1]},
        element_data={"material_id": [10, 20]},
    )

    assert isinstance(result, CohesiveInsertionResult)
    assert result.nodes.shape[0] == nodes.shape[0] + 2
    assert result.elements.shape == elements.shape
    assert result.duplicate_node_map == {0: 6, 1: 7}
    assert set(result.node_sets["interface"]) == {0, 1, 6, 7}
    assert set(result.node_sets["interface_top"]) == {0, 1}
    assert set(result.node_sets["interface_bottom"]) == {6, 7}
    np.testing.assert_array_equal(result.element_sets["top_material"], [0])
    np.testing.assert_array_equal(result.element_sets["bottom_material"], [1])
    np.testing.assert_array_equal(result.element_data["material_id"], [10, 20])
    assert np.count_nonzero(result.element_side == -1) == 1


def test_insert_cohesive_layer_with_metadata_validates_bad_external_mesh_metadata():
    from phast.cohesive_elements import insert_cohesive_layer_with_metadata

    nodes, elements, edges = _two_strip_mesh()
    with pytest.raises(ValueError, match="out-of-range node"):
        insert_cohesive_layer_with_metadata(
            nodes, elements, edges, node_sets={"bad": [999]})
    with pytest.raises(ValueError, match="out-of-range element"):
        insert_cohesive_layer_with_metadata(
            nodes, elements, edges, element_sets={"bad": [999]})
    with pytest.raises(ValueError, match="first dimension"):
        insert_cohesive_layer_with_metadata(
            nodes, elements, edges, element_data={"material_id": [1, 2]})
    with pytest.raises(ValueError, match="duplicate cohesive edge"):
        insert_cohesive_layer_with_metadata(nodes, elements, [(0, 1), (1, 0)])
    with pytest.raises(ValueError, match="T3 triangle or Q4 quad"):
        insert_cohesive_layer_with_metadata(
            nodes, np.array([[0, 1, 2, 3, 4]], dtype=int), edges)


def test_insert_cohesive_layer_meshio_preserves_format_metadata(tmp_path):
    import meshio

    from phast.cohesive_elements import (
        MeshIOCohesiveInsertionResult,
        insert_cohesive_layer_meshio,
    )

    nodes, elements, _edges = _two_strip_mesh()
    points = np.column_stack([nodes, np.zeros(nodes.shape[0])])
    lines = np.array([[0, 1], [2, 3]], dtype=int)
    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", elements), ("line", np.array([[1, 0], [2, 3]], dtype=int))],
        point_data={"temperature": np.arange(nodes.shape[0], dtype=float)},
        cell_data={
            "material_id": [
                np.array([10, 10, 20, 20], dtype=int),
                np.array([0, 0], dtype=int),
            ]
        },
        point_sets={
            "interface": np.array([0, 1], dtype=int),
            "top_boundary": np.array([2, 3], dtype=int),
        },
        cell_sets={
            "cohesive_interface": [
                np.array([], dtype=int),
                np.array([0], dtype=int),
            ],
            "top_bulk": [
                np.array([0, 1], dtype=int),
                np.array([], dtype=int),
            ],
        },
    )

    result = insert_cohesive_layer_meshio(
        mesh, interface_set="cohesive_interface")

    assert isinstance(result, MeshIOCohesiveInsertionResult)
    assert result.triangle_block_index == 0
    assert result.interface_edges == [(0, 1)]
    assert result.insertion.duplicate_node_map == {0: 6, 1: 7}
    assert result.mesh.points.shape == (8, 3)
    assert result.mesh.cells[0].type == "triangle"
    assert result.mesh.cells[1].type == "line"
    assert result.mesh.cells[0].data.shape == elements.shape
    assert (result.mesh.cells[0].data >= nodes.shape[0]).any(axis=1).sum() == 2
    assert set(result.mesh.point_sets["interface"]) == {0, 1, 6, 7}
    assert set(result.mesh.point_sets["interface_top"]) == {0, 1}
    assert set(result.mesh.point_sets["interface_bottom"]) == {6, 7}
    assert set(result.mesh.point_sets["cohesive_interface_top"]) == {0, 1}
    assert set(result.mesh.point_sets["cohesive_interface_bottom"]) == {6, 7}
    np.testing.assert_array_equal(
        result.mesh.point_data["temperature"][-2:], [0.0, 1.0])
    np.testing.assert_array_equal(
        result.mesh.cell_data["material_id"][0], [10, 10, 20, 20])
    np.testing.assert_array_equal(
        result.mesh.cell_data["material_id"][1], [0, 0])
    np.testing.assert_array_equal(
        result.mesh.cell_sets["top_bulk"][0], [0, 1])

    out = tmp_path / "cohesive_mesh.vtu"
    meshio.write(out, result.mesh)
    loaded = meshio.read(out)
    assert loaded.points.shape == (8, 3)
    assert loaded.cells_dict["triangle"].shape == elements.shape
    assert "temperature" in loaded.point_data


def test_insert_cohesive_layer_meshio_supports_q4_metadata(tmp_path):
    import meshio

    from phast.cohesive_elements import insert_cohesive_layer_meshio

    nodes, elements, _edges = _two_strip_quad_mesh()
    points = np.column_stack([nodes, np.zeros(nodes.shape[0])])
    lines = np.array([[1, 0]], dtype=int)
    mesh = meshio.Mesh(
        points=points,
        cells=[("quad", elements), ("line", lines)],
        point_data={"temperature": np.arange(nodes.shape[0], dtype=float)},
        cell_data={
            "material_id": [
                np.array([10, 20], dtype=int),
                np.array([0], dtype=int),
            ]
        },
        point_sets={"interface": np.array([0, 1], dtype=int)},
        cell_sets={
            "cohesive_interface": [
                np.array([], dtype=int),
                np.array([0], dtype=int),
            ],
            "top_bulk": [
                np.array([0], dtype=int),
                np.array([], dtype=int),
            ],
        },
    )

    result = insert_cohesive_layer_meshio(
        mesh, interface_set="cohesive_interface")

    assert result.triangle_block_index == 0
    assert result.cell_type == "quad"
    assert result.interface_edges == [(0, 1)]
    assert result.mesh.points.shape == (8, 3)
    assert result.mesh.cells[0].type == "quad"
    assert result.mesh.cells[0].data.shape == elements.shape
    assert set(result.mesh.point_sets["interface"]) == {0, 1, 6, 7}
    np.testing.assert_array_equal(
        result.mesh.point_data["temperature"][-2:], [0.0, 1.0])
    np.testing.assert_array_equal(
        result.mesh.cell_data["material_id"][0], [10, 20])
    np.testing.assert_array_equal(result.mesh.cell_sets["top_bulk"][0], [0])

    out = tmp_path / "cohesive_q4_mesh.vtu"
    meshio.write(out, result.mesh)
    loaded = meshio.read(out)
    assert loaded.points.shape == (8, 3)
    assert loaded.cells_dict["quad"].shape == elements.shape


def test_insert_cohesive_layer_meshio_validates_interface_source():
    import meshio

    from phast.cohesive_elements import insert_cohesive_layer_meshio

    nodes, elements, edges = _two_strip_mesh()
    points = np.column_stack([nodes, np.zeros(nodes.shape[0])])
    mesh = meshio.Mesh(
        points=points,
        cells=[
            ("triangle", elements[:2]),
            ("triangle", elements[2:]),
            ("line", np.array([[0, 1]], dtype=int)),
        ],
        cell_sets={
            "cohesive_interface": [
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([0], dtype=int),
            ]
        },
    )

    with pytest.raises(ValueError, match="not both"):
        insert_cohesive_layer_meshio(
            mesh, interface_edges=edges, interface_set="cohesive_interface")
    with pytest.raises(ValueError, match="multiple T3/Q4 bulk blocks"):
        insert_cohesive_layer_meshio(mesh, interface_set="cohesive_interface")
    with pytest.raises(ValueError, match="not found"):
        insert_cohesive_layer_meshio(
            mesh, interface_set="missing", triangle_block_index=0)
    with pytest.raises(ValueError, match="T3 triangle or Q4 quad cell block"):
        insert_cohesive_layer_meshio(
            mesh, interface_set="cohesive_interface", triangle_block_index=2)
    with pytest.raises(ValueError, match="disagree"):
        insert_cohesive_layer_meshio(
            mesh,
            interface_set="cohesive_interface",
            triangle_block_index=0,
            cell_block_index=1,
        )


def test_insert_cohesive_layer_meshio_validates_metadata_lengths():
    import meshio

    from phast.cohesive_elements import insert_cohesive_layer_meshio

    nodes, elements, edges = _two_strip_mesh()
    points = np.column_stack([nodes, np.zeros(nodes.shape[0])])
    bad_point_data = meshio.Mesh(
        points=points,
        cells=[("triangle", elements)],
        point_data={"temperature": np.arange(nodes.shape[0])},
    )
    bad_point_data.point_data["temperature"] = np.arange(nodes.shape[0] - 1)
    with pytest.raises(ValueError, match="point_data 'temperature'"):
        insert_cohesive_layer_meshio(bad_point_data, interface_edges=edges)

    bad_cell_data = meshio.Mesh(
        points=points,
        cells=[("triangle", elements)],
        cell_data={"material_id": [np.arange(elements.shape[0], dtype=int)]},
    )
    bad_cell_data.cell_data["material_id"] = [np.array([1, 2], dtype=int)]
    with pytest.raises(ValueError, match="cell_data 'material_id'"):
        insert_cohesive_layer_meshio(bad_cell_data, interface_edges=edges)

    missing_cell_block = meshio.Mesh(
        points=points,
        cells=[("triangle", elements), ("line", np.array([[0, 1]], dtype=int))],
        cell_data={
            "material_id": [
                np.arange(elements.shape[0], dtype=int),
                np.array([0], dtype=int),
            ]
        },
    )
    missing_cell_block.cell_data["material_id"] = []
    with pytest.raises(ValueError, match="no block"):
        insert_cohesive_layer_meshio(missing_cell_block, interface_edges=edges)


def _single_cohesive_operator():
    ce = CohesiveElement(
        nodes_top=(0, 1),
        nodes_bottom=(2, 3),
        normal=np.array([0.0, 1.0]),
        tangent=np.array([1.0, 0.0]),
        length=1.0,
    )
    law = BilinearCohesiveLaw(
        k_n=1000.0, k_t=500.0, sigma_max=10.0, delta_c=0.1)
    return CohesiveInterfaceOperator(
        [ce], law, n_nodes=4, device="cpu"), ce


def test_cohesive_operator_zero_jump_force_is_zero():
    import torch

    op, _ = _single_cohesive_operator()
    u = torch.zeros((4, 2), dtype=torch.float64)
    f = op.internal_force(u)

    assert torch.allclose(f, torch.zeros_like(f))


def test_cohesive_operator_opening_force_is_balanced():
    import torch

    op, _ = _single_cohesive_operator()
    u = torch.zeros((4, 2), dtype=torch.float64)
    u[0:2, 1] = 1.0e-3
    f = op.internal_force(u)

    assert torch.allclose(f.sum(dim=0), torch.zeros(2, dtype=torch.float64))
    assert f[0, 1].item() > 0.0
    assert f[2, 1].item() < 0.0


def test_cohesive_law_rejects_negative_contact_stiffness():
    import torch

    law = BilinearCohesiveLaw(
        k_n=1000.0, k_t=500.0, sigma_max=10.0, delta_c=0.1,
        contact_stiffness=-1.0)
    cohesive_state = CohesiveState.zeros(1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="contact_stiffness"):
        law.evaluate(torch.zeros((1, 1, 2), dtype=torch.float64), cohesive_state)


def test_cohesive_operator_contact_penalty_resists_compression_without_damage():
    import torch

    ce = CohesiveElement(
        nodes_top=(0, 1),
        nodes_bottom=(2, 3),
        normal=np.array([0.0, 1.0]),
        tangent=np.array([1.0, 0.0]),
        length=1.0,
    )
    law = BilinearCohesiveLaw(
        k_n=1000.0, k_t=500.0, sigma_max=10.0, delta_c=0.1,
        contact_stiffness=2000.0)
    op = CohesiveInterfaceOperator([ce], law, n_nodes=4, device="cpu")
    u = torch.zeros((4, 2), dtype=torch.float64)
    u[0:2, 1] = -1.0e-3
    f = op.internal_force(u)
    trial = op._trial_state

    assert torch.allclose(f.sum(dim=0), torch.zeros(2, dtype=torch.float64))
    assert f[0, 1].item() < 0.0
    assert f[2, 1].item() > 0.0
    assert trial is not None
    assert torch.allclose(trial.damage, torch.zeros_like(trial.damage))
    assert torch.allclose(
        trial.dissipated_energy,
        torch.zeros_like(trial.dissipated_energy),
    )


def test_cohesive_operator_contact_tangent_matches_finite_difference():
    import torch

    ce = CohesiveElement(
        nodes_top=(0, 1),
        nodes_bottom=(2, 3),
        normal=np.array([0.0, 1.0]),
        tangent=np.array([1.0, 0.0]),
        length=1.0,
    )
    law = BilinearCohesiveLaw(
        k_n=1000.0, k_t=500.0, sigma_max=10.0, delta_c=0.1,
        contact_stiffness=2000.0)
    op = CohesiveInterfaceOperator([ce], law, n_nodes=4, device="cpu")
    u = torch.zeros((4, 2), dtype=torch.float64)
    u[0:2, 1] = -1.0e-3
    u[0:2, 0] = 4.0e-4
    du = torch.zeros_like(u)
    du[0:2, 1] = -3.0e-4
    du[0:2, 0] = 2.0e-4

    K = op.assemble_tangent(u)
    action = torch.sparse.mm(K, du.reshape(-1, 1)).reshape_as(u)
    h = 1.0e-6
    f_plus = op.internal_force(u + h * du, state=op.state)
    f_minus = op.internal_force(u - h * du, state=op.state)
    fd = (f_plus - f_minus) / (2.0 * h)

    assert torch.allclose(action, fd, rtol=2.0e-5, atol=1.0e-8)


def test_cohesive_operator_explicit_state_diagnostics_are_read_only():
    import torch

    op, _ = _single_cohesive_operator()
    committed = op.state
    u = torch.zeros((4, 2), dtype=torch.float64)
    u[0:2, 1] = 3.0e-2

    assert op._trial_state is None
    op.internal_force(u, state=committed)
    assert op._trial_state is None
    op.assemble_tangent(u, state=committed)
    assert op._trial_state is None
    assert torch.allclose(op.state.damage, committed.damage)


def test_cohesive_operator_tangent_matches_finite_difference():
    import torch

    op, _ = _single_cohesive_operator()
    u = torch.zeros((4, 2), dtype=torch.float64)
    u[0:2, 1] = 1.0e-3
    du = torch.zeros_like(u)
    du[0:2, 1] = 2.0e-4
    du[2:4, 0] = -1.0e-4

    K = op.assemble_tangent(u)
    action = torch.sparse.mm(K, du.reshape(-1, 1)).reshape_as(u)
    h = 1.0e-6
    f_plus = op.internal_force(u + h * du)
    f_minus = op.internal_force(u - h * du)
    fd = (f_plus - f_minus) / (2.0 * h)

    assert torch.allclose(action, fd, rtol=5.0e-5, atol=1.0e-8)


def test_cohesive_operator_mixed_mode_damaged_tangent_matches_finite_difference():
    import torch

    op, _ = _single_cohesive_operator()
    u = torch.zeros((4, 2), dtype=torch.float64)
    u[0:2, 1] = 3.0e-2
    u[0:2, 0] = 1.5e-2
    du = torch.zeros_like(u)
    du[0:2, 1] = 1.0e-3
    du[0:2, 0] = -7.0e-4
    du[2:4, 1] = -2.0e-4

    K = op.assemble_tangent(u)
    action = torch.sparse.mm(K, du.reshape(-1, 1)).reshape_as(u)
    h = 1.0e-6
    f_plus = op.internal_force(u + h * du, state=op.state)
    f_minus = op.internal_force(u - h * du, state=op.state)
    fd = (f_plus - f_minus) / (2.0 * h)

    assert torch.allclose(action, fd, rtol=2.0e-5, atol=1.0e-8)


def test_cohesive_law_mode_i_fracture_energy_matches_bilinear_area():
    import torch

    law = BilinearCohesiveLaw(
        k_n=1000.0, k_t=500.0, sigma_max=10.0, delta_c=0.1)
    delta = torch.tensor([0.0, law.delta_0, law.delta_c], dtype=torch.float64)
    damage = torch.clamp(
        law.delta_c
        * (delta - law.delta_0)
        / (torch.clamp(delta, min=torch.finfo(delta.dtype).eps)
           * (law.delta_c - law.delta_0)),
        min=0.0,
        max=1.0,
    )
    dissipated = law.dissipated_energy_density(delta, damage)

    assert law.fracture_energy == pytest.approx(0.5)
    assert dissipated[0].item() == pytest.approx(0.0)
    assert dissipated[1].item() == pytest.approx(0.0)
    assert dissipated[2].item() == pytest.approx(law.fracture_energy)


def test_cohesive_operator_integrates_mode_i_dissipated_energy_capacity():
    import torch

    op, _ = _single_cohesive_operator()
    u = torch.zeros((4, 2), dtype=torch.float64)
    u[0:2, 1] = op.law.delta_c
    op.update_trial(u)
    op.commit()

    dissipated = op.integrated_dissipated_energy()
    capacity = op.integrated_fracture_energy_capacity()

    assert dissipated.item() == pytest.approx(capacity.item(), rel=0, abs=1.0e-12)
    assert capacity.item() == pytest.approx(op.law.fracture_energy)


def test_cohesive_operator_unloading_tangent_uses_committed_damage():
    import torch

    op, _ = _single_cohesive_operator()
    u_peak = torch.zeros((4, 2), dtype=torch.float64)
    u_peak[0:2, 1] = 3.0e-2
    op.update_trial(u_peak)
    op.commit()

    u = torch.zeros((4, 2), dtype=torch.float64)
    u[0:2, 1] = 2.0e-2
    du = torch.zeros_like(u)
    du[0:2, 1] = 1.0e-3

    K = op.assemble_tangent(u)
    action = torch.sparse.mm(K, du.reshape(-1, 1)).reshape_as(u)
    h = 1.0e-6
    f_plus = op.internal_force(u + h * du, state=op.state)
    f_minus = op.internal_force(u - h * du, state=op.state)
    fd = (f_plus - f_minus) / (2.0 * h)

    assert torch.allclose(action, fd, rtol=2.0e-5, atol=1.0e-8)


def test_cohesive_operator_commit_and_rollback_history():
    import torch

    op, _ = _single_cohesive_operator()
    u = torch.zeros((4, 2), dtype=torch.float64)
    u[0:2, 1] = 2.0e-2

    initial = op.state.clone()
    trial = op.update_trial(u)
    assert torch.any(trial.damage > 0.0)
    op.rollback()
    assert op._trial_state is None
    assert torch.allclose(op.state.damage, initial.damage)

    op.update_trial(u)
    committed = op.commit()
    assert torch.any(committed.damage > 0.0)
    assert op._trial_state is None


def test_quasistatic_solver_accepts_cohesive_operator_prescribed_opening():
    import torch

    from phast.fem_operators import FEMOperators
    from phast.material import Material
    from phast.mechanics_solver import QuasiStaticSolver
    from phast.mesh import FEMMesh
    from phast.cohesive_elements import insert_cohesive_layer

    nodes, elements, edges = _two_strip_mesh()
    new_nodes, new_elements, cohesives = insert_cohesive_layer(
        nodes, elements, edges)
    mesh = FEMMesh.from_tensors(
        torch.as_tensor(new_nodes, dtype=torch.float64),
        torch.as_tensor(new_elements, dtype=torch.long),
        device="cpu",
        dtype=torch.float64,
    )
    material = Material(energy_split="isotropic")
    fem = FEMOperators(mesh, material)
    law = BilinearCohesiveLaw(
        k_n=1000.0, k_t=500.0, sigma_max=10.0, delta_c=0.1)
    op = CohesiveInterfaceOperator(
        cohesives, law, n_nodes=mesh.n_nodes, device="cpu")
    solver = QuasiStaticSolver(
        fem, cohesive_operator=op, backend="auto", max_iter=5)

    bc_mask = torch.ones((mesh.n_nodes, 2), dtype=torch.bool)
    bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    bc_vals[0:2, 1] = 2.0e-2
    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)

    u, converged, n_iter = solver.solve(d, f_ext, bc_mask, bc_vals)

    assert converged
    assert n_iter == 0
    assert torch.allclose(u, bc_vals)
    assert torch.any(op.state.damage > 0.0)
