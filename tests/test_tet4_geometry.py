from __future__ import annotations

import pytest
import torch

from phast.core.tet4_geometry import Tet4Geometry, structured_tet4_block


def _scalar_element_gradients(geometry: Tet4Geometry, values: torch.Tensor) -> torch.Tensor:
    element_values = values[geometry.elements]
    return torch.einsum("eia,ei->ea", geometry.shape_gradients, element_values)


def _vector_element_gradients(geometry: Tet4Geometry, values: torch.Tensor) -> torch.Tensor:
    element_values = values[geometry.elements]
    return torch.einsum("eia,eij->eja", geometry.shape_gradients, element_values)


def test_tet4_orientation_is_repaired_and_shape_gradients_sum_to_zero():
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    # Local nodes 2 and 3 are intentionally swapped relative to a positive
    # unit tetrahedron. Construction should repair the orientation.
    geometry = Tet4Geometry.from_tensors(nodes, torch.tensor([[0, 1, 3, 2]]))

    assert geometry.elements.tolist() == [[0, 1, 2, 3]]
    assert geometry.n_nodes == 4
    assert geometry.n_elements == 1
    assert geometry.volumes.tolist() == pytest.approx([1.0 / 6.0])
    assert torch.all(geometry.volumes > 0.0)
    assert torch.allclose(
        geometry.shape_gradients.sum(dim=1),
        torch.zeros((1, 3), dtype=torch.float64),
        atol=1.0e-14,
    )


def test_tiny_rescaled_unit_tetra_is_not_rejected_by_default_tolerance():
    scale = 1.0e-6
    nodes = scale * torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    geometry = Tet4Geometry.from_tensors(nodes, torch.tensor([[0, 1, 2, 3]]))

    assert geometry.volumes.tolist() == pytest.approx([scale ** 3 / 6.0])
    expected_gradients = torch.tensor(
        [[[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    ) / scale
    assert torch.allclose(geometry.shape_gradients, expected_gradients, rtol=1.0e-12)


def test_large_translation_unit_tetra_has_stable_gradients_and_patch_result():
    offset = torch.tensor([1.0e12, -1.0e12, 2.0e12], dtype=torch.float64)
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    ) + offset

    geometry = Tet4Geometry.from_tensors(nodes, torch.tensor([[0, 1, 2, 3]]))

    expected_gradients = torch.tensor(
        [[[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    assert torch.allclose(geometry.shape_gradients, expected_gradients, atol=0.0, rtol=0.0)
    coordinate_gradient = _vector_element_gradients(geometry, geometry.nodes)
    assert torch.allclose(
        coordinate_gradient,
        torch.eye(3, dtype=torch.float64).expand(1, -1, -1),
        atol=1.0e-9,
    )
    assert geometry.nodes_on_plane(0, float(offset[0])).tolist() == [0, 2, 3]
    assert geometry.nodes_on_plane(0, float(offset[0] + 1.0)).tolist() == [1]


def test_nodes_on_plane_default_tolerance_is_translation_invariant_for_representable_shift():
    unit_nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    elements = torch.tensor([[0, 1, 2, 3]])
    origin_geometry = Tet4Geometry.from_tensors(unit_nodes, elements)
    translated_geometry = Tet4Geometry.from_tensors(
        unit_nodes + torch.tensor([1.0e12, -2.0e12, 3.0e12], dtype=torch.float64),
        elements,
    )

    assert origin_geometry.nodes_on_plane(0, 0.0).tolist() == [0, 2, 3]
    assert translated_geometry.nodes_on_plane(0, 1.0e12).tolist() == [0, 2, 3]


def test_nodes_on_plane_default_tolerance_resolves_very_small_scale():
    scale = 1.0e-6
    nodes = scale * torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    geometry = Tet4Geometry.from_tensors(nodes, torch.tensor([[0, 1, 2, 3]]))

    assert geometry.nodes_on_plane(0, 0.0).tolist() == [0, 2, 3]
    assert geometry.nodes_on_plane(0, scale).tolist() == [1]


def test_nodes_on_plane_default_tolerance_is_stable_for_float32_meshes():
    offset = torch.tensor([4096.0, -2048.0, 8192.0], dtype=torch.float32)
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ) + offset
    geometry = Tet4Geometry.from_tensors(nodes, torch.tensor([[0, 1, 2, 3]]))

    assert geometry.nodes_on_plane(0, float(offset[0])).tolist() == [0, 2, 3]
    assert geometry.nodes_on_plane(0, float(offset[0] + 1.0)).tolist() == [1]


def test_structured_tet4_block_preserves_volume_and_plane_node_selection():
    geometry = structured_tet4_block(
        length=2.0,
        height=3.0,
        thickness=4.0,
        nx=2,
        ny=3,
        nz=4,
    )

    assert geometry.structured_shape == (2, 3, 4)
    assert geometry.dimensions == (2.0, 3.0, 4.0)
    assert geometry.n_elements == 2 * 3 * 4 * 6
    assert torch.isclose(geometry.volumes.sum(), torch.tensor(24.0, dtype=geometry.dtype))
    assert torch.all(geometry.volumes > 0.0)

    x0_nodes = geometry.nodes_on_plane(0, 0.0)
    assert x0_nodes.numel() == (3 + 1) * (4 + 1)
    assert torch.allclose(
        geometry.nodes[x0_nodes, 0],
        torch.zeros(x0_nodes.numel(), dtype=geometry.dtype),
    )


def test_tet4_affine_scalar_patch_gradient_is_exact():
    geometry = structured_tet4_block(
        length=2.0,
        height=1.0,
        thickness=1.5,
        nx=2,
        ny=1,
        nz=2,
    )
    expected = torch.tensor([3.0, -4.0, 5.0], dtype=geometry.dtype)
    values = 2.0 + geometry.nodes @ expected

    gradients = _scalar_element_gradients(geometry, values)

    assert torch.allclose(
        gradients,
        expected.expand(geometry.n_elements, -1),
        atol=1.0e-12,
    )


def test_tet4_affine_vector_patch_gradient_is_exact():
    geometry = structured_tet4_block(
        length=2.0,
        height=1.0,
        thickness=1.0,
        nx=2,
        ny=1,
        nz=1,
    )
    gradient = torch.tensor(
        [
            [0.10, 0.20, -0.10],
            [0.03, -0.05, 0.04],
            [0.06, -0.02, 0.08],
        ],
        dtype=geometry.dtype,
    )
    translation = torch.tensor([0.4, -0.3, 0.2], dtype=geometry.dtype)
    displacement = geometry.nodes @ gradient.T + translation

    gradients = _vector_element_gradients(geometry, displacement)

    assert torch.allclose(
        gradients,
        gradient.expand(geometry.n_elements, -1, -1),
        atol=1.0e-12,
    )


def test_tet4_rigid_translation_has_zero_vector_gradient():
    geometry = structured_tet4_block(
        length=1.0,
        height=1.0,
        thickness=1.0,
        nx=1,
        ny=1,
        nz=1,
    )
    translation = torch.tensor([1.25, -0.5, 2.0], dtype=geometry.dtype)
    displacement = translation.expand(geometry.n_nodes, -1)

    gradients = _vector_element_gradients(geometry, displacement)

    assert torch.allclose(gradients, torch.zeros_like(gradients), atol=1.0e-14)


@pytest.mark.parametrize(
    ("nodes", "elements", "match"),
    [
        (
            torch.zeros((4, 2), dtype=torch.float64),
            torch.tensor([[0, 1, 2, 3]]),
            "nodes must have shape",
        ),
        (
            torch.arange(12).reshape(4, 3),
            torch.tensor([[0, 1, 2, 3]]),
            "nodes must use a floating-point dtype",
        ),
        (
            torch.zeros((4, 3), dtype=torch.float16),
            torch.tensor([[0, 1, 2, 3]]),
            "torch.float32 or torch.float64",
        ),
        (
            torch.zeros((4, 3), dtype=torch.float64),
            torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
            "elements must use an integer dtype",
        ),
        (
            torch.zeros((4, 3), dtype=torch.float64),
            torch.tensor([[0, 1, 2, 4]]),
            "out-of-range",
        ),
        (
            torch.zeros((4, 3), dtype=torch.float64),
            torch.tensor([[0, 1, 1, 3]]),
            "repeated node",
        ),
        (
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=torch.float64,
            ),
            torch.tensor([[0, 1, 2, 3]]),
            "degenerate",
        ),
    ],
)
def test_tet4_geometry_rejects_invalid_inputs(nodes, elements, match):
    with pytest.raises((TypeError, ValueError), match=match):
        Tet4Geometry.from_tensors(nodes, elements)


def test_structured_tet4_block_rejects_invalid_dimensions_and_counts():
    with pytest.raises(ValueError, match="length"):
        structured_tet4_block(length=0.0, height=1.0, thickness=1.0, nx=1, ny=1, nz=1)
    with pytest.raises(ValueError, match="nx"):
        structured_tet4_block(length=1.0, height=1.0, thickness=1.0, nx=0, ny=1, nz=1)
    with pytest.raises(ValueError, match="axis"):
        structured_tet4_block(length=1.0, height=1.0, thickness=1.0, nx=1, ny=1, nz=1).nodes_on_plane(3, 0.0)


@pytest.mark.parametrize("axis", [True, False, 0.0, 1.0, "0"])
def test_nodes_on_plane_rejects_bool_or_nonintegral_axis(axis):
    geometry = structured_tet4_block(length=1.0, height=1.0, thickness=1.0, nx=1, ny=1, nz=1)
    with pytest.raises(TypeError, match="axis"):
        geometry.nodes_on_plane(axis, 0.0)


@pytest.mark.parametrize("volume_tolerance", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_tet4_geometry_rejects_nonfinite_or_negative_volume_tolerance(volume_tolerance):
    nodes = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="volume_tolerance"):
        Tet4Geometry.from_tensors(
            nodes,
            torch.tensor([[0, 1, 2, 3]]),
            volume_tolerance=volume_tolerance,
        )


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_nodes_on_plane_rejects_nonfinite_or_negative_tolerance(tolerance):
    geometry = structured_tet4_block(length=1.0, height=1.0, thickness=1.0, nx=1, ny=1, nz=1)
    with pytest.raises(ValueError, match="tolerance"):
        geometry.nodes_on_plane(0, 0.0, tolerance=tolerance)
