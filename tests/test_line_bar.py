"""Analytical and derivative checks for the standalone 1D PhAST API."""
from __future__ import annotations

import pytest
import torch

from phast import LineMesh, line_mesh, solve_bar


@pytest.mark.parametrize("count", [1, 2, 10, 40])
def test_uniform_forward(count: int) -> None:
    mesh = line_mesh(100.0, count, origin=5.0)
    result = solve_bar(mesh, 210000.0, 10.0, 4000.0)
    expected = 4000.0 * (mesh.nodes - 5.0) / (210000.0 * 10.0)
    torch.testing.assert_close(result.displacement, expected, rtol=1e-11, atol=1e-12)
    assert mesh.elements.shape == (count, 2)
    torch.testing.assert_close(result.reaction, torch.tensor(-4000.0, dtype=torch.float64))
    assert result.free_residual.abs().max() < 1e-7
    torch.testing.assert_close(result.stress, torch.full((count,), 400.0, dtype=torch.float64))
    torch.testing.assert_close(result.axial_force, torch.full((count,), 4000.0, dtype=torch.float64))


@pytest.mark.parametrize("force", [0.0, -4000.0, 4000.0])
def test_nonuniform_mesh_and_prescribed_translation(force: float) -> None:
    mesh = LineMesh(torch.tensor([0.0, 0.1, 3.0, 10.0], dtype=torch.float64))
    result = solve_bar(mesh, 200000.0, 10.0, force, left_displacement=0.02)
    expected = 0.02 + force * mesh.nodes / 2000000.0
    torch.testing.assert_close(result.displacement, expected, rtol=1e-10, atol=1e-12)
    assert result.free_residual.abs().max() < 1e-7


def test_gradients_modulus_area_force_and_translation() -> None:
    mesh = line_mesh(100.0, 10)
    E, A, F, u0 = [torch.tensor(v, dtype=torch.float64, requires_grad=True)
                   for v in (100000.0, 10.0, 4000.0, 0.03)]
    tip = solve_bar(mesh, E, A, F, left_displacement=u0).displacement[-1]
    gradients = torch.autograd.grad(tip, (E, A, F, u0))
    expected = (-F * 100 / (A * E**2), -F * 100 / (E * A**2), 100 / (E * A), torch.ones_like(u0))
    for actual, reference in zip(gradients, expected):
        torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-12)


def test_coordinate_gradient() -> None:
    length = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    mesh = LineMesh(length * torch.linspace(0, 1, 11, dtype=torch.float64))
    tip = solve_bar(mesh, 210000.0, 10.0, 4000.0).displacement[-1]
    derivative = torch.autograd.grad(tip, length)[0]
    assert float(derivative) == pytest.approx(4000 / 2100000, rel=1e-10)


def test_sparse_backward_gradcheck() -> None:
    mesh = line_mesh(2.0, 3)
    E = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda e: solve_bar(mesh, e, 2.0, 1.0).displacement,
                                    (E,), eps=1e-6, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("length,count", [(0, 2), (-1, 2), (float('nan'), 2), (1, 0), (1, -1)])
def test_bad_uniform_mesh(length: float, count: int) -> None:
    with pytest.raises(ValueError):
        line_mesh(length, count)


@pytest.mark.parametrize("count", [True, 2.5])
def test_bad_element_count(count: object) -> None:
    with pytest.raises(TypeError):
        line_mesh(1.0, count)


@pytest.mark.parametrize("nodes", [[0], [0, 0], [1, 0], [0, float('nan')]])
def test_bad_nodes(nodes: list[float]) -> None:
    with pytest.raises(ValueError):
        LineMesh(torch.tensor(nodes, dtype=torch.float64))


@pytest.mark.parametrize("E,A,F", [(0, 1, 1), (-1, 1, 1), (1, 0, 1), (1, 1, float('inf'))])
def test_bad_inputs(E: float, A: float, F: float) -> None:
    with pytest.raises(ValueError):
        solve_bar(line_mesh(1), E, A, F)


def test_tensor_dtype_and_scalar_validation() -> None:
    with pytest.raises(ValueError, match="float64"):
        LineMesh(torch.tensor([0., 1.], dtype=torch.float32))
    with pytest.raises(ValueError, match="float64"):
        solve_bar(line_mesh(1), torch.tensor(1., dtype=torch.float32), 1, 1)
    with pytest.raises(ValueError, match="scalar"):
        solve_bar(line_mesh(1), torch.ones(2, dtype=torch.float64), 1, 1)
