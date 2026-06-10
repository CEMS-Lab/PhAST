"""Tests for Q4/Q8/Q9 quadrilateral primitives (issue #5)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_Q_PATH = Path(__file__).resolve().parents[1] / "src" / "phast" / "quad_elements.py"
_spec = importlib.util.spec_from_file_location("quad_elements", _Q_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

quad_element_stiffness = _mod.quad_element_stiffness
quad_gauss_points = _mod.quad_gauss_points
q4_internal_force = _mod.q4_internal_force
q4_laplacian_matvec = _mod.q4_laplacian_matvec
q4_mass_matvec = _mod.q4_mass_matvec
q4_quality = _mod.q4_quality
q4_quadrature_geometry = _mod.q4_quadrature_geometry
q4_strain_at_gauss = _mod.q4_strain_at_gauss
q4_to_triangles = _mod.q4_to_triangles
quad_ref_nodes = _mod.quad_ref_nodes
quad_shape_function_derivs = _mod.quad_shape_function_derivs
quad_shape_functions = _mod.quad_shape_functions
structured_q4_mesh = _mod.structured_q4_mesh


@pytest.mark.parametrize("kind,n_nodes", [("Q4", 4), ("Q8", 8), ("Q9", 9)])
def test_quad_shape_functions_partition_of_unity(kind, n_nodes):
    for xi, eta in [(-0.7, -0.5), (0.0, 0.0), (0.3, -0.2), (0.8, 0.6)]:
        N = quad_shape_functions(kind, xi, eta)
        dN = quad_shape_function_derivs(kind, xi, eta)
        assert N.shape == (n_nodes,)
        assert dN.shape == (n_nodes, 2)
        assert torch.isclose(N.sum(), torch.tensor(1.0, dtype=torch.float64),
                             atol=1e-12)
        assert torch.allclose(dN.sum(0), torch.zeros(2, dtype=torch.float64),
                              atol=1e-12)


@pytest.mark.parametrize("kind", ["Q4", "Q8", "Q9"])
def test_quad_shape_functions_are_nodal(kind):
    ref = quad_ref_nodes(kind)
    for j in range(ref.shape[0]):
        N = quad_shape_functions(kind, float(ref[j, 0]), float(ref[j, 1]))
        expected = torch.zeros(ref.shape[0], dtype=torch.float64)
        expected[j] = 1.0
        assert torch.allclose(N, expected, atol=1e-12), (kind, j, N)


@pytest.mark.parametrize(
    "fn, expected",
    [
        (lambda x, y: torch.ones_like(x), 4.0),
        (lambda x, y: x, 0.0),
        (lambda x, y: y, 0.0),
        (lambda x, y: x * x, 4.0 / 3.0),
        (lambda x, y: y * y, 4.0 / 3.0),
        (lambda x, y: x * y, 0.0),
        (lambda x, y: x * x * y * y, 4.0 / 9.0),
    ],
)
def test_quad_gauss_order_2_exactness(fn, expected):
    pts, wts = quad_gauss_points(order=2)
    val = (wts * fn(pts[:, 0], pts[:, 1])).sum()
    assert torch.isclose(val, torch.tensor(expected, dtype=torch.float64),
                         atol=1e-12)


@pytest.mark.parametrize("kind", ["Q4", "Q8", "Q9"])
def test_quad_element_stiffness_symmetry_and_rigid_translation(kind):
    coords = quad_ref_nodes(kind)
    # Map reference square to a 2x1 physical rectangle. This keeps Q8/Q9
    # midside/centre nodes exactly consistent with the isoparametric map.
    coords = torch.stack([coords[:, 0] + 1.0, 0.5 * (coords[:, 1] + 1.0)], dim=1)
    K = quad_element_stiffness(kind, coords, E=210.0, nu=0.3,
                               plane_strain=True)
    n = coords.shape[0]
    assert K.shape == (2 * n, 2 * n)
    assert torch.allclose(K, K.T, atol=1e-10)

    ux = torch.zeros(2 * n, dtype=torch.float64)
    ux[0::2] = 1.0
    uy = torch.zeros(2 * n, dtype=torch.float64)
    uy[1::2] = 1.0
    assert (K @ ux).abs().max().item() < 1e-9
    assert (K @ uy).abs().max().item() < 1e-9


@pytest.mark.parametrize("kind", ["Q4", "Q8", "Q9"])
def test_quad_constant_strain_reproduction(kind):
    coords = quad_ref_nodes(kind)
    coords = torch.stack([1.5 * coords[:, 0] + 0.2 * coords[:, 1] + 3.0,
                          0.1 * coords[:, 0] + 0.8 * coords[:, 1] - 2.0],
                         dim=1)
    eps_xx, eps_yy, gamma_xy = 0.01, -0.003, 0.004
    u = torch.zeros(2 * coords.shape[0], dtype=torch.float64)
    for i in range(coords.shape[0]):
        x, y = float(coords[i, 0]), float(coords[i, 1])
        u[2 * i] = eps_xx * x + 0.5 * gamma_xy * y
        u[2 * i + 1] = 0.5 * gamma_xy * x + eps_yy * y
    expected = torch.tensor([eps_xx, eps_yy, gamma_xy], dtype=torch.float64)

    pts, _ = quad_gauss_points(order=2 if kind == "Q4" else 3)
    for q in range(pts.shape[0]):
        xi, eta = float(pts[q, 0]), float(pts[q, 1])
        dN_ref = quad_shape_function_derivs(kind, xi, eta)
        J = dN_ref.T @ coords
        dN_xy = dN_ref @ torch.linalg.inv(J).T
        B = torch.zeros((3, 2 * coords.shape[0]), dtype=torch.float64)
        for i in range(coords.shape[0]):
            B[0, 2 * i] = dN_xy[i, 0]
            B[1, 2 * i + 1] = dN_xy[i, 1]
            B[2, 2 * i] = dN_xy[i, 1]
            B[2, 2 * i + 1] = dN_xy[i, 0]
        assert torch.allclose(B @ u, expected, atol=1e-12)


def test_structured_q4_mesh_and_quality_helpers():
    mesh = structured_q4_mesh(width=2.0, height=1.0, nx=2, ny=1)
    assert mesh.nodes.shape == (6, 2)
    assert mesh.quads.shape == (2, 4)
    assert q4_to_triangles(mesh.quads).shape == (4, 3)
    quality = q4_quality(mesh.nodes, mesh.quads)
    assert quality["all_positive_orientation"]
    assert quality["max_aspect_ratio"] == pytest.approx(1.0)


def test_native_q4_internal_force_matches_element_stiffness():
    nodes = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    quads = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    u = torch.tensor(
        [[0.0, 0.0], [0.01, 0.002], [0.012, -0.001], [0.001, -0.003]],
        dtype=torch.float64,
    )
    f = q4_internal_force(nodes, quads, u, E=210.0, nu=0.3,
                          plane_strain=True)
    K = quad_element_stiffness("Q4", nodes, E=210.0, nu=0.3,
                               plane_strain=True)
    expected = (K @ u.reshape(-1)).reshape(4, 2)

    assert torch.allclose(f, expected, atol=1.0e-10)
    assert torch.allclose(f.sum(dim=0), torch.zeros(2, dtype=torch.float64),
                          atol=1.0e-12)


def test_native_q4_affine_strain_exact_at_gauss_points():
    nodes = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    quads = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    eps_xx, eps_yy, gamma_xy = 0.01, -0.003, 0.004
    u = torch.zeros((4, 2), dtype=torch.float64)
    u[:, 0] = eps_xx * nodes[:, 0] + 0.5 * gamma_xy * nodes[:, 1]
    u[:, 1] = 0.5 * gamma_xy * nodes[:, 0] + eps_yy * nodes[:, 1]
    strain = q4_strain_at_gauss(nodes, quads, u)
    expected = torch.tensor([eps_xx, eps_yy, gamma_xy], dtype=torch.float64)

    assert torch.allclose(strain[0], expected.expand(4, 3), atol=1.0e-12)


def test_native_q4_scalar_laplacian_and_mass_match_quadrature_assembly():
    nodes = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    quads = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    x = torch.tensor([0.2, -0.1, 0.5, 0.3], dtype=torch.float64)
    N, grad, wdet = q4_quadrature_geometry(nodes, quads)
    K = torch.zeros((4, 4), dtype=torch.float64)
    M = torch.zeros((4, 4), dtype=torch.float64)
    for q in range(4):
        K += (
            grad[0, q, :, 0:1] @ grad[0, q, :, 0:1].T
            + grad[0, q, :, 1:2] @ grad[0, q, :, 1:2].T
        ) * wdet[0, q]
        M += (N[q:q + 1].T @ N[q:q + 1]) * wdet[0, q]

    assert torch.allclose(q4_laplacian_matvec(nodes, quads, x), K @ x,
                          atol=1.0e-12)
    assert torch.allclose(q4_mass_matvec(nodes, quads, x), M @ x,
                          atol=1.0e-12)
