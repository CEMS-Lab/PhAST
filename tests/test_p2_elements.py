"""Tests for P2 (T6) quadratic-triangle element primitives (issue #112)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_P2_PATH = Path(__file__).resolve().parents[1] / "src" / "phast" / "p2_elements.py"
_spec = importlib.util.spec_from_file_location("p2_elements", _P2_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

P2_REF_NODES = _mod.P2_REF_NODES
p2_element_stiffness = _mod.p2_element_stiffness
p2_gauss_points = _mod.p2_gauss_points
p2_mesh_from_p1 = _mod.p2_mesh_from_p1
p2_node_indices_from_p1_mesh = _mod.p2_node_indices_from_p1_mesh
p2_shape_function_derivs = _mod.p2_shape_function_derivs
p2_shape_functions = _mod.p2_shape_functions


def test_shape_functions_partition_of_unity():
    for xi, eta in [(0.1, 0.2), (0.3, 0.3), (0.0, 0.0), (0.5, 0.25), (1 / 3, 1 / 3)]:
        N = p2_shape_functions(xi, eta)
        assert torch.isclose(N.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)


def test_shape_functions_at_nodes():
    for j in range(6):
        xi, eta = float(P2_REF_NODES[j, 0]), float(P2_REF_NODES[j, 1])
        N = p2_shape_functions(xi, eta)
        expected = torch.zeros(6, dtype=torch.float64)
        expected[j] = 1.0
        assert torch.allclose(N, expected, atol=1e-12), f"node {j}: {N}"


def test_derivative_partition():
    for xi, eta in [(0.1, 0.2), (0.4, 0.1), (0.0, 0.0), (1 / 3, 1 / 3)]:
        dN = p2_shape_function_derivs(xi, eta)
        assert torch.isclose(dN[:, 0].sum(), torch.tensor(0.0, dtype=torch.float64), atol=1e-12)
        assert torch.isclose(dN[:, 1].sum(), torch.tensor(0.0, dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize(
    "fn, expected",
    [
        (lambda x, y: torch.ones_like(x), 0.5),
        (lambda x, y: x, 1 / 6),
        (lambda x, y: y, 1 / 6),
        (lambda x, y: x * x, 1 / 12),
        (lambda x, y: y * y, 1 / 12),
        (lambda x, y: x * y, 1 / 24),
    ],
)
def test_gauss_quadrature_exactness(fn, expected):
    pts, wts = p2_gauss_points(order=2)
    val = (wts * fn(pts[:, 0], pts[:, 1])).sum()
    assert torch.isclose(val, torch.tensor(expected, dtype=torch.float64), atol=1e-12), (
        f"got {val.item()}, expected {expected}"
    )


def test_element_stiffness_symmetry():
    # Reference triangle with midpoint nodes.
    K = p2_element_stiffness(P2_REF_NODES, E=210.0, nu=0.3, plane_strain=True)
    assert K.shape == (12, 12)
    asymm = (K - K.T).abs().max().item()
    assert asymm < 1e-12, f"asymmetry {asymm}"


def test_element_stiffness_rigid_translation():
    # Generic (non-degenerate) physical triangle plus its true midpoints.
    v = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.5, 1.5]], dtype=torch.float64)
    coords = torch.stack(
        [v[0], v[1], v[2], 0.5 * (v[0] + v[1]), 0.5 * (v[1] + v[2]), 0.5 * (v[2] + v[0])]
    )
    K = p2_element_stiffness(coords, E=100.0, nu=0.25, plane_strain=True)
    # Rigid x-translation: u_i = 1, v_i = 0 for all 6 nodes.
    u = torch.zeros(12, dtype=torch.float64)
    u[0::2] = 1.0
    f = K @ u
    assert f.abs().max().item() < 1e-10, f"rigid-x residual {f.abs().max().item()}"
    # Rigid y-translation as a bonus.
    u = torch.zeros(12, dtype=torch.float64)
    u[1::2] = 1.0
    f = K @ u
    assert f.abs().max().item() < 1e-10


def _build_B(coords, xi, eta):
    """Helper: assemble the 3x12 B-matrix at (xi, eta) for a P2 element."""
    dN_ref = p2_shape_function_derivs(xi, eta)
    J = dN_ref.T @ coords
    Jinv = torch.linalg.inv(J)
    dN_xy = dN_ref @ Jinv.T
    B = torch.zeros((3, 12), dtype=torch.float64)
    for i in range(6):
        B[0, 2 * i] = dN_xy[i, 0]
        B[1, 2 * i + 1] = dN_xy[i, 1]
        B[2, 2 * i] = dN_xy[i, 1]
        B[2, 2 * i + 1] = dN_xy[i, 0]
    return B


def test_constant_strain_reproduction_non_reference():
    """Patch test: constant strain field exactly reproduced on a non-reference triangle.

    Regression for the chain-rule transpose bug: with J[a,b] = dx_b/dxi_a, the
    physical-derivative map needs Jinv.T, not Jinv. Bug is invisible whenever J
    is symmetric (e.g. the reference triangle), so we use a generic triangle.
    """
    v = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.5, 1.5]], dtype=torch.float64)
    coords = torch.stack(
        [v[0], v[1], v[2], 0.5 * (v[0] + v[1]), 0.5 * (v[1] + v[2]), 0.5 * (v[2] + v[0])]
    )
    eps_xx, eps_yy, gamma_xy = 0.01, 0.005, 0.002
    # Linear displacement field producing this constant strain.
    u_nodal = torch.zeros(12, dtype=torch.float64)
    for i in range(6):
        x, y = float(coords[i, 0]), float(coords[i, 1])
        u_nodal[2 * i] = eps_xx * x + 0.5 * gamma_xy * y
        u_nodal[2 * i + 1] = 0.5 * gamma_xy * x + eps_yy * y
    expected = torch.tensor([eps_xx, eps_yy, gamma_xy], dtype=torch.float64)
    pts, _ = p2_gauss_points(order=2)
    for q in range(pts.shape[0]):
        xi, eta = float(pts[q, 0]), float(pts[q, 1])
        B = _build_B(coords, xi, eta)
        eps = B @ u_nodal
        err = (eps - expected).abs().max().item()
        assert err < 1e-12, f"gp{q}: eps={eps.tolist()} expected={expected.tolist()} err={err}"


def test_two_element_patch_constant_stress():
    """Two-element patch test: linear displacement field reproduced to 1e-12.

    Mesh: unit square (0,0)-(1,0)-(1,1)-(0,1) split along the (0,0)-(1,1)
    diagonal into two T6 triangles. A uniform-strain field is prescribed at
    every node (Dirichlet on every node — the strictest patch test); each
    element's B@u must reproduce the input strain exactly. Bug-affected
    output is ~10% off on the non-reference (upper) triangle.
    """
    # Vertex coordinates of the unit square plus midpoints.
    # Triangle A: (0,0), (1,0), (1,1)  with midpoints
    # Triangle B: (0,0), (1,1), (0,1)  with midpoints
    def coords_for(verts):
        v = torch.tensor(verts, dtype=torch.float64)
        return torch.stack(
            [v[0], v[1], v[2], 0.5 * (v[0] + v[1]), 0.5 * (v[1] + v[2]), 0.5 * (v[2] + v[0])]
        )

    coords_A = coords_for([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    coords_B = coords_for([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    eps_xx, eps_yy, gamma_xy = 0.01, 0.005, 0.0
    expected = torch.tensor([eps_xx, eps_yy, gamma_xy], dtype=torch.float64)

    def u_field(c):
        u = torch.zeros(12, dtype=torch.float64)
        for i in range(6):
            x, y = float(c[i, 0]), float(c[i, 1])
            u[2 * i] = eps_xx * x + 0.5 * gamma_xy * y
            u[2 * i + 1] = 0.5 * gamma_xy * x + eps_yy * y
        return u

    pts, _ = p2_gauss_points(order=2)
    for tag, c in [("A", coords_A), ("B", coords_B)]:
        u = u_field(c)
        for q in range(pts.shape[0]):
            xi, eta = float(pts[q, 0]), float(pts[q, 1])
            B = _build_B(c, xi, eta)
            eps = B @ u
            err = (eps - expected).abs().max().item()
            assert err < 1e-12, f"tri{tag} gp{q}: err={err}"


def test_p1_to_p2_connectivity_reuses_shared_edge_midpoint():
    """P1->T6 conversion must assign one midpoint node per unique edge."""
    nodes = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)

    p2_nodes, p2_elements = p2_mesh_from_p1(nodes, elements)

    assert p2_nodes.shape == (9, 2)  # 4 vertices + 5 unique square edges/diagonal.
    assert p2_elements.tolist() == [
        [0, 1, 2, 4, 5, 6],
        [0, 2, 3, 6, 7, 8],
    ]
    # The shared diagonal (0, 2) is local edge 2-0 in tri A and 0-1 in tri B.
    assert p2_elements[0, 5].item() == p2_elements[1, 3].item()
    assert torch.allclose(p2_nodes[6], torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_p2_node_indices_from_p1_mesh_matches_mesh_conversion():
    class MeshLike:
        def __init__(self):
            self.nodes = torch.zeros((4, 2), dtype=torch.float64)
            self.elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)

    mesh = MeshLike()
    p2_indices = p2_node_indices_from_p1_mesh(mesh)

    assert p2_indices.dtype == torch.long
    assert p2_indices.tolist() == [
        [0, 1, 2, 4, 5, 6],
        [0, 2, 3, 6, 7, 8],
    ]
