"""Issue #105: sparse-direct quasi-static solves preserve autograd."""

import torch

from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mechanics_solver import QuasiStaticSolver
from phast.mesh import FEMMesh


def _unit_square_fem():
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
    mesh = FEMMesh.from_tensors(nodes, elements, device="cpu")
    material = Material(
        E=210.0,
        nu=0.3,
        Gc=2.7,
        l0=0.1,
        pf_model="AT2",
        energy_split="isotropic",
    )
    return FEMOperators(mesh, material)


def _left_fixed_right_loaded(fem):
    n_nodes = fem.mesh.n_nodes
    bc_mask = torch.zeros(n_nodes, 2, dtype=torch.bool)
    bc_vals = torch.zeros(n_nodes, 2, dtype=torch.float64)
    left = fem.mesh.nodes[:, 0] < 1.0e-12
    right = fem.mesh.nodes[:, 0] > 1.0 - 1.0e-12
    bc_mask[left, :] = True

    f_ext = torch.zeros(n_nodes, 2, dtype=torch.float64)
    f_ext[right, 0] = 1.0
    return f_ext, bc_mask, bc_vals, right


def test_sparse_direct_quasistatic_matches_cg_on_isotropic_problem():
    fem = _unit_square_fem()
    f_ext, bc_mask, bc_vals, _ = _left_fixed_right_loaded(fem)
    d = torch.tensor([0.0, 0.15, 0.25, 0.05], dtype=torch.float64)

    scipy_solver = QuasiStaticSolver(
        fem, tol=1.0e-12, max_iter=5, backend="scipy"
    )
    cg_solver = QuasiStaticSolver(
        fem, tol=1.0e-12, max_iter=5, backend="cg", cg_tol=1.0e-14,
        cg_max_iter=500,
    )

    u_direct, conv_direct, _ = scipy_solver.solve(d, f_ext, bc_mask, bc_vals)
    u_cg, conv_cg, _ = cg_solver.solve(d, f_ext, bc_mask, bc_vals)

    assert conv_direct
    assert conv_cg
    torch.testing.assert_close(u_direct, u_cg, rtol=1.0e-9, atol=1.0e-10)


def test_sparse_direct_quasistatic_backpropagates_through_damage():
    fem = _unit_square_fem()
    f_ext, bc_mask, bc_vals, right = _left_fixed_right_loaded(fem)
    solver = QuasiStaticSolver(
        fem, tol=1.0e-12, max_iter=5, backend="scipy"
    )

    damage_scale = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    damage_shape = torch.tensor([0.0, 0.2, 0.5, 0.1], dtype=torch.float64)
    d = damage_scale * damage_shape

    u, converged, _ = solver.solve(d, f_ext, bc_mask, bc_vals)
    assert converged

    compliance = (u[right] * f_ext[right]).sum()
    compliance.backward()

    assert damage_scale.grad is not None
    assert torch.isfinite(damage_scale.grad)
    assert damage_scale.grad.abs() > 1.0e-10
