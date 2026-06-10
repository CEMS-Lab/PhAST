"""Finite-difference parity for constrained damage adjoints.

These tests cover the differentiable ``Gc_field`` path with and without
``pf_dirichlet``. Pinned damage DOFs are treated as eliminated unknowns:
their adjoint RHS is zero, but elements touching the pinned nodes must still
contribute to neighbouring free-node sensitivities.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_pf_dirichlet_lock import _build_tiny_sent_mesh  # noqa: E402


def _build_solver(tmp_path, *, pf_model: str, bounds_method: str):
    from phast.fem_operators import FEMOperators
    from phast.material import Material
    from phast.mesh import FEMMesh
    from phast.damage_solver import PhaseFieldDamageSolver

    msh_path = _build_tiny_sent_mesh(tmp_path, h=0.5)
    mesh = FEMMesh(msh_path, device="cpu", dtype=torch.float64)
    mesh.identify_boundaries()
    material = Material(
        E=210.0,
        nu=0.3,
        Gc=2.7,
        l0=0.4,
        rho=1.0,
        energy_split="amor",
        pf_model=pf_model,
        gamma_correction=True,
    )
    fem = FEMOperators(mesh, material)
    solver = PhaseFieldDamageSolver(
        fem,
        tol=1e-13,
        max_iter=1000,
        bounds_method=bounds_method,
        preconditioner="jacobi",
    )
    return mesh, material, solver


def _gc_field_gradient(mesh, solver, material, *, use_pf_dirichlet: bool):
    H_value = 5.0 if material.pf_model == "AT2" else 20.0
    H = torch.full((mesh.n_elems,), H_value, dtype=torch.float64)
    d_prev = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    l0 = torch.tensor(material.l0, dtype=torch.float64)
    weights = torch.linspace(0.2, 1.1, mesh.n_nodes, dtype=torch.float64)

    pf_mask = None
    pf_values = None
    pinned = None
    if use_pf_dirichlet:
        pinned = mesh.node_sets["bottom"][:1]
        pf_mask = torch.zeros(mesh.n_nodes, dtype=torch.bool)
        pf_mask[pinned] = True
        pf_values = torch.zeros(mesh.n_nodes, dtype=torch.float64)
        pf_values[pf_mask] = 1.0
        weights[pf_mask] = 0.0

    gc_base = torch.linspace(
        0.85 * material.Gc,
        1.15 * material.Gc,
        mesh.n_elems,
        dtype=torch.float64,
    )
    gc_var = gc_base.clone().requires_grad_(True)
    d_new = solver.solve(
        H,
        d_prev,
        Gc_field=gc_var,
        l0=l0,
        pf_dirichlet_mask=pf_mask,
        pf_dirichlet_values=pf_values,
    )
    loss = (weights * d_new).sum()
    loss.backward()
    grad_autograd = gc_var.grad.detach().clone()

    grad_fd = torch.empty_like(gc_base)
    eps = 1e-5
    with torch.no_grad():
        for i_elem in range(mesh.n_elems):
            gc_plus = gc_base.clone()
            gc_minus = gc_base.clone()
            gc_plus[i_elem] += eps
            gc_minus[i_elem] -= eps
            d_plus = solver.solve(
                H,
                d_prev,
                Gc_field=gc_plus,
                l0=l0,
                pf_dirichlet_mask=pf_mask,
                pf_dirichlet_values=pf_values,
            )
            d_minus = solver.solve(
                H,
                d_prev,
                Gc_field=gc_minus,
                l0=l0,
                pf_dirichlet_mask=pf_mask,
                pf_dirichlet_values=pf_values,
            )
            grad_fd[i_elem] = (
                (weights * d_plus).sum() - (weights * d_minus).sum()
            ) / (2.0 * eps)

    return grad_autograd, grad_fd, pinned


@pytest.mark.parametrize("pf_model", ["AT1", "AT2"])
@pytest.mark.parametrize("bounds_method", ["post_clamp", "projected_cg"])
@pytest.mark.parametrize("use_pf_dirichlet", [False, True])
def test_gc_field_adjoint_matches_fd_with_optional_pf_dirichlet(
    tmp_path, pf_model, bounds_method, use_pf_dirichlet
):
    mesh, material, solver = _build_solver(
        tmp_path, pf_model=pf_model, bounds_method=bounds_method)

    grad_autograd, grad_fd, pinned = _gc_field_gradient(
        mesh, solver, material, use_pf_dirichlet=use_pf_dirichlet)

    rel = torch.linalg.vector_norm(grad_autograd - grad_fd) / (
        torch.linalg.vector_norm(grad_fd) + 1e-12)
    assert rel.item() < 5e-6

    if pinned is not None:
        touching = (mesh.elements == int(pinned[0])).any(dim=1)
        assert touching.any()
        assert torch.isfinite(grad_autograd[touching]).all()
        assert torch.allclose(
            grad_autograd[touching],
            grad_fd[touching],
            rtol=5e-5,
            atol=5e-8,
        )
