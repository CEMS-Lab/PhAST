"""Tests for the parameter-free rational AT2 degradation law.

The law is

.. math:: g(d) = (1-d)^2 / ((1-d)^2 + d) = (1-d)^2 / (d^2 - d + 1)

It carries no length-scale coefficient and applies no residual-stiffness
blend, which distinguishes it from the Wu-style ``degradation_type='rational'``.
Because it is not quadratic, the damage subproblem is nonlinear and is solved
by a bound-constrained projected Newton iteration.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import phast
from phast.core.fem_operators import stress_split_of
from phast.fem_operators import FEMOperators
from phast.damage_solver import PhaseFieldDamageSolver
from phast.material import Material
from phast.mesh import FEMMesh

DTYPE = torch.float64


def make_material(**overrides):
    settings = dict(
        E=210000.0, nu=0.3, Gc=2.7, l0=0.4, rho=7.8e-9, eta_residual=1e-7,
        energy_split="spectral", pf_model="AT2", plane_stress=True,
    )
    settings.update(overrides)
    return Material(**settings)


def unit_square_mesh(n=6):
    """A structured triangulation of a unit square."""
    xs = np.linspace(0.0, 1.0, n)
    grid_x, grid_y = np.meshgrid(xs, xs, indexing="ij")
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            triangles.append([a, b, d])
            triangles.append([a, d, c])
    return (torch.as_tensor(nodes, dtype=DTYPE),
            torch.as_tensor(np.asarray(triangles), dtype=torch.long))


# ---------------------------------------------------------------- the law

def test_endpoints_are_exact():
    material = make_material(degradation_type="rational_at2")
    d = torch.tensor([0.0, 1.0], dtype=DTYPE)
    g, _, _ = material.rational_at2_degradation_derivatives(d)
    assert float(g[0]) == pytest.approx(1.0, abs=0.0)
    assert float(g[1]) == pytest.approx(0.0, abs=0.0)


def test_law_is_monotone_decreasing():
    material = make_material(degradation_type="rational_at2")
    d = torch.linspace(0.0, 1.0, 101, dtype=DTYPE)
    g, gp, _ = material.rational_at2_degradation_derivatives(d)
    assert bool((g.diff() < 0.0).all())
    assert bool((gp <= 0.0).all())


def test_derivatives_match_autograd():
    material = make_material(degradation_type="rational_at2")
    d = torch.linspace(0.0, 1.0, 51, dtype=DTYPE).requires_grad_(True)
    closed_g, closed_gp, closed_gpp = (
        material.rational_at2_degradation_derivatives(d.detach()))

    g = (1.0 - d) ** 2 / (d * d - d + 1.0)
    first = torch.autograd.grad(g.sum(), d, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), d)[0]

    assert torch.allclose(g.detach(), closed_g, atol=1e-14)
    assert torch.allclose(first.detach(), closed_gp, atol=1e-12)
    assert torch.allclose(second.detach(), closed_gpp, atol=1e-10)


def test_degradation_dispatches_to_the_law():
    material = make_material(degradation_type="rational_at2")
    d = torch.linspace(0.0, 1.0, 11, dtype=DTYPE)
    expected, _, _ = material.rational_at2_degradation_derivatives(d)
    assert torch.equal(material.degradation(d), expected)


def test_it_differs_from_the_wu_rational_law():
    d = torch.linspace(0.1, 0.9, 9, dtype=DTYPE)
    parameter_free = make_material(degradation_type="rational_at2").degradation(d)
    wu = make_material(degradation_type="rational").degradation(d)
    assert not torch.allclose(parameter_free, wu, atol=1e-6)


def test_it_requires_at2():
    with pytest.raises(ValueError, match="rational_at2.*AT2"):
        make_material(degradation_type="rational_at2", pf_model="AT1")


# ---------------------------------------------------------- stress degradation

def test_stress_degradation_defaults_to_the_energy_split():
    material = make_material()
    assert material.stress_degradation == "split"
    assert material.effective_stress_split() == "spectral"
    assert stress_split_of(material) == "spectral"


def test_full_stress_degradation_resolves_to_the_isotropic_tangent():
    material = make_material(stress_degradation="full")
    assert material.effective_stress_split() == "isotropic"
    assert stress_split_of(material) == "isotropic"


def test_full_stress_degradation_scales_the_undamaged_stress():
    nodes, elements = unit_square_mesh()
    mesh = FEMMesh.from_tensors(nodes, elements, device="cpu")
    material = make_material(stress_degradation="full")
    fem = FEMOperators(mesh, material)

    # A uniform biaxial stretch, so every element sees the same strain.
    u = nodes * 1.0e-3
    d_zero = torch.zeros(nodes.shape[0], dtype=DTYPE)
    d_half = torch.full((nodes.shape[0],), 0.5, dtype=DTYPE)

    intact = torch.stack(fem.compute_stress(u, d_zero))
    damaged = torch.stack(fem.compute_stress(u, d_half))
    factor = float(material.degradation(torch.tensor(0.5, dtype=DTYPE)))

    assert torch.allclose(damaged, intact * factor, rtol=1e-12, atol=0.0)


def test_an_unknown_stress_degradation_is_rejected():
    with pytest.raises(ValueError, match="stress_degradation must be"):
        make_material(stress_degradation="partial")


def test_stress_split_falls_back_for_a_material_without_the_contract():
    class Legacy:
        energy_split = "amor"

    assert stress_split_of(Legacy()) == "amor"


# ---------------------------------------------------------------- the solve

def build_solver(**material_overrides):
    nodes, elements = unit_square_mesh()
    mesh = FEMMesh.from_tensors(nodes, elements, device="cpu")
    material = make_material(**material_overrides)
    fem = FEMOperators(mesh, material)
    return mesh, PhaseFieldDamageSolver(fem, tol=1e-10, max_iter=500)


def driving_history(mesh, magnitude):
    centroids = mesh.nodes[mesh.elements].mean(1)
    radius = ((centroids[:, 0] - 0.5) ** 2 + (centroids[:, 1] - 0.5) ** 2)
    return magnitude * torch.exp(-radius / 0.02)


def test_the_projected_solve_is_selected_and_stays_in_bounds():
    mesh, solver = build_solver(degradation_type="rational_at2")
    d_previous = torch.zeros(mesh.nodes.shape[0], dtype=DTYPE)

    d = solver.solve(driving_history(mesh, 40.0), d_previous)

    assert solver.last_backend == "rational_at2_projected"
    assert bool((d >= -1e-12).all()) and bool((d <= 1.0 + 1e-12).all())
    assert float(d.max()) > 0.1


def test_the_solve_respects_irreversibility():
    mesh, solver = build_solver(degradation_type="rational_at2")
    d_previous = torch.full((mesh.nodes.shape[0],), 0.3, dtype=DTYPE)

    d = solver.solve(driving_history(mesh, 5.0), d_previous)

    assert bool((d >= d_previous - 1e-12).all())


def test_a_vanishing_history_leaves_damage_at_its_previous_value():
    mesh, solver = build_solver(degradation_type="rational_at2")
    d_previous = torch.full((mesh.nodes.shape[0],), 0.2, dtype=DTYPE)

    d = solver.solve(torch.zeros(mesh.elements.shape[0], dtype=DTYPE), d_previous)

    assert torch.allclose(d, d_previous, atol=1e-10)


def test_the_kkt_audit_accepts_the_converged_field():
    mesh, solver = build_solver(degradation_type="rational_at2")
    d_previous = torch.zeros(mesh.nodes.shape[0], dtype=DTYPE)
    history = driving_history(mesh, 40.0)

    d = solver.solve(history, d_previous)
    metrics = phast.solvers.damage_solver.damage_kkt_metrics(
        solver.compute_residual(history, d), d, d_previous)

    assert metrics["kkt_feasible"] is True
    assert metrics["kkt_projected_linf"] < 1e-4


def test_the_standard_law_still_uses_the_linear_route():
    mesh, solver = build_solver()
    d = solver.solve(driving_history(mesh, 40.0),
                     torch.zeros(mesh.nodes.shape[0], dtype=DTYPE))
    assert getattr(solver, "last_backend", None) != "rational_at2_projected"
    assert bool((d >= -1e-12).all()) and bool((d <= 1.0 + 1e-12).all())


def test_an_unsupported_degradation_law_is_still_rejected():
    nodes, elements = unit_square_mesh(4)
    mesh = FEMMesh.from_tensors(nodes, elements, device="cpu")
    fem = FEMOperators(mesh, make_material(degradation_type="cubic"))
    with pytest.raises(NotImplementedError, match="rational_at2"):
        PhaseFieldDamageSolver(fem, tol=1e-8, max_iter=50)
