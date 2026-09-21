"""Rigid-connector dispatch and small elastic mechanics regressions."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from phast.core.mesh import FEMMesh
from phast.physics.boundary_conditions import BoundaryConditions
from phast.physics.material import Material
from phast.solvers.mechanics_solver import QuasiStaticSolver
from phast.solvers.staggered_solver import SolverConfig, StaggeredSolver


def dispatch_fixture(rotation_free: bool) -> tuple[StaggeredSolver, Mock]:
    solver = StaggeredSolver.__new__(StaggeredSolver)
    solver.config = SolverConfig(solver_type="quasi_static")
    solver.bcs = BoundaryConditions(3, device="cpu")
    solver.bcs.add_rigid_connector(
        0, torch.tensor([1, 2]), [0, 1], {0: 0.01},
        rotation_free=rotation_free,
    )
    solver.u = torch.zeros(3, 2, dtype=torch.float64)
    solver.d = torch.zeros(3, dtype=torch.float64)
    solver.f_ext = torch.zeros_like(solver.u)
    solve = Mock(return_value=(torch.ones_like(solver.u), True, 2))
    solver.mechanics = SimpleNamespace(
        solve=solve, last_iter=2, last_converged=True,
        last_residual=1.0e-12, last_residual0=1.0,
        last_relative_residual=1.0e-12,
    )
    return solver, solve


@pytest.mark.parametrize("rotation_free", [False, True])
def test_quasistatic_dispatch_preserves_connector_and_state(
    rotation_free: bool,
) -> None:
    solver, solve = dispatch_fixture(rotation_free)
    original_u = solver.u
    mask, values = solver.bcs.get_masks_and_values()

    solver.step_mechanics()

    solve.assert_called_once()
    call = solve.call_args
    assert call.args[0] is solver.d
    assert call.args[1] is solver.f_ext
    torch.testing.assert_close(call.args[2], mask)
    torch.testing.assert_close(call.args[3], values)
    assert call.kwargs["u_init"] is original_u
    assert call.kwargs["rigid_connectors"] == (
        solver.bcs.get_active_rigid_connectors() or None
    )
    assert solver.u is solve.return_value[0]
    assert solver._last_mechanics_converged
    assert solver._last_mechanics_iter == 2


def test_quasistatic_connector_dispatch_keeps_nonconvergence_guard() -> None:
    solver, solve = dispatch_fixture(True)
    original_u = solver.u
    solve.return_value = (torch.ones_like(solver.u), False, 2)
    solver.mechanics.last_converged = False

    with pytest.raises(RuntimeError, match="Quasi-static mechanics solve"):
        solver.step_mechanics()

    assert solve.call_args.kwargs["rigid_connectors"] == solver.bcs.rigid_connectors
    assert solver.u is original_u
    assert not solver._last_mechanics_converged


def elastic_fixture(connector: bool) -> StaggeredSolver:
    nodes = torch.tensor(
        [[0, 0], [1, 0], [0, 0.5], [1, 0.5], [0, 1], [1, 1]],
        dtype=torch.float64,
    )
    elements = torch.tensor([[0, 1, 3], [0, 3, 2], [2, 3, 4], [3, 5, 4]])
    mesh = FEMMesh.from_tensors(nodes, elements, device="cpu")
    bcs = BoundaryConditions(mesh.n_nodes, device="cpu")
    for component in (0, 1):
        bcs.fix(torch.tensor([0, 2, 4]), component)
    if connector:
        bcs.add_rigid_connector(3, torch.tensor([1, 5]), [0, 1], {0: 0.01})
    else:
        bcs.add(torch.tensor([3]), 0, 0.01)
        bcs.fix(torch.tensor([3]), 1)
    return StaggeredSolver(
        mesh,
        Material(E=100.0, nu=0.25, Gc=1.0, l0=0.1, rho=1.0,
                 energy_split="isotropic", plane_stress=True, eta_residual=0.0),
        bcs,
        SolverConfig(solver_type="quasi_static", backend="scipy",
                     static_tol=1.0e-10, static_max_iter=5,
                     damage_max_iter=100,
                     use_multigrid=False, enable_damage=False),
    )


@pytest.mark.parametrize("couple", [0.0, 0.2])
def test_quasistatic_elastic_connector_constraints_and_equilibrium(couple: float) -> None:
    solver = elastic_fixture(True)
    solver.f_ext[1, 0] = -couple
    solver.f_ext[5, 0] = couple
    mask, values = solver.bcs.get_masks_and_values()
    reference, converged, _ = QuasiStaticSolver(
        solver.fem, backend="scipy", tol=1.0e-10, max_iter=5,
    ).solve(solver.d, solver.f_ext, mask, values, u_init=solver.u,
            rigid_connectors=solver.bcs.get_active_rigid_connectors())
    assert converged

    solver.step_mechanics()

    assert torch.isfinite(solver.u).all()
    torch.testing.assert_close(solver.u, reference, atol=1.0e-12, rtol=1.0e-10)
    torch.testing.assert_close(solver.u[mask], values[mask], atol=1.0e-12, rtol=0)
    slaves = torch.tensor([1, 5])
    offset_y = solver.mesh.nodes[slaves, 1] - solver.mesh.nodes[3, 1]
    theta = -(solver.u[5, 0] - solver.u[3, 0]) / offset_y[1]
    torch.testing.assert_close(
        solver.u[slaves, 0], solver.u[3, 0] - theta * offset_y,
        atol=1.0e-12, rtol=0,
    )
    torch.testing.assert_close(
        solver.u[slaves, 1], solver.u[3, 1].expand(2), atol=1.0e-12, rtol=0,
    )
    internal = solver.fem.internal_force(solver.u, solver.d)
    # The only free reduced degree of freedom is the connector rotation.
    moment_residual = torch.dot(-offset_y, (internal - solver.f_ext)[slaves, 0])
    assert abs(float(moment_residual)) < 1.0e-10
    torch.testing.assert_close(internal.sum(dim=0), torch.zeros(2, dtype=torch.float64),
                               atol=1.0e-10, rtol=0)
    if couple == 0.0:
        expected = torch.zeros_like(solver.u)
        expected[:, 0] = 0.01 * solver.mesh.nodes[:, 0]
        torch.testing.assert_close(solver.u, expected, atol=1.0e-12, rtol=0)
        expected_reaction = 100.0 * 0.01 / (1.0 - 0.25**2)
        assert float(internal[[1, 3, 5], 0].sum()) == pytest.approx(
            expected_reaction, rel=1.0e-10,
        )
    else:
        assert abs(float(theta)) > 1.0e-6


def test_quasistatic_without_connector_matches_direct_mechanics() -> None:
    solver = elastic_fixture(False)
    mask, values = solver.bcs.get_masks_and_values()
    reference, converged, _ = QuasiStaticSolver(
        solver.fem, backend="scipy", tol=1.0e-10, max_iter=5,
    ).solve(solver.d, solver.f_ext, mask, values, u_init=solver.u)
    assert converged

    solver.step_mechanics()

    torch.testing.assert_close(solver.u, reference, atol=1.0e-12, rtol=1.0e-10)
