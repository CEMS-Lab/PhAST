from __future__ import annotations

import pytest
import torch

from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mechanics_solver import QuasiStaticSolver
from phast.mesh import FEMMesh
from phast.cohesive_elements import (
    BilinearCohesiveLaw,
    CohesiveElement,
    CohesiveInterfaceOperator,
)
from phast.plasticity import (
    DuctilePhaseFieldCoupling,
    MeshJ2Elastoplasticity,
    SparseJ2QuasiStaticSolver,
    strain3d_from_mesh,
)


def _one_square_mesh() -> FEMMesh:
    nodes = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    elements = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return FEMMesh.from_tensors(nodes, elements, device="cpu", dtype=torch.float64)


def _j2_material(**overrides) -> Material:
    params = dict(
        E=210_000.0,
        nu=0.30,
        Gc=2.7,
        l0=0.1,
        rho=7.8e-9,
        energy_split="amor",
        plasticity_model="j2_isotropic",
        yield_stress=250.0,
        hardening_modulus=5_000.0,
        hardening_type="linear_iso",
        plane_stress=True,
    )
    params.update(overrides)
    return Material(**params)


def _affine_displacement(mesh: FEMMesh, eps_xx: float,
                         eps_yy: float = 0.0,
                         gamma_xy: float = 0.0) -> torch.Tensor:
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    u = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype, device=mesh.device)
    u[:, 0] = eps_xx * x + 0.5 * gamma_xy * y
    u[:, 1] = 0.5 * gamma_xy * x + eps_yy * y
    return u


def test_mesh_j2_uniform_affine_strain_and_commit():
    mesh = _one_square_mesh()
    mech = MeshJ2Elastoplasticity(mesh, _j2_material())
    u = _affine_displacement(mesh, eps_xx=4.0e-3)

    trial = mech.update_trial(u)
    expected = strain3d_from_mesh(mesh, u)
    assert torch.allclose(trial.strain, expected)
    assert torch.all(trial.eps_p_eq > 0.0)
    assert torch.all(trial.plastic_work_density > 0.0)

    committed = mech.commit()
    assert torch.allclose(committed.eps_p_eq, trial.eps_p_eq)
    assert mech._trial_state is None


def test_mesh_j2_rollback_discards_trial_state():
    mesh = _one_square_mesh()
    mech = MeshJ2Elastoplasticity(mesh, _j2_material())
    initial = mech.state.clone()
    mech.update_trial(_affine_displacement(mesh, eps_xx=4.0e-3))

    mech.rollback()
    assert mech._trial_state is None
    assert torch.allclose(mech.state.eps_p_eq, initial.eps_p_eq)
    assert torch.allclose(mech.state.plastic_work_density,
                          initial.plastic_work_density)


def test_mesh_j2_internal_force_degrades_with_damage():
    mesh = _one_square_mesh()
    mech = MeshJ2Elastoplasticity(mesh, _j2_material())
    trial = mech.update_trial(_affine_displacement(mesh, eps_xx=4.0e-3))
    f_intact = mech.internal_force(state=trial)
    d = torch.full((mesh.n_nodes,), 0.5, dtype=mesh.dtype, device=mesh.device)
    f_damaged = mech.internal_force(d=d, state=trial)

    assert f_damaged.norm().item() < f_intact.norm().item()


def test_mesh_j2_local_algorithmic_tangent_matches_finite_difference():
    mesh = _one_square_mesh()
    mech = MeshJ2Elastoplasticity(mesh, _j2_material())
    u = _affine_displacement(mesh, eps_xx=4.0e-3, eps_yy=-8.0e-4)

    C_alg = mech.inplane_algorithmic_tangent(u)
    strain_base = strain3d_from_mesh(mesh, u)
    eps0 = torch.stack(
        [strain_base[:, 0], strain_base[:, 1], strain_base[:, 3]], dim=1)
    h = 1.0e-7
    C_fd = torch.zeros_like(C_alg)
    for col in range(3):
        direction = torch.zeros_like(eps0)
        direction[:, col] = 1.0
        strain_plus = torch.zeros_like(strain_base)
        strain_minus = torch.zeros_like(strain_base)
        eps_plus = eps0 + h * direction
        eps_minus = eps0 - h * direction
        strain_plus[:, 0] = eps_plus[:, 0]
        strain_plus[:, 1] = eps_plus[:, 1]
        strain_plus[:, 3] = eps_plus[:, 2]
        strain_minus[:, 0] = eps_minus[:, 0]
        strain_minus[:, 1] = eps_minus[:, 1]
        strain_minus[:, 3] = eps_minus[:, 2]
        stress_plus, _, _ = mech.kernel.step(
            mech.state.strain, strain_plus, mech.state.stress,
            mech.state.plastic_strain, mech.state.eps_p_eq)
        stress_minus, _, _ = mech.kernel.step(
            mech.state.strain, strain_minus, mech.state.stress,
            mech.state.plastic_strain, mech.state.eps_p_eq)
        sig_plus = torch.stack(
            [stress_plus[:, 0], stress_plus[:, 1], stress_plus[:, 3]], dim=1)
        sig_minus = torch.stack(
            [stress_minus[:, 0], stress_minus[:, 1], stress_minus[:, 3]], dim=1)
        C_fd[:, :, col] = (sig_plus - sig_minus) / (2.0 * h)

    assert torch.allclose(C_alg, C_fd, rtol=3.0e-4, atol=5.0e-3)


def test_mesh_j2_local_algorithmic_tangent_plane_strain_matches_finite_difference():
    mesh = _one_square_mesh()
    mech = MeshJ2Elastoplasticity(mesh, _j2_material(plane_stress=False))
    u = _affine_displacement(mesh, eps_xx=4.0e-3, eps_yy=-8.0e-4)

    C_alg = mech.inplane_algorithmic_tangent(u)
    strain_base = strain3d_from_mesh(mesh, u)
    eps0 = torch.stack(
        [strain_base[:, 0], strain_base[:, 1], strain_base[:, 3]], dim=1)
    h = 1.0e-7
    C_fd = torch.zeros_like(C_alg)
    for col in range(3):
        direction = torch.zeros_like(eps0)
        direction[:, col] = 1.0
        strain_plus = torch.zeros_like(strain_base)
        strain_minus = torch.zeros_like(strain_base)
        eps_plus = eps0 + h * direction
        eps_minus = eps0 - h * direction
        strain_plus[:, 0] = eps_plus[:, 0]
        strain_plus[:, 1] = eps_plus[:, 1]
        strain_plus[:, 3] = eps_plus[:, 2]
        strain_minus[:, 0] = eps_minus[:, 0]
        strain_minus[:, 1] = eps_minus[:, 1]
        strain_minus[:, 3] = eps_minus[:, 2]
        stress_plus, _, _ = mech.kernel.step(
            mech.state.strain, strain_plus, mech.state.stress,
            mech.state.plastic_strain, mech.state.eps_p_eq)
        stress_minus, _, _ = mech.kernel.step(
            mech.state.strain, strain_minus, mech.state.stress,
            mech.state.plastic_strain, mech.state.eps_p_eq)
        sig_plus = torch.stack(
            [stress_plus[:, 0], stress_plus[:, 1], stress_plus[:, 3]], dim=1)
        sig_minus = torch.stack(
            [stress_minus[:, 0], stress_minus[:, 1], stress_minus[:, 3]], dim=1)
        C_fd[:, :, col] = (sig_plus - sig_minus) / (2.0 * h)

    assert torch.allclose(C_alg, C_fd, rtol=3.0e-4, atol=5.0e-3)


def test_mesh_j2_assembled_tangent_action_matches_internal_force_difference():
    mesh = _one_square_mesh()
    mech = MeshJ2Elastoplasticity(mesh, _j2_material())
    u = _affine_displacement(mesh, eps_xx=4.0e-3, eps_yy=-8.0e-4)
    du = _affine_displacement(mesh, eps_xx=2.0e-4, gamma_xy=1.0e-4)
    d = torch.full((mesh.n_nodes,), 0.2, dtype=mesh.dtype)

    K = mech.assemble_tangent(u, d=d)
    tangent_action = torch.sparse.mm(
        K, du.reshape(-1, 1)).reshape_as(du)
    h = 1.0e-6
    state_plus = mech.update_trial(u + h * du)
    f_plus = mech.internal_force(d=d, state=state_plus)
    state_minus = mech.update_trial(u - h * du)
    f_minus = mech.internal_force(d=d, state=state_minus)
    fd_action = (f_plus - f_minus) / (2.0 * h)

    assert torch.allclose(tangent_action, fd_action, rtol=5.0e-4, atol=2.0e-5)


def test_ductile_pf_driving_force_includes_plastic_work():
    mesh = _one_square_mesh()
    material = _j2_material()
    mech = MeshJ2Elastoplasticity(mesh, material)
    fem = FEMOperators(mesh, material)
    coupling = DuctilePhaseFieldCoupling(
        fem=fem, plasticity=mech, plastic_work_weight=2.0)
    u = _affine_displacement(mesh, eps_xx=4.0e-3)
    trial = mech.update_trial(u)

    psi = fem.compute_psi_plus(u)
    driving = coupling.driving_force(u, state=trial)
    assert torch.all(driving >= psi)
    assert torch.any(driving > psi)

    H_old = torch.zeros_like(driving)
    H_new = coupling.history_update(H_old, u, state=trial)
    assert torch.allclose(H_new, driving)
    H_again = coupling.history_update(H_new + 1.0, u, state=trial)
    assert torch.allclose(H_again, H_new + 1.0)


def test_sparse_j2_solver_commits_after_converged_prescribed_patch():
    mesh = _one_square_mesh()
    mech = MeshJ2Elastoplasticity(mesh, _j2_material())
    solver = SparseJ2QuasiStaticSolver(mech, max_iter=2)

    bc_mask = torch.ones((mesh.n_nodes, 2), dtype=torch.bool)
    bc_vals = _affine_displacement(mesh, eps_xx=4.0e-3)
    u, converged, n_iter = solver.solve(bc_mask, bc_vals)

    assert converged
    assert n_iter == 0
    assert torch.allclose(u, bc_vals)
    assert solver.last_converged
    assert mech._trial_state is None
    assert torch.all(mech.state.eps_p_eq > 0.0)
    assert torch.all(mech.state.plastic_work_density > 0.0)


def test_sparse_j2_solver_solves_free_patch_equilibrium():
    mesh = _one_square_mesh()
    mech = MeshJ2Elastoplasticity(mesh, _j2_material())
    solver = SparseJ2QuasiStaticSolver(
        mech, tol=1.0e-6, tol_rel=1.0e-7, max_iter=12)

    bc_mask = torch.zeros((mesh.n_nodes, 2), dtype=torch.bool)
    bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)
    left = torch.tensor([0, 3], dtype=torch.long)
    right = torch.tensor([1, 2], dtype=torch.long)
    bc_mask[left, :] = True
    bc_mask[right, 0] = True
    bc_vals[right, 0] = 4.0e-3

    u, converged, _ = solver.solve(bc_mask, bc_vals)
    free_mask = (~bc_mask).to(mesh.dtype)
    trial = mech.update_trial(u)
    residual = -mech.internal_force(state=trial) * free_mask

    assert converged
    assert solver.last_residual <= 2.0e-6
    assert residual.norm().item() <= 5.0e-5
    assert solver.last_backend in {"scipy", "mumps", "cudss"}
    assert mech._trial_state is not None
    mech.rollback()


def test_sparse_j2_solver_rolls_back_failed_step():
    mesh = _one_square_mesh()
    mech = MeshJ2Elastoplasticity(mesh, _j2_material())
    initial = mech.state.clone()
    solver = SparseJ2QuasiStaticSolver(mech, max_iter=0)

    bc_mask = torch.ones((mesh.n_nodes, 2), dtype=torch.bool)
    bc_mask[2, 1] = False
    bc_vals = _affine_displacement(mesh, eps_xx=4.0e-3)
    _, converged, _ = solver.solve(bc_mask, bc_vals)

    assert not converged
    assert solver.last_failure == "maximum iterations reached"
    assert mech._trial_state is None
    assert torch.allclose(mech.state.eps_p_eq, initial.eps_p_eq)
    assert torch.allclose(
        mech.state.plastic_work_density, initial.plastic_work_density)


def test_quasistatic_solver_dispatches_to_sparse_j2_operator():
    mesh = _one_square_mesh()
    material = _j2_material()
    fem = FEMOperators(mesh, material)
    mech = MeshJ2Elastoplasticity(mesh, material)
    solver = QuasiStaticSolver(
        fem, plasticity_operator=mech, backend="auto",
        tol=1.0e-6, tol_rel=1.0e-7, max_iter=12)

    bc_mask = torch.zeros((mesh.n_nodes, 2), dtype=torch.bool)
    bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)
    left = torch.tensor([0, 3], dtype=torch.long)
    right = torch.tensor([1, 2], dtype=torch.long)
    bc_mask[left, :] = True
    bc_mask[right, 0] = True
    bc_vals[right, 0] = 4.0e-3
    d = torch.zeros(mesh.n_nodes, dtype=mesh.dtype)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)

    u, converged, _ = solver.solve(d, f_ext, bc_mask, bc_vals)

    assert converged
    assert solver.last_backend in {"scipy", "mumps", "cudss"}
    assert torch.allclose(u[bc_mask], bc_vals[bc_mask])
    assert torch.all(mech.state.eps_p_eq > 0.0)


def test_quasistatic_solver_allows_cudss_for_sparse_j2_operator():
    mesh = _one_square_mesh()
    material = _j2_material()
    fem = FEMOperators(mesh, material)
    mech = MeshJ2Elastoplasticity(mesh, material)
    solver = QuasiStaticSolver(
        fem, plasticity_operator=mech, backend="cudss",
        tol=1.0e-6, tol_rel=1.0e-7, max_iter=12)

    bc_mask = torch.zeros((mesh.n_nodes, 2), dtype=torch.bool)
    bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)
    left = torch.tensor([0, 3], dtype=torch.long)
    right = torch.tensor([1, 2], dtype=torch.long)
    bc_mask[left, :] = True
    bc_mask[right, 0] = True
    bc_vals[right, 0] = 4.0e-3
    d = torch.zeros(mesh.n_nodes, dtype=mesh.dtype)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)

    u, converged, _ = solver.solve(d, f_ext, bc_mask, bc_vals)

    assert converged
    assert solver.last_backend in {"scipy", "mumps", "cudss"}
    assert torch.allclose(u[bc_mask], bc_vals[bc_mask])
    assert torch.all(mech.state.eps_p_eq > 0.0)


def test_quasistatic_solver_rejects_cudss_for_elastic_path():
    mesh = _one_square_mesh()
    material = Material(E=210_000.0, nu=0.30, energy_split="isotropic")
    fem = FEMOperators(mesh, material)
    solver = QuasiStaticSolver(fem, backend="cudss")
    d = torch.zeros(mesh.n_nodes, dtype=mesh.dtype)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)
    bc_mask = torch.ones((mesh.n_nodes, 2), dtype=torch.bool)
    bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)

    with pytest.raises(NotImplementedError, match="sparse J2"):
        solver.solve(d, f_ext, bc_mask, bc_vals)


def test_quasistatic_solver_rejects_j2_cg_backend():
    mesh = _one_square_mesh()
    material = _j2_material()
    fem = FEMOperators(mesh, material)
    mech = MeshJ2Elastoplasticity(mesh, material)
    solver = QuasiStaticSolver(fem, plasticity_operator=mech, backend="cg")
    bc_mask = torch.ones((mesh.n_nodes, 2), dtype=torch.bool)
    bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)
    d = torch.zeros(mesh.n_nodes, dtype=mesh.dtype)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=mesh.dtype)

    with pytest.raises(NotImplementedError, match="sparse tangent"):
        solver.solve(d, f_ext, bc_mask, bc_vals)


def test_quasistatic_solver_rejects_coupled_j2_and_cohesive_operator():
    import numpy as np

    mesh = _one_square_mesh()
    material = _j2_material(energy_split="isotropic")
    fem = FEMOperators(mesh, material)
    mech = MeshJ2Elastoplasticity(mesh, material)
    cohesive = CohesiveInterfaceOperator(
        [
            CohesiveElement(
                nodes_top=(0, 1),
                nodes_bottom=(3, 2),
                normal=np.array([0.0, 1.0]),
                tangent=np.array([1.0, 0.0]),
                length=1.0,
            )
        ],
        BilinearCohesiveLaw(
            k_n=1000.0, k_t=500.0, sigma_max=10.0, delta_c=0.1),
        n_nodes=mesh.n_nodes,
        device="cpu",
    )

    with pytest.raises(
            NotImplementedError,
            match="plasticity_operator \\+ cohesive_operator"):
        QuasiStaticSolver(
            fem,
            plasticity_operator=mech,
            cohesive_operator=cohesive,
            backend="auto",
        )


def test_mesh_j2_rejects_elastic_material():
    with pytest.raises(ValueError, match="plasticity_model"):
        MeshJ2Elastoplasticity(_one_square_mesh(), Material())
