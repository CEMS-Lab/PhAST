from __future__ import annotations

import meshio
import pytest
import torch

from phast.damage_solver import PhaseFieldDamageSolver
from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mechanics_solver import QuasiStaticSolver
from phast.mesh import FEMMesh
from phast.quad_elements import (
    q4_internal_force,
    q4_laplacian_matvec,
    q4_quality,
    q4_signed_areas,
    q4_to_triangles,
    structured_q4_mesh,
)


def test_structured_q4_mesh_boundary_sets_and_quality():
    mesh = structured_q4_mesh(width=2.0, height=1.0, nx=2, ny=1)
    assert mesh.nodes.shape == (6, 2)
    assert mesh.quads.shape == (2, 4)
    assert mesh.node_sets["left"].tolist() == [0, 3]
    assert mesh.node_sets["right"].tolist() == [2, 5]
    assert mesh.node_sets["bottom"].tolist() == [0, 1, 2]
    assert mesh.node_sets["top"].tolist() == [3, 4, 5]

    quality = q4_quality(mesh.nodes, mesh.quads)
    assert quality["n_quads"] == 2
    assert quality["all_positive_orientation"]
    assert quality["min_signed_area"] > 0.0
    assert quality["max_aspect_ratio"] == 1.0


def test_q4_to_triangles_preserves_area_and_solver_mesh_path():
    qmesh = structured_q4_mesh(width=2.0, height=1.0, nx=2, ny=2)
    triangles = q4_to_triangles(qmesh.quads)
    assert triangles.shape == (8, 3)

    tri_mesh = FEMMesh.from_tensors(
        qmesh.nodes, triangles, node_sets=qmesh.node_sets,
        device="cpu", dtype=torch.float64,
    )
    assert tri_mesh.n_nodes == qmesh.nodes.shape[0]
    assert tri_mesh.n_elems == 2 * qmesh.quads.shape[0]
    assert tri_mesh.areas.sum().item() == 2.0
    assert set(tri_mesh.node_sets) == {"left", "right", "bottom", "top"}


def test_q4_to_triangles_supports_both_diagonals():
    qmesh = structured_q4_mesh(width=1.0, height=1.0, nx=1, ny=1)
    tri_02 = q4_to_triangles(qmesh.quads, diagonal="02")
    tri_13 = q4_to_triangles(qmesh.quads, diagonal="13")
    assert tri_02.tolist() == [[0, 1, 3], [0, 3, 2]]
    assert tri_13.tolist() == [[0, 1, 2], [1, 3, 2]]

    area = q4_signed_areas(qmesh.nodes, qmesh.quads)
    assert area.shape == (1,)
    assert area[0].item() == 1.0


def test_femmesh_from_tensors_accepts_native_q4():
    qmesh = structured_q4_mesh(width=2.0, height=1.0, nx=2, ny=1)
    mesh = FEMMesh.from_tensors(
        qmesh.nodes, qmesh.quads, node_sets=qmesh.node_sets,
        device="cpu", dtype=torch.float64, element_type="Q4")

    assert mesh.element_type == "Q4"
    assert mesh.n_elem_nodes == 4
    assert mesh.quad_grad_phi.shape == (2, 4, 4, 2)
    assert mesh.quad_wdetJ.shape == (2, 4)
    assert mesh.areas.sum().item() == 2.0
    assert torch.all(mesh.M_scalar > 0.0)


def test_femmesh_loads_gmsh_quad_cells(tmp_path):
    qmesh = structured_q4_mesh(width=1.0, height=1.0, nx=1, ny=1)
    path = tmp_path / "one_quad.msh"
    meshio.write(
        path,
        meshio.Mesh(
            points=torch.nn.functional.pad(qmesh.nodes, (0, 1)).numpy(),
            cells=[("quad", qmesh.quads.numpy())],
        ),
    )

    mesh = FEMMesh(str(path), device="cpu", dtype=torch.float64)

    assert mesh.element_type == "Q4"
    assert mesh.elements.shape == (1, 4)
    assert mesh.areas.sum().item() == 1.0


def test_q4_femoperators_internal_force_matches_primitive():
    qmesh = structured_q4_mesh(width=1.0, height=1.0, nx=1, ny=1)
    mesh = FEMMesh.from_tensors(
        qmesh.nodes, qmesh.quads, node_sets=qmesh.node_sets,
        device="cpu", dtype=torch.float64, element_type="Q4")
    material = Material(E=210.0, nu=0.30, energy_split="isotropic")
    fem = FEMOperators(mesh, material)
    u = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    u[:, 0] = 0.01 * mesh.nodes[:, 0]
    u[:, 1] = -0.002 * mesh.nodes[:, 1]
    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)

    got = fem.internal_force(u, d)
    expected = q4_internal_force(
        mesh.nodes, mesh.elements, u, E=material.E, nu=material.nu,
        plane_strain=not material.plane_stress)

    assert torch.allclose(got, expected, atol=1.0e-10, rtol=1.0e-10)


def test_q4_femoperators_laplacian_matches_primitive():
    qmesh = structured_q4_mesh(width=1.0, height=1.0, nx=2, ny=2)
    mesh = FEMMesh.from_tensors(
        qmesh.nodes, qmesh.quads, node_sets=qmesh.node_sets,
        device="cpu", dtype=torch.float64, element_type="Q4")
    material = Material(E=210.0, nu=0.30, energy_split="isotropic")
    fem = FEMOperators(mesh, material)
    x = mesh.nodes[:, 0] ** 2 + 0.5 * mesh.nodes[:, 1]

    got = fem.laplacian_matvec(x)
    expected = q4_laplacian_matvec(mesh.nodes, mesh.elements, x)

    assert torch.allclose(got, expected, atol=1.0e-10, rtol=1.0e-10)


def test_q4_at2_damage_solver_accepts_gauss_point_history():
    qmesh = structured_q4_mesh(width=1.0, height=1.0, nx=2, ny=1)
    mesh = FEMMesh.from_tensors(
        qmesh.nodes, qmesh.quads, node_sets=qmesh.node_sets,
        device="cpu", dtype=torch.float64, element_type="Q4")
    material = Material(
        E=210.0, nu=0.30, Gc=1.0, l0=0.2,
        energy_split="isotropic", pf_model="AT2")
    fem = FEMOperators(mesh, material)
    u = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    u[:, 0] = 0.01 * mesh.nodes[:, 0]
    u[:, 1] = -0.002 * mesh.nodes[:, 1]
    H = fem.compute_psi_plus(u)
    d_prev = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    solver = PhaseFieldDamageSolver(
        fem, max_iter=500, tol=1.0e-10,
        use_multigrid=False, bounds_method="post_clamp")

    d = solver.solve(H, d_prev)
    residual = solver.compute_residual(H, d)

    assert H.shape == (mesh.n_elems, 4)
    assert solver.last_iter < solver.max_iter
    assert torch.isfinite(d).all()
    assert torch.all(d >= d_prev)
    assert torch.all(d <= 1.0)
    assert residual.norm().item() <= 1.0e-7


def test_q4_damage_solver_keeps_unsupported_variants_guarded():
    qmesh = structured_q4_mesh(width=1.0, height=1.0, nx=1, ny=1)
    mesh = FEMMesh.from_tensors(
        qmesh.nodes, qmesh.quads, node_sets=qmesh.node_sets,
        device="cpu", dtype=torch.float64, element_type="Q4")

    pfczm_material = Material(
        E=210.0, nu=0.30, energy_split="isotropic",
        pf_model="PFCZM", sigma_ts=3.0)
    with pytest.raises(NotImplementedError, match="PF-CZM"):
        PhaseFieldDamageSolver(FEMOperators(mesh, pfczm_material))

    at2_material = Material(
        E=210.0, nu=0.30, energy_split="isotropic", pf_model="AT2")
    solver = PhaseFieldDamageSolver(
        FEMOperators(mesh, at2_material), bounds_method="direct", max_iter=100)
    H = torch.zeros((mesh.n_elems, 4), dtype=torch.float64)
    d_prev = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="direct"):
        solver.solve(H, d_prev)


def test_q4_sparse_direct_stiffness_matches_internal_force_action():
    qmesh = structured_q4_mesh(width=1.0, height=1.0, nx=2, ny=1)
    mesh = FEMMesh.from_tensors(
        qmesh.nodes, qmesh.quads, node_sets=qmesh.node_sets,
        device="cpu", dtype=torch.float64, element_type="Q4")
    material = Material(E=210.0, nu=0.30, energy_split="isotropic")
    fem = FEMOperators(mesh, material)
    solver = QuasiStaticSolver(fem, backend="scipy", max_iter=1)
    d = 0.2 * mesh.nodes[:, 0] + 0.1 * mesh.nodes[:, 1]
    direction = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    direction[:, 0] = 0.01 * mesh.nodes[:, 0] + 0.002 * mesh.nodes[:, 1]
    direction[:, 1] = -0.003 * mesh.nodes[:, 0] + 0.004 * mesh.nodes[:, 1]

    indices, values, n_dof = solver._assemble_K_isotropic(d)
    K = torch.sparse_coo_tensor(
        indices, values, (n_dof, n_dof), dtype=torch.float64).coalesce()
    dense = K.to_dense()
    action = (dense @ direction.reshape(-1)).reshape_as(direction)
    expected = fem.internal_force(direction, d)

    assert torch.allclose(dense, dense.T, atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(action, expected, atol=1.0e-10, rtol=1.0e-10)


def test_q4_quasistatic_sparse_direct_solves_isotropic_patch():
    from phast.sparse_solve import scipy_available

    if not scipy_available():
        pytest.skip("SciPy sparse backend is not available")

    qmesh = structured_q4_mesh(width=1.0, height=1.0, nx=2, ny=1)
    mesh = FEMMesh.from_tensors(
        qmesh.nodes, qmesh.quads, node_sets=qmesh.node_sets,
        device="cpu", dtype=torch.float64, element_type="Q4")
    material = Material(E=210.0, nu=0.30, energy_split="isotropic")
    fem = FEMOperators(mesh, material)
    solver = QuasiStaticSolver(fem, backend="scipy", max_iter=4, tol=1.0e-10)
    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    f_ext = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)
    bc_mask = torch.zeros((mesh.n_nodes, 2), dtype=torch.bool)
    bc_vals = torch.zeros((mesh.n_nodes, 2), dtype=torch.float64)

    bc_mask[mesh.node_sets["left"], :] = True
    bc_mask[mesh.node_sets["right"], 0] = True
    bc_vals[mesh.node_sets["right"], 0] = 0.01

    u, converged, n_iter = solver.solve(d, f_ext, bc_mask, bc_vals)

    assert converged
    assert n_iter <= 2
    assert solver.last_backend == "scipy"
    assert solver.last_residual <= 1.0e-8
    assert torch.isfinite(u).all()
    assert torch.allclose(u[bc_mask], bc_vals[bc_mask])
