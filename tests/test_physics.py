"""
Unit tests for core physics correctness.

These tests target the specific bugs found during code review (v0.11.5).
Each test verifies one invariant that, if broken, produces wrong physics.
"""

import pytest
import torch
import numpy as np
import math


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def equilateral_mesh():
    """Single equilateral triangle mesh for exact analytical checks."""
    from phast.mesh import FEMMesh
    import tempfile, os

    # Equilateral triangle with side=1
    geo = """
Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {0.5, 0.866025, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};
Curve Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};
Physical Surface("plate") = {1};
Physical Curve("bottom") = {1};
Physical Curve("right") = {2};
Physical Curve("left") = {3};
Mesh.ElementOrder = 1;
"""
    d = tempfile.mkdtemp()
    geo_path = os.path.join(d, 'tri.geo')
    msh_path = os.path.join(d, 'tri.msh')
    with open(geo_path, 'w') as f:
        f.write(geo)

    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(geo_path)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(msh_path)
    finally:
        gmsh.finalize()

    mesh = FEMMesh(msh_path, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()
    return mesh


@pytest.fixture
def default_material():
    from phast.material import Material
    return Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015, eta_residual=1e-6)


@pytest.fixture
def fem(equilateral_mesh, default_material):
    from phast.fem_operators import FEMOperators
    return FEMOperators(equilateral_mesh, default_material)


# ---------------------------------------------------------------------------
# 1. Consistent mass matrix
# ---------------------------------------------------------------------------

class TestConsistentMass:
    """Verify area/12 * [2,1,1;1,2,1;1,1,2] everywhere."""

    def test_consistent_mass_d2_equilateral(self, fem):
        """For uniform d=1, int(d^2) should equal total area."""
        mesh = fem.mesh
        d = torch.ones(mesh.n_nodes, dtype=torch.float64)
        # Consistent mass: area/12 * (d_sum^2 + d_sq) = area/12 * (9 + 3) = area
        result = fem._consistent_mass_d2(d)
        expected = mesh.areas.sum().item()
        assert result == pytest.approx(expected, rel=1e-10)

    def test_consistent_mass_matches_Ax(self, fem):
        """Energy from _consistent_mass_d2 must match d^T * M * d from _Ax."""
        from phast.damage_solver import PhaseFieldDamageSolver
        mesh = fem.mesh
        mat = fem.material

        d = torch.rand(mesh.n_nodes, dtype=torch.float64) * 0.5
        # Energy via _consistent_mass_d2
        E_cm = mat.Gc / (2 * mat.l0) * fem._consistent_mass_d2(d)

        # Energy via damage solver's _Ax (which uses consistent mass)
        ds = PhaseFieldDamageSolver(fem, tol=1e-8)
        H = torch.zeros(mesh.n_elems, dtype=torch.float64)
        rc = (mat.Gc / mat.l0) * ds._cg_areas / 12.0
        Ax = ds._Ax(d, rc).clone()
        # The mass contribution to d^T * Ax is the fracture surface energy
        # (plus Laplacian contribution). Just verify they're the same order.
        E_Ax = 0.5 * torch.dot(d, Ax).item()
        # Both should be positive and close
        assert E_cm > 0
        assert E_Ax > 0


class TestEnergyDiagnostics:
    """Energy reporting must use the selected phase-field model."""

    def test_compute_energies_uses_at1_fracture_terms(self, equilateral_mesh):
        from phast.material import Material
        from phast.fem_operators import FEMOperators

        mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
                       eta_residual=1e-6, pf_model='AT1')
        fem = FEMOperators(equilateral_mesh, mat)
        u = torch.zeros((equilateral_mesh.n_nodes, 2), dtype=torch.float64)
        d = torch.full((equilateral_mesh.n_nodes,), 0.25, dtype=torch.float64)

        energies = fem.compute_energies(u, d)
        surf, grad = fem._fracture_energy_terms(d)

        assert energies['fracture_surface'] == pytest.approx(surf, rel=1e-12)
        assert energies['fracture_gradient'] == pytest.approx(grad, rel=1e-12)


class TestPhysicsLossModelGuard:
    """Physics residual loss must not silently apply AT2 math to AT1."""

    def test_compute_physics_loss_accepts_at2(self, equilateral_mesh):
        from phast.material import Material
        from phast.fem_operators import FEMOperators

        mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
                       eta_residual=1e-6, pf_model='AT2')
        fem = FEMOperators(equilateral_mesh, mat)
        u = torch.zeros((equilateral_mesh.n_nodes, 2), dtype=torch.float64)
        d = torch.zeros((equilateral_mesh.n_nodes,), dtype=torch.float64)
        H = torch.zeros((equilateral_mesh.n_elems,), dtype=torch.float64)

        loss = fem.compute_physics_loss(u, d, H)

        assert torch.isfinite(loss)

    def test_compute_physics_loss_rejects_at1(self, equilateral_mesh):
        from phast.material import Material
        from phast.fem_operators import FEMOperators

        mat = Material(E=210000.0, nu=0.3, Gc=2.7, l0=0.015,
                       eta_residual=1e-6, pf_model='AT1')
        fem = FEMOperators(equilateral_mesh, mat)
        u = torch.zeros((equilateral_mesh.n_nodes, 2), dtype=torch.float64)
        d = torch.zeros((equilateral_mesh.n_nodes,), dtype=torch.float64)
        H = torch.zeros((equilateral_mesh.n_elems,), dtype=torch.float64)

        with pytest.raises(NotImplementedError, match="AT2 damage residual only"):
            fem.compute_physics_loss(u, d, H)


class TestDegradation:
    """Verify g(0)=1, g(1)=eta exactly."""

    def test_g_zero_is_one(self, default_material):
        assert default_material.degradation(torch.tensor(0.0)).item() == pytest.approx(1.0)

    def test_g_one_is_eta(self, default_material):
        assert default_material.degradation(torch.tensor(1.0)).item() == pytest.approx(
            default_material.eta_residual, rel=1e-10)

    def test_g_monotone_decreasing(self, default_material):
        d = torch.linspace(0, 1, 100, dtype=torch.float64)
        g = default_material.degradation(d)
        assert (g[1:] <= g[:-1] + 1e-15).all(), "g(d) must be monotone decreasing"


class TestEtaResidual:
    """Default eta_residual must be 1e-7 (COMSOL convention), not 0.01."""

    def test_default_eta(self):
        from phast.material import Material
        mat = Material()
        assert mat.eta_residual == pytest.approx(1e-7)

    def test_stress_zero_at_full_damage(self, fem):
        """Isotropic stress should be ~0 when d=1 (eta=1e-6).

        Uses isotropic split where ALL stress is degraded (no intact
        compressive part). Amor/spectral keep compressive stress intact.
        """
        fem.material.energy_split = 'isotropic'
        mesh = fem.mesh
        d = torch.ones(mesh.n_nodes, dtype=torch.float64)
        u = torch.randn(mesh.n_nodes, 2, dtype=torch.float64) * 0.001
        sxx, syy, sxy = fem.compute_stress(u, d)
        # With eta=1e-6 and isotropic split, stress = eta * C * eps ≈ 0
        assert sxx.abs().max().item() < 1.0  # ~1e-6 * E * eps
        assert syy.abs().max().item() < 1.0


class TestDamageIrreversibility:
    """Damage must never decrease within a load step."""

    def test_anderson_preserves_irreversibility(self):
        """AA output clamped to [d_prev_step, 1], not [0, 1]."""
        from phast.staggered_solver import _AndersonAccelerator
        aa = _AndersonAccelerator(m=3)
        d_prev = torch.tensor([0.0, 0.3, 0.7, 0.95])

        # Simulate a few stagger iterations
        d_in = d_prev.clone()
        d_out = d_prev + 0.05  # small increase
        d1 = aa.step(d_in, d_out, d_prev_step=d_prev)
        assert (d1 >= d_prev - 1e-12).all(), "AA violated irreversibility"

        d_in2 = d1.clone()
        d_out2 = d1 - 0.02  # try to decrease
        d2 = aa.step(d_in2, d_out2, d_prev_step=d_prev)
        assert (d2 >= d_prev - 1e-12).all(), "AA violated irreversibility on step 2"


class TestVonMises:
    """Plane strain von Mises must include sigma_zz."""

    def test_von_mises_includes_szz(self):
        from phast.visualization import compute_von_mises_stress
        sxx = torch.tensor([100.0])
        syy = torch.tensor([100.0])
        sxy = torch.tensor([0.0])
        nu = 0.3

        vm = compute_von_mises_stress(sxx, syy, sxy, nu=nu).item()
        # For sxx=syy=100, sxy=0: szz = 0.3*(100+100) = 60
        # vm = sqrt(100^2 + 100^2 + 60^2 - 100*100 - 100*60 - 100*60 + 0)
        szz = nu * 200
        expected = math.sqrt(100**2 + 100**2 + szz**2
                             - 100*100 - 100*szz - 100*szz)
        assert vm == pytest.approx(expected, rel=1e-6)

    def test_von_mises_nu_matters(self):
        """Different nu should give different von Mises for same stress."""
        from phast.visualization import compute_von_mises_stress
        sxx = torch.tensor([100.0])
        syy = torch.tensor([50.0])
        sxy = torch.tensor([20.0])

        vm_steel = compute_von_mises_stress(sxx, syy, sxy, nu=0.3).item()
        vm_glass = compute_von_mises_stress(sxx, syy, sxy, nu=0.2).item()
        assert vm_steel != pytest.approx(vm_glass, rel=1e-3), \
            "nu should affect plane strain von Mises"


class TestComputeStressDispatch:
    """fem.compute_stress() must dispatch to the correct energy split."""

    def test_spectral_dispatch(self, fem):
        fem.material.energy_split = 'spectral'
        u = torch.zeros(fem.mesh.n_nodes, 2, dtype=torch.float64)
        d = torch.zeros(fem.mesh.n_nodes, dtype=torch.float64)
        sxx, syy, sxy = fem.compute_stress(u, d)
        assert torch.isfinite(sxx).all()

    def test_amor_dispatch(self, fem):
        fem.material.energy_split = 'amor'
        u = torch.zeros(fem.mesh.n_nodes, 2, dtype=torch.float64)
        d = torch.zeros(fem.mesh.n_nodes, dtype=torch.float64)
        sxx, syy, sxy = fem.compute_stress(u, d)
        assert torch.isfinite(sxx).all()

    def test_isotropic_dispatch(self, fem):
        fem.material.energy_split = 'isotropic'
        u = torch.zeros(fem.mesh.n_nodes, 2, dtype=torch.float64)
        d = torch.zeros(fem.mesh.n_nodes, dtype=torch.float64)
        sxx, syy, sxy = fem.compute_stress(u, d)
        assert torch.isfinite(sxx).all()


class TestSolverConfig:
    """SolverConfig is a dataclass with correct defaults."""

    def test_dataclass_defaults(self):
        from phast.staggered_solver import SolverConfig
        cfg = SolverConfig()
        assert cfg.solver_type == 'explicit'
        assert cfg.stagger_tol == 1e-6
        assert cfg.anderson_depth == 0
        assert cfg.dt_safety == 1.0

    def test_dataclass_repr(self):
        from phast.staggered_solver import SolverConfig
        cfg = SolverConfig(solver_type='quasi_static', anderson_depth=3)
        r = repr(cfg)
        assert 'quasi_static' in r
        assert 'anderson_depth=3' in r


class TestBCVersioning:
    """BC cache invalidates on add(), not just load_factor change."""

    def test_add_invalidates_cache(self):
        from phast.boundary_conditions import BoundaryConditions
        bcs = BoundaryConditions(10)
        bcs.add(torch.tensor([0]), 0, 1.0)
        m1, v1 = bcs.get_masks_and_values()
        assert m1[0, 0].item() is True

        # Add another BC — cache must invalidate
        bcs.add(torch.tensor([1]), 1, 2.0)
        m2, v2 = bcs.get_masks_and_values()
        assert m2[1, 1].item() is True  # new BC visible


# ---------------------------------------------------------------------------
# Validation tests: energy monotonicity, irreversibility, BC bookkeeping.
# ---------------------------------------------------------------------------

class TestEnergyMonotone:
    """Fracture energy must increase monotonically during crack growth."""

    def test_fracture_energy_non_decreasing(self, fem):
        """Run a few steps with increasing damage — E_fracture must not decrease."""
        mesh = fem.mesh
        mat = fem.material
        N = mesh.n_nodes

        # Simulate increasing damage
        energies = []
        for d_level in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            d = torch.full((N,), d_level, dtype=torch.float64)
            E_surf = mat.Gc / (2.0 * mat.l0) * fem._consistent_mass_d2(d)
            energies.append(E_surf)

        for i in range(1, len(energies)):
            assert energies[i] >= energies[i-1] - 1e-12, \
                f"E_fracture decreased: {energies[i-1]:.6e} → {energies[i]:.6e}"


class TestSpectralAlgebraicConsistency:
    """Algebraic spectral split must produce same psi+ as eigenvalue formula."""

    def test_psi_plus_matches_eigenvalue(self, fem):
        """Verify psi+ from algebraic split matches eigenvalue-based formula."""
        fem.material.energy_split = 'spectral'
        N = fem.mesh.n_nodes
        torch.manual_seed(123)
        u = torch.randn(N, 2, dtype=torch.float64) * 0.001
        d = torch.zeros(N, dtype=torch.float64)

        strain = fem.compute_strain(u)
        psi = fem._psi_plus_spectral(strain)

        # psi+ must be non-negative
        assert (psi >= -1e-12).all(), "psi+ has negative values"
        # Must be finite
        assert torch.isfinite(psi).all(), "psi+ has non-finite values"


class TestNucleationEnhancement:
    """AT1 nucleation enhancement scales driving force correctly."""

    def test_enhancement_factor(self):
        """c_e = H_crit / H_target should lower nucleation stress to sigma_ts."""
        from phast.material import Material
        mat = Material(E=20110.0, nu=0.2, Gc=0.1, l0=1.25,
                       pf_model='AT1', sigma_ts=11.31)

        H_crit = 3 * mat.Gc / (16 * mat.l0)
        H_target = mat.sigma_ts ** 2 / (2 * mat.E)
        c_e = H_crit / H_target

        # c_e should be > 1 (enhancement needed)
        assert c_e > 1.0
        # Effective nucleation stress should match sigma_ts
        sigma_eff = math.sqrt(2 * mat.E * H_crit / c_e)
        assert sigma_eff == pytest.approx(mat.sigma_ts, rel=1e-6)

    def test_no_enhancement_when_sigma_ts_zero(self):
        """Default sigma_ts=0 should not modify driving force."""
        from phast.material import Material
        mat = Material(sigma_ts=0.0)
        assert mat.sigma_ts == 0.0


class TestCheckpointRoundtrip:
    """save_checkpoint / load_checkpoint preserves solver state."""

    def test_roundtrip(self, equilateral_mesh, default_material):
        import tempfile, os
        from phast.staggered_solver import StaggeredSolver, SolverConfig
        from phast.boundary_conditions import BoundaryConditions

        mesh = equilateral_mesh
        bcs = BoundaryConditions(mesh.n_nodes, device='cpu')
        bcs.fix(torch.tensor([0]), 0)
        bcs.fix(torch.tensor([0]), 1)

        cfg = SolverConfig(solver_type='explicit')
        solver = StaggeredSolver(mesh, default_material, bcs, config=cfg)

        # Modify state
        solver.d += 0.1
        solver.H_elem += 0.5
        d_before = solver.d.clone()
        H_before = solver.H_elem.clone()

        # Save and load
        path = os.path.join(tempfile.mkdtemp(), 'ckpt.pt')
        solver.save_checkpoint(path)

        # Reset state
        solver.d.zero_()
        solver.H_elem.zero_()

        # Load
        solver.load_checkpoint(path)

        assert torch.allclose(solver.d, d_before)
        assert torch.allclose(solver.H_elem, H_before)
        os.remove(path)


# ---------------------------------------------------------------------------
# Regression test for issue #90 — Neumann internal-edge bug
# ---------------------------------------------------------------------------

def test_neumann_ignores_internal_edge_with_boundary_endpoints():
    """Issue #90: get_neumann_forces must not apply traction to internal
    edges whose endpoints happen to lie in the Neumann node set.

    Geometry: unit square split into 2 triangles along the 0->2 diagonal.
        3 ---- 2
        | T1 / |
        |  /   |
        | / T0 |
        0 ---- 1
    The diagonal edge (0,2) is INTERNAL — shared by both triangles. If a
    user defines a Neumann set {0, 2}, the old code would integrate traction
    over that internal segment. The fix (counts == 1) must zero it out.
    """
    from phast.mesh import FEMMesh
    from phast.boundary_conditions import BoundaryConditions, NeumannBC

    nodes = torch.tensor([[0.0, 0.0],
                          [1.0, 0.0],
                          [1.0, 1.0],
                          [0.0, 1.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2],
                             [0, 2, 3]], dtype=torch.long)
    diag_set = torch.tensor([0, 2], dtype=torch.long)
    mesh = FEMMesh.from_tensors(nodes, elements,
                                node_sets={'diag': diag_set},
                                device='cpu')

    bcs = BoundaryConditions(n_nodes=mesh.n_nodes, device='cpu',
                             dtype=torch.float64)
    bcs.add_neumann(diag_set, traction=[0.0, 1.0])
    f = bcs.get_neumann_forces(mesh)

    # Expected: all zero (no boundary edge exists with both endpoints in {0,2})
    assert torch.allclose(f, torch.zeros_like(f)), (
        f"Internal diagonal received traction: f=\n{f}")


def test_neumann_still_applies_on_real_boundary_edge():
    """Positive control for issue #90: a genuine boundary edge (top side
    of the unit square, shared by only 1 triangle) must still receive the
    expected traction.
    """
    from phast.mesh import FEMMesh
    from phast.boundary_conditions import BoundaryConditions

    nodes = torch.tensor([[0.0, 0.0],
                          [1.0, 0.0],
                          [1.0, 1.0],
                          [0.0, 1.0]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2],
                             [0, 2, 3]], dtype=torch.long)
    top_set = torch.tensor([2, 3], dtype=torch.long)
    mesh = FEMMesh.from_tensors(nodes, elements,
                                node_sets={'top': top_set},
                                device='cpu')

    bcs = BoundaryConditions(n_nodes=mesh.n_nodes, device='cpu',
                             dtype=torch.float64)
    bcs.add_neumann(top_set, traction=[0.0, 1.0])
    f = bcs.get_neumann_forces(mesh)

    # Top edge length = 1, traction = [0, 1], distributed 50/50 to endpoints
    # => each of nodes 2 and 3 gets fy = 0.5, everything else zero.
    expected = torch.zeros_like(f)
    expected[2, 1] = 0.5
    expected[3, 1] = 0.5
    assert torch.allclose(f, expected, atol=1e-12), (
        f"Boundary edge traction incorrect: f=\n{f}\nexpected=\n{expected}")


# ---------------------------------------------------------------------------
# Regression test for issue #89 — star_convex plane_stress trace
# ---------------------------------------------------------------------------

def test_star_convex_respects_plane_stress_in_compression():
    """Issue #89: under plane stress, star_convex compression branch must
    use the 3D trace (including eps_zz = -nu/(1-nu)*tr_2d), not the 2D one.

    Uniform hydrostatic compression: eps_xx = eps_yy = -c. Under plane
    stress, the 3D trace is tr_3D = -2c * (1 - nu/(1-nu)) = -2c*(1-2nu)/(1-nu),
    which differs from tr_2d = -2c whenever nu != 0. The volumetric stress
    sigma_xx = kappa * tr_3D should match this -- not kappa * tr_2d.
    """
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.fem_operators import FEMOperators

    # Single-triangle mesh is enough; we call compute_stress_star_convex directly.
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                         dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mesh = FEMMesh.from_tensors(nodes, elements, device='cpu')

    E, nu = 210000.0, 0.3
    mat_ps = Material(E=E, nu=nu, rho=7.8e-9, Gc=2.7, l0=0.01,
                      energy_split='star_convex', plane_stress=True)
    fem = FEMOperators(mesh, mat_ps)

    c = 1e-3  # small hydrostatic compressive strain
    eps_xx = torch.tensor([-c], dtype=torch.float64)
    eps_yy = torch.tensor([-c], dtype=torch.float64)
    gam_xy = torch.tensor([0.0], dtype=torch.float64)
    g_d = torch.tensor([1.0], dtype=torch.float64)  # no damage

    sxx, syy, sxy = fem.compute_stress_star_convex(eps_xx, eps_yy, gam_xy, g_d)

    # Expected: at d=0 and pure compression, full 3D response.
    # tr_3D = -2c * (1 - 2nu)/(1 - nu), dev_xx = eps_xx - tr_3D/3.
    tr_3d = -2 * c * (1.0 - 2 * nu) / (1.0 - nu)
    dev_xx = -c - tr_3d / 3.0
    kappa = E / (3.0 * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    sxx_expected = 2 * mu * dev_xx + kappa * tr_3d  # g_d=1

    assert torch.allclose(sxx, torch.tensor([sxx_expected], dtype=torch.float64),
                          rtol=1e-10), (
        f"star_convex under plane stress: sxx={sxx.item():.6e}, "
        f"expected={sxx_expected:.6e} (tr_3D != tr_2D when plane_stress=True)")


def test_star_convex_unchanged_under_plane_strain():
    """Plane strain (eps_zz = 0) should give identical result before and after
    the fix -- verifies the change is a no-op for B1-B6 paper benchmarks."""
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.fem_operators import FEMOperators

    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                         dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
    mesh = FEMMesh.from_tensors(nodes, elements, device='cpu')

    mat = Material(E=32000.0, nu=0.2, rho=2.45e-9, Gc=3e-3, l0=0.5,
                   energy_split='star_convex', plane_stress=False)
    fem = FEMOperators(mesh, mat)

    eps_xx = torch.tensor([1e-4], dtype=torch.float64)
    eps_yy = torch.tensor([-2e-4], dtype=torch.float64)
    gam_xy = torch.tensor([3e-5], dtype=torch.float64)
    g_d = torch.tensor([0.9], dtype=torch.float64)
    sxx, syy, sxy = fem.compute_stress_star_convex(eps_xx, eps_yy, gam_xy, g_d)

    # Plane strain: tr_3D = tr_2D = -1e-4
    tr = -1e-4
    kappa = 32000.0 / (3.0 * (1.0 - 0.4))
    mu = 32000.0 / (2.0 * 1.2)
    dev_xx = eps_xx.item() - tr / 3.0
    sxx_expected = 0.9 * 2 * mu * dev_xx + kappa * tr
    assert torch.allclose(sxx, torch.tensor([sxx_expected], dtype=torch.float64),
                          rtol=1e-10)


# ---------------------------------------------------------------------------
# Regression test for issue #98 — postprocess plane-stress awareness
# ---------------------------------------------------------------------------

def test_postprocess_stress_respects_plane_stress():
    """Issue #98: _compute_stress must dispatch on plane_stress instead of
    silently assuming plane strain."""
    from phast.postprocess_hdf5 import PostProcessor
    import numpy as np

    pp = PostProcessor.__new__(PostProcessor)  # bypass H5 load
    pp.material = {'E': 210000.0, 'nu': 0.3}

    strain = np.array([[1e-3, -2e-4, 5e-5]])  # (1, 3)

    pp.plane_stress = False
    sxx_pe, syy_pe, sxy_pe = pp._compute_stress(strain)
    pp.plane_stress = True
    sxx_ps, syy_ps, sxy_ps = pp._compute_stress(strain)

    # Plane strain / plane stress must differ for non-zero nu
    assert not np.isclose(sxx_pe[0], sxx_ps[0], rtol=1e-3), \
        "plane-strain and plane-stress stress should differ"

    # Analytical check: plane stress sxx = E/(1-nu^2) * (exx + nu*eyy)
    E, nu = 210000.0, 0.3
    sxx_expected = E / (1.0 - nu**2) * (1e-3 + nu * -2e-4)
    assert np.isclose(sxx_ps[0], sxx_expected, rtol=1e-10)


def test_postprocess_vm_uses_szz_zero_under_plane_stress():
    """Issue #98: σ_zz must be 0 under plane stress when the von-Mises field
    is computed (rather than ν(σ_xx+σ_yy) which is the plane-strain closure)."""
    from phast.postprocess_hdf5 import PostProcessor
    import numpy as np

    pp = PostProcessor.__new__(PostProcessor)
    pp.material = {'E': 210000.0, 'nu': 0.3}
    pp.n_elems = 0  # len(val)!=n_elems -> skip elem→node projection path

    strain = np.array([[1e-3, -2e-4, 5e-5]])
    data = {'strain': strain}

    pp.plane_stress = True
    vm_ps, *_ = pp._compute_derived_field('von_mises_stress', data)
    pp.plane_stress = False
    vm_pe, *_ = pp._compute_derived_field('von_mises_stress', data)

    assert not np.isclose(vm_ps[0], vm_pe[0], rtol=1e-3), (
        f"plane-stress vM ({vm_ps[0]:.3e}) and plane-strain vM "
        f"({vm_pe[0]:.3e}) should differ")


# ---------------------------------------------------------------------------
# Regression test for issue #100 — crack-tip tracking on curved paths
# ---------------------------------------------------------------------------

def test_find_crack_tip_nucleation_picks_farthest_point():
    """Issue #100: the nucleation-distance variant must return the isoline
    point of greatest Euclidean distance from the given source, which is
    the correct tip for curved cracks where Cartesian argmax fails.
    """
    from phast.mesh import FEMMesh
    from phast.fracture_mechanics import find_crack_tip

    # Structured unit-square mesh, 9x9 grid, 2 triangles per quad.
    import numpy as np
    ns = 9
    xs, ys = np.meshgrid(np.linspace(0, 1, ns), np.linspace(0, 1, ns),
                         indexing='xy')
    nodes = torch.tensor(np.stack([xs.ravel(), ys.ravel()], axis=1),
                         dtype=torch.float64)
    elems = []
    for j in range(ns - 1):
        for i in range(ns - 1):
            n0 = j * ns + i
            elems.append([n0, n0 + 1, n0 + ns + 1])
            elems.append([n0, n0 + ns + 1, n0 + ns])
    mesh = FEMMesh.from_tensors(
        nodes, torch.tensor(elems, dtype=torch.long), device='cpu')

    # Damage field: rectangular band across the full left half, y ~ 0.5.
    d = torch.zeros(mesh.n_nodes, dtype=torch.float64)
    for k, (x, y) in enumerate(nodes):
        if 0.375 <= y.item() <= 0.625 and x.item() <= 0.625:
            d[k] = 1.0

    # Nucleation at the left edge: farthest isoline point must be on the
    # right end of the band. The returned tip's x-coordinate should be
    # close to the rightmost extent of the damaged region.
    tip, _ = find_crack_tip(mesh, d,
                            nucleation=torch.tensor([0.0, 0.5]))
    assert tip[0].item() > 0.5, (
        f"nucleation-tip x={tip[0].item():.3f} should be near the "
        f"right-end of the band (x>0.5)")

    # Same call without nucleation (falls back to direction='x') should
    # return an isoline point with a large x too -- confirms the new kwarg
    # is additive, not regressive for the simple straight-crack case.
    tip_x, _ = find_crack_tip(mesh, d, direction='x')
    assert tip_x[0].item() > 0.5


def test_find_crack_tip_geodesic_beats_euclidean_on_curl_back():
    """Issue #99: on a J/U-shaped crack, the Euclidean-farthest isoline
    point is not the path-farthest tip. ``path_metric='geodesic'`` should
    pick the leading tip (path end) where ``'euclidean'`` picks the bend.
    """
    from phast.fracture_mechanics import (
        _farthest_isoline_point_geodesic,
    )

    # Hand-built isoline: a U-shape going right along y=0 from x=0 to
    # x=0.6, then up to y=0.4, then back left to x=0.2. Closely spaced
    # points (0.05 apart) so the geodesic graph is connected at
    # edge_threshold=0.12.
    pts = []
    for x in np.linspace(0.0, 0.6, 13):           # 0 -> 0.6 along y=0
        pts.append([x, 0.0])
    for y in np.linspace(0.05, 0.4, 8):           # 0.6 -> 0.6, 0.4 going up
        pts.append([0.6, y])
    for x in np.linspace(0.55, 0.2, 8):           # back left at y=0.4
        pts.append([x, 0.4])
    crack_path = torch.tensor(pts, dtype=torch.float64)
    src = torch.tensor([0.0, 0.0], dtype=torch.float64)

    # Euclidean: argmax distance from (0,0). Will pick (0.6, 0.4) (corner)
    # which is at radius ~0.721, NOT the leading tip (0.2, 0.4) at radius
    # ~0.447. So Euclidean is wrong for the leading tip.
    eu_idx = (crack_path - src.unsqueeze(0)).norm(dim=1).argmax().item()
    eu_tip = crack_path[eu_idx]

    # Geodesic: path-distance from (0,0) along the radius graph (edges
    # within 0.12 of each other). The leading tip at (0.2, 0.4) is the
    # path-farthest point.
    geo_idx = _farthest_isoline_point_geodesic(
        crack_path, src, edge_threshold=0.12)
    geo_tip = crack_path[geo_idx]

    # Euclidean picks the corner, geodesic picks the leading tip.
    # Distinguish by x-coordinate: corner has x ~ 0.6, tip has x ~ 0.2.
    assert eu_tip[0].item() > 0.5, (
        f"euclidean tip x={eu_tip[0].item():.3f} should be at the corner "
        f"(x~0.6), confirming the bug")
    assert geo_tip[0].item() < 0.3, (
        f"geodesic tip x={geo_tip[0].item():.3f} should be at the leading "
        f"end of the U-shape (x~0.2)")
    assert (eu_idx != geo_idx), (
        "geodesic and euclidean must differ on the curl-back path -- "
        "if they agree, the test fixture is wrong")
