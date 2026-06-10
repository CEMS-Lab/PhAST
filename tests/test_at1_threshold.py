"""
Tests for the AT1 strain-energy source term S_H (issue #140).

For AT1 phase-field, the weak-form RHS is (2H − S_H) M·1, where
S_H = 3 Gc / (8 l0) by default. This is equivalent to enforcing
H_eff = max(ψ⁺, W_c0) with W_c0 = S_H / 2 = 3 Gc / (16 l0)
(Pham 2011, Tanné 2018, Bleyer 2017, COMSOL dynamic crack-branching Eq. 6).

These tests cover:
  1. Material.at1_source resolves 'auto' → 3 Gc / (8 l0).
  2. Material.at1_source honours an explicit float override.
  3. AT2 ignores at1_threshold.
  4. Validation rejects bad values.
  5. The damage solver writes the resolved value into _at1_source.
  6. Mini-AT1 simulation: damage stays at 0 below the threshold, grows above.
"""

import pytest
import torch

from phast.material import Material, create_material


# ---------------------------------------------------------------------------
# 1–4: pure dataclass behaviour
# ---------------------------------------------------------------------------

def test_at1_auto_matches_analytical_formula():
    Gc, l0 = 2.7, 0.005
    mat = Material(Gc=Gc, l0=l0, pf_model='AT1')
    assert mat.at1_threshold == 'auto'
    expected = 3.0 * Gc / (8.0 * l0)
    assert mat.at1_source == pytest.approx(expected, rel=1e-12)


def test_at1_explicit_override_used():
    mat = Material(Gc=2.7, l0=0.005, pf_model='AT1', at1_threshold=42.0)
    assert mat.at1_source == pytest.approx(42.0, rel=1e-12)


def test_at2_ignores_at1_threshold_field():
    mat_auto = Material(Gc=2.7, l0=0.005, pf_model='AT2')
    assert mat_auto.at1_source == 0.0
    # Even if user sets a value, AT2 ignores it.
    mat_set = Material(Gc=2.7, l0=0.005, pf_model='AT2', at1_threshold=99.0)
    assert mat_set.at1_source == 0.0


def test_at1_threshold_validation():
    with pytest.raises(ValueError):
        Material(Gc=2.7, l0=0.005, pf_model='AT1', at1_threshold='bogus')
    with pytest.raises(ValueError):
        Material(Gc=2.7, l0=0.005, pf_model='AT1', at1_threshold=-1.0)


def test_create_material_forwards_override():
    mat = create_material('pmma_bleyer', at1_threshold=0.5)
    assert mat.pf_model == 'AT1'
    assert mat.at1_source == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# 5: damage solver consumes the override
# ---------------------------------------------------------------------------

def test_damage_solver_caches_resolved_at1_source(tmp_path):
    """Verify PhaseFieldDamageSolver._at1_source equals material.at1_source."""
    from phast.mesh import FEMMesh
    from phast.fem_operators import FEMOperators
    from phast.damage_solver import PhaseFieldDamageSolver

    geo = """
Point(1) = {0,0,0,1.0};
Point(2) = {1,0,0,1.0};
Point(3) = {1,1,0,1.0};
Point(4) = {0,1,0,1.0};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
Curve Loop(1)={1,2,3,4}; Plane Surface(1)={1};
Physical Surface("plate")={1};
Physical Curve("bottom")={1};
Physical Curve("top")={3};
Mesh.ElementOrder=1;
Mesh.CharacteristicLengthMax=0.5;
"""
    geo_file = tmp_path / "sq.geo"
    msh_file = tmp_path / "sq.msh"
    geo_file.write_text(geo)
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(str(geo_file))
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_file))
    finally:
        gmsh.finalize()
    mesh = FEMMesh(str(msh_file), device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()

    # auto → expected 3 Gc / (8 l0)
    Gc, l0 = 0.3, 0.1
    mat_auto = Material(
        E=3090.0, nu=0.35, Gc=Gc, l0=l0, rho=1.18e-9,
        energy_split='amor', pf_model='AT1',
    )
    fem_auto = FEMOperators(mesh, mat_auto)
    solver_auto = PhaseFieldDamageSolver(fem_auto)
    assert solver_auto._at1_source == pytest.approx(
        3.0 * Gc / (8.0 * l0), rel=1e-12)

    # explicit override
    mat_ovr = Material(
        E=3090.0, nu=0.35, Gc=Gc, l0=l0, rho=1.18e-9,
        energy_split='amor', pf_model='AT1', at1_threshold=7.5,
    )
    fem_ovr = FEMOperators(mesh, mat_ovr)
    solver_ovr = PhaseFieldDamageSolver(fem_ovr)
    assert solver_ovr._at1_source == pytest.approx(7.5, rel=1e-12)

    # AT2 → zero regardless of field value
    mat_at2 = Material(
        E=3090.0, nu=0.35, Gc=Gc, l0=l0, rho=1.18e-9,
        energy_split='amor', pf_model='AT2', at1_threshold=99.0,
    )
    fem_at2 = FEMOperators(mesh, mat_at2)
    solver_at2 = PhaseFieldDamageSolver(fem_at2)
    assert solver_at2._at1_source == 0.0


# ---------------------------------------------------------------------------
# 6: mini AT1 simulation — damage stays at 0 below W_c0, grows above
# ---------------------------------------------------------------------------

def test_at1_threshold_gates_damage_onset(tmp_path):
    """With pf_model=AT1, a uniform driving force ψ⁺ < W_c0 yields d≡0,
    while ψ⁺ > W_c0 produces d > 0.

    The elastic threshold is W_c0 = S_H / 2 = 3 Gc / (16 l0).
    """
    from phast.mesh import FEMMesh
    from phast.fem_operators import FEMOperators
    from phast.damage_solver import PhaseFieldDamageSolver

    geo = """
Point(1) = {0,0,0,1.0};
Point(2) = {1,0,0,1.0};
Point(3) = {1,1,0,1.0};
Point(4) = {0,1,0,1.0};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
Curve Loop(1)={1,2,3,4}; Plane Surface(1)={1};
Physical Surface("plate")={1};
Physical Curve("bottom")={1};
Physical Curve("top")={3};
Mesh.ElementOrder=1;
Mesh.CharacteristicLengthMax=0.25;
"""
    geo_file = tmp_path / "sq.geo"
    msh_file = tmp_path / "sq.msh"
    geo_file.write_text(geo)
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(str(geo_file))
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_file))
    finally:
        gmsh.finalize()
    mesh = FEMMesh(str(msh_file), device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()

    Gc, l0 = 1.0, 0.2
    W_c0 = 3.0 * Gc / (16.0 * l0)  # damage-onset threshold

    mat = Material(
        E=3000.0, nu=0.3, Gc=Gc, l0=l0, rho=1.0e-9,
        energy_split='amor', pf_model='AT1',
    )
    fem = FEMOperators(mesh, mat)
    solver = PhaseFieldDamageSolver(fem)

    n_elem = mesh.elements.shape[0]
    d_prev = torch.zeros(mesh.n_nodes, dtype=torch.float64)

    # (a) Below threshold: d should remain 0.
    H_low = torch.full(
        (n_elem,), 0.5 * W_c0, dtype=torch.float64, device='cpu')
    d_low = solver.solve(H_low, d_prev)
    assert d_low.max().item() < 1e-8, (
        f"Expected d=0 below W_c0, got max(d)={d_low.max().item():.3e}")

    # (b) Above threshold: d should be strictly positive.
    H_hi = torch.full(
        (n_elem,), 5.0 * W_c0, dtype=torch.float64, device='cpu')
    d_hi = solver.solve(H_hi, d_prev)
    assert d_hi.max().item() > 1e-3, (
        f"Expected d>0 above W_c0, got max(d)={d_hi.max().item():.3e}")


def test_at1_total_energy_uses_linear_fracture_functional():
    from phast.fem_operators import FEMOperators

    class _Mesh:
        device = 'cpu'
        dtype = torch.float64
        n_nodes = 3
        n_elems = 1
        elements = torch.tensor([[0, 1, 2]], dtype=torch.long)
        areas = torch.tensor([2.0], dtype=torch.float64)
        grad_phi = torch.zeros(1, 3, 2, dtype=torch.float64)
        M_scalar = torch.ones(3, dtype=torch.float64)
        _elem_flat = elements.flatten()
        h_min = 1.0

    mat = Material(
        E=210.0, nu=0.3, Gc=4.0, l0=2.0,
        pf_model='AT1', energy_split='isotropic',
    )
    fem = FEMOperators(_Mesh(), mat)
    u = torch.zeros(3, 2, dtype=torch.float64)
    d = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)

    expected_surface = 3.0 * mat.Gc / (8.0 * mat.l0) * (
        (fem.mesh.areas / 3.0) * d[fem.mesh.elements].sum(dim=1)
    ).sum().item()
    assert fem.compute_total_energy(u, d) == pytest.approx(
        expected_surface, rel=1e-12)
    comps = fem.compute_energy_components(u, d)
    assert comps['fracture'] == pytest.approx(expected_surface, rel=1e-12)
