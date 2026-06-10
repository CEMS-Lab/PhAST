"""
Regression test for issue #218 (HIGH): PhaseFieldDamageSolver constructed
with pf_model='AT1' and nodal_H=True must raise NotImplementedError.

Background
----------
The CG nodal-H path in `_prepare_cg_nodal` (damage_solver.py:1485 onward)
assembles the RHS as

    b_a = A/6 * (H_a + S_H_nodal)

without subtracting the AT1 elastic-threshold source term
S_H = 3 Gc / (8 l0). This is the AT2 RHS form. Prior to this fix, AT1
configs that opted into nodal_H silently produced AT2 numerics — wrong
damage fields with no warning. The eager guard in
PhaseFieldDamageSolver.__init__ converts that silent failure into a
clear error.

Supported combinations after this fix:
  (AT2, nodal_H=False), (AT2, nodal_H=True), (AT1, nodal_H=False).
Unsupported: (AT1, nodal_H=True) and (PFCZM, nodal_H=True) — must raise
NotImplementedError.
"""

import pytest
import torch

from phast.material import Material


def _build_unit_square_mesh(tmp_path):
    """Tiny 1x1 mesh — minimal infrastructure for solver construction."""
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

    from phast.mesh import FEMMesh
    mesh = FEMMesh(str(msh_file), device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()
    return mesh


@pytest.mark.parametrize("pf_model,extra", [
    ("AT1", {}),
    ("PFCZM", {"sigma_ts": 3.0}),
])
def test_finite_threshold_models_with_nodal_H_raise_not_implemented(
        tmp_path, pf_model, extra):
    """AT1/PF-CZM + nodal_H=True must raise NotImplementedError eagerly."""
    from phast.fem_operators import FEMOperators
    from phast.damage_solver import PhaseFieldDamageSolver

    mesh = _build_unit_square_mesh(tmp_path)
    mat = Material(
        E=3090.0, nu=0.35, Gc=0.3, l0=0.1, rho=1.18e-9,
        energy_split='amor', pf_model=pf_model, **extra,
    )
    fem = FEMOperators(mesh, mat)

    with pytest.raises(NotImplementedError) as exc_info:
        PhaseFieldDamageSolver(fem, nodal_H=True)

    msg = str(exc_info.value)
    # The error message must be specific enough to be actionable.
    assert pf_model in msg
    assert "nodal_H" in msg
    # Must mention which combinations ARE supported, so the user can fix it.
    assert "AT2" in msg


@pytest.mark.parametrize("pf_model,nodal_H", [
    ('AT2', False),
    ('AT2', True),
    ('AT1', False),
    ('PFCZM', False),
])
def test_supported_combinations_construct_cleanly(tmp_path, pf_model, nodal_H):
    """The three supported (pf_model, nodal_H) combinations must construct
    without raising — a guard that's too aggressive would break valid
    configs."""
    from phast.fem_operators import FEMOperators
    from phast.damage_solver import PhaseFieldDamageSolver

    mesh = _build_unit_square_mesh(tmp_path)
    extra = {"sigma_ts": 3.0} if pf_model == "PFCZM" else {}
    mat = Material(
        E=3090.0, nu=0.35, Gc=0.3, l0=0.1, rho=1.18e-9,
        energy_split='amor', pf_model=pf_model, **extra,
    )
    fem = FEMOperators(mesh, mat)
    # Should not raise.
    solver = PhaseFieldDamageSolver(fem, nodal_H=nodal_H)
    assert solver._nodal_H is nodal_H
    assert solver._pf_model == pf_model
