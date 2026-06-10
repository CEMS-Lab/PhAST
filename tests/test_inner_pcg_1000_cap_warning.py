"""Inner adjoint/active-set PCG 1000-cap warning test.

Audit T1.6 (W4 SOLVER_BUGS_OPTIMISATION_AUDIT_2026-05-07.md): the inner
PCG loops in the ``_AdjointDamage*`` autograd Functions cap iterations
at ``min(self.max_iter, 1000)`` (and 500 in the simpler adjoint path).
A user that configures ``max_iter=2000`` for an ill-conditioned mesh
silently has adjoint solves capped to 1000 with no signal.

Fix: at solver init, when ``max_iter`` exceeds the inner-PCG hard cap,
emit a one-shot ``RuntimeWarning`` so the caller knows adjoint solves
will be capped.

Sibling to ``test_deterministic_max_iter_cap_warning.py``; this file
focuses specifically on the 1000-cap and one-shot semantics per
solver instance.
"""

import warnings

import pytest
import torch


def _build_mesh_and_material(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    from phast.mesh import FEMMesh
    from phast.material import Material

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
Mesh.CharacteristicLengthMax=0.4;
"""
    geo_file = tmp_path / "innercap.geo"
    geo_file.write_text(geo)
    msh_file = tmp_path / "innercap.msh"
    gmsh.initialize()
    try:
        gmsh.open(str(geo_file))
        gmsh.model.mesh.generate(2)
        gmsh.write(str(msh_file))
    finally:
        gmsh.finalize()

    mesh = FEMMesh(str(msh_file), device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()
    mat = Material(
        E=2100.0, nu=0.3, Gc=2.7, l0=0.05, rho=7.8e-9,
        energy_split='amor', pf_model='AT2',
    )
    return mesh, mat


def test_inner_pcg_cap_warning_text_mentions_1000(tmp_path):
    """The warning must explicitly mention the 1000 hard cap so the user
    can find it in logs and link to the audit entry."""
    from phast.damage_solver import PhaseFieldDamageSolver
    from phast.fem_operators import FEMOperators
    mesh, mat = _build_mesh_and_material(tmp_path)
    fem = FEMOperators(mesh, mat)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solver = PhaseFieldDamageSolver(fem, max_iter=5000)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]
    matched = [m for m in msgs if '1000' in m and 'max_iter' in m]
    assert len(matched) == 1, (
        f"Expected exactly one RuntimeWarning citing '1000' and 'max_iter', "
        f"got {len(matched)} of {len(msgs)} RuntimeWarnings: {msgs}"
    )
    # Solver flag is also set so any downstream code can check.
    assert solver._inner_pcg_cap_warned is True


def test_inner_pcg_cap_warning_one_shot_per_instance(tmp_path):
    """Two solver instances should each emit exactly one warning -- the
    flag is per-instance, so the second instance still warns. But within
    a single instance the warning fires once at init, never again."""
    from phast.damage_solver import PhaseFieldDamageSolver
    from phast.fem_operators import FEMOperators
    mesh, mat = _build_mesh_and_material(tmp_path)
    fem = FEMOperators(mesh, mat)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        s1 = PhaseFieldDamageSolver(fem, max_iter=2000)
        s2 = PhaseFieldDamageSolver(fem, max_iter=2000)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning) and '1000' in str(w.message)]
    # Two instances → two warnings (one per instance), and each instance
    # has its flag set.
    assert len(msgs) == 2
    assert s1._inner_pcg_cap_warned is True
    assert s2._inner_pcg_cap_warned is True
