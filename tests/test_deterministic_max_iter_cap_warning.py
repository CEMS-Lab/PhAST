"""``TORCH_PF_CG_DETERMINISTIC`` silent ``max_iter`` cap warning test.

Audit T1.5 (W4 SOLVER_BUGS_OPTIMISATION_AUDIT_2026-05-07.md): when the
env var ``TORCH_PF_CG_DETERMINISTIC=1`` is set, ``PhaseFieldDamageSolver``
silently caps ``max_iter`` to 50 (or to ``$TORCH_PF_CG_FIXED_ITERS``).
CG can return non-converged at the capped count, and the caller had no
signal that the cap was applied.

Fix: at solver init, when the deterministic cap is binding (user-requested
``max_iter`` exceeds it), emit one ``RuntimeWarning`` mentioning both
"deterministic" and the cap value.

Also covers T1.6 (1000-cap on inner adjoint/active-set PCG): when
user-requested ``max_iter > 1000``, init emits a one-shot ``RuntimeWarning``
mentioning "1000".

Both warnings are one-shot per solver instance.
"""

import importlib
import os
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
    geo_file = tmp_path / "cap.geo"
    geo_file.write_text(geo)
    msh_file = tmp_path / "cap.msh"
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


def test_deterministic_mode_warns_when_max_iter_capped(tmp_path, monkeypatch):
    """T1.5: deterministic env var + max_iter=2000 must emit a one-shot
    ``RuntimeWarning`` mentioning "deterministic" and "50"."""
    monkeypatch.setenv('TORCH_PF_CG_DETERMINISTIC', '1')

    # Reload damage_solver so the module-level _CG_DETERMINISTIC flag
    # picks up the env var.
    import phast.damage_solver as ds
    importlib.reload(ds)
    assert ds._CG_DETERMINISTIC is True

    from phast.fem_operators import FEMOperators
    mesh, mat = _build_mesh_and_material(tmp_path)
    fem = FEMOperators(mesh, mat)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ds.PhaseFieldDamageSolver(fem, max_iter=2000)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]
    matched = [m for m in msgs
               if 'deterministic' in m.lower() and '50' in m]
    assert len(matched) == 1, (
        f"Expected exactly one deterministic-cap RuntimeWarning mentioning "
        f"'deterministic' and '50'; got {len(matched)} matching of "
        f"{len(msgs)} RuntimeWarnings: {msgs}"
    )

    # Cleanup: clear env and reload so other tests are unaffected.
    monkeypatch.delenv('TORCH_PF_CG_DETERMINISTIC', raising=False)
    importlib.reload(ds)


def test_deterministic_mode_no_warning_when_max_iter_below_cap(tmp_path, monkeypatch):
    """If user requested ``max_iter <= 50``, the cap is not binding -- no
    warning should fire."""
    monkeypatch.setenv('TORCH_PF_CG_DETERMINISTIC', '1')

    import phast.damage_solver as ds
    importlib.reload(ds)

    from phast.fem_operators import FEMOperators
    mesh, mat = _build_mesh_and_material(tmp_path)
    fem = FEMOperators(mesh, mat)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ds.PhaseFieldDamageSolver(fem, max_iter=20)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]
    matched = [m for m in msgs
               if 'deterministic' in m.lower() and '50' in m]
    assert len(matched) == 0, (
        f"Cap is not binding (20 < 50) -- no deterministic warning expected. "
        f"Got: {msgs}"
    )

    monkeypatch.delenv('TORCH_PF_CG_DETERMINISTIC', raising=False)
    importlib.reload(ds)


def test_inner_pcg_1000_cap_warns_when_max_iter_exceeds(tmp_path):
    """T1.6: ``max_iter > 1000`` triggers a one-shot ``RuntimeWarning``
    mentioning "1000" -- the inner adjoint/active-set PCG hard cap."""
    from phast.damage_solver import PhaseFieldDamageSolver
    from phast.fem_operators import FEMOperators
    mesh, mat = _build_mesh_and_material(tmp_path)
    fem = FEMOperators(mesh, mat)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        PhaseFieldDamageSolver(fem, max_iter=2000)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]
    matched = [m for m in msgs if '1000' in m]
    assert len(matched) == 1, (
        f"Expected exactly one inner-PCG cap RuntimeWarning mentioning "
        f"'1000'; got {len(matched)} matching of {len(msgs)} "
        f"RuntimeWarnings: {msgs}"
    )


def test_inner_pcg_1000_cap_silent_when_max_iter_below(tmp_path):
    """``max_iter <= 1000`` must not emit the inner-PCG cap warning."""
    from phast.damage_solver import PhaseFieldDamageSolver
    from phast.fem_operators import FEMOperators
    mesh, mat = _build_mesh_and_material(tmp_path)
    fem = FEMOperators(mesh, mat)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        PhaseFieldDamageSolver(fem, max_iter=500)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]
    matched = [m for m in msgs if 'PCG hard cap' in m or 'hard cap of 1000' in m]
    assert len(matched) == 0, (
        f"Cap not binding (500 < 1000) -- no warning expected. Got: {msgs}"
    )
