"""Spectral split + plane stress one-shot ``RuntimeWarning`` test.

Audit T1.4 (W4 SOLVER_BUGS_OPTIMISATION_AUDIT_2026-05-07.md): the
strain-spectral split decomposes only the in-plane strain tensor; under
plane stress the out-of-plane strain is not eigen-decomposed, making
the +/- energy partition approximate. Several ``Material`` presets
(`pmma_bleyer`, `cement_mortar_ambati`, ...) silently activate this
combination.

Fix: ``FEMOperators.compute_stress_spectral_*`` and
``FEMOperators._psi_plus_spectral`` emit a class-level one-shot
``RuntimeWarning`` on first call when ``material.plane_stress=True``.

Test contract:
- Warning fires exactly once per session when plane_stress=True + spectral.
- Warning does NOT fire for plane_strain.
- Warning does NOT fire for non-spectral splits (amor / isotropic).
"""

import warnings

import pytest
import torch


def _make_fem(plane_stress: bool, energy_split: str = 'spectral'):
    pytest.importorskip("gmsh")
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.fem_operators import FEMOperators

    import gmsh
    import tempfile
    import os
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
    tmp = tempfile.mkdtemp()
    geo_file = os.path.join(tmp, "ps.geo")
    msh_file = os.path.join(tmp, "ps.msh")
    with open(geo_file, 'w') as f:
        f.write(geo)
    gmsh.initialize()
    try:
        gmsh.open(geo_file)
        gmsh.model.mesh.generate(2)
        gmsh.write(msh_file)
    finally:
        gmsh.finalize()

    mesh = FEMMesh(msh_file, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()
    mat = Material(
        E=3090.0, nu=0.35, Gc=0.3, l0=0.1, rho=1.18e-9,
        energy_split=energy_split, pf_model='AT2',
        plane_stress=plane_stress,
    )
    fem = FEMOperators(mesh, mat)
    return fem


def _reset_class_warning_flag():
    """Reset the class-level one-shot flag so each test starts clean."""
    from phast.fem_operators import FEMOperators
    FEMOperators._plane_stress_spectral_warning_emitted = False


def _make_strain(fem):
    n_elem = fem.mesh.elements.shape[0]
    torch.manual_seed(0)
    exx = torch.rand(n_elem, dtype=torch.float64) * 1e-3
    eyy = torch.rand(n_elem, dtype=torch.float64) * 1e-3
    gxy = torch.rand(n_elem, dtype=torch.float64) * 1e-3
    return exx, eyy, gxy


def test_warning_fires_for_plane_stress_spectral_psi_plus():
    """``compute_psi_plus`` (which dispatches to ``_psi_plus_spectral``) must
    emit a ``RuntimeWarning`` on first call when plane_stress=True + spectral."""
    _reset_class_warning_flag()
    fem = _make_fem(plane_stress=True, energy_split='spectral')

    n_nodes = fem.mesh.n_nodes
    u = torch.zeros((n_nodes, 2), dtype=torch.float64)
    u[:, 0] = torch.linspace(0, 1e-3, n_nodes, dtype=torch.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fem.compute_psi_plus(u)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)
               and 'plane stress' in str(w.message).lower()
               and 'spectral' in str(w.message).lower()]
    assert len(runtime) == 1, (
        f"Expected exactly one plane-stress+spectral RuntimeWarning, got "
        f"{len(runtime)}: {[str(w.message) for w in caught]}"
    )


def test_warning_is_one_shot_per_session():
    """A second call must NOT emit the warning again."""
    _reset_class_warning_flag()
    fem = _make_fem(plane_stress=True, energy_split='spectral')

    n_nodes = fem.mesh.n_nodes
    u = torch.zeros((n_nodes, 2), dtype=torch.float64)
    u[:, 0] = torch.linspace(0, 1e-3, n_nodes, dtype=torch.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fem.compute_psi_plus(u)  # emits
        fem.compute_psi_plus(u)  # must NOT emit
        fem.compute_psi_plus(u)  # must NOT emit
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)
               and 'plane stress' in str(w.message).lower()
               and 'spectral' in str(w.message).lower()]
    assert len(runtime) == 1, (
        f"One-shot guarantee violated: warning emitted "
        f"{len(runtime)} times, expected 1."
    )


def test_warning_does_not_fire_for_plane_strain():
    """Plane strain + spectral must NOT trigger the approximation warning."""
    _reset_class_warning_flag()
    fem = _make_fem(plane_stress=False, energy_split='spectral')

    n_nodes = fem.mesh.n_nodes
    u = torch.zeros((n_nodes, 2), dtype=torch.float64)
    u[:, 0] = torch.linspace(0, 1e-3, n_nodes, dtype=torch.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fem.compute_psi_plus(u)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)
               and 'plane stress' in str(w.message).lower()
               and 'spectral' in str(w.message).lower()]
    assert len(runtime) == 0, (
        f"Plane-strain spectral spuriously emitted plane-stress warning: "
        f"{[str(w.message) for w in caught]}"
    )


def test_warning_does_not_fire_for_amor_split():
    """Plane stress + non-spectral (amor) must NOT trigger the warning."""
    _reset_class_warning_flag()
    fem = _make_fem(plane_stress=True, energy_split='amor')

    n_nodes = fem.mesh.n_nodes
    u = torch.zeros((n_nodes, 2), dtype=torch.float64)
    u[:, 0] = torch.linspace(0, 1e-3, n_nodes, dtype=torch.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fem.compute_psi_plus(u)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)
               and 'plane stress' in str(w.message).lower()
               and 'spectral' in str(w.message).lower()]
    assert len(runtime) == 0
