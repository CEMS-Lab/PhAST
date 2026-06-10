"""``_Ax`` aliased-buffer safety test.

Audit T1.3 (W4 SOLVER_BUGS_OPTIMISATION_AUDIT_2026-05-07.md): the eager
``PhaseFieldDamageSolver._Ax`` used to return a reference to the
preallocated buffer ``self._ax_out``. The compiled fast path returned a
freshly-allocated tensor. Any future refactor that interleaved a second
``_Ax`` call (e.g. a preconditioner reusing the matvec) would silently
corrupt the caller's cached residual / search direction without raising.

Fix: the eager path now returns ``out.clone()`` so semantics match the
compiled path -- two consecutive ``_Ax`` calls produce two independent
tensors. The cost is one N-vector copy per matvec, negligible against
the matvec itself.

Test contract: call ``_Ax`` twice; the first result must NOT be mutated
by the second call. Pre-fix this assertion fails on the eager path
(both calls return the same aliased buffer); post-fix it passes.
"""

import pytest
import torch


def _build_solver(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    from phast.mesh import FEMMesh
    from phast.material import Material
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
Mesh.CharacteristicLengthMax=0.4;
"""
    geo_file = tmp_path / "alias.geo"
    geo_file.write_text(geo)
    msh_file = tmp_path / "alias.msh"
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
    fem = FEMOperators(mesh, mat)
    solver = PhaseFieldDamageSolver(fem)
    return solver


def test_ax_returns_independent_tensors_eager_path(tmp_path):
    """Two consecutive ``_Ax`` calls (eager path) must return tensors that
    do not alias the same storage. Pre-fix they aliased ``self._ax_out``."""
    solver = _build_solver(tmp_path)
    # Force eager path explicitly
    solver._compiled_Ax = None
    assert solver._Gc_l0_e is None  # exercise the scalar-Gc branch

    n = solver._cg_n_nodes
    n_elem = solver._cg_elements.shape[0]
    torch.manual_seed(0)
    d1 = torch.rand(n, dtype=solver._cg_dtype, device=solver._cg_device)
    d2 = torch.rand(n, dtype=solver._cg_dtype, device=solver._cg_device)
    rc1 = torch.rand(n_elem, dtype=solver._cg_dtype, device=solver._cg_device)
    rc2 = torch.rand(n_elem, dtype=solver._cg_dtype, device=solver._cg_device)

    out1 = solver._Ax(d1, rc1)              # do NOT clone here
    out1_snapshot = out1.clone()             # capture value
    out2 = solver._Ax(d2, rc2)               # second call must not mutate out1

    assert not out1.data_ptr() == out2.data_ptr(), (
        "Eager _Ax returned aliased storage — two consecutive calls share "
        "the same underlying buffer. Post-fix they must be independent."
    )
    assert torch.allclose(out1, out1_snapshot, rtol=0, atol=0), (
        "Eager _Ax mutated the first result when called a second time."
    )


def test_ax_nodal_returns_independent_tensors(tmp_path):
    """Same alias-safety contract for the nodal-H matvec ``_Ax_nodal``."""
    gmsh = pytest.importorskip("gmsh")
    from phast.mesh import FEMMesh
    from phast.material import Material
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
Mesh.CharacteristicLengthMax=0.4;
"""
    geo_file = tmp_path / "alias_nodal.geo"
    geo_file.write_text(geo)
    msh_file = tmp_path / "alias_nodal.msh"
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
    fem = FEMOperators(mesh, mat)
    solver = PhaseFieldDamageSolver(fem, nodal_H=True)
    n = solver._cg_n_nodes
    torch.manual_seed(0)
    d1 = torch.rand(n, dtype=solver._cg_dtype, device=solver._cg_device)
    d2 = torch.rand(n, dtype=solver._cg_dtype, device=solver._cg_device)
    H1 = torch.rand(n, dtype=solver._cg_dtype, device=solver._cg_device)
    H2 = torch.rand(n, dtype=solver._cg_dtype, device=solver._cg_device)

    out1 = solver._Ax_nodal(d1, H1)
    out1_snapshot = out1.clone()
    out2 = solver._Ax_nodal(d2, H2)

    assert out1.data_ptr() != out2.data_ptr()
    assert torch.allclose(out1, out1_snapshot, rtol=0, atol=0)
