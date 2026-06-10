"""Compiled ``_Ax`` Gc_l0-freshness test.

Audit T1.2 (W4 SOLVER_BUGS_OPTIMISATION_AUDIT_2026-05-07.md): the inner
``_Ax_impl`` defined inside ``PhaseFieldDamageSolver._try_compile_matvec``
used to *capture* the Python float ``Gc_l0 = self._Gc_l0`` in its closure.
After ``torch.compile``, mutating ``solver._Gc_l0`` (which the
``_AdjointDamage*`` autograd Functions in inverse-problem demos do
in-place when they swap in trial Gc/l0 values) silently desynchronised
the compiled fast path from the eager fallback: eager re-read
``self._Gc_l0`` live, compiled kept the stale capture.

Fix: ``_Ax_impl`` now takes ``Gc_l0`` as a positional argument and the
dispatch site at ``_Ax`` passes ``self._Gc_l0`` on every call.

This test does not require CUDA. It bypasses the
``ctx.compile_solvers and self._cg_device.type == 'cuda'`` gate by
manually attaching the *un-compiled* ``_Ax_impl`` (a plain Python
callable that takes ``Gc_l0``) as ``solver._compiled_Ax``. The dispatch
codepath through ``_Ax`` is the same regardless of whether the callable
went through ``torch.compile``; what we are pinning is the argument
contract between dispatch and inner implementation.

Pre-fix (``_Ax_impl`` captured ``Gc_l0`` from the enclosing scope and
``_Ax`` called ``self._compiled_Ax(d, reaction_coeff)``): mutating
``solver._Gc_l0`` left the captured value stale, the compiled path
diverged from eager, and this test would fail.

Post-fix: ``_Ax`` passes ``self._Gc_l0`` as a third argument; the test
passes.
"""

import pytest
import torch


def _build_solver(tmp_path):
    """Construct a tiny PhaseFieldDamageSolver on CPU (float64) for testing."""
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
    geo_file = tmp_path / "freshness.geo"
    geo_file.write_text(geo)
    msh_file = tmp_path / "freshness.msh"
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
    return mesh, mat, solver


def _build_compiled_ax_callable(solver):
    """Return a plain Python callable matching the post-fix _Ax_impl signature.

    Mirrors ``PhaseFieldDamageSolver._try_compile_matvec`` but skips the
    ``torch.compile`` wrap (so the test runs anywhere, no CUDA required).
    The callable takes ``(d, reaction_coeff, Gc_l0)`` — the post-fix
    contract.
    """
    gp = solver._cg_grad_phi
    areas_col = solver._areas_col
    elem_flat = solver._elem_flat
    elements = solver._cg_elements
    n_nodes = solver._cg_n_nodes
    cg_dtype = solver._cg_dtype
    cg_device = solver._cg_device

    def _Ax_impl(d, reaction_coeff, Gc_l0):
        d_e = d[elements]
        gd_x = (gp[:, :, 0] * d_e).sum(1)
        gd_y = (gp[:, :, 1] * d_e).sum(1)
        lap = areas_col * (gp[:, :, 0] * gd_x.unsqueeze(1) +
                           gp[:, :, 1] * gd_y.unsqueeze(1))
        out = torch.zeros(n_nodes, dtype=cg_dtype, device=cg_device)
        out.scatter_add_(0, elem_flat, lap.flatten())
        d_sum = d_e.sum(1)
        mass_contrib = reaction_coeff.unsqueeze(1) * (d_e + d_sum.unsqueeze(1))
        react = torch.zeros(n_nodes, dtype=cg_dtype, device=cg_device)
        react.scatter_add_(0, elem_flat, mass_contrib.flatten())
        return Gc_l0 * out + react

    return _Ax_impl


def test_compiled_ax_picks_up_gc_l0_mutation(tmp_path):
    """After mutating ``solver._Gc_l0`` post-compile, the compiled-path
    ``_Ax`` output must match the eager-path ``_Ax`` output.

    Pre-fix the compiled callable captured ``Gc_l0`` once at compile time
    and silently desynced; this assertion catches that regression.
    """
    mesh, mat, solver = _build_solver(tmp_path)

    # Sanity: the bug only matters when there's no per-element Gc_l0
    # (otherwise ``_Ax`` skips the compiled path; see damage_solver.py:1188).
    assert solver._Gc_l0_e is None

    # Attach the un-compiled inner callable as the "compiled" matvec.
    # The dispatch logic in ``_Ax`` is the only thing under test.
    solver._compiled_Ax = _build_compiled_ax_callable(solver)

    n = solver._cg_n_nodes
    torch.manual_seed(0)
    d = torch.rand(n, dtype=solver._cg_dtype, device=solver._cg_device)
    n_elem = solver._cg_elements.shape[0]
    reaction_coeff = torch.rand(n_elem, dtype=solver._cg_dtype,
                                device=solver._cg_device)

    # ---- Step 1: baseline at original Gc/l0. Compiled and eager must agree. ----
    out_compiled_orig = solver._Ax(d, reaction_coeff).clone()
    solver._compiled_Ax_saved = solver._compiled_Ax
    solver._compiled_Ax = None  # force eager path
    out_eager_orig = solver._Ax(d, reaction_coeff).clone()
    solver._compiled_Ax = solver._compiled_Ax_saved

    assert torch.allclose(out_compiled_orig, out_eager_orig, rtol=1e-12, atol=1e-14), (
        "Compiled and eager _Ax disagree even at original Gc/l0 — "
        "test scaffolding is broken."
    )

    # ---- Step 2: mutate solver._Gc_l0 (mimics _AdjointDamage* swap-in). ----
    new_Gc_l0 = solver._Gc_l0 * 2.0
    solver._Gc_l0 = new_Gc_l0

    out_compiled_mut = solver._Ax(d, reaction_coeff).clone()
    solver._compiled_Ax_saved = solver._compiled_Ax
    solver._compiled_Ax = None
    out_eager_mut = solver._Ax(d, reaction_coeff).clone()
    solver._compiled_Ax = solver._compiled_Ax_saved

    # Eager must reflect the new Gc_l0 (it reads self._Gc_l0 live at line 1220).
    assert not torch.allclose(out_eager_mut, out_eager_orig), (
        "Eager _Ax did not respond to a 2x Gc_l0 mutation — "
        "test pre-condition violated."
    )

    # The fix: compiled path now also reflects the mutation.
    # Pre-fix this assertion failed (compiled returned out_compiled_orig
    # because Gc_l0 was captured by closure at compile time).
    assert torch.allclose(out_compiled_mut, out_eager_mut, rtol=1e-12, atol=1e-14), (
        "Compiled _Ax did NOT pick up Gc_l0 mutation — audit T1.2 regression. "
        f"Diff norm: {(out_compiled_mut - out_eager_mut).norm().item():.3e}"
    )

    # Stronger: the difference between mutated-compiled and mutated-eager
    # vs mutated-compiled and original-compiled tells us the compiled path
    # responded to the mutation rather than ignored it.
    delta_response = (out_compiled_mut - out_compiled_orig).norm().item()
    assert delta_response > 1e-8, (
        "Compiled _Ax output did not change after Gc_l0 mutation — "
        "stale-closure regression."
    )
