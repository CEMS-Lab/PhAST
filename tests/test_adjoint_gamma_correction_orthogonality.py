"""Issue #93 — orthogonality between scalar/field adjoint Functions.

The two adjoint Functions in ``damage_solver.py`` historically had
non-orthogonal handling of the ``gamma_correction`` flag:

* ``_AdjointDamageSolveScalar`` raised ``NotImplementedError`` when
  ``solver._gamma_correction=True``.
* ``_AdjointDamageSolveField`` requires ``_gamma_correction=True``.

This test suite locks the four-corner orthogonality matrix:

    | adjoint  | gamma=True | gamma=False |
    | scalar   |    works   |    works    |
    | field    |    works   |   raises    |

Plus a finite-difference check on the scalar+gamma path (the new
behavior) so we are sure the chain rule
``dL/dGc_scalar = sum_e gamma_factor_e * dL/dGc_eff_e``
is wired correctly.

Runtime is kept under 2 s by using a tiny ~3x3 unit-square mesh.
"""
import torch
import pytest


def _build_tiny_unit_square(tmp_path, h: float = 0.5):
    """Produce a minimal 2-element-style unit square gmsh mesh.

    ``h=0.5`` gives a ~5-node, 4-element mesh — small enough for a sub-2-s
    test, large enough for the per-element gamma_factor_e to be nontrivial
    (i.e. ``elem_h / l0 != 0`` so gamma_correction actually does something).
    """
    geo = f"""
SetFactory("OpenCASCADE");
Point(1) = {{0,0,0,{h}}};
Point(2) = {{1,0,0,{h}}};
Point(3) = {{1,1,0,{h}}};
Point(4) = {{0,1,0,{h}}};
Line(1)={{1,2}}; Line(2)={{2,3}}; Line(3)={{3,4}}; Line(4)={{4,1}};
Curve Loop(1)={{1,2,3,4}}; Plane Surface(1)={{1}};
Physical Surface("plate")={{1}};
Physical Curve("bottom")={{1}};
Physical Curve("right") ={{2}};
Physical Curve("top")   ={{3}};
Physical Curve("left")  ={{4}};
Mesh.ElementOrder=1;
Mesh.CharacteristicLengthMax={h};
"""
    geo_file = tmp_path / "ortho.geo"
    msh_file = tmp_path / "ortho.msh"
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
    return str(msh_file)


def _build_solver(tmp_path, gamma_correction: bool):
    pytest.importorskip("gmsh")
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.fem_operators import FEMOperators
    from phast.damage_solver import PhaseFieldDamageSolver

    msh_path = _build_tiny_unit_square(tmp_path, h=0.5)
    mesh = FEMMesh(msh_path, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()
    mat = Material(
        E=210.0, nu=0.3, Gc=2.7, l0=0.3, pf_model='AT2',
        energy_split='spectral', gamma_correction=gamma_correction,
    )
    fem = FEMOperators(mesh, mat)
    solver = PhaseFieldDamageSolver(fem)
    # Tighten CG so FD comparisons are not dominated by solver residual.
    solver.tol = 1e-12
    solver.max_iter = 5000
    return mesh, mat, solver


def _make_H_and_d_prev(solver):
    """Build a synthetic, nontrivial H_input and zero d_prev."""
    n_nodes = solver._cg_n_nodes
    n_elem = solver._cg_elements.shape[0]
    torch.manual_seed(0)
    # Element-wise H with mild spatial variation, comfortably above the
    # AT1 nucleation threshold so the solve is non-degenerate.
    H_input = (1.0 + 0.3 * torch.rand(n_elem, dtype=torch.float64))
    d_prev = torch.zeros(n_nodes, dtype=torch.float64)
    return H_input, d_prev


# ----------------------------------------------------------------------------
# Orthogonality matrix
# ----------------------------------------------------------------------------

def test_scalar_gamma_true_now_works(tmp_path):
    """Issue #93: scalar adjoint with gamma_correction=True must not raise."""
    from phast.damage_solver import _AdjointDamageSolveScalar

    _, _, solver = _build_solver(tmp_path, gamma_correction=True)
    H_input, d_prev = _make_H_and_d_prev(solver)

    Gc_t = torch.tensor(2.7, dtype=torch.float64, requires_grad=True)
    l0_t = torch.tensor(0.3, dtype=torch.float64, requires_grad=False)
    d_new = _AdjointDamageSolveScalar.apply(solver, H_input, d_prev, Gc_t, l0_t)
    loss = d_new.sum()
    loss.backward()

    assert Gc_t.grad is not None
    assert torch.isfinite(Gc_t.grad).all()


def test_scalar_gamma_false_unchanged(tmp_path):
    """Regression: scalar adjoint without gamma_correction still works."""
    from phast.damage_solver import _AdjointDamageSolveScalar

    _, _, solver = _build_solver(tmp_path, gamma_correction=False)
    H_input, d_prev = _make_H_and_d_prev(solver)

    Gc_t = torch.tensor(2.7, dtype=torch.float64, requires_grad=True)
    l0_t = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    d_new = _AdjointDamageSolveScalar.apply(solver, H_input, d_prev, Gc_t, l0_t)
    loss = d_new.sum()
    loss.backward()

    assert Gc_t.grad is not None and torch.isfinite(Gc_t.grad).all()
    assert l0_t.grad is not None and torch.isfinite(l0_t.grad).all()


def test_field_gamma_true_unchanged(tmp_path):
    """Regression: field adjoint with gamma_correction works as before."""
    from phast.damage_solver import _AdjointDamageSolveField

    _, _, solver = _build_solver(tmp_path, gamma_correction=True)
    H_input, d_prev = _make_H_and_d_prev(solver)
    n_elem = solver._cg_elements.shape[0]

    Gc_e_t = torch.full((n_elem,), 2.7, dtype=torch.float64,
                        requires_grad=True)
    l0_t = torch.tensor(0.3, dtype=torch.float64, requires_grad=False)
    d_new = _AdjointDamageSolveField.apply(solver, H_input, d_prev, Gc_e_t, l0_t)
    loss = d_new.sum()
    loss.backward()

    assert Gc_e_t.grad is not None
    assert torch.isfinite(Gc_e_t.grad).all()


def test_field_gamma_false_still_raises(tmp_path):
    """Field adjoint requires gamma_correction=True (unchanged contract).

    The acceptance bar in the issue is *orthogonality*, not "field works
    everywhere". The field path needs the per-element ``_Gc_l0_e`` arrays
    that only exist when the solver is built with ``gamma_correction=True``.
    """
    from phast.damage_solver import _AdjointDamageSolveField

    _, _, solver = _build_solver(tmp_path, gamma_correction=False)
    H_input, d_prev = _make_H_and_d_prev(solver)
    n_elem = solver._cg_elements.shape[0]

    Gc_e_t = torch.full((n_elem,), 2.7, dtype=torch.float64,
                        requires_grad=True)
    l0_t = torch.tensor(0.3, dtype=torch.float64, requires_grad=False)
    with pytest.raises(RuntimeError, match="gamma_correction"):
        _AdjointDamageSolveField.apply(
            solver, H_input, d_prev, Gc_e_t, l0_t)


# ----------------------------------------------------------------------------
# Correctness: scalar+gamma_correction matches finite-difference Gc gradient.
# ----------------------------------------------------------------------------

def test_scalar_gamma_correction_matches_finite_difference(tmp_path):
    """The new chain-rule reduction must match central FD on Gc_scalar.

    This is the acceptance criterion in issue #93: ``dL/dGc_scalar`` from
    autograd matches ``(L(Gc+h) - L(Gc-h))/(2h)`` to within 1e-5 relative.
    """
    from phast.damage_solver import _AdjointDamageSolveScalar

    _, _, solver = _build_solver(tmp_path, gamma_correction=True)
    H_input, d_prev = _make_H_and_d_prev(solver)

    # Random output projection so the scalar loss has nontrivial structure.
    torch.manual_seed(7)
    n_nodes = solver._cg_n_nodes
    w = torch.randn(n_nodes, dtype=torch.float64)

    Gc_val = 2.7
    l0_t = torch.tensor(0.3, dtype=torch.float64, requires_grad=False)

    # ---- Autograd ----
    Gc_t = torch.tensor(Gc_val, dtype=torch.float64, requires_grad=True)
    d_new = _AdjointDamageSolveScalar.apply(solver, H_input, d_prev, Gc_t, l0_t)
    loss = (w * d_new).sum()
    loss.backward()
    grad_ad = float(Gc_t.grad.item())

    # ---- Central FD ----
    h = 1e-5 * Gc_val
    def loss_at(gc):
        Gc_p = torch.tensor(gc, dtype=torch.float64, requires_grad=False)
        d_p = _AdjointDamageSolveScalar.apply(
            solver, H_input, d_prev, Gc_p, l0_t)
        return float((w * d_p).sum().item())
    grad_fd = (loss_at(Gc_val + h) - loss_at(Gc_val - h)) / (2.0 * h)

    rel = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-30)
    assert rel < 1e-5, (
        f"Autograd dL/dGc={grad_ad:+.6e} disagrees with FD={grad_fd:+.6e} "
        f"(rel err {rel:.2e})")


def test_scalar_gamma_warns_when_l0_requires_grad(tmp_path):
    """Issue #93: l0 grad under gamma_correction is the direct-only term;
    a RuntimeWarning must be emitted so users know the gamma_factor(l0)
    indirect dependence is not chain-ruled."""
    import warnings
    from phast.damage_solver import _AdjointDamageSolveScalar

    _, _, solver = _build_solver(tmp_path, gamma_correction=True)
    H_input, d_prev = _make_H_and_d_prev(solver)

    Gc_t = torch.tensor(2.7, dtype=torch.float64, requires_grad=True)
    l0_t = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d_new = _AdjointDamageSolveScalar.apply(
            solver, H_input, d_prev, Gc_t, l0_t)
        d_new.sum().backward()

    msg_hits = [w for w in caught
                if issubclass(w.category, RuntimeWarning)
                and "gamma_correction" in str(w.message)]
    assert msg_hits, "expected a gamma_correction l0-grad warning"
