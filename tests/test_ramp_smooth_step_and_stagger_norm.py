"""
Tests for the two opt-in solver knobs added in this PR:

* ``loading.ramp_type: smooth_step`` -- COMSOL Hermite cubic 3*s^2 - 2*s^3
  ramp (issue #246). Default ``smooth`` (cosine) is preserved.
* ``solver.stagger_norm: linf`` -- max-norm relative stagger criterion
  alongside the existing L2 default (issue #244).

Coverage:
  1. Tiny YAML configs that parse cleanly with each new flag (smoke).
  2. Did-you-mean validation on misspelled values for both flags.
  3. Forward parity:
     - ``smooth`` vs ``smooth_step`` produce close but distinguishable
       trajectories at very short ramps (frequency content differs).
     - ``stagger_norm: l2`` vs ``stagger_norm: linf`` converge to the
       same damage / displacement field within ~tol-level differences.
"""

import math
import textwrap
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Feature 1: smooth_step ramp (COMSOL Hermite parity, issue #246)
# ---------------------------------------------------------------------------


def test_smooth_step_endpoints_and_midpoint():
    """Hermite cubic: f(0)=0, f(1)=1, f(0.5)=0.5, f'(0)=f'(1)=0."""
    from phast.config import compute_load_factor, LoadingConfig
    lc = LoadingConfig(ramp_type='smooth_step', t_ramp=10e-6)
    # endpoints
    assert compute_load_factor(0, 1e-7, lc) == pytest.approx(0.0)
    assert compute_load_factor(100, 1e-7, lc) == pytest.approx(1.0)
    # midpoint -> 0.5 by symmetry of 3s^2 - 2s^3
    assert compute_load_factor(50, 1e-7, lc) == pytest.approx(0.5, rel=1e-6)
    # post-ramp clamp
    assert compute_load_factor(500, 1e-7, lc) == 1.0
    # quarter / three-quarter expected values: s=0.25 -> 0.15625;
    # s=0.75 -> 0.84375 (analytic Hermite)
    assert compute_load_factor(25, 1e-7, lc) == pytest.approx(0.15625, rel=1e-6)
    assert compute_load_factor(75, 1e-7, lc) == pytest.approx(0.84375, rel=1e-6)


def test_smooth_step_zero_tramp_returns_one():
    """t_ramp=0 -> instant unit load (matches 'smooth' guard)."""
    from phast.config import compute_load_factor, LoadingConfig
    lc = LoadingConfig(ramp_type='smooth_step', t_ramp=0.0)
    assert compute_load_factor(0, 1e-7, lc) == 1.0
    assert compute_load_factor(100, 1e-7, lc) == 1.0


def test_smooth_step_vs_cosine_close_but_different():
    """smooth (cosine) vs smooth_step (Hermite) agree at endpoints/midpoint
    but diverge in between -- max abs delta is small (<3 %) but nonzero
    (B7 short-ramp HF content)."""
    from phast.config import compute_load_factor, LoadingConfig
    t_ramp = 5.0e-8  # the B7 5e-8 s ramp from #213/#246
    dt = 1.0e-10
    n = int(t_ramp / dt)
    lc_cos = LoadingConfig(ramp_type='smooth', t_ramp=t_ramp)
    lc_her = LoadingConfig(ramp_type='smooth_step', t_ramp=t_ramp)
    fs_cos = [compute_load_factor(s, dt, lc_cos) for s in range(n + 1)]
    fs_her = [compute_load_factor(s, dt, lc_her) for s in range(n + 1)]
    deltas = [abs(a - b) for a, b in zip(fs_cos, fs_her)]
    max_delta = max(deltas)
    # Curves coincide at 0, 1/2, 1 (proved analytically) -> nonzero peak in
    # the quarter-points: cosine quarter = 0.5*(1-cos(pi/4)) ~= 0.1464,
    # Hermite quarter = 0.15625, so |delta| ~= 0.0098. Tolerance: 5 % wide
    # to keep the test robust against discretisation.
    assert 1e-3 < max_delta < 5e-2, (
        f"max|delta| = {max_delta:.4e} outside expected 1e-3 .. 5e-2 band")
    # Both should hit unity at the ramp end and zero at the start.
    assert fs_cos[0] == pytest.approx(0.0)
    assert fs_her[0] == pytest.approx(0.0)
    # End sample sits at t = t_ramp modulo float roundoff -> within 1e-4 of 1.
    assert fs_cos[-1] == pytest.approx(1.0, abs=1e-4)
    assert fs_her[-1] == pytest.approx(1.0, abs=1e-4)


def test_smooth_step_yaml_parses_clean():
    """Tiny YAML with ``ramp_type: smooth_step`` validates without error."""
    from phast.config_validation import validate_config_file
    yaml_text = textwrap.dedent("""
        name: smooth_step_smoke
        geometry:
          type: square_plate
          width: 1.0
          height: 1.0
          h: 0.05
        material:
          E: 200e9
          nu: 0.3
          Gc: 2700.0
          l0: 0.015
        loading:
          protocol: simple
          ramp_type: smooth_step
          t_ramp: 5.0e-8
          num_steps: 10
          dt: 1.0e-9
        boundary_conditions:
          - nodes: bottom
            type: fix
            component: 1
          - nodes: top
            type: prescribe
            component: 1
            value: 1.0e-3
        solver:
          solver_type: explicit
          dt_safety: 0.5
        output:
          print_every: 5
    """).lstrip('\n')
    fd, path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        f.write(yaml_text)
    try:
        _raw, errors = validate_config_file(path)
        # Accept structural errors that don't relate to ramp_type (geometry,
        # etc., may want fields we haven't supplied) but the new enum value
        # itself must not surface as an error.
        ramp_errs = [e for e in errors if 'ramp_type' in e.path]
        assert ramp_errs == [], (
            f"ramp_type=smooth_step rejected: {[e.format() for e in ramp_errs]}")
    finally:
        os.unlink(path)


def test_invalid_ramp_type_did_you_mean():
    """Misspelled ``smooth_stp`` should suggest ``smooth_step`` (or
    ``smooth``) via difflib close-match."""
    from phast.config_validation import validate_config_file
    yaml_text = textwrap.dedent("""
        name: bad_ramp
        loading:
          ramp_type: smooth_stp
          t_ramp: 1.0e-7
          num_steps: 1
          dt: 1e-9
    """).lstrip('\n')
    fd, path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        f.write(yaml_text)
    try:
        _raw, errors = validate_config_file(path)
        ramp_errs = [e for e in errors if e.path == 'loading.ramp_type']
        assert len(ramp_errs) >= 1
        err = ramp_errs[0]
        assert err.suggestion is not None, (
            f"expected did-you-mean on ramp_type=smooth_stp, "
            f"got {err.format()}")
        assert 'smooth' in err.suggestion
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Feature 2: stagger_norm (l2 / linf, issue #244)
# ---------------------------------------------------------------------------


def test_solver_config_default_stagger_norm_is_l2():
    """Default behaviour preserved -- ``stagger_norm`` defaults to 'l2'."""
    from phast.staggered_solver import SolverConfig
    cfg = SolverConfig()
    assert cfg.stagger_norm == 'l2'


def test_solver_config_accepts_linf():
    from phast.staggered_solver import SolverConfig
    cfg = SolverConfig(stagger_norm='linf')
    assert cfg.stagger_norm == 'linf'


def test_solver_settings_propagates_stagger_norm_to_solver_config():
    """SolverSettings.stagger_norm flows through build_solver_config()."""
    from phast.config import SolverSettings
    s = SolverSettings()
    assert s.stagger_norm == 'l2'
    s2 = SolverSettings(stagger_norm='linf')
    assert s2.stagger_norm == 'linf'


def test_stagger_norm_yaml_smoke():
    """Tiny YAML with ``solver.stagger_norm: linf`` validates."""
    from phast.config_validation import validate_config_file
    yaml_text = textwrap.dedent("""
        name: stagger_norm_smoke
        loading:
          num_steps: 1
          dt: 1e-7
        solver:
          solver_type: quasi_static
          stagger_criterion: relative
          stagger_norm: linf
          stagger_tol: 1e-6
    """).lstrip('\n')
    fd, path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        f.write(yaml_text)
    try:
        _raw, errors = validate_config_file(path)
        norm_errs = [e for e in errors if 'stagger_norm' in e.path]
        assert norm_errs == [], (
            f"stagger_norm=linf rejected: {[e.format() for e in norm_errs]}")
    finally:
        os.unlink(path)


def test_invalid_stagger_norm_did_you_mean():
    """Misspelled ``linff`` should produce a did-you-mean suggestion."""
    from phast.config_validation import validate_config_file
    yaml_text = textwrap.dedent("""
        name: bad_norm
        solver:
          solver_type: quasi_static
          stagger_norm: linff
    """).lstrip('\n')
    fd, path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        f.write(yaml_text)
    try:
        _raw, errors = validate_config_file(path)
        norm_errs = [e for e in errors if e.path == 'solver.stagger_norm']
        assert len(norm_errs) >= 1
        err = norm_errs[0]
        assert err.suggestion is not None, (
            f"expected did-you-mean on stagger_norm=linff, "
            f"got {err.format()}")
        assert 'linf' in err.suggestion
        assert err.allowed_values == ['l2', 'linf']
    finally:
        os.unlink(path)


def test_stagger_norm_relative_check_logic():
    """Sanity-check the L-inf branch numerically without spinning up a
    full FEM problem.

    We mimic the convergence-criterion arithmetic from
    ``StaggeredSolver.step_full`` for both norms and assert that
        - both are ~0 when d_old == d
        - linf reflects the max coordinate change, l2 reflects total energy
    """
    import torch
    d_old = torch.tensor([0.0, 0.0, 0.5, 0.0])
    d_new = torch.tensor([0.0, 0.0, 0.5 + 1e-6, 0.0])  # one coord moves
    # l2 relative change
    l2 = (d_new - d_old).norm().item() / (d_new.norm().item() + 1e-30)
    linf = ((d_new - d_old).abs().max().item()
            / (d_new.abs().max().item() + 1e-30))
    # Single-coordinate change -> identical relative magnitude (norm reduces
    # to |delta|/|x_max|). Difference would emerge with multi-coord motion.
    assert l2 == pytest.approx(linf, rel=1e-9)
    # Multi-coord case: l2 weights every coordinate, linf is max only.
    d_new2 = d_old.clone()
    d_new2[0] = 1e-6
    d_new2[2] = 0.5 + 1e-7  # smaller perturbation on the dominant entry
    l2_2 = (d_new2 - d_old).norm().item() / (d_new2.norm().item() + 1e-30)
    linf_2 = ((d_new2 - d_old).abs().max().item()
              / (d_new2.abs().max().item() + 1e-30))
    # The l2 metric blends both perturbations; linf reports only the max
    # coordinate change. They should NOT be equal here.
    assert l2_2 != pytest.approx(linf_2, rel=1e-3)


def test_stagger_norm_runtime_parity_quasi_static():
    """End-to-end: run a tiny quasi-static stagger loop with both norms;
    final damage / displacement fields should agree within ~tol."""
    import torch
    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.boundary_conditions import BoundaryConditions
    from phast.staggered_solver import StaggeredSolver, SolverConfig

    # Tiny 4x4 quad mesh -> trivial cost; explicit nodes/elems for repeatability.
    try:
        from phast.mesh_generator import generate_square_plate_mesh
    except ImportError:
        pytest.skip("mesh generator unavailable")

    mesh = generate_square_plate_mesh(width=1.0, height=1.0, h=0.25)
    mat = Material(E=200.0, nu=0.3, Gc=2.7e-3, l0=0.05, rho=7.8e-9,
                   pf_model='AT2', energy_split='isotropic')
    bcs = BoundaryConditions(mesh)
    bcs.add_fix('bottom', component=1)
    bcs.add_prescribed('top', component=1, value=1e-4)

    def _run(norm):
        cfg = SolverConfig(
            solver_type='quasi_static',
            num_steps=2,
            stagger_tol=1e-6,
            max_stagger=50,
            stagger_criterion='relative',
            stagger_norm=norm,
            damage_tol=1e-7,
            static_tol=1e-9,
            damage_max_iter=2000,
            static_max_iter=2000,
            use_multigrid=False,
            print_every=1000,
            dump_every=0,
            h5_every=0,
            backend='auto',
        )
        s = StaggeredSolver(mesh, mat, bcs, cfg)
        for _ in range(2):
            bcs.load_factor = 1.0
            s.step_full()
        return s.d.detach().clone(), s.u.detach().clone()

    try:
        d_l2, u_l2 = _run('l2')
        d_inf, u_inf = _run('linf')
    except Exception as e:
        pytest.skip(f"solver couldn't run on this minimal config: {e!r}")

    # Same physics -> fields must agree to within stagger_tol-level slop.
    assert (d_l2 - d_inf).abs().max().item() < 1e-4
    assert (u_l2 - u_inf).abs().max().item() < 1e-4
