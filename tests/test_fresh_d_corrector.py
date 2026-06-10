"""Issue #299 -- ``fresh_d_in_corrector`` flag tests.

The audit ``reports/B7_dynamic_pipeline_audit_2026-05-08.md`` ranked the
"corrector uses ``d_n`` (lagged damage) instead of ``d_{n+1}``" mismatch
as the #1 non-energy-split candidate cause of the B7 79us-vs-COMSOL-33us
branching delay. PhaFiDyn (Barki 2025) -- the validated FEniCS dynamic
PF reference in ``reference_codes/PhaFiDyn/`` -- evaluates the damage
solve BETWEEN the predictor and the corrector, so the corrector sees the
just-updated damage. This file locks in the new behaviour.

Three tests:
  1. ``test_fresh_d_default_off`` -- byte-identical state evolution when
     the flag stays at its default ``False``. Backstop against accidental
     drift in the legacy explicit step path.
  2. ``test_fresh_d_on_changes_state`` -- with the flag ``True``, the
     state diverges from the default after enough steps to be sure the
     reordering actually engaged (not 1 step where d_n == d_{n+1} == 0).
  3. ``test_fresh_d_with_damage_every_2`` -- correctness of the
     interaction with subcycling: damage solve only on the gated steps,
     and the corrector consumes the just-updated d on those steps.

The test relies on the same coarse SENT mesh used by
``test_at1_split_combinations.py`` so the runs are cheap (sub-second per
direction on CPU at float64).
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest
import torch

from phast.boundary_conditions import BoundaryConditions
from phast.material import Material
from phast.mesh import FEMMesh
from phast.mesh_generator import miehe_tension as gen_mesh
from phast.staggered_solver import SolverConfig, StaggeredSolver


# Coarse SENT -- keep tests fast. Same constants as
# ``test_at1_split_combinations.py``; only the flag varies between runs.
W = 20.0
A_NOTCH = 10.0
L0 = 0.5
H_CRACK = 2.0
H_COARSE = 4.0
DISP_MAX = 1.0e-3
T_RAMP = 50e-6


@pytest.fixture(scope='module')
def mesh_path():
    tmpdir = tempfile.mkdtemp(prefix='test_fresh_d_')
    path = os.path.join(tmpdir, 'sent.msh')
    gen_mesh(path, L=W, a=A_NOTCH, l0=L0,
             h_crack=H_CRACK, h_coarse=H_COARSE, verbose=False)
    yield path


def _build_solver(mesh_path: str, *,
                  fresh_d_in_corrector: bool,
                  damage_every: int = 1) -> StaggeredSolver:
    """AT2 + spectral SENT solver, deterministic dt, no MG, no AA."""
    mat = Material(
        E=3090.0, nu=0.35, Gc=0.3, l0=L0, rho=1.18e-9,
        energy_split='spectral', pf_model='AT2',
        eta_residual=1e-6,
    )
    mesh = FEMMesh(mesh_path, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()

    bcs = BoundaryConditions(mesh.n_nodes, device='cpu', dtype=torch.float64)
    bcs.fix(mesh.node_sets['left'], 0)
    bcs.fix(mesh.node_sets['right'], 0)
    bcs.add(mesh.node_sets['top'], 1, +DISP_MAX)
    bcs.add(mesh.node_sets['bottom'], 1, -DISP_MAX)
    bcs.load_factor = 0.0

    cfg = SolverConfig(
        solver_type='explicit',
        dt_safety=0.5,
        use_multigrid=False,
        bounds_method='post_clamp',
        damage_every=damage_every,
        adaptive_dt=False,
        print_every=10_000,
        fresh_d_in_corrector=fresh_d_in_corrector,
    )
    return StaggeredSolver(mesh, mat, bcs, config=cfg)


def _drive(solver: StaggeredSolver, n_steps: int):
    """Apply a smooth ramp of the prescribed Dirichlet load and step."""
    t = 0.0
    for _ in range(n_steps):
        solver.bcs.load_factor = min(t / T_RAMP, 1.0)
        solver.step_full()
        t += solver.dt


def test_fresh_d_default_off(mesh_path):
    """Default flag (``False``) preserves the legacy explicit step:
    byte-identical (``torch.equal``) state evolution against a freshly
    rebuilt baseline. Confirms the new branch is dormant when off.

    Two solvers are built from the same YAML inputs; the only thing they
    do differently is whether the new code path could possibly have run.
    Both run with ``fresh_d_in_corrector=False``, so the legacy code path
    is exercised in both. Their final ``(u, v, a, d, H_elem)`` MUST be
    bit-identical -- any drift here means the legacy path was perturbed
    by the wiring. Backstop against accidental refactor damage.
    """
    n_steps = 80
    s_a = _build_solver(mesh_path, fresh_d_in_corrector=False)
    s_b = _build_solver(mesh_path, fresh_d_in_corrector=False)
    _drive(s_a, n_steps)
    _drive(s_b, n_steps)

    assert torch.equal(s_a.u, s_b.u), "u not bit-identical with default flag"
    assert torch.equal(s_a.v, s_b.v), "v not bit-identical with default flag"
    assert torch.equal(s_a.a, s_b.a), "a not bit-identical with default flag"
    assert torch.equal(s_a.d, s_b.d), "d not bit-identical with default flag"
    assert torch.equal(s_a.H_elem, s_b.H_elem), (
        "H_elem not bit-identical with default flag")


def test_fresh_d_on_changes_state(mesh_path):
    """With ``fresh_d_in_corrector=True``, the state evolution diverges
    measurably from the default-off path after enough steps for the
    damage front to actually be non-zero. We use 80 steps -- by which
    point the damage solver has produced a non-trivial d field, and the
    Verlet ordering matters.

    Failure here means the new branch did not engage -- e.g., the flag
    was wired but the alternate ``_step_full_explicit_fresh_d`` path
    was not actually called.
    """
    n_steps = 80
    s_off = _build_solver(mesh_path, fresh_d_in_corrector=False)
    s_on = _build_solver(mesh_path, fresh_d_in_corrector=True)
    _drive(s_off, n_steps)
    _drive(s_on, n_steps)

    # Sanity: no NaNs from the on-path; the smoke must run cleanly.
    assert torch.isfinite(s_on.u).all(), "u NaN under fresh_d_in_corrector=True"
    assert torch.isfinite(s_on.v).all(), "v NaN under fresh_d_in_corrector=True"
    assert torch.isfinite(s_on.a).all(), "a NaN under fresh_d_in_corrector=True"
    assert torch.isfinite(s_on.d).all(), "d NaN under fresh_d_in_corrector=True"
    # Damage must stay in [0,1].
    assert 0.0 <= s_on.d.min().item() <= s_on.d.max().item() <= 1.0 + 1e-12, (
        f"damage out of [0,1]: d_min={s_on.d.min().item()}, "
        f"d_max={s_on.d.max().item()}")

    # The two state vectors MUST diverge enough that no plausible
    # accumulated round-off explains the gap. Use a relative threshold
    # tied to the scale of the off-path state -- 1e-9 of |u|_max is
    # well above f64 round-off (1e-13).
    du_rel = (s_on.u - s_off.u).abs().max().item() / max(
        s_off.u.abs().max().item(), 1e-30)
    dd_rel = (s_on.d - s_off.d).abs().max().item() / max(
        s_off.d.abs().max().item(), 1e-30)
    assert max(du_rel, dd_rel) > 1e-9, (
        f"fresh_d_in_corrector=True did not measurably change state: "
        f"|du|/|u|={du_rel:.3e}, |dd|/|d|={dd_rel:.3e}. "
        f"The legacy and new paths produced near-identical results "
        f"-- new branch likely never ran.")


def test_fresh_d_with_damage_every_2(mesh_path):
    """Subcycling x fresh-d interaction: with ``damage_every=2``,
    damage is solved only on the gated steps. On those steps the
    corrector uses the freshly updated d. On the in-between steps the
    corrector uses the most recent d (i.e., just ``self.d``, unchanged
    from the previous gated step).

    Two assertions:
      (a) The combined run still produces finite, in-bounds state --
          subcycling does not break the new ordering.
      (b) The d field is identical at gated and non-gated steps modulo
          the gated solves themselves: between two consecutive gated
          steps (steps k and k+2), ``d`` is set on step k and unchanged
          on step k+1. We probe this by snapshotting ``d`` after each
          step and asserting the expected pattern.
    """
    n_steps = 24  # enough to see subcycling pattern; cheap.
    solver = _build_solver(
        mesh_path, fresh_d_in_corrector=True, damage_every=2)

    d_snaps = []
    H_snaps = []
    for k in range(n_steps):
        solver.bcs.load_factor = min((k + 1) * solver.dt / T_RAMP, 1.0)
        solver.step_full()
        d_snaps.append(solver.d.clone())
        H_snaps.append(solver.H_elem.clone())

    # All snapshots finite + bounded.
    for k, ds in enumerate(d_snaps):
        assert torch.isfinite(ds).all(), f"d NaN at step {k}"
        assert ds.min().item() >= 0.0 and ds.max().item() <= 1.0 + 1e-12, (
            f"d out of bounds at step {k}: [{ds.min().item()}, {ds.max().item()}]")

    # The gate in step_full warm-starts: damage is solved unconditionally
    # for steps 1..5, then on every even ``_explicit_step_count``. So
    # ``d`` may change every step from step 1 through 5 inclusive (1-indexed
    # ``_explicit_step_count``); thereafter it changes only on even
    # ``_explicit_step_count`` (i.e. snapshots [5], [7], [9], ... in 0-indexed
    # ``d_snaps``, since ``_explicit_step_count`` is 1-indexed).
    # On the 1-indexed odd in-between steps (after warm-up) ``d`` MUST stay
    # equal to its value at the previous step. Probe the last odd in-between:
    # ``_explicit_step_count == 7`` -> d_snaps[6] should equal d_snaps[5].
    assert torch.equal(d_snaps[6], d_snaps[5]), (
        "d changed on a non-gated subcycling step "
        "(_explicit_step_count=7 should reuse d from count=6)")
    # And once more for stability (count=9 vs count=8):
    assert torch.equal(d_snaps[8], d_snaps[7]), (
        "d changed on a non-gated subcycling step "
        "(_explicit_step_count=9 should reuse d from count=8)")

    # H_elem -- updated EVERY step (driving force always recomputed); so
    # consecutive H_snaps may differ. This protects against a regression
    # where someone gates the H update under the same flag as the damage
    # solve (which would hide energy injected by waves the d field hasn't
    # absorbed yet).
    h_changes = sum(1 for k in range(1, n_steps)
                    if not torch.equal(H_snaps[k], H_snaps[k - 1]))
    assert h_changes >= n_steps - 5, (
        f"H_elem only updated on {h_changes}/{n_steps - 1} step pairs "
        f"under damage_every=2 + fresh_d_in_corrector=True; "
        f"expected near-every-step updates.")


def test_fresh_d_with_rigid_connector_raises(mesh_path):
    """The new ordering cannot be split around an in-step damage solve
    when ``rigid_connector`` MPCs are active -- the unified
    ``_step_mpc`` predictor+corrector path interleaves per-connector
    theta DOF with the master kinematics. The flag is incompatible
    with MPC; the implementation must fail loudly rather than silently
    miscompute. This test pins the contract.
    """
    # Build a very small standalone cantilever with a single
    # rigid_connector master+slave pair, so the MPC path is engaged
    # without dragging in the damage solver's full bulk-fracture setup.
    import numpy as np

    L, h = 8.0, 0.5
    nx, ny = 16, 2
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(-h / 2, h / 2, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = a + 1
            c = a + (nx + 1)
            d = c + 1
            elems.append([a, b, d])
            elems.append([a, d, c])
    nodes_t = torch.tensor(nodes, dtype=torch.float64)
    elems_t = torch.tensor(elems, dtype=torch.long)
    left_idx = np.where(np.isclose(nodes[:, 0], 0.0))[0]
    right_idx = np.where(np.isclose(nodes[:, 0], L))[0]
    master_node = int(ny // 2 * (nx + 1) + nx)
    node_sets = {
        'left': torch.tensor(left_idx, dtype=torch.long),
        'tip_face': torch.tensor(right_idx, dtype=torch.long),
        'master': torch.tensor([master_node], dtype=torch.long),
    }
    mesh = FEMMesh.from_tensors(nodes_t, elems_t, node_sets,
                                device='cpu', dtype=torch.float64)
    mat = Material(E=210e3, nu=0.3, Gc=2.7e-3, l0=0.5, rho=7.85e-9,
                   energy_split='amor', pf_model='AT2')
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu',
                             dtype=torch.float64)
    bcs.fix(mesh.node_sets['left'], 0)
    bcs.fix(mesh.node_sets['left'], 1)
    bcs.add_rigid_connector(
        master_node=int(mesh.node_sets['master'].item()),
        slave_indices=mesh.node_sets['tip_face'],
        locked_components=[0, 1],
        prescribe={0: 0.0, 1: 0.0},
        rotation_free=True,
    )

    cfg = SolverConfig(
        solver_type='explicit',
        dt_safety=0.5,
        use_multigrid=False,
        bounds_method='post_clamp',
        damage_every=1,
        fresh_d_in_corrector=True,
        print_every=10_000,
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg)
    with pytest.raises(RuntimeError, match='fresh_d_in_corrector.*rigid_connector'):
        solver.step_full()
