"""Shared helpers for standalone quasistatic benchmark drivers."""

from __future__ import annotations

import os
import math
from dataclasses import asdict, dataclass

import torch

from phast.io_utils import (
    init_h5,
    init_zarr,
    write_h5_snapshot,
    write_zarr_snapshot,
)


@dataclass
class TrajectoryWriter:
    """Small adapter hiding legacy H5 vs Zarr trajectory writers."""

    root: object
    kind: str
    path: str

    def write(self, *args, **kwargs) -> None:
        if self.kind == "h5":
            write_h5_snapshot(self.root, *args, **kwargs)
        else:
            write_zarr_snapshot(self.root, *args, **kwargs)

    def close(self, num_steps: int) -> None:
        if self.kind == "h5":
            self.root.attrs["num_steps"] = int(num_steps)
            self.root.close()
        else:
            self.root.attrs["num_steps"] = int(num_steps)


def open_trajectory_writers(output_dir, mesh, material, enabled, fmt="zarr"):
    """Open requested trajectory stores.

    The public benchmark flag is ``--trajectory``. The old ``--h5`` spelling
    remains as a backwards-compatible alias, but the default archived store is
    Zarr. Use ``--trajectory_format h5`` or ``both`` only for explicit legacy
    needs.
    """
    if not enabled:
        return []
    writers = []
    if fmt in ("zarr", "both"):
        path = os.path.join(output_dir, "training_data.zarr")
        writers.append(TrajectoryWriter(init_zarr(path, mesh, material),
                                        "zarr", path))
    if fmt in ("h5", "both"):
        path = os.path.join(output_dir, "training_data.h5")
        writers.append(TrajectoryWriter(init_h5(path, mesh, material),
                                        "h5", path))
    return writers


def snapshot_solver_state(solver):
    """Clone mutable solver state so a rejected load step can be retried."""
    damage_solver = getattr(solver, "damage_solver", None)
    visc_ref = getattr(damage_solver, "damage_viscosity_reference", None) if damage_solver is not None else None
    last_visc = getattr(damage_solver, "_last_viscous_d_prev", None) if damage_solver is not None else None
    if (damage_solver is not None
            and getattr(damage_solver, "damage_dt", None) is not None):
        damage_dt = damage_solver.damage_dt
    else:
        damage_dt = None
    return {
        "u": solver.u.detach().clone(),
        "v": solver.v.detach().clone(),
        "a": solver.a.detach().clone(),
        "d": solver.d.detach().clone(),
        "H_elem": solver.H_elem.detach().clone(),
        "H_nodal": solver.H_nodal.detach().clone(),
        "f_ext": solver.f_ext.detach().clone(),
        "_step_count": int(getattr(solver, "_step_count", 0)),
        "_last_stagger_iter": int(getattr(solver, "_last_stagger_iter", 0)),
        "_last_residual": float(getattr(solver, "_last_residual", float("nan"))),
        "_last_residual0": float(getattr(solver, "_last_residual0", float("nan"))),
        "_last_relative_residual": float(getattr(
            solver, "_last_relative_residual", float("nan"))),
        "_last_mechanics_residual": float(getattr(
            solver, "_last_mechanics_residual", float("nan"))),
        "_last_mechanics_residual0": float(getattr(
            solver, "_last_mechanics_residual0", float("nan"))),
        "_last_mechanics_relative_residual": float(getattr(
            solver, "_last_mechanics_relative_residual", float("nan"))),
        "_last_damage_load_factor": float(getattr(
            solver, "_last_damage_load_factor", 0.0)),
        "damage_solver_damage_dt": damage_dt,
        "damage_solver_viscosity_reference": (
            None if visc_ref is None else visc_ref.detach().clone()),
        "damage_solver_last_viscous_d_prev": (
            None if last_visc is None else last_visc.detach().clone()),
    }


def restore_solver_state(solver, state):
    """Restore a state captured by :func:`snapshot_solver_state`."""
    solver.u = state["u"].clone()
    solver.v = state["v"].clone()
    solver.a = state["a"].clone()
    solver.d = state["d"].clone()
    solver.H_elem = state["H_elem"].clone()
    solver.H_nodal = state["H_nodal"].clone()
    solver.f_ext = state["f_ext"].clone()
    solver._step_count = state["_step_count"]
    solver._last_stagger_iter = state["_last_stagger_iter"]
    solver._last_residual = state["_last_residual"]
    solver._last_residual0 = state.get("_last_residual0", float("nan"))
    solver._last_relative_residual = state.get(
        "_last_relative_residual", float("nan"))
    solver._last_mechanics_residual = state.get(
        "_last_mechanics_residual", float("nan"))
    solver._last_mechanics_residual0 = state.get(
        "_last_mechanics_residual0", float("nan"))
    solver._last_mechanics_relative_residual = state.get(
        "_last_mechanics_relative_residual", float("nan"))
    solver._last_damage_load_factor = state.get("_last_damage_load_factor", 0.0)
    damage_solver = getattr(solver, "damage_solver", None)
    if damage_solver is None:
        if (state.get("damage_solver_damage_dt") is not None
                or state.get("damage_solver_viscosity_reference") is not None
                or state.get("damage_solver_last_viscous_d_prev") is not None):
            # Legacy checkpoints may be restored against solvers with no
            # external damage-solver attribute. Keep this path strict and
            # fail fast to avoid silent attribute drift.
            raise AttributeError(
                "solver has no damage_solver but checkpoint contains "
                "damage solver state."
            )
        return
    damage_solver.damage_dt = state.get("damage_solver_damage_dt")
    damage_solver.damage_viscosity_reference = state.get(
        "damage_solver_viscosity_reference")
    damage_solver._last_viscous_d_prev = state.get(
        "damage_solver_last_viscous_d_prev")


def save_run_checkpoint(
    path,
    *,
    solver,
    bcs,
    pending_displacements,
    history,
    energy_rows,
    accepted_step,
    last_disp,
    cutback_count,
    consecutive_cutbacks,
    crack_step=None,
    crack_stage_snapshots=None,
    control_rows=None,
    continuation=None,
):
    """Write a restartable QS benchmark checkpoint.

    The trajectory/video stores are append-only outputs and are not resumed
    in-place. Restarted runs should use a fresh output directory, but the
    accepted nonlinear state, pending load schedule, history, and diagnostics
    are preserved exactly enough to continue the solve.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    payload = {
        "solver_state": snapshot_solver_state(solver),
        "load_factor": float(getattr(bcs, "load_factor", 0.0)),
        "pending_displacements": list(pending_displacements),
        "history": list(history),
        "energy_rows": list(energy_rows),
        "accepted_step": int(accepted_step),
        "last_disp": float(last_disp),
        "cutback_count": int(cutback_count),
        "consecutive_cutbacks": int(consecutive_cutbacks),
        "crack_step": crack_step,
        "crack_stage_snapshots": crack_stage_snapshots or {},
        "control_rows": list(control_rows or []),
        "continuation_state": (
            continuation.state_dict() if continuation is not None else None),
    }
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_run_checkpoint(path, *, solver, bcs):
    """Restore a checkpoint written by :func:`save_run_checkpoint`."""
    try:
        payload = torch.load(path, map_location=solver.u.device,
                             weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=solver.u.device)
    restore_solver_state(solver, payload["solver_state"])
    bcs.load_factor = float(payload.get("load_factor", 0.0))
    return payload


def write_load_step_control_csv(path, rows):
    """Write accepted/rejected adaptive-load decisions for auditability."""
    with open(path, "w") as fh:
        fh.write(
            "attempt,status,prev_disp,target_disp,accepted_step,reason,"
            "stagger_iter,residual,relative_residual,mechanics_residual,"
            "mechanics_relative_residual,max_d,total_energy,line_search_alpha,"
            "line_search_reductions,continuation_mode,arc_length_residual,"
            "arc_length_constraint,load_factor\n"
        )
        for i, row in enumerate(rows):
            fh.write(
                f"{i},{row.get('status', '')},"
                f"{float(row.get('prev_disp', 0.0)):.9e},"
                f"{float(row.get('target_disp', 0.0)):.9e},"
                f"{int(row.get('accepted_step', -1))},"
                f"{str(row.get('reason', '')).replace(',', ';')},"
                f"{int(row.get('stagger_iter', 0))},"
                f"{float(row.get('residual', float('nan'))):.9e},"
                f"{float(row.get('relative_residual', float('nan'))):.9e},"
                f"{float(row.get('mechanics_residual', float('nan'))):.9e},"
                f"{float(row.get('mechanics_relative_residual', float('nan'))):.9e},"
                f"{float(row.get('max_d', float('nan'))):.9e},"
                f"{float(row.get('total_energy', float('nan'))):.9e},"
                f"{float(row.get('line_search_alpha', 1.0)):.9e},"
                f"{int(row.get('line_search_reductions', 0))},"
                f"{row.get('continuation_mode', '')},"
                f"{float(row.get('arc_length_residual', float('nan'))):.9e},"
                f"{float(row.get('arc_length_constraint', float('nan'))):.9e},"
                f"{float(row.get('load_factor', float('nan'))):.9e}\n"
            )


def add_continuation_args(parser):
    """Add path-following options shared by standalone QS drivers.

    This is an incremental displacement/load-factor continuation controller,
    not a monolithic Crisfield-Riks augmented Newton solve. It lets the
    validated staggered equilibrium solve continue after peak by changing the
    next prescribed load factor based on accepted reaction/damage history.
    """
    parser.add_argument('--arc_length', action='store_true',
                        help=('Enable displacement/load-factor path-following '
                              'for post-peak QS continuation.'))
    parser.add_argument('--arc_length_solver', default='riks',
                        choices=('riks', 'controller'),
                        help=('Continuation implementation. "riks" solves an '
                              'augmented mechanics system with the load factor '
                              'as an unknown; "controller" keeps the older '
                              'external load-factor scheduler.'))
    parser.add_argument('--arc_length_alpha', type=float, default=1.0,
                        help=('Load-factor weight in the augmented arc-length '
                              'constraint. Increase when displacement DOFs '
                              'dominate the constraint norm.'))
    parser.add_argument('--arc_length_steps', type=int, default=None,
                        help='Maximum accepted steps under --arc_length.')
    parser.add_argument('--arc_length_ds', type=float, default=None,
                        help='Initial absolute load-factor increment.')
    parser.add_argument('--arc_length_min_ds', type=float, default=None,
                        help='Minimum absolute load-factor increment.')
    parser.add_argument('--arc_length_max_ds', type=float, default=None,
                        help='Maximum absolute load-factor increment.')
    parser.add_argument('--arc_length_min_disp', type=float, default=0.0,
                        help='Lower load-factor bound for continuation.')
    parser.add_argument('--arc_length_max_disp', type=float, default=None,
                        help='Upper load-factor bound for continuation.')
    parser.add_argument('--arc_length_damage_trigger', type=float, default=0.35,
                        help='Damage level above which peak/reversal logic is active.')
    parser.add_argument('--arc_length_reaction_drop', type=float, default=0.005,
                        help='Relative accepted reaction drop that triggers reversal.')
    parser.add_argument('--arc_length_post_peak_steps', type=int, default=20,
                        help='Accepted continuation steps after the first reversal.')
    parser.add_argument('--arc_length_allow_reversal', default=True,
                        action=torch_bool_action(),
                        help='Allow displacement/load-factor reversal after peak.')


def add_stagger_acceptance_args(parser):
    """Add QS stagger non-convergence policy controls.

    ``fail`` is the production default.  ``warn`` preserves the historical
    diagnostic behaviour.  ``phasefieldx`` mirrors the reference PhaseFieldX
    examples: a load step that reaches the stagger cap is written with its
    reported stagger count instead of being rejected by the outer driver.
    """
    parser.add_argument(
        '--stagger_nonconvergence_policy',
        choices=('fail', 'warn', 'phasefieldx'),
        default='fail',
        help=(
            'Policy when the implicit stagger loop reaches --max_stagger. '
            '"fail" raises and lets adaptive cutback retry. "warn" accepts '
            'with a warning. "phasefieldx" accepts capped stagger steps and '
            'disables stagger/residual cutback gates, matching the bundled '
            'PhaseFieldX benchmark scripts for parity diagnostics.'
        ),
    )


def apply_stagger_acceptance_policy(args):
    """Mutate parsed QS CLI args according to the requested policy.

    Returns the value to pass into ``SolverConfig.fail_on_stagger_nonconvergence``.
    """
    policy = getattr(args, 'stagger_nonconvergence_policy', 'fail')
    if policy == 'fail':
        return True
    if policy == 'phasefieldx':
        # PhaseFieldX writes output when the outer stagger loop reaches its cap.
        # Prevent the adaptive driver from immediately rejecting that same step.
        if hasattr(args, 'cutback_stagger_fraction'):
            args.cutback_stagger_fraction = max(
                float(args.cutback_stagger_fraction), 1.01)
        if hasattr(args, 'residual_cutback_limit'):
            args.residual_cutback_limit = math.inf
    return False


def torch_bool_action():
    """Return argparse's BooleanOptionalAction without importing argparse here."""
    import argparse
    return argparse.BooleanOptionalAction


@dataclass
class ArcLengthController:
    """Small stateful load-factor path-following controller.

    The controller operates outside the mechanics Newton solve: each accepted
    equilibrium point is still solved by the existing staggered solver. The
    controller decides the next prescribed displacement/load factor, including
    reversing the increment after a detected peak so unstable post-peak crack
    growth can be sampled without hand-authored schedules.
    """

    enabled: bool
    max_steps: int
    ds: float
    min_ds: float
    max_ds: float
    min_disp: float
    max_disp: float
    damage_trigger: float
    reaction_drop: float
    post_peak_steps: int
    allow_reversal: bool = True
    direction: float = 1.0
    peak_abs_reaction: float = 0.0
    reversed: bool = False
    steps_after_reversal: int = 0

    def state_dict(self):
        """Return restartable controller state."""
        return asdict(self)

    def load_state_dict(self, state):
        """Restore controller state from a checkpoint payload."""
        if not state:
            return
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_args(cls, args, displacements):
        if not getattr(args, 'arc_length', False):
            return cls(False, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0, True)
        if len(displacements) >= 2:
            base_ds = abs(float(displacements[1]) - float(displacements[0]))
        elif displacements:
            base_ds = max(abs(float(displacements[0])), 1e-6)
        else:
            base_ds = 1e-6
        ds = float(args.arc_length_ds or base_ds)
        min_ds = float(args.arc_length_min_ds or max(ds * 0.05, 1e-9))
        max_ds = float(args.arc_length_max_ds or max(ds, base_ds))
        max_disp = args.arc_length_max_disp
        if max_disp is None:
            max_disp = max(float(x) for x in displacements) if displacements else ds
        max_steps = int(args.arc_length_steps or args.num_steps)
        return cls(
            enabled=True,
            max_steps=max_steps,
            ds=abs(ds),
            min_ds=abs(min_ds),
            max_ds=abs(max_ds),
            min_disp=float(args.arc_length_min_disp),
            max_disp=float(max_disp),
            damage_trigger=float(args.arc_length_damage_trigger),
            reaction_drop=float(args.arc_length_reaction_drop),
            post_peak_steps=int(args.arc_length_post_peak_steps),
            allow_reversal=bool(args.arc_length_allow_reversal),
        )

    def seed_pending(self, displacements):
        """Return initial pending list for a run."""
        if not self.enabled:
            return list(displacements)
        if not displacements:
            return []
        return [float(displacements[0])]

    def on_cutback(self):
        if self.enabled:
            self.ds = max(self.min_ds, 0.5 * self.ds)

    def next_after_accept(self, *, accepted_step, disp, reaction, max_d):
        """Return next target displacement or ``None`` when complete."""
        if not self.enabled:
            return None
        if accepted_step >= self.max_steps:
            return None

        abs_r = abs(float(reaction))
        if abs_r > self.peak_abs_reaction:
            self.peak_abs_reaction = abs_r
            if not self.reversed:
                self.ds = min(self.max_ds, self.ds * 1.05)

        peak_ready = (
            self.allow_reversal
            and not self.reversed
            and max_d >= self.damage_trigger
            and self.peak_abs_reaction > 0.0
            and abs_r < (1.0 - self.reaction_drop) * self.peak_abs_reaction
        )
        if peak_ready:
            self.direction *= -1.0
            self.reversed = True
            self.ds = max(self.min_ds, min(self.ds, self.max_ds))

        if self.reversed:
            self.steps_after_reversal += 1
            if self.steps_after_reversal > self.post_peak_steps:
                return None

        nxt = float(disp) + self.direction * self.ds
        if nxt > self.max_disp:
            if self.allow_reversal:
                self.direction = -1.0
                self.reversed = True
                nxt = float(disp) + self.direction * self.ds
            else:
                return None
        if nxt < self.min_disp:
            if self.reversed:
                return None
            nxt = self.min_disp
        return max(self.min_disp, min(self.max_disp, nxt))


def should_cutback_step(
    *,
    enabled: bool,
    finite: bool,
    prev_disp: float,
    target_disp: float,
    prev_max_d: float,
    max_d: float,
    stagger_iter: int,
    max_stagger: int,
    min_load_step: float,
    damage_trigger: float,
    damage_increment_limit: float,
    stagger_fraction: float,
    residual: float = float("nan"),
    residual_limit: float = float("inf"),
    exception_reason: str | None = None,
    prev_total_energy: float = float("nan"),
    total_energy: float = float("nan"),
    energy_jump_limit: float = float("inf"),
):
    """Return ``(cutback, reason)`` for QS load-step rejection.

    This is intentionally conservative: it only refines when the run is in
    or near crack growth, the solver hit the stagger cap, non-finite state
    appeared, or one load increment creates an unrealistically large damage
    jump. It does not change the accepted physics state.
    """
    if not enabled:
        return False, ""

    step_size = abs(target_disp - prev_disp)
    if step_size <= min_load_step:
        return False, ""

    if not finite:
        if exception_reason:
            return True, exception_reason
        return True, "non-finite state"

    if residual_limit < float("inf"):
        residual_t = torch.as_tensor(residual)
        if not torch.isfinite(residual_t) or float(residual) > residual_limit:
            return True, f"residual {float(residual):.3e}"

    if stagger_iter >= max(1, int(stagger_fraction * max_stagger)):
        return True, f"stagger iterations {stagger_iter}/{max_stagger}"

    in_damage_zone = max(prev_max_d, max_d) >= damage_trigger
    if in_damage_zone and energy_jump_limit < float("inf"):
        prev_t = torch.as_tensor(prev_total_energy)
        total_t = torch.as_tensor(total_energy)
        if torch.isfinite(prev_t) and torch.isfinite(total_t):
            rel_jump = abs(float(total_energy) - float(prev_total_energy)) \
                / max(abs(float(prev_total_energy)), 1e-30)
            if rel_jump > energy_jump_limit:
                return True, f"energy jump {rel_jump:.3e}"

    damage_jump = max_d - prev_max_d
    if in_damage_zone and damage_jump > damage_increment_limit:
        return True, f"damage jump {damage_jump:.3f}"

    return False, ""
