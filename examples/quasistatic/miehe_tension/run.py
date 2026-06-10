#!/usr/bin/env python
"""
Miehe SENT (Single-Edge-Notch Tension) — Full benchmark run.

Uses StaggeredSolver for stagger iterations (no custom stagger loop).
All stagger logic, Anderson acceleration, convergence criteria, and
H capping are handled by the solver.

Usage:
    python -u examples/quasistatic/miehe_tension/run.py
    python -u examples/quasistatic/miehe_tension/run.py --all_outputs
    python -u examples/quasistatic/miehe_tension/run.py --trajectory --gif
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from phast.boundary_conditions import BoundaryConditions
from phast.device import DeviceContext
from phast.io_utils import (
    generate_run_tag, save_run_metadata, write_energy_csv,
    write_solver_telemetry_csv, plot_energy_history, write_vtu,
)
from phast.mesh import FEMMesh
from phast.mesh_generator import miehe_tension as gen_mesh
from phast.material import create_material
from phast.staggered_solver import SolverConfig, StaggeredSolver
from phast.visualization import (
    GIFRecorder,
    plot_field,
    plot_initial_conditions,
    plot_quasistatic_convergence,
)
from examples.quasistatic._run_utils import (
    ArcLengthController,
    add_continuation_args,
    load_run_checkpoint,
    open_trajectory_writers,
    restore_solver_state,
    save_run_checkpoint,
    should_cutback_step,
    snapshot_solver_state,
    write_load_step_control_csv,
)
# theory_report module not present; use fem.compute_energies inline below.


# ------------------------------------------------------------------ #
# Problem parameters (PhaseFieldX Example 1711)
# ------------------------------------------------------------------ #
PROBLEM = {
    'name': 'Miehe SENT (Single-Edge-Notch Tension)',
    'reference': 'Miehe et al. (2010), PhaseFieldX Example 1711',
    'L': 1.0,            # plate side length [mm]
    'a': 0.5,            # notch length [mm]
    'E': 210000.0,       # Young modulus [MPa]
    'nu': 0.3,
    'Gc': 2.7,           # fracture toughness [N/mm]
    'l0': 0.015,         # regularization length [mm]
    'energy_split': 'isotropic',
    'pf_model': 'AT2',
    'eta_residual': 0.0,  # PhaseFieldX k=0; Material applies tiny numerical floor
    # Loading: 50 fast steps (dt=1e-4) + slow steps (dt=1e-5)
    'dt_fast': 1e-4,     # [mm]
    'n_fast': 50,
    'dt_slow': 1e-5,     # [mm]
}


def setup_bcs(mesh):
    """SENT BCs matching PhaseFieldX 1711.

    Bottom is fixed in x and y. Top has prescribed vertical displacement
    only; horizontal sliding is free.
    """
    bcs = BoundaryConditions(mesh.n_nodes, mesh.device, mesh.dtype)
    bcs.fix(mesh.node_sets['bottom'], 0)
    bcs.fix(mesh.node_sets['bottom'], 1)
    bcs.add(mesh.node_sets['top'], 1, 1.0)  # unit disp, scaled by load_factor
    return bcs


def build_loading_schedule(n_steps):
    """PhaseFieldX-style loading: fast ramp then slow ramp."""
    P = PROBLEM
    disps = []
    for t in range(1, n_steps + 1):
        if t <= P['n_fast']:
            disps.append(P['dt_fast'] * t)
        else:
            disps.append(P['n_fast'] * P['dt_fast'] +
                         P['dt_slow'] * (t - P['n_fast']))
    return disps


def main():
    parser = argparse.ArgumentParser(description=PROBLEM['name'])
    # Mesh
    parser.add_argument('--h_crack', type=float, default=0.002)
    parser.add_argument('--h_coarse', type=float, default=0.05)
    parser.add_argument('--l0', type=float, default=PROBLEM['l0'])
    # Solver
    parser.add_argument('--num_steps', type=int, default=150)
    parser.add_argument('--energy_split', type=str, default=None,
                        choices=['isotropic', 'amor', 'spectral', 'star_convex'])
    parser.add_argument('--at_mode', type=str, default='at2',
                        choices=['at1', 'at2'],
                        help='Phase-field model: AT1 or AT2 (default: at2)')
    parser.add_argument('--stagger_tol', type=float, default=1e-8)
    parser.add_argument('--max_stagger', type=int, default=500)
    parser.add_argument('--static_max_iter', type=int, default=5000,
                        help='Maximum Newton iterations for the mechanics solve.')
    parser.add_argument('--stagger_criterion', type=str, default='relative',
                        choices=['absolute', 'relative', 'linf', 'residual',
                                 'am_energy'])
    parser.add_argument('--damage_cg_tol', type=float, default=1e-6)
    parser.add_argument('--mechanics_cg_tol', type=float, default=1e-8)
    parser.add_argument('--H_cap_factor', type=float, default=0.0,
                        help='Optional non-reference cap on H as a multiple of Gc/(2*l0).')
    parser.add_argument('--anderson_depth', type=int, default=0)
    parser.add_argument('--adaptive_stagger_tol', action='store_true')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'scipy', 'mumps', 'cg'],
                        help='Quasistatic mechanics linear solver backend.')
    parser.add_argument('--solver_type', type=str, default='quasi_static',
                        choices=['quasi_static', 'quasi_static_legacy',
                                 'monolithic'])
    parser.add_argument('--newton_line_search', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='Use residual backtracking in QuasiStaticSolver.')
    parser.add_argument('--line_search_max_steps', type=int, default=8)
    parser.add_argument('--line_search_min_alpha', type=float, default=1e-4)
    parser.add_argument('--line_search_c', type=float, default=1e-4)
    parser.add_argument('--preconditioner', type=str, default='jacobi',
                        choices=['jacobi', 'spectral', 'gmg', 'amg', 'amgx', 'auto'],
                        help='Damage CG preconditioner. Default avoids fragile CPU AMG in QS runs.')
    parser.add_argument('--multigrid', default=True,
                        action=argparse.BooleanOptionalAction)
    # Device
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--compile', action='store_true')
    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--trajectory', '--h5', dest='h5', action='store_true',
                        help='Write trajectory snapshots. Default store is Zarr; use --trajectory_format h5 for legacy H5.')
    parser.add_argument('--trajectory_format', choices=['zarr', 'h5', 'both'],
                        default='zarr')
    parser.add_argument('--vtu', action='store_true')
    parser.add_argument('--gif', action='store_true')
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--all_outputs', action='store_true')
    parser.add_argument('--vtu_every', type=int, default=1)
    parser.add_argument('--gif_frames', type=int, default=150)
    parser.add_argument('--gif_fields', type=str,
                        default='damage,von_mises_stress,H')
    parser.add_argument('--animation_format', choices=['gif', 'apng', 'mp4'],
                        default='mp4')
    parser.add_argument('--stop_at_crack', action='store_true')
    parser.add_argument('--stop_after_crack_steps', type=int, default=1,
                        help='When --stop_at_crack is set, continue this many extra load steps after max(d)>0.99.')
    parser.add_argument('--adaptive_load_steps', action='store_true',
                        help='Reject and bisect difficult quasistatic load increments near crack growth.')
    parser.add_argument('--min_load_step', type=float, default=1e-6)
    parser.add_argument('--damage_refine_trigger', type=float, default=0.20)
    parser.add_argument('--damage_increment_limit', type=float, default=0.04)
    parser.add_argument('--residual_cutback_limit', type=float, default=float('inf'),
                        help='Reject adaptive QS steps whose final stagger residual exceeds this value.')
    parser.add_argument('--energy_jump_limit', type=float, default=float('inf'),
                        help='Reject adaptive QS steps with a relative total-energy jump above this value.')
    parser.add_argument('--cutback_stagger_fraction', type=float, default=0.85)
    parser.add_argument('--max_cutbacks', type=int, default=16)
    parser.add_argument('--checkpoint_every', type=int, default=1,
                        help='Write restart checkpoint every N accepted steps; 0 disables.')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--restart_from', type=str, default=None,
                        help='Resume from a checkpoint produced with --checkpoint_every.')
    parser.add_argument('--print_every', type=int, default=1)
    add_continuation_args(parser)
    args = parser.parse_args()

    if args.all_outputs:
        args.h5 = args.vtu = args.gif = args.plots = args.profile = True

    if not any([args.h5, args.vtu, args.gif, args.plots, args.profile]):
        print("No output flags set. Use --all_outputs or --trajectory for training data.\n")

    active = [f for f, v in [('VTU', args.vtu), ('GIF', args.gif),
                              ('plots', args.plots), ('profiler', args.profile),
                              (args.trajectory_format.upper(), args.h5)] if v]
    print(f"  Outputs: {', '.join(active) if active else 'none'}")

    compile_flag = True if args.compile else None
    ctx = DeviceContext(device=args.device, profile=args.profile,
                        compile_solvers=compile_flag)

    if args.output_dir is None:
        base_dir = os.path.dirname(__file__)
        args.output_dir = os.path.join(
            base_dir,
            generate_run_tag(ctx.device, extra=f'sent_{args.at_mode}'))
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  {PROBLEM['name']}")
    print(f"  {PROBLEM['reference']}")
    print("=" * 60)

    # ---- Material ----
    l0 = args.l0
    mat_kwargs = dict(l0=l0, eta_residual=PROBLEM['eta_residual'],
                      pf_model=args.at_mode.upper())
    if args.energy_split:
        mat_kwargs['energy_split'] = args.energy_split
    mat = create_material('miehe_tension', **mat_kwargs)
    print(mat)

    # ---- H cap ----
    H_cap = None
    if args.H_cap_factor > 0:
        H_cap = args.H_cap_factor * mat.Gc / (2.0 * l0)

    # ---- Mesh ----
    mesh_path = os.path.join(args.output_dir, 'mesh.msh')
    gen_mesh(mesh_path, L=PROBLEM['L'], a=PROBLEM['a'], l0=l0,
             h_crack=args.h_crack, h_coarse=args.h_coarse)
    mesh = FEMMesh(mesh_path, device=ctx.device, dtype=ctx.dtype)
    mesh.identify_boundaries()
    print(mesh)

    # ---- BCs (unit displacement, scaled by load_factor per step) ----
    bcs = setup_bcs(mesh)

    # ---- Solver (all stagger logic delegated here) ----
    cfg = SolverConfig(
        solver_type=args.solver_type,
        num_steps=args.num_steps,
        damage_tol=args.damage_cg_tol,
        static_tol=args.mechanics_cg_tol,
        static_max_iter=args.static_max_iter,
        stagger_tol=args.stagger_tol,
        max_stagger=args.max_stagger,
        stagger_criterion=args.stagger_criterion,
        use_multigrid=args.multigrid,
        preconditioner=args.preconditioner,
        anderson_depth=args.anderson_depth,
        adaptive_stagger_tol=args.adaptive_stagger_tol,
        H_cap=H_cap,
        backend=args.backend,
        newton_line_search=args.newton_line_search,
        line_search_max_steps=args.line_search_max_steps,
        line_search_min_alpha=args.line_search_min_alpha,
        line_search_c=args.line_search_c,
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg, ctx=ctx)
    fem = solver.fem

    # ---- Loading schedule ----
    displacements = build_loading_schedule(args.num_steps)
    continuation = ArcLengthController.from_args(args, displacements)
    reaction_nodes = mesh.node_sets['bottom']

    # ---- Initial conditions plot ----
    if args.plots:
        bcs.load_factor = displacements[0]
        plot_initial_conditions(mesh, mat, bcs, cfg,
                               save_path=os.path.join(args.output_dir,
                                                      'initial_conditions.png'))

    # ---- Trajectory store ----
    trajectory_writers = open_trajectory_writers(
        args.output_dir, mesh, mat, args.h5, fmt=args.trajectory_format)

    # ---- GIF ----
    gif_recorder = None
    gif_every = 1
    if args.gif:
        gif_recorder = GIFRecorder(mesh, output_dir=args.output_dir,
                                  fields=args.gif_fields.split(','))
        gif_every = max(1, args.num_steps // args.gif_frames)

    # ---- CSV ----
    from phast.io_utils import CSVHistory
    csv_hist = CSVHistory(os.path.join(args.output_dir, 'history.csv'))

    # ---- Run ----
    history = []
    energy_rows = []
    control_rows = []
    t_total_start = time.time()
    crack_step = None
    crack_stage_targets = [0.0030, 0.0045, 0.0060]
    crack_stage_snapshots = {}

    mode = "path-following" if continuation.enabled else "load"
    print(f"\nRunning {args.num_steps} {mode} steps, "
          f"max_disp={displacements[-1]:.6f} mm\n")

    pending_displacements = continuation.seed_pending(displacements)
    accepted_step = 0
    cutback_count = 0
    consecutive_cutbacks = 0
    last_disp = 0.0
    checkpoint_path = args.checkpoint_path or os.path.join(
        args.output_dir, 'restart_checkpoint.pt')

    if args.restart_from:
        payload = load_run_checkpoint(args.restart_from, solver=solver, bcs=bcs)
        pending_displacements = list(payload['pending_displacements'])
        history = list(payload.get('history', []))
        energy_rows = list(payload.get('energy_rows', []))
        accepted_step = int(payload.get('accepted_step', len(history)))
        last_disp = float(payload.get('last_disp', last_disp))
        cutback_count = int(payload.get('cutback_count', 0))
        consecutive_cutbacks = int(payload.get('consecutive_cutbacks', 0))
        crack_step = payload.get('crack_step', crack_step)
        crack_stage_snapshots = payload.get('crack_stage_snapshots', crack_stage_snapshots)
        control_rows = list(payload.get('control_rows', control_rows))
        continuation.load_state_dict(payload.get('continuation_state'))
        print(f"Restarted from {args.restart_from}: "
              f"{accepted_step} accepted steps, {len(pending_displacements)} pending.")

    while pending_displacements:
        disp_val = pending_displacements.pop(0)
        t0 = time.time()
        state0 = snapshot_solver_state(solver)
        prev_max_d = float(state0['d'].max().item())

        # Update load factor -- BCs scale automatically. In Riks mode this
        # is only the predictor; the augmented mechanics solve returns the
        # accepted load factor and rewrites bcs.load_factor.
        bcs.load_factor = disp_val

        # Full stagger step (all Anderson, convergence, H capping inside solver)
        try:
            if (continuation.enabled
                    and getattr(args, 'arc_length_solver', 'riks') == 'riks'):
                arc_ds = max(abs(disp_val - last_disp),
                             float(getattr(continuation, 'min_ds', 1e-12)))
                psi = solver.step_full_arc_length(
                    lambda_prev=last_disp,
                    lambda_init=disp_val,
                    ds=arc_ds,
                    alpha=args.arc_length_alpha,
                    u_prev=state0['u'])
                disp_val = float(bcs.load_factor)
            else:
                psi = solver.step_full()
            finite = bool(torch.isfinite(solver.u).all()
                          and torch.isfinite(solver.d).all()
                          and torch.isfinite(solver.H_nodal).all())
        except Exception as exc:
            psi = None
            finite = False
            step_error = exc
        else:
            step_error = None

        elapsed = (time.time() - t0) * 1000
        stag_count = solver._last_stagger_iter
        max_d = solver.d.max().item() if finite else float('nan')
        trial_energies = None
        if finite and psi is not None:
            trial_energies = fem.compute_energy_components(
                solver.u, solver.d, getattr(solver, 'v', None), psi_plus=psi)
        trial_total_energy = (trial_energies or {}).get('total', float('nan'))
        prev_total_energy = (energy_rows[-1]['total'] if energy_rows
                            else float('nan'))

        cutback, reason = should_cutback_step(
            enabled=args.adaptive_load_steps,
            finite=finite,
            prev_disp=last_disp,
            target_disp=disp_val,
            prev_max_d=prev_max_d,
            max_d=max_d,
            stagger_iter=stag_count,
            max_stagger=args.max_stagger,
            min_load_step=args.min_load_step,
            damage_trigger=args.damage_refine_trigger,
            damage_increment_limit=args.damage_increment_limit,
            stagger_fraction=args.cutback_stagger_fraction,
            residual=float(getattr(solver, '_last_residual', float('nan'))),
            residual_limit=args.residual_cutback_limit,
            exception_reason=str(step_error) if step_error is not None else None,
            prev_total_energy=prev_total_energy,
            total_energy=trial_total_energy,
            energy_jump_limit=args.energy_jump_limit,
        )
        if cutback:
            control_rows.append({
                'status': 'rejected',
                'prev_disp': last_disp,
                'target_disp': disp_val,
                'accepted_step': accepted_step,
                'reason': reason,
                'stagger_iter': stag_count,
                'residual': float(getattr(solver, '_last_residual', float('nan'))),
                'max_d': max_d,
                'total_energy': trial_total_energy,
                'line_search_alpha': float(getattr(
                    solver.mechanics, 'last_line_search_alpha', 1.0)),
                'line_search_reductions': int(getattr(
                    solver.mechanics, 'last_line_search_reductions', 0)),
                'continuation_mode': (
                    args.arc_length_solver if continuation.enabled else ''),
                'arc_length_residual': float(getattr(
                    solver.mechanics, 'last_arc_length_residual', float('nan'))),
                'arc_length_constraint': float(getattr(
                    solver.mechanics, 'last_arc_length_constraint', float('nan'))),
                'load_factor': float(getattr(
                    solver.mechanics, 'last_load_factor', bcs.load_factor)),
            })
            restore_solver_state(solver, state0)
            bcs.load_factor = last_disp
            continuation.on_cutback()
            mid = 0.5 * (last_disp + disp_val)
            pending_displacements.insert(0, disp_val)
            pending_displacements.insert(0, mid)
            cutback_count += 1
            consecutive_cutbacks += 1
            print(f"  [cutback {cutback_count:02d}] {last_disp:.8f} -> "
                  f"{disp_val:.8f} split at {mid:.8f} ({reason})")
            if consecutive_cutbacks > args.max_cutbacks:
                raise RuntimeError(
                    f"Exceeded --max_cutbacks={args.max_cutbacks}; "
                    f"last rejected step {last_disp:.8f}->{disp_val:.8f} "
                    f"because {reason}") from step_error
            continue
        if step_error is not None:
            raise step_error

        # Reaction force
        R = fem.compute_reaction_force(solver.u, solver.d, reaction_nodes,
                                      component=1)

        # Divergence detection
        if not torch.isfinite(solver.u).all():
            print(f"  ** DIVERGENCE at accepted step {accepted_step}. Stopping. **")
            break

        dt_used = abs(disp_val - history[-1]['disp']) if history else abs(disp_val)
        record = {
            'step': accepted_step, 'disp': disp_val,
            'reaction_N': R, 'reaction_kN': -R / 1000.0,
            'max_d': max_d, 'max_H': solver.H_nodal.max().item(),
            'stagger_iter': stag_count, 'elapsed_ms': elapsed,
            'time': disp_val, 'newton_iters': stag_count,
            'pcg_iters_mech': int(getattr(solver.mechanics, 'last_iter', 0)),
            'pcg_iters_pf': int(getattr(solver.damage_solver, 'last_iter', 0)),
            'residual': float(getattr(solver, '_last_residual', float('nan'))),
            'relative_residual': float(getattr(
                solver, '_last_relative_residual', float('nan'))),
            'mechanics_residual': float(getattr(
                solver, '_last_mechanics_residual', float('nan'))),
            'mechanics_relative_residual': float(getattr(
                solver, '_last_mechanics_relative_residual', float('nan'))),
            'dt': dt_used,
            'line_search_alpha': float(getattr(
                solver.mechanics, 'last_line_search_alpha', 1.0)),
            'line_search_reductions': int(getattr(
                solver.mechanics, 'last_line_search_reductions', 0)),
            'continuation_direction': continuation.direction,
            'continuation_ds': continuation.ds,
            'continuation_reversed': int(continuation.reversed),
            'continuation_mode': (
                args.arc_length_solver if continuation.enabled else ''),
            'arc_length_residual': float(getattr(
                solver.mechanics, 'last_arc_length_residual', float('nan'))),
            'arc_length_constraint': float(getattr(
                solver.mechanics, 'last_arc_length_constraint', float('nan'))),
            'load_factor': float(getattr(
                solver.mechanics, 'last_load_factor', bcs.load_factor)),
        }
        history.append(record)
        control_rows.append({
            'status': 'accepted',
            'prev_disp': last_disp,
            'target_disp': disp_val,
            'accepted_step': accepted_step,
            'reason': '',
            'stagger_iter': stag_count,
            'residual': record['residual'],
            'relative_residual': record['relative_residual'],
            'mechanics_residual': record['mechanics_residual'],
            'mechanics_relative_residual':
                record['mechanics_relative_residual'],
            'max_d': max_d,
            'total_energy': trial_total_energy,
            'line_search_alpha': record['line_search_alpha'],
            'line_search_reductions': record['line_search_reductions'],
            'continuation_mode': record['continuation_mode'],
            'arc_length_residual': record['arc_length_residual'],
            'arc_length_constraint': record['arc_length_constraint'],
            'load_factor': record['load_factor'],
        })
        energies = trial_energies or fem.compute_energy_components(
            solver.u, solver.d, getattr(solver, 'v', None), psi_plus=psi)
        energy_rows.append({
            'step': accepted_step, 'time': disp_val,
            'elastic': energies['elastic'],
            'fracture': energies['fracture'],
            'kinetic': energies['kinetic'],
            'external': 0.0,
            'total': energies['total'],
        })

        if accepted_step % args.print_every == 0:
            print(f"  Step {accepted_step:3d}: u={disp_val:.6f}, "
                  f"R={-R/1000:.4f} kN, max(d)={max_d:.6f}, "
                  f"stag={stag_count} ({elapsed:.0f}ms)")

        # ---- Per-step outputs ----
        strain = solver._last_strain
        exx, eyy, gxy = strain

        csv_hist.write_row(accepted_step, solver.H_nodal.max().item(),
                          psi.max().item(), max_d, 0.0, 0.0,
                          reaction_force=R, applied_disp=disp_val)

        if args.vtu and accepted_step % args.vtu_every == 0:
            write_vtu(os.path.join(args.output_dir, f'step_{accepted_step:04d}.vtu'),
                      mesh,
                      point_data={'displacement': solver.u, 'damage': solver.d,
                                  'H': solver.H_nodal},
                      cell_data={'psi_plus': psi, 'H_elem': solver.H_elem})

        stress = None
        if trajectory_writers or (gif_recorder and accepted_step % gif_every == 0):
            stress = fem.compute_stress(
                solver.u, solver.d, strain=(exx, eyy, gxy))

        if trajectory_writers:
            sxx, syy, sxy = stress
            for writer in trajectory_writers:
                writer.write(accepted_step, mesh, solver.u, solver.d,
                             psi, solver.H_elem,
                             eps_xx=exx, eps_yy=eyy, gam_xy=gxy,
                             sxx=sxx, syy=syy, sxy=sxy,
                             H_nodal=solver.H_nodal, reaction_force=R,
                             applied_disp=disp_val)

        if gif_recorder and accepted_step % gif_every == 0:
            sxx, syy, sxy = stress
            gif_recorder.add_frame(accepted_step, solver.d, sxx, syy, sxy,
                                  exx, eyy, gxy, H=solver.H_nodal)

        for target in crack_stage_targets:
            if target not in crack_stage_snapshots and disp_val >= target:
                crack_stage_snapshots[target] = {
                    'step': accepted_step,
                    'disp': disp_val,
                    'damage': solver.d.detach().clone(),
                }

        if max_d > 0.99 and crack_step is None:
            crack_step = accepted_step
            print(f"  ** Crack fully developed at step {accepted_step} **")

        if (args.stop_at_crack and crack_step is not None
                and accepted_step >= crack_step + args.stop_after_crack_steps):
            print(f"  [Stopping early (--stop_at_crack, "
                  f"{args.stop_after_crack_steps} post-crack steps)]")
            break

        last_disp = disp_val
        consecutive_cutbacks = 0
        accepted_step += 1

        if continuation.enabled:
            nxt = continuation.next_after_accept(
                accepted_step=accepted_step,
                disp=disp_val,
                reaction=R,
                max_d=max_d)
            pending_displacements.clear()
            if nxt is not None:
                pending_displacements.append(nxt)

        if args.checkpoint_every and accepted_step % args.checkpoint_every == 0:
            save_run_checkpoint(
                checkpoint_path, solver=solver, bcs=bcs,
                pending_displacements=pending_displacements,
                history=history, energy_rows=energy_rows,
                accepted_step=accepted_step, last_disp=last_disp,
                cutback_count=cutback_count,
                consecutive_cutbacks=consecutive_cutbacks,
                crack_step=crack_step,
                crack_stage_snapshots=crack_stage_snapshots,
                control_rows=control_rows,
                continuation=continuation)

    t_total = time.time() - t_total_start
    print(f"\nTotal time: {t_total:.1f}s ({len(history)} steps)")

    # ---- Close trajectory stores ----
    for writer in trajectory_writers:
        writer.close(len(history))
    csv_hist.close()

    # ---- GIF ----
    if gif_recorder:
        anim_ext = 'png' if args.animation_format == 'apng' \
            else args.animation_format
        gif_recorder.save_gif(os.path.join(args.output_dir,
                                          f'damage_evolution.{anim_ext}'),
                             fps=8)

    # ---- Final energies ----
    try:
        _eng = fem.compute_energies(solver.u, solver.d, strain=strain)
        sim_energies = {'E_fracture': _eng.get('fracture_total', float('nan'))}
    except Exception:
        sim_energies = {'E_fracture': float('nan')}

    # ---- Results CSV ----
    csv_path = os.path.join(args.output_dir, 'results.csv')
    with open(csv_path, 'w') as f:
        f.write("step,displacement,reaction_kN,max_d,max_H,"
                "stagger_iter,elapsed_ms,continuation_mode,"
                "arc_length_residual,arc_length_constraint,load_factor,"
                "line_search_alpha,line_search_reductions\n")
        for r in history:
            f.write(f"{r['step']},{r['disp']:.8f},{r['reaction_kN']:.6f},"
                    f"{r['max_d']:.8f},{r['max_H']:.4f},"
                    f"{r['stagger_iter']},{r['elapsed_ms']:.1f},"
                    f"{r.get('continuation_mode', '')},"
                    f"{float(r.get('arc_length_residual', float('nan'))):.9e},"
                    f"{float(r.get('arc_length_constraint', float('nan'))):.9e},"
                    f"{float(r.get('load_factor', r['disp'])):.9e},"
                    f"{float(r.get('line_search_alpha', 1.0)):.9e},"
                    f"{int(r.get('line_search_reductions', 0))}\n")
    telemetry_path = os.path.join(args.output_dir, 'solver_telemetry.csv')
    write_solver_telemetry_csv(telemetry_path, history)
    print(f"Solver telemetry CSV: {telemetry_path}")
    timing_path = os.path.join(args.output_dir, 'timing_per_step.csv')
    with open(timing_path, 'w') as f:
        f.write("step,max_d,max_H,Solid solve Time,Compute Strain Time,"
                "Driving Force Time,Phase Solve Time,Total Step Time\n")
        for r in history:
            f.write(f"{r['step']},{r['max_d']:.8f},{r['max_H']:.9e},"
                    f"0.000000000,0.000000000,0.000000000,0.000000000,"
                    f"{r['elapsed_ms'] / 1000.0:.9f}\n")
    print(f"Timing CSV: {timing_path}")
    energy_path = os.path.join(args.output_dir, 'energy.csv')
    write_energy_csv(energy_path, energy_rows)
    print(f"Energy CSV: {energy_path}")
    control_path = os.path.join(args.output_dir, 'load_step_control.csv')
    write_load_step_control_csv(control_path, control_rows)
    print(f"Load-step control CSV: {control_path}")

    # ---- Load-displacement plot ----
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ref_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                               'reference_solutions')
        ref_file = os.path.join(ref_dir, 'miehe_sent_load_displacement.csv')
        if os.path.exists(ref_file):
            ref = np.loadtxt(ref_file)
            ax.plot(ref[:, 0], ref[:, 1], 'g-', lw=2.5, label='Miehe (reference)')
        ax.plot([r['disp'] for r in history],
                [r['reaction_kN'] for r in history],
                'b.-', lw=1.5, ms=3, label='phast')
        ax.set_xlabel('Displacement [mm]')
        ax.set_ylabel('Reaction Force [kN]')
        ax.set_title(f'SENT Validation — l0={l0}, h={args.h_crack}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        E_theory = mat.Gc * PROBLEM['a']
        ax.text(0.98, 0.55,
                f"Gc = {mat.Gc} N/mm\n"
                f"E_theory = {E_theory:.4f} N·mm\n"
                f"E_frac = {sim_energies['E_fracture']:.4f} N·mm",
                transform=ax.transAxes, fontsize=8, va='center', ha='right',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        fig.savefig(os.path.join(args.output_dir, 'load_displacement.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Saved: load_displacement.png")
        plot_energy_history(
            energy_rows, os.path.join(args.output_dir, 'energy.png'),
            xlabel='Applied displacement [mm]')
        print("Saved: energy.png")

        plot_quasistatic_convergence(
            history, os.path.join(args.output_dir, 'staggered_convergence.png'))
        print("Saved: staggered_convergence.png")

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
        plot_field(mesh, solver.d,
                   title=f'Final Damage (step {len(history)})',
                   cmap='hot', vmin=0, vmax=1, ax=ax2)
        ax2.plot([0.0, PROBLEM['a']],
                 [0.5 * PROBLEM['L'], 0.5 * PROBLEM['L']],
                 color='white', lw=1.6, zorder=5,
                 solid_capstyle='butt')
        fig2.savefig(os.path.join(args.output_dir, 'damage_final.png'),
                     dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print("Saved: damage_final.png")

        if crack_stage_snapshots:
            ncols = len(crack_stage_snapshots)
            fig3, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.0),
                                      squeeze=False)
            for ax, target in zip(axes[0], sorted(crack_stage_snapshots)):
                snap = crack_stage_snapshots[target]
                plot_field(
                    mesh, snap['damage'],
                    title=(f"u={snap['disp']:.4f} mm\n"
                           f"step {snap['step']}, target {target:.4f}"),
                    cmap='hot', vmin=0, vmax=1, ax=ax,
                    colorbar=(ax is axes[0, -1]))
            fig3.suptitle('SENT crack-path propagation stages', fontsize=12)
            fig3.tight_layout()
            fig3.savefig(os.path.join(args.output_dir, 'crack_path_stages.png'),
                         dpi=180, bbox_inches='tight')
            plt.close(fig3)
            print("Saved: crack_path_stages.png")

    # ---- Profiler ----
    if args.profile and ctx.profiler._timings:
        from phast.io_utils import write_profiler_csv
        prof_path = os.path.join(args.output_dir, 'profiler.csv')
        write_profiler_csv(prof_path, ctx.profiler)
        print(ctx.profiler.summary())

    # ---- Metadata ----
    save_run_metadata(
        args.output_dir,
        problem_name=PROBLEM['name'],
        device=ctx.device,
        material=mat,
        mesh=mesh,
        solver_config={
            'solver_type': args.solver_type,
            'num_steps': args.num_steps,
            'stagger_tol': args.stagger_tol,
            'stagger_criterion': args.stagger_criterion,
            'anderson_depth': args.anderson_depth,
            'newton_line_search': args.newton_line_search,
            'continuation': 'displacement_path_following'
            if continuation.enabled else 'prescribed_displacement',
        },
        extra={
            'total_time_s': round(t_total, 2),
            'peak_reaction_kN': round(max(r['reaction_kN'] for r in history), 4)
            if history else 0.0,
        },
    )

    print(f"\nAll outputs in: {args.output_dir}/")


if __name__ == '__main__':
    main()
