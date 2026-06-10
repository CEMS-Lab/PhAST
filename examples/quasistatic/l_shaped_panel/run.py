#!/usr/bin/env python
"""
L-Shaped Panel — Phase-field fracture benchmark.

Uses StaggeredSolver for stagger iterations (no custom stagger loop).
All stagger logic, Anderson acceleration, convergence criteria, and
H capping are handled by the solver.

An L-shaped specimen (500x500 mm with 250x250 mm cutout) loaded vertically
at a point on the cutout horizontal edge, 30 mm from the re-entrant corner. Crack
initiates at the re-entrant corner and propagates along the arm, matching
the Ambati/Winkler L-panel setup.

References:
  - Winkler (2001): Experimental crack path in concrete
  - Ambati, Gerasimov, De Lorenzis (2015): Phase-field modeling
  - Rudshaug, Hopperstad, Borvik (2024): LS-DYNA benchmark (glass)

Material presets:
  - 'l_shaped_glass': E=70 GPa, nu=0.23, Gc=8 J/m^2, l0=0.4 mm (Rudshaug 2024)
  - 'l_shaped_concrete': E=25.85 GPa, nu=0.18, Gc=89 N/m, l0=2.5 mm (Winkler 2001)

IMPORTANT: Spectral (Miehe) split is REQUIRED for correct crack path.
           Amor (vol-dev) split produces WRONG crack direction at the corner.

Usage:
    cd /path/to/PhAST
    python -u examples/quasistatic/l_shaped_panel/run.py --plots
    python -u examples/quasistatic/l_shaped_panel/run.py --all_outputs
    python -u examples/quasistatic/l_shaped_panel/run.py --material concrete --plots
"""

import sys
import os
import time
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from phast.mesh_generator import l_shaped_panel as gen_mesh
from phast.mesh import FEMMesh
from phast.material import create_material
from phast.boundary_conditions import BoundaryConditions
from phast.staggered_solver import StaggeredSolver, SolverConfig
from phast.device import DeviceContext
from phast.io_utils import (
    write_vtu,
    generate_run_tag, save_run_metadata, write_solver_telemetry_csv,
    write_energy_csv, plot_energy_history,
)
from phast.visualization import (
    plot_field, plot_initial_conditions, GIFRecorder,
    plot_quasistatic_convergence,
)
from examples.quasistatic._run_utils import open_trajectory_writers
# theory_report module not present; use fem.compute_energies inline below.

# ------------------------------------------------------------------ #
# Problem parameters
# ------------------------------------------------------------------ #
PROBLEM = {
    'name': 'L-Shaped Panel (Phase-Field Fracture)',
    'reference': 'Winkler (2001), Ambati (2015), Rudshaug (2024)',
    'L': 250.0,          # half-side length [mm] (total 500x500)
    'energy_split': 'spectral',  # REQUIRED — amor fails for L-shaped
    'pf_model': 'AT2',
    # Loading: slow linear ramp
    'dt': 1e-3,          # [mm/step] displacement increment
}

# Material configurations
MATERIALS = {
    'glass': {
        'preset': 'l_shaped_glass',
        'description': 'Soda-lime glass (Rudshaug et al. 2024)',
        'l0': 0.4,
        'h_crack': 0.1,       # l0/4 for proper AT2 resolution
        'h_coarse': 25.0,
    },
    'concrete': {
        'preset': 'l_shaped_concrete',
        'description': 'Concrete (Winkler 2001, Ambati 2015)',
        'l0': 1.1875,
        'h_crack': 0.3,       # l0/4 for proper AT2 resolution
        'h_coarse': 25.0,
    },
}


def setup_bcs(mesh):
    """L-shaped panel BCs matching Ambati (2015) Fig. 16a exactly.

    - Bottom (y=0, arm base): fully clamped (u_i = 0)
    - Load point on the cutout horizontal edge, 30 mm right of the
      re-entrant corner: u_2 = u_2^app (downward). The node set is
      still named load_segment for compatibility with old configs.
    """
    bcs = BoundaryConditions(mesh.n_nodes, mesh.device, mesh.dtype)
    # Bottom of arm: fully clamped
    bcs.fix(mesh.node_sets['bottom'], 0)
    bcs.fix(mesh.node_sets['bottom'], 1)
    # Ambati Fig. 16a point displacement: u_y = -1 (downward), scaled by load_factor.
    bcs.add(mesh.node_sets['load_segment'], 1, -1.0)
    return bcs


def build_loading_schedule(n_steps, du):
    """Cyclic loading schedule matching Ambati (2015) Fig. 17.

    Fixed displacement increment |Δu| = du per step. The protocol is:
      - Phase 1: ramp up    0    -> +0.3 mm  (300 steps)
      - Phase 2: ramp down +0.3  -> -0.2 mm  (500 steps)
      - Phase 3: ramp up   -0.2  -> +U mm    (remaining steps until failure)

    Total ~1400 steps with du=1e-3 mm reproduces Ambati's Fig. 17.
    """
    schedule = []
    u = 0.0
    phase = 1
    for i in range(n_steps):
        if phase == 1:
            u += du
            if u >= 0.3:
                u = 0.3
                phase = 2
        elif phase == 2:
            u -= du
            if u <= -0.2:
                u = -0.2
                phase = 3
        else:
            u += du
        schedule.append(u)
    return schedule


def main():
    parser = argparse.ArgumentParser(description=PROBLEM['name'])
    # Material
    parser.add_argument('--material', type=str, default='glass',
                        choices=['glass', 'concrete'],
                        help='Material preset (default: glass)')
    # Mesh overrides
    parser.add_argument('--h_crack', type=float, default=None)
    parser.add_argument('--h_coarse', type=float, default=None)
    parser.add_argument('--l0', type=float, default=None)
    parser.add_argument('--L', type=float, default=PROBLEM['L'],
                        help='Half-side length in mm (default: 250)')
    # Solver
    parser.add_argument('--num_steps', type=int, default=1400)
    parser.add_argument('--dt', type=float, default=PROBLEM['dt'])
    parser.add_argument('--stagger_tol', type=float, default=1e-6)
    parser.add_argument('--max_stagger', type=int, default=500)
    parser.add_argument('--static_max_iter', type=int, default=5000,
                        help='Maximum Newton iterations for the mechanics solve.')
    parser.add_argument('--damage_cg_tol', type=float, default=1e-6)
    parser.add_argument('--mechanics_cg_tol', type=float, default=1e-8)
    parser.add_argument('--eta_residual', type=float, default=1e-2)
    parser.add_argument('--stagger_criterion', type=str, default='relative',
                        choices=['absolute', 'relative', 'linf', 'residual',
                                 'am_energy'])
    parser.add_argument('--anderson_depth', type=int, default=0)
    parser.add_argument('--H_cap_factor', type=float, default=0.0,
                        help='Optional non-reference cap on H as a multiple of Gc/(2*l0).')
    parser.add_argument('--energy_split', type=str, default=None,
                        choices=['isotropic', 'amor', 'spectral', 'star_convex'])
    parser.add_argument('--at_mode', type=str, default='at2',
                        choices=['at1', 'at2'],
                        help='Phase-field model: AT1 or AT2 (default: at2)')
    parser.add_argument('--adaptive_stagger_tol', action='store_true')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'scipy', 'mumps', 'cg'],
                        help='Quasistatic mechanics linear solver backend.')
    parser.add_argument('--preconditioner', type=str, default='jacobi',
                        choices=['jacobi', 'spectral', 'gmg', 'amg', 'amgx', 'auto'])
    parser.add_argument('--multigrid', default=True,
                        action=argparse.BooleanOptionalAction)
    # Device
    parser.add_argument('--device', type=str, default=None)
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
    parser.add_argument('--gif_frames', type=int, default=200)
    parser.add_argument('--gif_fields', type=str,
                        default='damage,max_principal_stress,hydrostatic_stress')
    parser.add_argument('--animation_format', choices=['gif', 'apng', 'mp4'],
                        default='mp4')
    parser.add_argument('--stop_at_crack', action='store_true')
    parser.add_argument('--stop_after_crack_steps', type=int, default=1,
                        help=('When --stop_at_crack is set, keep this many '
                              'accepted increments after crack detection.'))
    parser.add_argument('--print_every', type=int, default=1)
    args = parser.parse_args()

    if args.all_outputs:
        args.h5 = args.vtu = args.gif = args.plots = args.profile = True

    if not any([args.h5, args.vtu, args.gif, args.plots, args.profile]):
        print("No output flags set. Use --all_outputs or --trajectory for training data.\n")

    active = [f for f, v in [('VTU', args.vtu), ('GIF', args.gif),
                              ('plots', args.plots), ('profiler', args.profile),
                              (args.trajectory_format.upper(), args.h5)] if v]
    print(f"  Outputs: {', '.join(active) if active else 'none'}")

    # Material config
    mat_cfg = MATERIALS[args.material]

    compile_flag = True if args.compile else None
    ctx = DeviceContext(device=args.device, profile=args.profile,
                        compile_solvers=compile_flag)

    if args.output_dir is None:
        base_dir = os.path.dirname(__file__)
        args.output_dir = os.path.join(
            base_dir,
            generate_run_tag(ctx.device,
                             extra=f'{args.material}_{args.at_mode}'))
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  {PROBLEM['name']}")
    print(f"  Material: {mat_cfg['description']}")
    print(f"  {PROBLEM['reference']}")
    print("=" * 60)

    # ---- Material ----
    l0 = args.l0 or mat_cfg['l0']
    energy_split = args.energy_split or PROBLEM['energy_split']
    if energy_split != 'spectral':
        print(f"  WARNING: Using {energy_split} split. "
              f"Spectral is required for correct L-shaped crack path!")
    mat = create_material(mat_cfg['preset'], l0=l0, energy_split=energy_split,
                          eta_residual=args.eta_residual,
                          pf_model=args.at_mode.upper())
    print(mat)

    # ---- H cap ----
    H_cap = None
    if args.H_cap_factor > 0:
        H_cap = args.H_cap_factor * mat.Gc / (2.0 * l0)
        print(f"H cap: {H_cap:.4f} ({args.H_cap_factor}x Gc/(2*l0))")

    # ---- Mesh ----
    h_crack = args.h_crack or mat_cfg['h_crack']
    h_coarse = args.h_coarse or mat_cfg['h_coarse']
    mesh_path = os.path.join(args.output_dir, 'mesh.msh')
    gen_mesh(mesh_path, L=args.L, l0=l0, h_crack=h_crack, h_coarse=h_coarse)
    mesh = FEMMesh(mesh_path, device=ctx.device, dtype=ctx.dtype)
    mesh.identify_boundaries()
    print(mesh)

    # ---- BCs (unit displacement, scaled by load_factor per step) ----
    bcs = setup_bcs(mesh)

    # ---- Solver (all stagger logic delegated here) ----
    cfg = SolverConfig(
        solver_type='quasi_static',
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
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg, ctx=ctx)
    fem = solver.fem

    # ---- Loading schedule (cyclic, Ambati 2015 Fig. 17) ----
    displacements = build_loading_schedule(args.num_steps, args.dt)
    reaction_nodes = mesh.node_sets['load_segment']

    # ---- Initial conditions plot ----
    if args.plots:
        bcs.load_factor = displacements[0]
        plot_initial_conditions(mesh, mat, bcs, cfg,
                                save_path=os.path.join(args.output_dir,
                                                       'initial_conditions.png'))
        print("Saved: initial_conditions.png")

    # ---- Trajectory store ----
    trajectory_writers = open_trajectory_writers(
        args.output_dir, mesh, mat, args.h5, fmt=args.trajectory_format)
    for writer in trajectory_writers:
        print(f"Trajectory initialized: {writer.path}")

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
    t_total_start = time.time()
    crack_step = None

    print(f"\nRunning {args.num_steps} load steps (cyclic: 0→+0.3→-0.2→+U), "
          f"Δu={args.dt} mm\n")

    for step_i, disp_val in enumerate(displacements):
        t0 = time.time()

        # Update load factor -- BCs scale automatically
        bcs.load_factor = disp_val

        # Full stagger step (all Anderson, convergence, H capping inside solver)
        psi = solver.step_full()

        elapsed = (time.time() - t0) * 1000
        stag_count = solver._last_stagger_iter

        # Reaction force (y-component at top of right arm)
        R = fem.compute_reaction_force(solver.u, solver.d, reaction_nodes,
                                        component=1)
        max_d = solver.d.max().item()

        # Divergence detection
        if not torch.isfinite(solver.u).all():
            print(f"  ** DIVERGENCE at step {step_i}. Stopping. **")
            break

        dt_used = abs(disp_val - history[-1]['disp']) if history else abs(disp_val)
        record = {
            'step': step_i, 'disp': disp_val,
            'reaction_N': R, 'reaction_kN': R / 1000.0,
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
        }
        history.append(record)
        energies = fem.compute_energy_components(
            solver.u, solver.d, getattr(solver, 'v', None), psi_plus=psi)
        energy_rows.append({
            'step': step_i, 'time': disp_val,
            'elastic': energies['elastic'],
            'fracture': energies['fracture'],
            'kinetic': energies['kinetic'],
            'external': 0.0,
            'total': energies['total'],
        })

        if step_i % args.print_every == 0:
            print(f"  Step {step_i:3d}: u_y={disp_val:.6f}, R={R/1000:.4f} kN, "
                  f"max(d)={max_d:.6f}, stag={stag_count} ({elapsed:.0f}ms)")

        # ---- Per-step outputs ----
        strain = solver._last_strain
        exx, eyy, gxy = strain

        csv_hist.write_row(step_i, solver.H_nodal.max().item(),
                           psi.max().item(), max_d, 0.0, 0.0,
                           reaction_force=R, applied_disp=disp_val)

        if args.vtu and step_i % args.vtu_every == 0:
            write_vtu(os.path.join(args.output_dir, f'step_{step_i:04d}.vtu'),
                      mesh,
                      point_data={'displacement': solver.u, 'damage': solver.d,
                                  'H': solver.H_nodal},
                      cell_data={'psi_plus': psi, 'H_elem': solver.H_elem})

        stress = None
        if trajectory_writers or (gif_recorder and step_i % gif_every == 0):
            stress = fem.compute_stress(
                solver.u, solver.d, strain=(exx, eyy, gxy))

        if trajectory_writers:
            sxx, syy, sxy = stress
            for writer in trajectory_writers:
                writer.write(step_i, mesh, solver.u, solver.d,
                             psi, solver.H_elem,
                             eps_xx=exx, eps_yy=eyy, gam_xy=gxy,
                             sxx=sxx, syy=syy, sxy=sxy,
                             H_nodal=solver.H_nodal, reaction_force=R,
                             applied_disp=disp_val)

        if gif_recorder and step_i % gif_every == 0:
            sxx, syy, sxy = stress
            gif_recorder.add_frame(step_i, solver.d, sxx, syy, sxy,
                                    exx, eyy, gxy, H=solver.H_nodal)

        if max_d > 0.99 and crack_step is None:
            crack_step = step_i
            print(f"  ** Crack fully developed at step {step_i} **")

        if args.stop_at_crack and crack_step is not None:
            stop_step = crack_step + max(args.stop_after_crack_steps, 0)
        else:
            stop_step = None
        if stop_step is not None and step_i >= stop_step:
            print(f"  [Stopping early (--stop_at_crack, "
                  f"post_crack_steps={args.stop_after_crack_steps})]")
            break

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
        sim_energies = {
            'E_fracture': _eng.get('fracture_total', float('nan')),
            'E_elastic': _eng.get('elastic_total', float('nan')),
            'E_total': _eng.get('total', float('nan')),
        }
    except Exception:
        sim_energies = {'E_fracture': float('nan'),
                        'E_elastic': float('nan'),
                        'E_total': float('nan')}

    # ---- Results CSV ----
    csv_path = os.path.join(args.output_dir, 'results.csv')
    with open(csv_path, 'w') as f:
        f.write("step,displacement,reaction_kN,max_d,max_H,"
                "stagger_iter,elapsed_ms\n")
        for r in history:
            f.write(f"{r['step']},{r['disp']:.8f},{r['reaction_kN']:.6f},"
                    f"{r['max_d']:.8f},{r['max_H']:.6f},"
                    f"{r['stagger_iter']},{r['elapsed_ms']:.1f}\n")
        f.write(f"\n# Final Energies [N*mm]\n")
        f.write(f"# E_elastic,{sim_energies['E_elastic']:.6f}\n")
        f.write(f"# E_fracture,{sim_energies['E_fracture']:.6f}\n")
        f.write(f"# E_total,{sim_energies['E_total']:.6f}\n")
    print(f"Results CSV: {csv_path}")
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

    # ---- Load-displacement plot ----
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot([r['disp'] for r in history],
                [r['reaction_kN'] for r in history],
                'b.-', lw=1.5, ms=3, label='phast')
        ax.set_xlabel('Displacement [mm]')
        ax.set_ylabel('Reaction Force [kN]')
        ax.set_title(f'L-Shaped Panel ({args.material}) — l0={l0}, h={h_crack}')
        ax.legend()
        ax.grid(True, alpha=0.3)
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

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        plot_field(mesh, solver.d,
                   title=f'Final Damage (step {len(history)})',
                   cmap='hot', vmin=0, vmax=1, ax=ax2)
        fig2.savefig(os.path.join(args.output_dir, 'damage_final.png'),
                     dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print("Saved: damage_final.png")

    # ---- Theory report (skipped: theory_report module not present) ----

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
            'solver_type': 'quasi_static',
            'num_steps': args.num_steps,
            'stagger_tol': args.stagger_tol,
            'stagger_criterion': args.stagger_criterion,
            'anderson_depth': args.anderson_depth,
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
