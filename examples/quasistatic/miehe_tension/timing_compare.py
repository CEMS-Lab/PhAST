#!/usr/bin/env python
"""Timing harness sweeping mesh sizes to validate #105/#106 SciPy SuperLU vs CG.

Sweeps a series of SENT meshes from ~1k to ~30k DOFs and times the staggered
loop with backend='cg' (matrix-free PCG, baseline) vs backend='scipy'
(SciPy SuperLU sparse-direct, the SparseSolveAutograd path from #106). For
each tier the wall-time, max displacement, and stagger iterations are
recorded; results are dumped to CSV and a log-log plot.

Hard caps:
  - per-cell wall budget: --max_cell_seconds (default 60s); larger meshes
    are skipped if either backend exceeds the cap.
  - 4-6 pre-peak load steps (--num_steps).
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from phast.mesh_generator import miehe_tension as gen_mesh
from phast.mesh import FEMMesh
from phast.material import create_material
from phast.boundary_conditions import BoundaryConditions
from phast.staggered_solver import StaggeredSolver, SolverConfig
from phast.device import DeviceContext


L, A_NOTCH = 1.0, 0.5
L0 = 0.02
ETA_RES = 0.0

# (h_crack, h_coarse) ladder calibrated on the SENT mesher to hit the DOF
# tiers below; values measured empirically.
DOF_LADDER = [
    (1000,  0.0100, 0.080),
    (4000,  0.0040, 0.038),
    (9000,  0.0027, 0.025),
    (16000, 0.0020, 0.018),
    (30000, 0.0014, 0.0125),
]


def _params_for_target(target_dofs):
    # Pick the ladder rung whose DOF target is closest to the requested value.
    best = min(DOF_LADDER, key=lambda r: abs(r[0] - target_dofs))
    return best[1], best[2]


def setup_bcs(mesh):
    bcs = BoundaryConditions(mesh.n_nodes, mesh.device, mesh.dtype)
    bcs.fix(mesh.node_sets['bottom'], 0)
    bcs.fix(mesh.node_sets['bottom'], 1)
    bcs.add(mesh.node_sets['top'], 1, 1.0)
    return bcs


def run_once(backend, num_steps, dt, h_crack, h_coarse, work_dir, tag):
    ctx = DeviceContext(device='cpu', compile_solvers=False)
    mesh_path = os.path.join(work_dir, f'mesh_{tag}.msh')
    gen_mesh(mesh_path, L=L, a=A_NOTCH, l0=L0,
             h_crack=h_crack, h_coarse=h_coarse, verbose=False)
    mesh = FEMMesh(mesh_path, device=ctx.device, dtype=ctx.dtype)
    mesh.identify_boundaries()
    n_dof = 2 * mesh.n_nodes

    mat = create_material('miehe_tension', l0=L0, eta_residual=ETA_RES)
    bcs = setup_bcs(mesh)
    cfg = SolverConfig(
        solver_type='quasi_static',
        num_steps=num_steps,
        damage_tol=1e-6,
        static_tol=1e-8,
        stagger_tol=1e-6,
        max_stagger=200,
        stagger_criterion='relative',
        use_multigrid=False,
        backend=backend,
        H_cap=None,
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg, ctx=ctx)

    disps = [dt * (i + 1) for i in range(num_steps)]
    fem = solver.fem
    diverged = False

    t0 = time.perf_counter()
    for i, u_val in enumerate(disps):
        bcs.load_factor = u_val
        solver.step_full()
        if not torch.isfinite(solver.u).all():
            diverged = True
            break
    wall = time.perf_counter() - t0

    return {
        'tag': tag,
        'backend': backend,
        'wall': wall,
        'n_dof': n_dof,
        'max_u': solver.u.abs().max().item(),
        'max_d': solver.d.max().item(),
        'diverged': diverged,
    }


def _run_tier(target, num_steps, dt, work_dir, max_cell_s):
    h_crack, h_coarse = _params_for_target(target)
    print(f"  --- target ~{target} DOFs (h_crack={h_crack}, h_coarse={h_coarse}) ---")
    res_cg = run_once('cg', num_steps, dt, h_crack, h_coarse, work_dir,
                      f'cg_{target}')
    n_dof = res_cg['n_dof']
    print(f"    [cg]    n_dof={n_dof}  wall={res_cg['wall']:.3f}s  "
          f"max|u|={res_cg['max_u']:.4e}  max d={res_cg['max_d']:.3e}")
    if res_cg['wall'] > max_cell_s or res_cg['diverged']:
        print(f"    cg exceeded cap or diverged; aborting larger tiers")
        return n_dof, res_cg, None, True

    res_sp = run_once('scipy', num_steps, dt, h_crack, h_coarse, work_dir,
                      f'sp_{target}')
    print(f"    [scipy] n_dof={n_dof}  wall={res_sp['wall']:.3f}s  "
          f"max|u|={res_sp['max_u']:.4e}  max d={res_sp['max_d']:.3e}")
    stop = res_sp['wall'] > max_cell_s or res_sp['diverged']
    return n_dof, res_cg, res_sp, stop


def _make_plot(rows, out_path):
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })
    import matplotlib.pyplot as plt

    dofs = np.array([r['dofs'] for r in rows], dtype=float)
    w_cg = np.array([r['wall_cg_s'] for r in rows], dtype=float)
    w_sp = np.array([r['wall_scipy_s'] for r in rows], dtype=float)
    sp = w_cg / np.maximum(w_sp, 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True,
                             constrained_layout=True)
    ax1, ax2 = axes
    ax1.loglog(dofs, w_cg, 'o-', color='C0', label='CG (matrix-free)')
    ax1.loglog(dofs, w_sp, 's-', color='C3', label='SciPy SuperLU')
    ax1.set_ylabel('Wall time (s)')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(frameon=False)
    ax1.set_title(r'Miehe SENT: CG vs SparseSolveAutograd')

    ax2.semilogx(dofs, sp, 'd-', color='C2')
    ax2.axhline(1.0, color='k', lw=0.5, ls='--', alpha=0.6)
    ax2.set_xlabel('Degrees of freedom')
    ax2.set_ylabel(r'Speedup (CG / SciPy)')
    ax2.grid(True, which='both', alpha=0.3)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=4)
    p.add_argument('--dt', type=float, default=1e-4)
    p.add_argument('--mesh_sizes', type=str,
                   default='1000,4000,9000,16000,30000',
                   help='Comma-separated DOF targets')
    p.add_argument('--max_cell_seconds', type=float, default=60.0)
    args = p.parse_args()

    targets = [int(x) for x in args.mesh_sizes.split(',') if x.strip()]
    work_dir = os.path.join(os.path.dirname(__file__), '_timing_compare_out')
    os.makedirs(work_dir, exist_ok=True)
    here = os.path.dirname(__file__)

    print("=" * 72)
    print("  Miehe SENT timing sweep — #105 SparseSolveAutograd vs CG")
    print("=" * 72)
    print(f"  num_steps={args.num_steps}  dt={args.dt}  "
          f"per-cell cap={args.max_cell_seconds:.0f}s")
    print(f"  targets   = {targets}")
    print()

    rows = []
    skipped = []
    for tgt in targets:
        try:
            n_dof, res_cg, res_sp, stop = _run_tier(
                tgt, args.num_steps, args.dt, work_dir,
                args.max_cell_seconds)
        except Exception as exc:  # noqa: BLE001
            print(f"  ** tier {tgt} failed: {exc} ** — skipping rest")
            skipped.append(tgt)
            break
        if res_sp is None:
            skipped.append(tgt)
            break
        du = abs(res_cg['max_u'] - res_sp['max_u'])
        speedup = res_cg['wall'] / res_sp['wall'] if res_sp['wall'] > 0 \
            else float('nan')
        rows.append({
            'dofs': n_dof,
            'wall_cg_s': res_cg['wall'],
            'wall_scipy_s': res_sp['wall'],
            'speedup': speedup,
            'abs_disp_err': du,
        })
        if stop:
            print(f"    cap hit at {n_dof} DOFs; not climbing higher")
            break

    print()
    print("=" * 72)
    print("  Sweep summary")
    print("=" * 72)
    print(f"  {'dofs':>8} {'wall_cg_s':>11} {'wall_scipy_s':>13} "
          f"{'speedup':>9} {'abs_disp_err':>13}")
    for r in rows:
        print(f"  {r['dofs']:>8d} {r['wall_cg_s']:>11.3f} "
              f"{r['wall_scipy_s']:>13.3f} {r['speedup']:>9.2f} "
              f"{r['abs_disp_err']:>13.2e}")
    if skipped:
        print(f"  skipped tiers: {skipped}")

    csv_path = os.path.join(here, 'timing_speedup.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows
                           else ['dofs', 'wall_cg_s', 'wall_scipy_s',
                                 'speedup', 'abs_disp_err'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  CSV  : {csv_path}")

    if rows:
        png_path = os.path.join(here, 'timing_speedup.png')
        _make_plot(rows, png_path)
        print(f"  Plot : {png_path}")


if __name__ == '__main__':
    main()
