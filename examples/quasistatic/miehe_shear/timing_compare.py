#!/usr/bin/env python
"""Timing harness sweeping mesh sizes to validate #105/#106 on Miehe SENS.

Companion of the SENT timing_compare.py (#105). Sweeps Miehe single-edge-
notch SHEAR meshes from ~1k to ~16k DOFs and times the staggered loop with
backend='cg' (matrix-free PCG, baseline) vs backend='scipy' (SciPy SuperLU
sparse-direct, the SparseSolveAutograd path from #106).

SENS is finickier than SENT pre-peak, so:
  - DOF ladder spans 1k–30k (mirrors SENT)
  - per-cell wall budget defaults to 120s (extended from 30s for 14k+ tier)
  - whole-script wall budget should be enforced by the caller (e.g. 15 min)
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import torch

# cloud-synced path workaround: /tmp/phast symlinks to this worktree
# so that "from phast.X import ..." resolves regardless of cwd.
sys.path.insert(0, '/tmp')

from phast.mesh_generator import miehe_shear as gen_mesh
from phast.mesh import FEMMesh
from phast.material import create_material
from phast.boundary_conditions import BoundaryConditions
from phast.staggered_solver import StaggeredSolver, SolverConfig
from phast.device import DeviceContext


L, A_NOTCH = 1.0, 0.5
L0 = 0.06              # PhaseFieldX 1712 default — larger for shear
ETA_RES = 0.0

# (target_dofs, h_crack, h_coarse) — SENS uses larger l0 so the crack-band
# h scales accordingly. Empirical ladder hitting target DOFs roughly.
DOF_LADDER = [
    (1000,  0.030, 0.080),
    (4000,  0.012, 0.040),
    (9000,  0.0080, 0.028),
    (16000, 0.0060, 0.020),
    (30000, 0.0044, 0.015),
]


def _params_for_target(target_dofs):
    best = min(DOF_LADDER, key=lambda r: abs(r[0] - target_dofs))
    return best[1], best[2]


def setup_bcs(mesh):
    """SENS BCs (PhaseFieldX 1712)."""
    bcs = BoundaryConditions(mesh.n_nodes, mesh.device, mesh.dtype)
    bcs.fix(mesh.node_sets['bottom'], 0)
    bcs.fix(mesh.node_sets['bottom'], 1)
    bcs.add(mesh.node_sets['top'], 0, 1.0)
    bcs.fix(mesh.node_sets['top'], 1)
    if 'left' in mesh.node_sets:
        bcs.fix(mesh.node_sets['left'], 1)
    if 'right' in mesh.node_sets:
        bcs.fix(mesh.node_sets['right'], 1)
    return bcs


def run_once(backend, num_steps, dt, h_crack, h_coarse, work_dir, tag):
    ctx = DeviceContext(device='cpu', compile_solvers=False)
    mesh_path = os.path.join(work_dir, f'mesh_{tag}.msh')
    gen_mesh(mesh_path, L=L, a=A_NOTCH, l0=L0,
             h_crack=h_crack, h_coarse=h_coarse, verbose=False)
    mesh = FEMMesh(mesh_path, device=ctx.device, dtype=ctx.dtype)
    mesh.identify_boundaries()
    n_dof = 2 * mesh.n_nodes

    mat = create_material('miehe_shear', l0=L0, eta_residual=ETA_RES)
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
        H_cap=10.0 * mat.Gc / (2.0 * L0),
    )
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg, ctx=ctx)

    disps = [dt * (i + 1) for i in range(num_steps)]
    diverged = False

    t0 = time.perf_counter()
    for u_val in disps:
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
    print(f"  --- target ~{target} DOFs (h_crack={h_crack}, h_coarse={h_coarse}) ---", flush=True)
    res_cg = run_once('cg', num_steps, dt, h_crack, h_coarse, work_dir,
                      f'cg_{target}')
    n_dof = res_cg['n_dof']
    print(f"    [cg]    n_dof={n_dof}  wall={res_cg['wall']:.3f}s  "
          f"max|u|={res_cg['max_u']:.4e}  max d={res_cg['max_d']:.3e}", flush=True)
    if res_cg['wall'] > max_cell_s or res_cg['diverged']:
        print(f"    cg exceeded cap ({max_cell_s}s) or diverged; aborting larger tiers", flush=True)
        return n_dof, res_cg, None, True

    res_sp = run_once('scipy', num_steps, dt, h_crack, h_coarse, work_dir,
                      f'sp_{target}')
    print(f"    [scipy] n_dof={n_dof}  wall={res_sp['wall']:.3f}s  "
          f"max|u|={res_sp['max_u']:.4e}  max d={res_sp['max_d']:.3e}", flush=True)
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
    ax1.set_title(r'Miehe SENS: CG vs SparseSolveAutograd')

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
    p.add_argument('--max_cell_seconds', type=float, default=120.0)
    args = p.parse_args()

    targets = [int(x) for x in args.mesh_sizes.split(',') if x.strip()]
    work_dir = os.path.join(os.path.dirname(__file__), '_timing_compare_out')
    os.makedirs(work_dir, exist_ok=True)
    here = os.path.dirname(__file__)

    print("=" * 72, flush=True)
    print("  Miehe SENS timing sweep — #105 SparseSolveAutograd vs CG", flush=True)
    print("=" * 72, flush=True)
    print(f"  num_steps={args.num_steps}  dt={args.dt}  "
          f"per-cell cap={args.max_cell_seconds:.0f}s", flush=True)
    print(f"  targets   = {targets}", flush=True)
    print(flush=True)

    rows = []
    skipped = []
    for tgt in targets:
        try:
            n_dof, res_cg, res_sp, stop = _run_tier(
                tgt, args.num_steps, args.dt, work_dir,
                args.max_cell_seconds)
        except Exception as exc:  # noqa: BLE001
            print(f"  ** tier {tgt} failed: {exc} ** — skipping rest", flush=True)
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
            print(f"    cap hit at {n_dof} DOFs; not climbing higher", flush=True)
            break

    print(flush=True)
    print("=" * 72, flush=True)
    print("  Sweep summary", flush=True)
    print("=" * 72, flush=True)
    print(f"  {'dofs':>8} {'wall_cg_s':>11} {'wall_scipy_s':>13} "
          f"{'speedup':>9} {'abs_disp_err':>13}", flush=True)
    for r in rows:
        print(f"  {r['dofs']:>8d} {r['wall_cg_s']:>11.3f} "
              f"{r['wall_scipy_s']:>13.3f} {r['speedup']:>9.2f} "
              f"{r['abs_disp_err']:>13.2e}", flush=True)
    if skipped:
        print(f"  skipped tiers: {skipped}", flush=True)

    csv_path = os.path.join(here, 'timing_speedup.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows
                           else ['dofs', 'wall_cg_s', 'wall_scipy_s',
                                 'speedup', 'abs_disp_err'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  CSV  : {csv_path}", flush=True)

    if len(rows) >= 2:
        png_path = os.path.join(here, 'timing_speedup.png')
        _make_plot(rows, png_path)
        print(f"  Plot : {png_path}", flush=True)


if __name__ == '__main__':
    main()
