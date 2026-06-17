#!/usr/bin/env python
"""
Run a phase-field fracture simulation from a YAML config file.

Usage:
    python -m phast.run_config configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml
    python -m phast.run_config configs/benchmarks/dynamic/B5_pmma_branching.yaml --device cuda --fast
    python -m phast.run_config configs/benchmarks/dynamic/B3_dynamic_sent.yaml --h5 --plots

CLI flags override YAML values.
"""

import argparse
import os
import sys
import time
import math
import shutil
import torch
import yaml

from .config import load_config, save_config, resolve_config, compute_load_factor
from .config_validation import (
    validate_config_file_with_warnings,
    format_errors,
    format_warnings,
)
from ..utils.provenance import write_run_lockfile
from ..solvers.staggered_solver import StaggeredSolver
from ..utils.io_utils import (
    init_zarr, write_zarr_snapshot, init_h5, write_h5_snapshot, CSVHistory,
    generate_run_tag, save_run_metadata, write_solver_telemetry_csv,
    write_energy_csv, plot_energy_history,
)
from ..utils.visualization import plot_field, plot_initial_conditions


def _copy_mesh_provenance(mesh, output_dir: str) -> None:
    """Copy mesh provenance into the run directory when it came from disk."""
    mesh_path = getattr(mesh, 'mesh_path', None)
    if not mesh_path or mesh_path == '<from_tensors>':
        return
    if not os.path.exists(mesh_path):
        return
    dst = os.path.join(output_dir, 'mesh.msh')
    try:
        if os.path.abspath(mesh_path) != os.path.abspath(dst):
            shutil.copy2(mesh_path, dst)
            print(f"Saved: mesh.msh (from {mesh_path})")
        geo_path = os.path.splitext(mesh_path)[0] + '.geo'
        geo_dst = os.path.join(output_dir, 'mesh.geo')
        if os.path.exists(geo_path) and os.path.abspath(geo_path) != os.path.abspath(geo_dst):
            shutil.copy2(geo_path, geo_dst)
            print(f"Saved: mesh.geo (from {geo_path})")
    except OSError as exc:
        print(f"WARNING: could not copy mesh provenance for '{mesh_path}' "
              f"into '{output_dir}': {exc}", flush=True)


def _print_precheck_summary(mesh, mat, solver_cfg, cfg) -> None:
    """Compact pre-run diagnostic block.

    Prints the key physical + numerical invariants that
    ``python -m phast precheck <yaml>`` reports in full. Kept
    tight (10 lines) so it does not drown the run log; the full report is
    still one command away. Categories mirror the paper's §3 diagnostic
    tool discussion (wave speeds, CFL, phase-field resolution, damage
    subcycling bound).
    """
    h_min = float(mesh.elem_h.min())
    c_p = float(mat.c_p)
    c_s = float(mat.c_s)
    c_R = float(mat.c_R)
    dt_cfl = h_min / c_p
    dt = solver_cfg.dt_safety * dt_cfl
    h_over_l0 = h_min / mat.l0 if mat.l0 > 0 else float('inf')
    N_max = max(1, int(c_p / (0.6 * c_R)))
    # Match the step-count resolution rule used by the main loop: an
    # explicit YAML num_steps wins over t_total-derived counts.
    if cfg.loading.num_steps > 0:
        n_steps = cfg.loading.num_steps
        n_steps_src = 'YAML num_steps'
    elif cfg.loading.t_total > 0:
        n_steps = int(math.ceil(cfg.loading.t_total / dt))
        n_steps_src = f't_total={cfg.loading.t_total*1e6:.1f} us'
    else:
        n_steps = 0
        n_steps_src = 'unset'
    elems_in_band = int(2 * mat.l0 / h_min) if h_min > 0 else 0

    print("")
    print("-" * 66)
    print(f"  Pre-simulation diagnostic (set SKIP_PRECHECK=1 to hide)")
    print("-" * 66)
    print(f"  Wave speeds      : c_p={c_p*1e-3:.0f}  c_s={c_s*1e-3:.0f}  "
          f"c_R={c_R*1e-3:.0f}  m/s")
    print(f"  CFL time step    : dt_CFL={dt_cfl*1e9:.3f} ns  "
          f"(×{solver_cfg.dt_safety} safety = {dt*1e9:.3f} ns)")
    print(f"  Step count       : n_steps={n_steps:,}  (source: {n_steps_src})")
    print(f"  Resolution       : h_min={h_min:.4f} mm, "
          f"h/l0={h_over_l0:.2f} ({'GOOD' if h_over_l0 <= 0.5 else ('MARGINAL' if h_over_l0 <= 1.0 else 'TOO COARSE')}),"
          f" elems/(2l0)={elems_in_band}")
    print(f"  Subcycling bound : N_max = floor(c_p/(0.6*c_R)) = {N_max}"
          f"  (current damage_every={solver_cfg.damage_every})")
    if h_over_l0 > 1.0:
        print("  WARNING: mesh too coarse for phase-field. "
              "Expect under-resolved crack band.")
    if solver_cfg.damage_every > N_max:
        print(f"  WARNING: damage_every={solver_cfg.damage_every} exceeds N_max={N_max}; "
              "damage front may overshoot between solves.")
    if solver_cfg.dt_safety > 1.0:
        print("  WARNING: dt_safety>1.0 violates CFL. Expect instability.")
    zeta = getattr(solver_cfg, 'damping_ratio_max', 0.0)
    if zeta > 0.0:
        print(f"  Damping         : Kelvin-Voigt stiffness-prop, "
              f"zeta_max={zeta:.3f} at omega_max (CFL shrinks by "
              f"{(1 + zeta*zeta)**0.5 - zeta:.3f})")
    print("-" * 66)


def main():
    parser = argparse.ArgumentParser(
        description='Run phase-field fracture from YAML config')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    # CLI overrides
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--h5', action='store_true', default=None)
    parser.add_argument('--trajectory', action='store_true', default=None,
                        help='Enable trajectory snapshots (Zarr). Alias for the '
                             'output.trajectory / legacy --h5.')
    parser.add_argument('--trajectory-format',
                        choices=['zarr', 'h5', 'both'],
                        default=None,
                        help='Trajectory backend for snapshots. Default is '
                             'output.trajectory_format from YAML, normally '
                             'zarr. H5 is legacy compatibility.')
    parser.add_argument('--plots', action='store_true', default=None)
    parser.add_argument('--gif', action='store_true', default=None)
    parser.add_argument('--fast', action='store_true', default=None)
    parser.add_argument('--profile', action='store_true', default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--validation-id', type=str, default=None,
                        help='Select a curated validation entry from a '
                             'plasticity/interface reproducibility contract.')
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--print_every', type=int, default=None)
    parser.add_argument('--h5_every', type=int, default=None)
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate the YAML config and exit (no run). '
                             'Useful in CI / pre-commit hooks.')
    # Time-integrator opt-in (#573/#570). The canonical name for the
    # current production dynamic path is now 'central_difference' because
    # it is Velocity-Verlet / explicit Newmark-beta with beta=0, gamma=1/2.
    # 'verlet' and legacy 'newmark' remain accepted aliases for old scripts.
    # 'generalized_alpha' / 'gen_alpha' is reserved for the future COMSOL-style
    # implicit production path; until #570 lands, it validates parameters and
    # points users at the standalone demo.
    parser.add_argument('--time_integrator',
                        choices=['central_difference', 'verlet', 'newmark',
                                 'generalized_alpha', 'gen_alpha'],
                        default=None,
                        help='Dynamic time integrator. Default: '
                             'solver.time_integrator from YAML, normally '
                             'central_difference (current Velocity-Verlet / '
                             'explicit Newmark central-difference path). '
                             'Aliases: verlet, legacy newmark. '
                             'generalized_alpha/gen_alpha is reserved for the '
                             'future implicit COMSOL-style path and currently '
                             'routes to a parameter check/demo pointer; see '
                             '#570.')
    parser.add_argument('--rho_inf', type=float, default=None,
                        help='Spectral radius rho_inf in [0,1] for '
                             'generalized_alpha (1.0 = no dissipation, '
                             '0.5 = Borden 2012 / COMSOL-style damping).')
    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            raw_for_workflow = yaml.safe_load(fh) or {}
    except OSError:
        raw_for_workflow = {}
    if isinstance(raw_for_workflow, dict) and raw_for_workflow.get("schema_version", 1) in (2, "2"):
        from ..workflow import (
            execution_plan_from_spec,
            problem_spec_from_yaml,
            run_problem_spec,
            validate_problem_spec,
        )
        from ..workflow.execution import WorkflowExecutionError

        try:
            spec = problem_spec_from_yaml(args.config)
        except Exception as exc:
            print(f"error: cannot compile schema-v2 workflow contract: {exc}", file=sys.stderr)
            sys.exit(2)
        issues = validate_problem_spec(spec)
        if issues:
            for issue in issues:
                print(f"error: {issue.category}: {issue.message}", file=sys.stderr)
            sys.exit(2)
        if args.validate_only:
            print(f"OK: {args.config} passes schema-v2 workflow contract validation.")
            sys.exit(0)
        try:
            plan = execution_plan_from_spec(spec)
            if not plan.direct_execution_supported or plan.route not in {
                "solid_mechanics_runner",
                "run_config",
            }:
                print(
                    "error: schema_version 2 workflow decks are currently "
                    "executable only for promoted solid_mechanics examples "
                    "and supported quasi_static fracture specs; "
                    "use --validate-only or run an equivalent v1 compatibility YAML.",
                    file=sys.stderr,
                )
                sys.exit(2)
            if plan.route == "solid_mechanics_runner":
                print(
                    "schema-v2 solid mechanics execution adapter: "
                    f"{spec.solver.parameters.get('example')}"
                )
            else:
                print(f"schema-v2 fracture execution adapter: {spec.solver.kind}")
            sys.exit(run_problem_spec(spec, output_dir=args.output_dir))
        except WorkflowExecutionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)

    from ..plasticity_interface_runner import (
        is_plasticity_interface_contract,
        run_plasticity_interface_contract,
    )
    if is_plasticity_interface_contract(args.config):
        try:
            code = run_plasticity_interface_contract(
                args.config,
                validation_id=args.validation_id,
                output_dir=args.output_dir,
                validate_only=args.validate_only,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
        raise SystemExit(code)

    from ..solid_mechanics_runner import (
        is_solid_mechanics_config,
        run_solid_mechanics_config,
    )
    if is_solid_mechanics_config(args.config):
        raise SystemExit(run_solid_mechanics_config(
            args.config,
            output_dir=args.output_dir,
            validate_only=args.validate_only,
        ))

    # Schema validation (issue #150). Runs before the main loader so the
    # user sees a line-numbered error block instead of a deep KeyError.
    try:
        _, _errs, _warnings = validate_config_file_with_warnings(args.config)
    except OSError as exc:
        print(f"error: cannot read config {args.config!r}: {exc}",
              file=sys.stderr)
        sys.exit(2)
    if _errs:
        print(format_errors(_errs, args.config), file=sys.stderr)
        sys.exit(2)
    if _warnings:
        print(format_warnings(_warnings, args.config), file=sys.stderr)
    if args.validate_only:
        print(f"OK: {args.config} passes schema validation.")
        sys.exit(0)

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.time_integrator is not None:
        ti = args.time_integrator
        if ti in ('verlet', 'newmark'):
            ti = 'central_difference'
        elif ti == 'gen_alpha':
            ti = 'generalized_alpha'
        cfg.solver.time_integrator = ti
    if args.rho_inf is not None:
        cfg.solver.rho_inf = args.rho_inf
    if args.device is not None:
        cfg.device.device = args.device
    if args.h5 is not None:
        cfg.output.h5 = args.h5
    if args.trajectory is not None:
        cfg.output.trajectory = args.trajectory
        cfg.output.h5 = bool(args.trajectory)
    if args.trajectory_format is not None:
        cfg.output.trajectory_format = args.trajectory_format
    if args.plots is not None:
        cfg.output.plots = args.plots
    if args.gif is not None:
        cfg.output.gif = args.gif
    if args.fast is not None:
        cfg.output.fast = args.fast
    if args.profile is not None:
        cfg.output.profile = args.profile
    if args.num_steps is not None:
        cfg.loading.num_steps = args.num_steps
    if args.print_every is not None:
        cfg.output.print_every = args.print_every
    if args.h5_every is not None:
        cfg.output.h5_every = args.h5_every

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif cfg.output.output_dir:
        output_dir = cfg.output.output_dir
    else:
        device_arg = cfg.device.device or 'auto'
        try:
            out_device = torch.device(device_arg)
        except (TypeError, ValueError):
            out_device = torch.device('cpu')
        output_dir = generate_run_tag(out_device,
                                       extra=cfg.name.replace(' ', '_')[:20])
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  {cfg.name}")
    if cfg.reference:
        print(f"  {cfg.reference}")
    print(f"  Config: {args.config}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Save resolved config for reproducibility
    save_config(cfg, os.path.join(output_dir, 'config.yaml'))

    # Resolve config to solver objects
    objs = resolve_config(cfg)
    mesh = objs['mesh']
    mat = objs['material']
    bcs = objs['bcs']
    solver_cfg = objs['solver_config']
    ctx = objs['ctx']
    _copy_mesh_provenance(mesh, output_dir)
    write_run_lockfile(
        os.path.join(output_dir, 'run_lockfile.json'),
        config=cfg,
        config_path=args.config,
        output_dir=output_dir,
        args=args,
        mesh=mesh,
        material=mat,
        solver_config=solver_cfg,
        ctx=ctx,
    )
    print("Saved: run_lockfile.json")

    if solver_cfg.solver_type == 'explicit':
        print(f"Dynamic integrator: {solver_cfg.time_integrator}", flush=True)
        if solver_cfg.time_integrator in ('generalized_alpha', 'gen_alpha'):
            from ..solvers.time_integrators import gen_alpha_params
            am, af, beta, gamma = gen_alpha_params(solver_cfg.rho_inf)
            print(
                f"  rho_inf={solver_cfg.rho_inf} -> alpha_m={am:.4f} "
                f"alpha_f={af:.4f} beta={beta:.4f} gamma={gamma:.4f}",
                flush=True)

    print(mat)
    print(mesh)

    # --- Inline pre-simulation diagnostic (always on). A compact summary
    # of the key invariants the full `python -m phast precheck`
    # tool reports: wave speeds, CFL dt, mesh/ℓ₀ resolution, safe
    # subcycling bound. Catches an obvious mesh-too-coarse or material-
    # mis-specified setup before the explicit loop starts. Silence by
    # setting the env var SKIP_PRECHECK=1 (e.g. scripts that already run
    # precheck separately).
    if os.environ.get('SKIP_PRECHECK') != '1':
        _print_precheck_summary(mesh, mat, solver_cfg, cfg)

    # Create solver
    solver = StaggeredSolver(mesh, mat, bcs, config=solver_cfg, ctx=ctx)

    # Pre-seed notches as phase-field cracks. Standard convention
    # (Borden 2012, Bleyer 2017): a pre-existing notch is broken
    # material with d = 1 along its line, not just a geometric slit.
    # We pin the one-ring of elements adjacent to the listed
    # node-sets to d = 1 by injecting a very large H, so that
    # d_eq = 2H / (Gc/l0 + 2H) -> 1 at t = 0. This prevents the
    # spurious horizontal-band damage that AT2 (which has no elastic
    # threshold) would otherwise accumulate on traction-free notch
    # walls from reflected stress waves.
    ic = cfg.initial_conditions
    preseed_specs = []
    if ic is not None:
        if ic.preseed_notch_nodesets:
            from ..physics.initial_conditions import normalise_legacy_preseed
            preseed_specs.extend(
                normalise_legacy_preseed(ic.preseed_notch_nodesets))
        if ic.preseed_damage:
            preseed_specs.extend(ic.preseed_damage)
    if preseed_specs:
        import torch as _torch
        from ..physics.initial_conditions import (
            resolve_preseed_specs, value_to_H_seed)
        node_mask, value_per_node = resolve_preseed_specs(mesh, preseed_specs)
        if not bool(node_mask.any()):
            print("preseed_damage: no nodes matched any region/nodeset; "
                  "nothing preseeded.")
        else:
            # One-ring of elements: any element touching a seeded node.
            elem_node_vals = value_per_node[mesh.elements]  # (E, nv)
            elem_node_in = node_mask[mesh.elements]
            # Per-element value = max value across its seeded nodes
            # (zeros where no node is seeded — those stay unmasked).
            elem_value = (elem_node_vals * elem_node_in.to(elem_node_vals.dtype)
                          ).max(dim=1).values
            elem_mask = elem_node_in.any(dim=1)
            # Use the resolved Material, not raw YAML fields: inline YAML may
            # carry unit-suffixed strings such as "3 J/m^2" and "0.25 mm".
            l0_mm = mat.l0
            Gc = mat.Gc
            # Group by unique value to avoid Python-loop per element.
            unique_vals = _torch.unique(elem_value[elem_mask])
            for v in unique_vals.tolist():
                if v <= 0.0:
                    continue
                H_seed = value_to_H_seed(v, Gc, l0_mm)
                sel = elem_mask & (elem_value == v)
                solver.H_elem[sel] = _torch.maximum(
                    solver.H_elem[sel],
                    _torch.tensor(H_seed, dtype=solver.H_elem.dtype,
                                  device=solver.H_elem.device))
            n_seeded = int(elem_mask.sum().item())
            print(f"Pre-seeded damage: {n_seeded} one-ring elements from "
                  f"{len(preseed_specs)} spec(s); unique target values = "
                  f"{[round(v, 3) for v in unique_vals.tolist()]}")
            # Converge damage once so d ~= value on seeded elements at t = 0
            solver.step_solve_damage()

    # Apply Neumann/traction forces. Two ramp paths coexist:
    #   * Per-BC ``ramp_type`` (issue #138): if any Neumann BC has a
    #     non-constant ramp_type we recompute f_ext every step using
    #     ``bcs.get_neumann_forces(mesh, t=t_now)``.
    #   * Legacy global load_factor: for the constant-ramp case we cache
    #     the assembled force and scale it by ``bcs.load_factor`` each
    #     step, so a global ``loading.ramp_type: smooth`` / ``linear``
    #     still ramps the traction rather than applying a Heaviside step.
    has_time_traction = any(
        getattr(nbc, 'ramp_type', 'constant') != 'constant'
        for nbc in bcs.neumann_bcs
    )
    neumann_forces = bcs.get_neumann_forces(mesh)
    if neumann_forces is not None:
        solver.f_ext = neumann_forces.clone()

    # Pre-strain if needed
    if cfg.loading.protocol == 'two_step_prestrain':
        print(f"\n--- Pre-strain: u = {cfg.loading.prestrain_displacement} mm ---")
        solver.pre_strain(coupled=bool(cfg.loading.coupled_prestrain))
        print(f"Pre-strain done: max(u_y) = {solver.u[:, 1].max().item():.6e} mm\n")

    # Initial conditions plot (always)
    plot_initial_conditions(mesh, mat, bcs, solver_cfg,
                            save_path=os.path.join(output_dir, 'initial_conditions.png'))

    # Trajectory snapshots. Zarr is the public default; H5 remains available
    # for legacy postprocessors and paper artifacts.
    trajectory_format = getattr(cfg.output, 'trajectory_format', 'zarr')
    if trajectory_format not in ('zarr', 'h5', 'both'):
        raise ValueError(
            "output.trajectory_format must be one of: zarr, h5, both")
    zarr_root = None
    h5f = None
    if cfg.output.h5:
        if trajectory_format in ('zarr', 'both'):
            zarr_path = os.path.join(output_dir, 'training_data.zarr')
            zarr_root = init_zarr(zarr_path, mesh, mat)
            print(f"Zarr snapshots: {zarr_path}")
        if trajectory_format in ('h5', 'both'):
            h5_path = os.path.join(output_dir, 'training_data.h5')
            h5f = init_h5(h5_path, mesh, mat)
            print(f"H5 snapshots: {h5_path} (legacy compatibility)")

    # Determine step count. ``num_steps`` (YAML or --num_steps CLI) takes
    # precedence; only fall back to the CFL-derived n_steps = t_total/dt
    # when num_steps was left unset (0).
    if solver_cfg.solver_type == 'explicit':
        dt = solver.dt
        n_steps = cfg.loading.num_steps
        if n_steps == 0 and cfg.loading.t_total > 0:
            n_steps = int(math.ceil(cfg.loading.t_total / dt))
        print(f"\nRunning {n_steps} explicit steps, dt={dt:.4e}\n")
    else:
        dt = cfg.loading.dt
        n_steps = cfg.loading.num_steps
        loading = objs['loading']
        print(f"\nRunning {n_steps} quasi-static steps, du={dt}\n")

    # Main loop
    t_start = time.time()
    crack_step = None
    fast = cfg.output.fast
    print_every = cfg.output.print_every
    snapshot_every = cfg.output.h5_every

    # Per-step timing CSV for three-way comparator (compare.py reads
    # "Total Step Time" in seconds). Column schema matches
    # the archived Paper-1 timing-comparison runner schema so either
    # producer is drop-in compatible. Sub-component columns (Solid,
    # Strain, Driving Force, Phase) are zeros here because step_full()
    # doesn't break out internal timings; downstream tooling only
    # consumes Total Step Time.
    timing_csv_path = os.path.join(output_dir, 'timing_per_step.csv')
    timing_rows = []  # accumulate, write once at end for speed

    # Per-step solver telemetry (issue #300, sub of #298 PF-Hetero-Bench).
    # Emits ``solver_telemetry.csv`` next to ``timing_per_step.csv`` /
    # ``energy.csv`` / ``crack_tip.csv`` so the dataset pipeline can
    # carry per-step Newton/PCG iteration counts + absolute/relative
    # stagger and mechanics residuals + dt without re-running the full
    # bench.  Pure observability — reads ``solver._last_*`` and
    # ``mechanics.last_iter`` / ``damage_solver.last_iter`` (already
    # tracked internally).
    telemetry_csv_path = os.path.join(output_dir, 'solver_telemetry.csv')
    telemetry_rows = []
    energy_rows = []

    # Optional reaction-force CSV (load-displacement curve for QS
    # benchmarks). Columns follow docs/output_standards/index.md and include
    # convergence fields so compare scripts and audits can share one file.
    # Activated by setting output.reaction_node_set in YAML.
    # Streamed (line-buffered) so a killed run still has usable data.
    react_set_name = getattr(cfg.output, 'reaction_node_set', None)
    react_comp = int(getattr(cfg.output, 'reaction_component', 1))
    react_nodes = None
    react_disp_value = 0.0
    react_fh = None
    react_count = 0
    if react_set_name:
        if react_set_name not in mesh.node_sets:
            raise RuntimeError(
                f"output.reaction_node_set='{react_set_name}' not found in "
                f"mesh node sets {sorted(mesh.node_sets.keys())}")
        react_nodes = mesh.node_sets[react_set_name]
        results_csv = os.path.join(output_dir, 'results.csv')
        react_fh = open(results_csv, 'w', buffering=1)
        react_fh.write(
            "step,time,displacement,reaction_kN,max_d,max_H,"
            "stagger_iter,elapsed_ms\n"
        )
        # Pick the prescribed displacement value for this node-set/component
        # so we can plot reaction vs applied displacement (the BC ramp scales
        # this value by load_factor each step).
        comp_key = {0: 'x', 1: 'y'}.get(react_comp, react_comp)
        for entry in cfg.boundary_conditions:
            if entry.nodes != react_set_name:
                continue
            if entry.type == 'prescribe' and entry.component == react_comp:
                react_disp_value = float(entry.value)
                break
            if entry.type == 'rigid_connector' and entry.prescribe:
                # rigid_connector prescribes per-component displacement on
                # the master; it scales the same way (value * load_factor).
                if comp_key in entry.prescribe:
                    react_disp_value = float(entry.prescribe[comp_key])
                    break
                if react_comp in entry.prescribe:
                    react_disp_value = float(entry.prescribe[react_comp])
                    break

    # Cached Neumann reference for ramping (see note above)
    neumann_ref = (solver.f_ext.clone() if (
        bcs.get_neumann_forces(mesh) is not None) else None)

    for step in range(n_steps):
        t0 = time.time()

        if solver_cfg.solver_type != 'explicit' and objs['loading']:
            bcs.load_factor = objs['loading'][step]
        elif solver_cfg.solver_type == 'explicit':
            bcs.load_factor = compute_load_factor(step, dt, cfg.loading)

        # Update Neumann (traction) external force for this step.
        # Per-BC ramp_type takes precedence over the global load_factor
        # scaling; if any Neumann BC declares a non-constant ramp we
        # rebuild f_ext at the current physical time. Otherwise we just
        # scale the cached one-shot assembly by the Dirichlet load_factor
        # (so a global ``loading.ramp_type: smooth`` still ramps tractions).
        if has_time_traction and solver_cfg.solver_type == 'explicit':
            t_now = step * dt
            global_factor = bcs.load_factor
            bcs.load_factor = 1.0
            try:
                solver.f_ext = bcs.get_neumann_forces(mesh, t=t_now)
            finally:
                bcs.load_factor = global_factor
        elif neumann_ref is not None:
            solver.f_ext = neumann_ref * bcs.load_factor

        psi = solver.step_full()
        elapsed_s = time.time() - t0
        elapsed = elapsed_s * 1000

        max_d = solver.d.max().item()
        max_H = float(solver.H_nodal.max().item())
        timing_rows.append((step, max_d, max_H, 0.0, 0.0, 0.0, 0.0, elapsed_s))
        energies = solver.fem.compute_energy_components(
            solver.u, solver.d,
            getattr(solver, 'v', None),
            psi_plus=psi)
        if solver_cfg.solver_type == 'explicit':
            energy_time = step * dt
        else:
            energy_time = objs['loading'][step] if objs.get('loading') else step * dt
        energy_rows.append({
            'step': step,
            'time': float(energy_time),
            'elastic': energies['elastic'],
            'fracture': energies['fracture'],
            'kinetic': energies['kinetic'],
            'external': 0.0,
            'total': energies['total'],
        })

        # Solver telemetry (#300): one row per step. ``getattr`` with NaN
        # sentinels covers solver paths (e.g. ExplicitDynamics) that lack
        # CG counters, plus the very first step before any solver internal
        # has populated ``last_iter``.
        if solver_cfg.solver_type == 'explicit':
            t_now = step * dt
            dt_used = float(getattr(solver, '_last_dt_used', dt) or dt)
        else:
            t_now = (objs['loading'][step]
                     if objs.get('loading') else step * dt)
            dt_used = float(dt)
        telemetry_rows.append((
            step,
            float(t_now),
            int(getattr(solver, '_last_stagger_iter', 0)),
            int(getattr(solver.mechanics, 'last_iter', 0)),
            int(getattr(solver.damage_solver, 'last_iter', 0)),
            float(getattr(solver, '_last_residual', float('nan'))),
            float(getattr(solver, '_last_relative_residual', float('nan'))),
            float(getattr(solver, '_last_mechanics_residual', float('nan'))),
            float(getattr(solver, '_last_mechanics_relative_residual',
                          float('nan'))),
            dt_used,
        ))

        # Reaction-force logging (results.csv) -- streamed each step
        if react_fh is not None:
            R = solver.fem.compute_reaction_force(
                solver.u, solver.d, react_nodes, component=react_comp)
            disp_now = bcs.load_factor * react_disp_value
            react_fh.write(
                f"{step},{float(t_now):.9e},{disp_now:.8f},"
                f"{abs(R)/1000.0:.8f},{max_d:.8f},{max_H:.9e},"
                f"{int(getattr(solver, '_last_stagger_iter', 0))},"
                f"{elapsed:.3f}\n")
            react_count += 1

        if step % print_every == 0 or step < 5:
            if solver_cfg.solver_type == 'explicit':
                t_now = step * dt
                print(f"  Step {step:5d} | t={t_now*1e6:6.2f}us | "
                      f"max(d)={max_d:.5f} | {elapsed:.1f}ms", flush=True)
            else:
                disp = objs['loading'][step] if objs['loading'] else step * dt
                print(f"  Step {step:3d} | u={disp:.6f} | "
                      f"max(d)={max_d:.5f} | {elapsed:.0f}ms", flush=True)

        # Trajectory snapshot
        if (zarr_root or h5f) and step % snapshot_every == 0:
            strain = solver.fem.compute_strain(solver.u)
            exx, eyy, gxy = strain
            sxx, syy, sxy = solver.fem.compute_stress(
                solver.u, solver.d, strain=(exx, eyy, gxy))
            snapshot_kwargs = dict(
                eps_xx=exx, eps_yy=eyy, gam_xy=gxy,
                sxx=sxx, syy=syy, sxy=sxy,
                H_nodal=solver.H_nodal,
                velocity=getattr(solver, 'v', None),
                acceleration=getattr(solver, 'a', None),
                energies=energies,
                time_s=(step * dt if solver_cfg.solver_type == 'explicit'
                        else float(t_now)),
            )
            if zarr_root:
                write_zarr_snapshot(
                    zarr_root, step, mesh, solver.u, solver.d, psi,
                    solver.H_elem, **snapshot_kwargs)
            if h5f:
                write_h5_snapshot(
                    h5f, step, mesh, solver.u, solver.d, psi,
                    solver.H_elem, **snapshot_kwargs)

        # Crack detection
        if max_d > 0.99 and crack_step is None:
            crack_step = step
            print(f"  ** Crack at step {step} **", flush=True)

    t_total = time.time() - t_start
    n_done = step + 1
    print(f"\nTotal: {t_total:.1f}s ({n_done} steps, "
          f"{t_total/max(n_done,1)*1000:.1f} ms/step)")

    # Flush timing CSV
    with open(timing_csv_path, 'w') as fh:
        fh.write("step,max_d,max_H,Solid solve Time,Compute Strain Time,"
                 "Driving Force Time,Phase Solve Time,Total Step Time\n")
        for r in timing_rows:
            fh.write(f"{r[0]},{r[1]:.8f},{r[2]:.4e},"
                     f"{r[3]:.9f},{r[4]:.9f},{r[5]:.9f},{r[6]:.9f},{r[7]:.9f}\n")
    print(f"Saved: timing_per_step.csv ({len(timing_rows)} rows)")

    energy_csv_path = os.path.join(output_dir, 'energy.csv')
    write_energy_csv(energy_csv_path, energy_rows)
    print(f"Saved: energy.csv ({len(energy_rows)} rows)")
    if cfg.output.plots:
        plot_energy_history(
            energy_rows,
            os.path.join(output_dir, 'energy.png'),
            xlabel='Time [s]' if solver_cfg.solver_type == 'explicit'
            else 'Load parameter / displacement [mm]',
        )
        print("Saved: energy.png")

    # Flush solver-telemetry CSV (#300). Schema:
    #   step, time, newton_iters, pcg_iters_mech, pcg_iters_pf, residual, dt
    # ``residual`` is max(u_change, d_change) at staggered convergence;
    # NaN for explicit-dynamics rows (no stagger residual).
    write_solver_telemetry_csv(telemetry_csv_path, telemetry_rows)
    print(f"Saved: solver_telemetry.csv ({len(telemetry_rows)} rows)")

    if react_fh is not None:
        react_fh.close()
        print(f"Saved: results.csv ({react_count} rows; "
              f"node_set='{react_set_name}', component={react_comp})")

    if zarr_root:
        zarr_root.attrs['num_steps'] = n_done
        print("Zarr snapshots saved.")
    if h5f:
        h5f.attrs['num_steps'] = n_done
        h5f.close()
        print("H5 snapshots saved.")

    if cfg.output.profile and ctx.profiler._timings:
        from ..utils.io_utils import write_profiler_csv
        prof_path = os.path.join(output_dir, 'profiler.csv')
        write_profiler_csv(prof_path, ctx.profiler)
        print(ctx.profiler.summary())

    # Final damage plot (always)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_field(mesh, solver.d, title=f'Final Damage (step {n_done})',
               cmap='inferno', vmin=0, vmax=1, ax=ax)
    fig.savefig(os.path.join(output_dir, 'damage_final.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: damage_final.png")

    # Metadata
    save_run_metadata(
        output_dir,
        problem_name=cfg.name,
        device=ctx.device,
        material=mat,
        mesh=mesh,
        solver_config={
            'solver_type': solver_cfg.solver_type,
            'time_integrator': solver_cfg.time_integrator,
            'rho_inf': solver_cfg.rho_inf,
            'num_steps': n_done,
            'dt': dt if solver_cfg.solver_type == 'explicit' else cfg.loading.dt,
            # Wave speeds in mm/s (solver-internal units). Saved here so
            # post-processing (e.g. crack_tip.csv normalisation by c_R)
            # can recover the value without re-deriving it from material
            # parameters. ``c_R_m_s`` is the legacy alias for
            # backward-compatible readers.
            'c_R_mm_s': float(mat.c_R),
            'c_R_m_s': float(mat.c_R) * 1e-3,
            'c_p_mm_s': float(mat.c_p),
            'c_s_mm_s': float(mat.c_s),
        },
        extra={
            'total_time_s': round(t_total, 2),
            'crack_step': crack_step,
            'config_file': os.path.abspath(args.config),
            # Pre-seeded notch nodesets (issue #213): post-processing
            # detectors (initiation, branching-onset, full-Y) must
            # exclude these nodes when computing max(d) over the mesh,
            # because pf_dirichlet locks them at d=1 from t=0.
            'preseed_notch_nodesets': (
                list(cfg.initial_conditions.preseed_notch_nodesets)
                if (cfg.initial_conditions is not None
                    and cfg.initial_conditions.preseed_notch_nodesets)
                else []
            ),
        },
    )
    print(f"Saved: run_metadata.json")

    # Forward-pass visualisation (issue: --plots/--gif used to be no-ops).
    # When the user passes --plots or --gif (or sets cfg.output.plots/gif
    # in the YAML), invoke the inline postprocess pipeline so the user
    # gets damage_t*.png snapshots and damage_evolution.gif without a
    # second command. The current postprocess bridge is still H5-backed;
    # Zarr postprocess parity is tracked separately.
    want_plots = bool(getattr(cfg.output, 'plots', False))
    want_gif = bool(getattr(cfg.output, 'gif', False))
    if want_plots or want_gif:
        h5_path = os.path.join(output_dir, 'training_data.h5')
        zarr_path = os.path.join(output_dir, 'training_data.zarr')
        if os.path.exists(h5_path) or os.path.exists(zarr_path):
            try:
                from postprocess_paper import BenchmarkPostProcessor
                print(f"\n[forward-viz] generating "
                      f"{'snapshots' if want_plots else ''}"
                      f"{' + ' if want_plots and want_gif else ''}"
                      f"{'gif' if want_gif else ''}...")
                bp = BenchmarkPostProcessor(output_dir)
                animation_format = getattr(cfg.output, 'animation_format', 'mp4')
                animation_fields = getattr(
                    cfg.output, 'gif_fields', 'damage')
                animation_fields = animation_fields.replace(
                    'max_principal_stress', 'stress')
                max_frames = int(getattr(cfg.output, 'gif_frames', 80))
                animation_renderer = getattr(
                    cfg.output, 'animation_renderer', 'raster')
                raster_width = int(getattr(
                    cfg.output, 'animation_raster_width', 960))
                if want_plots and want_gif:
                    bp.generate_all(skip_gif=False,
                                    animation_format=animation_format,
                                    animation_fields=animation_fields,
                                    max_frames=max_frames,
                                    animation_renderer=animation_renderer,
                                    raster_width=raster_width)
                elif want_gif:
                    bp.generate_all(skip_gif=False, fields='gif',
                                    animation_format=animation_format,
                                    animation_fields=animation_fields,
                                    max_frames=max_frames,
                                    animation_renderer=animation_renderer,
                                    raster_width=raster_width)
                else:
                    # plots only -- skip the (slow) gif step.
                    bp.generate_all(skip_gif=True)
                bp.close()
                print(f"[forward-viz] outputs in {output_dir}/figures/")
            except Exception as exc:
                print(f"[forward-viz] failed: {exc}")
                print(f"  fallback: python -m phast postprocess "
                      f"{output_dir}")
        else:
            print("[forward-viz] skipped: no trajectory snapshots in run "
                  "dir (set output.trajectory: true or pass --trajectory)")

    # Post-processing
    print(f"\nRun: python -m phast postprocess {output_dir}")
    print(f"All outputs in: {output_dir}/")


if __name__ == '__main__':
    main()
