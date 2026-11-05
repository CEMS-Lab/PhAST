"""Displacement-based dynamic and quasistatic AT2 forward models."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phast.boundary_conditions import BoundaryConditions
from phast.device import DeviceContext
from phast.material import Material
from phast.mesh import FEMMesh
from phast.mesh_generator import miehe_tension
from phast.staggered_solver import SolverConfig, StaggeredSolver

from .forward import Sampler, positive, positive_int


@dataclass
class ObservedRun:
    observations: dict[str, torch.Tensor]
    damage: torch.Tensor
    displacement: torch.Tensor
    history: list[dict[str, Any]]
    snapshots: np.ndarray
    observation_times: list[float]
    dt: float | None


def validate(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    name = config.get("example")
    if name not in {"cross_mesh_toughness", "joint_toughness_length"}:
        raise ValueError("unsupported displacement recovery example")
    if config["numerics"]["device"] != "cpu":
        raise ValueError("these checked displacement examples require cpu")
    material = config["material"]
    for key in ("E_MPa", "rho_tonne_per_mm3", "eta_residual"):
        positive(material[key], key)
    if not -1.0 < material["nu"] < 0.5 or material["pf_model"] != "AT2":
        raise ValueError("invalid AT2 material")
    if material["energy_split"] != ("spectral" if name == "cross_mesh_toughness" else "isotropic"):
        raise ValueError("incorrect energy split for this recovery example")
    for key in ("width_mm", "height_mm", "notch_length_mm"):
        positive(config["geometry"][key], key)
    if config["geometry"]["notch_length_mm"] >= config["geometry"]["width_mm"]:
        raise ValueError("notch must lie inside the plate")
    if name == "joint_toughness_length" and config["geometry"]["height_mm"] != config["geometry"]["width_mm"]:
        raise ValueError("the joint mesh generator requires a square plate")
    for spec in config["parameters"].values():
        positive(spec["truth"], "parameter truth")
        positive(spec["initial"], "parameter initial")
        if "bounds" in spec:
            low, high = spec["bounds"]
            if not 0 < low <= spec["initial"] <= high or not low < spec["truth"] < high:
                raise ValueError("invalid parameter bounds")
    if config["numerics"]["history_update"] != "hard_max" or config["numerics"]["bounds"] != "post_clamp":
        raise ValueError("hard_max and post_clamp are required")
    for key in ("damage_tolerance", "adjoint_relative_tolerance"):
        positive(config["numerics"][key], key)
    for key in ("threads", "damage_max_iterations"):
        positive_int(config["numerics"][key], key)
    for key in ("max_iterations", "max_evaluations"):
        positive_int(config["inverse"][key], key)
    positive(config["inverse"]["fd_step"], "fd_step")
    for key in ("parameter_relative_tolerance", "maximum_check_relative_error", "minimum_loss_reduction"):
        if not 0 < positive(config["inverse"][key], key) < 1:
            raise ValueError(f"{key} must be less than one")
    if not isinstance(config["inverse"]["truth_based_stopping"], bool):
        raise ValueError("truth_based_stopping must be boolean")
    if "full" not in config["profiles"] or set(config["profiles"]) - {"quick", "full"}:
        raise ValueError("a full profile is required; quick is optional")
    for profile in config["profiles"].values():
        for key in ("h_target_mm", "h_trial_mm", "h_coarse_mm"):
            positive(profile[key], key)
        if name == "cross_mesh_toughness":
            if profile["h_target_mm"] >= profile["h_trial_mm"]:
                raise ValueError("target must be finer than recovery mesh")
            if profile["dt_s"] is not None:
                positive(profile["dt_s"], "dt_s")
                positive_int(profile["steps"], "steps")
        else:
            if profile["h_target_mm"] != profile["h_trial_mm"]:
                raise ValueError("joint recovery requires one fixed mesh")
            positive_int(profile["frames"], "frames")
    if name == "cross_mesh_toughness":
        if config["loading"]["type"] != "ramped_displacement" or config["inverse"]["method"] != "lbfgsb":
            raise ValueError("cross-mesh requires ramped_displacement and lbfgsb")
        for key in ("notch_half_width_mm", "refinement_band_mm"):
            positive(config["geometry"][key], key)
        if config["parameters"]["l0"]["initial"] != config["parameters"]["l0"]["truth"]:
            raise ValueError("cross-mesh l0 must remain fixed")
        for key in ("ramp_time_s", "end_time_s", "top_displacement_mm"):
            positive(config["loading"][key], key)
        if config["loading"]["bottom_displacement_mm"] != -config["loading"]["top_displacement_mm"]:
            raise ValueError("top and bottom displacements must be equal and opposite")
        if config["loading"]["ramp_time_s"] > config["loading"]["end_time_s"]:
            raise ValueError("ramp exceeds observation duration")
        times = config["observations"]["requested_times_s"]
        if any(not 0 < t <= config["loading"]["end_time_s"] for t in times) or times != sorted(set(times)):
            raise ValueError("invalid observation times")
        positive(config["observations"]["loss_scale_mm"], "loss_scale_mm")
        positive(config["observations"]["grid_spacing_mm"], "grid_spacing_mm")
        if not 0 <= config["observations"]["grid_start_mm"] < config["observations"]["grid_stop_mm"]:
            raise ValueError("invalid observation grid extent")
    else:
        if config["loading"]["type"] != "quasistatic_displacement" or config["inverse"]["method"] != "adam":
            raise ValueError("joint requires quasistatic_displacement and adam")
        positive(config["inverse"]["learning_rate"], "learning_rate")
        positive_int(config["observations"]["node_stride"], "node_stride")
        margin = config["observations"]["interior_margin_mm"]
        if not 0 <= margin < config["geometry"]["width_mm"] / 2:
            raise ValueError("invalid joint observation margin")
        for channel in ("displacement", "strain", "reaction"):
            positive(config["observations"]["weights"][channel], channel)
        positive(config["loading"]["maximum_displacement_mm"], "maximum_displacement_mm")
        for key in ("mechanics_tolerance", "stagger_tolerance"):
            positive(config["numerics"][key], key)
        positive_int(config["numerics"]["max_stagger"], "max_stagger")
        if config["observations"]["weights"]["damage"] != 0:
            raise ValueError("damage is not an observation channel in joint recovery")


def make_mesh(path: Path, config: dict[str, Any], h: float, coarse: float) -> FEMMesh:
    if config["example"] == "joint_toughness_length":
        miehe_tension(str(path), L=config["geometry"]["width_mm"],
                      a=config["geometry"]["notch_length_mm"],
                      l0=config["parameters"]["l0"]["truth"], h_crack=h, h_coarse=coarse,
                      verbose=False)
    else:
        import gmsh

        geo = config["geometry"]
        width, height, notch = (geo[k] for k in ("width_mm", "height_mm", "notch_length_mm"))
        mouth, band = geo["notch_half_width_mm"], geo["refinement_band_mm"]
        # The centre line aligns triangles; it is not an imposed crack or damage band.
        text = f"""// Displacement-driven SENT, coordinates in mm.
h = {h}; h_coarse = {coarse}; W = {width}; H = {height}; a = {notch};
Point(1) = {{0, 0, 0, h_coarse}}; Point(2) = {{W, 0, 0, h_coarse}};
Point(3) = {{W, H, 0, h_coarse}}; Point(4) = {{0, H, 0, h_coarse}};
Point(5) = {{0, H/2 + {mouth}, 0, h}}; Point(6) = {{0, H/2 - {mouth}, 0, h}};
Point(7) = {{a, H/2, 0, h}}; Point(8) = {{W, H/2, 0, h}};
Line(1) = {{1, 2}}; Line(2) = {{2, 8}}; Line(3) = {{8, 3}};
Line(4) = {{3, 4}}; Line(5) = {{4, 5}}; Line(6) = {{5, 7}};
Line(7) = {{7, 6}}; Line(8) = {{6, 1}}; Line(9) = {{7, 8}};
Curve Loop(1) = {{1, 2, -9, 7, 8}}; Plane Surface(1) = {{1}};
Curve Loop(2) = {{9, 3, 4, 5, 6}}; Plane Surface(2) = {{2}};
Physical Curve("bottom") = {{1}}; Physical Curve("right") = {{2, 3}};
Physical Curve("top") = {{4}}; Physical Curve("left") = {{5, 8}};
Physical Curve("notch_upper") = {{6}}; Physical Curve("notch_lower") = {{7}};
Physical Surface("plate") = {{1, 2}};
Field[1] = Distance; Field[1].CurvesList = {{6, 7, 9}}; Field[1].Sampling = 400;
Field[2] = Threshold; Field[2].InField = 1;
Field[2].SizeMin = h; Field[2].SizeMax = h_coarse;
Field[2].DistMin = {band}; Field[2].DistMax = {2 * band}; Background Field = 2;
Mesh.MeshSizeExtendFromBoundary = 0; Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0; Mesh.MeshSizeMin = h; Mesh.MeshSizeMax = h_coarse;
Mesh.Algorithm = 6; Mesh.ElementOrder = 1;
"""
        path.with_suffix(".geo").write_text(text, encoding="utf-8")
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.NumThreads", 1)
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(path.with_suffix(".geo")))
            gmsh.model.mesh.generate(2)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(str(path))
        finally:
            gmsh.finalize()
    mesh = FEMMesh(str(path), device="cpu", dtype=torch.float64)
    mesh.identify_boundaries()
    return mesh


def make_solver(mesh: FEMMesh, config: dict[str, Any], gc: torch.Tensor,
                l0: torch.Tensor, differentiable: bool, dt: float | None = None) -> StaggeredSolver:
    mat = config["material"]
    material = Material(E=mat["E_MPa"], nu=mat["nu"], rho=mat["rho_tonne_per_mm3"],
                        Gc=float(gc.detach()), l0=float(l0.detach()),
                        energy_split=mat["energy_split"], eta_residual=mat["eta_residual"],
                        pf_model="AT2", gamma_correction=False, plane_stress=False)
    bcs = BoundaryConditions(mesh.n_nodes, device=mesh.device, dtype=mesh.dtype)
    dynamic = config["example"] == "cross_mesh_toughness"
    if dynamic:
        bcs.fix(mesh.node_sets["left"], 0)
        bcs.fix(mesh.node_sets["right"], 0)
        bcs.add(mesh.node_sets["top"], 1, config["loading"]["top_displacement_mm"])
        bcs.add(mesh.node_sets["bottom"], 1, config["loading"]["bottom_displacement_mm"])
    else:
        bcs.fix(mesh.node_sets["bottom"], 0)
        bcs.fix(mesh.node_sets["bottom"], 1)
        bcs.add(mesh.node_sets["top"], 1, 1.0)
    num = config["numerics"]
    kwargs = {} if dynamic else {
        "static_tol": num["mechanics_tolerance"], "stagger_tol": num["stagger_tolerance"],
        "max_stagger": num["max_stagger"], "backend": "scipy",
    }
    cfg = SolverConfig(solver_type="explicit" if dynamic else "quasi_static",
                       differentiable=differentiable, dt_safety=0.8, damage_every=1,
                       preconditioner="jacobi", use_multigrid=False,
                       damage_tol=num["damage_tolerance"], damage_max_iter=num["damage_max_iterations"],
                       bounds_method="post_clamp", H_update_method="hard_max",
                       damage_true_residual_audit=True, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = DeviceContext(device="cpu", dtype=torch.float64, compile_solvers=False)
        solver = StaggeredSolver(mesh, material, bcs, config=cfg, ctx=ctx)
    if dt is not None:
        if not 0 < dt <= 0.8 * float(solver.fem.dt_cfl) * (1.0 + 1.0e-12):
            raise ValueError("shared time step exceeds the mesh CFL limit")
        solver.dt = solver.mechanics.dt = dt
    solver.diff_Gc, solver.diff_l0 = gc, l0
    solver.damage_solver.adjoint_pcg_rel_tol = num["adjoint_relative_tolerance"]
    solver.damage_solver.adjoint_pcg_abs_tol = 0.0
    solver.damage_solver.record_adjoint_pcg_telemetry = True
    return solver


def fixed_points(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    spec = config["observations"]
    axis = np.arange(spec["grid_start_mm"], spec["grid_stop_mm"], spec["grid_spacing_mm"])
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel()))
    geo = config["geometry"]
    points = points[~((points[:, 0] <= geo["notch_length_mm"])
                      & (abs(points[:, 1] - geo["height_mm"] / 2) < 0.01))]
    check = (np.rint(points.sum(axis=1) / spec["grid_spacing_mm"]).astype(int) % 4) == 0
    return points, check


def time_grid(target_mesh: FEMMesh, trial_mesh: FEMMesh, config: dict[str, Any],
              profile: dict[str, Any]) -> tuple[float, int, list[int]]:
    values = [torch.tensor(config["parameters"][n]["truth"], dtype=torch.float64) for n in ("Gc", "l0")]
    cfl = [float(make_solver(mesh, config, *values, False).fem.dt_cfl) for mesh in (target_mesh, trial_mesh)]
    end = config["loading"]["end_time_s"]
    if profile["dt_s"] is not None:
        dt, steps = profile["dt_s"], profile["steps"]
        if not math.isclose(steps * dt, end, rel_tol=1.0e-12):
            raise ValueError("duration and fixed time step differ")
    else:
        steps = math.ceil(end / (0.8 * min(cfl)))
        dt = end / steps
    if dt > 0.8 * min(cfl) * (1.0 + 1.0e-12):
        raise ValueError("fixed shared time step is unstable for a generated mesh")
    wanted = [max(1, min(steps, round(t / dt))) for t in config["observations"]["requested_times_s"]]
    if len(set(wanted)) != len(wanted):
        raise ValueError("time grid collapses observation times")
    return dt, steps, wanted


def rollout(mesh: FEMMesh, config: dict[str, Any], gc: torch.Tensor, l0: torch.Tensor,
            *, differentiable: bool, dt: float | None = None, steps: int | None = None,
            observation_steps: list[int] | None = None, sampler: Sampler | None = None,
            frames: int | None = None, record: bool = False) -> ObservedRun:
    solver = make_solver(mesh, config, gc, l0, differentiable, dt)
    dynamic = config["example"] == "cross_mesh_toughness"
    if dynamic:
        assert steps is not None and dt is not None and observation_steps is not None and sampler is not None
        sequence = [min(i * dt / config["loading"]["ramp_time_s"], 1.0) for i in range(1, steps + 1)]
        wanted = set(observation_steps)
    else:
        assert frames is not None
        maximum = config["loading"]["maximum_displacement_mm"]
        sequence = np.linspace(maximum / frames, maximum, frames).tolist()
        wanted = set(range(1, frames + 1))
    data: dict[str, list[torch.Tensor]] = {"displacement": [], "strain": [], "reaction": []}
    coords = mesh.nodes
    margin = config["observations"].get("interior_margin_mm", 0.02)
    width, height = config["geometry"]["width_mm"], config["geometry"]["height_mm"]
    obs = torch.where((coords[:, 0] >= margin) & (coords[:, 0] <= width - margin)
                      & (coords[:, 1] >= margin) & (coords[:, 1] <= height - margin))[0]
    obs = obs[::config["observations"].get("node_stride", 1)]
    rows: list[dict[str, Any]] = []
    snapshots = [solver.d.detach().cpu().numpy().copy()] if record else []
    snapshot_steps = set(np.linspace(1, len(sequence), min(20, len(sequence)), dtype=int))
    realised: list[float] = []
    for step, value in enumerate(sequence, 1):
        damage_floor = solver.d.clone()
        previous = solver.d.detach().clone()
        solver.bcs.load_factor = float(value)
        solver.step_full()
        if solver.damage_solver.last_converged is not True:
            raise RuntimeError(f"damage linear solve did not converge at step {step}")
        if not torch.isfinite(solver.u.detach()).all() or not torch.isfinite(solver.d.detach()).all():
            raise RuntimeError(f"non-finite fields at step {step}")
        if (solver.d.detach() - previous).min() < -1.0e-10:
            raise RuntimeError(f"damage decreased at step {step}")
        if not dynamic:
            if not solver._last_mechanics_converged:
                raise RuntimeError(f"mechanics did not converge at frame {step}")
            if not math.isfinite(solver._last_residual) or solver._last_residual >= config["numerics"]["stagger_tolerance"]:
                raise RuntimeError(f"staggered state did not converge at frame {step}")
            mask, _ = solver.bcs.get_masks_and_values()

            def force_residual() -> float:
                with torch.no_grad():
                    force = solver.fem.internal_force(solver.u.detach(), solver.d.detach()) - solver.f_ext
                    value = float(force[~mask].norm())
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite final force residual at frame {step}")
                return value

            final_force_residual = force_residual()
            final_change = float(solver._last_residual)
            corrections = 0
            # Check equilibrium on the returned pair, after damage changes.
            # Continue the same coupled equations with the original frame floor.
            while final_force_residual > config["numerics"]["mechanics_tolerance"]:
                if solver._last_stagger_iter + corrections >= config["numerics"]["max_stagger"]:
                    raise RuntimeError(f"final coupled equilibrium did not converge at frame {step}")
                old_u, old_d = solver.u, solver.d
                solver.step_mechanics()
                solver.step_compute_driving_force(strain=solver.fem.compute_strain(solver.u))
                solver.step_solve_damage(d_prev_step=damage_floor)
                if not solver._last_mechanics_converged or solver.damage_solver.last_converged is not True:
                    raise RuntimeError(f"final coupled correction did not converge at frame {step}")
                final_change = max(float((solver.u.detach() - old_u.detach()).norm()
                                         / solver.u.detach().norm().clamp(min=1.0e-30)),
                                   float((solver.d.detach() - old_d.detach()).norm()
                                         / solver.d.detach().norm().clamp(min=1.0e-30)))
                corrections += 1
                final_force_residual = force_residual()
            if final_change >= config["numerics"]["stagger_tolerance"]:
                raise RuntimeError(f"final staggered change did not converge at frame {step}")
        else:
            final_force_residual = None
            final_change, corrections = None, 0
        if step in wanted:
            if dynamic:
                displacement = (solver.u[sampler.node_ids] * sampler.weights[..., None]).sum(dim=1)
            else:
                displacement = solver.u[obs]
                data["strain"].append(torch.stack(solver.fem.compute_strain(solver.u), dim=-1))
                data["reaction"].append(solver.fem.internal_force(solver.u, solver.d)[mesh.node_sets["top"], 1].sum())
            data["displacement"].append(displacement)
            realised.append(step * dt if dynamic else float(value))
        if record:
            rows.append({"step": step, "time_s" if dynamic else "applied_displacement_mm":
                         step * dt if dynamic else float(value), "max_damage": float(solver.d.detach().max()),
                         "damage_converged": True,
                         "damage_relative_residual": float(solver.damage_solver.last_relative_residual),
                         "stagger_iterations": int(solver._last_stagger_iter) + corrections,
                         "final_coupled_corrections": corrections,
                         "stagger_change": final_change,
                         "mechanics_converged": None if dynamic else bool(solver._last_mechanics_converged),
                         "final_free_force_residual_N": final_force_residual})
            if step in snapshot_steps:
                snapshots.append(solver.d.detach().cpu().numpy().copy())
    observations = {key: torch.stack(values) for key, values in data.items() if values}
    return ObservedRun(observations, solver.d, solver.u, rows, np.asarray(snapshots), realised, dt)
