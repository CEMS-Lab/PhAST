"""Recovery from displacement histories and quasistatic observation channels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize
import torch
import yaml

from .current_forward import (ObservedRun, fixed_points, make_mesh, rollout, time_grid, validate)
from .forward import build_sampler
from .recovery import EvaluationLimit, ROOT, core_hashes, git_info, save_fields, write_csv, write_json


def detached(run: ObservedRun) -> ObservedRun:
    return ObservedRun({key: value.detach() for key, value in run.observations.items()},
                       run.damage.detach(), run.displacement.detach(), run.history,
                       run.snapshots, run.observation_times, run.dt)


def example_source_hashes(example: Path) -> dict[str, str]:
    shared = Path(__file__).resolve().parent
    paths = [shared / name for name in (
        "current_forward.py", "current_recovery.py", "current_plotting.py",
        "forward.py", "recovery.py", "plotting.py",
    )] + [path for path in (example / "run.py", example / "plot.py") if path.is_file()]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def package_source_hashes() -> dict[str, str]:
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src/phast").rglob("*.py"))}


def run_case(example: Path, config: dict[str, Any], profile_name: str,
             output: Path, mode: str) -> dict[str, Any]:
    import phast

    started = time.perf_counter()
    hashes = core_hashes()
    source_hashes = example_source_hashes(example)
    package_hashes = package_source_hashes()
    started_utc = datetime.now(timezone.utc).isoformat()
    validate(config)
    if Path(phast.__file__).resolve().parent != ROOT / "src/phast":
        raise RuntimeError("wrong PhAST checkout imported; run with PYTHONPATH=src")
    torch.set_num_threads(config["numerics"]["threads"])
    profile = config["profiles"][profile_name]
    dynamic = config["example"] == "cross_mesh_toughness"

    def checkpoint(phase: str, **details: Any) -> None:
        temporary = output / "progress.json.tmp"
        write_json(temporary, {
            "phase": phase, "success": False, "started_utc": started_utc,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": time.perf_counter() - started, "profile": profile_name,
            "resolved_config": config, "core_sha256": hashes,
            "example_source_sha256": source_hashes, "package_source_sha256": package_hashes,
            **details,
        })
        temporary.replace(output / "progress.json")

    checkpoint("mesh_generation")
    target_mesh = make_mesh(output / "target_mesh.msh", config, profile["h_target_mm"], profile["h_coarse_mm"])
    trial_mesh = make_mesh(output / "trial_mesh.msh", config, profile["h_trial_mm"], profile["h_coarse_mm"]) if dynamic else target_mesh
    truth = {name: spec["truth"] for name, spec in config["parameters"].items()}
    points, check = fixed_points(config) if dynamic else (None, None)
    target_sampler = build_sampler(target_mesh, points) if dynamic else None
    trial_sampler = build_sampler(trial_mesh, points) if dynamic else None
    if dynamic:
        dt, steps, observation_steps = time_grid(target_mesh, trial_mesh, config, profile)
        controls = {"dt": dt, "steps": steps, "observation_steps": observation_steps}
    else:
        dt, steps, controls = None, profile["frames"], {"frames": profile["frames"]}
    values = [torch.tensor(truth[name], dtype=torch.float64) for name in ("Gc", "l0")]
    checkpoint("target_forward", steps=steps, dt_s=dt)
    target_started = time.perf_counter()
    with torch.no_grad():
        target = rollout(target_mesh, config, *values, differentiable=False,
                         sampler=target_sampler, record=True, **controls)
    target_elapsed = time.perf_counter() - target_started
    save_fields(output, "original", target_mesh, target)
    if dynamic:
        scales = {"displacement": config["observations"]["loss_scale_mm"]}
        weights = {"displacement": 1.0}
        train_mask = torch.as_tensor(~check)
        for label, sampler in (("target", target_sampler), ("trial", trial_sampler)):
            np.savetxt(output / f"{label}_observation_operator.csv",
                       np.column_stack((points, check, sampler.node_ids.numpy(), sampler.weights.numpy())),
                       delimiter=",", fmt="%.12g", comments="",
                       header="x_mm,y_mm,check_point,node_0,node_1,node_2,w_0,w_1,w_2")
    else:
        scales = {name: max(float(field.abs().max()), 1.0 if name == "reaction" else 1.0e-12)
                  for name, field in target.observations.items()}
        weights = {name: config["observations"]["weights"][name] for name in target.observations}
        train_mask = None
    names = ["Gc"] if dynamic else ["Gc", "l0"]
    inverse = config["inverse"]
    initial_q = np.log([config["parameters"][name]["initial"] for name in names])
    bounds = [tuple(np.log(config["parameters"][name]["bounds"])) for name in names]
    evaluations: list[dict[str, Any]] = []
    best_loss = math.inf
    best_q = initial_q.copy()
    best: ObservedRun | None = None
    count = 0

    def forward(q: torch.Tensor, differentiable: bool, record: bool = False) -> ObservedRun:
        gc_value = q[0].exp()
        l0_value = torch.tensor(truth["l0"], dtype=torch.float64) if dynamic else q[1].exp()
        return rollout(trial_mesh, config, gc_value, l0_value, differentiable=differentiable,
                       sampler=trial_sampler, record=record, **controls)

    def loss_of(prediction: ObservedRun) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = {}
        for name, predicted in prediction.observations.items():
            error = predicted - target.observations[name]
            if dynamic:
                error = error[:, train_mask]
            components[name] = (error / scales[name]).square().mean()
        loss = sum(weights[name] * component for name, component in components.items())
        return loss, components

    checkpoint("initial_ad_check", steps=steps, dt_s=dt, target_elapsed_s=target_elapsed)
    q = torch.tensor(initial_q, dtype=torch.float64, requires_grad=True)
    nominal = forward(q, True)
    loss, _ = loss_of(nominal)
    ad = torch.autograd.grad(loss, q)[0].detach().numpy()
    initial_loss = float(loss.detach())
    del loss, nominal
    checks = []
    for index, name in enumerate(names):
        checkpoint("initial_fd_check", parameter=name, steps=steps, dt_s=dt,
                   target_elapsed_s=target_elapsed)
        delta = np.zeros_like(initial_q)
        delta[index] = inverse["fd_step"]
        with torch.no_grad():
            plus, _ = loss_of(forward(torch.tensor(initial_q + delta), False))
            minus, _ = loss_of(forward(torch.tensor(initial_q - delta), False))
        fd = float(plus - minus) / (2.0 * inverse["fd_step"])
        relative = abs(ad[index] - fd) / max(abs(ad[index]), abs(fd), 1.0e-14)
        checks.append({"parameter": name, "coordinate": "log_parameter", "fd_step": inverse["fd_step"],
                       "gradient_ad": float(ad[index]), "gradient_fd": fd,
                       "relative_error": relative, "within_5_percent": bool(relative <= 0.05)})
        del plus, minus
        gc.collect()
    write_csv(output / "gradient_check.csv", checks)
    if not dynamic and not all(row["within_5_percent"] for row in checks):
        raise RuntimeError("joint initial AD/FD check exceeds 5%; the recovery is not run")

    def evaluate(q_array: np.ndarray, iteration: int = 0) -> tuple[float, np.ndarray]:
        nonlocal count, best_loss, best_q, best
        if count >= inverse["max_evaluations"]:
            raise EvaluationLimit("maximum objective evaluations reached")
        count += 1
        evaluation_started = time.perf_counter()
        checkpoint("optimisation", objective_evaluations=count - 1,
                   evaluating_Gc_N_per_mm=float(np.exp(q_array[0])))
        q = torch.tensor(q_array, dtype=torch.float64, requires_grad=mode == "ad")
        with torch.set_grad_enabled(mode == "ad"):
            prediction = forward(q, mode == "ad")
            loss, components = loss_of(prediction)
            if mode == "ad":
                gradient = torch.autograd.grad(loss, q)[0].detach().numpy().copy()
            else:
                gradient = np.zeros_like(q_array)
                for index in range(len(names)):
                    offset = np.zeros_like(q_array)
                    offset[index] = inverse["fd_step"]
                    with torch.no_grad():
                        plus, _ = loss_of(forward(torch.tensor(q_array + offset), False))
                        minus, _ = loss_of(forward(torch.tensor(q_array - offset), False))
                    gradient[index] = float(plus - minus) / (2.0 * inverse["fd_step"])
        value = float(loss.detach())
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError("non-finite observation loss or gradient")
        row = {"iteration": iteration, "evaluation": count,
               "Gc_N_per_mm": float(np.exp(q_array[0])),
               "l0_mm": truth["l0"] if dynamic else float(np.exp(q_array[1])),
               "loss": value, "evaluation_elapsed_s": time.perf_counter() - evaluation_started,
               **{f"gradient_log_{name}": float(gradient[i]) for i, name in enumerate(names)},
               **{f"loss_{name}": float(component.detach()) for name, component in components.items()}}
        evaluations.append(row)
        if value < best_loss:
            best_loss, best_q, best = value, q_array.copy(), detached(prediction)
        write_csv(output / "history.csv", evaluations)
        write_json(output / "best_candidate.json", {
            "selection": "minimum_finite_training_loss", "loss": best_loss,
            "Gc_N_per_mm": float(np.exp(best_q[0])),
            "l0_mm": truth["l0"] if dynamic else float(np.exp(best_q[1])),
        })
        checkpoint("optimisation", objective_evaluations=count, best_loss=best_loss,
                   best_Gc_N_per_mm=float(np.exp(best_q[0])))
        print(f"evaluation {count}: loss={value:.5g}, Gc={row['Gc_N_per_mm']:.7g}, l0={row['l0_mm']:.7g}", flush=True)
        del loss, components, prediction
        gc.collect()
        return value, gradient

    stop_reason = "maximum_iterations"
    if dynamic:
        try:
            result = minimize(evaluate, initial_q, method="L-BFGS-B", jac=True, bounds=bounds,
                              options={"maxiter": inverse["max_iterations"], "maxfun": inverse["max_evaluations"],
                                       "maxls": 8, "maxcor": 5, "ftol": 1.0e-12, "gtol": 1.0e-10})
            stop_reason = str(result.message)
        except EvaluationLimit:
            stop_reason = "maximum_evaluations"
    else:
        q = torch.nn.Parameter(torch.tensor(initial_q, dtype=torch.float64))
        optimizer = torch.optim.Adam([q], lr=inverse["learning_rate"])
        for iteration in range(inverse["max_iterations"] + 1):
            try:
                _, gradient = evaluate(q.detach().numpy().copy(), iteration)
            except EvaluationLimit:
                stop_reason = "maximum_evaluations"
                break
            parameter_error = max(abs(float(np.exp(best_q[i])) / truth[name] - 1.0) for i, name in enumerate(names))
            if (profile_name == "quick" and inverse["truth_based_stopping"]
                    and parameter_error <= inverse["parameter_relative_tolerance"]
                    and best_loss <= 0.0025 and best_loss < initial_loss * (1.0 - inverse["minimum_loss_reduction"])):
                stop_reason = "synthetic_success"
                break
            if iteration < inverse["max_iterations"]:
                optimizer.zero_grad()
                q.grad = torch.tensor(gradient, dtype=torch.float64)
                optimizer.step()
                with torch.no_grad():
                    q.copy_(torch.maximum(torch.minimum(q, torch.tensor([b[1] for b in bounds])),
                                          torch.tensor([b[0] for b in bounds])))
    assert best is not None
    checkpoint("selected_forward", objective_evaluations=count, stop_reason=stop_reason,
               best_loss=best_loss, best_Gc_N_per_mm=float(np.exp(best_q[0])))
    with torch.no_grad():
        final = forward(torch.tensor(best_q, dtype=torch.float64), False, record=True)
    save_fields(output, "recovered", trial_mesh, final)
    errors = {name: abs(float(np.exp(best_q[index])) / truth[name] - 1.0) for index, name in enumerate(names)}
    relative = {name: float(torch.linalg.vector_norm(final.observations[name] - target.observations[name])
                           / torch.linalg.vector_norm(target.observations[name]).clamp(min=1.0e-30))
                for name in target.observations}
    check_error = None
    if dynamic:
        original = target.observations["displacement"][:, check]
        prediction = final.observations["displacement"][:, check]
        check_error = float(torch.linalg.vector_norm(prediction - original) / torch.linalg.vector_norm(original))
    reduction = 1.0 - best_loss / initial_loss
    success = (max(errors.values()) <= inverse["parameter_relative_tolerance"]
               and reduction >= inverse["minimum_loss_reduction"]
               and (check_error <= inverse["maximum_check_relative_error"] if dynamic else best_loss <= 0.0025)
               and all(row["within_5_percent"] for row in checks))
    rows = []
    original_u, recovered_u = (run.observations["displacement"].numpy() for run in (target, final))
    for frame, coordinate in enumerate(final.observation_times):
        for sensor in range(original_u.shape[1]):
            rows.append({"frame": frame, "time_s" if dynamic else "applied_displacement_mm": coordinate,
                         "sensor": sensor, "check_point": bool(check[sensor]) if dynamic else False,
                         "original_u_x_mm": float(original_u[frame, sensor, 0]),
                         "original_u_y_mm": float(original_u[frame, sensor, 1]),
                         "recovered_u_x_mm": float(recovered_u[frame, sensor, 0]),
                         "recovered_u_y_mm": float(recovered_u[frame, sensor, 1])})
    write_csv(output / "observations.csv", rows)
    if not dynamic:
        write_csv(output / "response.csv", [
            {"frame": i, "applied_displacement_mm": value, "original_reaction_N": float(target.observations["reaction"][i]),
             "recovered_reaction_N": float(final.observations["reaction"][i])}
            for i, value in enumerate(final.observation_times)])
        for label, run in (("original", target), ("recovered", final)):
            np.savetxt(output / f"{label}_strain.csv", run.observations["strain"].numpy().reshape(-1, 3),
                       delimiter=",", fmt="%.12g", header="eps_xx,eps_yy,eps_xy", comments="")
    write_csv(output / "history.csv", evaluations)
    write_csv(output / "parameter_history.csv", [{k: r[k] for k in ("evaluation", "Gc_N_per_mm", "l0_mm")} for r in evaluations])
    write_csv(output / "loss_history.csv", [{k: r[k] for k in ("evaluation", "loss")} for r in evaluations])
    summary = {"example": config["example"], "profile": profile_name, "success": bool(success),
               "observation_kind": config["observation_kind"], "truth": truth,
               "initial": {name: spec["initial"] for name, spec in config["parameters"].items()},
               "recovered": {"Gc": float(np.exp(best_q[0])), "l0": truth["l0"] if dynamic else float(np.exp(best_q[1]))},
               "relative_parameter_error": errors, "initial_loss": initial_loss, "final_loss": best_loss,
               "loss_reduction": reduction, "relative_observation_error": relative,
               "check_relative_field_error": check_error, "initial_gradient_check_passed": all(r["within_5_percent"] for r in checks),
               "gradient_check_role": "informational" if dynamic else "required_initial_check",
               "gradient_mode": mode, "objective_evaluations": count, "stop_reason": stop_reason,
               "steps": steps, "dt_s": dt, "observation_times": final.observation_times,
               "observation_coordinate": "time_s" if dynamic else "applied_displacement_mm",
               "training_sensors": int((~check).sum()) if dynamic else original_u.shape[1],
               "check_sensors": int(check.sum()) if dynamic else 0,
               "target_max_damage": float(target.damage.max()), "primal_checks_passed": True,
               "elapsed_s": time.perf_counter() - started,
               "target_elapsed_s": target_elapsed,
               "selection": "minimum_finite_training_loss",
               "selected_evaluation": min(evaluations, key=lambda row: row["loss"])["evaluation"],
               "mesh": {label: {"nodes": mesh.n_nodes, "elements": mesh.n_elems,
                                  "crack_size_mm": profile[f"h_{label}_mm"]}
                        for label, mesh in (("target", target_mesh), ("trial", trial_mesh))}}
    if core_hashes() != hashes:
        raise RuntimeError("solver source changed during the run; rerun in a new directory")
    if example_source_hashes(example) != source_hashes:
        raise RuntimeError("example source changed during the run; rerun in a new directory")
    if package_source_hashes() != package_hashes:
        raise RuntimeError("package source changed during the run; rerun in a new directory")
    write_json(output / "summary.json", summary)
    text = yaml.safe_dump(config, sort_keys=False)
    (output / "config.yaml").write_text(text, encoding="utf-8")
    versions = {"phast": version("phast"), "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()}
    units = {"length": "mm", "time": "s", "Gc": "N/mm", "stress": "MPa", "force": "N", "loss": "1"}
    write_json(output / "run_metadata.json", {"config": config, "units": units, "versions": versions,
                                              "git": git_info(), "mesh": summary["mesh"],
                                              "device": "cpu", "dtype": "float64", "threads": torch.get_num_threads(),
                                              "elapsed_s": summary["elapsed_s"], "scales": scales})
    write_json(output / "run_lockfile.json", {"resolved_config": config, "profile": profile_name,
                                              "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                              "core_sha256": hashes, "example_source_sha256": source_hashes,
                                              "package_source_sha256": package_hashes,
                                              "versions": versions, "dt_s": dt, "steps": steps})
    from .current_plotting import plot_output
    plot_output(output)
    if (core_hashes() != hashes or example_source_hashes(example) != source_hashes
            or package_source_hashes() != package_hashes):
        raise RuntimeError("source changed during postprocessing; no completed bundle")
    write_json(output / "run_manifest.json", {"schema_version": 1, "example": config["example"],
                                               "status": "beta", "success": bool(success), "units": units,
                                               "command": f"PYTHONPATH=src python {example.relative_to(ROOT).as_posix()}/run.py --profile {profile_name} --output-dir <new_directory>",
                                               "artifacts": sorted(path.name for path in output.iterdir() if path.is_file())})
    checkpoint("complete", success=bool(success), objective_evaluations=count,
               relative_parameter_error=errors, check_relative_field_error=check_error,
               stop_reason=stop_reason)
    return summary


def main(example: Path) -> int:
    parser = argparse.ArgumentParser(description="Synthetic displacement-based AT2 parameter recovery")
    parser.add_argument("--config", type=Path, default=example / "config.yaml")
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--gradient-mode", choices=("ad", "fd"), default="ad")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate(config)
    if args.profile not in config["profiles"]:
        parser.error("the selected profile is not included in this configuration")
    if args.validate_only:
        print(f"Valid {config['example']} displacement recovery configuration")
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    output = args.output_dir.resolve()
    if ROOT / "examples" in output.parents or (output.exists() and any(output.iterdir())):
        parser.error("use a new empty output directory outside examples/")
    output.mkdir(parents=True, exist_ok=True)
    try:
        summary = run_case(example, config, args.profile, output, args.gradient_mode)
    except (RuntimeError, ValueError, NotImplementedError) as error:
        write_json(output / "failure.json", {"success": False, "error": str(error)})
        print(f"Run stopped: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 2
