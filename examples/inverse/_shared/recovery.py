"""Bounded synthetic parameter recovery through the PhAST forward solver."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from .forward import (FEMMesh, ForwardResult, build_mesh,
                      build_solver, run_forward, validate_config)


ROOT = Path(__file__).resolve().parents[3]
CORE_FILES = (
    "src/phast/solvers/damage_solver.py", "src/phast/solvers/staggered_solver.py",
    "src/phast/solvers/mechanics_solver.py", "src/phast/core/fem_operators.py",
    "src/phast/core/mesh_generator.py", "src/phast/core/mesh.py",
    "src/phast/physics/material.py", "src/phast/physics/boundary_conditions.py",
)


class EvaluationLimit(RuntimeError):
    """Stop a run when its configured evaluation limit is reached."""


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bounded_parameters(z: torch.Tensor, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Compatibility helper selecting the configuration's positive parameter map."""
    if config["inverse"].get("parameterization", "bounded_logit") == "log":
        return parameter_values(z, config)
    values = {
        name: torch.tensor(spec["truth"], device=z.device, dtype=z.dtype)
        for name, spec in config["parameters"].items()
    }
    for index, name in enumerate(config["recover"]):
        lower, upper = config["parameters"][name]["bounds"]
        values[name] = lower + (upper - lower) * torch.sigmoid(z[index])
    return values


def initial_latent(config: dict[str, Any]) -> torch.Tensor:
    coordinates = []
    for name in config["recover"]:
        spec = config["parameters"][name]
        if config["inverse"].get("parameterization", "bounded_logit") == "log":
            coordinates.append(math.log(spec["initial"]))
            continue
        lower, upper = spec["bounds"]
        probability = (spec["initial"] - lower) / (upper - lower)
        coordinates.append(math.log(probability / (1.0 - probability)))
    return torch.tensor(coordinates, dtype=torch.float64,
                        device=config["numerics"]["device"], requires_grad=True)


def parameter_values(z: torch.Tensor, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    if config["inverse"].get("parameterization", "bounded_logit") == "bounded_logit":
        return bounded_parameters(z, config)
    values = {name: torch.tensor(spec["truth"], device=z.device, dtype=z.dtype)
              for name, spec in config["parameters"].items()}
    for index, name in enumerate(config["recover"]):
        values[name] = z[index].exp()
    return values


def save_fields(output: Path, label: str, mesh: FEMMesh, result: ForwardResult) -> None:
    nodes = mesh.nodes.detach().cpu().numpy()
    damage = result.damage.detach().cpu().numpy()
    displacement = result.displacement.detach().cpu().numpy()
    data = np.column_stack((np.arange(mesh.n_nodes), nodes[:, :2], damage, displacement))
    np.savetxt(output / f"{label}_fields.csv", data, delimiter=",", fmt="%.12g",
               header="node,x_mm,y_mm,damage,u_x_mm,u_y_mm", comments="")
    np.savetxt(output / f"{label}_elements.csv", mesh.elements.detach().cpu().numpy(),
               delimiter=",", fmt="%d", header="node_0,node_1,node_2", comments="")
    write_csv(output / f"{label}_history.csv", result.history)
    np.savez_compressed(output / f"{label}_snapshots.npz", damage=result.snapshots)


def git_info() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                            text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def core_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CORE_FILES}


def gradient_check(z: torch.Tensor, objective: Callable[[torch.Tensor, bool], torch.Tensor],
                   config: dict[str, Any]) -> list[dict[str, Any]]:
    loss = objective(z, True)
    gradient = torch.autograd.grad(loss, z)[0].detach()
    rows = []
    step = config["inverse"]["fd_step"]
    for index, name in enumerate(config["recover"]):
        direction = torch.zeros_like(z)
        direction[index] = step
        with torch.no_grad():
            plus = float(objective(z.detach() + direction, False))
            minus = float(objective(z.detach() - direction, False))
        fd = (plus - minus) / (2.0 * step)
        ad = float(gradient[index])
        error = abs(ad - fd) / max(abs(ad), abs(fd), 1.0e-14)
        passed = (math.isfinite(ad) and math.isfinite(fd) and abs(ad) > 1e-14
                  and abs(fd) > 1e-14 and error <= config["inverse"]["gradient_relative_tolerance"])
        rows.append({"parameter": name,
                     "coordinate": config["inverse"].get("parameterization", "bounded_logit"), "fd_step": step,
                     "gradient_ad": ad, "gradient_fd": fd, "relative_error": error,
                     "passed": passed})
    return rows


def run_case(example_dir: Path, config: dict[str, Any], profile_name: str,
             output: Path, gradient_mode: str) -> dict[str, Any]:
    import phast

    started = time.perf_counter()
    initial_core_hashes = core_hashes()
    validate_config(config)
    profile = config["profiles"][profile_name]
    torch.set_num_threads(config["numerics"]["threads"])
    core = Path(phast.__file__).resolve()
    # A source checkout must not silently use an editable install from another tree.
    if (ROOT / "src" / "phast").is_dir() and core.parent != ROOT / "src" / "phast":
        raise RuntimeError("wrong PhAST checkout imported; run with PYTHONPATH=src")
    target_mesh = build_mesh(output / "target_mesh.msh", config,
                             profile["h_target_mm"], profile["h_coarse_mm"])
    trial_mesh = target_mesh
    truth = {name: spec["truth"] for name, spec in config["parameters"].items()}
    steps = profile["steps"]
    dt = float(build_solver(target_mesh, config, truth["Gc"], truth["l0"],
                            differentiable=False).dt)
    requested_time = steps * dt
    with torch.no_grad():
        target = run_forward(target_mesh, config, truth["Gc"], truth["l0"], steps,
                             differentiable=False, dt=dt, record=True)
    save_fields(output, "original", target_mesh, target)
    target_observations = target.damage.detach()
    observe: Callable[[torch.Tensor], torch.Tensor] = lambda field: field
    points = target_mesh.nodes.detach().cpu().numpy()[:, :2]
    normalization = float(torch.mean(target_observations ** 2))
    if normalization <= 1.0e-20:
        raise RuntimeError("target damage is too small for parameter recovery")
    inverse = config["inverse"]
    loss_scale = normalization if inverse.get("loss_scaling", "target_mse") == "target_mse" else 1.0
    count = 0

    def objective(latent: torch.Tensor, differentiable: bool) -> torch.Tensor:
        nonlocal count
        if count >= inverse["max_evaluations"]:
            raise EvaluationLimit("maximum forward evaluations reached")
        count += 1
        values = parameter_values(latent, config)
        for name in config["recover"]:
            value = float(values[name].detach())
            lower, upper = config["parameters"][name]["bounds"]
            if not math.isfinite(value) or not lower < value < upper:
                raise RuntimeError(f"{name} left the admissible interval; no clipping or restart applied")
        result = run_forward(trial_mesh, config, values["Gc"], values["l0"], steps,
                             differentiable=differentiable, dt=dt)
        return torch.mean((observe(result.damage) - target_observations) ** 2) / loss_scale

    z = initial_latent(config)
    checks = gradient_check(z, objective, config)
    write_csv(output / "gradient_check.csv", checks)
    if not all(row["passed"] for row in checks):
        raise RuntimeError("initial AD/FD check did not meet the configured tolerance")
    accepted: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []
    optimiser = torch.optim.LBFGS([z], lr=inverse.get("learning_rate", 1.0),
                                  max_iter=inverse["inner_iterations"],
                                  max_eval=inverse.get("max_inner_evaluations", 8),
                                  history_size=10, line_search_fn="strong_wolfe",
                                  tolerance_grad=inverse.get("tolerance_grad", 1e-10),
                                  tolerance_change=inverse.get("tolerance_change", 1e-12))
    with torch.no_grad():
        initial_loss = float(objective(z, False))

    def state_row(iteration: int, loss: float) -> dict[str, Any]:
        values = parameter_values(z, config)
        return {"iteration": iteration, "forward_evaluations": count,
                "Gc_N_per_mm": float(values["Gc"].detach()),
                "l0_mm": float(values["l0"].detach()),
                "loss": loss * loss_scale, "relative_loss": loss * loss_scale / normalization}

    accepted.append(state_row(0, initial_loss))
    stop_reason = "maximum_iterations"
    for iteration in range(1, inverse["max_iterations"] + 1):
        accepted_z = z.detach().clone()
        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            if gradient_mode == "ad":
                loss = objective(z, True)
                loss.backward()
            else:
                step = inverse["fd_step"]
                with torch.no_grad():
                    loss = objective(z, False)
                    gradient = torch.zeros_like(z)
                    for index in range(z.numel()):
                        offset = torch.zeros_like(z)
                        offset[index] = step
                        gradient[index] = (objective(z + offset, False)
                                           - objective(z - offset, False)) / (2.0 * step)
                z.grad = gradient
            if not torch.isfinite(loss) or z.grad is None or not torch.isfinite(z.grad).all():
                raise RuntimeError("non-finite loss or gradient")
            evaluated.append(state_row(iteration, float(loss.detach())))
            return loss

        try:
            optimiser.step(closure)
            with torch.no_grad():
                loss_value = float(objective(z, False))
        except EvaluationLimit:
            with torch.no_grad():
                z.copy_(accepted_z)
            stop_reason = "maximum_evaluations"
            break
        row = state_row(iteration, loss_value)
        accepted.append(row)
        errors = [abs(row["Gc_N_per_mm" if name == "Gc" else "l0_mm"] - truth[name]) / truth[name]
                  for name in config["recover"]]
        loss_reduction = 1.0 - loss_value / initial_loss
        print(f"iteration {iteration}: Gc={row['Gc_N_per_mm']:.8g}, "
              f"l0={row['l0_mm']:.8g}, damage MSE={row['loss']:.4g}", flush=True)
        if (inverse.get("stop_on_synthetic_success", True)
                and max(errors) <= inverse["parameter_relative_tolerance"]
                and loss_reduction >= inverse["minimum_loss_reduction"]
                and math.sqrt(loss_value * loss_scale / normalization) <= inverse["maximum_relative_field_error"]):
            stop_reason = "synthetic_success"
            break
    values = {name: float(value.detach()) for name, value in parameter_values(z, config).items()}
    with torch.no_grad():
        final = run_forward(trial_mesh, config, values["Gc"], values["l0"], steps,
                            differentiable=False, dt=dt, record=True)
    save_fields(output, "recovered", trial_mesh, final)
    final_observations = observe(final.damage).detach()
    final_loss = float(torch.mean((final_observations - target_observations) ** 2))
    relative_field_error = math.sqrt(final_loss / normalization)
    errors = {name: abs(values[name] - truth[name]) / truth[name] for name in config["recover"]}
    reduction = 1.0 - final_loss / (initial_loss * loss_scale)
    success = (max(errors.values()) <= inverse["parameter_relative_tolerance"]
               and reduction >= inverse["minimum_loss_reduction"]
               and relative_field_error <= inverse["maximum_relative_field_error"])
    check_error = None
    observation_rows = [
        {"x_mm": float(point[0]), "y_mm": float(point[1]),
         "original_damage": float(original), "recovered_damage": float(recovered)}
        for point, original, recovered in zip(points, target_observations.cpu().numpy(),
                                              final_observations.cpu().numpy())
    ]
    write_csv(output / "observations.csv", observation_rows)
    write_csv(output / "history.csv", accepted)
    write_csv(output / "evaluation_history.csv", evaluated)
    write_csv(output / "parameter_history.csv", [
        {key: row[key] for key in ("iteration", "Gc_N_per_mm", "l0_mm")} for row in accepted])
    write_csv(output / "loss_history.csv", [
        {key: row[key] for key in ("iteration", "loss", "relative_loss")} for row in accepted])
    mesh_metadata = {
        label: {"nodes": mesh.n_nodes, "elements": mesh.n_elems,
                "crack_size_mm": profile[f"h_{label}_mm"]}
        for label, mesh in (("target", target_mesh), ("trial", trial_mesh))
    }
    summary = {
        "example": config["example"], "profile": profile_name,
        "success": bool(success), "stop_reason": stop_reason,
        "truth": truth, "initial": {name: spec["initial"] for name, spec in config["parameters"].items()},
        "recovered": values, "relative_parameter_error": errors,
        "initial_loss": initial_loss * loss_scale, "final_loss": final_loss,
        "loss_reduction": reduction, "relative_field_error": relative_field_error,
        "check_relative_field_error": check_error,
        "target_max_damage": float(target.damage.max()),
        "target_nodes_above_0_2": int((target.damage > 0.2).sum()),
        "steps": steps, "dt_s": dt, "final_time_s": steps * dt,
        "requested_time_s": requested_time,
        "time_relative_rounding_error": abs(steps * dt - requested_time) / requested_time,
        "observations": len(points), "forward_evaluations": count + 2,
        "gradient_mode": gradient_mode, "initial_gradient_check_passed": True,
        "parameterization": inverse.get("parameterization", "bounded_logit"),
        "loss_scaling": inverse.get("loss_scaling", "target_mse"),
        "loss_logging": "accepted_state_re_evaluation",
        "damage_update_interval": config["numerics"].get("damage_update_interval", 1),
        "exact_historical_curve_reproduction": False,
        "elapsed_s": time.perf_counter() - started, "mesh": mesh_metadata,
    }
    if core_hashes() != initial_core_hashes:
        raise RuntimeError("solver source changed during this run; rerun in a new directory")
    write_json(output / "summary.json", summary)
    config_text = yaml.safe_dump(config, sort_keys=False)
    (output / "config.yaml").write_text(config_text, encoding="utf-8")
    try:
        phast_version = version("phast")
    except PackageNotFoundError:
        phast_version = "uninstalled source checkout"
    versions = {"phast": phast_version, "torch": torch.__version__,
                "numpy": np.__version__, "python": platform.python_version()}
    units = {"length": "mm", "time": "s", "stress": "MPa", "Gc": "N/mm",
             "density": "tonne/mm^3", "damage": "1", "loss": "1"}
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "device": config["numerics"]["device"], "dtype": "float64",
                "threads": torch.get_num_threads(), "platform": platform.platform(),
                "versions": versions, "git": git_info(), "mesh": mesh_metadata,
                "units": units, "config": config, "elapsed_s": summary["elapsed_s"]}
    write_json(output / "run_metadata.json", metadata)
    script = example_dir.relative_to(ROOT).as_posix() + "/run.py"
    command = f"PYTHONPATH=src python {script} --config <resolved_config.yaml> --profile {profile_name} --gradient-mode {gradient_mode} --output-dir <new_directory>"
    write_json(output / "run_lockfile.json", {
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "resolved_config": config, "versions": versions, "git": metadata["git"],
        "profile": profile_name, "steps": steps, "dt_s": dt,
        "gradient_mode": gradient_mode,
        "core_sha256": initial_core_hashes,
    })
    from .plotting import plot_output
    plot_output(output)
    artifacts = sorted(path.name for path in output.iterdir() if path.is_file())
    write_json(output / "run_manifest.json", {
        "schema_version": 1, "example": config["example"], "status": "beta",
        "success": bool(success), "command": command, "git": metadata["git"],
        "config": config, "units": units, "versions": versions, "artifacts": artifacts,
    })
    return summary


def main(example_dir: Path) -> int:
    parser = argparse.ArgumentParser(description="Synthetic dynamic AT2 parameter recovery")
    parser.add_argument("--config", type=Path, default=example_dir / "config.yaml")
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--gradient-mode", choices=("ad", "fd"), default="ad")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.validate_only:
        print(f"Valid {config['example']} configuration; profile={args.profile}")
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for a run")
    output = args.output_dir.resolve()
    if output == ROOT / "examples" or ROOT / "examples" in output.parents:
        parser.error("run outputs must be outside examples/; use a new runs/ directory")
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    try:
        summary = run_case(example_dir, config, args.profile, output, args.gradient_mode)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        write_json(output / "failure.json", {"success": False, "error": str(exc)})
        print(f"Run stopped: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 2
