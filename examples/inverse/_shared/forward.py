"""Dynamic AT2 forward model and fixed-position observation operators."""

from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phast.boundary_conditions import BoundaryConditions
from phast.device import DeviceContext
from phast.material import Material
from phast.mesh import FEMMesh
from phast.mesh_generator import rectangular_sent
from phast.staggered_solver import SolverConfig, StaggeredSolver


@dataclass
class ForwardResult:
    damage: torch.Tensor
    displacement: torch.Tensor
    dt: float
    history: list[dict[str, Any]]
    snapshots: np.ndarray


@dataclass(frozen=True)
class Sampler:
    points: np.ndarray
    node_ids: torch.Tensor
    weights: torch.Tensor

    def sample(self, field: torch.Tensor) -> torch.Tensor:
        return (field[self.node_ids] * self.weights).sum(dim=1)


def positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def positive_int(value: Any, name: str) -> int:
    number = positive(value, name)
    if int(number) != number:
        raise ValueError(f"{name} must be an integer")
    return int(number)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if config.get("example") not in {"scalar_toughness", "scalar_toughness_alumina"}:
        raise ValueError("this traction forward model is only for scalar_toughness")
    geometry = config["geometry"]
    for key in ("width_mm", "height_mm", "notch_length_mm"):
        positive(geometry[key], key)
    if not 0.0 < geometry["notch_length_mm"] < geometry["width_mm"]:
        raise ValueError("notch length must lie inside the plate")
    if not isinstance(geometry["branching_refinement"], bool):
        raise ValueError("branching_refinement must be boolean")
    material = config["material"]
    for key in ("E_MPa", "rho_tonne_per_mm3", "eta_residual"):
        positive(material[key], key)
    if not -1.0 < material["nu"] < 0.5:
        raise ValueError("nu must lie between -1 and 0.5")
    if material["energy_split"] != "spectral" or material["pf_model"] != "AT2":
        raise ValueError("these examples require spectral AT2")
    if config["loading"]["type"] != "constant_traction":
        raise ValueError("only constant equal-and-opposite traction is supported")
    positive(config["loading"]["traction_MPa"], "traction_MPa")
    numerics = config["numerics"]
    for key in ("damage_tolerance", "adjoint_relative_tolerance"):
        positive(numerics[key], key)
    positive_int(numerics["damage_max_iterations"], "damage_max_iterations")
    positive_int(numerics["threads"], "threads")
    positive_int(numerics.get("damage_update_interval", 1), "damage_update_interval")
    if not isinstance(numerics.get("use_multigrid", False), bool):
        raise ValueError("use_multigrid must be boolean")
    if not 0.0 < positive(numerics["dt_safety"], "dt_safety") <= 1.0:
        raise ValueError("dt_safety must be at most 1")
    if numerics["history_update"] != "hard_max" or numerics["bounds"] != "post_clamp":
        raise ValueError("these examples require hard_max and post_clamp")
    for name in ("Gc", "l0"):
        spec = config["parameters"][name]
        for key in ("truth", "initial"):
            positive(spec[key], f"{name}.{key}")
        if name in config["recover"]:
            lower, upper = spec["bounds"]
            positive(lower, f"{name}.lower")
            positive(upper, f"{name}.upper")
            if not lower < spec["truth"] < upper or not lower < spec["initial"] < upper:
                raise ValueError(f"{name} truth and initial must lie inside bounds")
        elif spec["initial"] != spec["truth"]:
            raise ValueError(f"fixed {name} must use its known generating value")
    expected = ["Gc"]
    if config["recover"] != expected:
        raise ValueError(f"recover must be {expected}")
    if all(config["parameters"][name]["truth"] == config["parameters"][name]["initial"]
           for name in config["recover"]):
        raise ValueError("at least one recovered parameter must start away from its truth")
    if set(config["profiles"]) != {"quick", "full"}:
        raise ValueError("quick and full profiles are required")
    for profile in config["profiles"].values():
        for key in ("h_target_mm", "h_trial_mm", "h_coarse_mm"):
            positive(profile[key], key)
        if profile["h_target_mm"] != profile["h_trial_mm"]:
            raise ValueError("same-mesh recovery requires equal mesh sizes")
        positive_int(profile["steps"], "steps")
    inverse = config["inverse"]
    if inverse.get("parameterization", "bounded_logit") not in {"bounded_logit", "log"}:
        raise ValueError("parameterization must be bounded_logit or log")
    if inverse.get("loss_scaling", "target_mse") not in {"target_mse", "none"}:
        raise ValueError("loss_scaling must be target_mse or none")
    if not isinstance(inverse.get("stop_on_synthetic_success", True), bool):
        raise ValueError("stop_on_synthetic_success must be boolean")
    for key, default in (("learning_rate", 1.0), ("tolerance_grad", 1e-10),
                         ("tolerance_change", 1e-12)):
        positive(inverse.get(key, default), key)
    positive_int(inverse.get("max_inner_evaluations", 8), "max_inner_evaluations")
    for key in ("max_iterations", "inner_iterations", "max_evaluations"):
        positive_int(inverse[key], key)
    for key in ("fd_step", "gradient_relative_tolerance", "parameter_relative_tolerance",
                "minimum_loss_reduction", "maximum_relative_field_error"):
        positive(inverse[key], key)
    for key in ("parameter_relative_tolerance", "minimum_loss_reduction"):
        if inverse[key] >= 1.0:
            raise ValueError(f"{key} must be less than 1")


def build_mesh(path: Path, config: dict[str, Any], size: float,
               coarse_size: float) -> FEMMesh:
    geometry = config["geometry"]
    rectangular_sent(
        str(path), W=geometry["width_mm"], H=geometry["height_mm"],
        a=geometry["notch_length_mm"], l0=config["parameters"]["l0"]["truth"],
        h_crack=size, h_coarse=coarse_size,
        branching=geometry["branching_refinement"], verbose=False,
    )
    mesh = FEMMesh(str(path), device=config["numerics"]["device"], dtype=torch.float64)
    mesh.identify_boundaries()
    return mesh


def build_solver(mesh: FEMMesh, config: dict[str, Any], gc: float | torch.Tensor,
                 l0: float | torch.Tensor, *, differentiable: bool,
                 dt: float | None = None) -> StaggeredSolver:
    gc_value = float(gc.detach()) if isinstance(gc, torch.Tensor) else float(gc)
    l0_value = float(l0.detach()) if isinstance(l0, torch.Tensor) else float(l0)
    positive(gc_value, "Gc")
    positive(l0_value, "l0")
    mat = config["material"]
    material = Material(
        E=mat["E_MPa"], nu=mat["nu"], rho=mat["rho_tonne_per_mm3"],
        Gc=gc_value, l0=l0_value, eta_residual=mat["eta_residual"],
        energy_split="spectral", pf_model="AT2", plane_stress=False,
        gamma_correction=False,
    )
    bcs = BoundaryConditions(mesh.n_nodes, device=mesh.device, dtype=mesh.dtype)
    bcs.add_neumann(mesh.node_sets["top"], [0.0, 1.0])
    bcs.add_neumann(mesh.node_sets["bottom"], [0.0, -1.0])
    bcs.fix(mesh.node_sets["left"], 0)
    numerics = config["numerics"]
    kwargs: dict[str, Any] = {}
    if "damage_reaction_mass" in SolverConfig.__dataclass_fields__:
        kwargs["damage_reaction_mass"] = "consistent"
    if "damage_true_residual_audit" in SolverConfig.__dataclass_fields__:
        kwargs["damage_true_residual_audit"] = True
    cfg = SolverConfig(
        solver_type="explicit", time_integrator="central_difference",
        dt_safety=numerics["dt_safety"], differentiable=differentiable,
        damage_every=numerics.get("damage_update_interval", 1), preconditioner="jacobi",
        use_multigrid=numerics.get("use_multigrid", False),
        damage_tol=numerics["damage_tolerance"],
        damage_max_iter=numerics["damage_max_iterations"],
        bounds_method="post_clamp", H_update_method="hard_max", **kwargs,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        context = DeviceContext(device=str(mesh.device), profile=False, compile_solvers=False)
        solver = StaggeredSolver(mesh, material, bcs, config=cfg, ctx=context)
    solver.f_ext = config["loading"]["traction_MPa"] * bcs.get_neumann_forces(mesh)
    if dt is not None:
        positive(dt, "dt")
        if dt > float(solver.dt) * (1.0 + 1.0e-12):
            raise ValueError("requested time step exceeds this mesh's CFL limit")
        solver.dt = dt
        solver.mechanics.dt = dt
    if isinstance(gc, torch.Tensor):
        solver.diff_Gc = gc
    if isinstance(l0, torch.Tensor):
        solver.diff_l0 = l0
    solver.damage_solver.adjoint_pcg_rel_tol = numerics["adjoint_relative_tolerance"]
    solver.damage_solver.adjoint_pcg_abs_tol = 1.0e-14
    return solver


def run_forward(mesh: FEMMesh, config: dict[str, Any], gc: float | torch.Tensor,
                l0: float | torch.Tensor, steps: int, *, differentiable: bool,
                dt: float | None = None, record: bool = False) -> ForwardResult:
    positive_int(steps, "steps")
    solver = build_solver(mesh, config, gc, l0, differentiable=differentiable, dt=dt)
    history: list[dict[str, Any]] = []
    snapshots: list[np.ndarray] = []
    frame_steps = set(np.linspace(1, steps, min(20, steps), dtype=int))
    if record:
        snapshots.append(solver.d.detach().cpu().numpy().copy())
    for step in range(1, steps + 1):
        previous = solver.d.detach()
        solver.step_full()
        damage_solver = solver.damage_solver
        interval = config["numerics"].get("damage_update_interval", 1)
        updated = step <= 5 or step % interval == 0
        convergence = getattr(damage_solver, "last_converged", None) if updated else None
        if updated and convergence is None:
            raise RuntimeError("damage convergence status is unavailable in this solver version")
        if updated and convergence is False:
            raise RuntimeError(f"damage linear solve did not converge at step {step}")
        detached = solver.d.detach()
        if not torch.isfinite(detached).all() or not torch.isfinite(solver.u.detach()).all():
            raise RuntimeError(f"non-finite field at step {step}")
        if detached.min() < -1.0e-10 or detached.max() > 1.0 + 1.0e-10:
            raise RuntimeError(f"damage outside [0,1] at step {step}")
        if (detached - previous).min() < -1.0e-10:
            raise RuntimeError(f"damage decreased at step {step}")
        if record:
            history.append({
                "step": step, "time_s": step * float(solver.dt),
                "max_damage": float(detached.max()),
                "damage_updated": updated,
                "damage_iterations": int(getattr(damage_solver, "last_iter", 0)) if updated else 0,
                "damage_converged": convergence,
                "damage_relative_residual": float(getattr(damage_solver, "last_relative_residual", 0.0)) if updated else None,
            })
            if step in frame_steps:
                snapshots.append(detached.cpu().numpy().copy())
    return ForwardResult(solver.d, solver.u, float(solver.dt), history, np.asarray(snapshots))


def build_sampler(mesh: FEMMesh, points: np.ndarray) -> Sampler:
    from matplotlib.tri import Triangulation

    nodes = mesh.nodes.detach().cpu().numpy()[:, :2]
    elements = mesh.elements.detach().cpu().numpy()
    triangles = Triangulation(nodes[:, 0], nodes[:, 1], elements)
    ids = triangles.get_trifinder()(points[:, 0], points[:, 1])
    if np.any(ids < 0):
        raise ValueError("observation point outside the mesh; extrapolation is not allowed")
    node_ids = elements[ids]
    a, b, c = (nodes[node_ids[:, index]] for index in range(3))
    denominator = ((b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0])
                   + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1]))
    if np.any(np.abs(denominator) < 1.0e-14):
        raise ValueError("observation operator encountered a degenerate triangle")
    w0 = ((b[:, 1] - c[:, 1]) * (points[:, 0] - c[:, 0])
          + (c[:, 0] - b[:, 0]) * (points[:, 1] - c[:, 1])) / denominator
    w1 = ((c[:, 1] - a[:, 1]) * (points[:, 0] - c[:, 0])
          + (a[:, 0] - c[:, 0]) * (points[:, 1] - c[:, 1])) / denominator
    weights = np.column_stack((w0, w1, 1.0 - w0 - w1))
    if weights.min() < -2.0e-9 or weights.max() > 1.0 + 2.0e-9:
        raise ValueError("invalid barycentric observation weights")
    return Sampler(points.copy(), torch.as_tensor(node_ids, device=mesh.device),
                   torch.as_tensor(weights, device=mesh.device, dtype=mesh.dtype))
