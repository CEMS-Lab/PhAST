"""Run the B6 perforated 30-hole plate example without a YAML configuration."""

from __future__ import annotations

import argparse

import phast
from phast.config import BoundaryConditionEntry, GeometryConfig, LoadingConfig, MaterialConfig, OutputConfig, SolverSettings


def _apply_num_steps(problem: phast.Problem, num_steps: int | None) -> phast.Problem:
    if num_steps is not None:
        if num_steps <= 0:
            raise ValueError("--num-steps must be positive")
        problem.config.loading.num_steps = num_steps
    return problem


def build_problem(num_steps: int | None = None) -> phast.Problem:
    problem = phast.Problem("Perforated Plate — 30 holes")
    problem.config.example = "phast.examples.dynamic_perforated_plate.run"
    problem.config.reference = "Bleyer et al. (2017), Sec 4.2"
    problem.config.geometry = GeometryConfig(
        type="perforated_sent",
        parameters={
            "W": 32.0,
            "H": 16.0,
            "a": 4.0,
            "h_crack": 0.05,
            "h_coarse": 1.0,
            "n_holes": 30,
            "hole_spacing": 0.9,
        },
    )
    problem.config.material = MaterialConfig(
        E=3090.0,
        nu=0.35,
        Gc=0.3,
        l0=0.1,
        rho=1.18e-9,
        energy_split="amor",
        pf_model="AT1",
        plane_stress=True,
        eta_residual=1.0e-7,
    )
    problem.config.boundary_conditions = [
        BoundaryConditionEntry(nodes="left", type="fix", component=0),
        BoundaryConditionEntry(nodes="top", type="prescribe", component=1, value=1.0),
        BoundaryConditionEntry(nodes="bottom", type="prescribe", component=1, value=-1.0),
    ]
    problem.config.loading = LoadingConfig(
        protocol="two_step_prestrain",
        prestrain_displacement=0.05,
        t_total=7.5e-05,
        ramp_type="constant",
        num_steps=0,
    )
    problem.config.solver = SolverSettings(solver_type="explicit", dt_safety=0.8)
    problem.config.output = OutputConfig(
        trajectory=True,
        h5=True,
        trajectory_format="zarr",
        h5_every=20,
        fast=True,
        print_every=100,
    )
    return _apply_num_steps(problem, num_steps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B6 30-hole perforated plate through the Python API.")
    parser.add_argument("--output-dir", default="examples/dynamic/B6_perforated_30holes/run_fluent")
    parser.add_argument("--num-steps", type=int, default=None, help="Optional short-run explicit step count.")
    args = parser.parse_args()
    build_problem(args.num_steps).run(output_dir=args.output_dir, return_result=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
