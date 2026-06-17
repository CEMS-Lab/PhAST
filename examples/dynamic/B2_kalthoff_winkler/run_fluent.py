"""Run the B2 Kalthoff-Winkler impact example without a YAML deck."""

from __future__ import annotations

import argparse

import phast
from phast.config import (
    BoundaryConditionEntry,
    GeometryConfig,
    LoadingConfig,
    MaterialConfig,
    OutputConfig,
    SolverSettings,
)


def _apply_num_steps(problem: phast.Problem, num_steps: int | None) -> phast.Problem:
    if num_steps is not None:
        if num_steps <= 0:
            raise ValueError("--num-steps must be positive")
        problem.config.loading.num_steps = num_steps
    return problem


def build_problem(num_steps: int | None = None) -> phast.Problem:
    """Build the Kalthoff-Winkler impact setup through Python."""
    problem = phast.Problem("Kalthoff-Winkler Impact")
    problem.config.example = "phast.examples.dynamic_kalthoff_shear.run"
    problem.config.reference = "Borden et al. (2012), CMAME"
    problem.config.geometry = GeometryConfig(
        type="kalthoff_winkler",
        parameters={
            "W": 100.0,
            "H": 100.0,
            "a": 50.0,
            "h_crack": 0.25,
            "h_coarse": 5.0,
        },
    )
    problem.config.material = MaterialConfig(
        E=190000.0,
        nu=0.30,
        Gc=22.13,
        l0=0.195,
        rho=8.0e-9,
        energy_split="spectral",
        pf_model="AT2",
        eta_residual=1.0e-7,
        plane_stress=False,
    )
    problem.config.boundary_conditions = [
        BoundaryConditionEntry(
            nodes="left_impact",
            type="prescribe",
            component=0,
            value=1.0,
        ),
        BoundaryConditionEntry(nodes="bottom", type="symmetry", axis="y"),
    ]
    problem.config.loading = LoadingConfig(
        protocol="simple",
        t_total=1.0e-04,
        ramp_type="velocity_impact",
        v0=16.5,
        t_ramp=1.0e-06,
        num_steps=0,
    )
    problem.config.solver = SolverSettings(
        solver_type="explicit",
        dt_safety=0.8,
        damage_every=1,
    )
    problem.config.output = OutputConfig(
        trajectory=True,
        h5=True,
        trajectory_format="zarr",
        h5_every=50,
        fast=True,
        print_every=100,
    )
    return _apply_num_steps(problem, num_steps)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run B2 Kalthoff-Winkler through the Python API."
    )
    parser.add_argument(
        "--output-dir",
        default="examples/dynamic/B2_kalthoff_winkler/run_fluent",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Optional short-run explicit step count.",
    )
    args = parser.parse_args()
    build_problem(args.num_steps).run(output_dir=args.output_dir, return_result=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
