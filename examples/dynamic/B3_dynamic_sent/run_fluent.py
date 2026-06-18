"""Run the B3 dynamic SENT example without a YAML configuration."""

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
    problem = phast.Problem("Dynamic SENT")
    problem.config.example = "phast.examples.dynamic_sent_tension.run"
    problem.config.reference = "Borden et al. (2012), CMAME"
    problem.config.geometry = GeometryConfig(
        units="mm",
        primitives={
            "plate": {"type": "rectangle", "origin": [0, 0], "size": [40.0, 40.0]},
            "notch": {
                "type": "polygon",
                "vertices": [[0.0, 20.001], [20.0, 20.0], [0.0, 19.999]],
            },
        },
        domain={"base": "plate", "subtract": ["notch"]},
        named_groups={
            "left": {"region": {"type": "rectangle", "origin": [-0.01, 0.0], "size": [0.02, 40.0]}},
            "right": {"region": {"type": "rectangle", "origin": [39.99, 0.0], "size": [0.02, 40.0]}},
            "top": {"region": {"type": "rectangle", "origin": [0.0, 39.99], "size": [40.0, 0.02]}},
            "bottom": {"region": {"type": "rectangle", "origin": [0.0, -0.01], "size": [40.0, 0.02]}},
        },
        mesh={"element_size": {"default": 2.0, "refined": [{"primitive": "notch", "size": 0.25, "margin": 1.5}]}},
    )
    problem.config.material = MaterialConfig(
        E=32000.0,
        nu=0.20,
        Gc=3.0e-3,
        l0=0.5,
        rho=2.45e-9,
        energy_split="spectral",
        pf_model="AT2",
        eta_residual=1.0e-7,
    )
    problem.config.boundary_conditions = [
        BoundaryConditionEntry(nodes="left", type="fix", component=0),
        BoundaryConditionEntry(nodes="right", type="fix", component=0),
        BoundaryConditionEntry(nodes="top", type="prescribe", component=1, value=0.002),
        BoundaryConditionEntry(nodes="bottom", type="prescribe", component=1, value=-0.002),
    ]
    problem.config.loading = LoadingConfig(
        protocol="simple",
        t_total=1.0e-04,
        ramp_type="smooth_step",
        t_ramp=2.0e-05,
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
    parser = argparse.ArgumentParser(description="Run B3 dynamic SENT through the Python API.")
    parser.add_argument("--output-dir", default="examples/dynamic/B3_dynamic_sent/run_fluent")
    parser.add_argument("--num-steps", type=int, default=None, help="Optional short-run explicit step count.")
    args = parser.parse_args()
    build_problem(args.num_steps).run(output_dir=args.output_dir, return_result=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
