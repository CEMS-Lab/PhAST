"""Run the B5 PMMA branching example without a YAML deck."""

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
    problem = phast.Problem("PMMA Dynamic Crack Branching")
    problem.config.example = "phast.examples.dynamic_branching_pmma.run"
    problem.config.reference = "Bleyer, Roux-Langlois & Molinari (2017), Comput. Mech."
    problem.config.geometry = GeometryConfig(
        units="mm",
        primitives={
            "plate": {"type": "rectangle", "origin": [0, 0], "size": [32.0, 16.0]},
            "notch": {"type": "polygon", "vertices": [[0.0, 8.01], [4.0, 8.0], [0.0, 7.99]]},
        },
        domain={"base": "plate", "subtract": ["notch"]},
        named_groups={
            "left": {"region": {"type": "rectangle", "origin": [-0.005, 0.0], "size": [0.01, 16.0]}},
            "top": {"region": {"type": "rectangle", "origin": [0.0, 15.995], "size": [32.0, 0.01]}},
            "bottom": {"region": {"type": "rectangle", "origin": [0.0, -0.005], "size": [32.0, 0.01]}},
        },
        mesh={
            "element_size": {
                "default": 1.0,
                "refined": [
                    {"primitive": "notch", "size": 0.02, "margin": 0.5},
                    {"region": {"type": "box", "x": [-1.0, 32.0], "y": [0.0, 16.0]}, "size": 0.02, "thickness": 1.0},
                ],
            }
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
        coupled_prestrain=True,
        t_total=7.5e-05,
        ramp_type="constant",
        num_steps=0,
    )
    problem.config.solver = SolverSettings(solver_type="explicit", dt_safety=0.8, damage_every=1)
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
    parser = argparse.ArgumentParser(description="Run B5 PMMA branching through the Python API.")
    parser.add_argument("--output-dir", default="examples/dynamic/B5_pmma_branching/run_fluent")
    parser.add_argument("--num-steps", type=int, default=None, help="Optional short-run explicit step count.")
    args = parser.parse_args()
    build_problem(args.num_steps).run(output_dir=args.output_dir, return_result=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
