"""Fluent authoring companion for the neo-Hookean plate YAML tutorial."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import phast
from phast.workflow import validate_problem_spec


def build_problem() -> phast.Problem:
    return (
        phast.Problem("Neo-Hookean cantilever")
        .geometry("structured_grid", nx=20, ny=10, length=1.0, height=0.2)
        .region("body", kind="domain")
        .material("rubber", region="body", E=2.1e11, nu=0.3)
        .analysis_step(
            "load",
            kind="solid_mechanics",
            controls={
                "load_steps": 5,
                "target_linear_tip_displacement_fraction": 0.05,
                "load_scale": 0.5,
            },
        )
        .solver("solid_mechanics", example="solid_mechanics.neohookean_plate")
        .outputs(fields=["displacement", "von_mises", "strain_energy", "jacobian"],
                 histories=["response"], plots=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Run the fluent problem into --output-dir.")
    parser.add_argument("--output-dir", default="runs/neohookean_plate")
    args = parser.parse_args()

    problem = build_problem()
    issues = validate_problem_spec(problem.to_spec())
    if issues:
        detail = "; ".join(issue.message for issue in issues)
        raise SystemExit(f"invalid fluent setup: {detail}")
    print("OK: fluent setup compiles to the workflow contract.")
    print("Canonical deck: examples/solid_mechanics/neohookean_plate/config.yaml")
    if args.run:
        problem.run(output_dir=args.output_dir, return_result=True)


if __name__ == "__main__":
    main()
