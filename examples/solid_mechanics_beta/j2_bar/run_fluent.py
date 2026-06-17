"""Fluent authoring companion for the J2 bar YAML tutorial."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import phast
from phast.workflow import validate_problem_spec


def build_problem() -> phast.Problem:
    return (
        phast.Problem("J2 plasticity bar")
        .geometry(
            "structured_grid",
            nx=18,
            ny=6,
            length=1.0,
            height=0.25,
            waist_depth=0.35,
            waist_width_fraction=0.18,
        )
        .region("body", kind="domain")
        .material(
            "j2_linear_hardening",
            region="body",
            E=210000.0,
            nu=0.3,
            sigma_y0=250.0,
            hardening_modulus=5000.0,
        )
        .analysis_step(
            "load",
            kind="solid_mechanics",
            controls={"n_steps": 30, "max_strain_xx": 4.5e-3},
        )
        .solver("solid_mechanics", example="solid_mechanics.j2_bar",
                backend="auto")
        .outputs(fields=["displacement", "von_mises", "equivalent_plastic_strain"],
                 histories=["response"], plots=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Run the fluent problem into --output-dir.")
    parser.add_argument("--output-dir", default="runs/j2_bar")
    args = parser.parse_args()

    problem = build_problem()
    issues = validate_problem_spec(problem.to_spec())
    if issues:
        detail = "; ".join(issue.message for issue in issues)
        raise SystemExit(f"invalid fluent setup: {detail}")
    print("OK: fluent setup compiles to the workflow contract.")
    print("Canonical deck: examples/solid_mechanics_beta/j2_bar/config.yaml")
    if args.run:
        problem.run(output_dir=args.output_dir, return_result=True)


if __name__ == "__main__":
    main()
