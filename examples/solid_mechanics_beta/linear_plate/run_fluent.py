"""Fluent authoring companion for the linear-plate YAML tutorial."""
from __future__ import annotations

import argparse

from fluent_setup import build_problem
from phast.workflow import run_problem_spec, validate_problem_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Run the fluent problem into --output-dir.")
    parser.add_argument("--output-dir", default="runs/linear_plate")
    args = parser.parse_args()

    problem = build_problem()
    issues = validate_problem_spec(problem.to_spec())
    if issues:
        detail = "; ".join(issue.message for issue in issues)
        raise SystemExit(f"invalid fluent setup: {detail}")
    print("OK: fluent setup compiles to the workflow contract.")
    print("Reference deck: examples/solid_mechanics_beta/linear_plate/config.yaml")
    if args.run:
        raise SystemExit(run_problem_spec(problem.to_spec(), output_dir=args.output_dir))


if __name__ == "__main__":
    main()
