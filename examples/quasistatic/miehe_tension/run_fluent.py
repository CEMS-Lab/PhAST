"""Run the Miehe SENT quasi-static fracture example without a YAML deck.

This script is intentionally close to ``config.yaml``. It demonstrates the
Python authoring path for users who want to construct a PhAST problem directly
with the fluent API, while keeping the checked-in YAML deck as the canonical
reproducibility input.
"""

from __future__ import annotations

import argparse

import phast


def loading_phases(num_steps: int) -> str:
    """Return a valid displacement schedule for full or shortened runs."""
    if num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if num_steps < 50:
        return f"0.005:{num_steps}"
    if num_steps == 50:
        return "0.005:50"
    return f"0.005:50,0.008:{num_steps - 50}"


def build_problem(num_steps: int) -> phast.Problem:
    """Build the Miehe single-edge-notched tension benchmark."""
    return (
        phast.Problem("Miehe SENT")
        .geometry(
            "miehe_tension",
            L=1.0,
            a=0.5,
            l0=0.015,
            h_crack=0.001875,
            h_coarse=0.05,
        )
        .region("body", kind="domain")
        .region("bottom", from_mesh="bottom")
        .region("top", from_mesh="top")
        .material(
            "glass",
            region="body",
            E=210000.0,
            nu=0.3,
            Gc=2.7,
            l0=0.015,
            rho=7.8e-09,
            eta_residual=1.0e-07,
            energy_split="isotropic",
            pf_model="AT2",
            plane_stress=False,
        )
        .boundary_condition("fix", region="bottom", dof="x", name="clamp_x")
        .boundary_condition("fix", region="bottom", dof="y", name="clamp_y")
        .boundary_condition(
            "displacement",
            region="top",
            dof="y",
            value=1.0,
            name="pull_top",
        )
        .analysis_step(
            "load",
            kind="quasi_static",
            controls={
                "protocol": "cyclic",
                "cyclic_phases": loading_phases(num_steps),
                "num_steps": num_steps,
                "dt": 1.0,
            },
            active_boundary_conditions=["clamp_x", "clamp_y", "pull_top"],
        )
        .solver(
            "quasi_static",
            stagger_tol=1.0e-08,
            max_stagger=500,
            stagger_criterion="relative",
            anderson_depth=0,
            adaptive_stagger_tol=False,
            use_multigrid=True,
            preconditioner="jacobi",
            damage_tol=1.0e-06,
            static_tol=1.0e-08,
            bounds_method="post_clamp",
            damage_max_iter=5000,
            static_max_iter=5000,
            backend="auto",
            fail_on_mechanics_nonconvergence=False,
            eta_residual=1.0e-07,
        )
        .outputs(
            fields=[{"name": "trajectory", "every": 1, "format": "zarr"}],
            histories=[{"name": "reaction_force", "region": "bottom", "dof": "y"}],
            plots=True,
            profile=True,
            gif=True,
            gif_frames=150,
            gif_fields="damage",
            animation_format="mp4",
            print_every=1,
        )
        .device("cpu")
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Miehe SENT benchmark through the fluent Python API."
    )
    parser.add_argument(
        "--output-dir",
        default="examples/quasistatic/miehe_tension/run_fluent",
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=350,
        help="Number of quasi-static load steps. Use a small value for a quick check.",
    )
    args = parser.parse_args()

    problem = build_problem(args.num_steps)
    problem.run(output_dir=args.output_dir, return_result=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
