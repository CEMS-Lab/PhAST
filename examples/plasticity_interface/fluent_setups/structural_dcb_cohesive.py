"""Fluent setup for the retained structural DCB cohesive result bundle."""
from __future__ import annotations

from pathlib import Path

from phast import Problem


ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "examples" / "plasticity_interface" / "results" / "structural_dcb_cohesive"


def build_problem() -> Problem:
    return (
        Problem("promoted_structural_dcb_cohesive")
        .geometry(
            "dcb_structured_cohesive",
            length=6.0,
            arm_height=0.6,
            nx=16,
            ny_per_arm=2,
            initial_crack_elements=4,
        )
        .region("bulk", kind="domain")
        .region("bonded_interface", kind="interface", y=0.0)
        .material(
            "linear_elastic_with_bilinear_cohesive_interface",
            region="bulk",
            E=5_000.0,
            nu=0.30,
            cohesive_k_n=2_000.0,
            cohesive_k_t=2_000.0,
            cohesive_sigma_max=5.0,
            cohesive_delta_c=0.080,
        )
        .boundary_condition("fix", region="right_clamp", dof="xy", value=0.0)
        .boundary_condition("prescribe", region="top_load", dof="y", value=0.15)
        .boundary_condition("prescribe", region="bottom_load", dof="y", value=-0.15)
        .analysis_step(
            "mode_i_opening",
            kind="cohesive_validation",
            controls={"openings": 11, "max_opening": 0.30},
        )
        .solver("cohesive_validation", example="structural_dcb_cohesive")
        .outputs(
            fields=["displacement", "cohesive_damage"],
            histories=["reaction_force", "energy", "solver_telemetry"],
        )
        .device("cpu")
    )


if __name__ == "__main__":
    print(build_problem().to_spec())
