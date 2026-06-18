"""Fluent setup for the retained J2 validation result bundle."""
from __future__ import annotations

from pathlib import Path

from phast import Problem


ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "examples" / "plasticity_interface_beta" / "results" / "j2_validation"


def build_problem() -> Problem:
    return (
        Problem("promoted_j2_validation")
        .geometry("structured_bar", length=1.0, height=0.2, nx=24, ny=6)
        .region("body", kind="domain")
        .material(
            "j2_isotropic",
            region="body",
            E=210_000.0,
            nu=0.30,
            yield_stress=250.0,
            hardening_modulus=5_000.0,
            hardening_type="linear_iso",
            plane_stress=True,
        )
        .boundary_condition("fix", region="left", dof="xy", value=0.0)
        .boundary_condition("prescribe", region="right", dof="x", value=4.0e-3)
        .analysis_step(
            "cyclic_axial_loading",
            kind="plasticity_validation",
            controls={"n_load": 48, "n_unload": 18},
        )
        .solver("plasticity_validation", example="j2_validation")
        .outputs(
            fields=["displacement", "stress", "von_mises", "plastic_strain"],
            histories=["reaction_force", "solver_telemetry"],
        )
        .device("cpu")
    )


if __name__ == "__main__":
    print(build_problem().to_spec())
