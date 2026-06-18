"""Fluent setup for the retained PF-CZM uniaxial result bundle."""
from __future__ import annotations

from pathlib import Path

from phast import Problem


ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = ROOT / "examples" / "plasticity_interface_beta" / "results" / "pfczm_uniaxial_strength"


def build_problem() -> Problem:
    return (
        Problem("promoted_pfczm_uniaxial_strength")
        .geometry("structured_bar", width=1.0, height=0.25, nx=20, ny=5)
        .region("body", kind="domain")
        .material(
            "pfczm_wu",
            region="body",
            E=3_000.0,
            nu=0.30,
            Gc=0.12,
            sigma_ts=3.0,
            pf_model="PFCZM",
            pfczm_softening="linear",
        )
        .boundary_condition("fix", region="left", dof="x", value=0.0)
        .boundary_condition("prescribe", region="right", dof="x", value=2.1e-3)
        .analysis_step(
            "uniaxial_strength_sweep",
            kind="pfczm_validation",
            controls={"n_steps": 32, "l0_values": [0.08, 0.12, 0.18]},
        )
        .solver("pfczm_validation", example="pfczm_uniaxial_strength")
        .outputs(
            fields=["damage"],
            histories=["load_displacement", "energy", "solver_telemetry"],
        )
        .device("cpu")
    )


if __name__ == "__main__":
    print(build_problem().to_spec())
