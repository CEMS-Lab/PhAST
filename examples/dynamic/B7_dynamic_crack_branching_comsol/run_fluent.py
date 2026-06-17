"""Run the B7 dynamic crack-branching COMSOL cross-check without YAML."""

from __future__ import annotations

import argparse

import phast
from phast.config import (
    BoundaryConditionEntry,
    GeometryConfig,
    InitialConditionsConfig,
    LoadingConfig,
    MaterialConfig,
    OutputConfig,
    SolverSettings,
    DeviceConfig,
)


def _apply_num_steps(problem: phast.Problem, num_steps: int | None) -> phast.Problem:
    if num_steps is not None:
        if num_steps <= 0:
            raise ValueError("--num-steps must be positive")
        problem.config.loading.num_steps = num_steps
    return problem


def build_problem(num_steps: int | None = None) -> phast.Problem:
    problem = phast.Problem("COMSOL Dynamic Crack Branching (PMMA-equiv.)")
    problem.config.example = None
    problem.config.reference = "COMSOL 6.4 Application Library 'Dynamic Crack Branching' (PDFs retained in the private development archive)"
    problem.config.acceptance = {
        "status": "beta",
        "reference_result": "Ren 2019 dynamic branching onset near 68.2 us; COMSOL Application Library 33 us retained as secondary vendor context",
        "required_outputs": [
            "run_lockfile.json",
            "config.yaml",
            "training_data.zarr",
            "energy.csv",
            "crack_tip.csv",
            "compare_report.txt",
            "damage_final.png",
        ],
        "metrics": {
            "branch_onset": {
                "target": 68.2,
                "tolerance": "20%",
                "units": "us",
            },
            "final_morphology": {
                "target": "clean Y-shaped branch by 75-80 us",
                "tolerance": "visual/morphology comparator, no multiple spurious branches",
            },
            "elastic_energy_peak": {
                "target": "COMSOL half-plate value doubled for full-plate convention",
                "tolerance": "same domain convention must be stated in compare report",
            },
        },
        "notes": (
            "Root config is the stable public B7 entry point and now matches COMSOL's "
            "AT1 plus Amor/volumetric-deviatoric split. Issue #299 is resolved by using "
            "the published Ren/Borden-style timing window as the default validation target "
            "and treating the COMSOL 33 us event as secondary vendor context. Exact COMSOL "
            "timing parity remains diagnostic provenance, not the release acceptance gate.\n"
        ),
    }
    problem.config.geometry = GeometryConfig(
        units="mm",
        primitives={
            "plate": {"type": "rectangle", "origin": [0, 0], "size": [100.0, 40.0]},
            "notch": {
                "type": "polygon",
                "vertices": [[0.0, 20.01], [50.0, 20.0], [0.0, 19.99]],
            },
        },
        domain={"base": "plate", "subtract": ["notch"]},
        named_groups={
            "left": {"region": {"type": "rectangle", "origin": [-0.005, 0.0], "size": [0.01, 40.0]}},
            "top": {"region": {"type": "rectangle", "origin": [0.0, 39.995], "size": [100.0, 0.01]}},
            "bottom": {"region": {"type": "rectangle", "origin": [0.0, -0.005], "size": [100.0, 0.01]}},
        },
        mesh={
            "element_size": {
                "default": 1.0,
                "refined": [
                    {"primitive": "notch", "size": 0.125, "margin": 2.5},
                    {"region": {"type": "box", "x": [45.0, 100.0], "y": [0.0, 40.0]}, "size": 0.125, "thickness": 1.0},
                ],
            }
        },
    )
    problem.config.material = MaterialConfig(
        E=32000.0,
        nu=0.20,
        Gc=3.0e-3,
        l0=0.5,
        rho=2.45e-9,
        energy_split="amor",
        pf_model="AT1",
        eta_residual=1.0e-7,
    )
    problem.config.boundary_conditions = [
        BoundaryConditionEntry(nodes="top", type="neumann", component=1, value=1.0),
        BoundaryConditionEntry(nodes="bottom", type="neumann", component=1, value=-1.0),
        BoundaryConditionEntry(nodes="left", type="fix", component=0),
        BoundaryConditionEntry(nodes="notch.boundary", type="pf_dirichlet", value=1.0),
    ]
    problem.config.initial_conditions = InitialConditionsConfig(preseed_notch_nodesets=["notch.boundary"])
    problem.config.loading = LoadingConfig(
        protocol="simple",
        ramp_type="smooth",
        t_ramp=5.0e-8,
        t_total=80.0e-6,
        num_steps=0,
    )
    problem.config.solver = SolverSettings(
        solver_type="explicit",
        dt_safety=0.8,
        use_multigrid=False,
        bounds_method="projected_cg",
        damage_every=2,
        eta_residual=1.0e-7,
    )
    problem.config.output = OutputConfig(
        trajectory=True,
        h5=True,
        trajectory_format="zarr",
        h5_every=50,
        fast=True,
        print_every=200,
    )
    problem.config.device = DeviceConfig(device="cpu")
    return _apply_num_steps(problem, num_steps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B7 COMSOL cross-check through the Python API.")
    parser.add_argument("--output-dir", default="examples/dynamic/B7_dynamic_crack_branching_comsol/run_fluent")
    parser.add_argument("--num-steps", type=int, default=None, help="Optional short-run explicit step count.")
    args = parser.parse_args()
    build_problem(args.num_steps).run(output_dir=args.output_dir, return_result=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
