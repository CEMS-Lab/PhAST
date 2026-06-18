"""Run the notched-holed plate benchmark without a YAML configuration.

This script constructs the same configuration objects used by ``config.yaml``.
It is intended as a readable Python setup example for users who want to define
the geometry, material, rigid connectors, loading, solver, and outputs
programmatically while keeping the YAML file as the canonical reproduction
input.
"""

from __future__ import annotations

import argparse

import phast
from phast.config import (
    BoundaryConditionEntry,
    DeviceConfig,
    GeometryConfig,
    InitialConditionsConfig,
    LoadingConfig,
    MaterialConfig,
    OutputConfig,
    SolverSettings,
)


def loading_phases(num_steps: int) -> str:
    """Return a valid two-stage pin-displacement schedule."""
    if num_steps <= 0:
        raise ValueError("--num-steps must be positive")
    if num_steps < 140:
        return f"0.25:{num_steps}"
    if num_steps == 140:
        return "0.25:140"
    return f"0.25:140,1.0:{num_steps - 140}"


def build_problem(num_steps: int) -> phast.Problem:
    """Build the COMSOL/Ambati notched-holed plate benchmark."""
    problem = phast.Problem(
        "COMSOL Notched Holed Plate strict parity (volDev + eta=1e-5)"
    )
    problem.config.reference = (
        "COMSOL 6.4 holed_plate_fracture.mph Damage settings; "
        "Ambati et al. (2015), Comput. Mech. 55, 383-405."
    )
    problem.config.acceptance = {
        "status": "validated",
        "reference_result": (
            "COMSOL 6.4 Application Library / strict-parity matrix run 33819 task 34"
        ),
        "metrics": {
            "first_peak_load": {
                "reference": 0.63,
                "observed_error_percent": 4.68,
                "tolerance_percent": 10.0,
                "units": "kN",
            },
            "first_peak_displacement": {
                "reference": 0.165,
                "observed_error_percent": 9.09,
                "tolerance_percent": 15.0,
                "units": "mm_per_pin",
            },
            "second_peak_load": {
                "reference": 0.15,
                "observed_error_percent": 10.51,
                "tolerance_percent": 20.0,
                "units": "kN",
            },
        },
    }
    problem.config.geometry = GeometryConfig(
        units="mm",
        primitives={
            "plate": {"type": "rectangle", "origin": [0.0, 0.0], "size": [65.0, 120.0]},
            "notch": {"type": "rectangle", "origin": [0.0, 64.75], "size": [10.0, 0.5]},
            "big_hole": {"type": "circle", "center": [36.5, 51.0], "radius": 10.0},
            "upper_pin": {"type": "circle", "center": [20.0, 100.0], "radius": 5.0},
            "lower_pin": {"type": "circle", "center": [20.0, 20.0], "radius": 5.0},
        },
        domain={
            "base": "plate",
            "subtract": ["notch", "big_hole", "upper_pin", "lower_pin"],
        },
        named_groups={
            "upper_pin_centre": {"point": [20.0, 100.0]},
            "lower_pin_centre": {"point": [20.0, 20.0]},
        },
        mesh={
            "element_size": {
                "default": 4.0,
                "refined": [
                    {
                        "region": {"type": "box", "x": [0.0, 65.0], "y": [45.0, 70.0]},
                        "size": 0.3,
                        "thickness": 4.0,
                    },
                    {"primitive": "big_hole", "size": 0.3, "margin": 8.0},
                    {"primitive": "upper_pin", "size": 1.0, "margin": 6.0},
                    {"primitive": "lower_pin", "size": 1.0, "margin": 6.0},
                ],
            }
        },
    )
    problem.config.material = MaterialConfig(
        E=6000.0,
        nu=0.22,
        Gc=2.28,
        l0=0.25,
        rho=2.4e-09,
        eta_residual=1.0e-05,
        energy_split="amor",
        pf_model="AT2",
        plane_stress=True,
    )
    problem.config.boundary_conditions = [
        BoundaryConditionEntry(
            nodes="upper_pin.boundary",
            type="rigid_connector",
            master="upper_pin_centre",
            dofs=["x", "y"],
            prescribe={"y": 2.0},
            rotation_free=True,
        ),
        BoundaryConditionEntry(
            nodes="lower_pin.boundary",
            type="rigid_connector",
            master="lower_pin_centre",
            dofs=["x", "y"],
            prescribe={"y": -2.0},
            rotation_free=True,
        ),
    ]
    problem.config.loading = LoadingConfig(
        protocol="cyclic",
        num_steps=num_steps,
        dt=0.01,
        cyclic_phases=loading_phases(num_steps),
    )
    problem.config.solver = SolverSettings(
        solver_type="quasi_static_legacy",
        stagger_tol=1.0e-04,
        max_stagger=50,
        stagger_criterion="relative",
        stagger_norm="l2",
        anderson_depth=0,
        adaptive_stagger_tol=False,
        use_multigrid=True,
        preconditioner="jacobi",
        H_cap_factor=0.0,
        damage_tol=1.0e-06,
        static_tol=1.0e-08,
        bounds_method="post_clamp",
        damage_every=3,
        fresh_d_in_corrector=False,
        damage_max_iter=5000,
        static_max_iter=5000,
        H_update_method="hard_max",
        enable_damage=True,
        fail_on_mechanics_nonconvergence=True,
        adaptive_dt=False,
        adaptive_dt_d_threshold=0.01,
        eta_residual=1.0e-05,
        damping_ratio_max=0.0,
        backend="auto",
    )
    problem.config.output = OutputConfig(
        h5=True,
        trajectory=True,
        trajectory_format="zarr",
        h5_every=5,
        gif=False,
        gif_frames=200,
        gif_fields="damage",
        animation_format="mp4",
        animation_renderer="raster",
        animation_raster_width=960,
        plots=True,
        profile=True,
        fast=True,
        print_every=5,
        reaction_node_set="upper_pin.boundary",
        reaction_component=1,
    )
    problem.config.device = DeviceConfig(device="cpu", compile=False)
    problem.config.initial_conditions = InitialConditionsConfig()
    return problem


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the notched-holed plate benchmark through Python config objects."
    )
    parser.add_argument(
        "--output-dir",
        default="examples/quasistatic/notched_holed_plate/run_fluent",
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="Number of quasi-static load steps. Use a small value for a quick check.",
    )
    args = parser.parse_args()
    build_problem(args.num_steps).run(output_dir=args.output_dir, return_result=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
