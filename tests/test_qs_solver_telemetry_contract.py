"""Output-contract checks for standalone quasi-static benchmark drivers."""

from pathlib import Path

import pytest


pytestmark = [pytest.mark.fast, pytest.mark.artifact]

REPO_ROOT = Path(__file__).resolve().parents[1]
QS_RUNNERS = [
    "examples/quasistatic/miehe_tension/run.py",
    "examples/quasistatic/miehe_shear/run.py",
    "examples/quasistatic/three_point_bending/run.py",
    "examples/quasistatic/l_shaped_panel/run.py",
]


def _function_body(text: str, name: str) -> str:
    start = text.index(f"def {name}(")
    next_def = text.find("\ndef ", start + 1)
    return text[start:] if next_def == -1 else text[start:next_def]


def test_telemetry_writer_accepts_tuple_and_dict_rows(tmp_path):
    from phast.io_utils import write_solver_telemetry_csv

    out = tmp_path / "solver_telemetry.csv"
    write_solver_telemetry_csv(
        str(out),
        [
            (0, 1.0e-4, 3, 12, 7, 2.5e-8, 2.5e-4, 4.0e-6, 1.6e-4, 1.0e-4),
            {
                "step": 1,
                "time": 2.0e-4,
                "newton_iters": 4,
                "pcg_iters_mech": 14,
                "pcg_iters_pf": 8,
                "residual": 1.0e-8,
                "relative_residual": 5.0e-4,
                "mechanics_residual": 2.0e-7,
                "mechanics_relative_residual": 1.0e-4,
                "dt": 1.0e-4,
            },
        ],
    )

    lines = out.read_text().splitlines()
    assert lines[0] == (
        "step,time,newton_iters,pcg_iters_mech,pcg_iters_pf,residual,"
        "relative_residual,mechanics_residual,mechanics_relative_residual,"
        "dt,line_search_alpha,line_search_reductions,continuation_mode,"
        "arc_length_residual,arc_length_constraint,load_factor"
    )
    assert lines[1].startswith(
        "0,1.000000000e-04,3,12,7,2.500000000e-08,2.500000000e-04,"
    )
    assert lines[2].startswith(
        "1,2.000000000e-04,4,14,8,1.000000000e-08,5.000000000e-04,"
    )


def test_energy_writer_and_plot_use_canonical_schema(tmp_path):
    from phast.io_utils import write_energy_csv, plot_energy_history

    rows = [
        {
            "step": 0,
            "time": 1.0e-4,
            "elastic": 1.0,
            "fracture": 0.2,
            "kinetic": 0.0,
            "external": 0.0,
            "total": 1.2,
        },
        {
            "step": 1,
            "time": 2.0e-4,
            "elastic": 0.8,
            "fracture": 0.5,
            "kinetic": 0.0,
            "external": 0.0,
            "total": 1.3,
        },
    ]
    csv_path = tmp_path / "energy.csv"
    png_path = tmp_path / "energy.png"

    write_energy_csv(str(csv_path), rows)
    plot_energy_history(rows, str(png_path))

    lines = csv_path.read_text().splitlines()
    assert lines[0] == "step,time,elastic,fracture,kinetic,external,total"
    assert lines[1].startswith("0,1.000000000e-04,1.000000000e+00")
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_quasistatic_convergence_plot_writes_png(tmp_path):
    from phast.visualization import plot_quasistatic_convergence

    out = tmp_path / "staggered_convergence.png"
    plot_quasistatic_convergence(
        [
            {
                "step": 0,
                "disp": 1.0e-4,
                "stagger_iter": 3,
                "pcg_iters_mech": 12,
                "pcg_iters_pf": 7,
                "residual": 2.5e-8,
            },
            {
                "step": 1,
                "disp": 2.0e-4,
                "stagger_iter": 4,
                "pcg_iters_mech": 14,
                "pcg_iters_pf": 8,
                "residual": 1.0e-8,
            },
        ],
        str(out),
    )

    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.parametrize("runner", QS_RUNNERS)
def test_quasistatic_runners_emit_solver_telemetry(runner):
    text = (REPO_ROOT / runner).read_text()

    assert "write_solver_telemetry_csv" in text
    assert "'solver_telemetry.csv'" in text
    assert "write_energy_csv" in text
    assert "'energy.csv'" in text
    assert "plot_energy_history" in text
    assert "'energy.png'" in text
    assert "'timing_per_step.csv'" in text
    assert "Total Step Time" in text
    assert "damage_evolution.{anim_ext}" in text
    assert "default='mp4'" in text
    assert "plot_quasistatic_convergence" in text
    assert "'staggered_convergence.png'" in text
    assert "'pcg_iters_mech'" in text
    assert "'pcg_iters_pf'" in text
    assert "'residual'" in text
    assert "'relative_residual'" in text


def test_miehe_tension_bcs_match_phasefieldx_1711():
    text = (REPO_ROOT / "examples/quasistatic/miehe_tension/run.py").read_text()
    setup_bcs = _function_body(text, "setup_bcs")

    assert "bcs.fix(mesh.node_sets['bottom'], 0)" in setup_bcs
    assert "bcs.fix(mesh.node_sets['bottom'], 1)" in setup_bcs
    assert "bcs.add(mesh.node_sets['top'], 1, 1.0)" in setup_bcs
    assert "bcs.fix(mesh.node_sets['top'], 0)" not in setup_bcs
    assert "default=0.0" in text[text.index("'--H_cap_factor'"):text.index("'--H_cap_factor'") + 180]


def test_miehe_shear_reaction_matches_phasefieldx_1712_convention():
    text = (REPO_ROOT / "examples/quasistatic/miehe_shear/run.py").read_text()

    assert "reaction_nodes = mesh.node_sets['bottom']" in text
    assert "'reaction_N': R, 'reaction_kN': -R / 1000.0" in text


def test_lshape_validation_default_does_not_cap_history_field():
    text = (REPO_ROOT / "examples/quasistatic/l_shaped_panel/run.py").read_text()
    hcap_arg = text[text.index("'--H_cap_factor'"):text.index("'--H_cap_factor'") + 180]

    assert "default=0.0" in hcap_arg
