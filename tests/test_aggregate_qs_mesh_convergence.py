import csv
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/aggregate_qs_mesh_convergence.py"
spec = importlib.util.spec_from_file_location("aggregate_qs_mesh_convergence", SCRIPT_PATH)
aggregate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(aggregate)


def test_report_parser_accepts_spaced_overall_and_lshape_l2(tmp_path):
    run = tmp_path / "lshape_at2_h0.3_l1.0" / "run"
    run.mkdir(parents=True)
    (run / "compare_report.txt").write_text(
        "\n".join([
            "Peak rel error          :   6.46 %  (tol 15 %)  -> PASS",
            "L2 (envelope vs ref)    :   6.24 %  to u=0.300 mm",
            "OVERALL                 : PASS",
        ])
    )

    metrics = aggregate._report_metrics(run)

    assert metrics["overall"] == "PASS"
    assert metrics["peak_rel_error_pct"] == 6.46
    assert metrics["envelope_l2_pct"] == 6.24


def test_history_parser_reads_history_and_results_units(tmp_path):
    run = tmp_path / "tpb_at2_coarse_arc" / "run"
    run.mkdir(parents=True)
    (run / "history.csv").write_text(
        "\n".join([
            "step,max_damage,reaction_force,applied_disp",
            "0,0.1,-100.0,0.001",
            "1,0.3,-250.0,0.002",
        ])
    )

    metrics = aggregate._history_metrics(run)

    assert metrics["n_steps"] == 2
    assert metrics["reaction_peak_kN"] == 0.25
    assert metrics["latest_step"] == 1
    assert metrics["latest_reaction_kN"] == 0.25
    assert metrics["latest_max_damage"] == 0.3


def test_history_parser_falls_back_to_results_csv(tmp_path):
    run = tmp_path / "sent_at2_coarse" / "run"
    run.mkdir(parents=True)
    (run / "results.csv").write_text(
        "\n".join([
            "step,displacement,reaction_kN,max_d",
            "0,0.001,0.10,0.2",
            "1,0.002,0.15,0.4",
        ])
    )

    metrics = aggregate._history_metrics(run)

    assert metrics["u_max"] == 0.002
    assert metrics["reaction_peak_kN"] == 0.15
    assert metrics["latest_max_damage"] == 0.4


def test_main_writes_partial_progress_columns(tmp_path):
    run = tmp_path / "sens_at2_coarse_arc" / "run"
    run.mkdir(parents=True)
    (run / "case.yaml").write_text(
        "\n".join([
            "args:",
            "- --at_mode",
            "- at2",
            "- --h_crack",
            "- 0.010",
            "- --l0",
            "- 0.060",
            "- --arc_length",
        ])
    )
    (run / "history.csv").write_text(
        "step,max_damage,reaction_force,applied_disp\n0,0.2,-10,0.001\n"
    )
    (run / "compare_report.txt").write_text("OVERALL: PASS\n")
    (run / "training_data.zarr").mkdir()
    (run / "damage_evolution.mp4").write_bytes(b"mp4")

    out = tmp_path / "summary.csv"
    rc = aggregate.main([str(tmp_path), "--out", str(out)])
    assert rc == 0

    row = next(csv.DictReader(out.open()))
    assert row["benchmark"] == "sens"
    assert row["at_mode"] == "at2"
    assert float(row["h_crack"]) == 0.010
    assert float(row["l0"]) == 0.060
    assert row["continuation"] == "arc_length"
    assert row["has_zarr"] == "True"
    assert row["has_mp4"] == "True"
    assert row["latest_u"] == "0.001"
