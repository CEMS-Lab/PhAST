import csv
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/plot_qs_mesh_convergence.py"
spec = importlib.util.spec_from_file_location("plot_qs_mesh_convergence", SCRIPT_PATH)
plot_qs = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = plot_qs
spec.loader.exec_module(plot_qs)


def _write_summary(path: Path) -> None:
    fieldnames = [
        "case",
        "benchmark",
        "at_mode",
        "h_crack",
        "l0",
        "continuation",
        "overall",
        "has_compare_report",
        "peak_rel_error_pct",
        "pre_peak_l2_pct",
        "energy_error_pct",
        "envelope_l2_pct",
        "n_steps",
        "u_max",
        "reaction_peak_kN",
        "u_at_reaction_peak",
        "max_damage",
        "latest_step",
        "latest_u",
        "latest_reaction_kN",
        "latest_max_damage",
        "has_zarr",
        "has_mp4",
        "has_initial_conditions",
        "has_final_damage",
        "has_compare_png",
        "run_dir",
    ]
    rows = [
        {
            "case": "sent_at2_h0.02_l0.06",
            "benchmark": "sent",
            "at_mode": "at2",
            "h_crack": "0.02",
            "l0": "0.06",
            "continuation": "arc_length",
            "overall": "PASS",
            "has_compare_report": "True",
            "peak_rel_error_pct": "4.0",
            "pre_peak_l2_pct": "2.0",
            "energy_error_pct": "3.0",
            "envelope_l2_pct": "5.0",
            "n_steps": "10",
            "u_max": "0.1",
            "reaction_peak_kN": "2.5",
            "u_at_reaction_peak": "0.04",
            "max_damage": "0.95",
            "latest_step": "9",
            "latest_u": "0.1",
            "latest_reaction_kN": "1.8",
            "latest_max_damage": "0.95",
            "has_zarr": "True",
            "has_mp4": "True",
            "has_initial_conditions": "True",
            "has_final_damage": "True",
            "has_compare_png": "True",
            "run_dir": "/tmp/sent/run",
        },
        {
            "case": "tpb_at1_h0.12_l0.03",
            "benchmark": "tpb",
            "at_mode": "at1",
            "h_crack": "0.12",
            "l0": "0.03",
            "continuation": "path_following",
            "overall": "MISSING",
            "has_compare_report": "False",
            "peak_rel_error_pct": "",
            "pre_peak_l2_pct": "",
            "energy_error_pct": "",
            "envelope_l2_pct": "",
            "n_steps": "2",
            "u_max": "0.03",
            "reaction_peak_kN": "0.8",
            "u_at_reaction_peak": "0.02",
            "max_damage": "0.4",
            "latest_step": "1",
            "latest_u": "0.03",
            "latest_reaction_kN": "0.6",
            "latest_max_damage": "0.4",
            "has_zarr": "False",
            "has_mp4": "False",
            "has_initial_conditions": "True",
            "has_final_damage": "False",
            "has_compare_png": "False",
            "run_dir": "/tmp/tpb/run",
        },
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_report_groups_rows_and_includes_required_status_fields(tmp_path):
    summary = tmp_path / "mesh_convergence_summary.csv"
    _write_summary(summary)

    rows = plot_qs.load_rows(summary)
    md = plot_qs.report_markdown(rows, summary)

    assert "## sent" in md
    assert "## tpb" in md
    assert "peak reaction [kN]" in md
    assert "u_at_peak" in md
    assert "max_damage" in md
    assert "step 9, u=0.1, R=1.8 kN, dmax=0.95" in md
    assert "5/5 complete" in md
    assert "1/5; missing zarr, mp4, final damage, compare png" in md
    assert "Overall status counts: MISSING=1, PASS=1" in md


def test_main_writes_markdown_png_and_pdf_without_hpc_inputs(tmp_path):
    summary = tmp_path / "mesh_convergence_summary.csv"
    out_dir = tmp_path / "figures"
    _write_summary(summary)

    rc = plot_qs.main([str(summary), "--out-dir", str(out_dir), "--stem", "meshconv"])

    assert rc == 0
    assert (out_dir / "meshconv_report.md").is_file()
    assert (out_dir / "meshconv.png").stat().st_size > 0
    assert (out_dir / "meshconv.pdf").stat().st_size > 0
    assert (out_dir / "meshconv_sent.png").stat().st_size > 0
    assert (out_dir / "meshconv_tpb.pdf").stat().st_size > 0


def test_load_rows_rejects_wrong_csv_schema(tmp_path):
    bad = tmp_path / "mesh_convergence_summary.csv"
    bad.write_text("benchmark,h_crack\nsent,0.02\n")

    try:
        plot_qs.load_rows(bad)
    except ValueError as exc:
        assert "reaction_peak_kN" in str(exc)
    else:
        raise AssertionError("load_rows should reject missing required columns")
