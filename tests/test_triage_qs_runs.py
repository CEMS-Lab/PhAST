import csv
import importlib.util
import io
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "triage_qs_runs.py"
SPEC = importlib.util.spec_from_file_location("triage_qs_runs", SCRIPT_PATH)
triage_qs_runs = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = triage_qs_runs
SPEC.loader.exec_module(triage_qs_runs)


def write_file(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_contract_run(path: Path, overall: str = "PASS") -> Path:
    write_file(path / "compare_report.txt", f"Peak reaction: PASS\nOVERALL: {overall}\n")
    (path / "training_data.zarr").mkdir(parents=True)
    write_file(path / "training_data.zarr" / ".zgroup", "{}")
    write_file(path / "damage_evolution.mp4", "mp4")
    write_file(path / "initial_conditions.png", "png")
    write_file(path / "load_displacement.png", "png")
    write_file(path / "damage_final.png", "png")
    write_file(path / "compare.png", "png")
    write_file(path / "run.log", "log")
    write_file(path / "slurm.out", "log")
    return path


def test_classify_promote_for_complete_pass_contract(tmp_path: Path):
    run = make_contract_run(tmp_path / "qs_good")

    result = triage_qs_runs.classify_run(run)

    assert result.status == triage_qs_runs.Status.PROMOTE
    assert result.overall == "PASS"
    assert result.reasons == ()


def test_classify_failed_for_final_overall_fail(tmp_path: Path):
    run = make_contract_run(tmp_path / "qs_failed", overall="FAIL")

    result = triage_qs_runs.classify_run(run)

    assert result.status == triage_qs_runs.Status.FAILED
    assert "compare_report OVERALL is FAIL" in result.reasons


def test_classify_diagnostic_for_pass_with_contract_violations(tmp_path: Path):
    run = make_contract_run(tmp_path / "qs_needs_cleanup")
    (run / "damage_evolution.mp4").unlink()
    write_file(run / "checkpoint_0100.pt", "weights")

    result = triage_qs_runs.classify_run(run)

    assert result.status == triage_qs_runs.Status.DIAGNOSTIC
    assert result.missing == ("damage_evolution.mp4",)
    assert result.forbidden == ("checkpoint_0100.pt",)


def test_classify_incomplete_without_final_report(tmp_path: Path):
    run = tmp_path / "qs_running"
    write_file(run / "run.log", "still running")

    result = triage_qs_runs.classify_run(run)

    assert result.status == triage_qs_runs.Status.INCOMPLETE
    assert result.overall == "MISSING"


def test_nested_run_directory_is_supported(tmp_path: Path):
    package = tmp_path / "promoted_package"
    artifact_dir = make_contract_run(package / "run")

    result = triage_qs_runs.classify_run(package)

    assert result.status == triage_qs_runs.Status.PROMOTE
    assert result.artifact_dir == artifact_dir.resolve()


def test_retracted_promotion_note_overrides_complete_pass_contract(tmp_path: Path):
    package = tmp_path / "retracted_package"
    artifact_dir = make_contract_run(package / "run")
    write_file(
        package / "PROMOTION.md",
        "# Retraction Note\n\nRetracted from paper-grade status after visual audit.\n",
    )

    result = triage_qs_runs.classify_run(package)

    assert result.status == triage_qs_runs.Status.DIAGNOSTIC
    assert result.artifact_dir == artifact_dir.resolve()
    assert "retracted from paper-grade" in " ".join(result.reasons)


def test_supplemental_promotion_note_overrides_complete_pass_contract(tmp_path: Path):
    package = tmp_path / "supplemental_package"
    artifact_dir = make_contract_run(package / "run")
    write_file(
        package / "PROMOTION.md",
        "# Promotion Note\n\nSupplemental package only; placeholder reference.\n",
    )

    result = triage_qs_runs.classify_run(package)

    assert result.status == triage_qs_runs.Status.DIAGNOSTIC
    assert result.artifact_dir == artifact_dir.resolve()
    assert "supplemental/non-canonical" in " ".join(result.reasons)


def test_discover_root_with_many_run_dirs_and_render_outputs(tmp_path: Path):
    root = tmp_path / "root"
    good = make_contract_run(root / "qs_good")
    failed = make_contract_run(root / "qs_failed", overall="FAIL")
    (root / "notes").mkdir()

    run_dirs = triage_qs_runs.discover_run_dirs([root])
    results = [triage_qs_runs.classify_run(path) for path in run_dirs]

    assert [path.resolve() for path in run_dirs] == [failed.resolve(), good.resolve()]
    assert [result.status for result in results] == [
        triage_qs_runs.Status.FAILED,
        triage_qs_runs.Status.PROMOTE,
    ]

    csv_rows = list(csv.DictReader(io.StringIO(triage_qs_runs.render_csv(results))))
    assert [row["status"] for row in csv_rows] == ["FAILED", "PROMOTE"]

    markdown = triage_qs_runs.render_markdown(results)
    assert "| status | run_dir | overall | reasons |" in markdown
    assert "PROMOTE" in markdown
    assert "FAILED" in markdown
