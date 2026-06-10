import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "qs_post_hpc_triage.py"
SPEC = importlib.util.spec_from_file_location("qs_post_hpc_triage", SCRIPT_PATH)
qs_post_hpc_triage = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = qs_post_hpc_triage
SPEC.loader.exec_module(qs_post_hpc_triage)


def test_shipped_stage_dispatches_sens_and_tpb_to_correct_comparators(
    tmp_path: Path,
    monkeypatch,
):
    calls = []

    def fake_run_command(cmd, log_path):
        calls.append((cmd, log_path))

    monkeypatch.setattr(qs_post_hpc_triage, "_run_command", fake_run_command)

    sens = tmp_path / "staged" / "sens_at2_medium_arc"
    tpb = tmp_path / "staged" / "tpb_at2_medium_arc"
    (sens / "run").mkdir(parents=True)
    (tpb / "run").mkdir(parents=True)
    plan = qs_post_hpc_triage.StagePlan(
        name="shipped",
        source_root=tmp_path / "source",
        stage_root=tmp_path / "staged",
        patterns=("sens_at2_*_arc", "tpb_at2_*_arc"),
        compare_scripts=(
            (
                "sens_",
                REPO_ROOT / "examples/quasistatic/miehe_shear/compare.py",
                (),
            ),
            (
                "tpb_",
                REPO_ROOT / "examples/quasistatic/three_point_bending/compare.py",
                (),
            ),
        ),
    )

    qs_post_hpc_triage._compare_runs(plan, [sens, tpb], tmp_path / "logs")

    assert calls[0][0][1].endswith("examples/quasistatic/miehe_shear/compare.py")
    assert calls[0][0][-1] == str(sens / "run")
    assert calls[1][0][1].endswith("examples/quasistatic/three_point_bending/compare.py")
    assert calls[1][0][-1] == str(tpb / "run")
