import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/check_qs_reference_runs.py"
spec = importlib.util.spec_from_file_location("check_qs_reference_runs", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
has_pass_report = module.has_pass_report
DEFAULT_PACKAGES = module.DEFAULT_PACKAGES
DIAGNOSTIC_PACKAGES = module.DIAGNOSTIC_PACKAGES
SUPPLEMENTAL_PACKAGES = module.SUPPLEMENTAL_PACKAGES


def test_has_pass_report_requires_overall_pass_line(tmp_path: Path):
    report = tmp_path / "compare_report.txt"
    report.write_text("""
Peak reaction: PASS
Pre-peak L2: PASS
OVERALL: FAIL
""")

    assert not has_pass_report(report)


def test_has_pass_report_accepts_exact_overall_pass(tmp_path: Path):
    report = tmp_path / "compare_report.txt"
    report.write_text("""
Peak reaction: PASS
OVERALL: PASS
""")

    assert has_pass_report(report)


def test_default_promoted_qs_packages_are_paper_grade_only():
    paths = {package.as_posix() for package in DEFAULT_PACKAGES}

    assert any("qs_sent_41278_coarse" in path for path in paths)
    assert any("qs_sent_41278_medium" in path for path in paths)
    assert not any("qs_lshape_concrete_37993" in path for path in paths)
    assert not any("notched_holed" in path for path in paths)


def test_diagnostic_qs_packages_are_not_promoted_by_default():
    assert DIAGNOSTIC_PACKAGES == []


def test_supplemental_qs_packages_are_not_promoted_by_default():
    assert SUPPLEMENTAL_PACKAGES == []
