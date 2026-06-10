import importlib.util
from pathlib import Path


HELPER = Path(__file__).resolve().parent / "_tier_markers.py"
SPEC = importlib.util.spec_from_file_location("tier_markers", HELPER)
tier_markers = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tier_markers)
auto_tier_markers = tier_markers.auto_tier_markers
auto_timeout_seconds = tier_markers.auto_timeout_seconds


def test_auto_tier_markers_classifies_docs_and_fast_tests():
    markers = auto_tier_markers(
        "tests/test_docs_drift_check.py::test_rejects_stale_text",
        existing=[],
    )

    assert {"docs", "fast"} <= markers


def test_auto_tier_markers_does_not_make_explicit_benchmark_fast():
    markers = auto_tier_markers(
        "tests/test_dataset_generators.py::test_seed_reproducible[class]",
        existing=["benchmark", "artifact"],
    )

    assert markers == set()


def test_auto_tier_markers_marks_inverse_as_solver_not_fast():
    markers = auto_tier_markers(
        "tests/inverse/test_gradcheck.py::test_small_gradcheck",
        existing=[],
    )

    assert markers == {"solver"}


def test_auto_timeout_seconds_for_long_tiers():
    assert auto_timeout_seconds({"benchmark"}) == 600
    assert auto_timeout_seconds({"slow"}) == 600
    assert auto_timeout_seconds({"hpc"}) == 1800
    assert auto_timeout_seconds({"hpc", "timeout"}) is None
    assert auto_timeout_seconds({"solver"}) is None
