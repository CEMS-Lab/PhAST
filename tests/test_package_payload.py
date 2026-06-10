import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_package_payload.py"
SPEC = importlib.util.spec_from_file_location("check_package_payload", SCRIPT)
package_payload = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = package_payload
SPEC.loader.exec_module(package_payload)
find_package_payload_violations = package_payload.find_package_payload_violations


def _pyproject(package_data=None, *, include_package_data=False, packages=None):
    return {
        "tool": {
            "setuptools": {
                "include-package-data": include_package_data,
                "packages": packages or ["phast"],
                "package-data": package_data or {"phast": ["configs/*.yaml"]},
            }
        }
    }


def test_current_package_payload_policy_allows_configs_and_small_references():
    violations = find_package_payload_violations(_pyproject(
        {
            "phast": [
                "configs/*.yaml",
                "configs/*.json",
                "reference_solutions/*.csv",
                "examples/dynamic/timing_comparisons/*/*.msh",
            ]
        }
    ))

    assert violations == []


def test_package_payload_requires_explicit_package_data_only():
    violations = find_package_payload_violations(_pyproject(include_package_data=True))

    assert violations
    assert "include-package-data" in violations[0].path


def test_package_payload_rejects_private_and_raw_result_patterns():
    violations = find_package_payload_violations(_pyproject(
        {
            "phast": [
                "docs/molinari_meeting/*.md",
                "papers/paper2/results/**/*.json",
                "examples/demo/training_data.h5",
            ]
        }
    ))

    reasons = {item.path: item.reason for item in violations}
    assert "phast: docs/molinari_meeting/*.md" in reasons
    assert "private/research directories" in reasons[
        "phast: docs/molinari_meeting/*.md"
    ]
    assert "phast: papers/paper2/results/**/*.json" in reasons
    assert "private/research directories" in reasons[
        "phast: papers/paper2/results/**/*.json"
    ]
    assert "phast: examples/demo/training_data.h5" in reasons
    assert "raw/generated .h5" in reasons[
        "phast: examples/demo/training_data.h5"
    ]


def test_package_payload_rejects_hpc_result_tree_without_blocking_references():
    violations = find_package_payload_violations(_pyproject(
        {
            "phast": [
                "examples/quasistatic/*/reference_solutions/*.csv",
                "examples/quasistatic/*/hpc_results/**/*.json",
            ]
        }
    ))

    assert [item.path for item in violations] == [
        "phast: examples/quasistatic/*/hpc_results/**/*.json"
    ]
    assert "raw run/result trees" in violations[0].reason



def test_package_payload_rejects_research_directories_as_packages():
    violations = find_package_payload_violations(_pyproject(
        packages=["phast", "papers/paper2", "docs.molinari_meeting"],
    ))

    paths = {item.path for item in violations}
    assert "papers/paper2" in paths
    assert "docs.molinari_meeting" in paths
