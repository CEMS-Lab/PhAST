from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.docs


def test_ci_runs_config_and_docs_consistency_gates():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "ci-testing.yml"
    )
    text = workflow.read_text(encoding="utf-8")

    required = [
        "python scripts/generate_reference_yaml.py --check",
        "python scripts/generate_json_schema.py --check",
        "python scripts/check_docs_drift.py",
        "python scripts/check_artifact_hygiene.py",
        "python scripts/check_package_payload.py",
    ]
    for command in required:
        assert command in text

    consistency_step = text.index("Verify generated config reference is current")
    test_step = text.index("Run package test suite")
    assert consistency_step < test_step
