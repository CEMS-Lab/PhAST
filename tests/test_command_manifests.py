from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest_paths():
    return sorted((REPO_ROOT / "configs" / "benchmarks").glob("*/manifests/*.yaml"))


def test_command_manifests_are_parseable():
    paths = _manifest_paths()
    assert paths

    for path in paths:
        data = yaml.safe_load(path.read_text())
        assert data["schema_version"] == 1
        if "manifest_type" in data:
            assert data["manifest_type"] == "command_manifest"
        assert data["cases"]
        for case in data["cases"]:
            assert case.get("id") or case.get("label")
            assert case.get("module") or case.get("command")
            if case.get("module"):
                assert isinstance(case.get("args", []), list)
            if case.get("command"):
                assert isinstance(case["command"], list)
