"""Paper catalogue and retained-data integrity checks use tiny local fixtures."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "reproduction" / "paper1" / "reproduce.py"
SPEC = importlib.util.spec_from_file_location("paper_reproduce", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def data(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    value = b"time,energy\n0,0\n1,2\n"
    (root / "history.csv").write_bytes(value)
    manifest = {"status": "test_fixture", "source_file_count": 1,
                "payload_bytes": len(value), "files": [
                    {"path": "history.csv", "bytes": len(value),
                     "sha256": hashlib.sha256(value).hexdigest()}]}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def rewrite_manifest(root: Path, key: str, value: Any) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[key] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_catalogue_has_unique_resolved_items_and_separate_statuses() -> None:
    catalogue = MODULE.read_json(SCRIPT.with_name("catalogue.json"))
    report = MODULE.catalogue_check(catalogue)
    assert report["items"] == 38
    assert report["result_groups"] >= 36
    assert report["simulation_reproduction_verified"] is False


def test_data_check_preserves_bytes_and_distinguishes_integrity(data: Path) -> None:
    before = {file.name: file.read_bytes() for file in data.iterdir()}
    report = MODULE.verify_data(data, MODULE.digest(data / "manifest.json"))
    assert report["files"] == 1
    assert report["simulation_reproduction_verified"] is False
    assert report["archive_status"] == "test_fixture"
    assert before == {file.name: file.read_bytes() for file in data.iterdir()}


def test_changed_same_size_data_is_rejected(data: Path) -> None:
    path = data / "history.csv"
    path.write_bytes(path.read_bytes().replace(b"1,2", b"1,3"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        MODULE.verify_data(data)


def test_missing_data_is_rejected(data: Path) -> None:
    (data / "history.csv").unlink()
    with pytest.raises(ValueError, match="missing"):
        MODULE.verify_data(data)


@pytest.mark.parametrize("name", ["../history.csv", "/history.csv", "a/../../x", "a\\x",
                                  "C:/x", "a//b", "./a", "a/", "", "a\nb"])
def test_unsafe_paths_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        MODULE.relative_path(name)


def test_symlink_in_input_path_is_rejected(data: Path) -> None:
    link = data / "alias"
    try:
        link.symlink_to(data, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation unavailable on this platform")
    with pytest.raises(ValueError, match="symlink"):
        MODULE.data_file(data, "alias/history.csv")


def test_manifest_may_be_pinned_independently(data: Path) -> None:
    with pytest.raises(ValueError, match="expected SHA-256"):
        MODULE.verify_data(data, "0" * 64)


def test_data_is_bound_to_catalogued_document_edition(data: Path) -> None:
    identity_path = data / "document_identity.json"
    identity_path.write_text('{"edition": "fixture"}', encoding="utf-8")
    catalogue = {"data_identity": {
        "manifest_sha256": MODULE.digest(data / "manifest.json"),
        "document_identity_sha256": MODULE.digest(identity_path)}}
    assert MODULE.verify_catalogued_data(catalogue, data)["files"] == 1
    identity_path.write_text('{"edition": "different"}', encoding="utf-8")
    with pytest.raises(ValueError, match="document edition differs"):
        MODULE.verify_catalogued_data(catalogue, data)


def test_optional_hash_cannot_override_catalogued_data(data: Path) -> None:
    catalogue = {"data_identity": {"manifest_sha256": "a" * 64}}
    with pytest.raises(ValueError, match="differs from this paper catalogue"):
        MODULE.verify_catalogued_data(catalogue, data, "b" * 64)


@pytest.mark.parametrize("declaration", [
    {"items": [{"archive_assets": ["missing.pdf"]}]},
    {"studies": {"test": {"archive_paths": ["missing-study"]}}},
])
def test_dangling_catalogue_data_references_are_rejected(
        data: Path, declaration: dict[str, Any]) -> None:
    path = data / "document_identity.json"
    path.write_text("{}", encoding="utf-8")
    catalogue = {**declaration, "data_identity": {
        "manifest_sha256": MODULE.digest(data / "manifest.json"),
        "document_identity_sha256": MODULE.digest(path)}}
    with pytest.raises(ValueError, match="absent from the manifest"):
        MODULE.verify_catalogued_data(catalogue, data)


def test_duplicate_paths_are_rejected(data: Path) -> None:
    manifest = MODULE.read_json(data / "manifest.json")
    rewrite_manifest(data, "files", manifest["files"] * 2)
    with pytest.raises(ValueError, match="Duplicate archive entry"):
        MODULE.verify_data(data)


@pytest.mark.parametrize("key,value", [("payload_bytes", 0), ("source_file_count", 0)])
def test_incorrect_manifest_totals_are_rejected(data: Path, key: str, value: int) -> None:
    rewrite_manifest(data, key, value)
    with pytest.raises(ValueError, match="differs"):
        MODULE.verify_data(data)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"files": [], "files": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        MODULE.read_json(path)


def test_nonfinite_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"bytes": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="numeric constant"):
        MODULE.read_json(path)


def test_unknown_study_reference_is_rejected() -> None:
    catalogue = MODULE.read_json(SCRIPT.with_name("catalogue.json"))
    catalogue["items"][0]["studies"] = ["missing"]
    with pytest.raises(ValueError, match="Unknown study"):
        MODULE.catalogue_check(catalogue)


@pytest.mark.parametrize("field", ["data_identity", "studies", "items", "results"])
def test_invalid_catalogue_containers_are_rejected(field: str) -> None:
    catalogue = MODULE.read_json(SCRIPT.with_name("catalogue.json"))
    catalogue[field] = None
    with pytest.raises(ValueError):
        MODULE.catalogue_check(catalogue)


def test_invalid_study_record_is_rejected() -> None:
    catalogue = MODULE.read_json(SCRIPT.with_name("catalogue.json"))
    catalogue["studies"][next(iter(catalogue["studies"]))] = None
    with pytest.raises(ValueError, match="study object"):
        MODULE.catalogue_check(catalogue)


def test_replot_requires_fresh_output(data: Path) -> None:
    assert MODULE.main(["replot", "q2", "--data", str(data), "--output", str(data)]) == 2


def test_manifest_option_requires_data() -> None:
    assert MODULE.main(["check", "--manifest-sha256", "0" * 64]) == 2


def test_catalogue_and_list_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert MODULE.main(["check"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "catalogue_structure_checked"
    assert MODULE.main(["list"]) == 0
    assert "figure.1" in capsys.readouterr().out
