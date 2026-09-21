#!/usr/bin/env python3
"""Inspect the paper catalogue, check retained data and run bounded replots.

Catalogue and checksum checks use the Python standard library. Replotting
uses the separate numerical environment described in the adjacent README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPLOT_SCRIPTS = {
    "q2": "replot_q2.py",
    "inverse": "replot_inverse.py",
    "forward": "replot_forward.py",
}
SHA256 = re.compile(r"[0-9a-f]{64}")


def read_json(path: Path) -> dict[str, Any]:
    """Read an object while rejecting ambiguous duplicate keys."""
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path.name}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON numeric constant {value}")

    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                        parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return result


def relative_path(value: str) -> PurePosixPath:
    """Require a portable archive-relative file path."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid archive path {value!r}")
    path = PurePosixPath(value)
    if (path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/"))
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f"Unsafe archive path {value!r}")
    return path


def data_file(root: Path, name: str) -> Path:
    path = root
    for part in relative_path(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Archive symlink is unsupported {name!r}")
    if not path.is_file():
        raise ValueError(f"Required archive file is missing {name!r}")
    return path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def catalogue_check(catalogue: dict[str, Any]) -> dict[str, Any]:
    if catalogue.get("schema_version") != 1:
        raise ValueError("Unsupported catalogue schema")
    identity = catalogue["data_identity"]
    if not isinstance(identity, dict):
        raise ValueError("Expected a data identity object")
    for key in ("manifest_sha256", "document_identity_sha256"):
        if not isinstance(identity.get(key), str) or not SHA256.fullmatch(identity[key]):
            raise ValueError(f"Catalogue requires a valid {key}")
    studies = catalogue["studies"]
    if not isinstance(studies, dict) or not studies:
        raise ValueError("Expected a non-empty study index")
    if any(not isinstance(study, dict) for study in studies.values()):
        raise ValueError("Expected a study object for each study identifier")
    if any(not isinstance(catalogue.get(key), list) for key in ("items", "results")):
        raise ValueError("Expected item and result lists")
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for row in catalogue["items"] + catalogue["results"]:
        if not isinstance(row, dict) or not isinstance(row.get("studies"), list):
            raise ValueError("Each catalogue entry requires a study list")
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"Invalid or repeated catalogue identifier {identifier!r}")
        seen.add(identifier)
        if not row["studies"]:
            raise ValueError(f"Study mapping required for {identifier}")
        for study in row["studies"]:
            if study not in studies:
                raise ValueError(f"Unknown study {study!r} in {identifier}")
        for name in row.get("archive_assets", []):
            relative_path(name)
        if "document" in row:
            key = f"{row['document']}_{row['kind']}"
            counts[key] = counts.get(key, 0) + 1
    for identifier, study in studies.items():
        if study.get("replot_group") not in (None, *REPLOT_SCRIPTS):
            raise ValueError(f"Unknown replot group in {identifier}")
        if not study.get("replot_status") or not study.get("rerun_status"):
            raise ValueError(f"Separate replot and rerun status required for {identifier}")
        for name in study.get("archive_paths", []):
            relative_path(name)
        related = study.get("related_example")
        if related:
            relative_path(related)
            if not (HERE.parents[1] / related).is_dir():
                raise ValueError(f"Related public example is missing {related!r}")
    return {"status": "catalogue_structure_checked", "items": len(catalogue["items"]),
            "result_groups": len(catalogue["results"]), "studies": len(studies),
            "counts": counts, "simulation_reproduction_verified": False}


def verify_data(root: Path, expected_manifest: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = data_file(root, "manifest.json")
    manifest_sha256 = digest(manifest_path)
    if expected_manifest is not None:
        if not SHA256.fullmatch(expected_manifest):
            raise ValueError("Expected manifest hash must be a lowercase SHA-256")
        if manifest_sha256 != expected_manifest:
            raise ValueError("Archive manifest differs from the expected SHA-256")
    manifest = read_json(manifest_path)
    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Expected a non-empty archive file manifest")
    names: set[str] = set()
    total = 0
    for row in entries:
        name = row["path"]
        if name in names:
            raise ValueError(f"Duplicate archive entry {name!r}")
        names.add(name)
        size, expected = row["bytes"], row["sha256"]
        if type(size) is not int or size < 0 or not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError(f"Invalid size or checksum for {name!r}")
        path = data_file(root, name)
        if path.stat().st_size != size or digest(path) != expected:
            raise ValueError(f"Archive size or checksum mismatch for {name!r}")
        total += size
    if manifest.get("payload_bytes", total) != total:
        raise ValueError("Archive payload total differs from its file entries")
    if manifest.get("source_file_count", len(entries)) != len(entries):
        raise ValueError("Archive file count differs from its file entries")
    if digest(manifest_path) != manifest_sha256:
        raise ValueError("Archive manifest changed during verification")
    return {"status": "input_integrity_checked", "files": len(entries), "bytes": total,
            "manifest_sha256": manifest_sha256, "archive_status": manifest.get("status"),
            "simulation_reproduction_verified": False}


def verify_catalogued_data(catalogue: dict[str, Any], root: Path,
                           expected_manifest: str | None = None) -> dict[str, Any]:
    identity = catalogue["data_identity"]
    pinned = identity["manifest_sha256"]
    if expected_manifest is not None and expected_manifest != pinned:
        raise ValueError("Requested manifest hash differs from this paper catalogue")
    document_path = data_file(root.resolve(), "document_identity.json")
    if digest(document_path) != identity["document_identity_sha256"]:
        raise ValueError("Archive document edition differs from this paper catalogue")
    report = verify_data(root, pinned)
    manifest = read_json(root / "manifest.json")
    names = {row["path"] for row in manifest["files"]}
    assets = {name for row in catalogue.get("items", [])
              for name in row.get("archive_assets", [])}
    if assets - names:
        raise ValueError(f"Catalogued assets absent from the manifest: {sorted(assets - names)}")
    for study in catalogue.get("studies", {}).values():
        for path in study.get("archive_paths", []):
            if path not in names and not any(name.startswith(path + "/") for name in names):
                raise ValueError(f"Catalogued study path absent from the manifest: {path}")
    report["catalogued_asset_paths_checked"] = len(assets)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List manuscript and supplementary items in reading order")
    check = commands.add_parser("check", help="Check catalogue and optionally retained-data hashes")
    check.add_argument("--data", type=Path)
    check.add_argument("--manifest-sha256", help="Independently recorded manifest hash")
    replot = commands.add_parser("replot", help="Regenerate selected figures from retained data")
    replot.add_argument("group", choices=REPLOT_SCRIPTS)
    replot.add_argument("--data", type=Path, required=True)
    replot.add_argument("--output", type=Path, required=True)
    replot.add_argument("--manifest-sha256")
    args = parser.parse_args(argv)
    try:
        catalogue = read_json(HERE / "catalogue.json")
        report = catalogue_check(catalogue)
        if args.command == "list":
            for row in catalogue["items"]:
                print(f"{row['id']}  {' '.join(row['studies'])}")
            print(f"{report['result_groups']} numerical-result groups are also indexed.")
            return 0
        if args.manifest_sha256 and args.data is None:
            raise ValueError("--manifest-sha256 requires --data")
        if args.command == "replot":
            archive, output = args.data.resolve(), args.output.resolve()
            if output.exists() or output.is_relative_to(archive) or archive.is_relative_to(output):
                raise ValueError("Choose a fresh output directory separate from the archive")
        if args.data is not None:
            report["data"] = verify_catalogued_data(catalogue, args.data, args.manifest_sha256)
        print(json.dumps(report, indent=2), flush=True,
              file=sys.stderr if args.command == "replot" else sys.stdout)
        if args.command == "replot":
            command = [sys.executable, "-B", str(HERE / REPLOT_SCRIPTS[args.group]),
                       "--archive", str(archive), "--output", str(output)]
            return subprocess.run(command, check=False).returncode
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Reproduction check error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
