#!/usr/bin/env python3
"""Replot retained Q2 data and audit saved fields; no simulation or residual solve."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import runpy
import shutil
import sys
from typing import Any


# Historical relative paths are replay identifiers, not public workflow names.
STUDY = Path("papers/paper/reviewer_evidence/studies/q2_completion_20260918")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relative_path(value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest paths must be nonempty strings")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts
            or "\\" in value or not path.parts):
        raise ValueError(f"Unsafe manifest path {value!r}")
    return Path(*path.parts)


def verified_path(root: Path, relative: Path, entry: dict[str, Any]) -> Path:
    relative = relative_path(relative.as_posix())
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes its input root or is a symlink: {relative}")
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Missing input or changed byte count: {relative}")
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Changed SHA-256: {relative}")
    return path


def selected_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in manifest["files"] if "q2" in row.get("groups", [])]
    if not rows:
        raise ValueError("Manifest contains no q2 entries")
    archive_paths, legacy_paths = set(), set()
    for row in rows:
        archive = relative_path(row["path"])
        legacy = relative_path(row["legacy_path"])
        if archive in archive_paths or legacy in legacy_paths:
            raise ValueError("Duplicate archive or legacy destination in q2 selection")
        archive_paths.add(archive)
        legacy_paths.add(legacy)
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("Manifest byte counts must be nonnegative integers")
        if (not isinstance(row["sha256"], str) or len(row["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in row["sha256"])):
            raise ValueError("Invalid SHA-256 in manifest")
    return rows


def configure_runtime(output: Path) -> None:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    for variable, name in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "cache"), ("TMPDIR", "tmp")):
        directory = output / "runtime" / name
        directory.mkdir(parents=True)
        os.environ[variable] = str(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True,
                        help="Extracted archive root containing manifest.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh directory outside archive")
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(archive) or archive.is_relative_to(output):
        raise ValueError("Use a fresh output directory separate from the archive")
    manifest_path = archive / "manifest.json"
    manifest_hash = digest(manifest_path)
    rows = selected_entries(json.loads(manifest_path.read_text()))
    for row in rows:
        verified_path(archive, relative_path(row["path"]), row)

    output.mkdir(parents=True, exist_ok=False)
    configure_runtime(output)
    replay = output / "replay"
    replay.mkdir()
    for row in rows:
        target = replay / relative_path(row["legacy_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive / relative_path(row["path"]), target)
        verified_path(replay, relative_path(row["legacy_path"]), row)

    study = replay / STUDY
    run = study / "hpc_runs/zero_job134377"
    accepted = json.loads((run / "accepted.json").read_text())
    status = json.loads((run / "status.json").read_text())
    if (len(accepted) != 200 or status.get("status") != "completed"
            or status.get("original_targets_reached") != 200
            or not status.get("full_trajectory_from_zero")):
        raise ValueError("Selected data is not the completed fresh 200-target trajectory")

    audit_function = runpy.run_path(str(study / "audit_saved_fields.py"))["audit"]
    audit = audit_function(run, output / "field_audit")
    if not audit["field_archive_checks_pass"] or audit["states_inspected"] != 200:
        raise ValueError(f"Saved-field audit failed: {audit['field_failures']}")

    # Redirect only output destinations, leaving the archived plotting code intact.
    render = runpy.run_path(str(study / "render_manuscript.py"))["completed_figure"]
    figures = output / "figures"
    figures.mkdir()
    publication = output / "render_destinations/paper"
    (publication / "figures/version_2").mkdir(parents=True)
    (publication / "reviewer_evidence/assets/journal").mkdir(parents=True)
    render.__globals__["OUT"] = figures
    render.__globals__["PAPER"] = publication
    render()

    for row in rows:
        verified_path(archive, relative_path(row["path"]), row)
        verified_path(replay, relative_path(row["legacy_path"]), row)
    final_rows = selected_entries(json.loads(manifest_path.read_text()))
    ordered = lambda entries: sorted(entries, key=lambda entry: entry["path"])
    if ordered(final_rows) != ordered(rows):
        raise ValueError("Q2 manifest entries changed while replotting")
    final_manifest_hash = digest(manifest_path)
    q2_manifest_hash = hashlib.sha256(
        json.dumps(ordered(rows), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    products = [figures / "q2_completed.png", figures / "q2_completed.pdf",
                output / "field_audit/field_audit.json",
                output / "field_audit/full_fields.png", output / "field_audit/reaction.png"]
    receipt = {
        "status": "completed_replot_and_saved_field_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_hash,
        "final_manifest_sha256": final_manifest_hash,
        "q2_manifest_entries_sha256": q2_manifest_hash,
        "q2_manifest_entries_unchanged": True,
        "other_manifest_content_changed": final_manifest_hash != manifest_hash,
        "wrapper_sha256": digest(Path(__file__)),
        "selected_files": len(rows),
        "selected_bytes": sum(row["bytes"] for row in rows),
        "states_inspected": audit["states_inspected"],
        "field_archive_checks_pass": audit["field_archive_checks_pass"],
        "field_failures": audit["field_failures"],
        "archive_and_replay_input_hashes_unchanged": True,
        "full_simulation_run": False,
        "equilibrium_residuals_recomputed": False,
        "scope": "All saved states checked for file integrity, finite fields, damage bounds, "
                 "irreversibility, previous-damage consistency and load-factor consistency. "
                 "The figure uses archived reactions; equilibrium residuals are not recomputed.",
        "runtime": {"python": platform.python_version(),
                    **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")}},
        "products": [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                      "sha256": digest(path)} for path in products],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
