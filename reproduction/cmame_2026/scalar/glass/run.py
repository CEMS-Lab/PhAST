"""Verify or run the archived glass inversion with its isolated source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


BUNDLE = Path(__file__).resolve().parent
SOURCE_COMMIT = "56e4dda3236c1dad64756be8ed9ba83c3f7c1ce1"
DRIVER_SHA256 = "4ba7d5b4f87de4333a36a699f21bc1499700183d004537af1ccf1de6695c223c"
RESULT_SHA256 = "fce56da3dd087448756bd168e0e51216b7436f7b6cc0dd6a6560111eea5bef5b"
# Pin the public source inventory separately from the manifest.
SOURCE_INVENTORY_SHA256 = "85f05c2eaecabe969446c0b0ce45eda343b588bc774e339804f9e31056ffe6ab"


def verify_bundle(bundle: Path = BUNDLE) -> dict[str, Any]:
    """Check fixed source/result identities and the complete file inventory."""
    manifest = json.loads((bundle / "historical_manifest.json").read_text())
    expected = {
        "schema_version": 1,
        "kind": "historical-calculation-code",
        "source_commit": SOURCE_COMMIT,
        "source_path": "src/phast/paper/demos/demo_inversion.py",
        "driver_path": "original_driver.py",
        "driver_sha256": DRIVER_SHA256,
        "result_path": "recorded_result.json",
        "result_sha256": RESULT_SHA256,
        "runtime_recomputed": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Invalid historical identity: {key}")
    file_map = manifest.get("files_sha256")
    if not isinstance(file_map, dict) or not file_map:
        raise ValueError("Missing historical file inventory")
    actual_files = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are not permitted in the historical bundle")
        if path.is_file() and "__pycache__" not in path.relative_to(bundle).parts:
            relative = path.relative_to(bundle).as_posix()
            if relative not in ("README.md", "historical_manifest.json"):
                actual_files.add(relative)
    if actual_files != set(file_map):
        raise ValueError("Historical inventory does not cover exactly the bundle files")
    for relative, expected_sha in file_map.items():
        path = bundle / relative
        if bundle.resolve() not in path.resolve().parents:
            raise ValueError("Historical inventory path is outside the bundle")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise ValueError(f"Historical SHA256 mismatch: {relative}")
    if file_map["original_driver.py"] != DRIVER_SHA256:
        raise ValueError("Original driver hash cannot be redefined")
    if file_map["recorded_result.json"] != RESULT_SHA256:
        raise ValueError("Original result hash cannot be redefined")
    sources = manifest.get("source_files")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Missing original source inventory")
    inventory = {}
    for source_path, relative in sorted(sources.items()):
        if relative not in file_map:
            raise ValueError(f"Original source not covered by file inventory: {relative}")
        inventory[source_path] = hashlib.sha256((bundle / relative).read_bytes()).hexdigest()
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    if digest != SOURCE_INVENTORY_SHA256 or manifest.get("source_snapshot_sha256") != digest:
        raise ValueError("Public source inventory does not match the fixed snapshot")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="Verify hashes; do not import or solve.")
    mode.add_argument("--import-check", action="store_true", help="Inspect historical imports and defaults; do not solve.")
    mode.add_argument("--driver-help", action="store_true", help="Show unchanged original driver arguments; do not solve.")
    parser.add_argument("--output-dir", type=Path, help="New external directory, required only for a numerical run.")
    args = parser.parse_args()
    try:
        manifest = verify_bundle()
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(2, f"Historical bundle verification failed: {exc}\n")
    if args.verify_only:
        print(f"Verified {len(manifest['files_sha256'])} historical bundle files; no calculation executed.")
        return 0
    if args.import_check:
        driver_args = ["--import-check"]
    elif args.driver_help:
        driver_args = ["--help"]
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required to execute the historical inversion")
        output = args.output_dir.resolve()
        if output == BUNDLE or BUNDLE in output.parents:
            parser.error("numerical outputs must be outside the historical bundle")
        if output.exists():
            parser.error("use a new output directory; existing results are never overwritten")
        invocation = json.loads((BUNDLE / "invocation.json").read_text())
        driver_args = [*invocation["arguments"], "--output_dir", str(output)]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-I", "-B", "-u", str(BUNDLE / "isolated_entry.py"), *driver_args]
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
