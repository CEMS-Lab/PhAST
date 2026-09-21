#!/usr/bin/env python3
"""Replot retained standing-wave and SENT subcycling data; no simulations.

Use an existing environment with numpy, matplotlib, h5py, torch, meshio,
PyYAML and Pillow. No original repository, git checkout or network is used.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import shutil
import sys
from typing import Any


WAVE = "verification/standing_wave"
CADENCE = "forward/B1_dynamic_sent/checks/subcycling"
CODE = "shared/code/scripts/cmame_revision"
WAVE_FILES = ("summary.json", "standing_wave_verification.csv", "receipt_audit.json")
CADENCE_FILES = (
    "summary.json", "b1_submitted_mesh.msh", "trajectory_metrics.csv",
    "timing_measurements.csv", "terminal_metrics.csv",
    "trajectory_cadence_1.npz", "trajectory_cadence_2.npz", "trajectory_cadence_3.npz",
)
CODE_FILES = (
    "journal_figure_style.py", "render_known_journal_figures.py",
    "subcycling_sent_benchmark_audit.py",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checked_path(root: Path, row: dict[str, Any]) -> Path:
    name = row["path"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid input path: {name!r}")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or PureWindowsPath(name).drive or ".." in relative.parts
            or "\\" in name or not relative.parts):
        raise ValueError(f"Unsafe input path: {name}")
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Input escapes archive or is a symlink: {name}")
    if not path.is_file() or path.stat().st_size != row["bytes"]:
        raise ValueError(f"Input missing or resized: {name}")
    if digest(path) != row["sha256"]:
        raise ValueError(f"Input hash mismatch: {name}")
    return path


def read_csv(path: Path, numeric: bool = False) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: float(value) for key, value in row.items()} for row in rows] if numeric else rows


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def configure_runtime(output: Path) -> None:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    for key, name in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "cache"), ("TMPDIR", "tmp")):
        directory = output / "runtime" / name
        directory.mkdir(parents=True)
        os.environ[key] = str(directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Root containing manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory outside archive")
    parser.add_argument("--expected-file-count", type=int)
    args = parser.parse_args()
    archive, output = args.archive.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(archive) or archive.is_relative_to(output):
        raise ValueError("Use a fresh output directory separate from the archive")
    manifest_path = archive / "manifest.json"
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if args.expected_file_count is not None and len(manifest["files"]) != args.expected_file_count:
        raise ValueError("Manifest has not reached the requested materialisation count")
    index = {row["path"]: row for row in manifest["files"]}
    if len(index) != len(manifest["files"]):
        raise ValueError("Duplicate archive paths in manifest")
    names = ([f"{WAVE}/{name}" for name in WAVE_FILES]
             + [f"{CADENCE}/{name}" for name in CADENCE_FILES]
             + [f"{CODE}/{name}" for name in CODE_FILES])
    entries = [index[name] for name in names]
    for entry in entries:
        checked_path(archive, entry)
    output.mkdir(parents=True, exist_ok=False)
    configure_runtime(output)
    import numpy as np

    sys.path.insert(0, str(archive / CODE))
    style = importlib.import_module("journal_figure_style")
    known = importlib.import_module("render_known_journal_figures")
    cadence = importlib.import_module("subcycling_sent_benchmark_audit")
    for module in (style, known, cadence):
        if Path(module.__file__).resolve().parent != (archive / CODE).resolve():
            raise ValueError("Plotting module was not loaded from selected archive")
    style.apply_style()
    figures = output / "figures"
    figures.mkdir()

    wave_rows = read_csv(archive / WAVE / "standing_wave_verification.csv", numeric=True)
    wave_summary = json.loads((archive / WAVE / "summary.json").read_text())
    h = np.asarray([row["h"] for row in wave_rows])
    error = np.asarray([row["relative_l2_error"] for row in wave_rows])
    if len(h) != 4 or not np.isfinite([h, error]).all() or not (h > 0).all() or not (error > 0).all():
        raise ValueError("Expected four finite, positive standing-wave records")
    orders = np.log(error[:-1] / error[1:]) / np.log(h[:-1] / h[1:])
    np.testing.assert_allclose(orders, wave_summary["observed_orders_relative_l2"], rtol=1e-12, atol=1e-12)
    if not (np.diff(error) < 0).all():
        raise ValueError("Standing-wave errors are not monotone")

    # The historical renderer needs its old relative layout, but only these tiny
    # staged records are replayed. All renderer destinations are redirected.
    replay = output / "replay"
    wave_replay = replay / "results/r1_2_standing_wave_20260828_job119146"
    wave_replay.mkdir(parents=True)
    for name in WAVE_FILES:
        shutil.copyfile(archive / WAVE / name, wave_replay / name)
    known.ROOT = replay
    known.RESULTS = replay / "results"
    known.FIGURES_JOURNAL = figures
    known.ASSETS_JOURNAL = output / "renderer_assets"
    known.render_standing_wave([])

    data = archive / CADENCE
    timing_rows = read_csv(data / "timing_measurements.csv")
    timing = cadence.cadence_timing_summary(timing_rows)
    stored = json.loads((data / "summary.json").read_text())["timing"]
    for interval, record in timing.items():
        for key in ("count", "minimum", "median", "maximum"):
            np.testing.assert_allclose(record[key], stored[str(interval)]["explicit_update_wall_s"][key], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(record["saving_percent"] / 100,
                                   stored[str(interval)]["median_reduction_relative_to_cadence_1"], rtol=1e-12, atol=1e-12)
    if {int(row["steps"]) for row in timing_rows} != {6116}:
        raise ValueError("Unexpected timing step count")
    compacts = {}
    for interval in (1, 2, 3):
        with np.load(data / f"trajectory_cadence_{interval}.npz", allow_pickle=False) as arrays:
            compacts[interval] = {"nodes": arrays["nodes"].copy(), "elements": arrays["elements"].copy(),
                                 "damage": arrays["damage"][-1:].copy()}
        state = compacts[interval]
        if not np.isfinite(state["damage"]).all() or state["damage"].min() < -1e-12 or state["damage"].max() > 1 + 1e-12:
            raise ValueError("Invalid terminal damage field")
        for key in ("nodes", "elements"):
            np.testing.assert_array_equal(compacts[1][key], state[key])
    terminal = cadence.terminal_comparisons(compacts)
    recorded_terminal = read_csv(data / "terminal_metrics.csv", numeric=True)
    if len(terminal) != len(recorded_terminal):
        raise ValueError("Terminal metric row count differs")
    max_difference = 0.0
    for result, recorded in zip(terminal, recorded_terminal):
        if result.keys() != recorded.keys():
            raise ValueError("Terminal metric columns differ")
        for key, value in result.items():
            np.testing.assert_allclose(value, recorded[key], rtol=1e-10, atol=1e-12)
            max_difference = max(max_difference, abs(value - recorded[key]))
    trajectory = read_csv(data / "trajectory_metrics.csv", numeric=True)
    mesh = data / "b1_submitted_mesh.msh"
    notch = style.mesh_initial_crack(mesh, ("notch_upper", "notch_lower"), nodes=compacts[1]["nodes"])
    cadence.write_visual(compacts, trajectory, figures / "damage_cadence.png", journal=True,
                         compact=True, timing_rows=None, notch=notch, notch_source=mesh)

    from PIL import Image
    pixel_checks = []
    for name in ("standing_wave_verification", "damage_cadence"):
        with Image.open(figures / f"{name}.png") as image:
            pixels = np.asarray(image.convert("RGB"))
        nonwhite = float(np.mean(np.any(pixels < 240, axis=2)))
        if nonwhite < 0.005 or nonwhite > 0.95:
            raise ValueError(f"Blank or invalid raster output: {name}")
        pixel_checks.append({"path": f"figures/{name}.png", "width": int(pixels.shape[1]),
                             "height": int(pixels.shape[0]), "nonwhite_fraction": nonwhite})

    numeric = {"standing_wave": {"label": "fig:standing_wave_verification", "orders": orders.tolist(),
                                  "orders_match_retained_summary": True, "mechanics_errors_recomputed": False},
               "subcycling": {"labels": ["fig:damage_cadence", "tab:damage_cadence"], "timing": timing,
                               "timing_matches_retained_summary": True, "terminal_metrics": terminal,
                               "terminal_metrics_max_abs_difference": max_difference,
                               "terminal_metrics_match_retained_csv": True,
                               "trajectory_metrics_recomputed": False}}
    write_json(output / "numeric_checks.json", numeric)
    for entry in entries:
        checked_path(archive, entry)
    current = json.loads(manifest_path.read_text())
    final_index = {row["path"]: row for row in current["files"]}
    if [final_index[name] for name in names] != entries:
        raise ValueError("Selected manifest entries changed during replot")
    products = sorted(figures.iterdir()) + [output / "numeric_checks.json"]
    receipt = {
        "status": "completed_bounded_replot_and_numeric_checks", "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_file_count": len(manifest["files"]), "archive_payload_bytes": manifest["payload_bytes"],
        "manifest_sha256_before": manifest_hash, "manifest_sha256_after": digest(manifest_path),
        "wrapper_sha256": digest(Path(__file__)), "verified_input_files": len(entries),
        "verified_input_bytes": sum(row["bytes"] for row in entries), "inputs": entries,
        "selected_input_hashes_unchanged": True, "solver_runs": 0, "remote_transfers": 0,
        "original_repository_access_required": False, "pixel_checks": pixel_checks,
        "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                    for name in ("numpy", "matplotlib", "h5py", "torch", "meshio", "PyYAML", "Pillow")}},
        "limits": ["Retained-data replot checks, not full simulation reproduction or scientific validation.",
                   "Standing-wave convergence orders recomputed from retained errors, not displacement fields or mechanics.",
                   "Nine retained timing measurements summarised; no timings measured afresh.",
                   "Subcycling terminal metrics recomputed at three thresholds; full trajectory metrics reused, not recomputed.",
                   "No solver, equilibrium-residual or all-figure verification.",
                   "Only standing-wave and SENT subcycling inputs are included in these checks."],
        "products": [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                      "sha256": digest(path)} for path in products],
    }
    write_json(output / "receipt.json", receipt)
    report = ["# Bounded Forward Replot", "", f"Status: {receipt['status']}.", "",
              f"Archive manifest: {len(manifest['files'])} entries; {manifest['payload_bytes']:,} bytes.",
              f"Verified and rechecked: {len(entries)} selected inputs; {receipt['verified_input_bytes']:,} bytes.",
              "", "## Results", "",
              "S1 convergence orders: " + ", ".join(f"{value:.8f}" for value in orders) + ". Match retained summary.",
              "Subcycling terminal metrics match all nine retained threshold/cadence records.",
              "", "| Interval | Median update time (s) | Saving (%) |", "| --- | ---: | ---: |"]
    report.extend(f"| {key} | {value['median']:.9f} | {value['saving_percent']:.6f} |" for key, value in timing.items())
    report.extend(["", "Timing is accumulated explicit-update time over 6,116 steps, three retained one-thread CPU processes per interval. It excludes energy evaluation, telemetry and file output; these are not fresh timings.",
                   "", "## Replot", "", "From the repository root with the same plotting dependencies:", "", "```sh",
                   "python -B reproduction/paper1/replot_forward.py --archive /path/to/phast-paper-reproduction --output /path/to/new-forward-check",
                   "```", "", "Products: `figures/standing_wave_verification.{png,pdf}`, `figures/damage_cadence.{png,pdf}`, `numeric_checks.json`, `receipt.json`.",
                   "", "## Limits", ""])
    report.extend("- " + limit for limit in receipt["limits"])
    (output / "README.md").write_text("\n".join(report) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "archive_file_count", "verified_input_files",
                                                  "verified_input_bytes", "selected_input_hashes_unchanged", "solver_runs")}, indent=2))


if __name__ == "__main__":
    main()
