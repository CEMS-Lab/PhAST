"""Tests for the docs/config drift guard."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_docs_drift.py"
pytestmark = pytest.mark.docs


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_docs_drift", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_docs_drift_checker_rejects_old_stagger_energy_enum(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text("solver.stagger_criterion: energy\n", encoding="utf-8")

    errors = checker.check_paths([bad])

    assert errors
    assert "am_energy" in errors[0]


def test_docs_drift_checker_accepts_current_stagger_enum(tmp_path):
    checker = _load_checker()
    good = tmp_path / "good.yaml"
    good.write_text("solver:\n  stagger_criterion: am_energy\n",
                    encoding="utf-8")

    errors = checker.check_paths([good])

    assert errors == []


def test_docs_drift_checker_rejects_monolithic_production_claim(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text(
        "MonolithicSolver: L-BFGS joint solve bypassing the stagger loop entirely\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "experimental" in errors[0]


def test_docs_drift_checker_rejects_legacy_run_yaml_command(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text(
        "Run with: python -m phast.run_yaml B1_branching_glass.yaml\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "canonical YAML CLI" in errors[0]


def test_docs_drift_checker_rejects_legacy_run_yaml_example(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "example: phast.run_yaml\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "canonical YAML CLI" in errors[0]


def test_docs_drift_checker_rejects_example_key_required_claim(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text(
        "Every YAML must have a top-level `example` key.\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "optional" in errors[0]


def test_docs_drift_checker_rejects_at1_post_clamp_equivalence(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text(
        "AT1 uses projected_cg or post-clamp for the damage sub-problem.\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "projected bound enforcement" in errors[0]


def test_docs_drift_checker_rejects_qs_preconditioner_auto_default(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text(
        "| `--preconditioner` | str | auto | jacobi, amg, auto |\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "default to Jacobi" in errors[0]


def test_docs_drift_checker_rejects_legacy_timing_paths(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text(
        "Use examples/timing_comparisons/sent_clean/timing_table.md\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "examples/dynamic/timing_comparisons" in errors[0]


def test_docs_drift_checker_rejects_removed_tutorial_links(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text(
        "See tutorial/minimal-example.md and api/index.rst\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "current docs/tutorial" in errors[0]


def test_docs_drift_checker_rejects_hdf5_primary_output_claim(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text(
        "| HDF5 snapshots | Production | Main reusable simulation output. |\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "Zarr-first" in errors[0]


def test_docs_drift_checker_rejects_hdf5_packager_claim(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text("The generation pipeline uses an HDF5 packager.\n",
                   encoding="utf-8")

    errors = checker.check_paths([bad])

    assert errors
    assert "Zarr-only packager" in errors[0]


def test_docs_drift_checker_rejects_hdf5_fallback_claim(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text("SamplePackager writes Zarr with an HDF5 fallback.\n",
                   encoding="utf-8")

    errors = checker.check_paths([bad])

    assert errors
    assert "legacy solver/post-processing compatibility" in errors[0]


def test_docs_drift_checker_rejects_guaranteed_zarr_size_claim(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text("Zarr is guaranteed smaller than H5.\n", encoding="utf-8")

    errors = checker.check_paths([bad])

    assert errors
    assert "blanket size claims" in errors[0]


def test_docs_drift_checker_rejects_stale_b7_half_plate_height(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "# COMSOL: 100 x 40 mm half-plate, pre-crack along y=0\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "height/2 = 20 mm" in errors[0]


def test_docs_drift_checker_rejects_stale_b7_full_plate_height(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text(
        "Mirroring about y=0 -> 100 x 80 mm full plate.\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "100 x 40 mm" in errors[0]


def test_docs_drift_checker_rejects_stale_b7_verlet_parity(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.md"
    bad.write_text(
        "The COMSOL B7 parity setup uses explicit Verlet velocity.\n",
        encoding="utf-8",
    )

    errors = checker.check_paths([bad])

    assert errors
    assert "generalized-alpha" in errors[0]
