"""Tests for ``scripts/check_h5_integrity.py``.

Issue #265 (B2 Kalthoff mesh1 truncated training_data.h5) and the
broader follow-up: post-rsync H5 truncation has hit at least four
files on this machine. The fix in PR `fix-b2-kalthoff-mesh1-
truncated-265` ships a generic scanner; these tests pin its behaviour
on three synthetic fixtures so a future regression in the parser is
caught immediately.

Fixtures (all built with h5py inside ``tmp_path``):

1. *clean*  — a small but well-formed file
2. *truncated tail* — clean file then truncated by N bytes (matches
   the B2_kalthoff_mesh1 failure mode exactly: ``size < stored_eof``)
3. *not HDF5* — a random binary blob

For each, ``Report.ok`` and ``Report.deficit`` must match expected
values; ``parse_h5_eof`` must match what h5py says ``stored_eof`` is.
"""
from __future__ import annotations

import importlib.util
import os
import struct
import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPT = REPO_DIR / "scripts" / "check_h5_integrity.py"

h5py = pytest.importorskip("h5py")


def _load_module():
    """Import the script as a module without making it importable
    via ``scripts.check_h5_integrity`` (no ``__init__.py`` in scripts/).
    Mirrors the pattern used by tests/test_verify_citations.py.
    """
    spec = importlib.util.spec_from_file_location(
        "check_h5_integrity", SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


CHK = _load_module()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_h5(path: Path) -> int:
    """Write a small but well-formed HDF5 file. Returns final size."""
    import numpy as np
    with h5py.File(path, "w") as f:
        sim = f.create_group("simulation_data")
        sim.create_dataset("damage", data=np.linspace(0.0, 1.0, 1000))
        sim.attrs["test"] = "fixture"
    return path.stat().st_size


def _truncate(path: Path, lose_bytes: int) -> None:
    sz = path.stat().st_size
    new_size = max(0, sz - lose_bytes)
    with open(path, "r+b") as f:
        f.truncate(new_size)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_parse_h5_eof_matches_h5py(tmp_path: Path) -> None:
    """Pure-python EOF parser agrees with h5py on a clean file."""
    p = tmp_path / "clean.h5"
    _make_h5(p)
    parsed = CHK.parse_h5_eof(p)
    assert parsed is not None, "parser returned None on a clean H5"
    # Clean files always have stored_eof == file_size.
    assert parsed == p.stat().st_size, (
        f"parser says {parsed}, h5py says {p.stat().st_size}"
    )


def test_clean_file_passes(tmp_path: Path) -> None:
    p = tmp_path / "clean.h5"
    _make_h5(p)
    rep = CHK.check_file(p)
    assert rep.is_hdf5 is True
    assert rep.h5py_open_error is None
    assert rep.deficit == 0
    assert rep.ok is True


def test_truncated_tail_is_caught(tmp_path: Path) -> None:
    """Reproduce the B2_kalthoff_mesh1 failure mode: clean superblock,
    file size < stored_eof. Matches issue #265 exactly."""
    p = tmp_path / "truncated.h5"
    _make_h5(p)
    eof_clean = CHK.parse_h5_eof(p)
    assert eof_clean is not None
    deficit = 256
    _truncate(p, deficit)
    rep = CHK.check_file(p)
    assert rep.is_hdf5 is True, "superblock should still validate"
    assert rep.stored_eof == eof_clean, (
        "stored_eof must come from the superblock, not from file size"
    )
    assert rep.size == eof_clean - deficit
    assert rep.deficit == deficit
    assert rep.h5py_open_error is not None, (
        "h5py should refuse to open a tail-truncated file"
    )
    assert "truncated" in rep.h5py_open_error.lower()
    assert rep.ok is False


def test_non_hdf5_is_flagged(tmp_path: Path) -> None:
    p = tmp_path / "garbage.bin"
    p.write_bytes(b"hello, this is not HDF5\n" * 64)
    rep = CHK.check_file(p)
    assert rep.is_hdf5 is False
    assert rep.stored_eof is None
    assert rep.ok is False


def test_scan_returns_only_matches(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    p1 = tmp_path / "a" / "training_data.h5"
    p2 = tmp_path / "b" / "training_data.h5"
    p3 = tmp_path / "b" / "other.h5"
    _make_h5(p1)
    _make_h5(p2)
    _make_h5(p3)
    reports = CHK.scan([tmp_path])
    paths = sorted(str(r.path) for r in reports)
    assert paths == sorted([str(p1), str(p2)]), (
        f"scan should match only training_data.h5, got: {paths}"
    )


def test_scan_skips_dotdirs(tmp_path: Path) -> None:
    """``.git``, ``.claude``, ``__pycache__`` etc. must not be walked."""
    bad_in_dotgit = tmp_path / ".git" / "training_data.h5"
    bad_in_dotgit.parent.mkdir()
    _make_h5(bad_in_dotgit)
    bad_in_pycache = tmp_path / "__pycache__" / "training_data.h5"
    bad_in_pycache.parent.mkdir()
    _make_h5(bad_in_pycache)
    real = tmp_path / "real" / "training_data.h5"
    real.parent.mkdir()
    _make_h5(real)
    reports = CHK.scan([tmp_path])
    assert len(reports) == 1
    assert reports[0].path == real


def test_main_exits_nonzero_on_breakage(tmp_path: Path) -> None:
    p = tmp_path / "training_data.h5"
    _make_h5(p)
    _truncate(p, 64)
    rc = CHK.main([str(tmp_path)])
    assert rc == 1


def test_main_exits_zero_when_clean(tmp_path: Path) -> None:
    p = tmp_path / "training_data.h5"
    _make_h5(p)
    rc = CHK.main([str(tmp_path)])
    assert rc == 0


def test_main_handles_no_matches(tmp_path: Path) -> None:
    rc = CHK.main([str(tmp_path)])
    # No files found is not a failure — composes safely into CI.
    assert rc == 0


def test_b2_kalthoff_mesh1_assertion_on_local_copy(tmp_path: Path) -> None:
    """If the user's local copy of B2_kalthoff_mesh1/training_data.h5
    exists, this test asserts it is well-formed (i.e. they have run
    ``h5clear --increment=0`` per RECOVERY.md, or re-downloaded /
    re-run on HPC). Skipped when the file is absent — the file is
    gitignored so CI will skip; locally it gates the user's recovery.

    Issue #265.
    """
    target = (
        REPO_DIR
        / "examples"
        / "dynamic"
        / "kalthoff"
        / "reference_runs"
        / "B2_kalthoff_mesh1"
        / "training_data.h5"
    )
    if not target.is_file():
        pytest.skip(f"{target} not present; recovery has not been performed")
    rep = CHK.check_file(target)
    assert rep.is_hdf5, f"{target} is not a valid HDF5 file"
    assert rep.stored_eof is not None, f"{target} superblock unparseable"
    assert rep.size >= rep.stored_eof, (
        f"{target} is truncated: size={rep.size} < stored_eof={rep.stored_eof} "
        f"(deficit {rep.stored_eof - rep.size} B). "
        f"See examples/dynamic/kalthoff/reference_runs/B2_kalthoff_mesh1/RECOVERY.md to repair."
    )
    assert rep.h5py_open_error is None, (
        f"{target} fails h5py open: {rep.h5py_open_error}"
    )
    assert rep.ok, f"integrity check failed on {target}"
