"""Resume of an interrupted result file (``ResultWriter.resume``).

An interrupted run is cut at a checkpoint increment and continued; the finished file must equal an uninterrupted one
in every dataset, dtype and attribute except the resume record. The 'kill' is a writer dropped without ``close`` (the
file stays ``.partial`` with its flushed rows), as in the kill-after-flush test.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from phast.io.result import Field, ImageGrid, ResultWriter, ResumeRefused, UnstructuredMesh, sha256_file, validate

N = 9                                    # increments 0 .. 8
_SKIP_ROOT = {"created", "finished", "resume_count", "status_reason"}
_SKIP_META = {"resumes"}


# --- a small run that can be replayed step by step -------------------------------------------------

def _grid():
    return ImageGrid((3, 2, 2), (0.5, 0.5, 0.5), cell_type="H8")


def _grid_fields():
    n_cells = 3 * 2 * 2
    return [Field("phase_id", "cell", "id", "1", "state", static=True, data=np.arange(n_cells) % 3),
            Field("displacement", "node", "x,y,z", "mm", "state"),
            Field("damage", "node", "d", "1", "state"),
            Field("H", "quadrature_point", ",".join(f"qp{q}" for q in range(8)), "MPa", "history"),
            Field("stress", "cell", "s", "MPa", "derived", placement="derived")]


def _mesh():
    return UnstructuredMesh(np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]]), {"T3": [[0, 1, 2], [0, 2, 3]]})


def _mesh_fields():
    return [Field("damage", "node", "d", "1", "state"), Field("psi", "cell", "psi", "MPa", "derived", placement="derived")]


def _row(k):
    return dict(increment=k, time=float(k), dt=1.0, load_parameter=0.1 * k, damage_solved=int(k > 0),
                **{"macro_stress/xx": 2.0 * k + 0.25})


def _step_image(w, k):
    n_pts, n_cells = 4 * 3 * 3, 3 * 2 * 2
    fields = derived = None
    if k % 2 == 0:                                        # a stored step at even increments
        base = np.arange(n_pts, dtype=float)
        fields = dict(displacement=np.stack([base + k, base * 2 + k, base * 3 - k], 1), damage=base / 100 + k,
                      H=np.arange(n_cells * 8, dtype=float).reshape(n_cells, 8) + 0.5 * k)
    if k % 3 == 0:
        derived = dict(stress=np.arange(n_cells, dtype=float) * 7 + k)
    w.append(_row(k), fields=fields, derived=derived)


def _new_image(path, **kw):
    return ResultWriter(path, _grid(), _grid_fields(), dim=3, analysis="3d", run_id="resume-test", sidecars=False,
                        history_units={"load_parameter": "1"}, config={"case": "t"}, **kw)


def _resume_image(path, upto, **kw):
    return ResultWriter.resume(path, _grid(), _grid_fields(), upto_increment=upto, run_id="resume-test", sidecars=False,
                               history_units={"load_parameter": "1"}, **kw)


def _step_mesh(w, k):
    fields = dict(damage=np.full(4, 0.1 * k)) if k % 2 == 0 else None
    derived = dict(psi=np.full(2, 1.5 * k)) if k % 4 == 0 else None
    w.append(_row(k), fields=fields, derived=derived)
    w.append_crack_tip(dict(increment=k, tip_id=0, position=(0.1 * k, 0.5), threshold=0.5))
    w.append_substep(dict(increment=k, call=0, cg_iterations=7 + k, cg_converged=True, effective_preconditioner="jacobi" if k % 2 else "amg"))
    w.flush()


def _new_mesh(path):
    return ResultWriter(path, _mesh(), _mesh_fields(), dim=2, analysis="plane_strain", run_id="resume-mesh", sidecars=False,
                        history_units={"load_parameter": "1"}, config="x")


def _resume_mesh(path, upto):
    return ResultWriter.resume(path, _mesh(), _mesh_fields(), upto_increment=upto, run_id="resume-mesh", sidecars=False,
                               history_units={"load_parameter": "1"})


# --- comparison -------------------------------------------------------------------------------------

def _tree(path):
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = (obj.dtype, obj.shape, obj[...], {k: _plain(v) for k, v in obj.attrs.items()})
        f.visititems(visit)
        root = {k: _plain(v) for k, v in f.attrs.items() if k not in _SKIP_ROOT}
        groups = {}
        f.visititems(lambda n, o: groups.__setitem__(n, {k: _plain(v) for k, v in o.attrs.items() if not (n == "meta" and k in _SKIP_META)})
                     if isinstance(o, h5py.Group) else None)
    return out, root, groups


def _plain(v):
    return v.tolist() if isinstance(v, np.ndarray) else v.decode() if isinstance(v, bytes) else v


def assert_same_file(a, b):
    da, ra, ga = _tree(a)
    db, rb, gb = _tree(b)
    assert sorted(da) == sorted(db), set(da) ^ set(db)
    for name in da:
        (ta, sa, xa, aa), (tb, sb, xb, ab) = da[name], db[name]
        assert ta == tb and sa == sb, (name, ta, tb, sa, sb)
        if xa.dtype.kind in "fc":
            assert np.array_equal(xa, xb, equal_nan=True), name
        else:
            assert np.array_equal(xa, xb), name
        assert aa == ab, (name, aa, ab)
    assert ra == rb
    assert ga == gb


def _errors(path):
    return [x for x in validate(path).findings if x.level == "ERROR"]


# --- uninterrupted references ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference_image(tmp_path_factory):
    d = tmp_path_factory.mktemp("ref_image")
    w = _new_image(d / "result.vtkhdf")
    for k in range(N):
        _step_image(w, k)
    w.close()
    return d / "result.vtkhdf"


@pytest.fixture(scope="module")
def reference_mesh(tmp_path_factory):
    d = tmp_path_factory.mktemp("ref_mesh")
    w = _new_mesh(d / "result.vtkhdf")
    for k in range(N):
        _step_mesh(w, k)
    w.close()
    return d / "result.vtkhdf"


# --- resume equals an uninterrupted run --------------------------------------------------------------

@pytest.mark.parametrize("killed_after, checkpoint", [(0, 0), (3, 3), (5, 3), (4, 4), (8, 6), (7, 0)])
def test_kill_and_resume_equals_uninterrupted_image(tmp_path, reference_image, killed_after, checkpoint):
    path = tmp_path / "result.vtkhdf"
    w = _new_image(path)
    for k in range(killed_after + 1):
        _step_image(w, k)
    w.f.close()                                             # killed: the flushed rows remain, status running
    assert (tmp_path / "result.vtkhdf.partial").exists() and not path.exists()
    w = _resume_image(path, checkpoint)
    assert w.n_rows == checkpoint + 1
    for k in range(checkpoint + 1, N):
        _step_image(w, k)
    w.close()
    assert not _errors(path) and (tmp_path / "result.vtkhdf.sha256").exists()
    assert_same_file(reference_image, path)
    with h5py.File(path, "r") as f:
        assert f.attrs["status"] == "complete" and int(f.attrs["resume_count"]) == 1
        log = json.loads(f["meta"].attrs["resumes"])
    assert len(log) == 1 and log[0]["from_increment"] == checkpoint and log[0]["kept_rows"] == checkpoint + 1
    assert log[0]["cut_rows"] == killed_after - checkpoint and log[0]["previous_status"] == "running"
    bak = tmp_path / "result.vtkhdf.partial.pre-resume-1.bak"
    assert bak.exists() and log[0]["previous_sha256"] == sha256_file(bak)


def test_kill_between_the_stored_step_and_its_row(tmp_path, reference_image):
    """The fields of step k are on disk but its history row is not: the extra stored step is cut."""
    path = tmp_path / "result.vtkhdf"
    w = _new_image(path)
    for k in range(5):
        _step_image(w, k)
    base = np.arange(4 * 3 * 3, dtype=float)
    w._write_step(dict(displacement=np.stack([base, base, base], 1), damage=base, H=np.zeros((12, 8))), 6.0)
    w.f.flush()
    w.f.close()
    w = _resume_image(path, 4)
    for k in range(5, N):
        _step_image(w, k)
    w.close()
    assert_same_file(reference_image, path)


def test_resume_from_a_closed_time_limited_file(tmp_path, reference_image):
    path = tmp_path / "result.vtkhdf"
    w = _new_image(path)
    for k in range(6):
        _step_image(w, k)
    w.close("time_limited")
    before = sha256_file(path)
    w = _resume_image(path, 4)
    assert not path.exists() and not (tmp_path / "result.vtkhdf.sha256").exists()
    for k in range(5, N):
        _step_image(w, k)
    w.close()
    assert_same_file(reference_image, path)
    with h5py.File(path, "r") as f:
        log = json.loads(f["meta"].attrs["resumes"])
    assert log[0]["previous_status"] == "time_limited" and log[0]["previous_sha256"] == before
    assert (tmp_path / "result.vtkhdf.pre-resume-1.bak").exists()


def test_resume_twice(tmp_path, reference_image):
    path = tmp_path / "result.vtkhdf"
    w = _new_image(path)
    for k in range(4):
        _step_image(w, k)
    w.f.close()
    w = _resume_image(path, 2)
    for k in range(3, 6):
        _step_image(w, k)
    w.f.close()
    w = _resume_image(path, 4)
    for k in range(5, N):
        _step_image(w, k)
    w.close()
    assert_same_file(reference_image, path)
    with h5py.File(path, "r") as f:
        assert int(f.attrs["resume_count"]) == 2 and len(json.loads(f["meta"].attrs["resumes"])) == 2


@pytest.mark.parametrize("killed_after, checkpoint", [(3, 3), (6, 2), (8, 5)])
def test_kill_and_resume_equals_uninterrupted_with_long_tables(tmp_path, reference_mesh, killed_after, checkpoint):
    path = tmp_path / "result.vtkhdf"
    w = _new_mesh(path)
    for k in range(killed_after + 1):
        _step_mesh(w, k)
    w.f.close()
    w = _resume_mesh(path, checkpoint)
    for k in range(checkpoint + 1, N):
        _step_mesh(w, k)
    w.close()
    assert not _errors(path)
    assert_same_file(reference_mesh, path)             # crack tips (speed refitted), substeps with their codes


# --- refusals leave the file as it was ----------------------------------------------------------------

def _partial_image(tmp_path, n=5):
    path = tmp_path / "result.vtkhdf"
    w = _new_image(path)
    for k in range(n):
        _step_image(w, k)
    w.f.close()
    return path, tmp_path / "result.vtkhdf.partial"


def _refused(tmp_path, fn, match):
    before = {p.name: sha256_file(p) for p in tmp_path.iterdir()}
    with pytest.raises(ResumeRefused, match=match):
        fn()
    assert {p.name: sha256_file(p) for p in tmp_path.iterdir()} == before          # nothing renamed, cut or backed up


def test_refuses_a_complete_file(tmp_path, reference_image):
    shutil.copy(reference_image, tmp_path / "result.vtkhdf")
    _refused(tmp_path, lambda: _resume_image(tmp_path / "result.vtkhdf", 4), "complete")


def test_refuses_a_checkpoint_that_is_not_in_the_file(tmp_path):
    path, _ = _partial_image(tmp_path, 5)
    _refused(tmp_path, lambda: _resume_image(path, 6), "increment 6 is not in the file")
    _refused(tmp_path, lambda: _resume_image(path, -1), "not in the file")


def test_refuses_another_run(tmp_path):
    path, _ = _partial_image(tmp_path)
    _refused(tmp_path, lambda: ResultWriter.resume(path, _grid(), _grid_fields(), upto_increment=2, run_id="other"), "run identity")


def test_refuses_another_mesh(tmp_path):
    path, _ = _partial_image(tmp_path)
    other = ImageGrid((3, 2, 2), (0.25, 0.5, 0.5), cell_type="H8")
    _refused(tmp_path, lambda: ResultWriter.resume(path, other, _grid_fields(), upto_increment=2, run_id="resume-test"), "mesh differs")
    big = ImageGrid((4, 2, 2), (0.5, 0.5, 0.5), cell_type="H8")
    _refused(tmp_path, lambda: ResultWriter.resume(path, big, _grid_fields(), upto_increment=2, run_id="resume-test"), "mesh differs")
    _refused(tmp_path, lambda: ResultWriter.resume(path, _mesh(), _mesh_fields(), upto_increment=2, run_id="resume-test"), "holds ImageData")


def test_refuses_other_fields(tmp_path):
    path, _ = _partial_image(tmp_path)
    fewer = [f for f in _grid_fields() if f.name != "damage"]
    _refused(tmp_path, lambda: ResultWriter.resume(path, _grid(), fewer, upto_increment=2, run_id="resume-test"), "not declared")
    more = _grid_fields() + [Field("extra", "node", "e", "1", "state")]
    _refused(tmp_path, lambda: ResultWriter.resume(path, _grid(), more, upto_increment=2, run_id="resume-test"), "lacks the declared")


def test_refuses_both_files_and_missing_file(tmp_path):
    path, partial = _partial_image(tmp_path)
    shutil.copy(partial, path)
    _refused(tmp_path, lambda: _resume_image(path, 2), "both exist")
    other = tmp_path / "none" / "result.vtkhdf"
    other.parent.mkdir()
    with pytest.raises(ResumeRefused, match="no result file"):
        _resume_image(other, 2)


def test_refuses_a_changing_mesh_and_a_setup_file(tmp_path):
    path, partial = _partial_image(tmp_path)
    with h5py.File(partial, "r+") as f:
        f.attrs["mesh_varies"] = True
    _refused(tmp_path, lambda: _resume_image(path, 2), "mesh_varies")
    with h5py.File(partial, "r+") as f:
        f.attrs["mesh_varies"] = False
        f.attrs["profile"] = "setup"
    _refused(tmp_path, lambda: _resume_image(path, 2), "profile")


def test_refuses_when_stored_steps_are_missing_below_the_checkpoint(tmp_path):
    path, partial = _partial_image(tmp_path, 7)
    with h5py.File(partial, "r+") as f:                    # the field rows of the last steps never reached the disk
        d = f["VTKHDF/PointData/damage"]
        d.resize(d.shape[0] - 2, axis=0)
    _refused(tmp_path, lambda: _resume_image(path, 6), "complete stored steps")


def test_no_backup_option(tmp_path):
    path, partial = _partial_image(tmp_path)
    w = _resume_image(path, 2, backup=False)
    w.close("failed")
    assert not list(tmp_path.glob("*.bak"))


def test_refuses_a_damaged_file(tmp_path):
    path, partial = _partial_image(tmp_path)
    garbage = partial.read_bytes()[:200]
    partial.write_bytes(garbage)                                                        # cut in the middle of a write
    _refused(tmp_path, lambda: _resume_image(path, 2), "cannot open|damaged")
    other = tmp_path / "b"
    other.mkdir()
    path2, partial2 = _partial_image(other)
    with h5py.File(partial2, "r+") as f:
        del f["history/increment"]
    _refused(partial2.parent, lambda: _resume_image(path2, 2), "damaged or incomplete")


def test_a_failure_while_cutting_restores_the_original_file(tmp_path, monkeypatch):
    path, partial = _partial_image(tmp_path, 6)
    before = sha256_file(partial)

    def boom(self, *a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(ResultWriter, "_restore", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        _resume_image(path, 2)
    assert sha256_file(partial) == before and not path.exists()
    assert (tmp_path / "result.vtkhdf.partial.pre-resume-1.bak").exists()
