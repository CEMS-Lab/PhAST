"""Validator and recovery for layout-2.0 result files.

The conformance fixtures from ``tests/result_v2/fixtures.py`` must validate;
broken copies of them must give the expected ERROR code; recovery of the
interrupted fixtures must trim them to their complete steps without touching
the source file.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from phast.io.result import Report, recover, validate
from phast.io.result.cli import main as cli_main
from result_v2.fixtures import FIXTURES, KILL_STAGES, build_all

LIBVER = ("earliest", "v114")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("result_v2_validate")
    return out, build_all(out)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def broken(built, tmp_path, name="t3_notched_plate"):
    """A copy of a fixture for one mutation."""
    out, expected = built
    dst = tmp_path / f"broken_{name}.vtkhdf"
    shutil.copyfile(out / expected[name]["file"], dst)
    return dst


def error_codes(report: Report) -> set[str]:
    return report.codes("ERROR")


# --- fixtures validate ------------------------------------------------------

@pytest.mark.parametrize("name", list(FIXTURES))
def test_complete_fixture_has_no_error(built, name):
    out, expected = built
    report = validate(out / expected[name]["file"])
    assert report.ok, report.format_text()
    assert not report.warnings, report.format_text()


@pytest.mark.parametrize("stage", KILL_STAGES)
def test_interrupted_fixture_is_running(built, stage):
    out, expected = built
    path = out / expected[f"interrupted_{stage}"]["file"]
    assert error_codes(validate(path)) == {"status-running"}
    report = validate(path, allow_running=True)
    assert report.ok, report.format_text()
    assert "status-running" in report.codes("INFO")


def test_optional_groups_are_reported_as_info(built):
    out, expected = built
    report = validate(out / expected["t3_notched_plate"]["file"])
    assert {"/interfaces", "/quadrature"} <= {f.path for f in report.infos if f.code == "optional-absent"}


# --- recovery ---------------------------------------------------------------

@pytest.mark.parametrize("stage", KILL_STAGES)
def test_recover_trims_interrupted_fixture(built, tmp_path, stage):
    out, expected = built
    src = out / expected[f"interrupted_{stage}"]["file"]
    before = sha256(src)
    with h5py.File(src, "r") as f:
        n_cells = int(f["VTKHDF/NumberOfCells"][0])
        rows_before = f["VTKHDF/CellData/H"].shape[0]
        values_before = f["VTKHDF/Steps/Values"].shape[0]
        hist_before = f["history/step_index"][...].tolist()
    result = recover(src, tmp_path / "recovered.vtkhdf", use_h5clear=False)
    assert sha256(src) == before
    assert result.report.ok, result.report.format_text()
    assert result.flag_cleared_with == "h5py"
    with h5py.File(result.output, "r") as f:
        assert f.attrs["status"] == "recovered"
        assert int(f.attrs["recovered_from_steps"]) == 2 and "finished" in f.attrs
        st = f["VTKHDF/Steps"]
        assert int(st.attrs["NSteps"]) == 2 == result.nsteps
        assert st["Values"].shape[0] == 2 and st["CellDataOffsets/H"].shape[0] == 2
        assert f["VTKHDF/CellData/H"].shape[0] == 2 * n_cells
        # the row of increment 3 (no fields stored) follows the last kept step and is dropped
        assert hist_before == [0, -1, 1, -1]
        assert f["history/step_index"][...].tolist() == [0, -1, 1]
        assert f["history/time"].shape[0] == f["history/reaction/top"].shape[0] == 3
    if stage == "after_rows":
        assert rows_before == 3 * n_cells
    if stage == "after_offsets":
        assert values_before == 3


def test_recover_rewrites_a_contiguous_dataset(built, tmp_path):
    """A dataset that cannot be resized is rewritten with its rows, dtype and attributes."""
    path = broken(built, tmp_path, "interrupted_after_rows")
    with h5py.File(path, "r+", libver=LIBVER) as f:
        cd = f["VTKHDF/CellData"]
        n_cells = int(f["VTKHDF/NumberOfCells"][0])
        rows = cd["H"][...]
        attrs = dict(cd["H"].attrs)
        del cd["H"]
        ds = cd.create_dataset("H", data=rows)                  # contiguous, not resizable
        for k, v in attrs.items():
            ds.attrs[k] = v
        assert ds.chunks is None and ds.shape[0] == 3 * n_cells
    result = recover(path, tmp_path / "rec.vtkhdf", use_h5clear=False)
    assert result.report.ok, result.report.format_text()
    with h5py.File(result.output, "r") as f:
        ds = f["VTKHDF/CellData/H"]
        assert ds.chunks is None and ds.dtype == np.float64 and ds.shape[0] == 2 * n_cells
        assert dict(ds.attrs) == attrs
        np.testing.assert_array_equal(ds[...], rows[: 2 * n_cells])
        assert f["state/H"].shape == ds.shape


def test_recover_default_output_and_refuses_to_overwrite(built, tmp_path):
    out, expected = built
    src = tmp_path / "run.vtkhdf"
    shutil.copyfile(out / expected["interrupted_after_rows"]["file"], src)
    result = recover(src)
    assert Path(result.output) == tmp_path / "run.recovered.vtkhdf"
    with pytest.raises(FileExistsError):
        recover(src)
    with pytest.raises(ValueError):
        recover(src, src)


@pytest.mark.skipif(shutil.which("h5clear") is None, reason="h5clear not on PATH")
def test_recover_with_h5clear(built, tmp_path):
    out, expected = built
    src = out / expected["interrupted_after_offsets"]["file"]
    before = sha256(src)
    result = recover(src, tmp_path / "recovered.vtkhdf", use_h5clear=True)
    assert result.flag_cleared_with == "h5clear" and result.report.ok and result.nsteps == 2
    assert sha256(src) == before


def test_recover_trims_derived_arrays(built, tmp_path):
    """A file cut after step 3: the derived arrays of later increments go."""
    path = broken(built, tmp_path, "sent_lagged")
    with h5py.File(path, "r+", libver=LIBVER) as f:
        f["VTKHDF/Steps"].attrs["NSteps"] = 3          # steps 3 to 5 become an incomplete tail
        f.attrs["status"] = "running"
    result = recover(path, tmp_path / "rec.vtkhdf", use_h5clear=False)
    assert result.report.ok, result.report.format_text()
    with h5py.File(result.output, "r") as f:
        assert f["history/step_index"][...].tolist() == [0, 1, 2]
        assert f["derived/stress/step_index"][...].tolist() == [-1, -1, 0]
        assert f["derived/stress/values"].shape[0] == 1
        n_points = int(f["VTKHDF/NumberOfPoints"][0])
        assert f["VTKHDF/PointData/damage"].shape[0] == 3 * n_points


# --- broken copies ------------------------------------------------------------

def _replace(group, name, data):
    attrs = dict(group[name].attrs)
    del group[name]
    ds = group.create_dataset(name, data=data)
    for k, v in attrs.items():
        ds.attrs[k] = v


def mutate_dangling_link(f):
    del f["state/H"]
    f["state/H"] = h5py.SoftLink("/VTKHDF/CellData/H_missing")


def mutate_link_outside(f):
    f["state/extra"] = h5py.SoftLink("/history/time")


def mutate_float32_state(f):
    cd = f["VTKHDF/CellData"]
    _replace(cd, "H", cd["H"][...].astype(np.float32))


def mutate_float32_points(f):
    _replace(f["VTKHDF"], "Points", f["VTKHDF/Points"][...].astype(np.float32))


def mutate_float32_history(f):
    _replace(f["history"], "time", f["history/time"][...].astype(np.float32))


def mutate_float32_material(f):
    f["materials/phases/0"].attrs["E"] = np.float32(210000.0)


def mutate_short_array(f):
    ds = f["VTKHDF/CellData/H"]
    ds.resize(ds.shape[0] - 1, axis=0)                     # one row short of the last step


def mutate_short_steps(f):
    f["VTKHDF/Steps/Values"].resize(2, axis=0)


def mutate_field_units(f):
    del f["VTKHDF/PointData/damage"].attrs["units"]


def mutate_history_units(f):
    del f["history/energy/elastic"].attrs["units"]


def mutate_plugin_filter(f):
    # lzf (filter id 32000) ships with h5py only, so it stands for any plugin filter
    f["meta"].create_dataset("plugin_compressed", data=np.arange(100.0), compression="lzf")


def mutate_scaleoffset(f):
    f["meta"].create_dataset("lossy", data=np.arange(100.0), scaleoffset=2)


def mutate_abs_posix(f):
    f["meta"].attrs["source_manifest"] = "/home/x/manifest.json"


def mutate_abs_windows(f):
    f["meta"].attrs["source_manifest"] = "C:\\runs\\x.json"


def mutate_abs_unc(f):
    f["meta"].attrs["source_file"] = "\\\\server\\share\\run.vtkhdf"


def mutate_virtual(f):
    layout = h5py.VirtualLayout(shape=(5,), dtype="f8")
    layout[:] = h5py.VirtualSource(".", "history/time", shape=(5,))
    f["meta"].create_virtual_dataset("virtual_time", layout)


def mutate_major(f):
    f.attrs["phast_schema_version"] = "3.0.0"


def mutate_mesh_varies(f):
    f.attrs["mesh_varies"] = True


def mutate_lagged(f):
    f.attrs["history_coupling"] = "lagged"


def mutate_missing_group(f):
    del f["loading"]


def mutate_missing_attribute(f):
    del f.attrs["run_id"]


def mutate_setup_with_steps(f):
    f.attrs["profile"] = "setup"


def mutate_unknown_kind(f):
    f["VTKHDF/PointData/damage"].attrs["kind"] = "proposal"


BROKEN = {
    "dangling_link": (mutate_dangling_link, "dangling-link"),
    "link_outside": (mutate_link_outside, "link-outside-vtkhdf"),
    "float32_state": (mutate_float32_state, "dtype"),
    "float32_points": (mutate_float32_points, "dtype"),
    "float32_history": (mutate_float32_history, "dtype"),
    "float32_material": (mutate_float32_material, "dtype"),
    "short_array": (mutate_short_array, "array-short"),
    "short_steps": (mutate_short_steps, "steps-short"),
    "missing_field_units": (mutate_field_units, "field-attribute-missing"),
    "missing_history_units": (mutate_history_units, "history-units"),
    "plugin_filter_lzf": (mutate_plugin_filter, "filter-not-allowed"),
    "scaleoffset_filter": (mutate_scaleoffset, "filter-not-allowed"),
    "absolute_path_posix": (mutate_abs_posix, "absolute-path"),
    "absolute_path_windows": (mutate_abs_windows, "absolute-path"),
    "absolute_path_unc": (mutate_abs_unc, "absolute-path"),
    "virtual_dataset": (mutate_virtual, "virtual-dataset"),
    "unknown_major": (mutate_major, "schema-version-major"),
    "mesh_varies": (mutate_mesh_varies, "mesh-varies"),
    "lagged_without_input": (mutate_lagged, "damage-input-missing"),
    "missing_group": (mutate_missing_group, "missing-group"),
    "missing_attribute": (mutate_missing_attribute, "missing-attribute"),
    "setup_with_steps": (mutate_setup_with_steps, "setup-nsteps"),
    "unknown_kind": (mutate_unknown_kind, "field-attribute-value"),
}


@pytest.mark.parametrize("case", list(BROKEN))
def test_broken_copy_gives_expected_error(built, tmp_path, case):
    mutate, code = BROKEN[case]
    path = broken(built, tmp_path)
    with h5py.File(path, "r+", libver=LIBVER) as f:
        mutate(f)
    report = validate(path)
    assert code in error_codes(report), report.format_text()
    assert "read-error" not in error_codes(report), report.format_text()


def test_hdf5_paths_and_relative_references_are_not_flagged(built, tmp_path):
    path = broken(built, tmp_path)
    with h5py.File(path, "r+", libver=LIBVER) as f:
        f["meta"].attrs["source_manifest"] = "checkpoints/manifest.json"
        f["meta"].attrs["target_path"] = "/VTKHDF/CellData/H"
    assert validate(path).ok


# --- warnings -----------------------------------------------------------------

def test_cell_mean_history_on_q4_warns(built, tmp_path):
    path = broken(built, tmp_path, "q4_plate")
    with h5py.File(path, "r+", libver=LIBVER) as f:
        cd = f["VTKHDF/CellData"]
        _replace(cd, "H", cd["H"][...].mean(axis=1))
        cd["H"].attrs["location"] = "cell"
        cd["H"].attrs["components"] = "H"
    report = validate(path)
    assert report.ok and "history-cell-mean" in report.codes("WARN"), report.format_text()


def test_irregular_step_index_warns(built, tmp_path):
    path = broken(built, tmp_path)                        # stored at increments 0, 2, 4
    with h5py.File(path, "r+", libver=LIBVER) as f:
        f["history/step_index"][...] = [0, 1, -1, -1, 2]  # stored at increments 0, 1, 4
    report = validate(path)
    assert report.ok and "step-index-gaps" in report.codes("WARN"), report.format_text()


def test_two_component_vector_and_unreadable_codes_warn(built, tmp_path):
    path = broken(built, tmp_path, "sent_lagged")
    with h5py.File(path, "r+", libver=LIBVER) as f:
        pd = f["VTKHDF/PointData"]
        _replace(pd, "displacement", pd["displacement"][..., :2])
        pd["displacement"].attrs["components"] = "x,y"
        column = f["history"].create_dataset("mode", data=np.zeros(6, np.int8))
        column.attrs["units"] = "1"
        column.attrs["codes"] = "not a JSON object"
    report = validate(path)
    assert {"vector-2-components", "categorical-codes"} <= report.codes("WARN"), report.format_text()


# --- command line ---------------------------------------------------------------

def test_cli_validate_exit_codes_and_json(built, tmp_path, capsys):
    out, expected = built
    good = out / expected["t3_notched_plate"]["file"]
    assert cli_main(["validate", str(good)]) == 0
    assert "OK" in capsys.readouterr().out
    assert cli_main(["validate", "--json", str(good)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True and data["counts"]["ERROR"] == 0
    bad = broken(built, tmp_path)
    with h5py.File(bad, "r+", libver=LIBVER) as f:
        mutate_dangling_link(f)
    assert cli_main(["validate", "--json", str(bad)]) == 1
    data = json.loads(capsys.readouterr().out)
    assert any(x["code"] == "dangling-link" and x["level"] == "ERROR" for x in data["findings"])
    running = out / expected["interrupted_after_rows"]["file"]
    assert cli_main(["validate", str(running)]) == 1
    assert "ERROR:" in capsys.readouterr().out
    assert cli_main(["validate", "--running-ok", str(running)]) == 0


def test_cli_recover(built, tmp_path, capsys):
    out, expected = built
    dst = tmp_path / "r.vtkhdf"
    assert cli_main(["recover", str(out / expected["interrupted_after_offsets"]["file"]), "-o", str(dst)]) == 0
    assert "NSteps 2 -> 2" in capsys.readouterr().out and dst.exists()


def test_python_m_phast_result_dispatches(built):
    out, expected = built
    env_path = str(ROOT / "src")
    proc = subprocess.run([sys.executable, "-m", "phast", "result", "validate", "--json",
                           str(out / expected["setup_profile"]["file"])],
                          capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": env_path})
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["ok"] is True
