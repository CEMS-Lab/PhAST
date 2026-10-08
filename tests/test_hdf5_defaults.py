"""Focused storage-default, provenance, and legacy-reader contracts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from examples.plasticity_interface_beta import _promoted_result_utils as promoted
from examples.quasistatic import _run_utils as quasistatic
from phast.config.config import OutputConfig
from phast.result import Result, ResultLoadError
from phast.solid_mechanics_runners import _common as solid
from phast.utils.io_utils import load_state_from_h5, load_state_from_zarr


@pytest.fixture
def model() -> tuple[SimpleNamespace, SimpleNamespace]:
    mesh = SimpleNamespace(
        nodes=torch.tensor([[0., 0.], [1., 0.], [0., 1.]], dtype=torch.float64),
        elements=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        node_sets={},
    )
    material = SimpleNamespace(
        E=210000., nu=0.3, Gc=2.7, l0=0.1, rho=1.,
        energy_split="amor", pf_model="AT2", plane_stress=False,
    )
    return mesh, material


def write_history(
    out_dir: Path,
    family: str,
    model: tuple[SimpleNamespace, SimpleNamespace],
    fmt: str | None = None,
) -> list[Path]:
    mesh, material = model
    fields = {
        "damage_nodal": np.array([[0., 0.1, 0.2], [0., 0.25, 1.]]),
        "displacement": np.zeros((2, 3, 2)),
        "H_elem": np.ones((2, 1)),
    }
    steps = [7, 21]
    if family == "quasistatic":
        options = {} if fmt is None else {"fmt": fmt}
        writers = quasistatic.open_trajectory_writers(
            out_dir, mesh, material, enabled=True, **options,
        )
        try:
            for index, step in enumerate(steps):
                for writer in writers:
                    writer.write(
                        step, mesh,
                        u=torch.from_numpy(fields["displacement"][index]),
                        d=torch.from_numpy(fields["damage_nodal"][index]),
                        psi_plus_e=torch.ones(1),
                        H_e=torch.from_numpy(fields["H_elem"][index]),
                        time_s=float(step),
                    )
        finally:
            for writer in writers:
                writer.close(len(steps))
        return [Path(writer.path) for writer in writers]
    options = {} if fmt is None else {"trajectory_format": fmt}
    if family == "promoted":
        path = promoted.write_trajectory(
            out_dir, nodes=mesh.nodes.numpy(), elements=mesh.elements.numpy(),
            snapshots=[
                (step, {name: array[index] for name, array in fields.items()})
                for index, step in enumerate(steps)
            ],
            metadata={"example": "storage_contract"}, **options,
        )
    else:
        path = solid.write_solid_trajectory(
            out_dir, mesh=mesh, steps=[{"step": step} for step in steps],
            fields=fields, **options,
        )
    return [path]


def result_at(path: Path) -> Result:
    (path / "run_metadata.json").write_text("{}", encoding="utf-8")
    return Result(path)


def test_config_defaults_keep_trajectory_saving_opt_in() -> None:
    output = OutputConfig()
    assert output.trajectory_format == "h5"
    assert not output.trajectory
    assert not output.h5


def test_disabled_quasistatic_output_creates_no_store(tmp_path, model, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "h5py", None)
    monkeypatch.setitem(sys.modules, "zarr", None)
    assert quasistatic.open_trajectory_writers(tmp_path, *model, enabled=False) == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("family", ["quasistatic", "promoted", "solid"])
def test_default_writers_use_h5_without_zarr(tmp_path, model, monkeypatch, family) -> None:
    monkeypatch.setitem(sys.modules, "zarr", None)
    paths = write_history(tmp_path, family, model)
    assert paths == [tmp_path / "training_data.h5"]
    assert not (tmp_path / "training_data.zarr").exists()
    with h5py.File(paths[0], "r") as root:
        assert root.attrs["format"].endswith(".h5")
        assert sorted(root["simulation_data/steps"]) == ["step_0007", "step_0021"]
    result = result_at(tmp_path)
    np.testing.assert_allclose(result.field("damage"), [0., 0.25, 1.])
    np.testing.assert_allclose(result.field("damage", step=7), [0., 0.1, 0.2])
    assert result.mesh()["source"] == "training_data.h5"
    assert result.mesh()["n_nodes"] == 3


@pytest.mark.parametrize("family", ["quasistatic", "promoted", "solid"])
def test_explicit_zarr_remains_readable(tmp_path, model, family) -> None:
    pytest.importorskip("zarr")
    paths = write_history(tmp_path, family, model, "zarr")
    assert paths == [tmp_path / "training_data.zarr"]
    assert not (tmp_path / "training_data.h5").exists()
    result = result_at(tmp_path)
    np.testing.assert_allclose(result.field("damage"), [0., 0.25, 1.])
    np.testing.assert_allclose(result.field("damage", step=7), [0., 0.1, 0.2])
    assert result.mesh()["source"] == "training_data.zarr"


@pytest.mark.parametrize("family", ["quasistatic", "promoted", "solid"])
def test_missing_h5_backend_never_falls_back(tmp_path, model, monkeypatch, family) -> None:
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError):
        write_history(tmp_path, family, model)
    assert not (tmp_path / "training_data.zarr").exists()
    assert not (tmp_path / "training_data.h5").exists()


@pytest.mark.parametrize("family", ["quasistatic", "promoted", "solid"])
def test_invalid_format_creates_no_store(tmp_path, model, family) -> None:
    with pytest.raises(ValueError, match="format"):
        write_history(tmp_path, family, model, "invalid")
    assert not list(tmp_path.iterdir())


def test_both_is_explicit_and_restart_readers_keep_working(tmp_path, model) -> None:
    pytest.importorskip("zarr")
    paths = write_history(tmp_path, "quasistatic", model, "both")
    assert {path.name for path in paths} == {"training_data.h5", "training_data.zarr"}
    for loader, name in (
        (load_state_from_h5, "training_data.h5"),
        (load_state_from_zarr, "training_data.zarr"),
    ):
        state = loader(str(tmp_path / name))
        assert state["step"] == 21
        assert state["time_s"] == 21.
        np.testing.assert_allclose(state["d"].numpy(), [0., 0.25, 1.])


def test_h5_preference_preserves_and_does_not_open_old_zarr(tmp_path, model, monkeypatch) -> None:
    old_store = tmp_path / "training_data.zarr"
    old_store.mkdir()
    sentinel = old_store / "historical.txt"
    sentinel.write_bytes(b"historical data, not an HDF5 run")
    write_history(tmp_path, "promoted", model)
    monkeypatch.setitem(sys.modules, "zarr", None)
    result = result_at(tmp_path)
    np.testing.assert_allclose(result.field("damage"), [0., 0.25, 1.])
    assert result.mesh()["source"] == "training_data.h5"
    assert sentinel.read_bytes() == b"historical data, not an HDF5 run"


def test_result_does_not_mix_fields_from_two_stores(tmp_path, model, monkeypatch) -> None:
    write_history(tmp_path, "promoted", model)
    result = result_at(tmp_path)
    monkeypatch.setattr(result, "_discover_zarr_fields", lambda: {"stress"})
    assert "stress" not in result.field_names()
    with pytest.raises(ResultLoadError, match="Unknown field"):
        result.field("stress")


def test_missing_h5_mesh_does_not_select_an_old_zarr_mesh(tmp_path, monkeypatch) -> None:
    with h5py.File(tmp_path / "training_data.h5", "w"):
        pass
    result = result_at(tmp_path)
    monkeypatch.setattr(result, "_zarr_mesh_metadata", lambda: {"n_nodes": 999})
    with pytest.raises(ResultLoadError, match="No mesh metadata"):
        result.mesh()


def test_corrupt_h5_is_not_hidden_by_zarr(tmp_path, monkeypatch) -> None:
    (tmp_path / "training_data.h5").write_bytes(b"not HDF5")
    result = result_at(tmp_path)
    monkeypatch.setattr(result, "_discover_zarr_fields", lambda: {"damage_nodal"})
    with pytest.raises(OSError):
        result.field_names()


def test_promoted_zarr_compatibility_entry_point(tmp_path, model) -> None:
    pytest.importorskip("zarr")
    mesh, _ = model
    path = promoted.write_zarr_trajectory(
        tmp_path, nodes=mesh.nodes.numpy(), elements=mesh.elements.numpy(),
        snapshots=[(3, {"damage_nodal": np.zeros(3)})], metadata={},
    )
    assert path.name == "training_data.zarr"
    np.testing.assert_array_equal(result_at(tmp_path).field("damage", step=3), np.zeros(3))


@pytest.mark.parametrize("fmt", [None, "h5", "zarr", "both"])
def test_lockfile_records_actual_outputs_not_requested_default(tmp_path, model, fmt) -> None:
    paths = []
    if fmt is not None:
        if fmt in {"zarr", "both"}:
            pytest.importorskip("zarr")
        paths = write_history(tmp_path, "quasistatic", model, fmt)
    solid.write_run_lockfile(
        tmp_path, config={"output": {"trajectory_format": "h5"}},
        command="storage-contract", trajectory_paths=paths,
    )
    lockfile = json.loads((tmp_path / "run_lockfile.json").read_text())
    assert lockfile["trajectory_format"] == fmt
    assert lockfile["requested_trajectory_format"] == "h5"
    assert lockfile["trajectory_files"] == [path.name for path in paths]


def test_lockfile_does_not_relabel_a_previous_run(tmp_path, model) -> None:
    write_history(tmp_path, "solid", model)
    solid.write_run_lockfile(tmp_path, config={}, command="no-trajectory-run")
    lockfile = json.loads((tmp_path / "run_lockfile.json").read_text())
    assert lockfile["trajectory_format"] is None
    assert lockfile["trajectory_files"] == []


def test_lockfile_rejects_missing_trajectory_output(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing or unsupported"):
        solid.write_run_lockfile(
            tmp_path, config={}, command="missing-output",
            trajectory_paths=[tmp_path / "training_data.h5"],
        )
    assert not (tmp_path / "run_lockfile.json").exists()


def test_solid_h5_rejects_misaligned_history_before_writing(tmp_path, model) -> None:
    with pytest.raises(ValueError, match="one entry per trajectory step"):
        solid.write_solid_trajectory(
            tmp_path, mesh=model[0], steps=[{"step": 1}],
            fields={"damage_nodal": np.zeros((2, 3))},
        )
    assert not (tmp_path / "training_data.h5").exists()
