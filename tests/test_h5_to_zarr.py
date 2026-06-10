import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


zarr = pytest.importorskip("zarr")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "h5_to_zarr.py"
SPEC = importlib.util.spec_from_file_location("h5_to_zarr", SCRIPT)
h5_to_zarr = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = h5_to_zarr
SPEC.loader.exec_module(h5_to_zarr)
convert_h5_to_zarr = h5_to_zarr.convert_h5_to_zarr


def _write_legacy_h5(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["case"] = "legacy-smoke"
        sim = f.create_group("simulation_data")
        mesh = sim.create_group("mesh")
        mesh.create_dataset("node_coordinates", data=np.array([[0.0, 0.0], [1.0, 0.0]]))
        mesh.create_dataset("elements", data=np.array([[0, 1, 1]], dtype=np.int64))
        steps = sim.create_group("steps")
        step = steps.create_group("step_0001")
        step.attrs["time"] = 1.0e-6
        step.attrs["energy_elastic"] = 0.25
        step.create_dataset("damage_nodal", data=np.array([0.0, 1.0]))
        step.create_dataset("displacement", data=np.array([[0.0, 0.0], [0.1, 0.0]]))
        step.create_dataset("H_elem", data=np.array([2.0]))


def test_h5_to_zarr_preserves_legacy_solver_hierarchy(tmp_path):
    h5_path = tmp_path / "training_data.h5"
    zarr_path = tmp_path / "training_data.zarr"
    _write_legacy_h5(h5_path)

    out = convert_h5_to_zarr(h5_path, zarr_path)

    assert out == zarr_path
    root = zarr.open(str(zarr_path), mode="r")
    assert root.attrs["case"] == "legacy-smoke"
    assert root.attrs["converter"] == "scripts/h5_to_zarr.py"
    np.testing.assert_allclose(
        root["simulation_data/mesh/node_coordinates"][:],
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )
    step = root["simulation_data/steps/step_0001"]
    assert step.attrs["time"] == pytest.approx(1.0e-6)
    assert step.attrs["energy_elastic"] == pytest.approx(0.25)
    np.testing.assert_allclose(step["damage_nodal"][:], np.array([0.0, 1.0]))
    np.testing.assert_allclose(
        step["displacement"][:],
        np.array([[0.0, 0.0], [0.1, 0.0]]),
    )
    assert (zarr_path / "conversion_manifest.json").is_file()


def test_h5_to_zarr_requires_overwrite_for_existing_store(tmp_path):
    h5_path = tmp_path / "training_data.h5"
    zarr_path = tmp_path / "training_data.zarr"
    _write_legacy_h5(h5_path)
    convert_h5_to_zarr(h5_path, zarr_path)

    with pytest.raises(FileExistsError):
        convert_h5_to_zarr(h5_path, zarr_path)

    convert_h5_to_zarr(h5_path, zarr_path, overwrite=True)
    assert (zarr_path / "conversion_manifest.json").is_file()
