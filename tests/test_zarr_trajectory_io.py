from types import SimpleNamespace

import pytest
import torch


zarr = pytest.importorskip("zarr")


def _tiny_mesh():
    return SimpleNamespace(
        nodes=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float64,
        ),
        elements=torch.tensor([[0, 1, 2]], dtype=torch.long),
        node_sets={"left": torch.tensor([0, 2], dtype=torch.long)},
    )


def _tiny_material():
    return SimpleNamespace(
        Gc=2.7e-3,
        l0=0.5,
        E=210000.0,
        nu=0.3,
        rho=7.8e-9,
        energy_split="spectral",
        pf_model="AT2",
        plane_stress=True,
    )


def test_zarr_trajectory_writer_preserves_solver_snapshot_contract(tmp_path):
    from phast.io_utils import (
        init_zarr,
        load_state_from_zarr,
        write_zarr_snapshot,
    )

    mesh = _tiny_mesh()
    material = _tiny_material()
    path = tmp_path / "training_data.zarr"

    root = init_zarr(str(path), mesh, material)
    write_zarr_snapshot(
        root,
        step=0,
        mesh=mesh,
        u=torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.2]]),
        d=torch.tensor([0.0, 0.25, 0.75]),
        psi_plus_e=torch.tensor([1.5]),
        H_e=torch.tensor([2.5]),
        eps_xx=torch.tensor([0.01]),
        eps_yy=torch.tensor([0.02]),
        gam_xy=torch.tensor([0.03]),
        sxx=torch.tensor([1.0]),
        syy=torch.tensor([2.0]),
        sxy=torch.tensor([3.0]),
        H_nodal=torch.tensor([0.0, 0.1, 0.2]),
        velocity=torch.ones(3, 2),
        acceleration=torch.full((3, 2), 2.0),
        energies={"elastic": 1.0, "fracture": 2.0, "total": 3.0},
        time_s=4.0e-6,
    )
    write_zarr_snapshot(
        root,
        step=2,
        mesh=mesh,
        u=torch.tensor([[1.0, 1.0], [1.1, 1.0], [1.0, 1.2]]),
        d=torch.tensor([0.1, 0.35, 0.85]),
        psi_plus_e=torch.tensor([1.7]),
        H_e=torch.tensor([2.8]),
        H_nodal=torch.tensor([0.3, 0.4, 0.5]),
        velocity=torch.full((3, 2), 3.0),
        acceleration=torch.full((3, 2), 4.0),
        energies={"elastic": 4.0, "fracture": 5.0, "total": 9.0},
        time_s=8.0e-6,
    )
    root.attrs["num_steps"] = 2

    reopened = zarr.open(str(path), mode="r")
    assert reopened.attrs["format"] == "phast.trajectory.zarr"
    assert reopened.attrs["num_steps"] == 2

    mesh_group = reopened["simulation_data/mesh"]
    assert mesh_group.attrs["n_nodes"] == 3
    assert mesh_group["node_coordinates"].shape == (3, 2)
    assert mesh_group["element_connectivity"].shape == (1, 3)
    assert mesh_group["node_sets/left"][:].tolist() == [0, 2]

    metadata = reopened["simulation_data/metadata"].attrs
    assert metadata["pf_model"] == "AT2"
    assert metadata["plane_stress"] is True

    step = reopened["simulation_data/steps/step_0000"]
    assert step["damage_nodal"].shape == (3,)
    assert step["displacement"].shape == (3, 2)
    assert step["strain"].shape == (1, 3)
    assert step["stress"].shape == (1, 3)
    assert step["velocity"].shape == (3, 2)
    assert step["acceleration"].shape == (3, 2)
    assert step.attrs["time_s"] == pytest.approx(4.0e-6)
    assert step.attrs["energy_total"] == pytest.approx(3.0)

    traj = reopened["simulation_data/trajectory"]
    assert traj.attrs["layout"] == "dense_step_major_v1"
    assert traj["step"][:].tolist() == [0, 2]
    assert traj["damage_nodal"].shape == (2, 3)
    assert traj["displacement"].shape == (2, 3, 2)
    assert traj["time_s"][0] == pytest.approx(4.0e-6)

    state = load_state_from_zarr(str(path))
    assert state["step"] == 2
    assert state["time_s"] == pytest.approx(8.0e-6)
    assert state["u"].shape == (3, 2)
    assert state["H"].shape == (1,)
    state0 = load_state_from_zarr(str(path), step=0)
    assert state0["step"] == 0
    assert state0["time_s"] == pytest.approx(4.0e-6)
