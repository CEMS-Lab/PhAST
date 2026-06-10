from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from phast.config import (
    DeviceConfig,
    GeometryConfig,
    LoadingConfig,
    MaterialConfig,
    OutputConfig,
    ProblemConfig,
    SolverSettings,
)
from phast.provenance import build_run_lockfile, write_run_lockfile

pytestmark = pytest.mark.docs


def _config() -> ProblemConfig:
    return ProblemConfig(
        schema_version=1,
        name="lockfile smoke",
        reference="unit test",
        geometry=GeometryConfig(type="rectangular_sent"),
        material=MaterialConfig(E=32000.0, nu=0.2, Gc=0.003, l0=0.5),
        loading=LoadingConfig(num_steps=2, dt=1e-7),
        solver=SolverSettings(solver_type="explicit", damage_every=1),
        output=OutputConfig(h5=False, print_every=1),
        device=DeviceConfig(device="cpu"),
        boundary_conditions=[],
    )


def test_build_run_lockfile_captures_resolved_config_and_runtime(tmp_path):
    cfg_path = tmp_path / "problem.yaml"
    cfg_path.write_text("schema_version: 1\n", encoding="utf-8")
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    mesh = SimpleNamespace(
        mesh_path="mesh.msh",
        n_nodes=4,
        n_elems=2,
        h_min=0.25,
        node_sets={"left": [0], "right": [1]},
    )
    material = SimpleNamespace(
        E=32000.0,
        nu=0.2,
        Gc=0.003,
        l0=0.5,
        rho=2.45e-9,
        energy_split="spectral",
        pf_model="AT2",
        eta_residual=1e-7,
        plane_stress=False,
        kinematics="small_strain",
    )
    solver = SimpleNamespace(
        solver_type="explicit",
        time_integrator="central_difference",
        dt_safety=0.8,
        damage_every=1,
        H_update_method="hard_max",
        backend="auto",
    )
    ctx = SimpleNamespace(device="cpu", dtype="torch.float64")
    args = argparse.Namespace(config=str(cfg_path), device="cpu", num_steps=2)

    lock = build_run_lockfile(
        config=_config(),
        config_path=str(cfg_path),
        output_dir=str(out_dir),
        args=args,
        mesh=mesh,
        material=material,
        solver_config=solver,
        ctx=ctx,
    )

    assert lock["lockfile_schema"] == "phast.run_lockfile.v1"
    assert lock["input"]["config_sha256"]
    assert lock["resolved_config"]["name"] == "lockfile smoke"
    assert lock["runtime"]["cli_args"]["num_steps"] == 2
    assert "torch" in lock["runtime"]["dependencies"]
    assert "dirty" in lock["runtime"]["git"]
    assert lock["resolved_objects"]["mesh"]["node_sets"] == ["left", "right"]
    assert lock["resolved_objects"]["material"]["pf_model"] == "AT2"
    assert lock["resolved_objects"]["solver"]["solver_type"] == "explicit"
    assert lock["resolved_objects"]["device"]["device"] == "cpu"


def test_write_run_lockfile_roundtrips_json(tmp_path):
    cfg_path = tmp_path / "problem.yaml"
    cfg_path.write_text("schema_version: 1\n", encoding="utf-8")
    out = tmp_path / "run_lockfile.json"

    write_run_lockfile(
        out,
        config=_config(),
        config_path=str(cfg_path),
        output_dir=str(tmp_path),
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["resolved_config"]["schema_version"] == 1
    assert payload["input"]["config_file"] == str(cfg_path)
