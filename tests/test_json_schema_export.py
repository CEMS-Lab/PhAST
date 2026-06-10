from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from phast.config_schema import dumps_schema, generate_json_schema

pytestmark = pytest.mark.docs

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_PATH = _REPO_ROOT / "configs" / "phast.schema.json"


def test_json_schema_contains_core_solver_enums_and_ranges():
    schema = generate_json_schema()
    props = schema["properties"]

    solver = props["solver"]["properties"]
    assert {"explicit", "quasi_static", "monolithic"} <= set(
        solver["solver_type"]["enum"]
    )
    assert "generalized_alpha" in solver["time_integrator"]["enum"]
    assert solver["dt_safety"]["minimum"] == 0.0
    assert solver["dt_safety"]["maximum"] == 1.0

    output = props["output"]["properties"]
    assert output["h5"]["type"] == "boolean"
    assert output["reaction_component"]["minimum"] == 0
    assert output["reaction_component"]["maximum"] == 1

    assert props["acceptance"]["type"] == ["object", "null"]
    assert props["acceptance"]["additionalProperties"] is True


def test_json_schema_contains_boundary_and_material_constraints():
    schema = generate_json_schema()
    bc_item = schema["properties"]["boundary_conditions"]["items"]["properties"]
    assert "pf_dirichlet" in bc_item["type"]["enum"]
    assert bc_item["component"]["enum"] == [0, 1]
    assert set(bc_item["value"]["type"]) == {"number", "string"}
    assert set(bc_item["t_ramp"]["type"]) == {"number", "string"}
    assert set(bc_item["t_hold"]["type"]) == {"number", "null", "string"}

    material = schema["properties"]["material"]["properties"]
    assert material["pf_model"]["enum"] == ["AT1", "AT2", "PFCZM", "allencahn"]
    assert material["nu"]["minimum"] == -1.0
    assert material["nu"]["maximum"] == 0.5
    assert set(material["E"]["type"]) == {"number", "null", "string"}
    assert material["overrides"]["properties"]["energy_split"]["enum"]
    assert set(material["overrides"]["properties"]["Gc"]["type"]) == {
        "number",
        "string",
    }

    loading = schema["properties"]["loading"]["properties"]
    assert set(loading["t_total"]["type"]) == {"number", "string"}


def test_checked_in_json_schema_matches_generator():
    on_disk = _SCHEMA_PATH.read_text(encoding="utf-8")
    assert json.loads(on_disk) == generate_json_schema()
    assert on_disk == dumps_schema()


def test_schema_cli_prints_and_checks_checked_in_schema(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "phast", "schema"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(proc.stdout)["title"] == "phast YAML problem schema"
    assert proc.stderr == ""

    out = tmp_path / "schema.json"
    subprocess.run(
        [sys.executable, "-m", "phast", "schema", "--output", str(out)],
        check=True,
    )
    assert json.loads(out.read_text(encoding="utf-8")) == generate_json_schema()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "phast",
            "schema",
            "--output",
            str(out),
            "--check",
        ],
        check=True,
    )
