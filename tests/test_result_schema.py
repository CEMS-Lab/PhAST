"""The machine-readable result schema must agree with its specification."""
from __future__ import annotations

from pathlib import Path

import pytest

from phast.io.result import SCHEMA_NAME, SCHEMA_VERSION, load_schema, schema_text

ROOT = Path(__file__).resolve().parents[1]
SPEC = (ROOT / "docs" / "user_guide" / "result_file_layout.md").read_text(encoding="utf-8")


def test_schema_loads_with_declared_name_and_version():
    schema = load_schema()
    assert schema["schema"] == SCHEMA_NAME == "phast.result"
    assert schema["version"] == SCHEMA_VERSION == "2.0.0"
    assert "phast_schema_version" in schema_text()


def test_unknown_schema_version_is_refused():
    with pytest.raises(FileNotFoundError):
        load_schema("9.9.9")


def test_specification_states_the_schema_version():
    assert f'phast_schema_version = "{SCHEMA_VERSION}"' in SPEC


@pytest.mark.parametrize("section", ["required", "optional"])
def test_every_root_attribute_is_in_the_specification(section):
    for name in load_schema()["root_attributes"][section]:
        assert name in SPEC, f"root attribute {name!r} is not in result_file_layout.md"


def test_enumerated_root_values_are_in_the_specification():
    for name, rule in load_schema()["root_attributes"]["required"].items():
        for value in rule.get("values", []):
            assert str(value) in SPEC, f"{name} value {value!r} is not in result_file_layout.md"


def test_conventions_match_the_specification():
    conventions = load_schema()["conventions"]
    assert conventions["extension"] == ".vtkhdf" and "`.vtkhdf`" in SPEC
    assert conventions["derived_in_view_size_limit_gb"] == 4 and "4 GB" in SPEC
    assert conventions["result_file_is_checkpoint"] is False


def test_cell_types_and_point_counts_follow_the_specification_table():
    cells = load_schema()["cell_types"]
    for name in ("T3", "T6", "Q4", "Q8", "Q9", "H8", "TET4", "TET10"):
        assert name in cells and name in SPEC
    assert cells["T3"]["quadrature_points"] == [1]
    assert cells["TET4"]["quadrature_points"] == [1]
    assert cells["Q4"]["quadrature_points"] == [4]
    assert cells["H8"]["quadrature_points"][0] == 8
    assert {cells[n]["vtk"] for n in ("T3", "Q4", "H8", "T6", "Q9")} == {5, 9, 12, 22, 28}


def test_component_orders_follow_the_specification():
    comp = load_schema()["components"]
    assert ", ".join(comp["tensor_3d"]) in SPEC
    assert ", ".join(comp["tensor_2d"]) in SPEC
    assert comp["strain_shear"] == "engineering"


def test_field_vocabulary_is_in_the_specification():
    fa = load_schema()["field_attributes"]
    for key in ("location", "kind", "derived_method", "measure"):
        for value in fa[key]:
            assert value in SPEC, f"{key} value {value!r} is not in result_file_layout.md"


def test_history_columns_are_in_the_specification():
    schema = load_schema()
    for name in schema["history_columns"]["required"] + schema["history_columns"]["energy_terms"]:
        assert name in SPEC, f"history column {name!r} is not in result_file_layout.md"
    tables = load_schema("2.1.0")["history_tables"]
    for name in tables["crack_tips"]["required"] + tables["substeps"]["required"]:
        assert name in SPEC, f"table column {name!r} is not in result_file_layout.md"


def test_setup_profile_and_categorical_encoding_follow_the_specification():
    schema = load_schema()
    assert schema["profile_rules"]["setup"]["nsteps"] == 0 and "`NSteps = 0`" in SPEC
    enc = schema["categorical_encoding"]
    assert enc["dtype"] == "int8" and enc["attribute"] == "codes" and "attribute `codes`" in SPEC


def test_precision_rule():
    dtypes = load_schema()["dtypes"]
    assert set(dtypes["float64_required"]) >= {"state", "history"}
    assert set(dtypes["float32_allowed"]) == {"derived"}
