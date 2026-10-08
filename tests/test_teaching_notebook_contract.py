"""Bounded contracts for the three executable teaching notebooks."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "notebook_setup.ipynb",
    "problem_setup_walkthrough.ipynb",
    "notebook_square_plate_fracture.ipynb",
)


@pytest.fixture(scope="module")
def notebooks():
    return {
        name: json.loads((ROOT / "docs" / "tutorial" / name).read_text(encoding="utf-8"))
        for name in NAMES
    }


def sources(notebook, kind=None):
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if kind is None or cell["cell_type"] == kind
    ]


@pytest.mark.parametrize("name", NAMES)
def test_notebooks_have_clean_compilable_cells(notebooks, name):
    for index, cell in enumerate(notebooks[name]["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"{name}:cell-{index}", "exec")


@pytest.mark.parametrize("name", NAMES)
def test_installation_and_finite_output_contract(notebooks, name):
    all_source = "\n".join(sources(notebooks[name]))
    code = "\n".join(sources(notebooks[name], "code"))
    assert "Python >=3.10" in all_source
    assert "jupyterlab" in all_source
    assert "mutable" in all_source
    assert "nan_to_num" not in code
    assert "v0.16.2-arxiv.2606.23458" not in all_source
    assert "PHAST_NOTEBOOK_OUTPUT_ROOT" in code
    assert "require_finite" in code
    assert '"Reds"' in code


def test_setup_notebooks_share_single_configuration(notebooks):
    first, second = (notebooks[name] for name in NAMES[:2])
    assert sources(first, "code") == sources(second, "code")
    code = "\n".join(sources(first, "code"))
    assert "problem.save(config_path)" in code
    assert "execution_config" not in code
    assert "authored_problem_spec.yaml" not in code
    assert "fail_on_mechanics_nonconvergence=True" in code
    assert "fail_on_stagger_nonconvergence=True" in code
    assert "range(history_len)" not in code
    assert "stored_steps" in code
    assert "its plot is omitted" in code


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad", ([np.nan], [np.inf], [-np.inf], []))
def test_nonfinite_and_empty_fields_are_rejected(notebooks, name, bad):
    definitions = []
    for source in sources(notebooks[name], "code"):
        definitions.extend(
            node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "require_finite"
        )
    assert len(definitions) == 1
    namespace = {"np": np}
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, "<notebook finite-field guard>", "exec"), namespace)
    guard = namespace["require_finite"]
    with pytest.raises(ValueError, match="non-finite"):
        guard(bad, "test field")
    np.testing.assert_array_equal(guard([0.0, 1.0], "valid field"), [0.0, 1.0])


def test_square_plate_preserves_compatibility_and_stress_meaning(notebooks):
    notebook = notebooks[NAMES[2]]
    text = "\n".join(sources(notebook))
    code = "\n".join(sources(notebook, "code"))
    assert "schema-v1 compatibility" in text
    assert "notebook_setup.ipynb" in text
    assert "not the full three-dimensional von Mises stress for plane strain" in text
    assert '"In-plane equivalent stress [MPa]"' in code
    assert '"Von Mises stress [MPa]"' not in code
    assert "damage_frames" in code
    assert 'cmap="magma"' not in code
