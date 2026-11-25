"""The multi-material runner keeps the mesh it used next to its outputs."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import phast
from phast.workflow.multimaterial_fracture import _structured_t3, _write_mesh_copy


def test_generated_mesh_is_written_with_named_material_regions(tmp_path):
    nodes, elements = _structured_t3(2.0, 1.0, 4, 2, (0.0, 0.0))
    ids = (nodes[elements].mean(axis=1)[:, 0] > 1.0).astype(np.int64)
    written = _write_mesh_copy(tmp_path, SimpleNamespace(mesh=None), nodes, elements,
                               ids, ["matrix_region", "inclusion_region"])
    assert written == ["mesh.msh"]
    groups = phast.inspect_mesh(str(tmp_path / "mesh.msh"))["named_groups"]
    assert groups["matrix_region"]["cell_counts"] == {"triangle": int((ids == 0).sum())}
    assert groups["inclusion_region"]["cell_counts"] == {"triangle": int((ids == 1).sum())}


def test_external_mesh_and_its_geo_are_copied(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "part.msh").write_text("mesh bytes", encoding="utf-8")
    (source / "part.geo").write_text("geo text", encoding="utf-8")
    out = tmp_path / "run"
    out.mkdir()
    spec = SimpleNamespace(mesh=SimpleNamespace(path=str(source / "part.msh")))
    written = _write_mesh_copy(out, spec, np.zeros((3, 2)), np.zeros((1, 3), int),
                               np.zeros(1, int), ["body"])
    assert written == ["mesh.msh", "mesh.geo"]
    assert (out / "mesh.msh").read_text(encoding="utf-8") == "mesh bytes"
    assert (out / "mesh.geo").read_text(encoding="utf-8") == "geo text"
