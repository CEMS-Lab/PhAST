"""Small end-to-end checks of the 3D voxel API: a coarse pore cell loaded for three increments, and the legacy import paths."""
from pathlib import Path

import numpy as np
import pytest

from phast import solid3d


def _problem(voxel_um=2.0):
    cell = solid3d.PeriodicCell(size_um=(24, 24, 24), voxel_um=voxel_um).add_sphere_pore(diameter_fraction=0.4)
    matrix = solid3d.Material(E=345e3, nu=0.2, strength=600.0, Gc=25e-3)
    return solid3d.Problem(cell, {"matrix": matrix}, solid3d.MacroStrain("xx", to=1e-4, increments=3), solid3d.PFCZM(length_scale_um=3.0))


def test_pore_cell_runs_and_writes_a_result_file(tmp_path):
    import phast

    summary = _problem().run(tmp_path / "run", threads=2, store_every=1, quiet=True)
    assert summary.status == "complete" and summary.path.name == "result.vtkhdf"
    result = phast.load_result(str(summary.path))
    assert result.steps() == [0, 1, 2, 3]
    damage = result.field("damage", 3)
    assert damage.shape == (13, 13, 13) and np.isfinite(damage).all()
    rows = result.history("history")
    stress = [row["macro_stress_xx"] for row in rows]
    assert stress[0] == 0.0 and all(b > a for a, b in zip(stress, stress[1:]))      # elastic loading: stress rises with strain


def test_output_folder_is_never_overwritten(tmp_path):
    problem = _problem()
    problem.run(tmp_path / "run", threads=2, quiet=True)
    with pytest.raises(FileExistsError):
        problem.run(tmp_path / "run", threads=2, quiet=True)


def test_cell_size_must_be_a_multiple_of_the_voxel():
    with pytest.raises(ValueError):
        solid3d.PeriodicCell(size_um=(24, 24, 24), voxel_um=5.0)


def test_legacy_import_paths_return_the_current_modules():
    import importlib

    import phast.core.mesh
    import phast.io.config.run_config
    import phast.solvers.damage_solver

    assert importlib.import_module("phast.mesh") is phast.core.mesh
    assert importlib.import_module("phast.damage_solver") is phast.solvers.damage_solver
    assert importlib.import_module("phast.config.run_config") is phast.io.config.run_config
    assert importlib.import_module("phast.utils.visualization") is importlib.import_module("phast.viz.visualization")
