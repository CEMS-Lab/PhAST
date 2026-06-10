from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def _make_displacement_run(tmp_path: Path) -> Path:
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

    with h5py.File(tmp_path / "training_data.h5", "w") as h5:
        mesh = h5.create_group("simulation_data/mesh")
        mesh.create_dataset("node_coordinates", data=coords)
        mesh.create_dataset("element_connectivity", data=elems)
        steps = h5.create_group("simulation_data/steps")
        for step in range(3):
            grp = steps.create_group(f"step_{step:04d}")
            scale = float(step + 1)
            u = np.column_stack((coords[:, 0], coords[:, 1])) * 0.01 * scale
            grp.create_dataset("displacement", data=u)
            grp.create_dataset("damage_nodal", data=np.zeros(len(coords)))
            grp.attrs["time_s"] = step * 1.0e-6

    (tmp_path / "run_metadata.json").write_text(
        json.dumps(
            {
                "problem": "displacement animation smoke",
                "solver": {"solver_type": "explicit", "dt": 1.0e-6},
                "material": {"E": 1.0, "nu": 0.25, "Gc": 1.0, "l0": 0.1},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_make_displacement_animation_writes_gif(tmp_path):
    from phast.postprocess_paper import BenchmarkPostProcessor

    run_dir = _make_displacement_run(tmp_path)
    pp = BenchmarkPostProcessor(str(run_dir))
    try:
        pp.make_displacement_animation(
            max_frames=3,
            animation_format="gif",
            renderer="raster",
            raster_width=64,
        )
    finally:
        pp.close()

    out = run_dir / "figures" / "displacement_evolution.gif"
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_all_dispatches_displacement_animation(tmp_path):
    from phast.postprocess_paper import BenchmarkPostProcessor

    run_dir = _make_displacement_run(tmp_path)
    pp = BenchmarkPostProcessor(str(run_dir))
    try:
        pp.generate_all(
            skip_gif=False,
            fields="gif",
            animation_fields="displacement",
            animation_format="gif",
            animation_renderer="raster",
            raster_width=64,
            max_frames=3,
        )
    finally:
        pp.close()

    out = run_dir / "figures" / "displacement_evolution.gif"
    assert out.exists()
    assert out.stat().st_size > 0
