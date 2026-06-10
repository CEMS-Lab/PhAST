import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def test_arc_length_controller_checkpoint_roundtrip(tmp_path):
    module_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "quasistatic"
        / "_run_utils.py"
    )
    spec = importlib.util.spec_from_file_location("qs_run_utils", module_path)
    run_utils = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = run_utils
    spec.loader.exec_module(run_utils)
    ArcLengthController = run_utils.ArcLengthController
    load_run_checkpoint = run_utils.load_run_checkpoint
    save_run_checkpoint = run_utils.save_run_checkpoint

    continuation = ArcLengthController(
        enabled=True,
        max_steps=10,
        ds=0.2,
        min_ds=0.01,
        max_ds=0.3,
        min_disp=0.0,
        max_disp=1.0,
        damage_trigger=0.3,
        reaction_drop=0.01,
        post_peak_steps=4,
        direction=-1.0,
        peak_abs_reaction=12.5,
        reversed=True,
        steps_after_reversal=2,
    )
    solver = SimpleNamespace(
        u=torch.zeros(2, 2),
        v=torch.zeros(2, 2),
        a=torch.zeros(2, 2),
        d=torch.zeros(2),
        H_elem=torch.zeros(1),
        H_nodal=torch.zeros(2),
        f_ext=torch.zeros(2, 2),
        _step_count=3,
        _last_stagger_iter=2,
        _last_residual=1.0e-7,
    )
    bcs = SimpleNamespace(load_factor=0.4)
    path = tmp_path / "restart.pt"

    save_run_checkpoint(
        str(path),
        solver=solver,
        bcs=bcs,
        pending_displacements=[0.3],
        history=[{"step": 0}],
        energy_rows=[],
        accepted_step=3,
        last_disp=0.4,
        cutback_count=1,
        consecutive_cutbacks=0,
        continuation=continuation,
    )

    restored = ArcLengthController.from_args(
        SimpleNamespace(
            arc_length=True,
            arc_length_ds=0.1,
            arc_length_min_ds=None,
            arc_length_max_ds=None,
            arc_length_max_disp=1.0,
            arc_length_min_disp=0.0,
            arc_length_damage_trigger=0.3,
            arc_length_reaction_drop=0.01,
            arc_length_post_peak_steps=4,
            arc_length_allow_reversal=True,
            arc_length_steps=10,
            num_steps=10,
        ),
        [0.0, 0.1],
    )
    payload = load_run_checkpoint(str(path), solver=solver, bcs=bcs)
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    restored.load_state_dict(payload["continuation_state"])

    assert restored.direction == -1.0
    assert restored.peak_abs_reaction == 12.5
    assert restored.reversed is True
    assert restored.steps_after_reversal == 2
