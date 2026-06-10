import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_UTILS_PATH = Path(__file__).resolve().parents[1] / "examples" / "quasistatic" / "_run_utils.py"
_SPEC = importlib.util.spec_from_file_location("qs_run_utils", _UTILS_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)
ArcLengthController = _MOD.ArcLengthController


def _args(**overrides):
    base = dict(
        arc_length=True,
        arc_length_steps=4,
        arc_length_ds=0.1,
        arc_length_min_ds=None,
        arc_length_max_ds=None,
        arc_length_min_disp=0.0,
        arc_length_max_disp=0.4,
        arc_length_damage_trigger=0.2,
        arc_length_reaction_drop=0.05,
        arc_length_post_peak_steps=2,
        arc_length_allow_reversal=True,
        num_steps=4,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_arc_length_seeds_single_initial_load():
    c = ArcLengthController.from_args(_args(), [0.1, 0.2, 0.3])
    assert c.seed_pending([0.1, 0.2, 0.3]) == [0.1]


def test_arc_length_reverses_after_reaction_drop_in_damage_zone():
    c = ArcLengthController.from_args(_args(), [0.1, 0.2, 0.3])
    assert c.next_after_accept(
        accepted_step=1, disp=0.1, reaction=10.0, max_d=0.1) == 0.2
    nxt = c.next_after_accept(
        accepted_step=2, disp=0.2, reaction=9.0, max_d=0.3)
    assert c.reversed is True
    assert c.direction < 0.0
    assert nxt == 0.1


def test_arc_length_respects_max_steps_after_incremented_counter():
    c = ArcLengthController.from_args(_args(arc_length_steps=3), [0.1, 0.2])
    assert c.next_after_accept(
        accepted_step=1, disp=0.1, reaction=1.0, max_d=0.0) == 0.2
    assert c.next_after_accept(
        accepted_step=2, disp=0.2, reaction=2.0, max_d=0.0) == pytest.approx(0.3)
    assert c.next_after_accept(
        accepted_step=3, disp=0.3, reaction=3.0, max_d=0.0) is None
