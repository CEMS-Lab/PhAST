"""The fluent Problem refuses a second material for another region."""
from __future__ import annotations

import pytest

import phast


def test_second_material_for_another_region_is_rejected():
    problem = phast.Problem("two materials").material(
        "epoxy", region="matrix", E=3000.0, nu=0.3, Gc=0.3, l0=0.3)
    with pytest.raises(NotImplementedError, match="schema-v2 deck with element regions"):
        problem.material("steel", region="insert", E=2.0e5, nu=0.3, Gc=30.0, l0=0.3)


def test_repeated_material_for_same_region_still_replaces():
    problem = phast.Problem("one material")
    problem.material("epoxy", region="body", E=3000.0)
    problem.material("epoxy", region="body", E=3500.0)
    assert problem._config.material.overrides["E"] == 3500.0
