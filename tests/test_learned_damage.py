from __future__ import annotations

import pytest
import torch

from phast.learned_damage import (
    DamagePrediction,
    DamagePredictionRejected,
    DamageStepContext,
    DamageUpdateController,
)


class ConstantPredictor:
    name = "constant"

    def __init__(self, value):
        self.value = value

    def predict(self, context):
        return DamagePrediction(
            damage=torch.full_like(context.damage_previous, self.value)
        )


class IncrementPredictor:
    name = "increment"

    def predict(self, context):
        return DamagePrediction(
            damage=torch.full_like(context.damage_previous, 0.2),
            representation="damage_increment",
        )


class ResidualSolver:
    def __init__(self, target):
        self.target = target

    def compute_residual(self, history, damage):
        del history
        return damage - self.target


def make_context(previous=0.2):
    dtype = torch.float64
    previous_damage = torch.full((3,), previous, dtype=dtype)
    return DamageStepContext(
        step=2,
        time=0.1,
        load_factor=0.5,
        nodes=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype),
        elements=torch.tensor([[0, 1, 2]], dtype=torch.long),
        displacement=torch.zeros((3, 2), dtype=dtype),
        velocity=torch.zeros((3, 2), dtype=dtype),
        history_element=torch.zeros((1,), dtype=dtype),
        history_nodal=torch.zeros((3,), dtype=dtype),
        damage_previous=previous_damage,
        material={"pf_model": "AT2", "energy_split": "spectral"},
        phase_field_model="AT2",
        energy_split="spectral",
        device=torch.device("cpu"),
        dtype=dtype,
    )


def make_controller(mode, value, target, fallback=True):
    controller = DamageUpdateController(
        predictor=ConstantPredictor(value),
        mode=mode,
        fallback=fallback,
        residual_rtol=1.0e-6,
        residual_atol=1.0e-12,
        bound_tolerance=1.0e-8,
    )
    solver = ResidualSolver(torch.full((3,), target, dtype=torch.float64))
    return controller, solver


def test_proposal_is_projected_and_returned_as_initial_guess():
    context = make_context(previous=0.2)
    controller, damage_solver = make_controller(
        mode="learned_proposal", value=0.1, target=0.4
    )
    decision = controller.decide(context, damage_solver=damage_solver)

    assert decision.route == "learned_proposal_exact_correction"
    assert not decision.accepted_replacement
    assert torch.equal(decision.candidate, context.damage_previous)


def test_replacement_accepts_audited_prediction():
    context = make_context(previous=0.2)
    controller, damage_solver = make_controller(
        mode="learned_replacement", value=0.4, target=0.4
    )
    decision = controller.decide(context, damage_solver=damage_solver)

    assert decision.route == "learned_replacement"
    assert decision.accepted_replacement
    assert torch.allclose(
        decision.candidate,
        torch.full((3,), 0.4, dtype=torch.float64),
    )


def test_replacement_accepts_damage_increment_representation():
    context = make_context(previous=0.2)
    controller = DamageUpdateController(
        predictor=IncrementPredictor(),
        mode="learned_replacement",
        fallback=True,
        residual_rtol=1.0e-6,
        residual_atol=1.0e-12,
        bound_tolerance=1.0e-8,
    )
    damage_solver = ResidualSolver(
        torch.full((3,), 0.4, dtype=torch.float64)
    )

    decision = controller.decide(context, damage_solver=damage_solver)
    assert decision.accepted_replacement
    assert torch.allclose(
        decision.candidate,
        torch.full((3,), 0.4, dtype=torch.float64),
    )


def test_replacement_rejects_invalid_prediction_and_requests_fallback():
    context = make_context(previous=0.2)
    controller, damage_solver = make_controller(
        mode="learned_replacement", value=float("nan"), target=0.4
    )
    decision = controller.decide(context, damage_solver=damage_solver)

    assert decision.route == "classical_fallback"
    assert not decision.accepted_replacement
    assert decision.candidate is None
    assert "NaN" in decision.reason


def test_replacement_can_raise_instead_of_falling_back():
    context = make_context(previous=0.2)
    controller, damage_solver = make_controller(
        mode="learned_replacement",
        value=float("nan"),
        target=0.4,
        fallback=False,
    )

    with pytest.raises(DamagePredictionRejected):
        controller.decide(context, damage_solver=damage_solver)
