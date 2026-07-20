"""Minimal learned-damage plug-in contract.

The persistence predictor is an interface demonstration, not a useful damage
model. The TorchScript adapter shows one portable way to load a trained model
without coupling PhAST to a particular neural-network architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from phast.learned_damage import DamagePrediction, DamageStepContext


class PersistencePredictor:
    """Return the previous converged damage field."""

    name = "persistence-demonstration"

    def predict(self, context: DamageStepContext) -> DamagePrediction:
        return DamagePrediction(
            damage=context.damage_previous.clone(),
            diagnostics={"purpose": "interface demonstration"},
        )


class TorchScriptDamagePredictor:
    """Adapt a TorchScript model to the PhAST damage-predictor protocol."""

    def __init__(
        self,
        checkpoint: str,
        device: torch.device,
        representation: str = "damage",
    ) -> None:
        self.name = f"torchscript:{Path(checkpoint).name}"
        self.representation = representation
        self.model = torch.jit.load(checkpoint, map_location=device)
        self.model.eval()

    def predict(self, context: DamageStepContext) -> DamagePrediction:
        with torch.no_grad():
            damage = self.model(
                context.canonical_node_features(),
                context.elements,
            )
        return DamagePrediction(
            damage=torch.as_tensor(
                damage,
                device=context.device,
                dtype=context.dtype,
            ).reshape_as(context.damage_previous),
            representation=self.representation,
            diagnostics={"adapter": "torchscript"},
        )


def create_predictor(
    checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
    options: Optional[Mapping[str, Any]] = None,
):
    """Factory used by ``solver.damage_predictor`` configuration."""

    options = dict(options or {})
    if checkpoint:
        return TorchScriptDamagePredictor(
            checkpoint=checkpoint,
            device=device or torch.device("cpu"),
            representation=str(options.get("representation", "damage")),
        )
    return PersistencePredictor()
