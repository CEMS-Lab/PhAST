"""Reference learned-damage plug-ins for PhAST."""

from .predictor_plugin import (
    PersistencePredictor,
    TorchScriptDamagePredictor,
    create_predictor,
)

__all__ = [
    "PersistencePredictor",
    "TorchScriptDamagePredictor",
    "create_predictor",
]
