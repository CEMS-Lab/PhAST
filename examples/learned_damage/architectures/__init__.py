"""Per-architecture learned-damage adapters.

Each module here wraps one neural architecture so that it satisfies the PhAST
damage-predictor protocol. The solver, the finite-element workflow and the YAML
schema are identical for all of them; only the ``damage_predictor`` entry in the
solver block changes. Use :mod:`template` as the starting point for a new
architecture.
"""
