# Learned damage plug-in example

PhAST exposes a small predictor protocol for research on learned damage
updates. The finite-element solver remains responsible for mechanics,
history-field construction, constraints, and acceptance.

The example provides two adapters:

- `PersistencePredictor` returns the previously converged damage field. It is
  only an interface demonstration and is not a trained surrogate.
- `TorchScriptDamagePredictor` loads a user-supplied TorchScript checkpoint.
  The model receives canonical nodal features and element connectivity.

Add the following fields to a runnable PhAST YAML configuration:

```yaml
solver:
  damage_update: learned_proposal
  damage_predictor: examples.learned_damage.predictor_plugin:create_predictor
  damage_checkpoint: path/to/model.pt
  damage_predictor_options:
    representation: damage_increment
```

`learned_proposal` uses the prediction only as an initial guess for the
classical damage solve. The converged finite-element update remains
authoritative.

`learned_replacement` is an experimental option. Before a prediction can
replace the classical update, PhAST checks its shape, finite values, physical
bounds, irreversibility, prescribed phase-field values, and projected damage
residual. A rejected prediction falls back to the classical solve by default.

The canonical nodal feature columns are:

```text
x, y, history field, previous damage, displacement x, displacement y
```

The canonical features are a teaching convenience, not a mandatory model
schema. A plug-in receives the complete `DamageStepContext` and can call
`context.graph_edge_index()` for mesh-graph models, use coordinates as a
DeepONet trunk, retain a temporal state buffer, project fields to a regular
grid, or construct node-element incidence operators. Architecture-specific
normalization and checkpoint schemas remain inside the plug-in.

Use `representation: damage` when the model predicts the next field, or
`representation: damage_increment` when it predicts an increment relative to
the previous accepted damage. Both pass through the same PhAST projection and
audit.

This interface does not prescribe a neural architecture or training
procedure. A predictor intended for scientific use must be trained and
evaluated on data consistent with the mesh representation, material model,
loading protocol, nondimensionalization, and fracture formulation used at
inference time.

Python callers can install a predictor directly:

```python
from examples.learned_damage import create_predictor

predictor = create_predictor(checkpoint="path/to/model.pt", device=solver.device)
solver.set_damage_predictor(predictor, mode="learned_proposal")
```

Start with `learned_proposal`. Treat direct replacement as a separate
experimental study requiring independent finite-element comparisons and
complete reporting of all fallback events.
