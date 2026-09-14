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

## Swapping architectures

The finite-element workflow does not change when a learned predictor is used.
Geometry construction, meshing, named region identification, boundary
conditions, loading, material properties, solver selection and time integration
are written exactly as for a classical run. One key inside the solver block
selects which operation performs the damage update, and a second names the
adapter:

```yaml
solver:
  solver_type: explicit
  damage_update: learned_proposal
  damage_predictor: examples.learned_damage.architectures.mesh_graph_net:create_predictor
  damage_checkpoint: checkpoints/mesh_graph_net.pt
  damage_fallback: true
```

Changing architecture means changing those two lines:

| Architecture | `damage_predictor` |
|---|---|
| Persistence, interface demonstration | `examples.learned_damage.predictor_plugin:create_predictor` |
| TorchScript, architecture-agnostic | `examples.learned_damage.predictor_plugin:create_predictor` |
| Mesh-graph network | `examples.learned_damage.architectures.mesh_graph_net:create_predictor` |
| Your own | `my_package.my_architecture:create_predictor` |

Each architecture needs a wrapper, because each one has its own graph
construction, feature ordering and normalization. The wrapper is the only place
that knowledge belongs. `examples/learned_damage/architectures/template.py` is a
commented skeleton with the two methods to fill in.

## Finite-element helpers available to any predictor

`DamageStepContext` supplies the quantities that predictors of every
architecture tend to need, so that a wrapper carries model-specific code only:

| Method | Returns |
|---|---|
| `canonical_node_features()` | `[x, y, H, d_prev, u_x, u_y]` per node |
| `graph_edge_index()` | directed, duplicate-free mesh edges |
| `element_areas()` | area of every triangle |
| `assemble_element_to_nodes(v)` | `sum_e int N_i v_e dOmega` |
| `boundary_node_mask()` | topological boundary nodes, interior voids included |
| `edge_lengths(edge_index)` | length of every edge |

## Portable checkpoints

`architectures/mesh_graph_net.py` provides `export_torchscript`, which converts a
state-dict checkpoint into a TorchScript archive. The archive carries its own
architecture, so it evaluates with no model source on the import path and can be
distributed as a single file. Optimizer state is dropped in the process.

```python
from examples.learned_damage.architectures.mesh_graph_net import export_torchscript

export_torchscript("training_checkpoint.pt", "mesh_graph_net.pt")
```

Load checkpoints only from sources you trust. The state-dict loader accepts
tensor-and-metadata archives through PyTorch's restricted `weights_only` mode;
the example does not deserialize arbitrary Python objects.

The `examples.learned_damage...` module paths are available when running from
the PhAST source checkout. They are examples rather than modules installed in
the core `phast` wheel. A deployed predictor should therefore live in its own
installed package or use an application-specific import path.

## Auditing an accepted damage field

`phast.solvers.damage_solver.damage_kkt_metrics` reports the box-constrained
KKT conditions for one damage field: projected stationarity, feasibility of the
irreversibility interval, dual sign conditions on the active sets, and
complementarity. Post-clamping and relaxed iterates can hide exactly these
violations, so the projected norms are the right quantity to report when
comparing a learned update against a classical one.
