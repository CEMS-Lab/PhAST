# Learned damage predictor interface

## Status and claim boundary

| Route | Public status | Accepted state |
|---|---|---|
| `classical` | Supported default | Classical phase-field damage solve |
| `learned_proposal` | Experimental interface | Classical solve initialized by a projected prediction |
| `learned_replacement` | Experimental interface | Prediction only after admissibility and projected-residual audits |
| Bundled trained model | Not provided | Users supply and validate their own checkpoint |

The interface supports research integration. It does not establish the
accuracy, generalization, or speed of any neural architecture.

## Standard computational route

PhAST retains one standard finite-element workflow. Geometry definition, mesh
construction, material assignment, initial conditions, loading, boundary
conditions, mechanical equilibrium or time integration, history-field
evaluation, and result generation are independent of the selected damage
update. Only the implementation used to obtain the next phase-field damage
state is selectable.

```{mermaid}
flowchart LR
    A["Geometry, mesh, material,<br/>loads and boundary conditions"]
    B["Mechanical FEM update"]
    C["Evaluate strain energy<br/>and history field H"]
    D{"Damage-update route"}
    E["Classical phase-field<br/>damage solve"]
    F["Learned damage proposal"]
    G["Projection and exact<br/>damage correction"]
    H["Admissibility and<br/>projected-residual audit"]
    I["Accepted damage state"]
    J["Classical fallback"]
    K["Advance time or<br/>load increment"]

    A --> B
    B --> C
    C --> D
    D -->|"classical"| E
    D -->|"learned_proposal"| F
    F --> G
    G --> E
    D -->|"learned_replacement"| F
    F --> H
    H -->|"accepted"| I
    H -->|"rejected"| J
    J --> E
    E --> I
    I --> K
    K --> B

    classDef input fill:#f5f5f4,stroke:#57534e,color:#1c1917,stroke-width:1.2px;
    classDef fem fill:#e7e5e4,stroke:#44403c,color:#1c1917,stroke-width:1.2px;
    classDef learned fill:#e2e8f0,stroke:#475569,color:#0f172a,stroke-width:1.2px;
    classDef audit fill:#f1f5f9,stroke:#64748b,color:#0f172a,stroke-width:1.2px;
    classDef decision fill:#fafaf9,stroke:#57534e,color:#1c1917,stroke-width:1.4px;

    class A input;
    class B,C,E,I,J,K fem;
    class F learned;
    class G,H audit;
    class D decision;
```

### Meaning of replacement

The optional learned component replaces only the classical solution of the
damage subproblem for an individual update. It does not replace:

- the finite-element mesh or interpolation;
- the mechanical displacement solve;
- constitutive evaluation;
- the tensile/compressive energy decomposition;
- history-field construction;
- material degradation in the mechanical equations;
- loading, boundary conditions, or time integration;
- output, provenance, or route reporting.

The phrase *learned replacement* therefore means replacement of the damage
subproblem solve by a predicted nodal damage state. The classical damage solver
remains available as the default route and as the fallback authority.

### Recommended standard route

For ordinary simulations, use `damage_update: classical`. For initial research
with a trained model, use `damage_update: learned_proposal`. In this mode the
prediction is projected onto the admissible set and used only as an initial
guess; the classical phase-field solver still determines the accepted damage
state.

Use `damage_update: learned_replacement` only for an explicitly identified
experimental study. A prediction is accepted only after the configured
physical and residual audits. Failure of any audit invokes the classical
damage solve when fallback is enabled.

## Configuration

```yaml
solver:
  damage_update: learned_proposal
  damage_predictor: my_package.damage_adapter:create_predictor
  damage_checkpoint: checkpoints/model.pt
  damage_predictor_options:
    representation: damage_increment
  damage_residual_rtol: 1.0e-3
  damage_residual_atol: 1.0e-8
  damage_bound_tolerance: 1.0e-8
  damage_fallback: true
```

The factory is loaded from `module:factory` and called as:

```python
factory(checkpoint=checkpoint, device=device, options=options)
```

It returns an object with:

```python
predictor.name
predictor.predict(context)
```

`predict(context)` returns either a nodal tensor, interpreted as the next
damage field, or:

```python
DamagePrediction(
    damage=prediction,
    representation="damage_increment",
    diagnostics={"checkpoint": "..."},
)
```

## State available to a plug-in

`DamageStepContext` exposes:

- step, time, and load factor;
- node coordinates and element connectivity;
- displacement and velocity;
- element and nodal history fields;
- previous accepted damage;
- material and formulation metadata;
- execution device and dtype.

`canonical_node_features()` provides a small teaching representation.
`graph_edge_index()` provides dependency-free directed mesh edges. Neither is
mandatory. Architecture adapters may construct graph edge features,
node-element incidence operators, temporal windows, DeepONet branch/trunk
inputs, regular-grid projections, reduced coordinates, or their own normalized
features.

This separation is deliberate. PhAST owns finite-element state and acceptance;
the plug-in owns architecture-specific preprocessing, checkpoint
interpretation, and inference.

## Plug-and-play boundary

The solver-facing interface is plug-and-play: installing a predictor through a
factory and selecting it in YAML does not require modification of the PhAST
mechanics or damage-solver source code. A conforming adapter receives
`DamageStepContext` and returns `DamagePrediction`.

An arbitrary checkpoint is not automatically portable. Neural architectures
can differ in feature ordering, normalization, temporal history, graph
construction, finite-element incidence operators, grid projection, output
representation, and checkpoint serialization. These differences must be
declared and implemented by the model adapter.

The reference TorchScript factory demonstrates the contract but is not a
registry of validated models. A future YAML-level model manifest may standardize
known adapters, normalization records, output representations, and checkpoint
metadata. Until such a registry is implemented and tested, the accurate
description is **plug-in compatible**, rather than universally
checkpoint-compatible.

## Acceptance and fallback

For `learned_proposal`, PhAST:

1. checks shape and finite values;
2. projects damage to `[0, 1]`;
3. enforces irreversibility and phase-field Dirichlet values;
4. supplies the result as the classical solver initial guess.

For `learned_replacement`, PhAST additionally checks:

1. unprojected bound and irreversibility violations;
2. phase-field Dirichlet consistency;
3. the projected phase-field residual relative to the previous state;
4. an absolute residual floor.

Rejected replacements use the classical damage solver when
`damage_fallback: true`. Route reports include calls, proposals, accepted
replacements, failures, and fallbacks.

## Portability

The predictor receives the device selected by PhAST, but its factory remains
responsible for verifying that the checkpoint and operators support that
device. CPU execution is the portable baseline. CUDA, MPS, and optional sparse
backends depend on the installed PyTorch build and platform-specific packages.

Automatic route selection is deterministic according to availability; it is
not a guarantee of the fastest route for every mesh or machine. The public CI
workflow is configured to check the core CPU installation and configuration
path on Linux, macOS, and Windows. Optional accelerator and sparse-direct paths
require their own environment-specific evidence.

See [Modular fracture problems and learned damage updates](../tutorial/03_modular_fem_and_learned_damage.md)
for the student workflow and `examples/learned_damage/` for a minimal factory.
