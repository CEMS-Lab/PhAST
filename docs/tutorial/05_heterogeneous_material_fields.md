# Heterogeneous Material Fields

PhAST supports elementwise Young's modulus and fracture-toughness fields through
its programmatic finite-element operators. This tutorial introduces the field
ordering and damage-solver interface using a small CPU example. It does not
claim that arbitrary material maps are available through the general YAML
runner.

## Run The Teaching Example

From the repository root:

```bash
python examples/heterogeneous_fields/run.py \
  --config examples/heterogeneous_fields/parameters.yaml \
  --output-dir runs/heterogeneous_fields
```

Inspect the result bundle:

```python
import phast

result = phast.load_result("runs/heterogeneous_fields")
print(result.metadata())
print(result.manifest())
print(result.visuals())
```

## What The Example Solves

The script creates a structured two-dimensional T3 mesh and defines two arrays:

- `E_field[e]` is the Young's modulus assigned to element `e`;
- `Gc_field[e]` is the fracture toughness assigned to element `e`.

Array index `e` is the row index of `mesh.elements[e]`. Material values are not
indexed by node, pixel, geometric region name, or an independently sorted list.
Any image-to-mesh or segmentation workflow must preserve this element ordering.

The example installs `E_field` in `FEMOperators`, evaluates tensile strain
energy under an imposed affine strain field, and calls
`PhaseFieldDamageSolver.solve(..., Gc_field=Gc_field)` for bounded AT2 damage.
The output includes the exact element ordering, material values, history field,
and final nodal damage.

## Capability Boundary

This is a damage-subproblem teaching calculation. It is not:

- a coupled displacement-damage equilibrium benchmark;
- a dynamic fracture trajectory;
- a zero-thickness cohesive interface or PF-CZM model;
- a calibrated multiphase material law;
- evidence that an arbitrary checkpoint or segmentation is portable.

For a research microstructure study, first establish the specimen geometry,
phase properties, interface assumption, mesh resolution relative to `l0`,
loading, and validation target. Then replace the analytic field construction in
`run.py` with an audited elementwise map and retain the same ordering checks and
provenance outputs.

## Differentiability

The example uses ordinary tensors because it is a forward teaching run. An
elementwise `E` field can remain in the autograd graph when constructed from
parameters with `requires_grad=True`. The spatial `Gc` damage route has a
specialized implicit-differentiation pathway in the staggered solver. History
maxima, bounds, active sets, and solver tolerances remain nonsmooth boundaries;
gradients require case-specific verification.

See the public [`examples/heterogeneous_fields/README.md`](https://github.com/CEMS-Lab/PhAST/blob/main/examples/heterogeneous_fields/README.md)
for parameters, outputs, and adaptation guidance.
