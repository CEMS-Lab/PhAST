# Elementwise E(x) and Gc(x) Teaching Example

This example demonstrates the implemented programmatic route for elementwise
Young's modulus `E(x)` and fracture toughness `Gc(x)` in a small two-dimensional
AT2 damage calculation.

## Run

From the repository root:

```bash
python examples/heterogeneous_fields/run.py \
  --config examples/heterogeneous_fields/parameters.yaml \
  --output-dir runs/heterogeneous_fields
```

This is a script-contract example. `parameters.yaml` is consumed by `run.py`; it
must not be passed to `python -m phast run`, whose general YAML schema does not
currently describe arbitrary elementwise material maps.

The default problem is intentionally small and runs on CPU in float64. The
actual elapsed time is written to `run_metadata.json`.

## Inputs

- `mesh`: structured T3 mesh dimensions and resolution;
- `material`: bulk `E`, `nu`, `Gc`, `l0`, energy split, and plane assumption;
- `heterogeneity.soft_inclusion`: circular Young's-modulus contrast;
- `heterogeneity.weak_band`: vertical fracture-toughness contrast;
- `loading.strain_yy`: imposed affine tensile strain used to evaluate the
  mechanics-derived history field;
- `solver`: projected-CG tolerance and iteration limit.

`E_field[e]` and `Gc_field[e]` correspond exactly to row `e` of
`mesh.elements`. `material_fields.csv` records this ordering and must remain the
reference when replacing the analytic fields with segmented data.

## Outputs

- `parameters_resolved.yaml`
- `material_fields.csv`
- `damage.csv`
- `material_fields.png`
- `damage_final.png`
- `summary.json`
- `run_manifest.json`
- `run_metadata.json`
- `run_lockfile.json`
- `visual_manifest.json`

Inspect the result bundle with:

```python
import phast

result = phast.load_result("runs/heterogeneous_fields")
print(result.metadata())
print(result.manifest())
print(result.visuals())
```

## Scientific Boundary

The calculation evaluates damage under an imposed affine displacement field. It
does not solve a fully coupled displacement-damage boundary-value problem and is
not a microstructure-fracture validation case. It does not represent a cohesive
interface, PF-CZM, plasticity, contact, anisotropy, or three-dimensional
fracture. Use it to learn field ordering and the implemented AT2 damage API
before constructing a project-specific model.
