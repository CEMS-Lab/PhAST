# Solid Mechanics

Solid mechanics examples are promoted public FEA workflows when they are listed
in `examples/PUBLIC_EXAMPLES_CONTRACT.yaml` and ship the flat result bundle
required by `docs/user_guide/example_contract.md`.

## Public path

Use YAML configurations for reproducible runs:

```bash
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml --validate-only
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml --output_dir runs/linear_plate
python -m phast run examples/solid_mechanics_beta/neohookean_plate/config.yaml --output_dir runs/neohookean_plate
python -m phast run examples/solid_mechanics_beta/j2_bar/config.yaml --output_dir runs/j2_bar
```

Use the fluent `phast.Problem` API to author new solid mechanics setups, then
promote reproducible examples through checked-in YAML configurations once the runner,
outputs, and tests are covered.

## Expected outputs

Promoted solid mechanics examples should include `run_manifest.json`,
`visual_manifest.json`, `initial_conditions.png`, `response.csv`,
`response.png`, `thumbnail.png`, and the field plots listed in the public
example contract. Inspect completed runs with:

```python
import phast

result = phast.load_result("runs/linear_plate")
print(result.metadata())
print(result.history_names())
print(result.visuals())
```

## Boundary

The public J2 bar is a promoted solid-mechanics example, not a claim that every
coupled elastoplastic phase-field/cohesive workflow is production-ready. Check
the [capability matrix](../user_guide/capability_matrix.md) before combining
solid mechanics with beta plasticity/interface features.
