# PF-CZM Uniaxial Strength Validation

Forward Wu PF-CZM damage-law strength calibration and length-scale validation
for a uniaxial bar. This retained result validates the damage-law threshold and
bounded nonlinear solve; it is not a full structural crack-growth, mixed-mode
delamination, or ductile PF-plasticity-cohesive benchmark.

## What This Validation Covers

- Model: Wu PF-CZM cohesive phase-field damage law.
- Geometry: small uniaxial bar mesh.
- Target tensile strength: `sigma_ts = 3.0`.
- Length scales: `l0 = 0.08`, `0.12`, `0.18`.
- Validation gates: peak degraded stress matches target strength, damage onset
  occurs near `sigma_ts`, nonlinear residuals are finite/bounded, and visual
  artifacts pass review checks.

| Initial conditions | Damage evolution |
|---|---|
| <img src="initial_conditions.png" alt="PF-CZM initial conditions" width="360"> | <img src="damage_evolution.gif" alt="PF-CZM damage evolution" width="360"> |


## Run Through The Reproducibility Contract

From the repository root:

```bash
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml \
  --validation-id pfczm_uniaxial_strength \
  --output_dir examples/plasticity_interface_beta/results/pfczm_uniaxial_strength
```

For direct script execution:

```bash
python examples/plasticity_interface_beta/run_pfczm_uniaxial_strength_validation.py \
  --output-dir outputs/plasticity_interface/pfczm_uniaxial_strength
```

## Manual Or Fluent Setup

The script-contract runner is canonical for this beta validation slice. The
fluent authoring companion is:

```text
examples/plasticity_interface_beta/fluent_setups/pfczm_uniaxial_strength.py
```

It constructs the same conceptual problem using `phast.Problem`, but public
reproduction should use the reproducibility contract command above.

## Reference Result

| Quantity | Value | Status |
| --- | ---: | --- |
| Target tensile strength | `3.0` | PASS |
| Max peak strength relative error | `1.20e-4` | PASS |
| Damage onset ratio | `1.016 * sigma_ts` | PASS |
| All solvers converged | true | PASS |
| Max residual norm | `1.12e-7` | PASS |
| Visual manifest | passed | PASS |
| Validation | passed | PASS |

