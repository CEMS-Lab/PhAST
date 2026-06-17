# J2 Plasticity Validation

Standalone J2/von-Mises material-point validation with linear isotropic
hardening. This retained result validates the return-mapping kernel and
stress-strain response; it is not a coupled phase-field plasticity solve.

## What This Validation Covers

- Constitutive model: small-strain J2/von-Mises plasticity.
- Hardening: linear isotropic hardening.
- Elastic parameters: `E = 210000 MPa`, `nu = 0.30`.
- Yield stress: `250 MPa`.
- Hardening modulus: `5000 MPa`.
- Loading: axial load/unload path over 66 steps.
- Validation gate: post-yield consistency relation
  `sigma_vm = sigma_y0 + H * eps_p_eq`.

| Initial conditions | Field evolution |
|---|---|
| <img src="initial_conditions.png" alt="J2 validation initial conditions" width="360"> | <img src="field_evolution.gif" alt="J2 validation field evolution" width="360"> |

## Files

| File | Purpose |
| --- | --- |
| `README.md` | This guide. |
| `config.yaml` | Resolved validation configuration saved with the retained result. |
| `summary.json` | Validation metrics, capability boundary, config hash, memory, and pass/fail status. |
| `j2_stress_strain.csv` | Stress-strain response and plasticity state history. |
| `j2_stress_strain.png` | Stress-strain validation plot. |
| `results.csv` | Per-step retained result table. |
| `history.csv` | History output for the validation run. |
| `solver_telemetry.csv` | Solver/kernel telemetry. |
| `timing_per_step.csv` | Per-step timing evidence. |
| `initial_conditions.png` | Initial setup visual. |
| `stress_strain.png` | Stress-strain response figure. |
| `equivalent_plastic_strain.png` | Equivalent plastic strain field. |
| `von_mises.png` | Von-Mises stress field. |
| `plastic_work.png` | Plastic-work diagnostic. |
| `mesh_deformed.png` | Deformed mesh visual. |
| `field_evolution.gif` | Lightweight evolution animation. |
| `visual_manifest.json` | Visual artifact dimensions and review status. |
| `run_manifest.json` | Retained result manifest. |
| `run_metadata.json` | Runtime and platform metadata. |
| `run_lockfile.json` | Resolved configuration and reproducibility metadata. |

artifact.

## Run Through The Reproducibility Contract

From the repository root:

```bash
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml \
  --validation-id j2_validation \
  --output_dir examples/plasticity_interface/results/j2_validation
```

For direct script execution:

```bash
python examples/plasticity_interface/run_j2_validation.py \
  --output-dir outputs/plasticity_interface/j2_validation
```

## Manual Or Fluent Setup

The script-contract runner is canonical for this beta validation slice. The
fluent authoring companion is:

```text
examples/plasticity_interface/fluent_setups/j2_validation.py
```

It constructs the same conceptual problem using `phast.Problem`, but public
reproduction should use the reproducibility contract command above.

## Reference Result

| Quantity | Value | Status |
| --- | ---: | --- |
| Plastic steps | 51 / 66 | PASS |
| First yield step | 16 | PASS |
| Max yield residual | `8.53e-14 MPa` | PASS |
| Visual manifest | passed | PASS |
| Validation | passed | PASS |

