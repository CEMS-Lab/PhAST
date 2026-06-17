# Structural DCB Cohesive Validation

DCB-style Mode-I structural cohesive delamination validation with a precrack,
free bulk degrees of freedom, post-peak softening, damage-front tracking, and
energy diagnostics. This is a solver-coupled structural baseline validation, not
ASTM D5528 material-property data reduction.

## What This Validation Covers

- Geometry: two-arm DCB-style specimen with a bonded zero-thickness interface.
- Cohesive law: bilinear traction-separation interface.
- Loading: clamped-end Mode-I opening.
- Solver path: sparse cohesive Newton path through the cohesive operator hook.
- Validation gates: convergence, post-peak softening, monotone cohesive
  dissipation, front advance, bounded diagnostic energy gap, and visual review.

| Initial conditions | Damage evolution |
|---|---|
| <img src="initial_conditions.png" alt="Structural DCB initial conditions" width="360"> | <img src="damage_evolution.gif" alt="Structural DCB damage evolution" width="360"> |

## Files

| File | Purpose |
| --- | --- |
| `README.md` | This guide. |
| `config.yaml` | Resolved validation configuration saved with the retained result. |
| `summary.json` | Validation metrics, capability boundary, references, energy checks, and pass/fail status. |
| `mesh.geo` | Gmsh geometry source. |
| `mesh.msh` | Generated mesh used by the retained result. |
| `structural_dcb_response.csv` | Load/opening response and cohesive-front data. |
| `history.csv` | History output for the validation run. |
| `energy.csv` | Energy ledger for the retained run. |
| `solver_telemetry.csv` | Solver residual and iteration telemetry. |
| `timing_per_step.csv` | Per-step timing evidence. |
| `initial_conditions.png` | Initial setup visual. |
| `cohesive_damage_final.png` | Final cohesive damage field. |
| `structural_dcb_load_displacement.png` | Load-displacement response. |
| `structural_dcb_damage_front.png` | Delamination-front progression. |
| `structural_dcb_energy.png` | Energy diagnostic plot. |
| `structural_dcb_deformed_mesh.png` | Deformed mesh visual. |
| `damage_evolution.gif` | Lightweight damage evolution animation. |
| `visual_manifest.json` | Visual artifact dimensions and review status. |
| `run_manifest.json` | Retained result manifest. |
| `run_metadata.json` | Runtime and platform metadata. |
| `run_lockfile.json` | Resolved configuration and reproducibility metadata. |

artifact.

## Run Through The Reproducibility Contract

From the repository root:

```bash
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml \
  --validation-id structural_dcb_cohesive \
  --output_dir examples/plasticity_interface/results/structural_dcb_cohesive
```

For direct script execution:

```bash
python examples/plasticity_interface/run_structural_dcb_cohesive_benchmark.py \
  --output-dir outputs/plasticity_interface/structural_dcb_cohesive
```

## Manual Or Fluent Setup

The script-contract runner is canonical for this beta validation slice. The
fluent authoring companion is:

```text
examples/plasticity_interface/fluent_setups/structural_dcb_cohesive.py
```

It constructs the same conceptual problem using `phast.Problem`, but public
reproduction should use the reproducibility contract command above.

## Reference Result

| Quantity | Value | Status |
| --- | ---: | --- |
| Nodes / elements | 102 / 128 | PASS |
| Cohesive elements | 12 | PASS |
| Max residual norm | `1.24e-10` | PASS |
| Peak opening force | `2.0585` | PASS |
| Final delamination front | `x = 3.75` | PASS |
| Post-peak softening | true | PASS |
| Energy gap fraction | `0.106` with `0.15` tolerance | PASS |
| Validation | passed | PASS |

The claim remains scoped to structural cohesive baseline validation. It does not
claim ASTM D5528 data reduction or calibrated engineering fracture toughness.

