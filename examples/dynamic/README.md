# Dynamic Fracture Examples

This folder holds the curated public dynamic fracture examples. Keep each
example flat, YAML-first, and free of raw cluster dumps or vendor binaries.

## Public release candidates

The clean public-candidate folders are:

| Example | Role | Command |
|---|---|---|
| `B2_kalthoff_winkler/` | Kalthoff-Winkler dynamic impact evidence. | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml` |
| `B3_dynamic_sent/` | Lightweight dynamic SENT smoke/example. | `python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml` |
| `B5_pmma_branching/` | Selected PMMA branching sweep. | `python -m phast run examples/dynamic/B5_pmma_branching/config.yaml` |
| `B6_perforated_30holes/` | Selected perforated PMMA plate; public B6 name replaces old source B4 naming. | `python -m phast run examples/dynamic/B6_perforated_30holes/config.yaml` |
| `B6_perforated_10holes/` | Curated perforated PMMA variant with ten holes. | `python -m phast run examples/dynamic/B6_perforated_10holes/config.yaml` |
| `B6_perforated_1hole_near/` | Curated single-hole PMMA variant with the hole near the crack path. | `python -m phast run examples/dynamic/B6_perforated_1hole_near/config.yaml` |
| `B6_perforated_1hole_far/` | Curated single-hole PMMA variant with the hole farther from the crack path. | `python -m phast run examples/dynamic/B6_perforated_1hole_far/config.yaml` |
| `B7_dynamic_crack_branching_comsol/` | Accepted dynamic branching comparison package from job 47961. | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml` |

| Public example | Source status | Public release action |
|---|---|---|
| `B2_kalthoff_winkler` | Mesh-1 curated outputs exist; local H5 is private and has recovery caveats. | Publish flat curated CSV/PNG/metadata outputs, not `training_data.h5`. |
| `B3_dynamic_sent` | Curated outputs exist; config validates with an `h/l0 = 1` warning. | Publish as a runnable dynamic smoke/example unless finer evidence is promoted. |
| `B5_pmma_branching` | Representative PMMA sweep retained as a flat public folder. | Publish the selected deck and curated visuals only. Keep other sweeps private. |
| `B6_perforated_*` | Perforated-plate variants retained as flat public folders. | Publish only the B6 public folders; do not expose old naming or raw run dumps. |
| `B7_dynamic_crack_branching_comsol` | Curated dynamic branching comparison package. | Publish PNG/CSV/report/metadata only. Keep proprietary binaries and vendor PDFs private. |

## Folder rule

Each public example folder should be flat and predictable:

```text
examples/dynamic/<example_name>/
  README.md
  config.yaml
  mesh.geo
  mesh.msh
  run_metadata.json
  history.csv
  energy.csv
  crack_tip.csv
  training_data.zarr/
  damage_final.png
  thumbnail.png
  damage_evolution.gif
  damage_evolution.mp4
```

Do not leave curated public files under `figures/`, `outputs/`,
`reference_runs/`, raw job dump folders, or dated run folders. Heavy trajectory
stores (`training_data.zarr/`) stay private unless a separate artifact release
is created.

## Timing material

Timing comparisons should not be mixed into the flat public examples gallery.
If timing material is exposed publicly, promote it as a named benchmark artifact
with its own README, exact commands, machine/backend notes, and regenerated
plots.

## Private-only material

- Raw run dumps used only for recovery and promotion.
- Proprietary binary model files and vendor PDFs.
- Historical PMMA sweeps that are not the selected public B5 representative.
- Any old B4 naming.

## Validation

The canonical dynamic configs currently pass schema validation with:

```bash
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only
python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml --validate-only
python -m phast run examples/dynamic/B5_pmma_branching/config.yaml --validate-only
python -m phast run examples/dynamic/B6_perforated_30holes/config.yaml --validate-only
python -m phast run examples/dynamic/B6_perforated_10holes/config.yaml --validate-only
python -m phast run examples/dynamic/B6_perforated_1hole_near/config.yaml --validate-only
python -m phast run examples/dynamic/B6_perforated_1hole_far/config.yaml --validate-only
python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml --validate-only
```

`B3_dynamic_sent` emits a mesh-resolution warning and should not be used as
quantitative convergence evidence without a finer promoted run.
