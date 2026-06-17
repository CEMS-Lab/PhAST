# Dynamic Fracture Examples

This folder holds the curated public dynamic fracture examples. Keep each
example flat, YAML-first, and free of raw cluster dumps or vendor binaries.

## Public Candidate Dynamic Fracture Examples

These folders are public dynamic-fracture examples. They are the primary
Paper-1 example family unless a row states that it is an external comparison or
supporting extension.

| Example | Status | Role | Command |
|---|---|---|---|
| `B2_kalthoff_winkler/` | Paper-1 public fracture example | Kalthoff-Winkler dynamic impact evidence. | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml` |
| `B3_dynamic_sent/` | Paper-1 baseline verification | Lightweight dynamic SENT baseline verification example. | `python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml` |
| `B5_pmma_branching/` | Paper-1 public fracture example | Selected PMMA branching parametric-study result. | `python -m phast run examples/dynamic/B5_pmma_branching/config.yaml` |
| `B6_perforated_30holes/` | Public fracture extension | Selected perforated PMMA plate; public B6 name replaces old source B4 naming. | `python -m phast run examples/dynamic/B6_perforated_30holes/config.yaml` |
| `B6_perforated_10holes/` | Public fracture extension | Curated perforated PMMA variant with ten holes. | `python -m phast run examples/dynamic/B6_perforated_10holes/config.yaml` |
| `B6_perforated_1hole_near/` | Public fracture extension | Curated single-hole PMMA variant with the hole near the crack path. | `python -m phast run examples/dynamic/B6_perforated_1hole_near/config.yaml` |
| `B6_perforated_1hole_far/` | Public fracture extension | Curated single-hole PMMA variant with the hole farther from the crack path. | `python -m phast run examples/dynamic/B6_perforated_1hole_far/config.yaml` |
| `B7_dynamic_crack_branching_comsol/` | External comparison example | Accepted dynamic branching comparison package from job 47961. | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml` |

| Public example | Source status | Public release action |
|---|---|---|
| `B2_kalthoff_winkler` | Mesh-1 curated outputs exist; local H5 trajectory is not distributed. | Publish flat curated CSV/PNG/metadata outputs, not `training_data.h5`. |
| `B3_dynamic_sent` | Curated outputs exist; config validates with an `h/l0 = 1` warning. | Publish as a runnable dynamic baseline verification example unless finer evidence is promoted. |
| `B5_pmma_branching` | Representative PMMA parametric-study result retained as a flat public folder. | Publish the selected deck and curated visuals only. Keep other parametric studies outside the public repository. |
| `B6_perforated_*` | Perforated-plate variants retained as flat public folders. | Publish only the B6 public folders; do not expose old naming or raw run dumps. |
| `B7_dynamic_crack_branching_comsol` | Curated dynamic branching comparison package. | Publish PNG/CSV/report/metadata only. Link to vendor documentation instead of distributing proprietary binaries or vendor PDFs. |

## Folder rule

Each public example folder should be flat and predictable:

```text
examples/dynamic/<example_name>/
  README.md
  config.yaml
  mesh.geo
  mesh.msh
  run_manifest.json
  visual_manifest.json
  history.csv
  energy.csv
  crack_tip.csv
  damage_final.png
  thumbnail.png
  damage_evolution.gif
  damage_evolution.mp4
```

Do not leave curated public files under `figures/`, `outputs/`,
`reference_runs/`, raw job dump folders, or dated run folders. Trajectory
stores such as `training_data.zarr/` are generated on demand or distributed
through a separate artifact release, not committed to the public repository.

## Timing material

Timing comparisons should not be mixed into the flat public examples gallery.
If timing material is exposed publicly, promote it as a named benchmark artifact
with its own README, exact commands, machine/backend notes, and regenerated
plots.

## Excluded Material

- Raw run dumps used only for recovery and promotion.
- Proprietary binary model files and vendor PDFs.
- Historical PMMA parametric studies that are not the selected public B5 representative.
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
