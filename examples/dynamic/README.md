# Dynamic Fracture Examples

This directory contains two-dimensional explicit-dynamics phase-field fracture
examples. Each case has an example-local `config.yaml` that can be validated or
executed from the repository root.

## Example Index

| Example | Role | Command |
|---|---|---|
| `B2_kalthoff_winkler/` | Kalthoff-Winkler impact benchmark with documented reference artifacts. | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml` |
| `B3_dynamic_sent/` | Compact dynamic single-edge-notched-tension verification case. | `python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml` |
| `B5_pmma_branching/` | Selected PMMA crack-branching case. | `python -m phast run examples/dynamic/B5_pmma_branching/config.yaml` |
| `B6_perforated_10holes/` | Perforated PMMA plate with ten holes. | `python -m phast run examples/dynamic/B6_perforated_10holes/config.yaml` |
| `B6_perforated_30holes/` | Perforated PMMA plate with thirty holes. | `python -m phast run examples/dynamic/B6_perforated_30holes/config.yaml` |
| `B6_perforated_1hole_near/` | Single-hole variant with the hole near the expected crack path. | `python -m phast run examples/dynamic/B6_perforated_1hole_near/config.yaml` |
| `B6_perforated_1hole_far/` | Single-hole variant with the hole farther from the expected crack path. | `python -m phast run examples/dynamic/B6_perforated_1hole_far/config.yaml` |
| `B7_dynamic_crack_branching_comsol/` | Dynamic crack-branching comparison case. | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml` |

## Recommended Workflow

Validate and inspect a case before allocating a full simulation:

```bash
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only
python -m phast explain-config examples/dynamic/B2_kalthoff_winkler/config.yaml
```

Run into a separate result directory:

```bash
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml \
  --output_dir runs/B2_kalthoff_winkler
```

Dynamic fracture calculations can be computationally demanding. Review the
mesh-to-length-scale ratio, stable time step, damage-update cadence, device, and
trajectory settings before a complete rerun.

## Evidence Boundaries

- The retained PNG, CSV, JSON, and animation files provide lightweight
  reference material for the specific checked-in configuration.
- A schema-valid configuration is not, by itself, evidence of mesh convergence
  or agreement with a literature reference.
- `B3_dynamic_sent` reports an `h/l0 = 1` resolution warning and should be
  treated as a compact verification case rather than quantitative convergence
  evidence.
- Vendor model files, proprietary documentation, raw cluster directories, and
  large trajectory stores are not distributed in these example folders.

## Typical Folder Contents

An example may contain:

```text
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
damage_evolution.gif
```

The exact set depends on the case and output settings. Generate complete local
results with `--output_dir runs/<case>` rather than writing into the example
directory.
