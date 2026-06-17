# Benchmark catalogue

YAML-driven benchmark suite covering quasi-static and explicit-dynamic
phase-field fracture problems.


New benchmarks should be added as YAML configs under `configs/` and
invoked via `python -m phast run configs/<name>.yaml`. The
schema is documented in `configs/REFERENCE.yaml` (every option with
defaults, allowed values, and units). `--validate-only` parses and
schema-checks without running.

Public examples list their canonical commands in `examples/README.md` and the
example gallery. Keep diagnostic variants and result-local `config.yaml`
snapshots out of this catalogue until they are promoted through the public
example contract.

### Quasi-static benchmarks

| Config | Material | Reference | Acceptance script | Status |
|---|---|---|---|---|
| `QS_lshaped_concrete.yaml` | concrete (Ambati 2015) | Winkler 2001 (path) + Ambati 2015 Fig 19 (load-displacement, ~16 kN peak) | deferred comparison script | reference CSV is retained outside the public repository; do not mark quantitative L2 validated until the rerun evidence is promoted into the public example contract. |
| `QS_notched_holed_plate.yaml` | cement_mortar_ambati (E=6 GPa, Gc=2.28 N/mm) | COMSOL 6.4 Application Library "Brittle Fracture of a Holed Plate" (Ambati 2015 PF formulation; first peak 0.63 kN at 0.33 mm) | `examples/quasistatic/notched_holed_plate/compare.py` | first-peak load 1% off, displacement 52% off; root-caused to simplified rigid-pin BC, unblocked by full Lagrange MPC (#154/#164) |

### Quasi-static benchmarks (deferred)

These predate the YAML-first migration and remain outside the public release
until they have the same flat YAML, manifest, and visualization contract as the
promoted examples:

- `examples/quasistatic/miehe_tension/run.py`
- Miehe shear
- three-point bending

### Dynamic benchmarks

| Config | Material | Reference | Notes |
|---|---|---|---|
| `B2_kalthoff_winkler.yaml` | maraging_steel_kw | Kalthoff 2000 / Borden 2012 Fig 12 | half-plate + symmetry; spectral split |
| `B3_dynamic_sent.yaml` | glass_borden | Borden 2012 §4.1 SENT | spectral split, AT2; public folder contains the curated runnable configuration file and retained visuals |
| `B5_pmma_branching.yaml` | pmma_bleyer (E=3.09 GPa, Gc=0.3 J/m²) | Bleyer 2017 PMMA branching | AT1 + Amor/volumetric-deviatoric split; two-step prestrain + dynamic release |
| `B6_perforated_*.yaml` | pmma_bleyer | Bleyer 2017 perforated plate | dynamic, AT1 + Amor; hole layouts: 1hole_near, 1hole_far, 10holes, 30holes |
| `B7_dynamic_crack_branching_comsol.yaml` | glass_borden override (PMMA equivalent) | COMSOL 6.4 Application Library "Phase-Field Modeling of Dynamic Crack Branching" | AT1 + Amor/vol-dev; traction-controlled (third independent reference for the Y-branching benchmark) |

### Per-benchmark `compare.py`

Each benchmark dir under `examples/{dynamic,quasistatic}/<name>/` ships a
`compare.py` that is invoked manually after a run completes. It loads
the most recent run output + the reference data and writes
`compare_report.txt` + `compare.png` into the same run directory.
`compare.py` is the sole writer of `compare.png`; the `run` subcommand
does not produce it. The acceptance metric varies per benchmark
(peak-load + displacement match; L2 norm on load-displacement envelope;
branching onset time; final crack-path morphology).

### Reaction-force CSV writer

For load-displacement benchmarks, set in the YAML's `output` block:

```yaml
output:
  reaction_node_set: <named_node_set>      # e.g. upper_pin
  reaction_component: 1                    # 0=x, 1=y
