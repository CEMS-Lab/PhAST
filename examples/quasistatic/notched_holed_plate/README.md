# Quasi-static Notched Holed Plate

Validated notched-holed plate benchmark based on the COMSOL 6.4 Geomechanics
Application Library example "Brittle Fracture of a Holed Plate" and the
Ambati, Gerasimov, and De Lorenzis phase-field fracture setup.

This folder is intentionally self-contained: the YAML configuration, mesh,
reference curve, validation report, and promoted visual outputs are kept beside
each other.

## Run

YAML-first journey: validate the checked-in deck, run
`python -m phast run examples/quasistatic/notched_holed_plate/config.yaml`, and
inspect the standard result directory. YAML is canonical for this example until
the fluent path is promoted because the public deck uses rigid connector
boundary conditions that are validated through the YAML runner.

```bash
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml \
  --output_dir examples/quasistatic/notched_holed_plate/run_local
```

## Compare

```bash
python -u examples/quasistatic/notched_holed_plate/compare.py \
  --run-dir examples/quasistatic/notched_holed_plate/run_local
```

## Promoted Result

The current promoted package is the strict-parity matrix run from Slurm job
`33819`, task `34` (`notched_holed_at2_h0.30_l0.25`).

| Final damage | Crack evolution |
|---|---|
| ![Notched-holed plate final damage](damage_final.png) | ![Notched-holed plate damage evolution](damage_evolution.gif) |

| Quantity | Reference | Promoted result | Status |
| --- | ---: | ---: | --- |
| First peak load | 0.63 kN | 4.68% error | PASS |
| First peak displacement | 0.165 mm per pin | 9.09% error | PASS |
| Second peak load | 0.15 kN | 10.51% error | PASS |

Required public artifacts are present:

- `config.yaml`
- `mesh.geo`
- `mesh.msh`
- `run_metadata.json`
- `visual_manifest.json`
- `initial_conditions.png`
- `results.csv`
- `history.csv`
- `energy.csv`
- `timing_per_step.csv`
- `solver_telemetry.csv`
- `load_displacement.png`
- `damage_final.png`
- `damage_evolution.mp4`
- `damage_evolution.gif`
- `damage_multipanel.png`
- `compare.png`
- `compare_report.txt`
- `thumbnail.png`

The crack-path comparison is currently qualitative against the COMSOL reference
morphology. The second peak is a post-peak diagnostic because it occurs after
crack reorientation toward the hole, where staggered tolerances,
crack-width-to-mesh ratio, and monolithic/staggered solver details affect the
response.

Large trajectory stores are generated on demand for local post-processing and
are not part of the lightweight public example bundle.
