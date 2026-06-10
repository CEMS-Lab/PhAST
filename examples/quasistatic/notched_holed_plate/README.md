# Quasistatic Notched Holed Plate

Canonical artifact folder for the COMSOL notched-holed-plate benchmark.

Canonical configs:

- `configs/benchmarks/quasistatic/QS_notched_holed_plate.yaml` (historical isotropic split baseline)
- `configs/benchmarks/quasistatic/QS_notched_holed_plate_comsol_strict.yaml` (COMSOL-parity split + residual settings)

Run locally:

```bash
python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate.yaml \
  --output_dir examples/quasistatic/notched_holed_plate/run_local
```

Run the COMSOL-strict variant:

```bash
python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate_comsol_strict.yaml \
  --output_dir examples/quasistatic/notched_holed_plate/run_local_strict
```

Compare outputs:

```bash
python -u examples/quasistatic/notched_holed_plate/compare.py \
  --run-dir examples/quasistatic/notched_holed_plate/run_local
```

Optional strict comparison:

```bash
python -u examples/quasistatic/notched_holed_plate/compare.py \
  --run-dir examples/quasistatic/notched_holed_plate/run_local_strict
```
