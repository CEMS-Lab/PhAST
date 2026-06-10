# Quickstart -- five minutes from clone to crack

This tutorial gets a phase-field crack-branching simulation running on
your machine in roughly five minutes. It uses the `B7` benchmark
(COMSOL 6.4 dynamic-crack-branching reference) because it ships with
its own `compare.py` and produces a clear Y-shaped branching pattern.

## 1. Install

From the repository root:

```bash
pip install -e .
```

The `pyproject.toml` pulls torch, meshio, gmsh, h5py, matplotlib, and
PyG. For a full platform-aware install (CUDA wheels, MPS, system libs)
use the helper script instead:

```bash
bash install.sh            # auto-detect platform
bash install.sh cuda       # force CUDA
bash install.sh mps        # macOS Apple Silicon
bash install.sh cpu        # CPU only
```

## 2. Run the smoke test

```bash
# macOS / CPU
python -m phast run configs/benchmarks/dynamic/B7_dynamic_crack_branching_comsol.yaml --device cpu

# Linux + GPU
python -m phast run configs/benchmarks/dynamic/B7_dynamic_crack_branching_comsol.yaml --device cuda
```

You should see log lines like:

```
[config] resolved: device=cpu, dtype=float64
[mesh]   loaded: ... nodes, ... elements
[solver] explicit dynamics, dt=...e-09 s, n_steps=...
[step]   100/...   max_d=0.07   E_kin=...
...
[output] writing damage_final.png
```

## 3. Inspect the output

The run writes to `runs/B7_*/` (or whatever you pass via
`--output_dir`). Two artefacts to look at first:

- `damage_final.png` -- a tricontour of the final damage field; you
  should see a single pre-crack splitting into a Y near `t = 33` us.
- `compare.png` -- produced by the per-benchmark comparator, which
  overlays our crack path on the COMSOL reference figure. Run it
  manually after the simulation finishes:

  ```bash
  python examples/dynamic/crack_branching_comsol/compare.py <run_dir>
  ```

If `damage_final.png` shows a single straight crack (no branching) the
run is under-resolved or the regulariser is too large; the
[primer](01_phase_field_primer.md) explains why, and
[setting up your problem](03_setting_up_your_problem.md) shows how to
tune the parameters. For an interactive walk-through of the same
flow see `quickstart.ipynb` in this directory.

## 4. Forward-pass visualisation (`--plots` / `--gif`)

Before committing to a long inversion, sanity-check the forward
physics with intermediate snapshots and a damage-evolution GIF. The
`run` subcommand exposes two opt-in flags:

```bash
python -m phast run \
    configs/benchmarks/dynamic/B7_dynamic_crack_branching_comsol.yaml \
    --device cpu --num_steps 200 --h5 --plots --gif
```

What each flag does (after the run finishes):

- `--plots` writes `damage_t<step>.png` snapshots and the standard
  multi-panel damage / stress / energy figures into
  `<run_dir>/figures/`.
- `--gif` additionally writes `<run_dir>/figures/damage_evolution.mp4`
  by default. Set `output.gif_fields: damage,stress,displacement` in
  YAML, or use `python -m phast postprocess <run_dir>
  --only-gifs --animation-fields damage,stress,displacement`, to render
  stress and displacement animations as well.

Both flags currently require legacy H5 snapshots, so pass `--h5` (or set
`output.h5: true` in the YAML) -- the snapshot interval is controlled
by `output.h5_every`. H5 here is for benchmark visualization compatibility
while the Zarr post-processing bridge is completed. The
flags reuse `BenchmarkPostProcessor` internally; if the inline call
fails (e.g. missing `imageio`) a fallback command is printed:

```bash
python -m phast postprocess <run_dir>
```

Use this same `postprocess` subcommand any time you want to regenerate
figures from an existing run dir without re-simulating.
