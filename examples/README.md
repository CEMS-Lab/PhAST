# PhAST Examples

Phase-field fracture benchmarks organized by solver type.

## Dynamic (Explicit)

| Example | Problem | Material | Status |
|---------|---------|----------|--------|
| YAML `configs/benchmarks/dynamic/B1_branching_glass.yaml` | Crack branching (Borden 4.2) | Glass | Dynamic YAML benchmark |
| YAML `configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml` | Kalthoff-Winkler impact (Borden 4.3) | Steel | Dynamic YAML benchmark |
| YAML `configs/benchmarks/dynamic/B3_dynamic_sent.yaml` | Straight crack propagation | Glass | Dynamic YAML benchmark |
| [dynamic/branching_pmma](dynamic/branching_pmma/) | PMMA branching (Bleyer) | PMMA | AT1 projected CG + plane stress |
| [dynamic/perforated_plate](dynamic/perforated_plate/) | Perforated plate | PMMA | Bleyer-style dynamic benchmark |
| [dynamic/crack_branching_comsol](dynamic/crack_branching_comsol/) | COMSOL dynamic branching | PMMA/glass preset | Vendor-reference benchmark |

## Quasi-Static (Implicit)

| Example | Problem | Material | Status |
|---------|---------|----------|--------|
| [quasistatic/miehe_tension](quasistatic/miehe_tension/) | SENT tension | Steel | Implemented; rerun with QS safe defaults |
| [quasistatic/miehe_shear](quasistatic/miehe_shear/) | SENT shear | Steel | Implemented; rerun with QS safe defaults |
| [quasistatic/three_point_bending](quasistatic/three_point_bending/) | Three-point bending | Concrete | Implemented; rerun with QS safe defaults |
| [quasistatic/notched_holed_plate](quasistatic/notched_holed_plate/) | COMSOL notched holed plate | Concrete | Implemented; direct YAML + manifest audit trail |
| [quasistatic/l_shaped_panel](quasistatic/l_shaped_panel/) | L-shaped panel | Glass/Concrete | Implemented; rerun with QS safe defaults |

## Customer Boundary Examples

| Example | Problem | Status |
|---------|---------|--------|
| [plasticity_interface](plasticity_interface/) | Sparse quasi-static J2 plasticity, guarded ductile AT2 damage validation, ductile sensitivity study, cohesive displacement-jump/mixed-mode/contact/delamination-patch/structural-DCB benchmarks, and two solid-interface fracture benchmarks: weak-interface deflection and strong-interface penetration | Runnable plasticity/interface examples; full benchmark-matched ductile/cohesive fracture workflows remain roadmap work |

## Quick Start

```bash
cd /path/to/phast

# Dynamic benchmarks (explicit, fast, O(N) per step)
python -u -m phast run configs/benchmarks/dynamic/B1_branching_glass.yaml --plots --gif
python -u -m phast run configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml --plots --gif
python -u -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --plots --gif
# Quasi-static benchmarks (implicit, iterative staggered)
python -u examples/quasistatic/miehe_tension/run.py --backend auto --preconditioner jacobi --all_outputs
python -u examples/quasistatic/miehe_shear/run.py --backend auto --preconditioner jacobi --all_outputs
python -u examples/quasistatic/three_point_bending/run.py --backend auto --preconditioner jacobi --all_outputs
python -u examples/quasistatic/l_shaped_panel/run.py --backend auto --preconditioner jacobi --plots
```

## Common CLI Flags

| Flag | Output |
|------|--------|
| `--plots` | PNG figures (damage, energy, convergence/load-displacement when relevant) |
| `--vtu` | VTU snapshots for ParaView |
| `--gif` | Animated GIF of damage evolution |
| `--h5` | H5 training data (must be requested explicitly) |
| `--all_outputs` | VTU + GIF + plots + profiler (does **not** include H5) |
| `--device cpu/cuda` | Compute device (always use `cpu` on Mac for float64) |
| `--output_dir DIR` | Custom output directory (default: run tag under example folder) |

## Output Layout

Standalone example runners write results under their canonical problem
folders. YAML runs write to the configured or CLI-provided `output_dir`;
for production and HPC runs, prefer an explicit run folder under the matching
example family.

```
examples/quasistatic/miehe_tension/
    run.py
    run_miehe_tension_cpu_YYYYMMDD_HHMMSS/  # standalone runner output

/path/to/results/B1_branching_glass/
    config.yaml
    run_metadata.json
    trajectory.zarr
    figures/
```

## Known Issues

- **MPS + float32 = wrong results**: CG damage solver needs float64. MPS on Mac
  doesn't support float64 natively, causing diffused (wrong) damage fields.
  Always use `--device cpu` on Mac.
- **Customer-ready plasticity/cohesive workflows**: sparse J2 mechanics,
  guarded ductile AT2 damage validation, cohesive residual/tangent smoke
  coverage, and mode-I/mixed-mode/contact/delamination-patch cohesive
  benchmarks plus a structural DCB-style cohesive smoke are available, but
  benchmark-matched coupled workflows remain roadmap work.
- **B7 COMSOL branching parity**: corrected H=20 half-plate and
  generalized-alpha parity runs are in flight; older H=40 half-plate debug
  runs should not be used as final COMSOL evidence.
