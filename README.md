# PhAST

<p align="center">
  <img src="docs/_static/brand/phast-banner.png" alt="PhAST: Phase-field Autograd Solver in Torch" width="820">
</p>

<p align="center">
  <a href="https://github.com/CEMS-Lab/PhAST/actions/workflows/ci-testing.yml"><img src="https://github.com/CEMS-Lab/PhAST/actions/workflows/ci-testing.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/CEMS-Lab/PhAST/actions/workflows/docs.yml"><img src="https://github.com/CEMS-Lab/PhAST/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <a href="https://cems-lab.github.io/PhAST/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-1f6feb" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
</p>

<p align="center">
  <a href="https://cems-lab.github.io/PhAST/">Documentation</a> |
  <a href="https://cems-lab.github.io/PhAST/example-gallery.html">Example gallery</a> |
  <a href="https://cems-lab.github.io/PhAST/installation.html">Installation</a> |
  <a href="https://colab.research.google.com/github/CEMS-Lab/PhAST/blob/main/notebooks/quickstart_colab.ipynb">Run in Colab</a>
</p>

**PhAST: Phase-field Autograd Solver in Torch** (`phast`) is a
PyTorch-native finite-element research solver for 2D phase-field fracture,
quasi-static validation, explicit dynamics, and beta plasticity/cohesive
validation workflows.

The project keeps the main fracture workflows in ordinary PyTorch tensors so
they can run on CPU, CUDA, and selected Apple Silicon paths, and so supported
solver components can participate in autograd-based optimisation. Public
capability boundaries are explicit: brittle phase-field fracture is the mature
core, while plasticity, cohesive interfaces, and PF-CZM are beta validation
slices until their production gates close.

## Representative Results

The panels below are lightweight documentation thumbnails generated from the
same workflows mapped in the hosted example gallery.

| | |
|---|---|
| <img src="docs/readme_showcase/dynamic_sent_damage.png" alt="Dynamic SENT damage evolution" width="390"><br><strong>Dynamic fracture</strong><br>Explicit SENT, branching, impact, and crack-propagation benchmark workflows. | <img src="docs/readme_showcase/qs_notched_holed_damage.png" alt="Quasi-static notched holed plate damage" width="390"><br><strong>Quasi-static fracture</strong><br>Implicit AT1/AT2 crack-path workflows with comparison artifacts and run metadata. |
| <img src="docs/readme_showcase/perforated_microstructure_damage.png" alt="Perforated plate microstructure damage field" width="390"><br><strong>Microstructured fracture</strong><br>Perforated and heterogeneous forward fracture cases with reproducible visualization artifacts. | <img src="docs/readme_showcase/solid_mechanics_materials.png" alt="Solid mechanics material kernels" width="390"><br><strong>Beta nonlinear failure</strong><br>J2 material kernels, cohesive operators, and ductile-validation slices under explicit capability gates. |

## What It Covers

| Area | What is included |
|---|---|
| Forward solver | Explicit dynamic fracture, staggered mechanics/damage solves, AT1/AT2 damage models, spectral and Amor-style splits, standard output/provenance files. |
| Quasistatic benchmarks | YAML-driven Ambati/COMSOL-style benchmark setups, rigid connector boundary conditions, compare scripts, and validation notes. |
| Plasticity and cohesive validation | Sparse quasi-static J2 mechanics, guarded ductile AT2 damage validation, cohesive mode-I/mixed-mode/contact/delamination-patch checks, and DCB-style structural benchmarks with retained validation evidence. |
| Documentation and reproducibility | Sphinx docs, config schema, run lockfiles, benchmark maps, capability matrix, and standard output conventions. |

## Install

Install directly from GitHub:

```bash
pip install "phast @ git+https://github.com/CEMS-Lab/PhAST.git"
python -m phast doctor
```

Or work from a local clone:

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
pip install -e .
```

Optional packages such as `pyamg`, `pymetis`, AmgX, PETSc/MUMPS, and cuDSS are
only needed for specific solver backends or HPC workflows. Documentation
dependencies live in `requirements-docs.txt`.

Quick import check:

```bash
python -c "import phast, torch; print(torch.__version__, torch.cuda.is_available())"
python -m phast doctor
```

The one-command installer attempts safe optional packages for the detected
machine and tries PETSc/MUMPS through conda-forge when `conda` or `mamba` is
available:

```bash
bash install.sh            # platform auto-detect + workflow backend checks
bash install.sh cpu --no-direct  # skip the PETSc/MUMPS attempt
```

## Recommended Defaults By Workflow

| Workflow | Default policy |
|---|---|
| Explicit dynamic fracture | `solver_type: explicit`, CFL-controlled `dt_safety: 0.8`, `damage_every: 1` for reference validation. Use `damage_every: 2` or `3` only after a subcycling sensitivity check. |
| Quasi-static fracture | `solver_type: quasi_static`, `backend: auto`, `preconditioner: jacobi`, `stagger_criterion: linf`, and MP4/Zarr outputs when trajectories or animations are requested. |
| Spectral/Amor implicit QS on CPU/HPC | Install PETSc/MUMPS where possible; `backend: auto` selects MUMPS after a smoke test, then SciPy SuperLU, then CG. |
| Cohesive contact | Use the sparse quasi-static path with `backend: auto`; enable normal-contact penalty only for contact benchmarks that need it. |
| J2 plasticity | Use the guarded sparse quasi-static plasticity path with `backend: auto`; unsupported material combinations fail early rather than falling back silently. |

Reproducibility YAMLs are organised by workflow under `configs/benchmarks/`.
Dynamic and quasi-static entries are direct `python -m phast run`
problem configs. Cohesive-contact and plasticity entries are command manifests
when the workflow is driven by a specialised example module.

## Quick Start

Validate a config without running the solver:

```bash
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --validate-only
```

Run a YAML benchmark:

```bash
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml
```

Inspect a config before spending compute:

```bash
python -m phast explain-config configs/benchmarks/dynamic/B3_dynamic_sent.yaml
```

Useful starting points:

| Need | Start here |
|---|---|
| First runnable workflow | `docs/getting-started.md` |
| Hosted/local documentation index | `docs/index.md` |
| Project capability boundaries | `docs/user_guide/capability_matrix.md` |
| Customer validation tutorial map | `docs/tutorials.md` |
| Plasticity/cohesive examples | `examples/plasticity_interface/README.md` |
| Benchmark-to-config map | `configs/BENCHMARK_RUN_MAP.md` |
| Runnable examples | `examples/README.md` |
| Standard output files | `docs/STANDARD_OUTPUTS.md` |
| Configuration reference | `configs/REFERENCE.yaml` |

## Customer Claim Boundary

Customer-facing claims should follow
[`docs/user_guide/capability_matrix.md`](docs/user_guide/capability_matrix.md).
The current plasticity/cohesive stack is validation-ready for sparse J2
mechanics, a guarded ductile AT2 coupling example, cohesive operator
benchmarks, and a DCB-style structural cohesive smoke. It is not yet an
Abaqus/COMSOL-equivalent coupled
PF-plasticity-CZM product workflow or ASTM-calibrated DCB data-reduction tool.

## Repository Map

| Path | Purpose |
|---|---|
| `src/phast/` | Solver package, mechanics/damage kernels, config handling, CLI entry points. |
| `configs/` | YAML benchmark and workflow definitions plus schema/reference files. |
| `examples/` | Runnable forward, quasistatic, plasticity/cohesive, and comparison workflows. |
| `docs/` | Sphinx documentation, user guide, benchmark notes, API notes, and developer docs. |
| `tests/` | Focused regression and capability tests. |
| `papers/` | Manuscript and paper-specific supporting material. |
| `scripts/` | Diagnostics, benchmark helpers, analysis, and artifact-generation utilities. |

## Documentation

Build the Sphinx documentation locally with:

```bash
pip install -r requirements-docs.txt
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

The hosted docs are published at:

https://cems-lab.github.io/PhAST/

GitHub Actions workflows are currently configured as manual-only
(`workflow_dispatch`) to avoid expensive automatic runs. Start CI, docs, wheel,
or Claude workflows from the GitHub Actions tab when you explicitly want them.

## Development Notes

Before starting new work, branch from current `main` and keep changes scoped to
one lane when possible:

```bash
git switch main
git pull --ff-only
git switch -c agent/<lane>-YYYY-MM-DD
```

For multi-agent work, read `docs/reconciliation/2026-06-02-agent-worktree-plan.md`
and keep ownership boundaries clear. Do not commit generated heavy outputs unless
they are explicitly part of the requested artifact.
