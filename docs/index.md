# PhAST

**Phase-field Autograd Solver in Torch**

<img class="phast-hero-brand" src="_static/brand/phast-banner.png" alt="PhAST brand banner: Phase-field Autograd Solver in Torch">

[![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CEMS-Lab/PhAST/blob/main/notebooks/quickstart_colab.ipynb)

PhAST is a PyTorch-native 2D phase-field fracture framework for forward
simulation, quasi-static validation, explicit dynamics, and beta
plasticity/cohesive validation workflows. It is GPU-aware,
autograd-integrated, and JIT-free, so users can inspect and debug mechanics,
damage, and cohesive operators directly in Python. Public documentation keeps
capability boundaries explicit: brittle fracture is the mature core, while
plasticity, cohesive interfaces, and PF-CZM are beta validation slices until
their production gates close.

<div class="phast-card-grid">
  <div class="phast-card">
    <h3>Differentiable Mechanics</h3>
    <p>Pure PyTorch tensor kernels keep mechanics and damage operators
    inspectable, testable, and compatible with autograd where the released
    solver path supports it.</p>
  </div>
  <div class="phast-card">
    <h3>Phase-Field Fracture</h3>
    <p>AT1/AT2 brittle fracture, energy splits, quasi-static/static solves,
    explicit dynamics, Zarr trajectories, and reproducible visualization
    artifacts.</p>
  </div>
  <div class="phast-card">
    <h3>Beta Nonlinear Failure</h3>
    <p>Sparse J2 plasticity, ductile-driving AT2 validation, cohesive
    elements, coupled brittle PF+cohesive smoke, and PF-CZM strength
    calibration.</p>
  </div>
  <div class="phast-card">
    <h3>CEMS Lab Release Path</h3>
    <p>Release tooling stages an explicit public payload and scans it before
    any synchronization to the CEMS Lab organization.</p>
  </div>
</div>

## Quick Commands

```bash
pip install -e .
python -m phast doctor
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --validate-only
python -m phast explain-config configs/benchmarks/dynamic/B3_dynamic_sent.yaml
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml
```

Legacy entry points remain available during the transition:

```bash
python -m phast doctor
```

Every normal YAML run writes reproducibility metadata, including the resolved
configuration and run lockfile, into the output directory.

## Start Here

| Goal | First page |
|---|---|
| Install and run a first case | [Getting started](getting-started.md) |
| Understand supported physics | [Capability matrix](user_guide/capability_matrix.md) |
| Review beta plasticity/cohesive scope | [Plasticity/cohesive beta release](plasticity_cohesive_beta_release.md) |
| Browse examples | [Example gallery](example-gallery.md) |

```{toctree}
:maxdepth: 2
:caption: Get Started

getting-started
installation
quickstart
verify-install
showcase
tutorials
tutorial/00_quickstart
tutorial/01_phase_field_primer
tutorial/03_setting_up_your_problem
plasticity-onboarding
plasticity_cohesive_beta_release
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/overview
user_guide/problem_types
user_guide/capability_matrix
user_guide/physics
user_guide/configuration
user_guide/sparse_solve
user_guide/meshes
user_guide/performance
```

```{toctree}
:maxdepth: 2
:caption: Example Gallery

example-gallery
examples/solid_mechanics
benchmarks/catalogue
benchmarks/examples
readme_showcase/README
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/sparse_solve
api/time_integrators
api/mixed_precision_cg
api/adaptive
api/process_zone
```

```{toctree}
:maxdepth: 1
:caption: Performance & Validation

STANDARD_OUTPUTS
paper1-benchmarks
customer_readiness
changelog
```

```{toctree}
:maxdepth: 2
:caption: Developer

developer/architecture
developer/modules
developer/matrix_free
developer/testing
```

```{toctree}
:maxdepth: 1
:caption: Supplemental

COMSOL_COMPARISON
BEAT_THIS_BENCHMARK
QUICKSTART_PACKAGED_BENCHMARK
distribution-strategy
visualisation_requirements
user_guide/data_and_devices
user_guide/roadmap
user_guide/setup_problems
```

```{toctree}
:maxdepth: 1
:caption: Community

community
```
