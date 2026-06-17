# Plasticity / Cohesive / PF-CZM Beta Release

Date: 2026-06-10

Recommended tag: `v0.16.2-plasticity-cohesive-beta.1`

## Release Position

This is a technical-preview beta validation release for plasticity,
cohesive-interface, diffuse-interface, and PF-CZM examples. It is appropriate
for researcher evaluation when accompanied by the capability boundary below.

Do not describe this release as a mature Abaqus/COMSOL-equivalent coupled
elastoplastic cohesive fracture product. The fully coupled plasticity +
phase-field + cohesive/PF-CZM production workflow remains gated.

## Supported Beta Slices

- Standalone J2 material-point validation.
- Sparse quasi-static mesh-level J2 mechanics with state commit/rollback,
  plastic-work accounting, and backend-selection evidence.
- Ductile plastic-work-driven AT2 damage validation and sensitivity study.
- Solver-driven diffuse-interface weak/strong deflection examples.
- Zero-thickness bilinear cohesive elements with mode-I, mixed-mode,
  contact-compression, delamination patch, structural DCB-style validation, and
  coupled brittle PF+cohesive validation examples.
- Wu PF-CZM uniaxial strength/length-scale validation with residual, convergence,
  telemetry, visuals, and manifest checks.
- Native Q4 isotropic mechanics + AT2 validation coverage.

## Reproduce the Validation Suite

The canonical reproducibility contract is:

```text
configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml
```

Each entry records the runner, exact launcher command, expected artifacts,
visual-manifest requirements, and claim boundary.

Focused local validation command:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_plasticity_interface_examples.py \
  tests/test_plasticity_j2.py \
  tests/test_mesh_j2_ductile_pf.py \
  tests/test_cohesive_elements.py \
  tests/test_pfczm_material_damage.py -q
```

Latest local result before this release note:

```text
83 passed, 2 warnings in 15.17s
```

Broader focused stack evidence from the merged PF-CZM/plasticity integration:

```text
139 passed, 6 warnings
```

Command used for that broader stack:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_plasticity_j2.py \
  tests/test_mesh_j2_ductile_pf.py \
  tests/test_plasticity_interface_examples.py \
  tests/test_cohesive_elements.py \
  tests/test_pfczm_material_damage.py \
  tests/test_quad_mesh_capability.py \
  tests/test_at1_nodal_h_guard.py \
  tests/test_degradation_functions.py \
  tests/test_quasi_static_spectral.py -q
```

## Required Outputs for Promotion

Promoted validation runs should retain:

- `summary.json`
- `config.yaml`
- `run_lockfile.json` and `run_manifest.json` when written by the runner
- CSV telemetry/results tables
- mesh artifacts where applicable
- `visual_manifest.json`
- review-safe PNG/GIF/MP4 outputs following `docs/visualisation_requirements.md`

## Known Release Blockers for Production Claim

- Fully coupled plasticity + phase-field + cohesive/PF-CZM product workflow.
- PF-CZM full TSL family, structural crack-growth, He-Hutchinson, Camanho
  mixed-mode, and PPR validation.
- Ductile fracture benchmark parity for Borden/Aldakheel/SENT/TPB-style cases.
- ASTM-style cohesive DCB/material data reduction and structural calibration.
- Functional cuDSS GPU backend promotion without fallback.
- General researcher YAML problem-definition layer for arbitrary plasticity,
  cohesive, and PF-CZM workflows.
