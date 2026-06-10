# Example gallery

This section maps core solver capabilities to runnable workflows in the
repository, so users can quickly check the project’s practical envelope.

## Representative results

The panels below are lightweight documentation thumbnails, not raw benchmark
archives. They point to the same workflows listed in the sections that follow.

<div class="phast-card-grid phast-thumb-grid">
  <div class="phast-card phast-thumb-card">
    <img src="readme_showcase/qs_notched_holed_damage.png" alt="Quasi-static notched holed plate damage field">
    <h3>Quasi-static fracture</h3>
    <p>Implicit AT1/AT2 crack-path workflows with comparison reports and
    standard run metadata.</p>
  </div>
  <div class="phast-card phast-thumb-card">
    <img src="readme_showcase/dynamic_sent_damage.png" alt="Dynamic SENT damage evolution">
    <h3>Dynamic fracture</h3>
    <p>Explicit dynamics for SENT, branching, impact, and crack-propagation
    benchmarks.</p>
  </div>
  <div class="phast-card phast-thumb-card">
    <img src="readme_showcase/perforated_microstructure_damage.png" alt="Perforated plate microstructure damage field">
    <h3>Microstructured media</h3>
    <p>Perforated-plate and heterogeneous fracture cases used for forward
    fracture morphology studies.</p>
  </div>
  <div class="phast-card phast-thumb-card">
    <img src="readme_showcase/qs_force_displacement.png" alt="Force displacement curve">
    <h3>Engineering outputs</h3>
    <p>Force-displacement, energy, convergence, lockfile, and visualization
    artifacts from reproducible runs.</p>
  </div>
  <div class="phast-card phast-thumb-card">
    <img src="readme_showcase/solid_mechanics_materials.png" alt="Solid mechanics material kernels">
    <h3>Solid mechanics kernels</h3>
    <p>Linear elastic, hyperelastic, and J2 material-kernel demonstrations used
    by the nonlinear validation path.</p>
  </div>
</div>

## Capability-first pathways

### Quasi-static fracture

The quasi-static family is the production path for many literature comparisons.

- **SENT / SENS / TPB / L-panel**: shipped benchmark configs under
  `configs/benchmarks/quasistatic/` and compare scripts in matching example
  directories.
- **Run example**:  
  `python -m phast run configs/benchmarks/quasistatic/QS_notched_holed_plate.yaml --validate-only`
- **Expected outputs**: `config.yaml`, `run_lockfile.json`, `results.csv`, plots.
  Run `compare.py` in `examples/quasistatic/notched_holed_plate/` for
  benchmark-level pass/fail summary.

### Explicit dynamics fracture

- **Branching, impact, perforation, Kalthoff-Winkler**:
  `configs/benchmarks/dynamic/*.yaml` plus matching run folders under
  `examples/dynamic/*`.
- **Benchmark workflow**:
  1. Launch via `python -m phast run <cfg>`
  2. Run `compare.py` in the corresponding `examples/dynamic/<case>/` directory.
  3. Save comparison artifacts (`compare.txt`, `compare.png`) in run folder.

### Solid mechanics and solver diagnostics

- `examples/solid_mechanics/linear_plate.py`  
  Lightweight CPU-safe baseline for baseline linear mechanics.
- `examples/solid_mechanics/mixed_precision_cg_demo.py`  
  Mixed-precision CG stability and residual behavior.
- `examples/solid_mechanics/dynamic_oscillator_genalpha.py`  
  generalized-alpha exploration for implicit-dynamics prototyping.

### Beta plasticity, cohesive, and PF-CZM validation

- **J2 plasticity and ductile PF-plasticity**:
  `examples/plasticity_interface/run_j2_validation.py` and
  `examples/plasticity_interface/run_ductile_pf_plasticity_validation.py`.
- **Cohesive/interface benchmarks**:
  mode-I jump, mixed-mode, contact-compression, delamination patch, structural
  DCB and coupled PF-cohesive smoke workflows live under
  `examples/plasticity_interface/`.
- **Manifested reproduction set**:
  `configs/benchmarks/plasticity_interface/manifests/customer_validation_examples.yaml`.
- **Generated visual outputs**:
  the scripts write `*.png`, manifest metadata, and summaries into their chosen
  output directories. These are beta validation artifacts until the nonlinear
  production gates in the capability matrix are closed.

## Tutorial map (ordered by onboarding value)

- [Getting started](getting-started.md)
- [Tutorial 0: Quickstart](tutorial/00_quickstart.md)
- [Tutorial 1: Phase-field primer](tutorial/01_phase_field_primer.md)
- [Tutorial 3: Problem setup](tutorial/03_setting_up_your_problem.md)

## Showcase assets

A compact media index is curated under
[`docs/readme_showcase/README.md`](readme_showcase/README.md).
These images are documentation-ready and can be reused for capability pages,
proposals, and release notes.

## Capability boundaries

For current non-hardened features, review
[Capability matrix](user_guide/capability_matrix.md).  
This doc is also the source of truth for what can be promised publicly.
