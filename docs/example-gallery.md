# Example gallery

This section maps core solver capabilities to runnable workflows in the
repository, so users can quickly check the project’s practical envelope.

## Representative results

The panels below are lightweight documentation thumbnails, not raw benchmark
archives. They point to the same workflows listed in the sections that follow.

## Start With These

| Example | Status | Modality | Command | Expected outputs | Inspect with |
|---|---|---|---|---|---|
| Miehe tension | Production | YAML-first | `python -m phast run examples/quasistatic/miehe_tension/config.yaml --output_dir runs/miehe_tension` | CSV histories, run metadata/lockfile, and checked-in gallery visuals/comparison artifacts; run explicit postprocessing if you need regenerated animations from a fresh solve | `phast.load_result("runs/miehe_tension")` |
| Notched-holed plate | Production | YAML-first | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --output_dir runs/notched_holed_plate` | final damage, response CSVs, comparison report, visual manifest | `phast.load_result("runs/notched_holed_plate")` |
| Linear plate | Production | YAML-first | `python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml --output_dir runs/linear_plate` | response curve, displacement, von Mises, strain energy, manifests | `phast.load_result("runs/linear_plate")` |
| Kalthoff-Winkler | Public candidate | YAML-first | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --output_dir runs/B2_kalthoff_winkler` | setup preview, damage image, energy/history CSVs, manifests | `phast.load_result("runs/B2_kalthoff_winkler")` |

Use the [capability matrix](user_guide/capability_matrix.md) before assuming a
workflow status. Use the
[promoted example contract](user_guide/example_contract.md) when adding or
auditing a gallery entry.

<div class="phast-card-grid phast-thumb-grid">
  <div class="phast-card phast-thumb-card">
    <img src="../assets/qs_notched_holed_damage.png" alt="Quasi-static notched holed plate damage field">
    <h3>Quasi-static fracture</h3>
    <p>Implicit AT1/AT2 crack-path workflows with comparison reports and
    standard run metadata.</p>
  </div>
  <div class="phast-card phast-thumb-card">
    <img src="../assets/kalthoff_winkler_long_crack.gif" alt="Kalthoff-Winkler crack growth">
    <h3>Dynamic fracture</h3>
    <p>Explicit dynamics for impact, branching, and perforated-plate
    crack-growth benchmarks.</p>
  </div>
  <div class="phast-card phast-thumb-card">
    <img src="../assets/perforated_microstructure_damage.png" alt="Perforated plate microstructure damage field">
    <h3>Microstructured media</h3>
    <p>Perforated-plate and heterogeneous fracture cases used for dataset and
    morphology studies.</p>
  </div>
  <div class="phast-card phast-thumb-card">
    <img src="../assets/qs_force_displacement.png" alt="Force displacement curve">
    <h3>Engineering outputs</h3>
    <p>Force-displacement, energy, convergence, lockfile, and visualization
    artifacts from reproducible runs.</p>
  </div>
  <div class="phast-card phast-thumb-card">
    <img src="../assets/solid_mechanics_materials.png" alt="Solid mechanics material models">
    <h3>Solid mechanics</h3>
    <p>Linear elastic, hyperelastic, and J2 mechanics examples backed by the
    same sparse-solve and material-update paths used by the solver.</p>
  </div>
</div>

## Capability-first pathways

### Quasi-static fracture

The quasi-static family is the production path for many literature comparisons.

- **SENT / SENS / TPB / L-panel**: shipped benchmark configs under
  `configs/benchmarks/quasistatic/` and compare scripts in matching example
  directories.
- **Run example**:
  `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --validate-only`
- **Expected outputs**: `config.yaml`, `run_lockfile.json`, `results.csv`, plots.
  Run `compare.py` in `examples/quasistatic/notched_holed_plate/` for
  benchmark-level pass/fail summary.

### Explicit dynamics fracture

- **Branching, impact, perforation, Kalthoff-Winkler**:
  `configs/benchmarks/dynamic/*.yaml` plus matching run folders under
  `examples/dynamic/*`.
- **B7 dynamic branching**:
  `examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml` is the
  public-candidate full-plate COMSOL cross-check.
- **Benchmark workflow**:
  1. Launch via `python -m phast run <cfg>`
  2. Run `compare.py` in the corresponding `examples/dynamic/<case>/` directory.
  3. Save comparison artifacts (`compare.txt`, `compare.png`) in run folder.

### Solid mechanics FEA

- `python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml`
  Linear elastic plate FEA with displacement, stress, deformed-shape, response,
  and visual-manifest outputs.
- `python -m phast run examples/solid_mechanics_beta/neohookean_plate/config.yaml`
  Nonlinear neo-Hookean cantilever FEA with Newton convergence and final-state
  field outputs.
- `python -m phast run examples/solid_mechanics_beta/j2_bar/config.yaml`
  Mesh-level J2 plasticity bar FEA with von Mises and equivalent-plastic-strain
  fields.

### Beta plasticity, cohesive, and PF-CZM validation

- **J2 plasticity and ductile PF-plasticity**:
  `examples/plasticity_interface_beta/run_j2_validation.py` and
  `examples/plasticity_interface_beta/run_ductile_pf_plasticity_validation.py`.
- **Cohesive/interface benchmarks**:
  mode-I jump, mixed-mode, contact-compression, delamination patch, structural
  DCB, diffuse-interface, and coupled PF-cohesive validation workflows live under
  `examples/plasticity_interface_beta/`.
- **Manifested reproduction set**:
  `configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml`.
- **Generated visual outputs**:
  the scripts write `*.png`, manifest metadata, and summaries into their chosen
  output directories. These are beta validation artifacts until the nonlinear
  production gates in the capability matrix are closed.

## Capability boundaries

For current non-hardened features, review
[Capability matrix](user_guide/capability_matrix.md).
This doc is also the source of truth for what can be promised publicly.

## Sparse direct vs CG inner solve


| Workflow | Public route | Evidence to keep |
| -------- | ------------ | ---------------- |
| Miehe tension | `python -m phast run examples/quasistatic/miehe_tension/config.yaml` | run manifests, CSV histories, damage animation |
| Notched-holed plate | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml` | setup preview, response plot, final damage |

Driver: [`sparse_solve`](api/sparse_solve.md). Speedup is wall-time of the
sparse direct path versus the matrix-free CG inner-solve path on the same mesh,
tolerance, backend stack, and output settings. See
[`Performance and Reproducibility`](performance_reproducibility/index.md) for
the reporting checklist before publishing numbers.
