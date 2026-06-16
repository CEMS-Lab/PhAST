# Quasi-static Miehe SENT

Validated single-edge-notched tension benchmark based on Miehe et al. (2010)
and the PhaseFieldX 1711 reference response.

This folder is self-contained: the YAML configuration, mesh, validation report,
and promoted visual outputs are kept beside each other.

## Run

Recommended journey: inspect the fluent authoring shape when useful -> run the
checked-in YAML deck with
`python -m phast run examples/quasistatic/miehe_tension/config.yaml` ->
standard result directory. The checked-in `config.yaml` is the canonical public
input deck for exact reproduction.

```bash
python -m phast run examples/quasistatic/miehe_tension/config.yaml \
  --output_dir examples/quasistatic/miehe_tension/run_local
```

## Equivalent Fluent API

Use this as the Python authoring shape for the same promoted runner family.
For public reproduction, run the checked-in YAML deck above.

```python
import phast

result = (
    phast.Problem("Miehe SENT")
    .geometry("miehe_tension", L=1.0, a=0.5, l0=0.015,
              h_crack=0.001875, h_coarse=0.05)
    .region("body", kind="domain")
    .region("bottom", from_mesh="bottom")
    .region("top", from_mesh="top")
    .material("glass", region="body", E=210000.0, nu=0.3,
              Gc=2.7, l0=0.015, rho=7.8e-09,
              eta_residual=1.0e-07, energy_split="isotropic",
              pf_model="AT2", plane_stress=False)
    .boundary_condition("fix", region="bottom", dof="x", name="clamp_x")
    .boundary_condition("fix", region="bottom", dof="y", name="clamp_y")
    .boundary_condition("displacement", region="top", dof="y",
                        value=1.0, name="pull_top")
    .analysis_step(
        "load",
        kind="quasi_static",
        controls={"protocol": "cyclic", "cyclic_phases": "0.005:50,0.008:300",
                  "num_steps": 350, "dt": 1.0},
        active_boundary_conditions=["clamp_x", "clamp_y", "pull_top"],
    )
    .solver("quasi_static", stagger_tol=1.0e-08, max_stagger=500,
            preconditioner="jacobi", backend="auto",
            fail_on_mechanics_nonconvergence=False)
    .outputs(
        fields=[{"name": "trajectory", "every": 1, "format": "zarr"}],
        histories=[{"name": "reaction_force", "region": "bottom", "dof": "y"}],
        plots=True,
        gif=True,
    )
    .run(output_dir="runs/miehe_tension", return_result=True)
)
```

The config.yaml is the canonical public input deck; the snippet documents
the equivalent fluent authoring surface and output intent.

## Promoted Result

The current promoted package is HPC job `37992`.

| Final damage | Crack evolution |
|---|---|
| ![Miehe SENT final damage](damage_final.png) | ![Miehe SENT damage evolution](damage_evolution.gif) |

| Quantity | Reference | Promoted result | Status |
| --- | ---: | ---: | --- |
| Peak reaction | 0.7012 kN | 0.6936 kN, 1.08% error | PASS |
| Pre-peak L2 error | PhaseFieldX 1711 | 1.70% | PASS |
| Dissipated-energy error | PhaseFieldX 1711 | 5.38% | PASS |

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
- `staggered_convergence.png`
- `damage_final.png`
- `damage_evolution.mp4`
- `damage_evolution.gif`
- `compare.png`
- `compare_report.txt`
- `thumbnail.png`

The full snap-back branch is reported as informational only. Robust post-peak
traversal requires arc-length or another continuation strategy, so acceptance is
gated on the peak, pre-peak response, and dissipated energy.
