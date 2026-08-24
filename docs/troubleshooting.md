# Troubleshooting and Failure Modes

Phase-field fracture simulations are sensitive to units, mesh resolution,
boundary conditions, and time-step choices. This page lists common symptoms and
the first checks to perform before changing solver code.

## Numerical Mechanics FAQ

### Is $h\leq\ell_0/2$ sufficient?

No. It is a useful starting resolution criterion near the expected crack path,
not a convergence certificate. Assess $h$-refinement at fixed $\ell_0$ using
load-displacement response, fracture energy, crack path, and solver convergence.
Study $\ell_0$ separately because changing it changes the regularized model.

### Does every route use staggered convergence?

No. Supported quasistatic calculations perform inner mechanics-damage
iterations and ordinarily monitor displacement and damage changes. Explicit
dynamics performs one segregated pass per time step, with damage solved at the
configured cadence. The monolithic $(\mathbf{u},d)$ minimizer is experimental.

### Is the AT2 damage update explicit because mechanics is explicit?

No. For fixed history and the standard quadratic AT2 operator, the damage
residual is affine in $d$, but its finite-element discretization remains a
global elliptic linear solve. Explicit dynamics can therefore use an explicit
mechanics update while invoking an implicit damage solve at selected steps.

### What is the difference between history, bounds, and an active set?

The history maximum makes the crack-driving field nondecreasing. Nodal
no-healing is imposed separately through $d_{n+1}\geq d_n$. Projected CG
maintains the bound through an active set; `post_clamp` solves the unconstrained
system and clamps afterward, so it is not an active-set solution.

### Does adaptive explicit time stepping retry a failed step?

The automatic explicit step is bounded by an elastic-wave CFL estimate. The
optional damage-increment controller adjusts the subsequent step after the
current step is completed; it does not reject and recompute that completed
step. In quasistatic runs, the configured increment is a continuation/load
increment rather than physical time.

## Quick Triage

| Symptom | Likely cause | First checks |
|---|---|---|
| Whole domain damages immediately | Unit mismatch, load too large, missing pre-crack control, or inappropriate AT2 nucleation setup | Check `E`, `G_c`, `l0`, load units, and whether the example should use AT1 or a seeded notch. |
| Jagged or mesh-biased crack path | Mesh too coarse near the expected crack, or `h/l0` too large | Inspect `initial_conditions.png`; refine near notches, holes, and crack paths so $h \leq \ell_0/2$ where possible. |
| Damage grows in compression | Wrong energy split for the loading state | Prefer `amor` or `spectral` over `isotropic` when compression or bending is present. |
| Explicit dynamic run produces NaNs | CFL time step too large, unstable load ramp, or invalid material density | Reduce `dt_safety`, check density units, and validate wave-speed/time-step values. |
| Quasi-static solve stalls | Poor conditioning, over-aggressive load step, or unsupported backend path | Reduce load increment, inspect `backend: auto` with `python -m phast doctor`, and validate the config before running. |
| Results differ between machines | Different PyTorch/backend versions, precision, optional sparse solver, or mesh regeneration | Compare `run_lockfile.json`, `run_metadata.json`, mesh statistics, and backend status. |
| Missing plots or animations | Optional visualization dependency missing or output disabled | Check `visual_manifest.json`, output settings, and install documentation requirements if building figures. |

## Commands To Run First

Validate the environment:

```bash
python -m phast doctor
```

Validate the configuration without launching the full solve:

```bash
python -m phast run <config.yaml> --validate-only
```

Inspect the parsed setup:

```bash
python -m phast explain-config <config.yaml>
```

After a completed run, inspect the result folder:

```python
import phast

result = phast.load_result("runs/<case>")
print(result.metadata())
print(result.manifest())
print(result.visuals())
```

## Mesh and Regularization

The phase-field length $\ell_0$ is not just a material number; it also sets the
width of the damage band that the mesh must resolve. Near notches, holes, and
expected crack paths, start from

$$
h \leq \frac{\ell_0}{2}.
$$

If the crack path changes when the mesh is refined, treat the coarse result as
diagnostic rather than validated.

## Units

The public examples generally use the mm-N-MPa-s convention. Mixing SI values
with mm-scale geometry is a common source of immediate failure. Check:

- $E$ in MPa when length is in mm;
- $G_c$ in N/mm;
- density and time units for explicit dynamics;
- prescribed displacements and velocities in the same length/time system as
  the mesh.

## When To Open An Issue

Open a GitHub issue when the problem persists after the checks above. Include:

- the exact command;
- the configuration file path;
- `python -m phast doctor` output;
- the first traceback or warning;
- `run_manifest.json`, `run_metadata.json`, and `run_lockfile.json` when a run
  directory exists;
- a small image or `initial_conditions.png` if the issue is geometric or visual.

Also open an issue when the documentation does not explain what to do next.
Students and first-time users are welcome to submit incomplete diagnostic
information; include the command and first observed failure, and the
maintainers can help identify the next check.
