# Troubleshooting and Failure Modes

Phase-field fracture simulations are sensitive to units, mesh resolution,
boundary conditions, and time-step choices. This page lists common symptoms and
the first checks to perform before changing solver code.

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
