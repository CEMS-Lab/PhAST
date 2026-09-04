# Verify Install

## Verification ladder

Use the checks in this order. Each rung is stronger and more expensive than
the previous one:

1. **Doctor** reports what the current environment can provide.
2. **Sanitizer** performs a bounded single-element CPU calculation and plot.
3. **Validate-only** checks a public YAML configuration without solving it.
4. **Completed run** executes the selected example and produces a result bundle.

Passing a lower rung does not certify the next one. In particular,
`--validate-only` is schema and path preflight; it is not runtime or scientific
validation.

## Quick single-element sanitizer

From a source checkout, run:

```bash
python run_sanitizer.py
```

The sanitizer exercises one T3 element, the fixed-history AT2 damage solve,
SciPy sparse linear algebra, and headless Matplotlib output. It reports the CPU,
CUDA, MPS, PETSc, PyAMG, and PyVista availability it observes. Optional backends
are reported as optional rather than required.

The script is designed to be quick, but two seconds is not a universal
wall-clock guarantee: Python and PyTorch cold imports vary across operating
systems and machines. The reported kernel-and-plot time excludes imports.

Use this page after installation, on a new workstation, or after loading a new
HPC environment. The goal is to confirm that PhAST can import, see the expected
optional backends, and validate a public configuration before launching a full
solve.

## Environment Doctor

Run:

```bash
python -m phast doctor
```

The command reports:

- Python, PyTorch, CUDA, and MPS availability;
- core packages such as NumPy, SciPy, Gmsh, meshio, Zarr, and ffmpeg;
- optional sparse-direct backends such as PETSc/MUMPS and cuDSS;
- the backend selected by `backend: auto` on the current machine;
- recommended defaults by problem class.

Typical CPU/macOS output begins like:

```text
PhAST environment doctor
==================================
Python:   3.10.x
PyTorch:  2.x
CUDA:     False
MPS:      True

Core packages:
  numpy:  ...
  scipy:  ...
  gmsh:   ...
  meshio: ...
```

Backend availability is machine-dependent. Missing PETSc/MUMPS or cuDSS is not
an installation failure unless the workflow you intend to run requires that
backend.

## Validate A Public Configuration

Before launching a full run, validate one public example:

```bash
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only
```

Expected output:

```text
OK: examples/dynamic/B2_kalthoff_winkler/config.yaml passes schema validation.
```

This check confirms that the YAML parser, schema validation, material presets,
and example path resolution are working without allocating the full solver.

## Inspect A Setup

For a human-readable setup summary, run:

```bash
python -m phast explain-config examples/dynamic/B2_kalthoff_winkler/config.yaml
```

Use this output to confirm the material, geometry, solver type, output settings,
and `device.compile` policy before running a costly simulation.

## Common Outcomes

| Outcome | Meaning | Next step |
|---|---|---|
| `scipy SuperLU: True` | Portable CPU sparse-direct fallback is available. | Good for small or moderate sparse checks. |
| `PETSc/MUMPS: False` | MUMPS is not active in this environment. | Install a consistent PETSc/petsc4py/MUMPS stack only if needed. |
| `CUDA: False` | PyTorch cannot see an NVIDIA GPU. | Use CPU validation or install a CUDA-compatible PyTorch wheel. |
| `MPS: True` | Apple Silicon acceleration is visible. | Prefer CPU float64 for verification-sensitive fracture runs. |
| Validation passes | The config is structurally valid. | Run the example with an explicit `--output_dir`. |

If these checks fail, use [Troubleshooting](troubleshooting.md) before opening an
issue. The sanitizer should be the first completed computation to compare
across platforms.

If the instructions themselves are unclear, or you cannot determine whether an
outcome is expected, open a
[documentation or support issue](https://github.com/CEMS-Lab/PhAST/issues/new/choose).
New users are not expected to diagnose solver internals before asking for help.
