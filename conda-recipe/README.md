# conda-forge recipe for PhAST

This directory stages the conda-forge recipe for `phast` so the
package can eventually be installed via:

```bash
conda install -c conda-forge phast
```

## Files

- `recipe.yaml` — the v1-format conda-forge recipe (schema_version: 1)
- `SUBMISSION.md` — step-by-step guide to submit to `conda-forge/staged-recipes`

## Layout decision

The recipe is `noarch: python` because the package is pure-Python (a thin
PyTorch + SciPy + gmsh layer). PyTorch handles its own CPU/CUDA/MPS variants;
no compiled extensions live in this repo.

PyAMG, PyMETIS, PETSc/MUMPS, and petsc4py are included in
`requirements.run` for the lab release recipe so `backend: auto` has a strong
CPU/HPC path out of the box. CUDA-specific packages such as CuPy/cuDSS and
AmgX remain user-installed because they depend on local driver and system
library availability.

## Pyproject metadata used

- name: `phast`
- version: `0.16.2`
- license: MIT
- python: `>=3.9`
- runtime deps: torch, numpy, scipy, matplotlib, h5py, meshio, gmsh, Pillow, pyyaml, pyamg, pymetis, petsc4py, mumps

## Notes

- conda-forge calls the gmsh Python package `python-gmsh`, and matplotlib's
  headless build is `matplotlib-base` (preferred over `matplotlib` to avoid
  pulling Qt by default).
- The AmgX extra has no conda-forge package and stays pip-only.
