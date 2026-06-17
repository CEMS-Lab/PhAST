# Submitting PhAST to conda-forge

## Pre-submission checklist

- [ ] Cut a tagged release `v0.16.2` on GitHub (recipe `source` must point to
      a tagged release, not `main`)
- [x] `LICENSE` file present at repo root (MIT)
- [ ] All `requirements.run` deps verified to exist on conda-forge:
      pytorch, numpy, scipy, matplotlib-base, h5py, meshio, python-gmsh,
      pillow, pyyaml, pyamg, pymetis, petsc4py, mumps
- [x] `requires-python` in pyproject.toml (`>=3.10`) matches recipe
      `python >=3.10`
- [x] No `editable=True` install hooks in pyproject (pip install . works)
- [ ] Prepare the conda-forge recipe in a separate staged-recipes working
      tree after the public tag exists. Prefer a release tarball URL plus
      sha256.
- [ ] Compute sha256: `curl -L https://github.com/CEMS-Lab/PhAST/archive/v0.16.2.tar.gz | sha256sum`
- [ ] Run the release preflight after the public tag exists:
      `python -m phast doctor`

## Submission steps

1. Fork `https://github.com/conda-forge/staged-recipes`.
2. Clone the fork and create a branch:
   ```bash
   git checkout -b add-phast
   ```
3. Create the recipe in place:
   ```bash
   mkdir -p recipes/phast
   # Draft recipes/phast/recipe.yaml from pyproject.toml and the tagged release.
   ```
4. Commit and push:
   ```bash
   git add recipes/phast/recipe.yaml
   git commit -m "Add phast recipe"
   git push -u origin add-phast
   ```
5. Open a PR against `conda-forge/staged-recipes` using the standard PR
   template (the template is auto-populated; tick the relevant boxes).
6. The `conda-forge` linter bot runs automatically. Address any feedback
   on platform builds, missing deps, or naming conventions.
7. A conda-forge maintainer reviews and merges. After merge, a
   `phast-feedstock` repo is auto-created under the
   `conda-forge` org with the recipe maintainers as collaborators.

## Post-merge maintenance

- The `regro-cf-autotick-bot` opens version-bump PRs on the feedstock
  whenever a new GitHub tag is pushed. Review, ensure tests pass, merge.
- To add or change runtime deps, edit `recipe.yaml` on the feedstock and
  open a PR; the same CI matrix runs.
- Maintain the `recipe-maintainers` list — currently `allamaprabhuani`.

## Naming note

The public conda-forge package and feedstock should be `phast` /
`phast-feedstock`. Older development repository names must not be used in
staged-recipes paths, package names, or validation imports.

## NOT to be done from this repo

This file documents the user-driven submission flow. The repo itself does
NOT open the PR to `conda-forge/staged-recipes`; that requires forking
under the user's GitHub identity.
