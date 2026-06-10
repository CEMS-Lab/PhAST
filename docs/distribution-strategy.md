# Distribution strategy

Strategic anchor for: "where does this new install path go?" and "should
this optional-backend PR merge?". Pairs with [`installation`](installation.md)
and the root-level
[`INSTALL.md`](https://github.com/CEMS-Lab/PhAST/blob/main/INSTALL.md) /
[`COMSOL_COMPARISON.md`](https://github.com/CEMS-Lab/PhAST/blob/main/COMSOL_COMPARISON.md)
(this doc does not duplicate their content; it explains the *shape* of the
install surface).

## Killer app

```bash
pip install git+https://github.com/CEMS-Lab/PhAST.git
```

One line. A short install later, the user can validate and run the documented
Miehe SENT configuration from a normal Python environment.

This is the FEniCS / Akantu UX advantage. They ship great science behind
a wall of CMake, dolfinx C++ ABI breaks, MFront, and dependency-version
gymnastics. We ship `pip install` and the user is done.

Every distribution decision below defends that one-liner. If a change
slows or complicates the default `pip install`, it does not belong on
tier 1 — push it up the ladder.

## Comparison: COMSOL / Abaqus

| Property                        | COMSOL / Abaqus              | phast           |
|---------------------------------|------------------------------|---------------------------|
| Install medium                  | bundled installer (~5-15 GB) | `pip install` (~50 MB)    |
| License                         | commercial, node-locked      | open source               |
| Dependency strategy             | vendored everything          | rely on PyPI ecosystem    |
| Platform variants               | Win / Mac / Linux pre-built  | any platform pip supports |
| GPU support                     | shipped, opt-in module       | torch native              |
| Autograd through the model      | no                           | yes (the differentiator)  |
| "Run installer, it works" UX    | yes                          | yes (tier 1 + pip)        |
| Time-to-first-solve             | minutes (after install)      | seconds                   |

We deliberately do **not** replicate the bundled-installer model 1:1:

1. Free + tiny team — we cannot afford to build, test, and sign 15
   platform variants the way COMSOL can.
2. The Python ecosystem expects `pip` / `conda`, not `~/comsol_install/`.
   A bundled installer would feel alien to our users (researchers on
   HPC, students with PyTorch already installed).
3. We can approximate the "everything works" property via **conda-forge**
   (binary ecosystem) and **wheel-bundled binaries** (cibuildwheel),
   which are the open-source analogues.

But we keep their *load-bearing property*: user runs install, everything
works, no version dance.

## The 5-tier install ladder

### Tier 1 — `pip install` default (shipped)

`pip install git+...` pulls torch + numpy + scipy and that's it. The
portable CPU path runs with SciPy SuperLU where sparse-direct mechanics are
needed, while supported PyTorch-native components remain available for
autograd workflows. Works on every platform pip works on (Mac arm64/x86_64,
Linux, WSL, Windows).

CI gate: [`.github/workflows/install-promise.yml`](../.github/workflows/install-promise.yml)
verifies Mac + Ubuntu x Py3.10 / 3.11 / 3.12 on the manual/scheduled
install-promise workflow. Merged in #457.

### Tier 2 — cibuildwheel pre-built wheels (shipped, #462)

Pre-built `linux-x86_64` and `macos-arm64` wheels x Py3.10 / 3.11 / 3.12
via the cibuildwheel matrix. Faster install (no source step) and a
clean place to bundle private `.so` artefacts later if needed.

### Tier 3 — conda-forge feedstock (shipped, #461)

```bash
conda install -c conda-forge phast
```

Pulls torch + scipy + optionally `petsc4py`, `mumps`, `nvmath`, all
binary-compatible against each other. This is the open-source analogue
of the COMSOL-bundled UX: the conda-forge build farm guarantees ABI
compatibility we cannot guarantee from PyPI alone.

### Tier 4 — Docker / Singularity image (deferred)

For HPC users who want zero install. Estimated ~3-8 GB image, single
`docker run` / `singularity exec`. Defer until tier 3 is in. Mac users
are already covered by tier 1; HPC users are mostly covered by the
native PETSc module path documented in #460.

### Tier 5 — bundled installer (skipped)

Python anti-pattern in our domain — researchers already have a Python
environment, a bundled installer would fight `conda` / `venv` / `module
load`. We skip and lean on tier 3 instead.

## Held-PR merge gate

Optional-backend PRs (currently #378 PETSc/MUMPS, #379 cuDSS, #382 MUMPS
symbolic cache, #458 cuDSS API follow-up) are **draft and stay draft**
until each clears two gates:

1. **Verified working install on at least one platform.** Today's HPC
   validation surfaced a `petsc4py` ABI mismatch against the pip-built
   wheel and an incorrect API guess for cuDSS / nvmath 0.9. The
   architecture is right; the install path is what we have not yet
   pinned. Each held PR needs an explicit comment of the form: "install
   path X works on platform Y, validated by SLURM run Z" before it
   leaves draft.
2. **Skipif gracefulness.** `_petsc_functional()` and
   `_cudss_functional()` runtime smoke-test the backend (not just
   import) and fall back to SciPy SuperLU if anything fails. Merged in
   #459 — this is the safety net that lets tier 1 stay clean regardless
   of which optional extras are installed.

The architecture (`pyproject.toml` extras + skipif fallback in
`sparse_solve.py` + auto dispatch in `damage_solver.py`) is correct.
What is missing per held PR is the documented working install path, not
new code.

## Artifact Hygiene

The repository should contain source, curated documentation, small
hand-digitised reference data, and reproducible scripts. Raw run payloads
belong outside git unless they have been explicitly promoted into a
curated result table or figure source.

Keep these off GitHub:

- `papers/paper2/results/**` raw checkpoints, truth bundles, generated meshes,
  GIF/MP4/PNG panels, and run JSON/CSV dumps.
- `docs/qs_hpc_results/**` raw HPC mirrors. Summarise these in Markdown
  tables or issue comments instead.
- `examples/**/hpc_results/**` raw benchmark mirrors.
- Local Molinari meeting packs under `docs/molinari_meeting/**`, including
  source scripts, storyboards, talk tracks, and generated media.

Allowed exceptions:

- Small hand-digitised reference curves under
  `examples/**/reference_solutions/`.
- Source scripts under `papers/paper2/scripts/`, manuscript sources under
  `papers/paper2/sections/`, and curated figure-generation code.
- A compact Markdown/TSV summary that records job ids, configs, PASS/FAIL,
  and paths to the HPC result directory.

If a generated artifact is needed for a paper figure, promote it through a
script that can recreate the figure from documented inputs rather than
committing the raw run directory.

The CI hygiene gate is:

```bash
python scripts/check_artifact_hygiene.py
python scripts/check_package_payload.py
```

`check_artifact_hygiene.py` rejects private/local paths such as
`docs/molinari_meeting/**` and `reference_codes/**`, blocks tracked heavy run
formats (`.h5`, `.vtu`, videos, COMSOL `.mph`), and size-gates curated binary
assets such as `.gif`, `.npz`, and `.pt`.

`check_package_payload.py` protects the wheel/source-install surface. It
requires explicit package data (`include-package-data = false`) and rejects
future package-data patterns that would ship docs, papers, private meeting
material, reference clones, raw result trees, or heavy generated payloads.

## Decision matrix for new install paths

When a contributor proposes a new optional backend or distribution path:

1. Does it slow or complicate the **default** `pip install`? If yes,
   reject. That is the killer-app UX.
2. Is it gated behind a clean `[extra]` with a runtime skipif fallback?
   If not, fix that first; do not merge.
3. Is there a documented working install on at least one platform,
   tied to a CI job or a SLURM log? If not, hold until validated.
4. Where on the 5-tier ladder does it sit? Tier 1 = high bar (must
   not regress the one-liner). Higher tiers = optional, easier to
   merge once the gates above are met.

## Today's snapshot (2026-05-09)

- Tier 1 verified: pip install path, autograd-compatible tensor kernels, and CI
  matrix on Mac + Ubuntu x Py3.10 / 3.11 / 3.12 (#457).
- Tier 2 shipped: cibuildwheel matrix on linux-x86_64 + macos-arm64 (#462).
- Tier 3 shipped: conda-forge recipe + submission notes (#461).
- HPC native PETSc module path documented (#460).
- Held PRs awaiting verified-install gate: #378, #379, #382, #458.
- Foundational fixes this session: packaging guard (#455), pytest
  collection (#454), skipif gracefulness (#459), install-promise CI
  (#457).

## How to apply

When a future contributor or agent asks "should we add X distribution
path?" or "is this optional-backend PR ready?" — open this file first.
The killer-app principle, the 5-tier ladder, and the held-PR merge gate
decide most cases without further debate.
