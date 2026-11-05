# CMAME 2026 article calculations

This folder contains the original scalar toughness-inversion drivers for
glass and alumina, their recorded parameter and loss histories, and the
solver source needed by each driver. These older source copies run in
isolation from the installed PhAST 0.18.0 package.

The article is [PhAST: A differentiable, matrix-free finite-element solver
for dynamic phase-field fracture: Formulation, benchmarks, and inverse
analysis](https://doi.org/10.1016/j.cma.2026.119547).
The added 3D examples extend the software; they are not calculations reported
in that article.

## Scalar inversion

Run these commands from the repository root. They launch a new calculation
and can take substantial time:

```bash
python reproduction/cmame_2026/scalar/glass/run.py --output-dir runs/paper_glass
python reproduction/cmame_2026/scalar/alumina/run.py --output-dir runs/paper_alumina
```

Use a new output directory. Each folder includes `requirements.txt`,
`historical_settings.json`, `invocation.json`, `original_driver.py` and
`recorded_result.json`. The dependencies are minimum requirements, not a
complete reconstruction of the original runtime environment.

To redraw the stored convergence curves without a simulation:

```bash
python reproduction/cmame_2026/plot_scalars.py --output-dir runs/paper_scalar_figures
```

For glass, recorded losses precede each optimiser update; recorded parameter
values follow it. For alumina, both are evaluated after the update. The
plotting routine keeps those different iteration indices.

The original full scalar fields and meshes are not included. New meshes or
different dependency versions can change a repeated calculation. The
recorded histories remain unchanged, and the release does not claim that
every paper figure is reproduced by this ZIP alone.

## Cross-mesh inversion

The [worked cross-mesh example](../../examples/inverse/README.md) includes
the YAML input, forward model, AD optimisation and recorded AD/Brent values
for the original 0.125 mm mesh and the 0.1875 mm inverse mesh.
It generates meshes from the documented geometry. A new solve uses the
current public solver; exact historical observations and meshes belong to
the paper-results archive.

## Other paper figures

The 2D examples provide YAML inputs, loading and geometry descriptions,
plots and stored numerical histories for related forward problems.
See [paper reproduction](../../docs/paper-reproduction.md) for the distinction
between those examples and exact paper calculations.

The existing `PhAST-paper-reproduction.zip` contains the larger study
records, including the selected joint recovery, six-start study, cross-code
comparisons and supplementary analyses. Its own `REPRODUCING.md` and
`analysis/run_analysis.py` describe the supported figure-generation route.
That archive is not embedded in this compact software ZIP.

## Source scope

The glass and alumina drivers and recorded histories are copied from the
prepared paper archive. Unused model-injection methods were removed from
the embedded source copies, and explanatory comments were cleaned.
The original source identities and exported-file hashes are recorded in
each `historical_manifest.json`. Numerical routines used by the scalar
drivers were not changed. These exported recipes were not rerun during
release preparation.

## Licence

The solver sources, drivers and plotting code use the repository's MIT
licence. The two `recorded_result.json` histories retain the paper archive's
CC BY 4.0 licence. Please attribute those records to the authors and article
identified above.
