# Paper reproduction

The [paper reproduction directory](https://github.com/CEMS-Lab/PhAST/tree/main/reproduction/paper1)
contains the manuscript and supplement catalogue and its commands. The
catalogue follows the paper's order and uses stable figure and table labels.
It also maps numerical statements in the text to their supporting studies.

The existing example folders and YAML configurations remain the starting
point for using PhAST. Each study records the relationship between its exact
paper configuration and the related public example. Large numerical fields,
meshes and historical execution sources belong in the companion archive.
Its publication details will be added after the archive is complete.

From the repository root, list the paper entries and check their mappings.

```bash
python reproduction/paper1/reproduce.py list
python reproduction/paper1/reproduce.py check
```

With the retained data available locally, check its file hashes and regenerate
the selected quasi-static holed-plate figure.

```bash
python reproduction/paper1/reproduce.py check --data /path/to/phast-paper-reproduction
python reproduction/paper1/reproduce.py replot q2 --data /path/to/phast-paper-reproduction --output runs/paper-q2
```

The `forward` group covers standing-wave and damage-update-interval plots. The
`inverse` group covers the selected scalar, cross-mesh, joint, finite-difference
and memory plots. Install the separate plotting requirements described in the
reproduction directory before running these commands. Output directories must
be fresh and separate from the retained data.

The commands check the archive manifest and document-edition hashes against
the catalogue before using the files. Use a trusted data archive because the
plotting adapters load its preserved experiment code.

A checksum check establishes input integrity. Replotting checks the recorded
data and figure-generation route. Complete simulation reproduction also
requires the executed solver version, all numerical inputs, the environment
and a verified run command. These statuses are recorded separately for each
study, including inputs still being collected.

This separation of the solver and experiment dependencies follows
[Firedrake's reproduction guidance](https://www.firedrakeproject.org/zenodo.html).
The problem-based ordering also follows the approach used in the
[DOLFINx demonstrations](https://docs.fenicsproject.org/dolfinx/main/python/demos.html).
