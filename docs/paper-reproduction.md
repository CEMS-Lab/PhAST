# Paper examples and recorded results

PhAST 0.18.0 contains the solver, worked examples and selected inversion
recipes for the [CMAME article](https://doi.org/10.1016/j.cma.2026.119547).
The 3D examples are a later extension, separate from the paper's 2D results.

## What this software ZIP contains

| Material | Location | How to use it |
| --- | --- | --- |
| Dynamic fracture inputs and stored outputs | `examples/dynamic/` | Run the example's `config.yaml`; inspect its README and plots |
| Quasi-static SENT and holed plate | `examples/quasistatic/` | Run the YAML input; inspect the stored load-displacement curves |
| Selected cross-mesh toughness recovery | [examples/inverse](https://github.com/CEMS-Lab/PhAST/tree/main/examples/inverse) | Original mesh h=0.125 mm, inverse mesh h=0.1875 mm; repeat AD or inspect the recorded AD/Brent values |
| Scalar glass and alumina inversions | [reproduction/cmame_2026](https://github.com/CEMS-Lab/PhAST/tree/main/reproduction/cmame_2026) | Run the isolated original driver or replot its recorded history |
| Three periodic voxel demonstrations | [3D examples](https://github.com/CEMS-Lab/PhAST/tree/main/examples/three_dimensional) | Run the Python setup; inspect the damage animations and stored stress-strain curves |

Most 2D examples use `config.yaml` and optionally `run_fluent.py`.
The 3D examples use `run.py` to define the geometry, material phases,
loading and voxel mesh. A YAML file is not required for those Python setups.
The fibre example also includes its microstructure array.

Public example names predate the paper's benchmark labels. Dynamic SENT is
paper B1 and repository `B3_dynamic_sent`; branching is paper B3 and
`B7_dynamic_crack_branching_comsol`; the `B6_perforated_*` examples relate
to paper B4. Similar names do not establish identical solver settings.

## Exact paper plots

The larger collection of recorded arrays, comparison-code sources and
historical settings is in the separate `PhAST-paper-reproduction.zip`.
For those materials, extract that archive and follow
its `REPRODUCING.md`, for example:

```bash
python -m pip install -r environments/replot-requirements.txt
python analysis/run_analysis.py --archive . --output ../phast-analysis
```

Run those commands from the extracted `phast-paper-reproduction` directory,
not from this software checkout. Some figures use saved images where the
original fields are unavailable. The archive identifies those limits.
Its large B4 energy-field file is only needed for the analyses that use it.

This release provides code and examples that users can extend. It does not
promise that the software ZIP alone regenerates every paper or supplementary
figure. Stored-result plotting, a fresh simulation, and comparison with the
published result are separate operations.
