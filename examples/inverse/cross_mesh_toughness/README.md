# Cross-mesh Toughness Recovery

Paper calculation: the selected cross-mesh toughness recovery ($h = 0.1875$ mm inverse mesh).

This example estimates the fracture toughness $G_c$ from synthetic
displacement readings generated on a finer mesh.
It generates a 0.125 mm original mesh and a 0.1875 mm recovery mesh from the
documented geometry.

## Recorded Results

Automatic differentiation (AD) and Brent's scalar search both use the
0.1875 mm recovery mesh and observations from the 0.125 mm original mesh.
[recorded_results.json](recorded_results.json) keeps the results at full
precision.

| Method | Recovered $G_c$ (N/mm) | Toughness error | Fitting-point displacement error | Check-point displacement error |
| --- | --- | --- | --- | --- |
| AD | 0.00293339 | 2.22 % | 6.01 % | 0.99 % |
| Brent | 0.00295045 | 1.65 % | 5.94 % | 1.21 % |

AD used 20 objective evaluations.
The displacement errors are relative Euclidean norms at the specified points
and times, not whole-field errors.
The check points do not enter the fitting loss.

Verify the recorded results without solving:

```bash
python examples/inverse/cross_mesh_toughness/run.py --verify-only
```

`recorded_results.json` also records the checksums of the original sources and
forward runs.
The original drivers, meshes, observation arrays and fields are in the
`inverse/` folder of the reproduction ZIP
(`PhAST-paper-reproduction.zip`) in the same Zenodo record.

## Problem And Recovery Recipe

The two-dimensional plate is 40 mm x 40 mm with a 20 mm edge notch at
mid-height.
It uses plane-strain AT2 with the Miehe spectral split.

| Quantity | Value |
| --- | --- |
| Young's modulus $E$ | 32,000 MPa |
| Poisson's ratio $\nu$ | 0.2 |
| Density $\rho$ | $2.45\times10^{-9}$ tonne/mm$^3$ |
| Original toughness $G_c$ | 0.003 N/mm |
| Fixed regularisation length $\ell_0$ | 0.5 mm |
| Residual stiffness | $10^{-6}$ |
| Original and recovery mesh sizes | 0.125 mm and 0.1875 mm |
| Mesh ratios $h/\ell_0$ | 0.25 and 0.375 |
| Initial toughness | 0.006 N/mm |

The top and bottom displacements ramp to $+0.002$ and $-0.002$ mm over
20 µs and then remain fixed until 30 µs.
Horizontal motion is fixed on both vertical edges.

The time grid has 3,179 steps.
Mesh-specific linear interpolation samples 351 fixed positions at
10, 15, 18, 20, 22, 25, 28 and 30 µs.
There are 263 fitting points and 88 check points.
The loss is the mean squared error (MSE) of the fitting displacements divided
by $(0.002\ \mathrm{mm})^2$.

The recipe uses CPU float64, explicit central-difference mechanics and damage
updates every step.
The damage solve uses Jacobi-preconditioned conjugate gradients (CG) with
tolerance $10^{-7}$ and a hard history maximum.
Damage is then projected to $[d_n, 1]$, where $d_n$ is the damage of the
previous step.
L-BFGS-B optimises the logarithm of the toughness, with limits of
20 iterations and 20 objective evaluations.
The selected result has the minimum fitting loss; the known original
toughness is not used for the selection.
A centred finite-difference check of the initial gradient and checks of the
forward solves and of the recovery decide whether a run is accepted.

## Running The Recovery

Check the input, then run into a new directory:

```bash
python examples/inverse/cross_mesh_toughness/run.py --validate-only
python examples/inverse/cross_mesh_toughness/run.py --profile full --output-dir runs/cross_mesh_full
```

These commands use the script interface, not `phast run`.
The script accepts only the full profile and this mesh pair, also when another
configuration file is given with `--config`.
Validation does not run the recovery.
The full inverse solve can need substantial time and memory.

A completed run writes loss and parameter histories, field CSV files,
derivative checks, figures and source and version records.
Plot its results with:

```bash
python examples/inverse/cross_mesh_toughness/plot.py --output-dir runs/cross_mesh_full
python tools/replot_inverse_setup.py --output-dir runs/cross_mesh_full
```

These commands plot the new run, not the fields of the paper calculation.
The script repeats the AD recovery with the current solver; its result can
differ slightly from the recorded AD values.
The Brent values come from the paper calculation only.

## Scope

The example uses synthetic observations and one initial value on one mesh
pair.
It is not a mesh-convergence or experimental calibration study.
