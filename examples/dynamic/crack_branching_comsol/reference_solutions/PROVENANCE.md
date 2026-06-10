# Reference data provenance -- Dynamic crack branching (COMSOL)

## Source

`models.geomech.dynamic_crack_branching.pdf` and
`models.nsm.dynamic_crack_branching.pdf` -- official COMSOL Multiphysics
6.4 Application Library reports (Geomechanics Module and Nonlinear
Structural Materials Module, respectively) for the example
**"Dynamic Crack Branching"**. The companion
`dynamic_crack_branching.mph` model file is also kept in this
directory but is not directly consumed by the comparison script.

## Contents

| File | What | How extracted |
|---|---|---|
| `models.geomech.dynamic_crack_branching.pdf` | COMSOL Geomechanics report | Downloaded by the user from the COMSOL Application Library on 2026-05-04. |
| `models.nsm.dynamic_crack_branching.pdf` | COMSOL NSM Module variant | Same source. |
| `dynamic_crack_branching.mph` | Original COMSOL model file | Same source. |
| `comsol_branching_times.txt` | Reference event times (initiation, branching, full Y) | Read off the figures + report text. |
| `comsol_energy_curve.csv` | Three reference points on the elastic-energy time series | Reported in the PDF's Fig 4 caption / discussion. |

## Reference values (per task spec)

| Event | Time | Source |
|---|---|---|
| Initiation        | ~ 10 us | COMSOL PDF (visual, Fig 3 first frame with crack growth) |
| Branching onset   | ~ 33 us | COMSOL PDF text (Y bifurcation visible) |
| Full Y morphology | by 45 / 75 us | COMSOL PDF Fig 3 last two frames |

| Quantity | Value | Source |
|---|---|---|
| Elastic-energy peak 1 | ~ 0.13 J at thickness 1 m | COMSOL PDF Fig 4 |
| Elastic-energy peak 2 | ~ 0.14 J at thickness 1 m | COMSOL PDF Fig 4 |
| Fracture energy at 75 us | ~ 0.5 J at thickness 1 m | COMSOL PDF Fig 4 |

**Thickness convention**: COMSOL runs the model with an out-of-plane
thickness of 1 m. Our `phast` solves a per-mm-thickness 2D
problem (mesh in mm; energies in mJ when forces are N and lengths mm).
To compare with the COMSOL energies, multiply our values by 1000
(factor of `1 m / 1 mm`). `compare.py` does this conversion.

## Acceptance criterion (issue #135 + this task)

| Metric | Reference | Tolerance |
|---|---|---|
| Branching onset time | ~ 33 us | +/-20 % |
| Full-Y morphology by | 75 us | qualitative |
| Elastic-energy peak (per-mm * 1000) | 0.13 - 0.14 J | +/-25 % |

## Naming note

The COMSOL example calls the material PMMA, but the parameter values
(E = 32 GPa, nu = 0.2, rho = 2450 kg/m^3, Gc = 3 J/m^2) match
Borden 2012's soda-lime glass exactly and are far stiffer than typical
PMMA. Our YAML config `B7_dynamic_crack_branching_comsol.yaml`
therefore starts from the `glass_borden` preset (which encodes those
numbers) and overrides `l0 = 0.5` and `pf_model = AT1` to match the
COMSOL setup. See the README in this directory for discussion.
