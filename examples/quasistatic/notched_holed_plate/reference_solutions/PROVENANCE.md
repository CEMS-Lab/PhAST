# Reference data provenance — Notched holed plate

## Source

`models.geomech.holed_plate_fracture.pdf` (also `models.nsm.holed_plate_fracture.pdf`)
— official COMSOL Multiphysics 6.4 Application Library report for the
Geomechanics Module example **"Brittle Fracture of a Holed Plate"**. The
companion `holed_plate_fracture.mph` model file is also kept in this
directory but is not directly consumed by the comparison script.

The original phase-field formulation reproduced by the COMSOL example is
Ambati, Gerasimov & De Lorenzis (2015), *A review on phase-field models
of brittle fracture and a new fast hybrid formulation*, Comput. Mech.
**55**, 383–405.

## Contents

| File | What | How extracted |
|---|---|---|
| `models.geomech.holed_plate_fracture.pdf` | COMSOL Application Library report (text + figures) | Downloaded by the user from the COMSOL Application Library on 2026-05-04. |
| `holed_plate_fracture.mph` | Original COMSOL model file | Same source; kept for cross-reference, not parsed. |
| `comsol_load_displacement.csv` | Reference reaction-force vs. **total elongation** (= 2 x per-pin = top_pin_y - bottom_pin_y) | Two **quantitative anchor points** taken verbatim from the report text on page 3-4 (first peak 0.63 kN at 0.33 mm; second peak 0.15 kN at 1.7 mm). The remaining points are read off Figure 2 of the report (green snapshot markers + visually-traced curve) at the displacements that COMSOL itself plotted. |

**Displacement-axis correction (issue #223, 2026-05-06).** The
column was originally labelled "per-pin displacement [mm]" — that
label was wrong. The canonical
`models.geomech.holed_plate_fracture.pdf` page 3-4 plots reaction
force against the **total elongation** of the specimen
(top_pin_y − bottom_pin_y), which equals 2 × per-pin displacement
because the loading is symmetric (`para * 2 mm` per-pin in COMSOL,
matched by `--u_max` in `run.py:282`). The simulator's
`results.csv` reports per-pin displacement; `compare.py` now
divides this reference column by 2 inside `load_reference()` so
the comparison is performed in a single (per-pin) unit system.

## Acceptance criterion (issue #119)

| Metric | Reference value (PDF) | Tolerance |
|---|---|---|
| First peak load           | 0.63 kN at 0.33 mm total (= 0.165 mm per-pin) | ±10 % on load, ±15 % on displacement |
| Second peak load          | 0.15 kN at 1.7 mm total (= 0.85 mm per-pin)   | ±20 % on load |
| Final crack-path morphology | Curved crack from notch tip to large hole (Fig. 4) | qualitative |

The intermediate curve points are *not* used for a strict L2 acceptance —
they are only present so `compare.py` can plot a meaningful side-by-side
overlay. Accepting/rejecting the run is done on the three numbers above.
