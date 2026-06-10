# SENS Reference Data Provenance

`miehe_sens_load_displacement.csv` -- load-displacement curve for the
single-edge-notch shear test of Miehe, Welschinger and Hofacker
(2010), *IJNME* **83**(10), reproduced via the PhaseFieldX example
**1712_Single_Edge_Notched_Shear_Test**.

- Columns: applied horizontal shear displacement `u_x` [mm], reaction
  force at the bottom edge in the loading direction [kN]. Whitespace
  separated, no header.
- Source mirror: repository root `reference_solutions/miehe_solution_shear.csv`
  and `reference_codes/phasefieldx-main/examples/PhaseFieldFracture/reference_solutions/miehe_solution_shear.csv`.
- Material: E=210 GPa, nu=0.3, Gc=2.7 N/mm, l0=0.06 mm, AT2,
  spectral (Miehe) split. The shear crack curves diagonally from the
  notch tip toward the bottom-right corner.

The peak reaction force in this curve is approximately 0.531 kN at
u ~ 0.0094 mm.

## 2026-05-17 audit caveat

The local executable driver follows the PhaseFieldX 1712 code path with
`l0=0.06 mm`, but the bundled PhaseFieldX output inspected under
`reference_codes/phasefieldx-main` peaks closer to 0.495 kN at
u ~ 0.0087 mm. Recent phast strict SENS runs also peak near
that lower value. This means the checked-in CSV and its label are not
currently sufficient provenance for a production acceptance claim.

Before closing SENS validation, choose and document one target:

1. regenerate the CSV from the bundled PhaseFieldX 1712 output for
   `l0=0.06 mm`; or
2. retarget the driver to the original Miehe-paper length scales
   `l=0.015 mm` / `0.0075 mm` and compare to the corresponding published
   curve and crack-stage figures.
