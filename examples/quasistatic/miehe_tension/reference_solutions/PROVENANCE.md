# SENT Reference Data Provenance

`miehe_sent_load_displacement.csv` — load–displacement curve for the
single-edge-notch tension test of Miehe, Welschinger and Hofacker
(2010), *IJNME* **83**(10), reproduced via the PhaseFieldX example
**1711_Single_Edge_Notched_Tension_Test**.

- Columns: applied vertical displacement `u_y` [mm], reaction force at
  the bottom edge in the loading direction [kN]. Whitespace separated,
  no header.
- Source mirror: repository root `reference_solutions/miehe_solution.csv`
  and `reference_codes/phasefieldx-main/examples/PhaseFieldFracture/reference_solutions/miehe_solution.csv`.
- Material: E=210 GPa, nu=0.3, Gc=2.7 N/mm, l0=0.015 mm, AT2,
  isotropic split.

The peak reaction force in this curve is approximately 0.701 kN at
u ≈ 0.0055 mm.
