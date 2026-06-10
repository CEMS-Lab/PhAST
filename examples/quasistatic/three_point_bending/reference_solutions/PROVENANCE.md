# Three-Point Bending Reference Data Provenance

`miehe_tpb_load_displacement.csv` -- load-displacement curve for the
three-point bending test of Miehe, Welschinger and Hofacker (2010),
*IJNME* **83**(10), reproduced via the PhaseFieldX example
**1714_Three_point_bending**.

- Columns: applied vertical displacement at the load point `|u_y|`
  [mm], reaction force at the load point [kN]. Comma separated, no
  header.
- Source mirror: repository root `reference_solutions/miehe_three_point.csv`
  and `reference_codes/phasefieldx-main/examples/PhaseFieldFracture/reference_solutions/miehe_three_point.csv`.
- Material: E=20.8 GPa, nu=0.3, Gc=0.5 N/mm, l0=0.06 mm, AT2,
  spectral split. Beam 8 mm x 2 mm with a 0.4 mm V-notch rising from
  the bottom centre. Pin-roller supports at the bottom corners,
  downward displacement at the top centre.

Peak reaction force in this curve is approximately 0.0347 kN at
u ~ 0.039 mm.
