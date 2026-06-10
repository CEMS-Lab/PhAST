# Provenance: Galvis 2026 cross-validation

## Known

- **Citation**: Galvis, A.F. et al. (2026). *Phase-field modelling of
  quasi-static and dynamic brittle fracture: A FreeFEM++ implementation*.
  Eng. Fract. Mech. DOI: 10.1016/j.engfracmech.2026.111846.
- **Companion code**: <https://github.com/GalvisA1087/Phase-field-for-brittle-fracture>
  (FreeFEM++ source for all 5 published benchmarks).
- **Quasi-static benchmarks (Sections 4.1-4.4)**: Miehe SENT tension,
  Miehe shear, L-shaped panel, notched-with-holes plate.
- **Dynamic benchmark (Section 4.5)**: out of scope for \#109; tracked
  separately under the dynamic-validation thread.

## TBD

- Digitised load-displacement curves for each of Galvis Figs 4.1-4.4.
- Initiation displacement, peak load, and kink-angle numerical values
  (the acceptance tolerances in \#109 are stated relative to these).
- Whether Galvis uses identical material parameters to our existing
  `quasistatic_*` configs, or whether material presets need a new
  `galvis_2026_*` variant.

## How this scaffold was produced

- Issue \#109 body provided the citation, DOI, and benchmark mapping.
- No paper PDF was available at scaffold time; no digitisation was
  performed.
- `reference_solutions/galvis_2026_PLACEHOLDER.csv` is a single-row
  schema stub so downstream tooling can be wired without committing
  fake numerical values.
