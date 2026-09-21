# Changelog

## Unreleased

### Quasi-static rigid connectors

- Forward active rotation-free rigid connectors from the staggered
  `quasi_static` path to the existing reduced mechanics solve.
- Add CPU float64 elastic checks for connector kinematics, reaction and moment
  balance, alongside dispatch and non-connector regression tests. Damage
  stopping rules, solver defaults and example configurations are unchanged.

### Paper reproduction

- Add a manuscript and supplement catalogue with separate records for figures,
  tables and numerical statements.
- Add portable retained-data checks and selected figure-generation commands.
  Record data replotting separately from complete simulation reproduction.
- Preserve the existing example folders and configuration schema. Large fields
  and historical execution sources remain in the companion data archive.

### Experimental tetrahedral geometry

- Add validated TET4 connectivity, orientation repair, volumes and scalar
  shape-function gradients for geometry experiments.
- Add a small structured block generator and scale-aware plane selection.
- Cover orientation, affine fields, translated coordinates and small element
  scales with CPU float32 and float64 regressions.

This module prepares geometry. Existing two-dimensional solver routes are
unchanged and three-dimensional mechanics and fracture remain separate work.

### Python workflow execution

- Run the supported linear-plate Python `ProblemSpec` through the existing
  YAML runner and example files.
- Validate the supported declarations before execution and reject inputs
  that the example runner cannot represent.
- Preserve geometry units as metadata in `Problem.geometry`. The linear-plate
  companion declares metres consistently with its SI inputs and results.
- Add Python/YAML parity tests for resolved inputs, response values and field
  images, together with validation-only and unsupported-input regressions.
- Preserve default and relative Python result directories after execution.
- Validate requests against the fixed visual bundle and require `plots=True`.

The geometry keyword records units and leaves numerical values unchanged.
The source-less execution bridge covers the documented single-material,
single-step linear plate with its built-in boundary conditions.
