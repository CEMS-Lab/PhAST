# Changelog

## Unreleased

### Result file

- Add the result file `result.vtkhdf`: one HDF5 file per run in the VTKHDF
  layout, so that ParaView opens it directly, with float64 state, the load
  history, the mesh, the materials, the loading and the resolved input.
- Add the writer, the validator, the recovery and archive commands
  (`python -m phast result`), the standard figures (`python -m phast figures`)
  and the layout specification and guide under `docs/user_guide/`.
- The YAML runner writes the file for two-dimensional T3 and Q4 runs by default
  (`output.result: true`); `output.result: false` writes no file and a mapping
  sets the options. `phast.load_result` reads the file and the existing run
  directories. A run directory that holds both `result.vtkhdf` and
  `training_data.h5` is read from the result file, where `step` counts stored
  steps and the field names are those of the file. The CSV files and the
  trajectory store are written as before.
- The tutorial notebooks `notebook_setup`, `problem_setup_walkthrough` and
  `notebook_square_plate_fracture` read the result file through
  `phast.load_result` with stored-step indices, and take the mesh from
  `mesh_arrays()`, instead of step labels from `training_data.h5`.
- Layout 2.1 adds crack tips and crack speed, initial conditions, wave speeds,
  velocity conditions, validation records and meshes that change during a run.
- `ResultWriter.resume` continues an interrupted file from a checkpoint increment
  (the cut, the backup and the record of the resume are described in the layout
  guide); `phast.io.result.interface_connectivity` builds the rows of cohesive
  interface cells in the stored order, the solver's `top` side first.

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
