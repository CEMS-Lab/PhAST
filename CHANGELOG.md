# Changelog

## Unreleased

### Fracture mechanics

- Fix: `compute_sif` returned stress intensity factors that were wrong by a fixed
  factor of the elastic modulus: `K_I` was 3/32 E' times too large and `K_II`
  1/32 E' times too large (E' is the plane modulus). The rotation term of the
  auxiliary Williams field had the units of a stress and was added to strains.
  The auxiliary displacement gradient is now the closed-form derivative of the
  Williams displacement, the auxiliary fields are rotated into the crack frame
  (so a crack at an angle to the x axis works), and the plane-strain compliance
  is used in plane strain. On a slit mesh under a Williams field, `K_I` and
  `K_II` now reproduce the applied value to 0.02 % or better for domains of 2 to
  8 regularisation lengths, in plane strain and plane stress, in modes I and II
  and in mixed mode; a mesh rotated by 30, -40, 90 and 160 degrees gives the same
  values to 1e-8.
- Change: the default `d_process_threshold` of `compute_j_integral` and
  `compute_sif` is now 0.9 (it was 0.01, fixed inside `compute_sif`). A cell with
  a mean damage of 0.9 keeps 1 % of its stiffness, so the cells above the cutoff
  carry no load; the former value also removed the damage halo of a regularised
  crack face, which lies inside the weight ramp, and J then depended on the domain
  radius (9.6 % between the smallest and the largest domain on a notched strip,
  0.05 % with the new default). `compute_sif` takes the cutoff as the argument
  `d_process_threshold`.
- Who is affected: callers of `compute_sif`, whose results change by the factors
  above, and callers of `compute_j_integral` on a field with damage between 0.01
  and 0.9 that rely on the default cutoff; pass `d_process_threshold=0.01` to
  restore the former behaviour of `compute_j_integral`. The runner does not call
  either function.
- For a crack that runs towards -x, pass `alpha` (the direction from the crack
  mouth to the tip): `find_crack_tip` returns an angle in [-90, 90] degrees
  without an orientation.

### Adaptive time stepping

- Change: with `solver.adaptive_dt: true`, the YAML runner (explicit dynamics) now
  derives the time of each step from the accumulated time, the sum of the steps
  already taken, instead of `step * dt`. The load factor, the time-dependent
  tractions (`ramp_type` of a traction condition), the time columns of
  `energy.csv`, `solver_telemetry.csv` and `results.csv`, the snapshot time and the
  time handed to the result file follow that time. The `dt` column of
  `solver_telemetry.csv` is the step used by the row, not the next one.
- Who is affected: only runs with `adaptive_dt: true`. Before, the load factor and
  the tractions followed the nominal time `step * dt`, so the stored `load_factor`
  differed from the `history/time` of the same file once the step size changed; a
  rerun of such a deck gives a different load history and different results.
  Runs with a constant time step, and all quasi-static runs, are unchanged and
  give identical histories.
- `compute_load_factor` takes an optional time `t`; without it the value is still
  computed from `step * dt`.

### Result file

- Add the result file `result.vtkhdf`: one HDF5 file per run in the VTKHDF
  layout, so that ParaView opens it directly, with float64 state, the load
  history, the mesh, the materials, the loading and the resolved input.
- Add the writer, the validator, the recovery and archive commands
  (`python -m phast result`), the standard figures (`python -m phast figures`)
  and the layout specification and guide under `docs/user_guide/`.
- The YAML runner writes the file for single-material two-dimensional T3 and Q4
  runs by default (`output.result: true`; the multi-material route does not write
  it); `output.result: false` writes no file and a mapping
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
