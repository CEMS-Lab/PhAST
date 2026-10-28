# Result file layout

PhAST stores the result of one run in one file, `result.vtkhdf`. The file is
an HDF5 file in the VTKHDF layout, so ParaView opens it directly, and it
carries the arrays, the load history, the mesh, the materials, the loading
and the resolved input that a later analysis needs. This page is the
specification of that layout. [Result files](result_file.md) explains how to
write, read, check and draw them.

The layout has version 2.0.0. Version 2.1.0 adds optional items for forward
runs (section 8); every 2.0 file remains valid. The machine-readable schema
is `src/phast/io/result/schema/phast_result_2.0.0.yaml` (and `2.1.0`). The
writer embeds the schema it used in the file, in `meta/schema_yaml`, and the
validator checks a file against the schema of its own version.

The keywords MUST, SHOULD and MAY have their usual meaning. In this page
`n_steps` is the number of stored steps, `N` the number of nodes and `E` the
number of cells.

## 1. Principles

1. One file per run, float64 state, one writer. The file is a valid VTKHDF 2.x
   file: ParaView 5.13 and later open it without a sidecar file. Groups beside
   `/VTKHDF` hold what VTK has no place for. The VTKHDF specification permits
   such groups and ParaView ignores them.
2. Arrays that ParaView shows are stored once, inside `/VTKHDF`, in its own
   layout. The other groups refer to them by HDF5 soft links that point into
   `/VTKHDF`, never out of it. A stored file contains no virtual datasets: a
   partial copy can turn them into silent zeros, they cannot grow while the
   run appends, and HDF5 single-writer mode does not support them.
3. Values are stored where the solver keeps them. History variables (the
   crack driving history `H`, plastic and hardening variables, cohesive
   damage) are stored at the solver's own location, per quadrature point when
   the solver keeps them per point, because averaging them makes a later
   recomputation inexact. Derived outputs (stress, strain, energy densities)
   are a choice.
4. Every value carries its meaning: location, component order, units, and,
   for derived values, the method that produced them.
5. Views are generated from the file and are never written as second copies:
   the ParaView view is the file itself, and figures are made by a command.
6. Run folders and file names are short and ASCII. Files hold no absolute
   paths.

The unit system is mm, MPa, N and s (`unit_system = "mm-MPa-N-s"`), the same
as the solver. Energies are in mJ, forces in N, stresses in MPa.

Four fixed choices apply to every file.

- The extension is `.vtkhdf`, which ParaView opens directly.
- The file is not compressed while the run appends to it. The archive command
  (`phast result finalize --archive`) rewrites a closed file with gzip level 4,
  shuffle and fletcher32.
- Stress and strain stay in the ParaView view, at every stored step, while the
  estimated size of the file is below 4 GB. Above that size the runner moves
  them to `/derived` (section 3).
- The result file is a record of the run and not a checkpoint. It supports an
  exact recomputation of the derived fields, but it is not a substitute for a
  checkpoint from which a run is restarted.

## 2. Normative layout

```
result.vtkhdf
├─ attrs (root)
│   phast_schema = "phast.result"           phast_schema_version = "2.0.0"
│   dim = 2 | 3                              analysis = plane_strain | plane_stress | 3d
│   history_coupling = fresh | lagged        profile = setup | run
│   run_id, parent_run_id                    (identity; the parent is the run this one continues or repeats)
│   history_update = hard_max | softmax | smooth_max | log_smooth | custom_subgrad
│   history_update_beta                      (smoothing parameter, or absent)
│   kinematics = small | finite              mesh_varies = false (true: layout 2.1, section 8.6)
│   index_basis = increment                  (rows of /history are load increments, not solver steps)
│   unit_system = "mm-MPa-N-s"               status = running | complete | time_limited | failed | recovered
│   created, finished (ISO 8601 with time zone)
├─ VTKHDF/                                   (VTK view and the canonical storage of shown arrays)
│   attrs: Version = [2, 2], Type = UnstructuredGrid | ImageData
│   UnstructuredGrid: NumberOfPoints, NumberOfCells, NumberOfConnectivityIds,
│                     Points (N, 3), Connectivity, Offsets, Types
│   ImageData:        attrs WholeExtent, Origin, Spacing, Direction
│   PointData/<name>  UnstructuredGrid: (n_steps * N[, c]);  ImageData: (n_steps, nz+1, ny+1, nx+1[, c])
│   CellData/<name>   UnstructuredGrid: (n_steps * E[, c]);  ImageData: (n_steps, nz, ny, nx[, c])
│   Steps/            attr NSteps; Values, PartOffsets, NumberOfParts, PointOffsets,
│                     CellOffsets, ConnectivityIdOffsets, PointDataOffsets/<name>,
│                     CellDataOffsets/<name>
├─ mesh/
│   cell_blocks/<block>/   attrs: cell_type (T3, T6, Q4, Q8, Q9, H8, TET4, TET10),
│                          vtk_type, node_order ("vtk" or a permutation dataset)
│      cells               index range of this block inside /VTKHDF cells
│      quadrature/         points (n_qp, ref_dim), weights (n_qp); attrs rule, point_order
│   grid/                  ImageData runs: shape (nx, ny, nz), spacing, origin; attr index_order = "x fastest"
│   node_sets/<name>       int64 node indices
│   face_sets/<name>       (n, 2) int64 pairs (cell, local face)
│   graph/edge_index       (2, n_edges) int64, both directions, from the cell edges of every cell type
├─ materials/
│   phase_id               soft link to /VTKHDF/CellData/phase_id
│   phases/<id>/           attrs: name, elastic_model (isotropic | orthotropic | transversely_isotropic | stiffness),
│                          constants (named attrs, or a "stiffness" dataset 6x6), fracture_law
│                          (AT1 | AT2 | PFCZM | none), Gc, strength, ell, and the units of each
│   orientation            optional soft link to /VTKHDF/CellData/orientation (E, 9), local axes as rows
├─ interfaces/<name>/      cohesive blocks: connectivity, quadrature, law, parameters, and their fields
├─ loading/
│   bc/<k>/                attrs: set, kind (displacement | traction | symmetry | periodic), components,
│                          value = constant or the name of a column in /history
│   periodic/              attrs: axes, controlled, free, fixed_zero
├─ history/                one row per converged increment, whether or not fields were stored for it;
│                          every column carries a units attribute
│   increment, time (accumulated, never step*dt), dt, load_parameter, load_factor
│   step_index             row in /VTKHDF/Steps, or -1 when no fields were stored for this increment
│   damage_solved          1 when the damage equation was solved in this increment
│   cutbacks, stagger_iterations, mechanics_iterations, residual
│   reaction/<set>         (n_inc, 3), signed; attrs components, units, sign_convention
│   macro_stress/<c>, macro_strain/<c>, mean_strain/<c>
│   energy/<term>          open list, each with attr method: external_work, elastic, fracture, kinetic,
│                          plastic_work, cohesive_dissipation, total, balance_residual, balance_relative
├─ state/                  soft links to the arrays in /VTKHDF: displacement, damage, H, velocity,
│                          acceleration, damage_lower_bound (d_prev), external_force (f_ext)
├─ quadrature/<block>/<name>   raw per-point arrays, (n_steps, E_block, n_qp[, c]); only when n_qp > 1
│                               and the field is not already shown per point in /VTKHDF
├─ derived/<name>/         optional arrays on another cadence than the stored steps: values
│                          (n_stored, ...) and step_index (one entry per increment, -1 when absent);
│                          never inside /VTKHDF
└─ meta/
    config_resolved_yaml   scalar UTF-8 string dataset, attr sha256
    schema_yaml            the schema this file was written against
    attrs: software, software_version (git revision and a dirty flag), dependencies (torch, h5py, HDF5),
           machine, slurm_job_id, command_line, solver_dtype, tolerances
```

### 2.1 Field attributes

Every field dataset has these attributes.

| Attribute | Values |
|---|---|
| `location` | `node`, `cell`, `quadrature_point`, `interface_point` |
| `components` | comma-separated names in stored order, for example `x,y,z` or `xx,yy,zz,yz,xz,xy` |
| `units` | a unit string; `"1"` for dimensionless |
| `kind` | `state` (needed to continue the solution), `history` (irreversibility or internal variable), `derived` |
| `measure` | stress and strain under `kinematics = finite`: `cauchy`, `pk1`, `pk2`, `green_lagrange`, `log` |
| `derived_method` | derived fields only: `exact` (one-point cells), `qp_mean_volume_weighted`, `qp_mean_unweighted`, `recomputed` |
| `shear` | strain-like tensors: `engineering` |

### 2.2 Conventions

- Profiles: a `setup` file is a `run` file before its first increment.
  `/VTKHDF/Steps` exists with `NSteps = 0` and empty step datasets, so the
  same file grows into a run by appending. ParaView shows its mesh and static
  arrays.
- Categorical columns are stored as int8 codes with an attribute `codes`, a
  JSON object that maps each code to its name, so that every reader sees
  numeric arrays.
- Write order and interruption: the rows of a new step are appended first,
  then the `Steps` offsets and `Values`, then `NSteps`, then the `history/`
  rows, and then the file is flushed. A file that is killed at any point of an
  increment opens in ParaView with the previous complete steps.
- Points are stored with 3 coordinates; 2D runs have z = 0.
- Vectors in `/VTKHDF` have 3 components and 2D runs store z = 0. ParaView
  loads a 2-component vector but warps it incorrectly and gives no warning.
  `Result.field()` returns arrays in the native dimension.
- Symmetric tensors use one order everywhere: in 3D `xx, yy, zz, yz, xz, xy`;
  in 2D `xx, yy, xy`, with `zz` stored as a separate field (`stress_zz`) for
  plane strain. Strain shear is the engineering shear (gamma = 2 epsilon).
  A solver that uses another order converts when it writes. ParaView reads
  these arrays as plain multi-component arrays with no tensor role and no
  component names, so VTK tensor filters must not be applied to them
  directly.
- Quadrature points follow the `point_order` attribute, for example
  `eta outer, xi inner, minus then plus` for Q4.
- The time axis inside `/VTKHDF` is shared by every array in it. An array is
  inside `/VTKHDF` only if it exists at every stored step. Repeated offsets
  are used only for static data (mesh, `phase_id`, `orientation`).
- Static arrays in ImageData have a leading time axis of length 1 and zero
  offsets, because ParaView requires the time axis. In a `setup` file
  (`NSteps = 0`) they have no time axis: ParaView reads a file with fewer than
  two steps as data without steps and refuses arrays that carry one. For the
  same reason ParaView 5.13 shows no arrays of an ImageData run with exactly
  one stored step. Such a file is valid and `phast.load_result` reads it, and
  the validator warns about it.
- Field names contain no `/` or `.`.

### 2.3 Where each value is stored

| Cell type | Points per cell | Stored |
|---|---|---|
| T3, TET4 | 1 | per cell in `/VTKHDF/CellData`; no `quadrature/` group needed |
| Q4 | 4 | history per point: `/VTKHDF/CellData/H` with 4 components, `components = qp0,qp1,qp2,qp3` |
| H8 | 8 | history per point: `/VTKHDF/CellData/H` with 8 components |
| T6, Q8, Q9, TET10 | 3 to 9 | as Q4 |
| Cohesive | points along the interface | `/interfaces/<name>/` |

Interface cells (`/interfaces/<name>/connectivity`, `cell_type` `COH2` in 2D and `COH4` in 3D) list the nodes of the first side
first and the nodes of the second side after them. The first side is the side the solver calls `top` (`CohesiveElement.nodes_top`: the
side the normal `n = (-t_y, t_x)` points to); a `COH2` row is `top0, top1, bottom1, bottom0`, a `COH4` row the four nodes of the top face
and then the four of the bottom face in the same order. A drawing of an interface uses its first side. The 2D order is tested against the
elements of `insert_cohesive_layer`; the 3D order is the same rule and has no test on solver elements yet.
`phast.io.result.interface_connectivity` builds the 2D rows from `CohesiveElement` records.

History variables per point go inside `/VTKHDF` as multi-component cell
arrays, so that they share the offsets and the truncation rule of the other
arrays and survive a copy of `/VTKHDF` alone. Derived tensors per point (stress
at 8 points with 6 components) go to `/quadrature/`, and their cell means go to
`/VTKHDF` with `derived_method` set.

### 2.4 History variable and damage

The root attribute `history_coupling` states how damage and history are
coupled: `fresh` (the history is updated from the current displacement and the
damage is then solved, within the same increment) or `lagged` (the damage of
increment i uses the history of the previous mechanics solve).

- `H` is always the history after the increment, the state carried forward.
- `H_damage_input` is the history that the damage solve of the increment used.
  It MUST be stored when `history_coupling = lagged` or when the damage is
  solved only every k-th increment. It MAY be omitted for `fresh` coupling
  with a damage solve in every increment, where it equals `H`. Readers never
  infer one from the other.
- `history/damage_solved` marks the increments in which the damage equation
  was solved (explicit dynamics with `damage_every > 1` skips some).

## 3. Output settings

```yaml
output:
  result:
    every: 1                     # stored steps; 1 = every increment
    fields: [displacement, damage, H, stress, strain]
    derived_location: cell       # cell | quadrature_point
    derived_every: 1             # 1 = in the ParaView view; N > 1 = in /derived
    size_limit_gb: 4             # above the estimate, derived fields move to /derived
    layout: 2.0                  # 2.1 adds the items of section 8 that the runner has data for
    crack_tips: false            # true or {threshold: 0.5, direction: x}; implies layout 2.1
```

Each simulation type supplies defaults, the user overrides them, and the
writer resolves the final settings into `meta/config_resolved_yaml`.

| Simulation type | Stored by default | `derived_location` |
|---|---|---|
| brittle phase field, T3 | u, d, H, stress, strain | cell (exact) |
| brittle phase field, Q4, H8 | u, d, H per point, stress, strain | cell, volume-weighted mean |
| plasticity | u, d, plastic variables per point, stress | quadrature_point |
| cohesive interfaces | u, interface opening, cohesive damage per point | interface_point |
| explicit dynamics | u, v, a, d, H | cell |
| periodic voxel cells | u, d, H per point, stress, strain | cell; macro stress and strain in `/history` |

The writer refuses settings that would drop a history variable or a state
variable, and it reports when a per-point request is satisfied exactly by cell
values (one-point cells). The sizes below show why derived fields can move out
of the ParaView view: a 2D mesh of 100,000 T3 cells stores 2.4 MB of primary
state per step and 4.8 MB of stress and strain cell means per step, and a 48
by 48 by 48 voxel cell with H at 8 points stores 10.8 MB and 10.6 MB.

The 2D YAML runner (`phast run`, implemented in
`src/phast/config/result_output.py`) uses the settings as follows.

- `output.result` is `true` by default and writes `<output_dir>/result.vtkhdf`
  with the defaults above. A mapping overrides the keys above, and `false` writes no file. Unknown keys, and `fields` lists that omit `displacement`, `damage`,
  `H` (and, for explicit dynamics, `velocity` and `acceleration`), are
  refused.
- `every`: the quasi-static default is 1 and the explicit-dynamics default is
  `output.h5_every`. Fields are stored at increments `0, every, 2*every, ...`,
  at the last increment and at a time-limit stop. When `every` is not given
  and the stored state alone is estimated above `size_limit_gb`, the runner
  multiplies `every` until the estimate fits and prints the value.
- `derived_every: N > 1` stores the derived fields in `/derived` every N-th
  increment. `derived_location: quadrature_point` (Q4 only) also writes
  `quadrature/Q4/stress` and `quadrature/Q4/strain` and needs
  `derived_every: 1`.
- `stress_zz` is written for plane strain with the splits `isotropic`
  (`nu*(stress_xx + stress_yy)`), `amor`, `star_convex`, `spectral` and
  `spectral_stress`. It is the out-of-plane stress of the same degraded law
  with `eps_zz = 0`, obtained from the 3D Lame parameters and a third
  principal strain of zero. Under `stress_degradation: full` the isotropic
  expression applies. Plane stress has `sigma_zz = 0`, so it is not stored.
- `fields` may also list `damage_lower_bound` (state, node, unit `1`): the
  damage at the start of the increment, which is the lower bound that the
  irreversibility constraint applied in its damage solve. It differs from
  `damage` by the change of the increment, and it cannot be recovered from the
  stored steps when `every > 1`. It is not in the default list.
- `H_damage_input` is written for explicit dynamics with `damage_every > 1`:
  the history used by the most recent damage solve.
- Quasi-static runs use a pseudo-time of one unit per load increment
  (`history/time` and `history/dt` have unit `1`). `load_parameter` is the
  prescribed displacement of the first non-zero `prescribe` condition (mm), or
  the load factor when there is none; `load_factor` is stored separately.
- Explicit runs store the accumulated physical time (s) and the time step that
  each increment used. The load factor of an increment is the value the run
  applied. Under `adaptive_dt` the runner evaluates the load ramp at the
  nominal time `step * dt`, so the stored `load_factor` follows that nominal
  time while `history/time` follows the accumulated step sizes.
- `loading/bc/<k>` (key `<index>_<node set>`) carries `set`, `kind`,
  `components` and `value`. Prescribed displacements and tractions have
  `value = load_factor` and the condition's value as `scale`. In explicit runs
  with a time-dependent traction ramp each traction has `value =
  traction_factor/<k>`, a history column with its ramp factor. `pf_dirichlet`
  entries (prescribed damage) go to `loading/damage_bc/<k>`.
- `history/reaction/<set>` is written for every node set with a fix,
  prescribe, symmetry or rigid-connector condition and for
  `output.reaction_node_set`. In explicit runs it excludes inertia and damping
  forces, which its `method` attribute states.
- For runs with applied tractions, `history/energy/external_work` is the
  accumulated work of the applied traction force with the left-point rule
  `f_n . (u_(n+1) - u_n)`. `balance_residual` is the total energy minus the
  total energy after the first increment minus that work, and
  `balance_relative` divides it by the largest of the three magnitudes.
- The runner does not write the writer's own CSV copies of the history
  (`sidecars=False`): its existing CSV files keep their names and columns.

## 4. Writer rules

1. The writer creates `<name>.partial`. After the final close it reopens the
   file, validates it, renames it atomically and writes `<name>.sha256`.
   Write on local disk. The writer warns when the target folder looks like a
   synchronised folder, because a synchronisation client can lock or partly
   upload an open HDF5 file.
2. Every group, attribute and the resolved configuration are created before
   the first increment.
3. Increments are appended one at a time: field rows first, then the `Steps`
   offsets and `Values`, then `NSteps`, then the `history/` rows, then
   `flush()`. A reader never sees offsets for a step whose rows are missing.
4. File format bounds are `libver=("earliest", "v114")`, never `"latest"`. The name `v114` exists only in an h5py built against HDF5 1.14 (the wheels of h5py 3.13 and later); older builds stop with a `KeyError` when the file is opened for writing.
   With h5py 3.16 the latest bound writes HDF5 2.0 structures that the HDF5
   1.13 to 1.14 library inside ParaView may not read, and a killed writer
   leaves a file that does not open.
5. Chunks: one stored step per chunk along the time axis (`N` or `E` rows for
   UnstructuredGrid, one full grid for ImageData); `history/` columns in
   chunks of 1024 rows, resizable along time.
6. Filters while running: none. The archive command applies gzip level 4,
   shuffle and fletcher32. Scale-offset and other lossy filters are never
   used.
7. Every dataset is created with an explicit dtype. Fields with
   `kind = state` or `kind = history`, mesh coordinates, material values and
   `history/` columns are float64. Fields with `kind = derived` MAY be
   float32, because they can be recomputed from the float64 state. Indices are
   int64 and phase ids uint8. A reader that needs float32 converts while
   reading; no second copy is written.
8. Time-limit stop: the signal handler only sets a flag. The run loop finishes
   the current increment, flushes, sets `status = time_limited` and
   `finished`, closes the file, and the process exits with code 11
   (`TIME_LIMIT_EXIT_CODE`).
9. Locking is best effort. Other processes do not open the file while it is
   running; opening a file that is being written can fail with errno 35.
10. Portability on macOS, Linux and Windows: readers depend only on h5py and
    NumPy; only the filters built into every HDF5 library are used (none,
    gzip, shuffle, fletcher32), never plugin codecs; files hold no absolute
    paths (references to other files are relative paths or SHA-256 values);
    all text is UTF-8.
11. Resume (`ResultWriter.resume`): a run killed or stopped by the scheduler continues in the same file from a
    checkpoint increment. The source is the `.partial` file of a killed run or the closed file with status
    `time_limited`, `failed` or `recovered`; a file closed `complete`, a changing mesh (`mesh_varies`), another
    `run_id` or mesh, another field list, a checkpoint increment that is not among the history rows, and stored steps
    missing below it are refused (`ResumeRefused`) before anything is changed. History rows after the checkpoint
    increment, the stored steps that only they refer to, and the long tables (`crack_tips`, `substeps`) after it are cut,
    the file is renamed to `.partial`, `status` becomes `running`, `finished` and `status_reason` are removed, and the
    resume is recorded in the root attribute `resume_count` and the JSON list `resumes` in the attributes of `meta`
    (time, increment, rows and steps kept and cut, previous status and SHA-256, scheduler job number). Both are outside
    the validated vocabulary and ignored by the validator. The bytes before the cut are kept as
    `<file>.pre-resume-<k>.bak`. A finished file equals an uninterrupted one in every dataset and attribute but
    these records and the wall-clock history columns. The code that records checkpoints must publish a checkpoint
    only after the history row of its step is in the file, so that a checkpoint is never ahead of the file.

## 5. Validator

`phast result validate <file>` reports findings of three levels.

- ERROR: a missing required group or attribute; an unknown
  `phast_schema_version` major; an array shorter than the offsets implied by
  `NSteps`; a field without `location`, `components`, `units` or `kind`; a
  dangling soft link (ParaView drops the array that follows a dangling link
  without a message); a `state` or `history` field, coordinate or material
  value that is not float64; a dataset that uses a plugin filter; an absolute
  path stored as a reference; a virtual dataset; `status = running` on a file
  that is not on scratch storage.
- WARN: history stored as a cell mean while the solver keeps it per point;
  `history/step_index` with gaps that the configuration does not explain; a
  vector in `/VTKHDF` with 2 components.
- INFO: optional groups that are absent.

The precise rules are as follows. The `validator` section of the schema holds
the lists.

- Array length: a time-indexed array MUST hold at least the rows implied by
  the offsets of the `NSteps` stored steps (`offset[NSteps-1] + N` or `+ E`;
  for ImageData `offset[NSteps-1] + 1` along time). Rows beyond them, left by an
  interrupted append, are allowed. `Steps/Values` and every offset dataset
  hold at least `NSteps` entries.
- Field attributes: `location`, `components`, `units` and `kind` are required
  on every field (section 2.1). A field is every dataset in
  `/VTKHDF/PointData` and `/VTKHDF/CellData` and any other dataset that carries
  a `kind` attribute. In `/VTKHDF` the number of names in `components` equals
  the number of stored components.
- Precision: `phase_id` may be uint8 or int8. In `/history`, floating columns
  are float64. Integer columns are allowed only for counters, indices and
  flags (`increment`, `step_index`, `damage_solved`, `cutbacks`,
  `stagger_iterations`, `mechanics_iterations`, `mesh_index`) and for
  categorical columns with `codes`. Every `/history` dataset has `units`.
- References: string attributes and datasets whose names end in `path`,
  `manifest` or `file` are file references. An absolute POSIX, Windows drive,
  UNC or `file:` value is an ERROR. Soft-link targets and HDF5 paths that
  resolve inside the file are not file references.
- Status: the validator cannot see where a file lives. `status = running` is an
  ERROR unless the caller states that the file is on scratch storage
  (`--running-ok`).
- Also an ERROR: `mesh_varies = true` in a 2.0 file (2.1 allows it, section
  8.6); `profile = setup` with `NSteps` other than 0; `history_coupling =
  lagged` without `H_damage_input` in `/VTKHDF`; a soft link that points
  outside `/VTKHDF`; an external link.
- `history/step_index` gaps are explained when the stored increments are evenly
  spaced (the cadence `output.result.every` from `meta/config_resolved_yaml`,
  or the first gap when the configuration does not state it). The last stored
  increment may come earlier.

Versioning: `MAJOR.MINOR.PATCH`. Minor versions only add optional items, and
readers accept any minor version of their major. The validator checks a file
with the schema of its own `phast_schema_version` (2.0.x with 2.0.0, 2.1.x
with 2.1.0, a later minor version with the newest schema it knows), so a 2.0
file receives the same findings that it received before 2.1 existed.

Recovery: `phast result recover <file>` works on a copy. It clears the
open-for-write flag (`h5clear -s`, or a read-write open with h5py, which is
enough for files written with the `libver` bounds of section 4 because their
superblock has no such flag), trims every time-indexed dataset to the longest
consistent step count, and sets `status = recovered`. The step count kept is
the smallest of `NSteps`, the length of every `Steps` dataset and the number of
steps whose rows are complete in every array. `history/` keeps the rows up to
the last increment whose fields are kept, and `derived/` and the long tables of
section 8 are trimmed to match. The root attribute `recovered_from_steps`
records `NSteps` before trimming, and `finished` the time of recovery.

## 6. Reader

`phast.load_result(path)` accepts a run directory, a layout-2.x file of any
name, or a run directory of the older formats. The methods that describe a
layout-2.x file are `result.mesh_arrays()`, `result.step_index()`,
`result.materials()`, `result.boundary_conditions()`,
`result.quadrature(name, step)`, `result.field_attributes(name)`,
`result.history_attributes()`, `result.history_columns()`,
`result.source_files()`, `result.file_metadata()`, `result.steps()`,
`result.times()` and `result.n_steps()`. Layout 2.1 adds `result.mesh_versions()`,
`result.step_mesh()`, `result.crack_tips()`, `result.validation()` and
`result.initial_conditions()`, and an optional `step` argument of
`result.mesh_arrays(step)` (section 8). The `field_attributes`,
`history_attributes` and `history_columns` methods return empty results for the
older formats, which store no attributes. The other methods listed here raise
`ResultLoadError` for the older formats.

Reader conventions:

- In a directory, `result.vtkhdf` is read in preference to any older store.
  `run_manifest.json` and `run_metadata.json` are optional when it exists.
  Without either JSON file, `metadata()` returns the root attributes and the
  `meta/` attributes. `file_metadata()` returns them in every case.
- `step` is the index of a stored step in `/VTKHDF/Steps`; negative values
  count from the end. Rows are located through the offsets. `NSteps` is
  authoritative, so rows of an incomplete step are never returned.
  `result.steps()` lists the valid values.
- `field()` returns 2D vectors with 2 components. UnstructuredGrid arrays are
  `(N[, c])` or `(E[, c])`, and history per point is `(E, n_qp)`. ImageData
  arrays are `(nz + 1, ny + 1, nx + 1[, c])` or `(nz, ny, nx[, c])`, x
  varying fastest. Static arrays are returned at any step. A `/derived` array is
  returned for a stored step only when `history/step_index` maps the step to an
  increment that has a row; otherwise `ResultLoadError` is raised.
- `field_names()` lists reference names (`H` appears as `history_field`, as for
  the older formats); the stored names remain valid arguments.
- `history(name)` keeps the CSV files under their names and adds, for names no
  CSV provides, tables built from `/history`: `history` (columns of a group
  joined with `_`, vector columns split by component, for example
  `reaction_top_y`), `energy`, and the long tables `crack_tips` and `substeps`
  when the file has them. `history_columns()` returns the `/history` datasets as
  arrays.

## 7. Figures

`phast figures <run> [-o OUT] [--steps first,peak,last] [--dpi N] [--fields
...] [--deform-scale F|auto]` (also `python -m phast.io.result.figures`) draws
the standard figures of a layout-2.x result from the file alone, with no
run-specific code. It writes, below 2000 px on both axes, a setup sheet, field
and component sheets at the chosen stored steps (default: the first, the step
nearest the largest load-like history value, and the last), a sheet of damage
across those steps and an evolution sheet, with `manifest.json` (kind, steps,
fields and the SHA-256 of the source files) beside them. Field maps are laid out
about 1900 px wide; when they need more than two rows, state and history fields
and derived fields go on separate sheets. What a file does not store is stated
in a footnote and is not raised as an error. The figures use the STIX serif
font.

## 8. Layout 2.1 additions

Version 2.1.0 adds optional items for forward runs. Its schema is a full copy of
the 2.0.0 schema plus the items below, so a file embeds the complete schema it
was written against. Nothing in 2.0 changes: every 2.0 file is a valid 2.1 file,
except that `mesh_varies = true` is accepted only from 2.1 on.

### 8.1 Version rules

- The writer writes 2.0.0 until the first item of this section is used. At
  that point it sets `phast_schema_version = "2.1.0"` and replaces
  `meta/schema_yaml` by the 2.1 text, both before the item is written, and
  records the items in the `layout_2_1_features` attribute of `/meta`. A file
  that is interrupted afterwards is therefore checked by the rules of its own
  version.
- The validator uses the schema of the version of the file (section 5). The
  notes about optional groups (INFO) of a 2.1 file include
  `/initial_conditions` and `/meta/validation`.
- The long tables below `/history` (`crack_tips` and `substeps`) have one row
  per event, and the datasets of one table have the same length. Their rows
  refer to rows of `/history/increment`. The caller adds the rows of an
  increment after `ResultWriter.append` of that increment and then flushes, so
  that an interrupted file never holds table rows for an increment without a
  history row; a run killed between the two loses only the table rows of its
  last increment.

### 8.2 Crack tips and crack speed

`history/crack_tips/` has one row per tip and increment, in increment order
(the tips of one increment in order of `tip_id`). An increment without a crack
has no row.

| Dataset | Type and unit | Content |
|---|---|---|
| `increment` | int64, `1` | row of `/history` |
| `tip_id` | int64, `1` | 0, 1, ... within the increment |
| `position` | float64 `(n, 3)`, `mm` | `x, y, z` (`components` attribute; `z = 0` in 2D); the attribute `method` states how the tip was found |
| `threshold` | float64, `1` | damage value that defines the tip, in (0, 1] |
| `speed` | float64 `(n,)`, `mm/s` | derived; added when the file is closed |

The speed is derived from the stored positions. Each coordinate of a tip is
smoothed along the rows of that tip with a Savitzky-Golay filter (window 5,
order 2; unchanged when the tip has fewer than five rows). The speed of a row is
the length of the backward difference of the smoothed position divided by the
difference of `history/time` between the two rows, and the first row of a tip
has no speed (NaN). Smoothing the position before differentiating keeps the jump
of one element width between snapshots from appearing as a speed spike. The
dataset carries `fit_method`, `fit_window`, `fit_polyorder` and `derived_from`;
its unit follows the time unit (`mm` per unit of pseudo-time in quasi-static
runs, whose `history/time` has unit `1`). A file that was closed before the
speed was written has no `speed` dataset. `Result.crack_tips()` then computes it
with the same fit and says so (`speed_source`). Recovery removes `speed` when it
trims rows.

The 2D runner finds the tip with `physics.fracture_mechanics.find_crack_tip`
(edge interpolation of the isoline `d = threshold`, farthest point along `x`,
`-x` or `y`). It follows one tip on T3 cells; the table itself holds any number
of tips.

### 8.3 Per-solve records

`history/substeps/` has one row per damage solve: `increment`, `call` (the index
of the solve within the increment, from 0) and the columns of the damage
solver's record: `damage_increment`, `cg_iterations`, `cg_residual`,
`cg_rhs_norm`, `cg_initial_residual`, `cg_residual_reference`,
`cg_relative_residual`, `cg_converged`, `kkt_max`, `relative_kkt_max`,
`projected_relative_residual`, `projected_relative_residual_max`, `damage_min`,
`damage_max`, `min_increment`, `bounds_satisfied`, `irreversibility_satisfied`,
`walltime_s` (s), `cooldown_remaining`, `retry_counter`, `amg_eligible`,
`rebuild_attempted` and `rebuild_succeeded`. Counters are int64, flags are int8
(0 or 1) and the other columns are float64. Text values
(`requested_preconditioner`, `effective_preconditioner`, `amg_skip_reason`,
`fallback_target`, `fallback_reason`) are int8 codes with a `codes` attribute
(section 2.2) that grows with the names that occur.

The writer and the validator support this table (`ResultWriter.append_substep`).
The 2D runner does not write it, because the damage solver does not report the
record of each solve.

### 8.4 Velocity conditions, initial conditions, wave speeds

- `loading/bc/<k>` accepts `kind = velocity` (with `components` and `value`, a
  velocity in `mm/s` or the name of a history column, and `value_units`). For
  the velocity-impact ramp the runner records the prescribed velocity: `value`
  is the boundary value times the impact velocity `v0` in mm/s,
  `ramp_type = velocity_impact`, `t_ramp` (s), and `displacement` describes the
  displacement that the solver imposes, `u = v0 t^2 / (2 t_ramp)` for
  `t < t_ramp` and `u = v0 (t - t_ramp / 2)` afterwards. The validator requires
  `kind` to be one of `displacement`, `traction`, `symmetry`, `periodic`,
  `velocity`.
- `initial_conditions/` holds the state at the start of the first increment, on
  the initial mesh: datasets named `displacement`, `velocity`, `acceleration`,
  `damage`, `H` or `damage_lower_bound`, each with the field attributes of
  section 2.1 (`kind = state` or `history`, float64). The attribute
  `specification` (JSON) of an array records how it was set. For `damage` the
  runner records the pre-seeded regions (`preseed_notch_nodesets` and
  `preseed_damage`). A pre-seeded notch is imposed as a large history value and
  one damage solve, so the stored damage is the state after that solve. The
  validator checks the length against the mesh.
- `materials/phases/<id>` may carry `c_p`, `c_s` and `c_R` (unit `mm/s`, listed
  in the `units` attribute) next to `rho` (`t/mm^3`): the P-wave speed
  `sqrt((lam + 2 mu) / rho)` with the Lame parameter of the analysis (plane
  stress: `E nu / (1 - nu^2)`), the shear-wave speed `sqrt(mu / rho)` and the
  Rayleigh speed `c_s (0.862 + 1.14 nu) / (1 + nu)`. For an isotropic phase with
  `E` and `nu` the validator checks each against these formulas to a relative
  difference of 1e-6.

### 8.5 Validation results

`meta/validation/` holds the comparisons that a run was judged by, one row each:
`name` (text), `value` (float64; NaN for a condition that is either met or not),
`tolerance` (float64; NaN when there is none), `passed` (int8, 0 or 1) and,
optionally, `units` and `description` (text). The attribute `overall_passed` is
1 when every row passed (the validator checks that it agrees with the rows), and
`source` names the summary that the rows come from by a relative file name. The
group is written once, at any time before the file closes
(`ResultWriter.write_validation`).

### 8.6 Meshes that change during a run

`mesh_varies = true` is allowed for UnstructuredGrid runs. The geometry follows
the time-dependent layout of VTKHDF, with one part for each mesh version.

- `NumberOfPoints`, `NumberOfCells` and `NumberOfConnectivityIds` have one entry
  per mesh version. `Points`, `Connectivity` (point numbers local to the
  version), `Offsets` (cells + 1 entries per version, each starting at 0) and
  `Types` are the versions concatenated.
- Each stored step names its mesh: `Steps/NumberOfParts = 1`, `PartOffsets[t]`
  is the version, and `PointOffsets[t]`, `CellOffsets[t]` and
  `ConnectivityIdOffsets[t]` are the first point, cell and connectivity entry of
  that version. Steps on the same version repeat the same offsets. Array rows
  are stored per step as before. The offset of a step in `PointDataOffsets` and
  `CellDataOffsets` is the number of rows stored before it, and a step holds the
  points or cells of its version. An array that is static on each mesh
  (`phase_id`, `orientation`) holds the rows of each version once, and the
  offsets of the steps point at the rows of their version.
- `history/mesh_index` (int64, `1`) gives the version in force at each
  increment, including increments that store no fields. It is non-decreasing and
  agrees with `PartOffsets` at the stored steps. A mesh is registered at the
  increment where it first applies, and the fields stored at that increment are
  on the new mesh.
- Version 0 uses the groups of section 2 (`mesh/cell_blocks`, `mesh/node_sets`,
  `mesh/graph`, ...). Version `k >= 1` has the same sub-groups in
  `mesh/versions/<k>/`, with attributes `n_points`, `n_cells` and
  `first_increment` and the maps to the previous version: `cell_parent` `(E_k,)`
  (the cell of version `k - 1` that each cell came from, -1 for none),
  `node_origin` `(N_k,)` (the node of version `k - 1`, -1 for a new node) and
  `node_parents` `(N_k, 2)` (the two nodes of version `k - 1` whose mean is a new
  node, -1 for existing nodes). A conforming bisection or a quadrilateral split
  produces these maps directly (`refinement_maps` converts the dictionaries of
  `phast.solvers.adaptive`).
- A non-conforming mesh stores its hanging-node constraints in `hanging_nodes/`
  of its version (`mesh/hanging_nodes` for version 0): `constrained_nodes`
  `(n,)`, `master_nodes` `(n, m)` and `weights` `(n, m)` (float64, each row
  summing to 1). The value at a constrained node is the weighted sum of the
  values at its master nodes, and a master node is not itself constrained.
- Arrays of a fixed shape cannot follow the mesh. With `mesh_varies` there are
  no `quadrature/`, `derived/` or `interfaces/` arrays, and every field is
  stored inside `/VTKHDF` (per-point history as a multi-component cell array,
  section 2.3). An ImageData grid is fixed and cannot change.
- The validator checks the counts and offsets of every version, the maps
  (ranges, shapes, one origin per node), the hanging-node constraints, the node
  sets against the node count of their version, `mesh_index`, and that the rows
  suffice for the mesh of each stored step. Recovery trims the arrays to the
  rows of the mesh version of each kept step.
- The reader returns each stored step on its own mesh: `result.field(name,
  step)` has the size of the version of that step, `result.mesh_arrays(step)`
  returns that version (with `mesh_version` and the maps), `result.step_mesh()`
  lists the version of every step and `result.mesh_versions()` lists the
  versions.

ParaView 5.13 shows such a file with the points, cells and arrays of each step.
The 2D YAML runner does not refine the mesh, so it stops the file at a mesh
change and continues with its other outputs. The writer and the format are what
a solver loop that refines the mesh needs.

### 8.7 Not part of layout 2.x

- One-dimensional runs (line cells) are outside the layout: `dim` is 2 or 3.
  The writer refuses `dim = 1`, one-column points and line cell types, and the
  validator reports a file whose `dim` is 1 with the message "one-dimensional
  runs (line cells) are outside layout 2.x: dim must be 2 or 3".
- J-integral and stress-intensity columns of `/history` are not defined. They
  can be added as optional `history/` columns when a run records them.

### 8.8 Settings of the 2D runner

`output.result.layout: 2.1` (default 2.0) records the initial state with the
pre-seeded damage, the wave speeds of the material and, for the velocity-impact
ramp, the prescribed velocity. `output.result.crack_tips: true` or `{threshold:
0.5, direction: x}` writes the crack-tip table of a T3 run and implies 2.1. The
numerical results of a run do not depend on these settings.
