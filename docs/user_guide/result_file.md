# Result files

A single-material two-dimensional phase-field run of the YAML runner writes one
`result.vtkhdf` by default (the multi-material route does not write it). The file is HDF5 and a valid VTKHDF file at the same
time. ParaView 5.13 or later opens it directly,
and `phast.load_result` reads it on macOS, Linux and Windows with h5py and
NumPy only. The layout is specified in [Result file layout](result_file_layout.md).
This page describes how to write, check, view and read these files.

The result file is the file to open, validate and cite. The runner still writes
its CSV files and, where configured, the trajectory store (`training_data.h5`),
so that existing scripts keep working. Older run directories load through the
same reader.

The runner writes the result file unless `output.result` is `false`. A run
directory that holds both `result.vtkhdf` and `training_data.h5` is read from the
result file, so `step` in `field()` counts stored steps and the field names are
those of the result file (section "Reading"). A script that reads step labels or
field names of the trajectory store (`step_0020`, `H_nodal`, `eps_xx`) from
such a directory changes to the stored-step convention, or reads the store
directly with h5py. Setting `output.result: false` restores the former output
set, which holds no result file.

## Writing

### The YAML runner

`python -m phast run <config.yaml>` writes `<output_dir>/result.vtkhdf` with the
defaults of the solver type. The configuration can set `output.result` to a
mapping of the settings below, which are all optional:

```yaml
output:
  result:
    every: 1                     # stored steps; history rows are written for every increment
    fields: [displacement, damage, H, stress, strain]
    derived_location: cell       # cell | quadrature_point
    derived_every: 1             # 1 = in the ParaView view; N > 1 = in /derived every N-th increment
    size_limit_gb: 4             # above the estimate, derived fields move to /derived
    layout: 2.0                  # 2.1 adds the items of section 8 of the layout
    crack_tips: false            # true or {threshold: 0.5, direction: x}; implies layout 2.1
```

`output.result: true`, or leaving the key out, uses the defaults and
`output.result: false` writes no file. The defaults of the settings
depend on the solver type: a quasi-static run stores every increment, and an explicit run stores
every `output.h5_every` increments and also stores velocity and acceleration.
When the estimated size of the stored state exceeds `size_limit_gb`, the runner
stores fewer increments (it prints the value of `every` that it chose) unless
`every` is set. The meaning of each setting is in section 3 of the layout.

The runner records one row of `/history` for every converged increment: the
time, the load parameter, the signed reaction of each fixed or loaded node set,
the energies, and the iteration counts. Quasi-static runs use a pseudo-time of
one unit per increment. Explicit runs store the accumulated physical time and
the time step that each increment used.

Four properties of the runner's file need attention.

- With `solver.adaptive_dt: true` the runner evaluates the load ramp at the
  nominal time `step * dt`, as it did before the result file existed, so the
  stored `load_factor` follows that nominal time while `history/time` follows the
  accumulated step sizes. Constant-step runs are not affected.
- The runner does not write the per-solve table `history/substeps` of layout
  2.1, because the damage solver does not report the record of each solve. The
  writer and the validator support the table.
- If the mesh changes during a run (adaptive refinement), the runner closes the
  result file at the last increment on the initial mesh, records the reason in
  `meta`, and continues with its other outputs. The writer supports changing
  meshes (section 8.6 of the layout); the runner does not use that yet.
- `crack_tips` follows one tip on T3 meshes. It is refused for Q4 meshes.

### Any other solver

```python
from phast.io.result import Field, ResultWriter, UnstructuredMesh

mesh = UnstructuredMesh(points, {"T3": cells}, node_sets={"top": top})
fields = [Field("displacement", "node", "x,y", "mm", "state"),
          Field("damage", "node", "d", "1", "state"),
          Field("H", "cell", "H", "MPa", "history")]
with ResultWriter(run_dir / "result.vtkhdf", mesh, fields, dim=2,
                  analysis="plane_strain", run_id="sent-001",
                  history_units={"load_parameter": "mm"}) as w:
    for inc in range(n_increments):
        ...                                   # solve the increment
        w.append(dict(increment=inc, time=t, dt=dt, load_parameter=u_top, damage_solved=1),
                 fields=dict(displacement=u, damage=d, H=H))
```

The writer refuses what the layout forbids: a state or history field that is not
float64, a missing `H_damage_input` under lagged coupling, a history column
without units, an energy column without its method, and a stored step that lacks
one of the fields in the view. `ImageGrid` replaces `UnstructuredMesh` for voxel
grids. `SoftStop` turns SIGTERM and SIGUSR1 into a flag: the loop finishes the
increment, closes with `status = time_limited` and exits with code 11
(`TIME_LIMIT_EXIT_CODE`).

The writer creates `result.vtkhdf.partial` and writes it on local disk, not in a
synchronised folder. A run whose output
directory lies under a synchronised folder (a path component `CloudStorage`,
`OneDrive`, `Dropbox`, `Mobile Documents` or `Google Drive`) issues a warning and
still writes the file; keep run directories on local disk and copy them afterwards. After the file closes, the writer validates it, renames it
to `result.vtkhdf` and writes `result.vtkhdf.sha256`. A `.partial` file that
remains belongs to a run that was killed. It still opens with all its complete
steps.

## Checking and archiving

```bash
python -m phast result validate run/result.vtkhdf
python -m phast result recover run/result.vtkhdf.partial -o run/result.recovered.vtkhdf
python -m phast result finalize --archive run/result.vtkhdf
```

`validate` reports ERROR, WARN and INFO findings and exits with 1 on any ERROR.
`recover` writes a trimmed, validated copy of an interrupted file and never
modifies the source. `finalize --archive` rewrites a closed file with gzip level
4, shuffle and fletcher32, compares the copy value by value with the source, and
then replaces the file and its checksum. Validate a file before copying it to
synchronised storage.

### Continuing an interrupted run

A solver that stores its own checkpoints can continue an interrupted file in place
instead of writing a new one:

```python
from phast.io.result import ResultWriter, ResumeRefused

w = ResultWriter.resume("run/result.vtkhdf", mesh, fields, upto_increment=40, run_id="run-001")
```

The call cuts the history rows and stored steps after the checkpoint increment,
keeps the bytes before the cut as `result.vtkhdf.pre-resume-1.bak`, records the
resume in the file (`resume_count` and the list `resumes` in the attributes of
`meta`) and returns a writer that appends from the next increment. It refuses
with `ResumeRefused`, before changing anything, a file that was closed complete,
a changing mesh, another run or mesh, another field list, or a checkpoint that
is not among the stored rows. The rules are item 11 of the writer rules in
`docs/user_guide/result_file_layout.md`. The YAML runner does not resume a run.

## Viewing

- ParaView: open `result.vtkhdf`. The time steps, the mesh or voxel grid and
  every array in `/VTKHDF` appear without a plugin or a sidecar file. Tensors
  are plain multi-component arrays in the order `xx, yy, zz, yz, xz, xy` (2D:
  `xx, yy, xy`), so VTK tensor filters must not be applied to them directly.
  ParaView 5.13 shows no arrays of a voxel run with exactly one stored step, and
  the validator warns about such files.
- Standard figures: `python -m phast figures <run>` writes the setup sheet, the
  field sheets at the first, peak-load and last stored steps (2D maps; for voxel
  runs mid-plane sections and an isometric view with a cut-away corner) and the
  evolution sheet, with `figures/manifest.json`. The images are below 2000 px on
  both axes. The command reads layout-2.x files and refuses other run
  directories.

## Reading

```python
import phast

r = phast.load_result("run")                 # a run folder or a .vtkhdf file
r.steps(), r.times()                         # stored steps and their accumulated time
d = r.field("damage", step=-1)               # 2D vectors come back with 2 components
H = r.field("H", step=0)                     # (E, n_qp) when kept per quadrature point
rows = r.history("history")                  # /history as rows; CSV files keep their names
cols = r.history_columns()                   # /history as arrays
mesh = r.mesh_arrays()                       # points, cells per block, node sets, graph edges
mats, bcs = r.materials(), r.boundary_conditions()
```

`step` is the index of a stored step, not the step label of the trajectory store
(`step_0020`): in an explicit run with `output.h5_every: 20` the label 40 is the stored step 2
of the result file, and `field("damage", step=40)` raises `ResultLoadError` when
the file holds fewer steps. The result file also names its fields as stored (`H` appears
as `history_field`, strain and stress as multi-component arrays), and it has none
of the trajectory-only fields (`H_nodal`, `psi_plus`, `eps_xx`). `history/step_index` maps each load
increment to its stored step, or to -1 when no fields were stored for it.
`field()` returns stored arrays only; it does not derive stress, strain or other
quantities that the run did not store. Code that needs float32 converts while
reading (`np.asarray(x, dtype=np.float32)`), because no second copy of a run is
written.

`phast.load_result` also accepts the older run directories (a manifest or
metadata file with `training_data.h5` or `training_data.zarr`). For those, only
the methods listed in [Results and visualization](results_visualization.md) are
available. Methods that describe the mesh, the steps and the layout need a
layout-2.x file and raise `ResultLoadError` otherwise.

### Layout 2.1 files

A file that uses an item of section 8 of the layout has
`phast_schema_version = 2.1.0`. `phast result validate` checks every file with the
schema of its own version, and `phast.load_result` reads both. The runner writes
2.1 files when `output.result.layout: 2.1` or `output.result.crack_tips` is set.
The reader methods for the added items are:

```python
r.crack_tips()                               # increment, tip_id, position, threshold, time, speed
r.history("substeps")                        # one row per damage solve (text columns as names)
r.initial_conditions()                       # initial state, 2D vectors with 2 components
r.validation()                               # name, value, tolerance, passed of the run's comparisons
r.mesh_versions(), r.step_mesh()             # runs whose mesh changes (refinement)
r.field("damage", step=1)                    # the array of that step, on that step's mesh
r.mesh_arrays(step=1)                        # points, cells and maps of that step's mesh
```

`phast figures` draws 2.1 files, except runs whose mesh changes. It draws one
mesh and reports that the mesh changes.

## Checks that need ParaView

The automated tests read the files with h5py and NumPy. Two scripts open the
conformance files in ParaView and compare what ParaView reads with the expected
values. They run with ParaView's own Python (`pvpython`), on macOS, Linux or
Windows, and exit with 1 on any mismatch.

```bash
cd tests
python -m result_v2.fixtures fixtures_out
python -m result_v2.fixtures_21 fixtures_out_21
pvpython result_v2/paraview_check.py fixtures_out
pvpython result_v2/paraview_check_21.py fixtures_out_21
```

Run them after any change of the writer that affects the `/VTKHDF` group, and on
each ParaView version that the project supports. The files written by the YAML
runner open in ParaView in the same way as the conformance files.

## Limits

- The layout covers two- and three-dimensional runs. One-dimensional runs (line
  cells) are outside it.
- The result file is a record of a run. It does not hold the solver state, so
  it is not a checkpoint from which a run is restarted; a solver that keeps its
  own checkpoints continues the file with `ResultWriter.resume`.
- The YAML runner writes two-dimensional T3 and Q4 results of the staggered
  phase-field routes. The solid-mechanics, plasticity and interface, and
  one-dimensional bar routes do not write a result file, and they accept
  `output.result` without effect. Results of other solvers need the writer API
  above.
- The run directories written before the result file existed keep their formats,
  and the reader does not convert or rewrite them.
