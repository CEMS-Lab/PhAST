# Promoted example contract

This is the canonical contract for promoted PhAST example folders. Link to
this page when asking, "What must a PhAST example folder contain?"

The rule is simple: a promoted example must be runnable, inspectable,
reproducible, lightweight enough for public documentation, and honest about
its capability boundary. This page supersedes the older scattered checklists in
`examples/PUBLIC_EXAMPLES_CONTRACT.yaml` and
`configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml`.
Those files remain machine-readable manifests, but this page is the
human-facing source of truth.

## Example tiers

| Tier | Command shape | Public meaning |
|---|---|---|
| YAML-first | `python -m phast run <config.yaml>` | Public runnable input deck with a curated result bundle. |
| Script contract | `python <run.py> --output-dir <dir>` | Beta or specialist workflow not yet expressible through the generic YAML runner. |
| Diagnostic | Case-specific | Developer evidence only; not a gallery example until deliberately promoted. |

YAML-first is the default. Script-contract examples are allowed only for beta
capabilities such as plasticity/interface validation slices where the public
YAML surface is not yet complete.

## example folder layout

Use a flat leaf folder. A public example should not make users search through a
raw run tree or unpublished scratch archive.

```text
examples/<family>/<case>/
  README.md
  config.yaml
  run.py                       # only for script-contract or compatibility examples
  run_manifest.json
  run_lockfile.json            # required for full solver runs
  run_metadata.json            # required for full solver runs
  visual_manifest.json         # required when visuals are present
  visual_manifest.md           # optional human-readable visual index
  run_fluent.py                # public Python/fluent companion for tutorial-ready YAML examples
  fluent_setup.py              # optional legacy alias when retained for compatibility
  initial_conditions.png
  thumbnail.png
  damage_final.png             # fracture examples
  displacement_final.png
  stress_final.png
  strain_final.png
  response.csv / results.csv
  history.csv
  energy.csv
  solver_telemetry.csv
  timing_per_step.csv
  field_evolution.mp4          # or damage/plastic-strain evolution equivalent
  compare.png                  # when a reference exists
  compare_report.txt           # when a reference exists
```

Do not commit raw cluster result folders, proprietary COMSOL binaries, full
scratch paths, trajectory stores, or archaeology from exploratory runs.
Raw trajectory stores, VTU/PV time series, and full solver-output archives
belong in local rerun output directories or named external release artifacts.
The public repository should retain the lightweight YAML, Python runner, CSV,
metadata, image, and animation artifacts needed to understand and rerun the
case.

## README contract

Every promoted example `README.md` must contain:

- the physics and benchmark family;
- the capability status: production, beta, diagnostic, or scaffold;
- the exact local command;
- the expected output directory;
- the required result artifacts;
- the evolution animation artifacts;
- input conditions: geometry, mesh, material, units, initial crack/notch,
  boundary conditions, loading, solver settings, and output settings;
- known limitations and claim boundary;
- rerun instructions from the checked-in `config.yaml`;
- result-inspection snippet using `phast.load_result(path)`.

The README should say what the example demonstrates and what it does not
demonstrate. Do not describe a beta workflow as a production solver path. Do
not imply that a qualitative crack image is a benchmark validation unless the
comparison report and pass/fail metrics are present.

## YAML-first rules

For public YAML examples, `config.yaml` is the exact input deck. It must be
checked in and runnable from the repository root:

```bash
python -m phast run examples/<family>/<case>/config.yaml --validate-only
python -m phast run examples/<family>/<case>/config.yaml --output_dir runs/<case>
```

The YAML must document or encode:

- schema version and problem type;
- geometry or mesh source;
- named regions, node sets, element sets, and crack/notch masks;
- material parameters and units;
- phase-field model, split, `Gc`, length scale, and residual stiffness where
  relevant;
- initial conditions and pre-cracks;
- boundary conditions and loading protocol;
- solver backend, convergence tolerances, time/load stepping, and device;
- output requests: plots, animations, trajectories, telemetry, manifests, and
  comparison settings.

Use the fluent `phast.Problem` API to author new models when convenient, but
the promoted example is reproduced through the saved YAML deck.

## Script-contract rules

Script-contract examples must be explicit about why they are not YAML-first.
They must include:

- `run.py`, `run_fluent.py`, or a named script path;
- command-line help;
- an `--output-dir` argument;
- deterministic defaults;
- `config.yaml` or equivalent resolved settings copied into the result folder;
- `run_manifest.json`, `run_metadata.json`, and `visual_manifest.json` when
  visuals are generated;
- a claim-boundary paragraph in the README.

If a script-contract workflow becomes a public standard path, promote it by adding
a YAML deck, updating `examples/PUBLIC_EXAMPLES_CONTRACT.yaml`, and adding a
test for the expected artifacts.

## Tutorial readiness flags

`examples/PUBLIC_EXAMPLES_CONTRACT.yaml` is deliberately stricter than the
gallery list. Every listed example must declare one of two states:

- `tutorial_ready: true`: the leaf folder has the complete tutorial bundle:
  canonical YAML, `run_fluent.py` or equivalent fluent authoring companion,
  setup visual, final field plots, field/damage animations, run
  manifests, and visual manifest. Trajectory stores are generated by local
  reruns or distributed through a named external artifact.
- `tutorial_ready: false`: the entry is useful but incomplete as a full
  tutorial. It must include non-empty `tutorial_blockers` naming the exact
  missing work, such as HPC trajectory regeneration, fluent lowering, rigid
  connector support, beta-script promotion, or trajectory externalization.

Do not set `tutorial_ready: true` for an example unless its artifacts are
generated by the solver or postprocessor and enforced by
`tests/test_public_examples_contract.py`. Do not create placeholder Zarr files
or hand-made animations to satisfy the flag.

## Inputs and provenance

A promoted example must make the setup inspectable before the solver runs.

### Geometry and mesh

Document:

- geometry dimensions and coordinate units;
- whether the mesh is generated or imported;
- mesh file path and generator;
- element type, typical mesh size, and refinement region;
- pre-crack/notch/void/inclusion geometry;
- boundary tags and named regions.

Required files for solver examples:

| File | Requirement |
|---|---|
| `mesh.geo` | Required when gmsh source exists. |
| `mesh.msh` | Required when the public example must rerun without remeshing. |
| `initial_conditions.png` | Required setup visual showing mesh, boundary tags, pre-crack/notch, and relevant material regions. |

### Materials and units

Document all material parameters and units. For fracture examples include
`E`, `nu`, `Gc`, phase-field length scale, density for dynamics, and the energy
split. For solid mechanics include constitutive model, stress/strain measure,
and unit conventions. For plasticity include yield stress, hardening law, and
claim boundary.

### Boundary conditions and loading

Document constrained boundaries, loaded boundaries, displacement/load schedule,
time/load units, sign convention, and reaction side. For benchmark comparisons,
the result CSV must use the same reaction side and sign convention as the
reference or state the conversion.

### Solver and output settings

Document solver type, backend, tolerances, time/load step control, damage
subcycling, device, dtype, and output cadence. If a result depends on an
optional backend such as PETSc/MUMPS, AmgX, cuDSS, PyVista, or `viz-fast`,
state that requirement and the fallback behavior.

## Required metadata files

| File | Required when | Contents |
|---|---|---|
| `run_manifest.json` | Every promoted example | command, git commit, script/config, result path, issue/review notes where applicable, and artifact list. |
| `run_lockfile.json` | Full solver run | input config hash, resolved config, CLI args, dependencies, platform, mesh/material/solver/device summaries. |
| `run_metadata.json` | Full solver run | timestamp, host class, device, dtype, thread count, package versions, git state. |
| `visual_manifest.json` | Any promoted visual output | file list, kind, source data, dimensions, generation command, and claim role. |
| `visual_manifest.md` | Optional but recommended for gallery examples | human-readable visual index with thumbnails or short descriptions. |

When a run was produced on HPC, the metadata must also include scheduler/job
context: job id, job name, partition, node list where public-safe, Slurm script
path, submit command, stdout/stderr log names, exit code, elapsed time, and
promotion date. Do not expose absolute scratch paths in public docs.

## Required CSV outputs

CSV files are the lightweight numerical audit source. Use stable column names.

| File | Required when | Typical columns |
|---|---|---|
| `results.csv` | reaction/load output exists | `step,time,displacement,reaction_kN,max_d,max_H,stagger_iter,elapsed_ms` |
| `response.csv` | solid mechanics examples | load/displacement or prescribed/response pair plus status columns |
| `history.csv` | solver examples | step, time/load, max damage, max history/driving force, reaction, key energy terms |
| `energy.csv` | fracture/dynamic examples | `step,time,elastic,fracture,kinetic,external,total` |
| `solver_telemetry.csv` | iterative solver examples | step, residuals, relative residuals, stagger/Newton/linear iterations, `dt` or load increment |
| `timing_per_step.csv` | promoted performance or HPC examples | step, wall time, solver components, output time where available |
| `crack_tip.csv` | crack-tip tracking exists | step, time/load, tip coordinates, speed |
| family-specific CSVs | beta/specialist examples | documented in README and manifest |

CSV files should be small enough to review in git. Large dense field histories
belong in trajectory stores outside the public example payload.

## Required visual outputs

Every promoted visual should use physically meaningful color limits and paper
style. A figure should help a reviewer understand the setup, the final state,
or the comparison. It should not be decoration.

### Pre-run visuals

- `initial_conditions.png`: mesh, boundary tags, pre-crack/notch, inclusions,
  material fields, and loading direction where relevant.

### Final-state visuals

- `damage_final.png` for fracture;
- `thumbnail.png` for gallery cards;
- deformation, displacement magnitude, von Mises, strain energy, equivalent
  plastic strain, or material-field plots for solid/plasticity examples;
- `material_fields.png` where heterogeneous fields drive the result.

### Evolution visuals

- Prefer `damage_evolution.mp4` for dense time histories.
- Every tutorial-ready example must include an animation that shows the
  physical response evolving. Use `damage_evolution.mp4` for fracture,
  `field_evolution.mp4` for solid mechanics, and a family-specific name such
  as `plastic_strain_evolution.mp4` when that is clearer.
  when the response is a load-displacement, stress-strain, residual, or
  oscillator history rather than a crack path.
- GIFs are acceptable for lightweight docs/review; cap them at roughly 60 to
  100 frames and about 1 to 5 MB when practical.
- Include enough frames to show setup, crack initiation, crack interaction,
  and final state.

### Curves and diagnostics

- `load_displacement.png` or `response.png` when loading is controlled;
- `energy.png` for fracture/dynamics;
- `staggered_convergence.png` or `convergence.png` for iterative runs;
- `crack_path.png` when a crack path metric is available;
- `compare.png` and `compare_report.txt` when a reference exists.

### Figure style

Use the repository paper style for promoted figures:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})
```

Keep review-facing PNGs below about 2000 pixels on the largest axis. Use
single-column widths near `3.5 in`, double-column widths near `7 in`, and
about `200 dpi` for raster paper figures unless a script has a documented
reason to differ.

## Trajectory and visualization stores

Use trajectory stores only when they are needed for replay, ML, restart,
derived fields, or high-fidelity post-processing.

- New trajectory/dataset workflows should prefer Zarr.
- Legacy H5 remains compatibility input/output for old artifacts.
- Tutorial-ready examples should document how to regenerate trajectory stores
  locally when field-level replay is needed.
- VTU/PVD remains the ParaView-native visualization path.
- `.pv`/PyVista-zstd is an optional fast-path for large visualization output.

Do not commit raw trajectory stores in public examples. Promote a lightweight
summary, manifest, plots, animations, and a documented regeneration command or
external artifact reference instead.

## Result inspection

Every promoted example README should include a short inspection snippet:

```python
import phast

result = phast.load_result("runs/my_example")
print(result.metadata())
print(result.manifest())
print(result.history_names())
history = result.history("energy")
visuals = result.visuals()

if result.has_field("damage"):
    damage = result.field("damage", step=-1)
    result.plot("damage", step=-1)
```

Use `phast.load_result(path)` for existing result folders. Do not require users
to rerun the solver just to inspect metadata, histories, visuals, or stored
fields.

## HPC execution and batch promotion

Promoted examples must be runnable locally first, but HPC is the expected path
for large dynamic runs, long quasi-static validations, and batch regeneration.

### Submit one example

Use validated YAML decks locally first. Site-specific Slurm templates and
queue wrappers are maintained outside the public repository:

```bash
# Validate locally first.
python -m phast run examples/quasistatic/miehe_tension/config.yaml --validate-only

# CPU or GPU batch runs should use your site's scheduler template.
sbatch <site_scheduler_template>.slurm
```

GPU submissions should record the selected partition, GPU type, dry-run plan,
and reason for pending jobs. CPU-only jobs may go through direct `sbatch` when
the Slurm header is explicit and runtime is acceptable.

### Submit a batch

Batch jobs should use Slurm arrays or a checked-in manifest. The manifest must
list config path, output key, device request, expected wall time, and promotion
target. Public examples should remain runnable through `python -m phast run`
before any site-specific scheduler wrapper is used.

Before submitting a batch:

1. run `--validate-only` on each YAML deck;
2. run your site scheduler wrapper in dry-run or plan-only mode for GPU jobs;
3. confirm output roots and log paths are public-safe or internal-only;
4. record the issue number or review target in the job manifest.

### Logs and output roots

Slurm logs should go to a `logs/` folder with job name and job id in the file
name. HPC results should go to an organized results root with one folder per
run or array task. Public docs should describe the relative structure, not an
absolute scratch path.

### Required HPC provenance

HPC-promoted outputs must record:

- scheduler name and job id;
- job name, array id when applicable, partition, requested CPUs/memory/time,
  and GPU request when applicable;
- Slurm script path and submit command;
- git commit and dirty-state summary;
- resolved command and config path;
- stdout/stderr log names;
- exit code and elapsed time;
- result folder key;
- promotion command/date and copied artifact list.

The separate `lab-infra` repository is the operational source for current
cluster paths, modules, queue aliases, and promotion conventions. Public docs
may reference it as the maintainer knowledge source, but they must not expose
user paths, secrets, or credentials.

### Promote back from HPC

Promote only the lightweight public payload:

- README/config/manifests;
- CSV histories and telemetry;
- setup/final/comparison PNGs;
- small MP4/GIF review media;
- comparison report and pass/fail summary.

Keep raw trajectories, unpublished scratch paths, and intermediate solver
checkpoints out of public example folders unless a separate release artifact is
explicitly declared.

## README, gallery, and root-doc integration

When promoting an example:

1. update the example `README.md`;
2. update `examples/PUBLIC_EXAMPLES_CONTRACT.yaml`;
3. update `examples/README.md`;
4. update `docs/example-gallery.md` and gallery thumbnails when the example is
   public-facing;
5. update README/root docs only if the example changes the public capability
   story.

All links should point back to this page for the full contract.

## Capability-boundary wording

Use precise labels:

- "production" for tested public-facing workflows;
- "beta" for useful but capability-limited workflows;
- "diagnostic" for numerical-method evidence;
- "scaffold" for API or config surfaces that exist but are not validated.

Avoid broad claims such as "general fracture solver", "validated for all
geometries", or "GPU production path" unless the relevant benchmark,
comparison, and performance evidence exists.

## Tests and drift checks

Promotion is not complete until tests or drift checks cover the contract:

- public example manifest checks;
- README command checks;
- artifact existence checks;
- visual manifest checks;
- docs link/drift checks;
- release export checks to exclude unpublished archives and heavy raw stores;
- HPC drift checks when helper scripts are mirrored in `lab-infra`.

At minimum, the canonical contract page must remain linked from
`examples/README.md`, `docs/output_standards/index.md`, and the public docs
index.

## Promotion checklist

Before declaring an example promoted:

- [ ] local `--validate-only` passes;
- [ ] local or HPC run completes;
- [ ] `README.md` documents setup, command, outputs, and claim boundary;
- [ ] `config.yaml` or script contract is present;
- [ ] manifests and metadata are present;
- [ ] required CSVs are present;
- [ ] setup and final-state visuals are present;
- [ ] comparison artifacts are present when a reference exists;
- [ ] visual files use STIX-style typography and review-safe dimensions;
- [ ] heavy or unpublished files are excluded;
- [ ] public contract YAML and docs are updated;
- [ ] issue or review thread records the promotion evidence.
