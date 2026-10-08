# Configuration authoring rules

These rules are mandatory for new public YAML examples, their READMEs, and
tutorials. Use them with [DOCUMENTATION_STYLE.md](DOCUMENTATION_STYLE.md) and
the [example contract](docs/user_guide/example_contract.md). Write for an
undergraduate who knows forces and displacements but is new to PhAST.

## One layout, supported execution routes

Use `schema_version: 2` for new standard-workflow teaching problems when the
required execution adapter exists and has passed its checks. Keep this
top-level order, omitting only genuinely unused optional sections:

| Order | Key | Question answered |
|---|---|---|
| 1 | `schema_version` | Which input format is this? |
| 2 | `name` | What problem is being solved? |
| 3 | `reference` | What is the source or teaching scope? |
| 4 | `geometry` | What domain and mesh are used? |
| 5 | `regions` | Which elements and nodes belong to each selection? |
| 6 | `materials` | What properties does each material have? |
| 7 | `assignments` | Which material belongs to each element region? |
| 8 | `initial_conditions` | What starting state or maintained damage is prescribed? |
| 9 | `boundary_conditions` | What is restrained or loaded? |
| 10 | `analysis_steps` | What physical analysis and loading are requested? |
| 11 | `solver` | How is the analysis solved numerically? |
| 12 | `outputs` | Where are histories, fields, and visuals written? |

Use actual parser keys, two-space indentation, descriptive region names, and
brief section comments. In schema 2, do not substitute the legacy singular
`material`, `loading`, or `output` sections. Put units and the effect of
changing a value next to the input. Label partial YAML excerpts as partial;
use a complete checked-in file for runnable instructions.

A common layout is not a promise that every solver supports every section.
Check the [capability matrix](docs/user_guide/capability_matrix.md). The
documented multi-material route is a CPU float64, structured-T3 rectangle,
quasi-static Amor AT2 model with elementwise `E` and `Gc`, shared `nu`
and `l0`, and Dirichlet conditions. It is not dynamic multi-material
fracture or an independent interface law.

Keep existing schema-1 benchmark inputs explicitly labelled. Do not migrate
frozen course assets or separate 3D research merely to make inputs uniform.
Unsupported routes remain separate until an adapter and evidence exist.

## Explain editable choices

- Separate geometry from mesh resolution. Name the boundary, crack, and
  material selectors that must change when dimensions or mesh change.
- Explain `E` as elastic stiffness and `Gc` as fracture resistance.
  A tougher region need not be stiffer; vary one property at a time.
- State a consistent unit system. A `units` label does not repair
  inconsistent bare numbers. Dynamics requires meaningful density and time.
- Separate physical analysis from integration. Quasi-static analysis omits
  inertia; dynamics includes inertia, and explicit integration needs a
  CFL-limited time step.
- Separate final physical time, step limits, load increments, solver
  iterations, and output cadence. Use the selected adapter's actual keys.
- Define the mesh measure in `h/l0`. Neither one resolution ratio nor a
  smaller time step is a universal convergence or physical-validation test.

## Use the same command sequence

Run from the installed repository root. Replace `CONFIG` with the YAML
path and `RUN` with a new output directory; neither is a literal filename.

```bash
python -m phast explain-config CONFIG
python -m phast run CONFIG --validate-only
python -m phast run CONFIG --output_dir RUN
```

Explain, check the supported input contract, then solve. Confirm all three
commands for the selected schema before promoting the example. Do not
present legacy `precheck --config` as a universal schema-2 command.

## README and tutorial minimum

Use this sequence: physical question; prerequisites and installation; exact
commands; input-section map; safe edits; outputs; evidence and limitations;
help. Source-install instructions require Python 3.10 or newer, separate
macOS/Linux and Windows environment commands, editable installation, and
the optional dependency route using `requirements.txt`. Do not claim a
checkout is the latest release.

The main teaching sequence is the small single-material quasi-static SENT,
small single-material dynamic SENT, and full layered DCB configuration.
Identify their execution-check status. A setup exercise need not propagate
a crack; initial `damage = 1` is not new growth. Distinguish algebraic
convergence, discretisation checks, matched controls, independent scientific
review, and experimental validation.

Compact complete comparison inputs are permitted duplication of model data,
not duplicated solver code. Keep the same section order, change only the
declared comparison inputs, use the common CLI and separate output directories,
and state what is held fixed. Do not add a comparison-specific solver driver.

When trajectory saving is requested, default to HDF5 (`training_data.h5`).
Preserve existing opt-in behaviour and keep large raw stores out of the public
payload. Show retained PNG/GIF assets with captions and a static fallback;
use repository-relative MyST figure paths that Sphinx can collect, not
unverified public raw URLs.

Before publication, record exact command checks and the documentation build;
identify checks not run. End substantial tutorials with a
[GitHub issue invitation](https://github.com/CEMS-Lab/PhAST/issues/new/choose)
requesting the config, command, OS, Python/PhAST version, and full error.
