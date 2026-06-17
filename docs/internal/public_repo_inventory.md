# Public Repository Inventory

This page records the public repository structure and the current documentation
coverage for each file category. It is intended as a release-hygiene checklist:
new public files should either fit one of these categories or update this page.

## Top-Level Tree

The public repository is organized as:

```text
.
├── .github/                 issue templates, PR template, workflows, CODEOWNERS
├── assets/                  lightweight README/docs media
├── conda-recipe/            experimental conda packaging recipe
├── configs/                 schemas, reference YAML, benchmark configurations
├── docs/                    Sphinx documentation source
├── examples/                runnable public examples and retained visuals
├── notebooks/               lightweight notebook quickstarts
├── src/phast/               solver package
├── tools/                   small public media/maintenance utilities
├── .cursorrules             agent coding constraints
├── llms.txt                 agent repository orientation
├── pyproject.toml           package metadata and CLI entry points
├── requirements*.txt        dependency lists
├── install.sh               convenience installer
├── README.md                public landing page
├── LICENSE                  license
├── CITATION.cff             citation metadata
├── CONTRIBUTING.md          contribution guide
└── docs/internal/ROADMAP.md internal roadmap
```

Tracked-file count at the time of this audit: 968 files.

## File Category Coverage

| Category | Files | Public explanation |
|---|---:|---|
| `examples/dynamic/` | 188 | README, example gallery, YAML workflow, dynamic supported-workflow page. |
| `examples/quasistatic/` | 50 | README, example gallery, quasi-static supported-workflow page. |
| `examples/solid_mechanics_beta/` | 188 | README, example gallery, solid-mechanics supported-workflow page. |
| `examples/plasticity_interface_beta/` | 298 | Beta supported-workflow page and capability matrix. |
| `examples/` root | 3 | Public examples overview and contract. |
| `src/phast/` | 116 | Python API, public API reference, YAML workflow, results API, API notes. |
| `docs/` | 56 | Sphinx toctree and this inventory page. |
| `configs/` | 32 | YAML workflow, configuration guide, benchmark catalogue. |
| `assets/` | 11 | README visuals, showcase page, internal assets provenance note. |
| `.github/` | 9 | Contribution/community pages and this inventory. |
| `conda-recipe/` | 3 | Packaging notes and this inventory. |
| Root metadata/scripts | 15 | README, installation, contribution docs, `llms.txt`, `.cursorrules`, this inventory. |

## Source Package Map

```text
src/phast/
├── __init__.py, __main__.py       public imports and CLI dispatch
├── core/                          mesh, geometry, FEM operators, Problem API
├── physics/                       material, BC, fracture, initial/history updates
├── solvers/                       damage, mechanics, sparse solve, multigrid, time integration
├── config/                        config dataclasses, validation, schema, run/explain CLI
├── workflow/                      validated workflow contracts and adapters
├── utils/                         doctor, IO, metrics, provenance, units, visualization
├── cohesive_elements/             cohesive law/operator helpers
├── plasticity/                    J2 plasticity helpers
├── solid_mechanics_runners/       promoted solid-mechanics example runners
└── training/                      lightweight training/data helpers
```

The stable public entry points for new users are `phast.Problem`,
`phast.load_result`, `phast.inspect_mesh`, and the CLI commands documented in
the getting-started and YAML workflow pages. Lower-level exported helpers are
available for advanced users, but they should not be treated as the primary
researcher API unless a dedicated docs page describes them.

## Example Tree

```text
examples/
├── dynamic/
│   ├── B2_kalthoff_winkler/
│   ├── B3_dynamic_sent/
│   ├── B5_pmma_branching/
│   ├── B6_perforated_10holes/
│   ├── B6_perforated_1hole_far/
│   ├── B6_perforated_1hole_near/
│   ├── B6_perforated_30holes/
│   └── B7_dynamic_crack_branching_comsol/
├── quasistatic/
│   ├── miehe_tension/
│   └── notched_holed_plate/
├── solid_mechanics/
│   ├── generalized_alpha_oscillator/
│   ├── j2_bar/
│   ├── linear_plate/
│   ├── mixed_precision_cg/
│   └── neohookean_plate/
└── plasticity_interface/
    ├── fluent_setups/
    └── results/
```

Public example folders should be flat and reproducible: `README.md`,
`config.yaml`, standard manifests, CSV histories, setup/final plots, and an
animation when the response is time-dependent.

## Multi-Agent Audit Findings

The current documentation pass used separate reviewers for examples/configs,
source/API, docs/fluent workflow, and root metadata/agent files. The main
findings to track are:

| Severity | Area | Finding |
|---|---|---|
| Fixed | `examples/dynamic/B7_dynamic_crack_branching_comsol/` | Public manifests now use a retained-run identifier instead of absolute machine paths. |
| High | `configs/` | Some top-level aliases are broken or diagnostic-only. Remove or replace with public benchmark configurations. |
| High | Examples | Several public manifests reference missing `run_metadata.json`, `run_lockfile.json`, or MP4 artifacts. Regenerate or update manifests. |
| High | Root/CI | Public docs and CI mention `tests/`, but this snapshot has no top-level `tests/` directory. Either restore public tests or remove those commands from public workflows. |
| High | Root/CI | `CHANGELOG.md` and `configs/status/` are referenced by contribution templates but absent. Add them or update the templates. |
| Fixed | Packaging | `pyproject.toml` discovers only the public `phast*` package namespace. |
| Medium | Docs | Some excluded scoping documents remain in the repo and should stay out of the public build or be rewritten as public notes. |
| Medium | Docs/API | Some older docs mention unsupported CLI flags or old H5 behavior. Update before publication. |
| Medium | Assets | Some asset provenance text mentions private job/source context. Replace with public-safe provenance. |
| Low | API boundary | `src/phast/__init__.py` exports more low-level helpers than the main docs present as stable researcher API. Clarify stability or reduce exports later. |

## Verification Commands

Use these lightweight checks before merging public documentation or example
changes:

```bash
python -m phast doctor
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --validate-only
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml --validate-only
sphinx-build -W -b html docs docs/_build/html
```

When public tests are restored, add the documented pytest commands back to this
page, `.cursorrules`, and the contribution guide.
