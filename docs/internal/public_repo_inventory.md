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
├── configs/                 schemas, reference YAML, benchmark configurations
├── docs/                    Sphinx documentation source
├── examples/                runnable public examples and retained visuals
├── notebooks/               lightweight notebook quickstarts
├── src/phast/               solver package
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

Tracked-file count at the time of this audit: 581 files.

## File Category Coverage

| Category | Files | Public explanation |
|---|---:|---|
| `examples/dynamic/` | 131 | README, example gallery, YAML workflow, dynamic supported-workflow page. |
| `examples/quasistatic/` | 51 | README, example gallery, quasi-static supported-workflow page. |
| `examples/solid_mechanics_beta/` | 73 | README, example gallery, solid-mechanics supported-workflow page. |
| `examples/plasticity_interface_beta/` | 88 | Beta supported-workflow page and capability matrix. |
| `examples/` root | 3 | Public examples overview and contract. |
| `src/phast/` | 116 | Python API, public API reference, YAML workflow, results API, API notes. |
| `docs/` | 69 | Sphinx toctree and internal release-audit material. |
| `configs/` | 20 | YAML workflow, configuration guide, benchmark catalogue. |
| `assets/` | 9 | README visuals, showcase page, internal assets provenance note. |
| `.github/` | 7 | Contribution/community pages and this inventory. |
| Root metadata/scripts | 13 | README, installation script, contribution docs, `llms.txt`, `.cursorrules`, citation, license, requirements, and package metadata. |

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
├── solid_mechanics_beta/
│   ├── j2_bar/
│   ├── linear_plate/
│   └── neohookean_plate/
└── plasticity_interface_beta/
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
| Fixed | `configs/` | Top-level aliases and diagnostic-only configs were removed; public docs now use canonical `configs/benchmarks/...` paths. |
| Fixed | Examples | Public example manifests were trimmed to retained artifacts and lightweight GIF outputs. |
| Fixed | Root/CI | Public docs and CI no longer require an absent top-level `tests/` directory. |
| Fixed | Root/CI | The contribution template now uses the current public validation checks. |
| Fixed | Packaging | `pyproject.toml` discovers only the public `phast*` package namespace. |
| Medium | Docs | Some excluded scoping documents remain in the repo and should stay out of the public build or be rewritten as public notes. |
| Fixed | Docs/API | Public docs use current validation commands and avoid old packaging/test promises. |
| Fixed | Assets | Public asset inventory now uses public-safe provenance wording, and unused assets were removed. |
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
