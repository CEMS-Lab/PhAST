<p align="center">
  <img src="assets/phast-banner.png" alt="PhAST logo" width="640">
</p>

<h1 align="center">PhAST</h1>

<p align="center">
  <strong>Phase-field Autograd Solver in Torch</strong><br>
  A GPU-accelerated, differentiable phase-field fracture solver built natively on PyTorch.
</p>

<p align="center">
  <a href="https://cems-lab.github.io/PhAST/">Documentation</a> |
  <a href="docs/installation.md">Installation</a> |
  <a href="#quick-start">Quickstart</a> |
  <a href="docs/example-gallery.md">Examples</a> |
  <a href="CITATION.cff">Citation</a>
</p>

<p align="center">
  <img alt="Python 3.10-3.12" src="https://img.shields.io/badge/python-3.10--3.12-3776ab">
  <img alt="PyTorch 2.0+" src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c">
  <img alt="License" src="https://img.shields.io/github/license/CEMS-Lab/PhAST">
  <img alt="Docs" src="https://img.shields.io/badge/docs-GitHub%20Pages-f97316">
  <img alt="Package" src="https://img.shields.io/badge/package-pyproject.toml-0f766e">
</p>

---

PhAST is a PyTorch-native finite-element solver for 2D phase-field fracture,
explicit dynamics, and solid mechanics. It combines phase-field fracture
workflows with familiar PyTorch execution, so mechanics, damage, and
postprocessing pipelines can run through inspectable tensor operations.

The public release focuses on brittle phase-field fracture and foundational
solid mechanics, with advanced plasticity, cohesive-interface, and PF-CZM
capabilities documented in the
[capability matrix](docs/user_guide/capability_matrix.md). Use the fluent
`phast.Problem` API to author models in Python, or use declarative YAML decks
for batch runs, HPC jobs, and shared simulations.

## Core Strengths

- **PyTorch-native mechanics.** Mechanics, damage, and solid-mechanics routines use ordinary PyTorch tensors, making device placement and precision choices explicit.
- **Phase-field fracture workflows.** Dynamic impact, crack branching, quasi-static fracture, and cohesive-interface studies share a consistent solver and output style.
- **Python and YAML entry points.** Author models with the fluent `phast.Problem` API or run declarative YAML decks for batch execution and shared simulations.
- **Animation-led examples.** Curated examples include damage evolution, response histories, final fields, and lightweight visual summaries.
- **Result inspection.** `phast.load_result` reads stored manifests, histories, visuals, mesh metadata, and trajectory fields when present.
- **Documented capabilities.** The capability matrix summarizes supported physics, optional backends, and current feature coverage.

## Quick Start

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
pip install -e .

python -m phast doctor
python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml --validate-only
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only
python -m phast run examples/quasistatic/notched_holed_plate/config.yaml --validate-only
python -m phast run examples/solid_mechanics/linear_plate/config.yaml --output_dir runs/linear_plate
```

Inspect an input deck before launching a solve:

```bash
python -m phast explain-config examples/quasistatic/notched_holed_plate/config.yaml
```

Read an existing result directory:

```python
import phast

result = phast.load_result("runs/linear_plate")
print(result.metadata())
print(result.visuals())
print(result.history_names())
```

## Visual Examples

<table>
  <tr>
    <td align="center" width="33%">
      <img src="assets/kalthoff_winkler_long_crack.gif" alt="Kalthoff-Winkler long crack-growth animation" width="100%">
    </td>
    <td align="center" width="33%">
      <img src="examples/quasistatic/notched_holed_plate/damage_evolution.gif" alt="Notched-holed plate damage evolution" width="78%">
    </td>
    <td align="center" width="33%">
      <img src="assets/b7_crack_branching_evolution.gif" alt="B7 dynamic crack branching damage evolution" width="100%">
    </td>
  </tr>
  <tr>
    <td><strong>Kalthoff-Winkler impact</strong></td>
    <td><strong>Quasi-static fracture</strong></td>
    <td><strong>Dynamic crack branching</strong></td>
  </tr>
</table>

| Workflow | What to run | What you get |
|---|---|---|
| Dynamic crack branching | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml --output_dir runs/B7_dynamic_crack_branching_comsol` | Crack-branching comparison package with curated damage fields, energy outputs, metadata, and visual summaries. |
| Dynamic fracture | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --output_dir runs/B2_kalthoff_winkler` | Explicit crack propagation with generated run metadata, CSV histories, damage plots, and trajectory/provenance outputs. |
| Quasi-static fracture | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml` | Load-displacement response, final damage, comparison-ready artifacts. |
| Solid mechanics | `python -m phast run examples/solid_mechanics/linear_plate/config.yaml` | Mesh-level FEA fields, response curves, visual manifest, result metadata. |

Browse the full [example gallery](docs/example-gallery.md) for runnable
YAML-first examples and visual result bundles.

## Documentation & Workflows

| Goal | Use | Start here |
|---|---|---|
| Author a new forward model | Fluent `phast.Problem` | [Python API](docs/user_guide/python_api.md) |
| Run public examples or submit to HPC | YAML `config.yaml` | [YAML workflow](docs/user_guide/yaml_workflow.md) |
| Inspect a completed run | `phast.load_result(path)` | [Results API](docs/user_guide/results_api.md) |
| Check supported physics and backends | Capability matrix | [Capability matrix](docs/user_guide/capability_matrix.md) |

Minimal Python authoring:

```python
import phast

problem = (
    phast.Problem("linear plate")
    .geometry("structured_grid", nx=40, ny=12, length=1.0, height=0.2)
    .region("body", kind="domain")
    .material("steel", model="solid_mechanics", region="body", E=2.1e11, nu=0.3)
    .analysis_step("load", kind="solid_mechanics", tip_force_y=-1.0e3)
    .outputs(fields=["displacement", "von_mises"], histories=["response"], plots=True)
)

spec = problem.to_spec()
```

Use Python when designing a new model interactively. Use YAML when you want an
exact input deck for sharing, CI, or cluster runs. Both paths write standard
result directories that can be inspected with `phast.load_result(...)`.

Result inspection:

```python
result = phast.load_result("runs/notched_holed_plate")
print(result.metadata())
print(result.visuals())
print(result.history_names())
```

## Repository Map

| Path | Purpose |
|---|---|
| `src/phast/` | Solver package, mechanics/damage kernels, workflow helpers, CLI entry points. |
| `examples/` | Runnable examples with YAML decks, visuals, and result metadata. |
| `configs/` | Benchmark YAMLs, schema files, and reusable configuration templates. |
| `docs/` | Sphinx documentation, gallery, user guide, and capability matrix. |
| `assets/` | Lightweight README and documentation visuals. |
| `tools/` | Small public maintenance utilities for regenerating curated visuals. |

## Build The Docs

```bash
pip install -r requirements-docs.txt
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

Hosted documentation is published at <https://cems-lab.github.io/PhAST/>.

## Citation

Use [`CITATION.cff`](CITATION.cff) when citing PhAST. Paper-specific links will
be updated as manuscripts and archived releases become public.

## Acknowledgments and AI Usage

The theoretical formulations, phase-field continuum equations, constitutive
assumptions, and numerical discretization choices in PhAST are derived from the
established computational solid mechanics literature and were selected,
interpreted, and validated by the human authors, as described in the associated
article and documentation. AI coding assistants, including Codex, GitHub
Copilot, and Claude/Gemini-class tools, were used as auxiliary software
engineering aids for tasks such as Python boilerplate generation, script
refactoring, docstring and documentation formatting, unit-test scaffolding, and
data-pipeline organization. These tools did not formulate the physics or define
the solver claims. The authors reviewed and verified the computational
mechanics kernels, benchmark configurations, and validation artifacts, and take
full responsibility for the correctness, limitations, and scientific content of
the codebase.

PhAST is also organized with modern agent-assisted scientific computing in
mind. Machine-readable manifests, structured result metadata, headless CLI/API
entry points, and repository-level guidance files such as `llms.txt` and
`.cursorrules` are provided so human researchers and their software agents can
inspect, reproduce, and extend simulations without relying on hidden local
state. These files are engineering aids; the scientific claims and solver
validity remain governed by the documented formulations, tests, and validation
artifacts above.
