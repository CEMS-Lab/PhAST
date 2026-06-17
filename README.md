<p align="center">
  <img src="assets/phast-banner.png" alt="PhAST logo" width="640">
</p>

<h1 align="center">PhAST</h1>

<p align="center">
  <strong>Phase-field Autograd Solver in Torch</strong><br>
  A PyTorch-native differentiable finite element framework for phase-field fracture mechanics.
</p>

<p align="center">
  <a href="https://cems-lab.github.io/PhAST/">Documentation</a> |
  <a href="docs/getting-started.md">Getting Started</a> |
  <a href="#quickstart">Quickstart</a> |
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

PhAST is a PyTorch-native finite-element solver for 2D phase-field fracture, explicit dynamics, and solid mechanics. It embeds phase-field fracture workflows within familiar PyTorch execution graphs, enabling mechanics, damage evolution, and post-processing pipelines to run as inspectable tensor operations.

The public release provides robust support for brittle phase-field fracture and foundational solid mechanics. Advanced implementations covering plasticity, cohesive-interfaces, and PF-CZM capabilities are detailed in the [capability matrix](docs/user_guide/capability_matrix.md).

Models can be authored programmatically via the fluent `phast.Problem` Python API, or executed declaratively via YAML configurations for batch processing, HPC submission, and exact reproducibility.

## Core Strengths

- **PyTorch-Native Mechanics:** Mechanics, damage evolution, and foundational routines use PyTorch tensors, offering explicit control over device placement (CPU/GPU) and precision.
- **Unified Phase-Field Workflows:** Dynamic impact, crack branching, and quasi-static fracture studies share a consistent algorithmic framework and output schema.
- **Dual Authoring Interfaces:** Formulate models using the `phast.Problem` API for iterative exploration, or deploy declarative YAML configurations for batch execution.
- **Curated Validation Suite:** Promoted examples include load-displacement histories, final damage fields, and strict-parity checks against reference literature.
- **Standardized Post-Processing:** `phast.load_result` automatically handles stored manifests, CSV histories, and Zarr trajectory fields.

## Quickstart

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
pip install -e .

python -m phast doctor
```

Verify the installation by running a foundational solid mechanics case:

```bash
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml --output_dir runs/linear_plate
```

To inspect the structural definition of an input deck before execution:

```bash
python -m phast explain-config examples/quasistatic/notched_holed_plate/config.yaml
```

## Reproducible Workflows

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
    <td align="center"><strong>Kalthoff-Winkler Impact</strong></td>
    <td align="center"><strong>Quasi-Static Fracture</strong></td>
    <td align="center"><strong>Dynamic Crack Branching</strong></td>
  </tr>
</table>

| Simulation Category | Execution Command | Expected Artifacts |
|---|---|---|
| **Dynamic Crack Branching** | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml` | Curated damage fields, kinetic energy histories, metadata, and visual summaries. |
| **Dynamic Fracture** | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml` | Explicit crack propagation states, CSV histories, damage plots, and Zarr trajectory outputs. |
| **Quasi-Static Fracture** | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml` | Load-displacement response curves, final phase-field damage, and parity comparison artifacts. |
| **Solid Mechanics** | `python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml` | Mesh-level FEA fields, nodal displacements, visual manifests, and structured metadata. |

Browse the full [example gallery](docs/example-gallery.md) for the complete list of runnable YAML-first benchmarks.

## Documentation & API

| Objective | Interface | Documentation Link |
|---|---|---|
| Author a forward model | Fluent `phast.Problem` API | [Python API](docs/user_guide/python_api.md) |
| Execute public benchmarks | Declarative `config.yaml` | [YAML Workflow](docs/user_guide/yaml_workflow.md) |
| Post-process simulation data | `phast.load_result(path)` | [Public API Reference](docs/user_guide/public_api_reference.md) |
| Review supported physics | Verification Matrix | [Capability Matrix](docs/user_guide/capability_matrix.md) |

### Programmatic Authoring

```python
import phast

problem = (
    phast.Problem("linear plate")
    .geometry("structured_grid", nx=40, ny=12, length=1.0, height=0.2)
    .region("body", kind="domain")
    .material("steel", model="solid_mechanics", region="body", E=2.1e11, nu=0.3)
    .analysis_step("load", kind="solid_mechanics", controls={"tip_force_y": -1.0e3})
    .solver("solid_mechanics", example="solid_mechanics.linear_plate")
    .outputs(fields=["displacement", "von_mises"], histories=["response"], plots=True)
)

spec = problem.to_spec()
```

### Result Inspection

```python
import phast

result = phast.load_result("runs/linear_plate")
print(result.metadata())
print(result.visuals())
print(result.history_names())
```

## Repository Map

| Path | Purpose |
|---|---|
| `src/phast/` | Core PyTorch solver packages, mechanics/damage kernels, and CLI entry points. |
| `examples/` | Publicly validated simulations featuring declarative YAML configurations and visual references. |
| `configs/` | JSON schema definitions and reusable configuration templates. |
| `docs/` | Comprehensive Sphinx documentation, capability matrices, and architectural guides. |
| `assets/` | Lightweight visual assets for repository documentation. |
| `tools/` | Maintenance utilities for regenerating documentation artifacts. |
| `.github/` | Issue templates, Pull Request guidelines, and CI/CD Action workflows. |
| `llms.txt`, `.cursorrules` | Agent-facing repository orientation and rigorous coding constraints. |

## Build The Docs

```bash
pip install -r requirements-docs.txt
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

Hosted documentation is published at <https://cems-lab.github.io/PhAST/>.

## Citation

Use [`CITATION.cff`](CITATION.cff) when referencing the PhAST solver framework. Links to specific technical manuscripts will be updated as accompanying articles become public.

## Acknowledgments and AI Usage

The theoretical formulations, phase-field continuum equations, constitutive assumptions, and numerical discretization choices in PhAST are derived from the established computational solid mechanics literature and were selected, interpreted, and validated by the human authors, as described in the associated article and documentation. AI coding assistants, including Codex, GitHub Copilot, and Claude/Gemini-class tools, were used as auxiliary software engineering aids for tasks such as Python boilerplate generation, script refactoring, docstring and documentation formatting, unit-test scaffolding, and data-pipeline organization. These tools did not formulate the physics or define the solver claims. The authors reviewed and verified the computational mechanics kernels, benchmark configurations, and validation artifacts, and take full responsibility for the correctness, limitations, and scientific content of the codebase.

PhAST is also organized with modern agent-assisted scientific computing in mind. Machine-readable manifests, structured result metadata, headless CLI/API entry points, and repository-level guidance files such as `llms.txt` and `.cursorrules` are provided so human researchers and their software agents can inspect, reproduce, and extend simulations without relying on hidden local state. These files are engineering aids; the scientific claims and solver validity remain governed by the documented formulations, tests, and validation artifacts above.
