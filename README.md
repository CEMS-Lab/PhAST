<p align="center">
  <img src="assets/phast-banner.png" alt="PhAST logo" width="640">
</p>

<h1 align="center">PhAST: A matrix-free, differentiable PyTorch Solver for Phase-Field Fracture</h1>

<p align="center">
  <strong>Phase-field Autograd Solver in Torch</strong><br>
  A matrix-free, differentiable PyTorch solver for phase-field fracture and FEM benchmarks.
  <br><br>
  <strong>PhAST is a matrix-free, differentiable PyTorch solver for phase-field fracture.</strong>
</p>

<p align="center">
  <a href="https://cems-lab.github.io/PhAST/">Documentation</a> |
  <a href="docs/getting-started.md">Getting Started</a> |
  <a href="#quickstart">Quickstart</a> |
  <a href="docs/example-gallery.md">Examples</a> |
  <a href="docs/community.md">Community</a> |
  <a href="CITATION.cff">Citation</a> |
  <a href="https://github.com/CEMS-Lab/PhAST/releases">Releases</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.23458"><img alt="arXiv:2606.23458" src="https://img.shields.io/badge/arXiv-2606.23458-b31b1b"></a>
  <img alt="Python 3.10-3.12" src="https://img.shields.io/badge/python-3.10--3.12-3776ab">
  <img alt="PyTorch 2.0+" src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c">
  <img alt="License" src="https://img.shields.io/github/license/CEMS-Lab/PhAST">
  <img alt="Docs" src="https://img.shields.io/badge/docs-GitHub%20Pages-f97316">
  <img alt="Package" src="https://img.shields.io/badge/package-pyproject.toml-0f766e">
</p>

---

## What is PhAST?

PhAST is a research finite-element solver implemented in PyTorch for
two-dimensional phase-field fracture. Its principal dynamic pathway evaluates
finite-element operators through tensor gather-compute-scatter operations
without retaining a global stiffness matrix. Selected operations remain
compatible with PyTorch autograd, subject to the documented limitations of
nonsmooth history updates, bounds, active sets, and optional sparse backends.

*(New to phase-field modeling? Read our [Phase-Field Primer](docs/tutorial/01_phase_field_primer.md) to learn the basics).*

Models can be authored programmatically through the `phast.Problem` Python API
or executed from YAML configurations. YAML is the reference format for shared
examples because it records geometry, materials, boundary conditions, solver
controls, and requested outputs in one reviewable file.

## Core Strengths

- **Matrix-Free Operators:** Explicit fracture mechanics and damage updates use operations on PyTorch tensors without persistent global stiffness assembly on the main dynamic path.
- **Differentiable Mechanics:** Supported tensor operations remain compatible
  with PyTorch autograd where documented, enabling carefully interpreted
  sensitivity studies.
- **Phase-Field Fracture Focus:** Dynamic impact, crack branching, and quasi-static fracture workflows share a consistent mechanics/damage formulation and output schema.
- **Public Benchmark Bundles:** Public examples provide `config.yaml`, setup figures, final field plots, response histories, manifests, and compact animations. Numerical fields are reloadable only when the result bundle retains a trajectory store.
- **YAML Plus Fluent API:** Use declarative YAML for reproducible runs and `phast.Problem` for programmatic model authoring.
- **Standardized Post-Processing:** `phast.load_result` handles stored manifests, CSV histories, visuals, and retained trajectory fields.

## How The Solver Works

`YAML / phast.Problem` -> `Mesh` -> `Operators` -> `Solver` -> `Result bundle`

For a phase-field fracture run, PhAST constructs or imports a two-dimensional
finite-element mesh, evaluates the mechanical state, updates the tensile
history field, solves the regularized damage problem, enforces damage bounds
and irreversibility, and writes fields, histories, manifests, and provenance.
Explicit dynamics and quasi-static fracture use different mechanics updates;
the [solver overview](docs/user_guide/overview.md) and
[formulation guide](docs/user_guide/physics.md) describe both pathways.

## For New Users

If you are new to PhAST, follow this sequence:

1. Read the [phase-field primer](docs/tutorial/01_phase_field_primer.md) if the
   formulation is new to you.
2. Follow the [installation and first-run guide](docs/getting-started.md).
3. Validate a fracture YAML before allocating a full simulation.
4. Run the small linear-elastic example to verify end-to-end execution and
   result loading.
5. Consult the [capability matrix](docs/user_guide/capability_matrix.md) before
   selecting a model for research use.

## Quickstart

```bash
git clone https://github.com/CEMS-Lab/PhAST.git
cd PhAST
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

python -m phast doctor
```

PhAST currently supports Python 3.10-3.12; Python 3.11 is recommended for a
first source installation.

Validate a public fracture configuration without launching a full solve:

```bash
python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --validate-only
```

`--validate-only` checks the schema and semantic consistency of the input. It
does not run the solver or establish mesh convergence, benchmark reproduction,
or physical validity.

Expected validation output:

```text
OK: examples/dynamic/B2_kalthoff_winkler/config.yaml passes schema validation.
```

Inspect the parsed problem definition before execution:

```bash
python -m phast explain-config examples/quasistatic/notched_holed_plate/config.yaml
```

Run a compact end-to-end mechanics example and inspect its output:

```bash
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml \
  --output_dir runs/linear_plate
```

```python
import phast

result = phast.load_result("runs/linear_plate")
print(result.metadata())
print(result.history_names())
print(result.visuals())
```

PhAST itself does not require a separate CMake build. Optional PETSc/MUMPS,
AmgX, cuDSS, and other platform-specific backends are not required for this
first workflow.

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
| **Dynamic Crack Branching** | `python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml` | Damage fields, kinetic-energy histories, metadata, and visual summaries. |
| **Dynamic Fracture** | `python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml` | Crack-propagation states, CSV histories, damage plots, and optional trajectory outputs. |
| **Quasi-Static Fracture** | `python -m phast run examples/quasistatic/notched_holed_plate/config.yaml` | Load-displacement response curves, final phase-field damage, and comparison artifacts. |
| **Solid Mechanics Beta** | `python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml` | Mesh-level FEA fields, nodal displacements, visual manifests, and structured metadata. |

Browse the full [example gallery](docs/example-gallery.md) for the complete list of runnable examples. Beta examples are provided for inspection, but they have not yet been validated as extensively as the included fracture benchmarks.

## Documentation & API

| Objective | Interface | Documentation Link |
|---|---|---|
| Author a forward model | Fluent `phast.Problem` API | [Python API](docs/user_guide/python_api.md) |
| Execute public benchmarks | Declarative `config.yaml` | [YAML Workflow](docs/user_guide/yaml_workflow.md) |
| Post-process simulation data | `phast.load_result(path)` | [Public API Reference](docs/user_guide/public_api_reference.md) |
| Review supported physics | Capability matrix | [Capability Matrix](docs/user_guide/capability_matrix.md) |
| Learn step-by-step setup | Tutorial notebook | [Problem Setup Walkthrough](docs/tutorial/problem_setup_walkthrough.ipynb) |
| Add an audited learned damage model | Predictor plug-in protocol | [Modular FEM and Learned Damage](docs/tutorial/03_modular_fem_and_learned_damage.md) |
| Learn elementwise `E(x)` and `Gc(x)` fields | Script-contract teaching example | [Heterogeneous Material Fields](docs/tutorial/05_heterogeneous_material_fields.md) |
| Diagnose failed runs | Troubleshooting guide | [Troubleshooting](docs/troubleshooting.md) |

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
| `examples/` | Runnable examples, their YAML inputs, and lightweight reference outputs. |
| `configs/` | Runnable benchmark decks, the YAML reference template and schema, and explicitly labelled reproducibility contracts. |
| `docs/` | Sphinx documentation, capability matrices, tutorials, and user guides. |
| `assets/` | Lightweight visual assets for repository documentation. |
| `tools/` | Maintenance utilities for documentation and release checks. |
| `.github/` | Issue templates, Pull Request guidelines, and CI/CD Action workflows. |
| `AGENTS.md`, `llms.txt`, `.cursorrules` | Agent-facing contribution guidance and repository orientation. |

## Contributing

Contributions are welcome for solver kernels, example cases, validation scripts,
post-processing utilities, documentation, and performance improvements. Start
with [CONTRIBUTING.md](CONTRIBUTING.md), then use the [capability matrix](docs/user_guide/capability_matrix.md)
and [example contract](docs/user_guide/example_contract.md) to keep public
claims, examples, and artifacts consistent.

Students, researchers, scientific-software developers, and users evaluating
PhAST are invited to review the code and documentation, propose reproducible
examples, and report unclear instructions. If you become stuck at any point,
[open an issue](https://github.com/CEMS-Lab/PhAST/issues/new/choose). A question
about installation or usage is a valid issue and helps improve the documentation
for subsequent users.

Agent-assisted contributions are also supported. Guidance lives in
[AGENTS.md](AGENTS.md), [llms.txt](llms.txt), [.cursorrules](.cursorrules), and
[docs/agent-contribution-guide.md](docs/agent-contribution-guide.md). These
files are intended to help contributors improve the solver and documentation
without inventing benchmark results, capabilities, or paper metadata.

## Build The Docs

```bash
pip install -r requirements-docs.txt
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

Hosted documentation is published at <https://cems-lab.github.io/PhAST/>.

Release history and version-specific notes are maintained in
[GitHub Releases](https://github.com/CEMS-Lab/PhAST/releases), which is the
project changelog referenced by the package metadata.

## Citation

If PhAST contributes to your research, please cite the associated arXiv
manuscript and the software repository metadata in [`CITATION.cff`](CITATION.cff).

The Sphinx documentation includes a short [how-to-cite page](docs/citing.md)
with a repository BibTeX entry and reproducibility notes.

```bibtex
@misc{ani2026phast,
  title={A matrix-free, differentiable PyTorch solver for phase-field fracture: Formulation, benchmarks, and inverse analysis},
  author={Ani, Allamaprabhu and Molinari, Jean-François and Subhash, Ghatu and Ponnusami, Sathiskumar Anusuya},
  year={2026},
  eprint={2606.23458},
  archivePrefix={arXiv},
  primaryClass={cs.CE},
  url={https://arxiv.org/abs/2606.23458}
}
```

Official code for the manuscript is hosted in this repository:
<https://github.com/CEMS-Lab/PhAST>.

## Acknowledgments

The theoretical formulations, phase-field continuum equations, constitutive assumptions, and numerical discretization choices in PhAST are derived from the established computational solid mechanics literature and were selected, interpreted, and validated by the human authors, as described in the associated article and documentation. AI coding assistants, including Codex, Claude, Gemini, and GitHub Copilot, were used as auxiliary software-engineering tools for repository organization, documentation editing, boilerplate generation, and code-review support; they did not define the physics, benchmark claims, validation criteria, or scientific conclusions. The authors reviewed and verified the computational mechanics kernels, benchmark configurations, and validation artifacts, and take full responsibility for the correctness, limitations, and scientific content of the codebase.

PhAST is organized with reproducible scientific computing in mind. Machine-readable manifests, structured result metadata, headless CLI/API entry points, and repository-level guidance files are provided so researchers can inspect, reproduce, and extend simulations without relying on hidden local state. These files are engineering aids; the scientific claims and solver validity remain governed by the documented formulations, tests, and validation artifacts above.
