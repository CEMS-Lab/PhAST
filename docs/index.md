# PhAST: Matrix-Free, Differentiable PyTorch Solver for Phase-Field Fracture

**Phase-field Autograd Solver in Torch**

<img class="phast-hero-brand" src="phast-banner.png" alt="PhAST logo">

<div class="phast-hero-panel">
  <div>
    <p class="phast-eyebrow">CEMS Lab · PyTorch-native FEM workflows</p>
    <h2>PhAST is a matrix-free, differentiable PyTorch solver for phase-field fracture and FEM benchmarks.</h2>
    <p>
      <b>What is PhAST?</b> PhAST is a PyTorch finite-element solver for
      two-dimensional phase-field fracture in explicit dynamics and
      quasi-static mechanics. Its principal dynamic pathway evaluates
      finite-element operators without retaining a global stiffness matrix.
      Selected tensor operations remain compatible with autograd, subject to
      the documented limitations of history updates, active sets, and optional
      sparse backends.
      <br><br>
      <i>(New to phase-field modeling? Read our <a href="tutorial/01_phase_field_primer.html">Phase-Field Primer</a> and the <a href="tutorial/02_visual_glossary.html">Visual Glossary</a> to learn the basics).</i>
    </p>
    <p>
      Use the fluent <code>phast.Problem</code> API to author new models. Use YAML
      configurations for documented examples, reproducibility, batch runs, and
      reviewable reruns of published simulations.
    </p>
    <p class="phast-hero-links">
      <a class="phast-button" href="getting-started.html">Get started</a>
      <a class="phast-button phast-button-secondary" href="example-gallery.html">View examples</a>
      <a class="phast-button phast-button-secondary" href="user_guide/capability_matrix.html">Capability matrix</a>
      <a class="phast-button phast-button-secondary" href="https://github.com/CEMS-Lab/PhAST">Source on GitHub</a>
    </p>
  </div>
  <div class="phast-command-card">
    <p class="phast-command-title">Run a first check</p>
    <p><small>The base source installation does not require a separate PhAST
    compilation step. Optional HPC backends are not required for validation.</small></p>
    <pre><code>pip install -e .
python run_sanitizer.py
python -m phast doctor
python -m phast run examples/solid_mechanics_beta/linear_plate/config.yaml --output_dir runs/linear_plate</code></pre>
  </div>
</div>

<figure class="phast-wide-figure">
  <img src="kalthoff_winkler_long_crack.gif" alt="Kalthoff-Winkler impact crack growth">
  <figcaption>Dynamic fracture showcase: Kalthoff-Winkler impact crack growth.</figcaption>
</figure>

## Core Strengths

<div class="phast-card-grid">
  <div class="phast-card">
    <h3>Matrix-free operators</h3>
    <p>Fracture and damage updates use operations on PyTorch tensors without
    persistent global stiffness assembly on the main dynamic path.</p>
    <p><a href="user_guide/physics.html">Formulation</a></p>
  </div>
  <div class="phast-card">
    <h3>Differentiable mechanics</h3>
    <p>Supported tensor operations remain compatible with PyTorch autograd,
    making forward runs inspectable and extensible for sensitivity studies.</p>
    <p><a href="user_guide/public_api_reference.html">API reference</a></p>
  </div>
  <div class="phast-card">
    <h3>Public benchmark bundles</h3>
    <p>Public examples include YAML inputs, setup figures, final field plots,
    response histories, manifests, and compact animations. Reloadable numerical
    fields require a retained trajectory store.</p>
    <p><a href="example-gallery.html">Example gallery</a></p>
  </div>
  <div class="phast-card">
    <h3>YAML plus fluent API</h3>
    <p>Use declarative YAML for reproducible runs and <code>phast.Problem</code>
    for programmatic model authoring.</p>
    <p><a href="user_guide/python_api.html">Python API</a></p>
  </div>
</div>

## Documentation

| Section | What it covers |
|---|---|
| [Install](install.md) | Recommended source, Conda, and Docker routes. |
| [Getting started](getting-started.md) | Installation, `phast doctor`, validation, first run, and result inspection. |
| [Verify install](verify-install.md) | Environment discovery, sanitizer, configuration preflight, and completed-run checks. |
| [User guide](user_guide/overview.md) | Problem setup, YAML, Python API, physics, meshes, sparse solves, and result APIs. |
| [Example gallery](example-gallery.md) | Runnable fracture, solid-mechanics, and beta validation examples with visual outputs. |
| [Performance and reproducibility](performance-reproducibility.md) | Device choice, backend policy, timing evidence, and `torch.compile` reporting. |
| [Community](community.md) | Issues, maintainer review, and contribution routes. |
| [Source repository](https://github.com/CEMS-Lab/PhAST) | Clone the code, open issues, inspect examples, and contribute through GitHub. |

## For New Users

If you are new to PhAST, follow one continuous route:

1. **Install and diagnose the environment:** Use [Install](install.md), then run the sanitizer and `python -m phast doctor`.
2. **Complete a bounded solve:** Run the linear-plate example and inspect its result directory.
3. **Learn the formulation:** Read the [Phase-Field Primer](tutorial/01_phase_field_primer.md) and [Visual Glossary](tutorial/02_visual_glossary.md).
4. **Construct a model:** Work through the [problem-setup notebook](tutorial/notebook_setup.ipynb) and the [Python API](user_guide/python_api.md).
5. **Check the capability boundary:** Review the [Capability Matrix](user_guide/capability_matrix.md) before selecting a fracture, beta, or experimental route.
6. **Progress to fracture and heterogeneity:** Use the tutorial sequence and example-local READMEs, which state runtime and evidence boundaries.

## Which Path Should I Use?

| Goal | First page | Stable surface |
|---|---|
| Install and run a first case | [Install](install.md) and [Getting started](getting-started.md) | `python run_sanitizer.py` followed by `python -m phast doctor` |
| Author new models | [Python API](user_guide/python_api.md) and [Setting up problems](user_guide/setup_problems.md) | `phast.Problem` |
| Reproduce or batch-run examples | [YAML workflow](user_guide/yaml_workflow.md) | `python -m phast run config.yaml` |
| Inspect completed runs | [Public API reference](user_guide/public_api_reference.md) | `phast.load_result(path)` |
| Browse runnable examples | [Example gallery](example-gallery.md) | flat public example folders |
| Diagnose failed runs | [Troubleshooting](troubleshooting.md) | units, mesh, backend, and output checks |
| Check supported physics | [Capability matrix](user_guide/capability_matrix.md) | supported / beta / experimental / scaffold labels |

## Workflow In One Line

`YAML / phast.Problem` -> `Mesh` -> `Operators` -> `Solver` -> `Result bundle`

For phase-field fracture, this sequence expands to configuration validation,
mesh construction, mechanics update, tensile-history update, bounded damage
solution, irreversibility enforcement, and result/provenance output. See the
[solver overview](user_guide/overview.md) for the algorithmic pathway.

```{toctree}
:maxdepth: 2
:caption: Start

install
getting-started
verify-install
troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Learn

tutorial/index
tutorial/01_phase_field_primer
tutorial/02_visual_glossary
tutorial/notebook_setup
tutorial/notebook_mesh_resolution
tutorial/notebook_retained_results
tutorial/03_modular_fem_and_learned_damage
tutorial/04_exploration_experiments
tutorial/05_heterogeneous_material_fields
```

```{toctree}
:maxdepth: 2
:caption: User Manual

user_guide/overview
user_guide/physics
user_guide/numerical_methods
user_guide/configuration
user_guide/meshes
user_guide/geometry_gallery
user_guide/sparse_solve
```

```{toctree}
:maxdepth: 2
:caption: How-to Guides

user_guide/setup_problems
user_guide/python_api
user_guide/yaml_workflow
user_guide/results_visualization
user_guide/learned_damage
performance-reproducibility
```

```{toctree}
:maxdepth: 2
:caption: Examples

example-gallery
user_guide/example_contract
supported_workflows/solid_mechanics
supported_workflows/quasistatic_fracture
supported_workflows/dynamic_fracture
supported_workflows/plasticity_interface_beta
supported_workflows/unsupported_experimental
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/index
reference/cli
reference/configuration
user_guide/public_api_reference
user_guide/glossary
user_guide/capability_matrix
api/public_workflow
api/sparse_solve
api/time_integrators
api/mixed_precision_cg
api/adaptive
```

```{toctree}
:maxdepth: 1
:caption: Project

community
citing
agent-contribution-guide
```
