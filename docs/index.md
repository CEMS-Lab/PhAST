# PhAST

**Phase-field Autograd Solver in Torch**

<img class="phast-hero-brand" src="phast-banner.png" alt="PhAST logo">

<div class="phast-hero-panel">
  <div>
    <p class="phast-eyebrow">CEMS Lab · PyTorch-native FEM workflows</p>
    <h2>Matrix-free, differentiable phase-field fracture and FEM benchmarks in PyTorch.</h2>
    <p>
      <b>What is PhAST?</b> PhAST is a Python library built on PyTorch that allows you to solve phase-field fracture problems without assembling large global matrices. It is designed to be differentiable, making it easier to combine with machine learning techniques.
      <br><br>
      <i>(New to phase-field modeling? Read our <a href="tutorial/01_phase_field_primer.html">Phase-Field Primer</a> and the <a href="tutorial/02_visual_glossary.html">Visual Glossary</a> to learn the basics).</i>
    </p>
    <p>
      Use the fluent <code>phast.Problem</code> API to author new models. Use YAML
      configurations for public examples, reproducibility, batch runs, and exact
      reruns of published simulations.
    </p>
    <p class="phast-hero-links">
      <a class="phast-button" href="getting-started.html">Get started</a>
      <a class="phast-button phast-button-secondary" href="example-gallery.html">View examples</a>
      <a class="phast-button phast-button-secondary" href="user_guide/capability_matrix.html">Capability matrix</a>
    </p>
  </div>
  <div class="phast-command-card">
    <p class="phast-command-title">Run a first check</p>
    <pre><code>pip install -e .
python -m phast doctor
python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml --validate-only
python -m phast run examples/quasistatic/miehe_tension/config.yaml --validate-only</code></pre>
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
    <p>Public examples include YAML inputs, setup figures, final fields, response
    histories, manifests, and compact animations.</p>
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
| [Getting started](getting-started.md) | Installation, `phast doctor`, validation, first run, and result inspection. |
| [Verify install](verify-install.md) | Backend visibility, expected doctor output, and a schema-validation smoke test. |
| [User guide](user_guide/overview.md) | Problem setup, YAML, Python API, physics, meshes, sparse solves, and result APIs. |
| [Example gallery](example-gallery.md) | Runnable fracture, solid-mechanics, and beta validation examples with visual outputs. |
| [Performance and reproducibility](performance-reproducibility.md) | Device choice, backend policy, timing evidence, and `torch.compile` reporting. |
| [Community](community.md) | Issues, discussions, maintainer review, and contribution route. |

## For New Users: Your First 15 Minutes

If you are new to PhAST, we recommend following this path:
1. **Learn the Basics:** Read the "What is PhAST?" summary above, then follow the [Phase-Field Primer](tutorial/01_phase_field_primer.md) and [Visual Glossary](tutorial/02_visual_glossary.md).
2. **Install:** Follow the [Getting Started](getting-started.md) guide.
3. **Run a Simple Example:** Run `python -m phast run examples/quasistatic/miehe_tension/config.yaml` to see a result quickly.
4. **Understand the API:** Read the [Python API](user_guide/python_api.md) to understand how the models are defined.
5. **Check Capabilities:** Review the [Capability Matrix](user_guide/capability_matrix.md) to ensure your target problem is supported.

## Which Path Should I Use?

| Goal | First page | Stable surface |
|---|---|
| Install and run a first case | [Getting started](getting-started.md) | `python -m phast doctor` |
| Author new models | [Python API](user_guide/python_api.md) and [Setting up problems](user_guide/setup_problems.md) | `phast.Problem` |
| Reproduce or batch-run examples | [YAML workflow](user_guide/yaml_workflow.md) | `python -m phast run config.yaml` |
| Inspect completed runs | [Public API reference](user_guide/public_api_reference.md) | `phast.load_result(path)` |
| Browse runnable examples | [Example gallery](example-gallery.md) | flat public example folders |
| Diagnose failed runs | [Troubleshooting](troubleshooting.md) | units, mesh, backend, and output checks |
| Check supported physics | [Capability matrix](user_guide/capability_matrix.md) | production / beta / scaffold labels |

## Workflow In One Line

`YAML / phast.Problem` -> `Mesh` -> `Operators` -> `Solver` -> `Result bundle`

```{toctree}
:maxdepth: 2
:caption: Get Started
getting-started
verify-install
user_guide/public_api_reference
agent-contribution-guide
citing
troubleshooting
tutorial/index
tutorial/01_phase_field_primer
tutorial/02_visual_glossary
tutorial/04_exploration_experiments
user_guide/capability_matrix
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/overview
user_guide/setup_problems
user_guide/python_api
user_guide/yaml_workflow
user_guide/physics
user_guide/configuration
user_guide/meshes
user_guide/example_contract
user_guide/capability_matrix
user_guide/sparse_solve
```

```{toctree}
:maxdepth: 2
:caption: Supported Workflows

supported_workflows/solid_mechanics
supported_workflows/quasistatic_fracture
supported_workflows/dynamic_fracture
supported_workflows/plasticity_interface_beta
supported_workflows/unsupported_experimental
```

```{toctree}
:maxdepth: 2
:caption: Example Gallery

example-gallery
```

```{toctree}
:maxdepth: 2
:caption: API Reference

user_guide/public_api_reference
api/public_workflow
api/sparse_solve
api/time_integrators
api/mixed_precision_cg
api/adaptive
```

```{toctree}
:maxdepth: 1
:caption: Performance & Reproducibility

performance-reproducibility
```

```{toctree}
:maxdepth: 1
:caption: Community

community
```
