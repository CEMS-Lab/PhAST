# Inverse Problems Beta

**Status: documentation scaffold.** This folder records the requirements for a
possible future inverse-analysis example based on differentiable PhAST forward
operations. It does not contain a runnable inverse benchmark.

The current public release foregrounds forward phase-field fracture examples.
Inverse workflows should be added here only when they include:

- a forward configuration or fluent setup that can be validated from the
  repository root;
- a clearly defined observation, loss function, and recovered parameter;
- retained lightweight outputs such as loss history, parameter history, final
  comparison plots, and manifests;
- a README explaining the claim boundary and expected runtime;
- reproducibility notes identifying the PhAST commit, PyTorch version, device,
  and random seeds where applicable.

Do not add raw trajectory stores, restricted calibration data, unpublished paper
artifacts, or large optimizer checkpoints to this folder. Store heavy data
outside git and provide a stable, citable archive link when appropriate.

For now, use the forward examples under `examples/dynamic/` and
`examples/quasistatic/` as the public reproducibility surface.
