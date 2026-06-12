# Paper 1 Benchmarks

Paper 1 positions PhAST as a matrix-free, autograd-enabled PyTorch solver for
2D phase-field fracture. The public repository keeps the evolving engineering
benchmark table here so the manuscript can stay concise while the reproducible
comparison surface remains visible.

Preprint link: pending arXiv release.

## Why Matrix-Free PyTorch

PhAST keeps the dynamic fracture time-stepping path in tensor operations rather
than assembling global sparse matrices. That design keeps memory use tied to
field and element arrays, makes GPU execution straightforward through PyTorch,
and avoids common serialization issues around external solver objects and
compiled linear-algebra handles. Supported tensor paths can also participate in
autograd-based checks, which makes gradient verification and differentiable
mechanics diagnostics easier to reproduce.

## Timing Snapshot

The Paper 1 timing table is under release refresh. Earlier Akantu numbers may
have been produced with a debug-mode build/configuration, so the public
documentation does not publish those ratios as validated results.

| Benchmark | Public status |
|---|---|
| SENT dynamic timing | Rerun PhAST, Akantu, and FEniCS with locked Release-mode environments before quoting cross-code ratios. |
| Kalthoff-Winkler dynamic timing | Rerun PhAST and FEniCS for the spectral split; include Akantu only for a compatible Amor/vol-dev variant. |
| Quasi-static timing | Keep separate from the Paper 1 explicit-dynamics table. |

## Caveats

- The intended table covers explicit dynamic timing harnesses, not quasi-static
  timing.
- Phase-field subcycling is disabled so every solver performs one damage solve
  per explicit step.
- SENT should use AT2/Amor so Akantu can participate. Kalthoff-Winkler uses the
  Miehe spectral split, so Akantu is omitted there.
- For broader reproducibility commands, use the canonical configs and example
  READMEs linked from this section.
