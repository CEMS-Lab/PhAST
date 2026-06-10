# Quasistatic Miehe Shear

Canonical artifact folder for the Miehe shear quasistatic benchmark.

Current runnable source: `examples/quasistatic/miehe_shear/`.

## Reference Targets

The default comparator uses `reference_solutions/miehe_sens_load_displacement.csv`,
which peaks at `0.53118 kN` and represents the current Miehe-style strict gate.

For PhaseFieldX-code parity with the local `plot_1712.py` executable setup
(`l0=0.06 mm`), use:

```bash
python examples/quasistatic/miehe_shear/compare.py \
  --run-dir <run> --reference-source phasefieldx-output \
  --report-name compare_phasefieldx_output_report.txt \
  --figure-name compare_phasefieldx_output.png
```

That mode compares against the bundled PhaseFieldX 1712 output
`top.dof`/`bottom.reaction`, which peaks at `0.49468 kN`. The provenance split
is documented in `docs/qs_sens_reference_provenance_2026-05-25.md`. Use the
nondefault output names above when running this as a side parity check so the
promotion artifacts `compare_report.txt` and `compare.png` remain tied to the
default shipped-reference gate.
