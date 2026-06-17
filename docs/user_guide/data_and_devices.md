# Training data & device support

H5 training-data format and the device-portability story (CUDA float64,
CPU float64, MPS float32 caveats).


**Note:** Running without any output flags prints a summary of all available
flags.

### Stagger Convergence Criteria

Four convergence criteria for the staggered iteration, selectable via
`--stagger_criterion`:

| Criterion | Formula | Best for |
|-----------|---------|----------|
| `absolute` | `‖d^{k+1} - d^k‖₂ < tol` | General use, fast |
| `relative` (default) | `‖d^{k+1} - d^k‖₂ / ‖d^{k+1}‖₂ < tol` | Crack nucleation, fine details |
| `am_energy` | `|E^{k+1} - E^k| / |E^k| < tol` | Peak load accuracy |
| `linf` | `max|d^{k+1} - d^k| < tol` | Pointwise convergence, crack tip accuracy |
| `residual` | `‖R_u‖₂ < tol AND ‖R_d‖₂ < tol` | Mathematically strictest, PDE satisfaction |

- **Absolute**: Simple L2 norm of damage change. Sensitive to mesh size.
- **Relative**: Normalized by damage magnitude. Adapts to current state.
- **Energy**: Uses `compute_total_energy()` (elastic + fracture energy).
  Physically motivated — converges when the energy functional is stationary.
  More iterations per step but most accurate for peak load prediction.
- **L∞ (linf)**: Maximum pointwise damage change. Detects single-node
  convergence failures that L2 norms can mask by averaging over thousands of
  zero-change nodes. Recommended when crack-tip accuracy is critical.
- **Residual**: Computes the assembled PDE residuals `R_u = f_int(u,d) - f_ext`
  (mechanics) and `R_d = A*d - b` (damage). Directly measures how well the
  PDEs are satisfied. Standard in computational mechanics (SNES). Most expensive
  per iteration but independent of iteration history.

### Output Directory (timestamped)

Each run creates a unique timestamped subdirectory to preserve previous results:

```
examples/quasistatic/miehe_tension/output/run_110326_143052/
```

Format: `run_ddmmyy_HHMMSS`. Use `--output_dir /path` to override.

