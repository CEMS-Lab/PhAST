# Performance optimisations

Phase-field subcycling, smooth-step loading, crack-front indicator, and the
residual-stiffness `eta` knob — the v0.13.0 performance package.


The only requirement is that the mesh contains 2D triangular elements and
(optionally) named node/element sets for boundary conditions.

---

## Performance Optimizations (v0.13.0)

### Phase-Field Subcycling

For explicit dynamics, the damage equation can be solved every N-th time step
instead of every step. Since damage propagates at ~0.6 c_R while the CFL
condition is based on c_p ≈ 3 × (0.6 c_R), the damage field barely changes
between consecutive steps.

`damage_every` is a YAML field in the `solver:` block (no CLI flag); set it
in the config file:

```yaml
solver:
  damage_every: 1   # validation: solve damage every explicit step
  # damage_every: 2   # COMSOL-style throughput sensitivity run
  # damage_every: 3   # aggressive throughput run; document as non-reference
```

```bash
python -m phast run configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml --device cuda
```

**Reference:** COMSOL Multiphysics 6.4, "Phase-Field Modeling of Dynamic Crack
Branching" — solves phase field every 2nd time step based on the ratio
c_p / (0.6 c_R) ≈ 3.

**Note:** The first 5 explicit steps always solve damage regardless of
`damage_every`, to ensure crack nucleation is captured.

### Smooth Step Loading

Instantaneous traction loading causes spurious high-frequency oscillations.
Use `smooth_step()` to ramp loads over a transition zone:

```python
from phast.boundary_conditions import smooth_step

# Ramp from 0 to 1 over the first 0.05 µs
for step in range(n_steps):
    t = step * dt
    factor = smooth_step(t, t_start=0.0, t_end=0.05e-6)
    bcs.load_factor = factor * max_load
    solver.step_full()
```

Also available as `smooth_step_tensor()` for vectorized inputs.

**Reference:** COMSOL step function with smoothing transition zone (0.05 µs default).

### Crack Front Indicator

Better crack tip detection using d × ∂d/∂t instead of simple threshold:

```python
# Compute crack front indicator (peaks at propagating crack tips)
indicator = solver.fem.compute_crack_front_indicator(solver.d, d_prev, dt)
# indicator is zero in undamaged (d=0) AND fully cracked (∂d/∂t=0) regions
crack_tip_nodes = (indicator > threshold).nonzero()
```

**Reference:** COMSOL 6.4, "The crack front is tracked by the product of the
phase field and its time derivative."

### Residual Stiffness (eta)

