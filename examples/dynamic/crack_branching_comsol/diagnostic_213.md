# B7 branching-onset diagnostic (#213) -- explore-only

HPC job 28652 reports branching onset at **78 µs** vs COMSOL **33 µs**
(out of the ±20 % tolerance band). Energy peak now PASS (0.34 J in
the 0.13–0.14 J × 1000-mm-thickness band, PR #211); morphology +
final Y-shape PASS visually. Only the *timing* is late, by ~2.4×.

This file ranks candidate causes against the COMSOL Geomechanics PDF
(`models.geomech.dynamic_crack_branching.pdf`) and Borden 2012, with
the YAML knob that would test each. **No code edits in this PR.**

## Reference setup (from PDF)

- AT1, plane strain, thickness `d0 = 1 m`, half-plate 100×40 mm with
  symmetry on the bottom edge (PDF p. 3 + p. 11 step "Symmetry 1").
- Element size `he = lint/4 = 0.125 mm` -- twice as fine as our
  `h_crack: 0.25` (l0/2). PDF p. 10.
- Smooth-step ramp **transition zone = 0.05 µs** (PDF p. 10, "Step 1").
  Our config has `t_ramp: 5.0e-6` = **5 µs**, **100× longer**.
- Phase-field every 2nd explicit step (subcycling). PDF p. 3 caption.
- `eta = 1e-7`. Driving force `H = max(psi+, W_c0)` with
  `W_c0 = 3 Gc / (8 l_int)` (Eq 6).

## Hypothesis ranking (most likely → least)

### 1. (f) Step-pulse rise time, `t_ramp` (HIGHEST CONFIDENCE)
COMSOL ramps over 0.05 µs; we ramp over 5 µs. A 100× slower ramp
attenuates the high-frequency content of the impulse that triggers
the branching micro-instability (Borden 2012 §4.3 attributes
branching to a Yoffe-type velocity instability driven by the
inertial overshoot near the crack tip). Slow loading delays the
moment the tip reaches the critical velocity ≈ 0.6 c_R. Ratio of
delays (≈ 2.4×) is consistent with first-arrival of the elastic
front at the crack tip being pushed back by a finite ramp.
Test knob: `loading.t_ramp: 5.0e-8` (= 0.05 µs, COMSOL value).

### 2. (d) Mesh + (a) AT1 nucleation interaction
COMSOL: `he = lint/4 = 0.125 mm`. We: `h_crack = 0.25 = lint/2`.
At AT1 the off-axis damage band has width ~`pi*l0`; under-resolving
the band by 2× delays when `psi+` exceeds `W_c0` over a
contiguous *band* (not just a node) -- the discrete spread is
slower. PhaFiDyn (Barki 2025) recommends `h ≤ l0/4` for branching.
This is consistent with our late-but-correct-shape outcome.
Test knob: `geometry.parameters.h_crack: 0.125`.

### 3. (c) `dt_safety = 0.8` vs Borden's 0.5
Borden 2012 §4.3 uses safety 0.5 for B7-equivalent runs. With
`damage_every: 2` (subcycling), the *damage* timestep is 1.6 ×
the explicit CFL bound -- on the edge of CFL for the phase field.
Coarse damage dt damps the instability seed.
Test knob: `solver.dt_safety: 0.5`.

### 4. (a' / damage_every) Subcycling delays band spreading
COMSOL also runs `damage_every: 2` so this is unlikely to be the
*sole* cause, but combined with (2) above it compounds. Falsifiable
cheaply.
Test knob: `solver.damage_every: 1`.

### 5. (e) Damping
Our YAML has no Rayleigh / Kelvin-Voigt damping (`damping_ratio_max`
unset, default 0). Same as COMSOL. Rule out.

### 6. (b) `_spectral_eps = 1e-12`
At E = 32 GPa, ε ~ 1e-3, principal strains O(1e-3); `delta` floor
is `sqrt(1e-12) = 1e-6` -- five orders below the operating point.
Cannot smear the eigenvalue split here. Rule out.

## Recommended experiment order
1. Ramp fix: set `t_ramp: 5.0e-8`. Cheapest, highest-prior. **Run first.**
2. If (1) only partially closes the gap: refine to `h_crack: 0.125`.
3. If still off: drop `dt_safety` to 0.5 (or `damage_every: 1`).

Cite: Bourdin/Borden 2012 §4.3; Barki 2025 (PhaFiDyn) §4;
COMSOL 6.4 *Phase-Field Modeling of Dynamic Crack Branching*,
Geomechanics Module, p. 3 + p. 10.
