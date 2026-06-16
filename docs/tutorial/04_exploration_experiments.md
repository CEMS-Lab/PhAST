# Exploration experiments

Once the [Getting Started](../getting-started.md) runs cleanly, the fastest way
to build intuition is to break the simulation in controlled ways and
watch what changes. Each experiment below is small (one parameter
edit), independent, and re-uses an existing config so there is no new
mesh to debug.

All commands assume you start from the repository root.

## 1. Sweep Poisson's ratio and watch the crack curve

**Motivation.** `nu` enters every energy split: it controls the
Lame coefficient `lambda` which, in turn, sets how compressive
strain is partitioned. Big-`nu` materials behave more like rubber and
crack paths tend to curve more sharply.

```bash
# Edit configs/B7_dynamic_crack_branching_comsol.yaml in a copy:
#   material.overrides.nu: 0.20    (default for glass_borden)
# Try 0.10, 0.30, 0.40 in three separate runs.
python -m phast run my_B7_nu010.yaml --device cpu
python -m phast run my_B7_nu030.yaml --device cpu
python -m phast run my_B7_nu040.yaml --device cpu
```

**Expected outcome.** As `nu` rises the branch angle widens (more
compressive locking on the inside of the curve) and branching onset
arrives a few microseconds earlier.

**Too far.** `nu >= 0.49` makes the bulk modulus blow up and the
explicit timestep collapse; `nu < 0` is unphysical and CG will stall.

## 2. Compare energy splits on the same SENT

**Motivation.** Section 3 of the [primer](01_phase_field_primer.md)
lists five splits. The one you pick matters: the same SENT geometry
under the same load can produce a single straight crack, a curving
crack, or a bilateral branch depending on the split.

```bash
# Make four copies of configs/B3_dynamic_sent.yaml with
#   material.overrides.energy_split: isotropic  | amor | spectral | star_convex
# (skip spectral_stress unless you are checking COMSOL parity)
python -m phast run B3_isotropic.yaml --device cpu
python -m phast run B3_amor.yaml --device cpu
python -m phast run B3_spectral.yaml --device cpu
python -m phast run B3_starconvex.yaml --device cpu
```

**Expected outcome.** `isotropic` propagates fastest (no compressive
lock-out), `amor` and `spectral` produce qualitatively similar crack
fronts, `star_convex` converges with fewer staggers per step at the
cost of slightly different nucleation timing.

**Too far.** `isotropic` will close cracks under reflected
compressive waves -- damage decreases, which violates irreversibility
unless `H` is enforced. If you see `max(d)` drop step-to-step, switch
to `amor` or `spectral`.

## 3. Sweep `l0` from `0.5 h` to `2 h`

**Motivation.** The regularisation length `l0` must resolve the
diffuse damage band, so the rule of thumb is `l0 >= 2 h`. Pushing
below that under-resolves the band; pushing above wastes resolution.

```bash
# In a copied dynamic SENT YAML deck:
#   geometry.parameters.h_crack: 0.25
#   material.overrides.l0:  0.125 / 0.25 / 0.5 / 1.0
python -m phast run sent_l0_0p125.yaml --device cpu  # under-resolved
python -m phast run sent_l0_0p25.yaml  --device cpu
python -m phast run sent_l0_0p50.yaml  --device cpu  # rule-of-thumb 2h
python -m phast run sent_l0_1p00.yaml  --device cpu
```

**Expected outcome.** `l0 = 2 h` (here `0.5`) converges fastest --
CG iteration counts hold steady through the run. At `l0 = 0.5 h` the
damage is jagged, CG iterations spike, and you may see
`max(d) < 1` at the crack tip. At `l0 = 4 h` the crack is too thick
and visually different from the reference but cheap.

**Too far.** `l0 < h` means fewer than one element in the band; the
crack pattern becomes mesh-dependent.

## 4. Compare output cadence

**Motivation.** Output cadence changes wall-clock cost and disk usage.
Before a long run, compare a dense trajectory against a sparse trajectory on a
short validation window.

```bash
python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml \
  --device cpu --num_steps 50 --output_dir output/sent_dense
python -m phast run examples/dynamic/B3_dynamic_sent/config.yaml \
  --device cpu --num_steps 50 --h5_every 10 --output_dir output/sent_sparse
```

**Expected outcome.** The dense run writes more snapshots and is easier to
animate. The sparse run is smaller and faster to move between machines.

**Too far.** Very sparse output can miss crack-initiation frames. Keep enough
snapshots to explain the response you plan to show.

## 5. Inspect result artifacts

**Motivation.** A result directory should be self-describing. Check manifests,
histories, visuals, and metadata before you compare or share a run.

```bash
python - <<'PY'
import phast

result = phast.load_result("output/sent_dense")
print(result.metadata())
print(result.visuals())
print(result.history_names())
PY
```

**Expected outcome.** You should see run metadata, visual artifact paths, and
CSV history names. Missing manifests or empty histories mean the run should be
regenerated before it becomes a public example.

**Too far.** Do not mix outputs from different runs in one directory. Use a new
`--output_dir` for each configuration.

## 6. Subcycle damage after a reference check

**Motivation.** In explicit dynamics the elastic CFL timestep is
about three times finer than the damage propagation needs. Solving
the damage equation every third step can shorten the wall-clock by
40-50%, but it should be treated as a throughput setting after the
`damage_every: 1` reference run has been checked.

```bash
# Edit solver.damage_every in any explicit config.
python -m phast run B1_de1.yaml --device cpu     # damage_every: 1
python -m phast run B1_de3.yaml --device cpu     # damage_every: 3 (throughput)
python -m phast run B1_de5.yaml --device cpu     # damage_every: 5 (aggressive)
```

**Expected outcome.** `damage_every: 3` matches `damage_every: 1`
visually and on energy budgets. `damage_every: 5` is faster still but
the branching onset shifts a few hundred nanoseconds late.

**Too far.** With AT1 + Amor splits, `damage_every > 1` can suppress
branching entirely (issue #131). Stick to `damage_every: 1` whenever
you are comparing against a reference figure.

## 7. Anderson acceleration for stagger convergence

**Motivation.** The staggered scheme is a fixed-point iteration; it
can stall. Anderson acceleration (Walker & Ni 2011) restarts the
fixed point with a least-squares-projected step.

```yaml
solver:
  solver_type: quasi_static
  anderson_depth: 5     # try 0 (off), 3, 5, 7
```

**Expected outcome.** Depth 3 cuts stagger iterations 30%; depth 5 by
50%. Above 7 the linear-least-squares cost dominates and you see
diminishing returns.

**Too far.** On highly nonlinear steps Anderson can over-shoot and
produce non-monotone damage. If you see CG counts spike right after
an Anderson restart, drop the depth.

## Where to next

These experiments cover parameters that move the crack pattern and change the
artifact bundle. The full parameter schema is in `configs/REFERENCE.yaml`, and
solid-mechanics API notes live under `docs/api/`.
