# Phase-field primer for mechanics engineers

This document is a one-page review for engineers who already know
finite elements, linear elasticity, and small-strain plasticity, but
have not used phase-field fracture before. It explains *why* the model
in `phast` looks the way it does and points to the original
references in `refs/`.

## Variational fracture in one minute

Griffith's 1921 fracture criterion balances elastic strain energy
against a surface energy proportional to the crack area. Francfort and
Marigo (1998) recast the criterion as a global energy minimisation:
find a displacement field `u` and a crack set `Gamma` minimising

```
E(u, Gamma) = integral_Omega psi(eps(u)) dV  +  Gc * H^{n-1}(Gamma)
```

where `psi` is the elastic strain energy density, `Gc` is the critical
energy release rate, and `H^{n-1}` measures the (n-1)-dimensional crack
surface.

The discrete-crack-set problem is intractable on a fixed mesh. Bourdin,
Francfort and Marigo (2000) proposed a *regularised* form that
approximates the sharp crack `Gamma` by a smooth scalar damage field
`d in [0, 1]` with a length scale `l0`. As `l0 -> 0` the regularised
energy gamma-converges to the sharp-crack energy. In `phast`
the regularised energy is

```
E(u, d) = integral_Omega g(d) psi+(eps(u)) dV
        + integral_Omega psi-(eps(u)) dV
        + Gc * integral_Omega ( w(d)/(c_w * l0)  +  l0 * |grad d|^2 / c_w ) dV
```

`g(d) = (1-d)^2 + eta_residual` is the degradation function; `w(d)` is
the local dissipation density; `c_w` is a normalisation constant. The
Euler-Lagrange system is two coupled PDEs (mechanics and damage) which
the staggered solver alternates between.

## AT1 vs AT2

The two standard regularisations differ in `w(d)` and `c_w`:

| Model | `w(d)` | `c_w` | Elastic threshold | Reference |
|-------|--------|-------|-------------------|-----------|
| AT2 | `d^2` | 1/2 | None (damage from t=0) | Bourdin et al. (2011) |
| AT1 | `d`   | 8/3 | `W_c0 = 3 Gc / (8 l0)` | Pham, Marigo, Maurini (2011) |

**AT2** is mathematically simpler -- the damage equation is linear in
`d` for fixed `H`, so a single CG solve does the job. The downside: at
any non-zero strain a tiny amount of damage develops everywhere,
because there is no elastic threshold. Most papers therefore enforce a
post-hoc nucleation threshold or a pre-existing notch.

**AT1** has a true elastic phase: damage stays at zero until the local
driving energy `H` exceeds `W_c0 = 3 Gc / (8 l0)`. This matches the
intuition of "no damage until the strength is reached" but the damage
sub-problem is now constrained (`d >= 0`), so production AT1 runs use
projected CG (`bounds_method='projected_cg'`). A post-clamp after an
unconstrained solve is not a valid replacement for the AT1 active-set solve.

In a YAML config, switch with `material.overrides.pf_model: AT1` or
`AT2`. AT1 is the right choice when you care about nucleation without
a pre-crack (Ambati et al. 2015, Bleyer et al. 2017); AT2 is the right
choice for propagation from an existing notch (Borden et al. 2012).

The history field `H = max_t psi+` enforces irreversibility (a node
cannot heal). Combined with the damage-bound constraint
`d_new >= d_old`, this gives the monotone crack growth observed in
experiments.

## Energy splits -- why we don't degrade `psi` directly

If `g(d)` multiplies the *full* strain energy `psi(eps)`, cracks can
close under compression and develop on the compressive side of a
bend -- both unphysical. The fix is to split

```
psi(eps) = psi+(eps)  +  psi-(eps)
```

where only `psi+` (the "damaging" part) is degraded. `phast`
ships five splits, all in `fem_operators.py`:

| `energy_split` | What gets degraded | When to use |
|----------------|--------------------|-------------|
| `isotropic` | full energy | Pure mode I tension; debugging |
| `amor` | volumetric tension + deviatoric | General default; robust under mixed loading (Amor, Marigo, Maurini 2009) |
| `spectral` | tensile principal strains | Curving / branching cracks (Miehe, Welschinger, Hofacker 2010) |
| `spectral_stress` | tensile principal stresses | Opt-in COMSOL parity; experimental |
| `star_convex` | tension full / compression deviatoric | Improved convergence, nucleation (Kumar, Francfort, Lopez-Pamies 2020) |

`amor` is a safe starting point. `spectral` is what most published
dynamic-fracture benchmarks use (Borden 2012, Bleyer 2017). For
isotropic Mode I loading with no compressive zones, `isotropic` is
faster and gives the same answer.

## The four parameters that matter

| Parameter | Symbol | Typical range | Effect |
|-----------|--------|---------------|--------|
| Regularisation length | `l0` | 1-4 elements (`l0 = 2 h`) | Smaller = sharper crack, more compute |
| Fracture toughness | `Gc` | material-dependent | Sets the load to fracture |
| Residual stiffness | `eta_residual` | `1e-7` (default) | Numerical floor on `g(d)`; prevents zero-stiffness rows |
| `pf_model` | -- | `AT1` or `AT2` | Sets whether nucleation has a threshold |

The mesh size `h` near the crack must satisfy roughly `h <= l0 / 2`
to resolve the diffuse damage band. If you double `l0`, you can halve
the element count -- but the apparent fracture toughness changes
slightly (the `Gc` of the *regularised* model is not exactly `Gc` of
the sharp-crack model unless the mesh is fine enough).

## Pointers into `refs/`

- Bourdin et al. (2011) - time-discrete dynamic fracture.pdf -- the
  canonical AT2 dynamic phase-field formulation.
- Borden et al. (2012) - phase-field dynamic brittle fracture.pdf --
  the SENT/Kalthoff/branching benchmarks reproduced as `B1`-`B4` in
  `configs/`.
- Ambati et al. (2015) - review phase-field brittle fracture.pdf --
  comprehensive review including energy-split comparison.
- Bleyer, Roux-Langlois, Molinari (2017) - phase-field dynamic
  branching velocity-toughening.pdf -- the PMMA branching benchmark
  reproduced as `B5`/`B6`.

Once you have the theory in mind, head to
[setting up your problem](03_setting_up_your_problem.md) to translate
it into a YAML config.
