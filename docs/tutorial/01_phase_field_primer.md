# Phase-field primer for mechanics engineers

This document is a compact review for engineers who already know
finite elements, linear elasticity, and small-strain plasticity, but
have not used phase-field fracture before. It explains *why* the model
in `phast` looks the way it does and points to the public references at
the end of this page.

## Variational fracture in one minute

Griffith's 1921 fracture criterion balances elastic strain energy
against a surface energy proportional to the crack area. Francfort and
Marigo (1998) recast the criterion as a global energy minimisation:
find a displacement field `u` and a crack set `Gamma` minimising

```{math}
\mathcal{E}(\mathbf{u}, \Gamma)
= \int_{\Omega} \psi(\boldsymbol{\varepsilon}(\mathbf{u})) \, d\Omega
  + G_c \, \mathcal{H}^{n-1}(\Gamma).
```

where $\psi$ is the elastic strain energy density, $G_c$ is the critical
energy release rate, and $\mathcal{H}^{n-1}$ measures the $(n-1)$-dimensional crack
surface.

The discrete-crack-set problem is intractable on a fixed mesh. Bourdin,
Francfort and Marigo (2000) proposed a *regularised* form that
approximates the sharp crack $\Gamma$ by a smooth scalar damage field
$d \in [0, 1]$ with a length scale $\ell_0$. As $\ell_0 \to 0$ the regularised
energy gamma-converges to the sharp-crack energy. In `phast`
the regularised energy is

```{math}
\mathcal{E}(\mathbf{u}, d)
= \int_{\Omega} g(d) \, \psi^+(\boldsymbol{\varepsilon}(\mathbf{u})) \, d\Omega
  + \int_{\Omega} \psi^-(\boldsymbol{\varepsilon}(\mathbf{u})) \, d\Omega
  + \frac{G_c}{4c_w} \int_{\Omega}
    \left[
      \frac{w(d)}{\ell_0}
      + \ell_0 |\nabla d|^2
    \right] d\Omega ,
\qquad c_w=\int_0^1\sqrt{w(s)}\,ds.
```

$g_\eta(d) = (1-\eta)(1-d)^2 + \eta$ is the standard degradation function;
$w(d)$ is the local dissipation density; $c_w$ is a normalisation constant. The
normalization above gives $c_w=1/2$ for AT2 and $c_w=2/3$ for AT1. The displayed
functional describes the standard split model before external work is
subtracted. PhAST treats the small residual stiffness as a mechanics
regularization; its conventional linear AT1/AT2 damage operator retains the
$\eta$-independent driving coefficients. Other degradation families have
separate capability boundaries and should not be inferred from this equation.

## AT1 vs AT2

The two standard regularisations differ in `w(d)` and `c_w`:

| Model | `w(d)` | `c_w` | Elastic threshold | Reference |
|-------|--------|-------|-------------------|-----------|
| AT2 | $d^2$ | $1/2$ | None; damage may start at nonzero strain | Bourdin et al. (2011) |
| AT1 | $d$ | $2/3$ | $\mathcal{H}_{c,0}=3G_c/(16\ell_0)$ | Pham, Marigo, Maurini (2011) |

```{figure} ../_static/at_dissipation.svg
:alt: AT1 and AT2 local crack-density terms
:width: 78%

AT1 and AT2 differ through the local term $w(d)$. The full crack-surface
density also includes the gradient penalty and the normalization constant
$c_w$, so this plot should not be read as a complete fracture-energy density or
as the one-dimensional crack profile.
```

**AT2** is mathematically simpler -- the damage equation is linear in
$d$ for fixed history field $\mathcal{H}$, so a single CG solve does the job. The downside: at
any non-zero strain a tiny amount of damage develops everywhere,
because there is no elastic threshold. Most papers therefore enforce a
post-hoc nucleation threshold or a pre-existing notch.

**AT1** has a true elastic phase: damage stays at zero until the local
driving energy $\mathcal{H}$ exceeds $\mathcal{H}_{c,0}=3G_c/(16\ell_0)$. This matches the
intuition of "no damage until the strength is reached" but the damage
sub-problem is now constrained ($d \geq 0$), so supported AT1 calculations use
projected CG (`bounds_method='projected_cg'`). A post-clamp after an
unconstrained solve is not a valid replacement for the AT1 active-set solve.

In a YAML config, switch with `material.overrides.pf_model: AT1` or
`AT2`. AT1 is the right choice when you care about nucleation without
a pre-crack (Ambati et al. 2015, Bleyer et al. 2017); AT2 is the right
choice for propagation from an existing notch (Borden et al. 2012).

For the default hard-history route, the history field

```{math}
\mathcal{H}(\mathbf{x}, t) =
\max_{\tau \leq t} \psi^+
\left(\boldsymbol{\varepsilon}(\mathbf{u}(\mathbf{x}, \tau))\right)
```

makes the crack-driving field nondecreasing. Nodal no-healing is imposed
separately through the lower bound $d_{n+1}\geq d_n$. The projected-CG route
maintains this bound through an active set; `post_clamp` instead enforces
admissibility after an unconstrained solve and is not an active-set solution.
The authoritative history is elementwise for T3 and quadrature-based for Q4;
nodal history is a projected field used for output and selected interfaces.
Optional smooth-history routes alter this update and must be interpreted as
separate differentiability approximations.

## Staggered minimisation loop

The supported quasistatic route uses alternate mechanics-damage iterations.
PhAST freezes damage while solving mechanics, updates the driving history, then
freezes mechanics while solving damage. Convergence ordinarily assesses both
displacement and damage changes. Explicit dynamics instead performs one
segregated pass per time step, with damage solved at the configured cadence; it
does not use the quasistatic inner convergence test. The joint monolithic
$(\mathbf{u},d)$ minimizer is an experimental comparison pathway and does not
provide the projected-CG active-set enforcement described above.

```{mermaid}
flowchart TD
    A[Load or time step] --> B[Freeze damage d]
    B --> C[Solve mechanics for u]
    C --> D[Update history H]
    D --> E[Freeze u]
    E --> F[Solve damage for d]
    F --> G[Project bounds and irreversibility]
    G --> H{displacement and damage updates satisfy criteria?}
    H -- no --> B
    H -- yes --> I[Advance step and write outputs]
```

One representative damage component of the quasistatic stopping test is

```{math}
\| d^{k+1} - d^k \|_{\infty}
< \varepsilon_{\mathrm{stag}} .
```

## Energy splits -- why we don't degrade `psi` directly

If `g(d)` multiplies the full strain energy, compressive energy can drive
damage and compressive stiffness can be removed from damaged zones. This may
produce spurious damage or interpenetration. Tension-compression splits mitigate
that behaviour:

```{math}
\psi(\boldsymbol{\varepsilon})
= \psi^+(\boldsymbol{\varepsilon})
  + \psi^-(\boldsymbol{\varepsilon}),
```

For split formulations, only `psi+` is degraded. The `isotropic` option is an
intentional unsplit exception. PhAST exposes the following routes:

| `energy_split` | What gets degraded | When to use |
|----------------|--------------------|-------------|
| `isotropic` | full energy | Pure mode I tension; debugging |
| `amor` | volumetric tension + deviatoric | General default; robust under mixed loading (Amor, Marigo, Maurini 2009) |
| `spectral` | tensile principal strains | Curving / branching cracks (Miehe, Welschinger, Hofacker 2010) |
| `spectral_stress` | tensile principal stresses | Opt-in COMSOL parity; experimental |
| `star_convex` | tension full / compression deviatoric | Improved convergence, nucleation (Kumar, Francfort, Lopez-Pamies 2020) |
| `spectral_plane_stress_condensed` | condensed plane-stress spectral contribution | Research comparison; use only with case-specific verification |

`amor` is a safe starting point. `spectral` is what most published
dynamic-fracture benchmarks use (Borden 2012, Bleyer 2017). For
isotropic Mode I loading with no compressive zones, `isotropic` is
faster and gives the same answer.

Plane-stress `spectral` is supported as a reduced 2D in-plane
strain-spectral projection. It is not a fully condensed 3D plane-stress
spectral decomposition with damage-dependent out-of-plane strain. For mature
validated paths, use plane-strain `spectral` for Miehe-style principal-strain
splits or plane-stress `amor` for thin PMMA-style dynamic benchmarks.

## The four parameters that matter

| Parameter | Symbol | Typical range | Effect |
|-----------|--------|---------------|--------|
| Regularisation length | `l0` | 1-4 elements (`l0 = 2 h`) | Smaller = sharper crack, more compute |
| Fracture toughness | `Gc` | material-dependent | Sets the load to fracture |
| Residual stiffness | `eta_residual` | `1e-7` (default) | Numerical floor on `g(d)`; prevents zero-stiffness rows |
| `pf_model` | -- | `AT1` or `AT2` | Sets whether nucleation has a threshold |

The mesh size `h` near the crack should commonly satisfy `h <= l0 / 2`
to resolve the diffuse damage band. This is a starting resolution criterion,
not a convergence guarantee. Mesh refinement should vary $h$ at fixed
$\ell_0$. A separate $\ell_0$ sensitivity study addresses the regularized
model: changing $\ell_0$ changes crack-band width, nucleation response, and
computational cost, and is not merely a mesh-coarsening device.

In mathematical form:

```{math}
h \leq \frac{\ell_0}{2}.
```

## References

- Bourdin et al. (2000, 2011) for regularized variational fracture and
  time-discrete dynamic fracture.
- Borden et al. (2012) for phase-field dynamic brittle-fracture benchmarks.
- Ambati et al. (2015) for a review of phase-field brittle fracture and energy
  splits.
- Bleyer, Roux-Langlois, and Molinari (2017) for dynamic branching and
  velocity-toughening studies.

Once you have the theory in mind, head to
[Setting up new problems](../user_guide/setup_problems.md) to translate it
into a PhAST model and durable YAML configuration.


For a picture-first companion, see the [visual glossary](02_visual_glossary.md).
