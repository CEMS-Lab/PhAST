# PF-CZM (Wu 2017) — Scoping for `phast`

Issue: #247 (PF-CZM implementation), epic #259 (cohesive-zone family).
Date: 2026-05-06. Status: scoping only — no code changes.

This document scopes the addition of Wu's phase-field cohesive-zone model
(PF-CZM) to `phast` alongside the existing AT1 / AT2 / Allen–Cahn
options. The motivation is twofold: (i) PF-CZM recovers a finite,
mesh-independent traction–separation law (TSL) and a true `K_n` (cohesive
strength), making it the canonical regularised counterpart of Abaqus
cohesive elements; (ii) COMSOL 6.4 ships PF-CZM as one of three built-in
phase-field models, so adopting it puts our DSL in 1:1 correspondence with
COMSOL's `Phase Field in Solids` interface.

## §1 — Mathematical formulation

We follow Wu (2017) §2–§3 and the Wu–Nguyen review (2020) §4.2.

**Crack surface density.** A unified family is

```
γ(d, ∇d) = (1 / c_α l₀) [ α(d) + l₀² |∇d|² ] ,    c_α = 4 ∫₀¹ √α(s) ds .
```

The three standard choices are

| Model | α(d)            | c_α        | nucleation threshold |
|-------|-----------------|------------|----------------------|
| AT2   | d²              | 2          | none (zero)          |
| AT1   | d               | 8/3        | finite (3G_c/8 l₀)    |
| PF-CZM| 2d − d²         | π          | finite, set by σ_ts  |

PF-CZM thus differs from AT1 only in α; the same `c_α` integral and the
same Laplacian survive. (Wu 2017 §3.2; Wu–Nguyen 2020 §4.2.1.)

**Geometric crack-driving (degradation) function.** Wu replaces the
standard `(1−d)²` with the rational form

```
g(d) = (1 − d)^p / [ (1 − d)^p + Q(d) ] ,
Q(d) = a₁ d (1 + a₂ d + a₂ a₃ d² ) ,
```

with the user choosing the integer exponent `p ≥ 2` and `a₁` set so that
`g'(0)` matches the desired tensile strength σ_ts via Irwin's relation:

```
a₁ = 4 E G_c / ( π l₀ σ_ts² )         (Wu 2017 Eq. 41)
```

`a₂, a₃` are TSL-shape parameters: `(a₂, a₃) = (−0.5, 0)` for linear
softening, `(2^{5/3} − 3, 0)` for exponential, more elaborate values for
Cornelissen / hyperbolic Wu–Nguyen 2020 Table 4.1.

**Critical crack-driving force.** Y_c = G_c / (c_α l₀). PF-CZM recovers a
sharp-crack TSL τ(w) in the `l₀ → 0` limit with `∫₀^{w_c} τ dw = G_c`
(Wu 2017 Theorem 1).

**Where this differs from AT1 in our solver.** The PF weak form is the
same as AT1 except (a) the reaction coefficient term `α'(d)/(c_α l₀)`
becomes `(2 − 2d)/(π l₀)` (i.e. linear in `d`, zero at `d=1`), and (b)
the degradation function and its derivatives are the new rational form
above instead of `(1−d)² + η`. The Laplacian / mass / RHS structure is
preserved.

## §2 — Connection to COMSOL and Abaqus

| Property                    | Wu PF-CZM                | Abaqus CZM (cohesive elt) | COMSOL Damage attribute       |
|-----------------------------|--------------------------|---------------------------|-------------------------------|
| Crack representation        | regularised, volumetric  | sharp interface           | regularised, volumetric       |
| TSL recovery in l₀→0        | exact                    | bilinear / exp / PPR      | exact (PowerLaw m=2 + Borden) |
| Length scale                | l₀                        | none                      | l_int                         |
| Element                     | volumetric PF FE         | zero-thickness cohesive   | volumetric PF FE              |
| Tensile strength input      | σ_ts (sets a₁)            | t_n^0 (initiation)        | σ_c                           |
| Implementation status here  | **this issue (#247)**     | **#261**                   | already match (B7 spectral)   |

COMSOL: per `reference_codes/COMSOL_comparison_audit.md` §2.2–§2.3, the
6.2/6.4 Damage attribute exposes Power-law (m=1, m=2 ⇒ PF-CZM-equivalent),
Cubic, and Borden evolution functions. `Borden` is `Power(m=2)` plus a
quartic correction equivalent to Wu's Q(d) with `(p, a₂, a₃) = (2, …, …)`.
COMSOL 6.4 release notes name `AT1`, `AT2`, and `PF-CZM` as the three
supported phase-field families (`Phase Field in Solids` interface);
B5/B6 PMMA in our suite already runs against COMSOL B7 spectral, so PF-CZM
parity unblocks B5/B6 head-to-head.

Abaqus: cohesive-surface and cohesive-element formulations (Abaqus 2024
*Materials Reference*, "Defining the constitutive response of cohesive
elements using a traction-separation description") use bilinear or
exponential TSLs with damage-initiation criteria (max stress, quadratic
stress) and damage-evolution by displacement or fracture energy. There is
no length-scale parameter; the connection is at the macro level — PF-CZM
and Abaqus CZM converge to the same (G_c, σ_ts, w_c) cohesive law at
sufficient mesh resolution. (WebFetch failed: 403 on `help.3ds.com`;
formulation summarised from open Abaqus theory manual references —
*Theory* §4.5.6 cohesive elements; bilinear/exponential TSL standard.)

## §3 — Minimal-viable implementation plan

DSL extension:

```yaml
material:
  pf_model: PFCZM      # new value alongside AT1, AT2, allencahn
  pfczm_p: 2           # exponent in (1-d)^p
  pfczm_softening: linear   # linear | exponential | cornelissen
  sigma_ts: 11.31      # MPa — already exists, drives a₁
  energy_split: spectral    # composes orthogonally
```

Files affected (line numbers from current HEAD):

- **`material.py`** — extend `pf_model: Literal['AT1','AT2','allencahn']`
  at L69 to include `'PFCZM'`; add `pfczm_p`, `pfczm_softening` fields;
  in `degradation()` (around L237) add a `pfczm` branch returning the
  rational `g(d)` and exposing `g'(d)` for the tangent. `c_α = π` constant
  needed for the RHS.
- **`fem_operators.py`** — `degradation()` is already routed via
  `material.degradation(d_e)` at L447, L528, L910, L975, L1018, L1326.
  No changes here if the new `g` is plumbed inside `material.py`.
- **`damage_solver.py`** — three sites:
  1. `c_w` selector at L94, L989: extend ternary to `{'AT2': 2,
     'AT1': 8/3, 'PFCZM': π}`.
  2. RHS source-term branch at L284, L429, L536, L659: PFCZM uses
     `α'(d)/(c_α l₀) = (2 − 2d)/(π l₀)` — mirrors the AT1 branch but with
     a `d`-dependent reaction coefficient (so it cannot be folded into a
     constant `reaction_coeff`; needs an extra Mass-times-d term).
  3. Adjoint sensitivity at L110–111, L498–499: derive
     `dL/dGc_e` for PFCZM = `-1/(π l₀) · (λ_M_1_e − 2 λ_M_d_e) − (l₀/π) λ_K_d_e`.
- **`staggered_solver.py`** — guard at L244 (AT1 ⇒ projected_cg) extends to
  PFCZM; PFCZM has the same finite threshold issue and `d≡0` post-clamp
  pathology. `softmax_H_beta` (L438) is independent.

## §4 — Risks and gotchas

1. **`d`-dependent reaction coefficient.** AT1's RHS has a *constant*
   source `−3G_c/(8l₀)` per element; PF-CZM's source `2(1−d)G_c/(π l₀)`
   is `d`-dependent. The current `_Ax(d, reaction_coeff)` API
   (`damage_solver.py` L1177) takes a constant `reaction_coeff` and adds
   it as a diagonal mass term. PF-CZM needs that term to migrate to the
   RHS or to be split: `(2/π l₀)·M·1 − (2/π l₀)·M·d`. The second is a
   **negative mass** contribution to A, which can make A indefinite for
   small l₀. Mitigation: keep the constant part on the RHS and put the
   `−(2/π l₀) d` on the LHS as a negative-definite shift; verify the
   `projected_cg` Hessian remains PSD on the active set (it does, since
   bound-projection regularises).
2. **g(d) denominator positivity.** With `a₁ > 0` and the Wu-recommended
   `(a₂, a₃)` table, `Q(d) > 0` on `d ∈ [0,1)` and `(1−d)^p + Q(d) > 0`
   throughout. Adding our `eta_residual` (1e-7) inside the denominator
   guards against the `d → 1` limit.
3. **Composition with `energy_split`.** PF-CZM is a `pf_model` choice (it
   sets α, g, c_α). Splits (Amor / spectral / spectral_stress / isotropic)
   act on the *driving force* `H` and are orthogonal. The
   `pf_model='PFCZM' + energy_split='spectral'` combination is the
   COMSOL-default for B5/B6 PMMA and the Wu 2017 SENT benchmark.
4. **Adjoint correctness.** The implicit-diff machinery
   (`_AdjointDamageSolveScalar/Field`) currently special-cases `AT1` vs
   `AT2`. The PFCZM branch needs a third case in `dL/dGc_e` and in the
   reaction-coeff propagation through CG.

## §5 — Test plan

1. **Unit tests** (`tests/test_material_pfczm.py`):
   - `α(0)=0`, `α(1)=1`, `α'(0)=2`, `α'(1)=0`.
   - `g(0)=1`, `g(1)=0`, `g'(0)= −a₁ < 0` matching Irwin: numerical match
     to `4 E G_c / (π l₀ σ_ts²)` within 1e-10.
2. **SENT vs Wu 2017 Fig. 12** (PMMA, E=32 GPa, ν=0.2, G_c=3 N/m,
   σ_ts=11.31 MPa). Peak load 0.6 kN at u≈5 µm; mesh-convergent for
   `h ≤ l₀/4`. Three meshes (l₀/2, l₀/4, l₀/8) check
   length-scale-insensitivity (the defining PF-CZM property).
3. **Cohesive analytical limit.** 1-D bar, prescribed displacement, run
   l₀ ∈ {0.5, 0.25, 0.1} mm; integrate dissipated energy per crack area
   vs G_c; require <2% deviation on the finest l₀.
4. **COMSOL parity.** Re-run B7 dynamic crack branching with
   `pf_model: PFCZM, energy_split: spectral`, compare crack-tip path and
   branching onset to the existing AT1/AT2 runs and the COMSOL 6.4 PDF
   (33 µs onset).

## References

- Wu, J.-Y. (2017). *A unified phase-field theory for the mechanics of
  damage and quasi-brittle failure*. **JMPS** 103, 72–99.
  DOI: 10.1016/j.jmps.2017.03.015.
- Wu, J.-Y., Nguyen, V.P., Nguyen, C.T., Sutula, D., Sinaie, S., Bordas, S.
  (2020). *Phase-field modeling of fracture*. **Adv. Appl. Mech.** 53,
  1–183. DOI: 10.1016/bs.aams.2019.08.001.
- Bourdin, B., Larsen, C.J., Richardson, C.L. (2011). *A time-discrete
  model for dynamic fracture based on crack regularization*.
  **Int. J. Fract.** 168, 133–143. DOI: 10.1007/s10704-010-9562-x.
  (Local copy: `refs/Bourdin et al. (2011) - time-discrete dynamic fracture.pdf`.)
- COMSOL Multiphysics 6.2 — *Damage* theory page.
  <https://doc.comsol.com/6.2/doc/com.comsol.help.sme/sme_ug_theory.06.035.html>
  (accessible; no PF-CZM by name, but Power-law and Borden evolution
  functions cover the family).
- COMSOL 6.4 release notes (Phase Field in Solids — AT1/AT2/PF-CZM).
  <https://www.comsol.com/release/6.4>.
- Abaqus 2024 *Materials Reference* — cohesive surfaces / cohesive
  elements with traction-separation behaviour.
  <https://help.3ds.com/2024/english/dssimulia_established/SIMACAEMATRefMap/simamat-c-cohesivecz.htm>
  (WebFetch failed: HTTP 403; formulation summary from open Abaqus theory
  manual conventions and Wu–Nguyen 2020 §4.4 cross-reference).
- Wu et al. (2022). PF-CZM in COMSOL (open source).
  <https://github.com/jianyingwu/pfczm-comsol>.
- Internal: `reference_codes/COMSOL_comparison_audit.md` §1, §2.1–§2.5.
