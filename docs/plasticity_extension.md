# Plasticity extension status and roadmap (issue #262)

Status, 2026-06-10: beta validation slices are implemented and merged. This is
no longer only a research scoping note. The solver now has a standalone J2
material-point kernel, sparse quasi-static mesh-level J2 mechanics with
per-element state/commit/rollback, a guarded T3 `j2_isotropic` + AT2 ductile
phase-field smoke path, ductile plastic-work validation examples, and
reproducible plasticity/interface YAML contracts.

Customer boundary: this is suitable for a technical-preview / beta validation
release. It is not yet a mature Abaqus/COMSOL-equivalent ductile fracture
product, and it does not yet provide a fully coupled plasticity + PF-CZM +
cohesive-interface production workflow.

## 1. Current state

`phast` has production coupled phase-field solves for **linear
elasticity** with a damage degradation multiplier. It also has a small-strain
J2 radial-return kernel (`plasticity/j2_vonmises.py`) for material-point
validation and a sparse mesh-level J2 mechanics path (`plasticity/mesh_j2.py`)
used by the plasticity/interface validation examples.

Implemented beta capabilities:

- Material-point J2 return mapping with linear isotropic hardening.
- Sparse quasi-static mesh-level J2 mechanics with per-element state,
  commit/rollback, internal-force assembly, plastic-work accounting, and
  backend-selectable validation harnesses.
- Ductile-driving AT2 damage validation, where accumulated plastic work
  contributes to the phase-field driving history.
- Guarded quasi-static T3 `j2_isotropic` + AT2 staggered smoke coverage.
- Reproducibility contracts in
  `configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml`.

Still gated:

- Benchmark-matched ductile SENT/TPB/notched-bar fracture validation.
- Voce/Swift hardening in the coupled benchmark path.
- Drucker-Prager, Mohr-Coulomb, Hill anisotropy, GTN, and rate-sensitive
  plasticity.
- Fully coupled plasticity + cohesive/PF-CZM interfaces.
- Production cuDSS promotion for the sparse elastoplastic path.

The coupled elastic constitutive routines still live in `fem_operators.py`:

- `compute_stress_linear`, `compute_stress_isotropic`, `compute_stress_amor`,
  `compute_stress_spectral_algebraic`, `compute_stress_spectral_stress`,
  `compute_stress_star_convex` — all are functions of total strain
  `(eps_xx, eps_yy, gam_xy)` and the degradation `g(d)` only.
- Plastic state for the beta J2 path is maintained by the plasticity mesh
  operator, not by the legacy elastic `FEMState`.
- The current promoted J2 examples exercise sparse quasi-static dispatch and
  state rollback; broader benchmark parity and larger production-scale studies
  remain open.
- Phase-field history `H` remains the primary nodal damage history variable.

A full production plasticity extension still needs the following hardening:

1. Broader per-quadrature state storage beyond the current element-mean beta
   path, especially for Q4 and higher-order elements.
2. More hardening/yield models and calibrated benchmark data.
3. Consistent production-scale tangent/backend evidence across SciPy,
   PETSc/MUMPS, and cuDSS.
4. Coupled residual/tangent contracts for plasticity together with cohesive
   interfaces.

## 2. What Abaqus and COMSOL do

References for this section:

- Abaqus rate-independent metal plasticity (classical Mises) — Abaqus
  Theory Manual section *Classical metal plasticity*, available on the
  Dassault help portal at
  `https://help.3ds.com/.../simathe-c-classmetalplas.htm`. Public copies of
  the older 6.6 / 6.14 manuals are mirrored at MIT
  (`abaqus-docs.mit.edu`) and WUSTL.
- Simo & Taylor (1985) — *Consistent tangent operators for rate-independent
  elastoplasticity*, CMAME 48(1), 101–118. DOI
  `10.1016/0045-7825(85)90070-2`.
- Simo & Hughes (1998) — *Computational Inelasticity*, Springer, ISBN
  978-0-387-97520-7. Boxes 3.1–3.2 give the classical radial-return.
- COMSOL 6.2 Structural Mechanics Module User's Guide — sections on
  *Elastoplastic Materials* and *Soil Plasticity*
  (`https://doc.comsol.com/6.2/...sme_ug_theory...html`). Direct
  WebFetches in scoping returned only adjacent sections (mixed
  formulation, viscoelasticity, Cam-Clay) — the plasticity chapter is in
  the same theory manual but at a different anchor; its structure mirrors
  the standard textbook treatment.

Both codes implement the same canonical algorithm for J2:

**Yield function**

    f(s, alpha) = sqrt(3/2 * s_dev:s_dev) - sigma_y(alpha)

with `s_dev` the deviatoric Cauchy stress and `alpha` the cumulative
equivalent plastic strain.

**Additive split** of small-strain tensor: `eps = eps_e + eps_p`.

**Backward-Euler / radial-return** at each quadrature point:

1. Trial elastic stress `s_trial = C : (eps_{n+1} - eps_p,n)`.
2. Test `f(s_trial, alpha_n) <= 0` — if so, step is elastic.
3. Otherwise solve scalar `dGamma >= 0` from the consistency
   `f(s_trial - 2 mu dGamma n, alpha_n + dGamma * sqrt(2/3)) = 0`, where
   `n` is the unit deviator of `s_trial`.
4. Update `eps_p`, `alpha`, and `sigma`.

For linear isotropic hardening (`sigma_y = sigma_y0 + H * alpha`) step 3 is
closed form. For Voce
(`sigma_y0 + Q*(1 - exp(-b*alpha))`), Swift
(`K*(eps0 + alpha)^n`), Hockett-Sherby, or Johnson-Cook
(`(A + B*alpha^n) * (1 + C*ln(eps_dot/eps_dot0)) * (1 - T_hat^m)`),
step 3 needs a 1-D Newton inner loop (typically 3–8 iterations).

**Internal state variables** stored at every quadrature point:

- `eps_p` — plastic strain tensor (6 components in 3D, 3 in plane-strain).
- `alpha` — cumulative equivalent plastic strain (1 scalar).
- For kinematic hardening: back-stress `beta` (deviatoric, 5 indep.).
- For Johnson-Cook rate term: temperature `T` (or it is supplied).
- For GTN: void volume fraction `f`.

**Consistent algorithmic tangent** (Simo–Taylor 1985 §4):

    C^alg = C^e
          - 2 mu * (2 mu / (2 mu + 2/3 H' )) * (n ⊗ n)
          - 4 mu^2 * dGamma / |s_trial| * (I_dev - n ⊗ n)

This is the tangent that *must* be used in the global Newton loop;
substituting the continuum elasto-plastic tangent destroys quadratic
convergence (Simo & Hughes 1998 §3.6.3). Both Abaqus and COMSOL
implement this form by default; Abaqus offers a continuum-tangent
fallback only for explicit dynamics where it does not matter.

COMSOL additionally exposes Drucker-Prager and Mohr-Coulomb yield
surfaces with non-associated flow and a tension cut-off (Cam-Clay and
Hardening-Soil are also documented in their soil-plasticity chapter). For
both, the return mapping is more delicate because the yield surface has
corners (Mohr-Coulomb) or apex singularity (Drucker-Prager); the standard
remedy is the de Souza Neto–Peric–Owen apex return (Souza Neto et al.
2008, ISBN 978-0-470-69462-6).

## 3. Roadmap, ranked by priority

**A — J2 + linear isotropic hardening.**
Status: beta implemented for material-point and sparse mesh-level validation
slices. Remaining work is production-scale benchmark parity, Q4/higher-order
coverage, and functional cuDSS promotion.

**B — PF + plasticity coupling (Hai 2026 / Miehe-Aldakheel 2016)**.
Two formulations to choose from:

- *Coupled split* (Miehe, Aldakheel, Teichtmeister 2016, IJNME 109(7),
  DOI `10.1002/nme.5234`): elastic and plastic free-energy parts are both
  degraded by `g(d)`, and plastic work enters the phase-field driving
  force.
- *Hai et al. 2026* (DOI `10.1016/j.jmps.2026.106591`): introduces a
  **separate degradation function** for the plastic part `g_p(omega)`
  with its own intrinsic threshold, plus an **independent compliance
  variable** `omega` that decouples stiffness degradation from the
  phase-field `phi`. Their abstract claim is that this removes geometric
  upper bounds on the length scale `b` and supports arbitrary cohesive
  softening laws (exponential, hyperbolic, Cornelissen).

Recommended implementation target: the Hai 2026 formulation. It is the
more recent, length-scale-insensitive alternative consistent with the
PFCZM line the solver is already chasing (cf. `pfczm_formulation.md`),
and it sidesteps Miehe-Aldakheel's known ad-hoc plastic-work threshold.
Their numerical scheme is semi-explicit (explicit central-difference
equilibrium, neighbour-element phase-field update); this maps cleanly
onto the explicit dynamic driver already in the repo.

Current beta implementation uses accumulated plastic work as an added damage
driving contribution with an energy ledger and sensitivity evidence. Remaining
work is benchmark-matched ductile fracture and, if selected, migration to a
newer length-scale-insensitive PF-CZM-plasticity formulation.

**C — Drucker-Prager / Mohr-Coulomb.**
Once A is in, geomechanics yield surfaces are an additive feature —
swap the yield function and use the Souza-Neto-Peric-Owen apex return.
Cite: de Souza Neto, Peric, Owen (2008) *Computational Methods for
Plasticity*, Wiley.

**D — Gurson-Tvergaard-Needleman.**
GTN provides a void-growth-based ductile damage that is an alternative
to PF for ductile fracture. Adds void volume fraction `f` to the state.
Cite: Tvergaard & Needleman (1984), Acta Metallurgica 32(1), 157–169,
DOI `10.1016/0001-6160(84)90213-X`. Worth doing only if we want a
non-PF ductile baseline to compare against B.

**E — Crystal plasticity.**
Grain-scale; rate-dependent slip-system flow rules (Asaro–Needleman
1985) plus rotation update. Lower priority — only justified if the
multiscale / texture story becomes part of the thesis. Cite: Roters et
al. (2010) Acta Materialia 58, 1152–1211, DOI
`10.1016/j.actamat.2009.10.058` for the DAMASK-style implementation
template.

**F — Strain-rate sensitivity.**
Trivial bolt-on once A exists: replace the static yield with
Johnson-Cook (Johnson & Cook 1983, *Proc. 7th Int. Symp. on Ballistics*,
The Hague) or Cowper-Symonds
(`sigma_y(alpha, eps_dot) = sigma_y0(alpha) * (1 + (eps_dot/D)^(1/q))`).
Only meaningful in the dynamic-explicit branch; matters for Kalthoff
and similar high-rate benchmarks already in the repo.

## 4. Test plan

- **A**: uniaxial cyclic test with linear isotropic hardening. Reference:
  Abaqus `*PLASTIC` benchmark (Abaqus Verification Manual,
  *Plasticity*). Match yield onset, hardening slope, unload elastic
  recovery to four significant figures.
- **A + B**: SENT specimen with mild steel (E = 200 GPa, sigma_y0 =
  300 MPa, H = 1 GPa, Gc = 50 N/mm). Compare peak load and post-peak
  softening curve to Hai 2026 Fig. 9 (the SENT verification example).
- **B**: UHPC three-point bend from Hai 2026 §5.3 — pseudo-elastoplastic
  fibre-bridging response should reproduce strain hardening followed by
  softening with negligible length-scale dependence.
- **C**: drained triaxial compression on a sand-like Drucker-Prager
  material; cf. Souza Neto et al. 2008 §8 example.
- **F**: Kalthoff-Winkler at 32 m/s with Johnson-Cook steel; compare
  shear-band angle and crack-tip velocity to existing brittle baseline.

## 5. Risks

- **Per-quadrature state breaks the data layout.** Today the solver
  carries only nodal fields and element-mean tensors. Plasticity needs
  one tensor per quadrature point per element. For the current 1-Gauss
  triangles this collapses to per-element, but the moment we move to
  P2 or quad elements (already on the roadmap) the storage and
  rollback-on-Newton-failure logic must be redesigned. Refactor blast
  radius is large and touches `FEMState`, the staggered driver, and the
  autograd path (state must be detached/cloned correctly).
- **Multiple incompatible PF + plasticity formulations.** Pick Hai 2026
  and stick with it; do **not** re-implement Miehe-Aldakheel mid-flight.
  Switching mid-paper rewrites every figure.
- **Inner Newton in return mapping kills throughput.** 5–10 inner
  iterations per quad point per global step is the textbook range; on
  CPU this is fine, on MPS the per-call dispatch overhead will dominate
  unless the inner loop is fully vectorised across elements. Plan a
  closed-form linear-isotropic fast path so the common case stays
  cheap.
- **Consistent tangent and autograd interplay.** The differentiable-
  inversion machinery currently relies on the analytic elastic tangent;
  swapping in `C^alg` must keep `torch.autograd` happy through the
  Simo-Taylor projection — testable with `gradcheck` on a single
  element before integration.

## 6. References

- Abaqus Theory Manual, *Classical metal plasticity*. Dassault Systèmes
  documentation portal, accessed 2026-05-06.
- Asaro, R. J., & Needleman, A. (1985). Texture development and strain
  hardening in rate dependent polycrystals. *Acta Metallurgica*, 33(6),
  923–953. DOI `10.1016/0001-6160(85)90188-9`.
- COMSOL Multiphysics 6.2, *Structural Mechanics Module User's Guide*,
  Theory chapter (Elastoplastic Materials, Soil Plasticity).
- de Souza Neto, E. A., Peric, D., & Owen, D. R. J. (2008).
  *Computational Methods for Plasticity*. Wiley. ISBN
  978-0-470-69462-6.
- Duan, Y., Ren, H., Bie, Y., Zhuang, X., & Rabcuk, T. (2026). A unified
  variational damage model and an efficient length scale insensitive
  phase-field model. *J. Mech. Phys. Solids*, 208, 106494. DOI
  `10.1016/j.jmps.2025.106494`.
- Hai, L., Zhao, X.-L., Zhang, H., Huang, Y.-J., & Wriggers, P. (2026).
  A thermodynamically consistent and length scale-insensitive
  phase-field methodology suitable for ductile fracture with coupled
  elastic-plastic driving forces. *J. Mech. Phys. Solids*. DOI
  `10.1016/j.jmps.2026.106591`.
- Johnson, G. R., & Cook, W. H. (1983). A constitutive model and data
  for metals subjected to large strains, high strain rates and high
  temperatures. *Proc. 7th Int. Symp. on Ballistics*, The Hague,
  541–547.
- Miehe, C., Aldakheel, F., & Teichtmeister, S. (2017). Phase-field
  modeling of ductile fracture at finite strains: A variational
  gradient-extended plasticity-damage theory. *Int. J. Numer. Methods
  Engng*, 109(7), 1051–1082. DOI `10.1002/nme.5234`.
- Roters, F., Eisenlohr, P., Hantcherli, L., Tjahjanto, D. D., Bieler,
  T. R., & Raabe, D. (2010). Overview of constitutive laws, kinematics,
  homogenization and multiscale methods in crystal plasticity finite
  element modeling. *Acta Materialia*, 58(4), 1152–1211. DOI
  `10.1016/j.actamat.2009.10.058`.
- Simo, J. C., & Hughes, T. J. R. (1998). *Computational Inelasticity*.
  Springer. ISBN 978-0-387-97520-7.
- Simo, J. C., & Taylor, R. L. (1985). Consistent tangent operators for
  rate-independent elastoplasticity. *Computer Methods in Applied
  Mechanics and Engineering*, 48(1), 101–118. DOI
  `10.1016/0045-7825(85)90070-2`.
- Tvergaard, V., & Needleman, A. (1984). Analysis of the cup-cone
  fracture in a round tensile bar. *Acta Metallurgica*, 32(1),
  157–169. DOI `10.1016/0001-6160(84)90213-X`.
