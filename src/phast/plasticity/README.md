# `phast.plasticity` — standalone elastoplastic kernels

**Status: SCAFFOLD.** This module is the FIRST building block of the
plasticity track (epic [#262](https://github.com/CEMS-Lab/PhAST/issues/262),
J2 sub-issue [#242](https://github.com/CEMS-Lab/PhAST/issues/242)).
It is **not yet coupled to the phase-field solver**. Coupled
PF + plasticity (Ambati 2015 / Borden 2016 / Miehe 2016 ductile
fracture) is a separate research-class PR tracked under #262.

## What this gives you

A quadrature-point return-mapping integrator that, given the previous
step's stress + plastic-strain history and the current total strain,
returns the updated state. It does NOT touch the FEM assembly, the
staggered solver, or the phase-field damage field.

```python
from phast.material import Material
from phast.plasticity import J2Plasticity, J2State

mat = Material(
    E=210000.0, nu=0.3,                # MPa
    plasticity_model='j2_isotropic',
    yield_stress=250.0,                # MPa
    hardening_modulus=1000.0,          # MPa (linear iso slope)
    hardening_type='linear_iso',
    plane_stress=False,                # plane strain
)
kernel = J2Plasticity(mat)

state = J2State.zeros((batch,), dtype=torch.float64)
strain_n   = ...   # (batch, 6) Voigt-6 [xx, yy, zz, xy, yz, xz], engineering shear
strain_np1 = ...
sigma, plastic_strain_np1, eps_p_eq_np1 = kernel.step(
    strain_n, strain_np1,
    state.stress, state.plastic_strain, state.eps_p_eq,
)
```

## Models supported

| `plasticity_model` | Status | Notes |
| --- | --- | --- |
| `none` | Pass-through | Default. Material extension preserves all elastic behaviour bit-for-bit. |
| `j2_isotropic` | **Implemented** | Rate-independent J2 + isotropic hardening (linear / Voce / Swift). Plane strain + plane stress. |
| `j2_kinematic` | Stub | Reserved for follow-up PR (needs back-stress α in `J2State`). |
| `drucker_prager` | Stub | Reserved for geomechanics follow-up. |

| `hardening_type` | Formula | Required Material fields |
| --- | --- | --- |
| `none` / `linear_iso` | `R(eps_p_eq) = H * eps_p_eq` | `hardening_modulus` |
| `voce` | `R = Q_inf * (1 - exp(-b*eps_p_eq)) + H*eps_p_eq` | `voce_q_inf`, `voce_b`, `hardening_modulus` (optional linear add-on) |
| `swift` | `R = K * (eps0 + eps_p_eq)^n - sigma_y0 + H*eps_p_eq` | `swift_K`, `swift_n`, `swift_eps0` (default 0), `hardening_modulus` (optional) |

All three obey `R(0) = 0` so the initial yield surface stays at
`sigma_y0`.

## When to use which

- **`linear_iso`**: cheapest, closed-form scalar return; good for
  textbook verification, mild hardening.
- **`voce`**: saturating hardening, captures the soft-then-stiffen
  shape seen in tempered alloys.
- **`swift`**: power-law, classical sheet-metal forming (cold-worked
  steels, aluminium).

## Plane strain vs plane stress

Both branches carry the **full 3D Voigt-6 stress / strain tensor** so
the von Mises deviator is computed correctly under triaxial stress:

- **Plane strain** (`plane_stress=False`): we enforce
  `eps_zz = eps_yz = eps_xz = 0` on the input strain; `sigma_zz`
  develops elastically and (after yielding) plastically. The
  through-thickness component matters because plane-strain stress
  states have a non-trivial hydrostatic component that the deviator
  must account for.

- **Plane stress** (`plane_stress=True`): `sigma_zz = sigma_yz = sigma_xz = 0`
  is enforced via a nested Newton iteration on `eps_zz` (Simo–Taylor
  1986). Convergence tol is `1e-10` MPa by default; max 30 outer
  Newton steps.

## Units

Consistent mm-tonne-N-s (MPa) — same as the rest of the solver. See
[`phast/units.py`](../units.py).

## Float64

The kernel is float64 throughout. Float32 will work mechanically but
the inner Newton tolerances (Voce/Swift, plane-stress nested loop)
were tuned for double precision and will likely overshoot at single
precision.

## Autograd

The kernel is implemented in plain torch ops; gradients flow through
the return-mapping path (the plastic multiplier is differentiable wrt
material params and strain). This makes the kernel compatible with
the solver's autograd-driven inverse-problem workflow once coupling
is wired in.

## Coupling to the phase-field solver — NOT YET DONE

The phase-field staggered solver, mechanics solver, and FEM operators
do **not** call into this module. To couple, the planned changes are:

1. Add a per-quadrature `J2State` field to `FEMOperators`'s element
   data.
2. In the mechanics solver inner Newton, replace the elastic stress
   evaluation with a call to `J2Plasticity.step(...)`.
3. Add a coupled energy split (Ambati 2015 / Borden 2016 / Miehe 2016)
   that drives the phase-field damage from a plastic-work-augmented
   driving force.
4. New benchmark target: shear-localisation / strip-yielding with
   ductile crack.

This is a separate PR tracked under epic #262.

## References

- Simo, J.C. & Hughes, T.J.R. (1998), *Computational Inelasticity*,
  Springer. §3.4 (radial return), §3.7 (plane stress).
- de Souza Neto, E.A., Perić, D. & Owen, D.R.J. (2008),
  *Computational Methods for Plasticity*, Wiley. §9.4.
- Belytschko, T., Liu, W.K. & Moran, B. (2000), *Nonlinear Finite
  Elements for Continua and Structures*, Wiley. §5.4.
- Ambati, M., Gerasimov, T. & De Lorenzis, L. (2015),
  "Phase-field modeling of ductile fracture", *Comput. Mech.* 55,
  1017–1040.
- Borden, M.J., Hughes, T.J.R., Landis, C.M., Anvari, A. & Lee, I.J.
  (2016), "A phase-field formulation for fracture in ductile
  materials", *Comput. Methods Appl. Mech. Engrg.* 312, 130–166.
- Miehe, C., Aldakheel, F. & Raina, A. (2016), "Phase field modeling
  of ductile fracture at finite strains: A variational
  gradient-extended plasticity-damage theory", *Int. J. Plast.* 84,
  1–32.
