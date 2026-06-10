"""
J2 (von Mises) plasticity — rate-independent radial-return kernel.

This is a *standalone* quadrature-point return-mapping integrator. It
takes a strain increment plus the previous step's stress / plastic
history and returns the updated stress + history. It does NOT touch the
FEM assembly, the staggered solver, or the phase-field damage field —
coupling is a separate PR (issue #262).

Formulation
-----------
We follow Simo & Hughes (1998), "Computational Inelasticity", §3.4
("J2 flow theory: radial return"). For a 3D state:

    Trial elastic step:
        sigma_trial = sigma_n + C : (eps_{n+1} - eps_n)

    Yield function:
        f(sigma, eps_p_eq) = ||s_dev|| - sqrt(2/3) * (sigma_y0 + R(eps_p_eq))

    where ``s_dev = dev(sigma)``, ``eps_p_eq`` is the accumulated
    equivalent plastic strain, and ``R`` is the hardening law:

        - ``linear_iso``: R = H * eps_p_eq
        - ``voce``      : R = Q_inf * (1 - exp(-b * eps_p_eq))
        - ``swift``     : R = K * (eps0 + eps_p_eq)^n  - sigma_y0
                          (so that R(0) = 0; sigma_yield(0) = sigma_y0)

    If f_trial > 0, the plastic multiplier dgamma is found by solving
    the (scalar) consistency equation in the deviatoric direction
    (radial return):

        ||s_trial|| - 2*mu*dgamma = sqrt(2/3) * (sigma_y0 + R(eps_p_eq + sqrt(2/3)*dgamma))

    For linear hardening this is closed-form; for Voce/Swift we use a
    local Newton iteration. The final updated state is:

        n_hat       = s_trial / ||s_trial||
        sigma_{n+1} = sigma_trial - 2*mu*dgamma * n_hat
        eps_p_{n+1} = eps_p_n + dgamma * n_hat
        eps_p_eq_{n+1} = eps_p_eq_n + sqrt(2/3) * dgamma

Plane strain
------------
We carry the full 3D stress / strain tensor in Voigt-6 form. Plane
strain means eps_zz = 0 (and eps_xz = eps_yz = 0). The von Mises
deviator is computed from the FULL 3D state, which is essential —
under plane strain, sigma_zz develops elastically (=ν(σ_xx+σ_yy)
in the elastic regime) and contributes to the deviator.

Plane stress
------------
Plane stress (sigma_zz = 0) is harder for J2 because the
through-thickness component of the plastic strain rate is unknown a
priori. We use the Simo–Taylor (1986) nested approach:

    Outer: standard radial return on the 5-component deviatoric system
           assuming a known eps_zz_trial.
    Inner: Newton on eps_zz to enforce sigma_zz_{n+1} = 0.

A simpler closed-form implementation by de Souza Neto et al. (2008)
reformulates plane-stress J2 directly in 3-component (σ_xx, σ_yy,
τ_xy) space, but that approach is harder to extend to mixed/kinematic
hardening, so we keep the nested Newton.

Reference
---------
- Simo, J.C. & Hughes, T.J.R., *Computational Inelasticity*, Springer
  (1998), §3.4 (radial return), §3.7 (plane stress).
- de Souza Neto, E.A., Perić, D. & Owen, D.R.J., *Computational
  Methods for Plasticity*, Wiley (2008), §9.4.
- Belytschko, T., Liu, W.K. & Moran, B., *Nonlinear Finite Elements
  for Continua and Structures*, Wiley (2000), §5.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Voigt-6 conventions (3D)
# ---------------------------------------------------------------------------
#
# Stress Voigt-6:   [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz]   (engineering shear)
# Strain Voigt-6:   [ε_xx, ε_yy, ε_zz, 2ε_xy, 2ε_yz, 2ε_xz]   (engineering shear)
#
# We follow the standard convention where strain Voigt entries 4-6 are
# *engineering* shears (γ = 2ε), so that contracted products
# σ:ε = sum(σ_voigt * ε_voigt) hold. Note the deviator and norms below
# reflect this.

_INV3 = 1.0 / 3.0
_SQRT_2_3 = math.sqrt(2.0 / 3.0)


def _trace3(voigt6: torch.Tensor) -> torch.Tensor:
    """Trace of a 3D tensor stored in Voigt-6 form."""
    return voigt6[..., 0] + voigt6[..., 1] + voigt6[..., 2]


def _stress_deviator_voigt6(stress: torch.Tensor) -> torch.Tensor:
    """Deviatoric part of a Voigt-6 stress tensor.

    Voigt-6 layout: [xx, yy, zz, xy, yz, xz] (engineering shear off-diag).
    The pressure is p = trace/3; only the diagonal entries (0..2) are
    shifted; the shear entries (3..5) pass through unchanged.
    """
    p = _trace3(stress) * _INV3
    s = stress.clone()
    s[..., 0] = s[..., 0] - p
    s[..., 1] = s[..., 1] - p
    s[..., 2] = s[..., 2] - p
    return s


def _stress_dev_norm(s: torch.Tensor) -> torch.Tensor:
    """L2 norm of a Voigt-6 deviatoric stress tensor.

    For symmetric tensor norms ||s|| = sqrt(s_ij s_ij), which in Voigt
    form (diagonal stored once, off-diagonal stored once) requires a
    factor 2 on the off-diagonal entries:

        ||s||^2 = s_xx^2 + s_yy^2 + s_zz^2 + 2(s_xy^2 + s_yz^2 + s_xz^2)
    """
    sq = (
        s[..., 0] ** 2 + s[..., 1] ** 2 + s[..., 2] ** 2
        + 2.0 * (s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2)
    )
    return torch.sqrt(sq)


def _stress_dev_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Metric inner product for symmetric deviatoric stress Voigt-6 tensors."""

    return (
        a[..., 0] * b[..., 0]
        + a[..., 1] * b[..., 1]
        + a[..., 2] * b[..., 2]
        + 2.0 * (
            a[..., 3] * b[..., 3]
            + a[..., 4] * b[..., 4]
            + a[..., 5] * b[..., 5]
        )
    )


def _elastic_C_voigt6(
    lam: float, mu: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """3D isotropic elastic stiffness matrix, Voigt-6 (engineering shear).

    C_ijkl in Voigt-6 with 2ε engineering shear:

        [λ+2μ   λ      λ      0    0    0  ]
        [λ      λ+2μ   λ      0    0    0  ]
        [λ      λ      λ+2μ   0    0    0  ]
        [0      0      0      μ    0    0  ]
        [0      0      0      0    μ    0  ]
        [0      0      0      0    0    μ  ]
    """
    C = torch.zeros((6, 6), dtype=dtype, device=device)
    a = lam + 2.0 * mu
    C[0, 0] = a; C[1, 1] = a; C[2, 2] = a
    C[0, 1] = lam; C[1, 0] = lam
    C[0, 2] = lam; C[2, 0] = lam
    C[1, 2] = lam; C[2, 1] = lam
    C[3, 3] = mu; C[4, 4] = mu; C[5, 5] = mu
    return C


# ---------------------------------------------------------------------------
# Hardening law evaluation
# ---------------------------------------------------------------------------


def _hardening_R_and_dR(
    eps_p_eq: torch.Tensor,
    sigma_y0: float,
    H: float,
    hardening_type: str,
    voce_q_inf: Optional[float],
    voce_b: Optional[float],
    swift_K: Optional[float],
    swift_n: Optional[float],
    swift_eps0: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (R(eps_p_eq), dR/d(eps_p_eq)) for the chosen hardening law.

    R(0) = 0 by construction so the initial yield surface stays at sigma_y0.
    """
    if hardening_type in ('none', 'linear_iso', 'linear_kin'):
        # Pure linear (no kinematic split here, since this is the iso part).
        return H * eps_p_eq, torch.full_like(eps_p_eq, H)
    if hardening_type == 'voce':
        # R = Q_inf * (1 - exp(-b * eps_p_eq))
        Q = float(voce_q_inf)
        b = float(voce_b)
        e = torch.exp(-b * eps_p_eq)
        R = Q * (1.0 - e)
        dR = Q * b * e + H  # allow linear+saturation
        return R + H * eps_p_eq, dR
    if hardening_type == 'swift':
        # R + sigma_y0 = K * (eps0 + eps_p_eq)^n  ->  R = K*(eps0+eps_p_eq)^n - sigma_y0
        K = float(swift_K)
        n = float(swift_n)
        eps0 = float(swift_eps0) if swift_eps0 is not None else 0.0
        x = eps0 + eps_p_eq
        # Avoid log(0) when eps0=0 and eps_p_eq=0:
        x_safe = torch.clamp(x, min=1e-300)
        Rn = K * torch.pow(x_safe, n) - sigma_y0
        dRn = K * n * torch.pow(x_safe, n - 1.0)
        return Rn + H * eps_p_eq, dRn + H
    raise ValueError(f"Unknown hardening_type: {hardening_type!r}")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class J2State:
    """Per-quadrature-point J2 plastic state.

    All Voigt-6 tensors follow the layout
    ``[xx, yy, zz, xy, yz, xz]`` with engineering-shear convention on
    strain (the off-diagonal slot stores 2 ε).

    Fields
    ------
    stress : (..., 6)
        Cauchy stress in Voigt-6.
    plastic_strain : (..., 6)
        Plastic strain in Voigt-6 (engineering shear). Total strain
        = plastic + elastic (small-strain regime).
    eps_p_eq : (...,)
        Accumulated equivalent plastic strain.
    """
    stress: torch.Tensor
    plastic_strain: torch.Tensor
    eps_p_eq: torch.Tensor

    @classmethod
    def zeros(cls, batch_shape, dtype=torch.float64, device='cpu'):
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        zero6 = torch.zeros((*batch_shape, 6), dtype=dtype, device=device)
        zero1 = torch.zeros(batch_shape, dtype=dtype, device=device)
        return cls(stress=zero6.clone(),
                   plastic_strain=zero6.clone(),
                   eps_p_eq=zero1)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class J2Plasticity:
    """Standalone J2 (von Mises) return-mapping integrator.

    Parameters
    ----------
    material : Material
        Must have ``plasticity_model`` set to one of
        ``'j2_isotropic'`` or ``'j2_kinematic'`` (kinematic placeholder
        — only isotropic + Voce/Swift are wired here; kinematic
        hardening is a follow-up). ``yield_stress``, ``hardening_modulus``,
        ``hardening_type``, and the optional Voce/Swift parameters are
        read off the Material.
    plane_stress : bool, optional
        If ``None`` (default), follows ``material.plane_stress``. The
        plane-stress branch uses a nested Newton iteration on the
        through-thickness elastic strain to enforce ``sigma_zz = 0``.
    dtype : torch.dtype
        Default ``float64``. Float32 is allowed but discouraged for J2
        (the inner Newton tolerances assume double precision).
    device : str or torch.device
        Default ``'cpu'``.

    Notes
    -----
    All tensors are stored in Voigt-6 with engineering-shear strain
    convention. The kernel is fully batched (the leading dims of all
    state tensors are broadcast freely), so a single call can update
    every quadrature point in a mesh in one shot.

    The kernel is implemented in plain torch ops; gradients flow
    through the return-mapping path (the plastic multiplier is
    differentiable wrt material params and strain), making the
    kernel compatible with the solver's autograd-driven inverse-
    problem workflow.
    """

    def __init__(
        self,
        material,
        plane_stress: Optional[bool] = None,
        dtype: torch.dtype = torch.float64,
        device='cpu',
        plane_stress_tol: float = 1e-10,
        plane_stress_max_iter: int = 30,
    ):
        if material.plasticity_model == 'none':
            raise ValueError(
                "J2Plasticity requires Material with plasticity_model "
                "set to 'j2_isotropic' (or 'j2_kinematic'); got 'none'.")
        if material.plasticity_model not in ('j2_isotropic', 'j2_kinematic'):
            raise ValueError(
                f"J2Plasticity only supports plasticity_model in "
                f"('j2_isotropic', 'j2_kinematic'); got "
                f"{material.plasticity_model!r}.")
        if material.plasticity_model == 'j2_kinematic':
            # Kinematic hardening requires a back-stress alpha; not yet
            # plumbed through J2State. Document and reject explicitly.
            raise NotImplementedError(
                "j2_kinematic hardening is reserved for a follow-up PR "
                "(needs back-stress alpha in J2State).")
        self.material = material
        self.lam = float(material.lam) if not torch.is_tensor(material.lam) \
            else float(material.lam.item())
        self.mu = float(material.mu) if not torch.is_tensor(material.mu) \
            else float(material.mu.item())
        self.E = float(material.E)
        self.nu = float(material.nu)
        self.sigma_y0 = float(material.yield_stress)
        self.H = float(material.hardening_modulus)
        self.hardening_type = material.hardening_type
        self.voce_q_inf = material.voce_q_inf
        self.voce_b = material.voce_b
        self.swift_K = material.swift_K
        self.swift_n = material.swift_n
        self.swift_eps0 = material.swift_eps0
        self.plane_stress = (
            material.plane_stress if plane_stress is None else plane_stress
        )
        self.dtype = dtype
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.plane_stress_tol = plane_stress_tol
        self.plane_stress_max_iter = plane_stress_max_iter
        self._C = _elastic_C_voigt6(self.lam, self.mu, dtype, self.device)

    # ------------------------------------------------------------------
    # Tangent / elastic helpers
    # ------------------------------------------------------------------

    @property
    def C_voigt6(self) -> torch.Tensor:
        """3D isotropic elastic stiffness matrix, Voigt-6."""
        return self._C

    def hardening_R(self, eps_p_eq: torch.Tensor) -> torch.Tensor:
        """Return R(eps_p_eq), the hardening function value (without sigma_y0)."""
        R, _ = _hardening_R_and_dR(
            eps_p_eq, self.sigma_y0, self.H, self.hardening_type,
            self.voce_q_inf, self.voce_b, self.swift_K, self.swift_n,
            self.swift_eps0,
        )
        return R

    def yield_stress_at(self, eps_p_eq: torch.Tensor) -> torch.Tensor:
        """Current uniaxial yield stress including hardening: σ_y0 + R(eps_p_eq)."""
        return self.sigma_y0 + self.hardening_R(eps_p_eq)

    # ------------------------------------------------------------------
    # Core scalar return-mapping (3D)
    # ------------------------------------------------------------------

    def _solve_dgamma(
        self,
        s_trial_norm: torch.Tensor,
        eps_p_eq_n: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the scalar consistency equation for the plastic multiplier.

        Equation (radial return):

            phi(dgamma) = ||s_trial|| - 2*mu*dgamma
                          - sqrt(2/3) * (sigma_y0 + R(eps_p_eq_n + sqrt(2/3)*dgamma))
                        = 0

        For linear isotropic hardening (R = H * eps_p_eq) this is
        closed-form:

            dgamma = (||s_trial|| - sqrt(2/3)*(sigma_y0 + H*eps_p_eq_n))
                     / (2*mu + 2/3 * H)

        For Voce / Swift / mixed laws, do a local Newton.
        """
        two_mu = 2.0 * self.mu
        sqrt23 = _SQRT_2_3
        if self.hardening_type in ('none', 'linear_iso'):
            num = s_trial_norm - sqrt23 * (self.sigma_y0 + self.H * eps_p_eq_n)
            den = two_mu + (2.0 / 3.0) * self.H
            dgamma = num / den
            return torch.clamp(dgamma, min=0.0)
        # Voce/Swift: Newton iterate, init from linear estimate.
        dgamma = torch.zeros_like(s_trial_norm)
        for _ in range(50):
            eps_p_eq_new = eps_p_eq_n + sqrt23 * dgamma
            R, dR = _hardening_R_and_dR(
                eps_p_eq_new, self.sigma_y0, self.H, self.hardening_type,
                self.voce_q_inf, self.voce_b, self.swift_K, self.swift_n,
                self.swift_eps0,
            )
            phi = (s_trial_norm - two_mu * dgamma
                   - sqrt23 * (self.sigma_y0 + R))
            # dphi/d(dgamma) = -2mu - 2/3 * dR
            dphi = -two_mu - (2.0 / 3.0) * dR
            step = -phi / dphi
            dgamma = dgamma + step
            if torch.max(torch.abs(step)).item() < 1e-12:
                break
        return torch.clamp(dgamma, min=0.0)

    # ------------------------------------------------------------------
    # Public step (dispatch on plane_stress)
    # ------------------------------------------------------------------

    def step(
        self,
        strain_n: torch.Tensor,
        strain_np1: torch.Tensor,
        stress_n: torch.Tensor,
        plastic_strain_n: torch.Tensor,
        eps_p_eq_n: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one return-mapping step.

        All Voigt-6 tensors follow the layout
        ``[xx, yy, zz, xy, yz, xz]`` with engineering-shear strain.

        Parameters
        ----------
        strain_n, strain_np1 : (..., 6) float64
            Total strain at the previous and current step. For plane
            problems, supply the full 3D strain — entries 2 (zz), 4
            (yz), 5 (xz) are 0 in plane strain; in plane stress, entry 2
            (zz) is computed inside this routine and the value passed in
            is ignored.
        stress_n : (..., 6) float64
            Stress at the previous step (assumed admissible: f<=0).
        plastic_strain_n : (..., 6) float64
            Plastic strain at the previous step.
        eps_p_eq_n : (...,) float64
            Accumulated equivalent plastic strain at the previous step.

        Returns
        -------
        stress_np1, plastic_strain_np1, eps_p_eq_np1
            Updated state.
        """
        if self.plane_stress:
            return self._step_plane_stress(
                strain_n, strain_np1, stress_n, plastic_strain_n, eps_p_eq_n
            )
        return self._step_3d(
            strain_n, strain_np1, stress_n, plastic_strain_n, eps_p_eq_n
        )

    def step_with_tangent(
        self,
        strain_n: torch.Tensor,
        strain_np1: torch.Tensor,
        stress_n: torch.Tensor,
        plastic_strain_n: torch.Tensor,
        eps_p_eq_n: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stress/history update plus algorithmic tangent.

        For 3D/plane-strain J2 this uses the explicit Simo-Taylor
        radial-return tangent. Plane stress is reduced from the same
        consistent 3D tangent after the through-thickness strain has been
        solved. Voce/Swift hardening still use the scalar local Newton for
        the consistency equation, but the tangent itself is explicit.
        """

        if self.plane_stress:
            strain_final = self._solve_plane_stress_strain(
                strain_n, strain_np1, stress_n, plastic_strain_n, eps_p_eq_n
            )
            stress, plastic_strain, eps_p_eq = self._step_3d(
                strain_n, strain_final, stress_n, plastic_strain_n, eps_p_eq_n
            )
            C3d = self._algorithmic_tangent_3d(
                strain_n, strain_final, stress_n, plastic_strain_n, eps_p_eq_n
            )
            return (
                stress,
                plastic_strain,
                eps_p_eq,
                self._plane_stress_tangent_from_3d(C3d),
            )

        stress, plastic_strain, eps_p_eq = self.step(
            strain_n, strain_np1, stress_n, plastic_strain_n, eps_p_eq_n)
        C_alg = self._algorithmic_tangent_3d(
            strain_n, strain_np1, stress_n, plastic_strain_n, eps_p_eq_n)
        return stress, plastic_strain, eps_p_eq, C_alg

    def algorithmic_tangent(
        self,
        strain_n: torch.Tensor,
        strain_np1: torch.Tensor,
        stress_n: torch.Tensor,
        plastic_strain_n: torch.Tensor,
        eps_p_eq_n: torch.Tensor,
    ) -> torch.Tensor:
        """Explicit J2 algorithmic tangent for the active strain measure.

        For plane stress, this returns the Schur-complement reduction of the
        consistent 3D tangent after solving for the through-thickness strain
        that enforces ``sigma_zz = 0``.
        """

        if self.plane_stress:
            strain_final = self._solve_plane_stress_strain(
                strain_n, strain_np1, stress_n, plastic_strain_n, eps_p_eq_n
            )
            C3d = self._algorithmic_tangent_3d(
                strain_n, strain_final, stress_n, plastic_strain_n, eps_p_eq_n
            )
            return self._plane_stress_tangent_from_3d(C3d)
        return self._algorithmic_tangent_3d(
            strain_n, strain_np1, stress_n, plastic_strain_n, eps_p_eq_n)

    # ------------------------------------------------------------------
    # Plane-strain / fully-3D path
    # ------------------------------------------------------------------

    def _step_3d(
        self,
        strain_n: torch.Tensor,
        strain_np1: torch.Tensor,
        stress_n: torch.Tensor,
        plastic_strain_n: torch.Tensor,
        eps_p_eq_n: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Trial elastic step (small-strain, additive split):
        # sigma_trial = stress_n + C : Δε
        d_eps = strain_np1 - strain_n
        d_sigma_trial = torch.einsum('ij,...j->...i', self._C, d_eps)
        sigma_trial = stress_n + d_sigma_trial
        # Deviator + norm
        s_trial = _stress_deviator_voigt6(sigma_trial)
        s_norm = _stress_dev_norm(s_trial)
        # Yield function f_trial
        sigma_y_n = self.yield_stress_at(eps_p_eq_n)
        f_trial = s_norm - _SQRT_2_3 * sigma_y_n
        # Branchless update with mask
        plastic_mask = (f_trial > 0.0)
        if not torch.any(plastic_mask):
            # Pure elastic step
            return sigma_trial, plastic_strain_n.clone(), eps_p_eq_n.clone()
        # Solve scalar consistency
        # Avoid division by zero in n_hat where s_norm=0 (only matters where
        # the mask says elastic — guard with a tiny floor).
        s_norm_safe = torch.clamp(s_norm, min=1e-300)
        n_hat = s_trial / s_norm_safe.unsqueeze(-1)
        dgamma = self._solve_dgamma(s_norm, eps_p_eq_n)
        # Apply only on plastic points; elastic points get dgamma=0.
        dgamma = torch.where(plastic_mask, dgamma, torch.zeros_like(dgamma))
        # Stress correction: subtract 2*mu*dgamma * n_hat from the deviator
        # (sigma_trial = pressure*I + s_trial; pressure unchanged).
        two_mu = 2.0 * self.mu
        sigma_np1 = sigma_trial - two_mu * dgamma.unsqueeze(-1) * n_hat
        # Plastic strain update: ε_p_{n+1} = ε_p_n + dgamma * n_hat.
        # Note: in Voigt-6 with engineering shear, the plastic-strain
        # off-diagonal slot is 2 ε_p_xy. n_hat in Voigt-6 (stress side)
        # has off-diagonal slot s_xy / ||s||. The flow rule is
        # dε_p = dgamma * n  with n the symmetric tensor; in Voigt
        # engineering-shear strain form the off-diagonal entry is
        # 2 dε_p_xy = 2 * dgamma * n_xy. So we need to multiply the
        # off-diagonal n_hat entries by 2 when storing on the strain side.
        n_hat_strain = n_hat.clone()
        n_hat_strain[..., 3] = n_hat_strain[..., 3] * 2.0
        n_hat_strain[..., 4] = n_hat_strain[..., 4] * 2.0
        n_hat_strain[..., 5] = n_hat_strain[..., 5] * 2.0
        plastic_strain_np1 = plastic_strain_n + dgamma.unsqueeze(-1) * n_hat_strain
        eps_p_eq_np1 = eps_p_eq_n + _SQRT_2_3 * dgamma
        return sigma_np1, plastic_strain_np1, eps_p_eq_np1

    def _algorithmic_tangent_3d(
        self,
        strain_n: torch.Tensor,
        strain_np1: torch.Tensor,
        stress_n: torch.Tensor,
        plastic_strain_n: torch.Tensor,
        eps_p_eq_n: torch.Tensor,
    ) -> torch.Tensor:
        d_eps = strain_np1 - strain_n
        sigma_trial = stress_n + torch.einsum('ij,...j->...i', self._C, d_eps)
        s_trial = _stress_deviator_voigt6(sigma_trial)
        s_norm = _stress_dev_norm(s_trial)
        sigma_y_n = self.yield_stress_at(eps_p_eq_n)
        f_trial = s_norm - _SQRT_2_3 * sigma_y_n
        plastic_mask = f_trial > 0.0

        C_alg = self._C.expand(*strain_np1.shape[:-1], 6, 6).clone()
        if not torch.any(plastic_mask):
            return C_alg

        s_norm_safe = torch.clamp(s_norm, min=1e-300)
        n_hat = s_trial / s_norm_safe.unsqueeze(-1)
        dgamma = self._solve_dgamma(s_norm, eps_p_eq_n)
        dgamma = torch.where(plastic_mask, dgamma, torch.zeros_like(dgamma))
        eps_p_eq_np1 = eps_p_eq_n + _SQRT_2_3 * dgamma
        _, dR = _hardening_R_and_dR(
            eps_p_eq_np1, self.sigma_y0, self.H, self.hardening_type,
            self.voce_q_inf, self.voce_b, self.swift_K, self.swift_n,
            self.swift_eps0,
        )
        den = 2.0 * self.mu + (2.0 / 3.0) * dR

        cols = []
        for col in range(6):
            trial_col = self._C[:, col].expand_as(strain_np1)
            dev_col = _stress_deviator_voigt6(trial_col)
            dr = _stress_dev_inner(n_hat, dev_col)
            dgamma_col = dr / den
            dn_col = (
                dev_col - n_hat * dr.unsqueeze(-1)
            ) / s_norm_safe.unsqueeze(-1)
            alg_col = (
                trial_col
                - 2.0 * self.mu * (
                    dgamma_col.unsqueeze(-1) * n_hat
                    + dgamma.unsqueeze(-1) * dn_col
                )
            )
            cols.append(alg_col)
        C_plastic = torch.stack(cols, dim=-1)
        C_alg = torch.where(
            plastic_mask[..., None, None], C_plastic, C_alg)
        return C_alg

    def _solve_plane_stress_strain(
        self,
        strain_n: torch.Tensor,
        strain_np1: torch.Tensor,
        stress_n: torch.Tensor,
        plastic_strain_n: torch.Tensor,
        eps_p_eq_n: torch.Tensor,
    ) -> torch.Tensor:
        """Return the converged strain tensor for the plane-stress solve."""

        nu = self.nu
        d_eps_xx = strain_np1[..., 0] - strain_n[..., 0]
        d_eps_yy = strain_np1[..., 1] - strain_n[..., 1]
        d_eps_zz_e = -(nu / (1.0 - nu)) * (d_eps_xx + d_eps_yy)
        eps_zz = strain_n[..., 2] + d_eps_zz_e

        for _it in range(self.plane_stress_max_iter):
            strain_try = strain_np1.clone()
            strain_try[..., 2] = eps_zz
            sigma_try, _, _ = self._step_3d(
                strain_n, strain_try, stress_n,
                plastic_strain_n, eps_p_eq_n
            )
            r = sigma_try[..., 2]
            if torch.max(torch.abs(r)).item() < self.plane_stress_tol:
                break
            h = 1e-7 * (1.0 + torch.abs(eps_zz))
            strain_try2 = strain_try.clone()
            strain_try2[..., 2] = eps_zz + h
            sigma_try2, _, _ = self._step_3d(
                strain_n, strain_try2, stress_n,
                plastic_strain_n, eps_p_eq_n
            )
            jac = (sigma_try2[..., 2] - r) / h
            jac = torch.where(torch.abs(jac) < 1e-300,
                              torch.full_like(jac, 1e-300), jac)
            eps_zz = eps_zz - r / jac

        strain_final = strain_np1.clone()
        strain_final[..., 2] = eps_zz
        return strain_final

    def _plane_stress_tangent_from_3d(self, C3d: torch.Tensor) -> torch.Tensor:
        """Reduce a 3D tangent to the plane-stress Jacobian."""

        active = (0, 1, 3, 4, 5)
        idx_active = torch.tensor(active, device=C3d.device, dtype=torch.long)
        idx_zz = torch.tensor([2], device=C3d.device, dtype=torch.long)

        Caa = C3d.index_select(-2, idx_active).index_select(-1, idx_active)
        Cab = C3d.index_select(-2, idx_active).index_select(-1, idx_zz)
        Cba = C3d.index_select(-2, idx_zz).index_select(-1, idx_active)
        Cbb = C3d[..., 2, 2]
        Cbb = torch.where(torch.abs(Cbb) < 1e-300,
                           torch.full_like(Cbb, 1e-300), Cbb)
        Cred = Caa - Cab * Cba / Cbb[..., None, None]

        C_out = torch.zeros_like(C3d)
        for i, row in enumerate(active):
            for j, col in enumerate(active):
                C_out[..., row, col] = Cred[..., i, j]
        return C_out

    # ------------------------------------------------------------------
    # Plane-stress path (nested Newton on eps_zz)
    # ------------------------------------------------------------------

    def _step_plane_stress(
        self,
        strain_n: torch.Tensor,
        strain_np1: torch.Tensor,
        stress_n: torch.Tensor,
        plastic_strain_n: torch.Tensor,
        eps_p_eq_n: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plane-stress J2 with a nested Newton on ε_zz.

        Outer: solve f(ε_zz) = sigma_zz_after_return(ε_zz) = 0 by Newton
               (numerical Jacobian, since the return mapping is non-smooth
                only at the elastic-plastic transition; we use a
                finite-difference Jacobian for robustness).
        Inner: standard 3D radial return for fixed ε_zz.

        Initial guess for ε_zz uses the elastic plane-stress relation
            ε_zz_e = -(ν / (1-ν)) * (ε_xx + ε_yy)
        which is exact in the purely elastic regime.
        """
        strain_final = self._solve_plane_stress_strain(
            strain_n, strain_np1, stress_n, plastic_strain_n, eps_p_eq_n
        )
        return self._step_3d(
            strain_n, strain_final, stress_n,
            plastic_strain_n, eps_p_eq_n
        )
