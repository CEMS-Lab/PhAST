"""
Material properties for phase-field fracture.

Supports:
  - Isotropic linear elasticity (plane strain)
  - AT1 / AT2 phase-field models
  - Energy splits: isotropic, amor (vol-dev), spectral (Miehe), star_convex (Kumar)
  - Configurable degradation function g(d)
"""

import torch
from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass
class Material:
    """Isotropic elastic material with phase-field fracture parameters.

    Default values match the Akantu benchmark (mm-tonne-N-s-MPa).

    Unit convention
    ---------------
    All numeric fields are stored in the solver's *internal* unit system,
    a consistent mm-tonne-N-s system driven by the mesh generators
    (which produce millimetre coordinates). See ``units.py`` for the full
    table; SI-suffixed strings (e.g. ``"32 GPa"``) are accepted at the
    YAML / ``create_material`` boundary and normalised to internal units
    before reaching this class.

    Parameters
    ----------
    E : float
        Young's modulus (MPa).      [SI equivalent: 1 MPa = 1e6 Pa]
    nu : float
        Poisson's ratio.
    Gc : float
        Fracture toughness (N/mm = kJ/m^2 = 1000 J/m^2).
    l0 : float
        Regularization length (mm). [SI equivalent: 1 mm = 1e-3 m]
    rho : float
        Density (tonne/mm^3).       [SI equivalent: 1 t/mm^3 = 1e12 kg/m^3]
    eta_residual : float
        Residual stiffness in degradation: g(d) = (1-d)^2 + eta_residual.
    energy_split : str
        'amor' (vol-dev), 'spectral' (Miehe §3.1, strain-spectral),
        'spectral_stress' (Miehe §3.2, stress-spectral; opt-in for B7
        COMSOL parity, issue #213), 'isotropic', or 'star_convex' (Kumar).
    pf_model : str
        'AT2' (quadratic), 'AT1' (linear), 'PFCZM' (Wu cohesive
        phase-field model with finite tensile strength), or the legacy
        'allencahn' gradient-flow variant.
    gamma_correction : bool
        If True, apply per-element Gc correction for gamma-convergence
        (Bourdin et al. 2000). Reduces effective Gc on coarse elements to
        compensate for over-prediction when h ~ l0. Default: False.
    degradation_type : str
        'standard' (default): g(d) = (1-d)^2 + eta.
        'cubic': Borden cubic degradation family.
        'rational': Wu (2017) rational degradation — less length-scale sensitive.
    cubic_s : float
        Shape parameter for the Borden cubic degradation family. Used only
        when ``degradation_type == 'cubic'``.
    """
    E: float = 210000.0
    nu: float = 0.3
    Gc: float = 2.7
    l0: float = 0.005
    rho: float = 7.8e-9
    eta_residual: float = 1e-7
    energy_split: Literal[
        'amor', 'spectral', 'spectral_stress', 'isotropic', 'star_convex'
    ] = 'amor'
    pf_model: Literal['AT1', 'AT2', 'PFCZM', 'allencahn'] = 'AT2'
    gamma_correction: bool = False
    degradation_type: Literal['standard', 'cubic', 'rational'] = 'standard'
    cubic_s: float = 1.0
    sigma_ts: float = 0.0  # tensile strength [MPa] for AT1 nucleation enhancement (0=off)
    pfczm_p: int = 2
    pfczm_softening: Literal['linear', 'exponential'] = 'linear'
    # Crack-driving force selector (issue #248):
    #   'strain_energy'    : Ψ⁺ from the configured ``energy_split`` (default;
    #                        bit-identical to legacy behaviour).
    #   'principal_stress' : D = ⟨σ₁⟩²/(2E) — Wu (2020)-style maximum principal
    #                        stress criterion. Useful where tensile strength
    #                        dominates (brittle ceramics, COMSOL parity).
    # Only the H-update / damage residual path branches on this knob; energy
    # bookkeeping (``compute_total_energy``, J-integral postprocess, dataset
    # labels) continues to use the true Ψ⁺ regardless.
    driving_force: Literal['strain_energy', 'principal_stress'] = 'strain_energy'
    plane_stress: bool = False  # True for plane stress (σ_zz=0), False for plane strain (ε_zz=0)
    # AT1 strain-energy source term S_H (Pham 2011, Tanné 2018, Bleyer 2017,
    # COMSOL dynamic-crack-branching Eq. 6). The AT1 weak-form RHS is
    # (2H − S_H) M·1, equivalent to enforcing damage-onset only when
    # ψ⁺ exceeds the elastic threshold W_c0 = 3 Gc / (16 l0)
    # (RHS becomes positive for H > S_H/2 = 3 Gc / (16 l0)).
    # 'auto' → S_H = 3 Gc / (8 l0); explicit float overrides the formula.
    # Ignored when pf_model == 'AT2'.
    at1_threshold: Union[float, str] = 'auto'

    # ------------------------------------------------------------------
    # Plasticity fields (issue #262 / #242, scaffold).
    #
    # NOTE: These are stored on Material so a single object describes the
    # full constitutive model. The plasticity kernel itself is implemented
    # in ``phast.plasticity`` as a *standalone* module — it is
    # NOT yet wired into the staggered phase-field solver. Coupled
    # PF–plasticity (Ambati 2015 / Borden 2016 / Miehe 2016 ductile
    # fracture) is tracked as a separate sub-issue under #262.
    #
    # Default ``plasticity_model = 'none'`` preserves all existing,
    # purely-elastic behaviour bit-for-bit. The remaining fields are read
    # only when ``plasticity_model != 'none'``.
    #
    # Units (consistent mm-tonne-N-s, MPa system, see ``units.py``):
    #   yield_stress       : sigma_y0    [MPa]   (initial yield in tension)
    #   hardening_modulus  : H           [MPa]   (linear hardening slope)
    #   voce_q_inf         : Q_inf       [MPa]   (Voce saturation stress)
    #   voce_b             : b           [-]     (Voce saturation rate)
    #   swift_K            : K           [MPa]   (Swift power-law strength)
    #   swift_n            : n           [-]     (Swift hardening exponent)
    #   swift_eps0         : eps0        [-]     (Swift offset strain)
    plasticity_model: Literal[
        'none', 'j2_isotropic', 'j2_kinematic', 'drucker_prager'
    ] = 'none'
    yield_stress: float = 0.0
    hardening_modulus: float = 0.0
    hardening_type: Literal[
        'none', 'linear_iso', 'linear_kin', 'voce', 'swift'
    ] = 'none'
    voce_q_inf: Optional[float] = None
    voce_b: Optional[float] = None
    swift_K: Optional[float] = None
    swift_n: Optional[float] = None
    swift_eps0: Optional[float] = None

    def __post_init__(self):
        if self.eta_residual <= 0.0:
            self.eta_residual = 1e-7  # minimum to prevent singular stiffness at d=1
        model = str(self.pf_model)
        self.pf_model = 'allencahn' if model.lower() == 'allencahn' else model.upper()
        if self.pf_model not in {'AT1', 'AT2', 'PFCZM', 'allencahn'}:
            raise ValueError(
                f"pf_model must be 'AT1', 'AT2', 'PFCZM', or 'allencahn', got "
                f"{self.pf_model!r}")
        if self.pfczm_p < 2:
            raise ValueError(f"pfczm_p must be >= 2, got {self.pfczm_p}")
        if self.pfczm_softening not in {'linear', 'exponential'}:
            raise ValueError(
                "pfczm_softening must be 'linear' or 'exponential', "
                f"got {self.pfczm_softening!r}")
        if self.pf_model == 'PFCZM' and self.sigma_ts <= 0.0:
            raise ValueError(
                "pf_model='PFCZM' requires sigma_ts > 0 so the Wu "
                "degradation parameter a1 and tensile strength are defined.")
        # Validate at1_threshold
        if isinstance(self.at1_threshold, str):
            if self.at1_threshold != 'auto':
                raise ValueError(
                    f"at1_threshold must be 'auto' or a float, got "
                    f"{self.at1_threshold!r}")
        elif not isinstance(self.at1_threshold, (int, float)):
            raise ValueError(
                f"at1_threshold must be 'auto' or a float, got type "
                f"{type(self.at1_threshold).__name__}")
        elif self.at1_threshold < 0.0:
            raise ValueError(
                f"at1_threshold must be non-negative, got {self.at1_threshold}")
        if self.cubic_s < 0.0:
            raise ValueError(f"cubic_s must be non-negative, got {self.cubic_s}")
        # Plasticity field validation — only when actually engaged.
        if self.plasticity_model != 'none':
            if self.yield_stress <= 0.0:
                raise ValueError(
                    f"plasticity_model={self.plasticity_model!r} requires "
                    f"yield_stress > 0, got {self.yield_stress}")
            if self.hardening_modulus < 0.0:
                raise ValueError(
                    f"hardening_modulus must be non-negative, "
                    f"got {self.hardening_modulus}")
            if self.hardening_type == 'voce':
                if self.voce_q_inf is None or self.voce_b is None:
                    raise ValueError(
                        "hardening_type='voce' requires voce_q_inf and "
                        "voce_b to be set")
                if self.voce_b <= 0.0:
                    raise ValueError(
                        f"voce_b must be > 0, got {self.voce_b}")
            if self.hardening_type == 'swift':
                if self.swift_K is None or self.swift_n is None:
                    raise ValueError(
                        "hardening_type='swift' requires swift_K and "
                        "swift_n to be set")
                if self.swift_K <= 0.0 or self.swift_n <= 0.0:
                    raise ValueError(
                        f"swift_K and swift_n must be > 0, got "
                        f"K={self.swift_K}, n={self.swift_n}")

    @property
    def at1_source(self) -> float:
        """AT1 strain-energy source term S_H in the damage RHS (2H − S_H).

        Returns 0.0 for AT2 (ignored). For AT1, returns the user-specified
        override or the auto-derived value 3 Gc / (8 l0). The corresponding
        elastic threshold for damage onset is W_c0 = S_H / 2 = 3 Gc / (16 l0).
        """
        if self.pf_model != 'AT1':
            return 0.0
        if isinstance(self.at1_threshold, str) and self.at1_threshold == 'auto':
            return 3.0 * self.Gc / (8.0 * self.l0)
        return float(self.at1_threshold)

    @property
    def pfczm_c_alpha(self) -> float:
        return 3.141592653589793

    @property
    def pfczm_a1(self) -> float:
        import math
        if self.sigma_ts <= 0.0:
            raise ValueError("PFCZM a1 requires sigma_ts > 0")
        scale = max(1.0 - float(self.eta_residual), 1.0e-30)
        return 4.0 * self.E * self.Gc / (
            math.pi * self.l0 * self.sigma_ts ** 2 * scale)

    @property
    def pfczm_softening_parameters(self) -> tuple[float, float]:
        if self.pfczm_softening == 'linear':
            return -0.5, 0.0
        # Wu-Nguyen exponential softening parameter with a3=0.
        return 2.0 ** (5.0 / 3.0) - 3.0, 0.0

    def pfczm_alpha(self, d: torch.Tensor) -> torch.Tensor:
        """Wu PF-CZM crack geometric function alpha(d)=2d-d^2."""
        return 2.0 * d - d * d

    def pfczm_alpha_prime(self, d: torch.Tensor) -> torch.Tensor:
        return 2.0 - 2.0 * d

    def pfczm_degradation_derivatives(
            self, d: torch.Tensor,
            a1: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Wu PF-CZM rational degradation g, g' and g''.

        The implementation follows the unified PF-CZM family with
        A=(1-d)^p and Q=a1*d*(1+a2*d+a2*a3*d^2). The residual-stiffness
        blend keeps g(0)=1 and g(1)=eta_residual, matching the other
        degradation families in this solver.
        """
        if a1 is None:
            a1_eff = torch.as_tensor(
                self.pfczm_a1, dtype=d.dtype, device=d.device)
        else:
            a1_eff = a1.to(dtype=d.dtype, device=d.device)
            while a1_eff.ndim < d.ndim:
                a1_eff = a1_eff.unsqueeze(-1)
        a2, a3 = self.pfczm_softening_parameters
        p = int(self.pfczm_p)
        eta = self.eta_residual

        omd = torch.clamp(1.0 - d, min=0.0)
        A = omd ** p
        Ap = -float(p) * omd ** (p - 1)
        if p == 2:
            App = torch.full_like(d, 2.0)
        else:
            App = float(p * (p - 1)) * omd ** (p - 2)

        Q = a1_eff * d * (1.0 + a2 * d + a2 * a3 * d * d)
        Qp = a1_eff * (1.0 + 2.0 * a2 * d + 3.0 * a2 * a3 * d * d)
        Qpp = a1_eff * (2.0 * a2 + 6.0 * a2 * a3 * d)

        S = torch.clamp(A + Q, min=1.0e-30)
        N = Ap * Q - A * Qp
        Np = App * Q - A * Qpp
        ratio = A / S
        ratio_p = N / (S * S)
        ratio_pp = Np / (S * S) - 2.0 * N * (Ap + Qp) / (S * S * S)
        scale = 1.0 - eta
        return (
            scale * ratio + eta,
            scale * ratio_p,
            scale * ratio_pp,
        )

    @property
    def lam(self) -> float:
        """First Lamé parameter (lambda).

        Plane strain: λ = Eν / ((1+ν)(1-2ν))
        Plane stress: λ_ps = Eν / (1-ν²)  (effective 2D Lamé parameter)
        """
        if self.plane_stress:
            return self.E * self.nu / (1.0 - self.nu ** 2)
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> float:
        """Second Lamé parameter (shear modulus)."""
        return self.E / (2 * (1 + self.nu))

    @property
    def kappa(self) -> float:
        """3D bulk modulus: K = E / (3*(1-2*nu)).

        Always returns the true 3D bulk modulus, independent of plane
        stress/strain assumption. Used by the Amor volumetric-deviatoric
        split which decomposes the full 3D strain energy.
        """
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    @property
    def Gc_over_l0(self) -> float:
        return self.Gc / self.l0

    @property
    def Gc_times_l0(self) -> float:
        return self.Gc * self.l0

    @property
    def c_p(self):
        """P-wave speed (mm/s or m/s equivalent in the unit system).

        Plane strain:  c_p = sqrt((λ + 2μ) / ρ)
        Plane stress:  c_p = sqrt(E / (ρ(1-ν²)))  [= sqrt((λ_ps + 2μ) / ρ)]

        Returns float when material params are floats; returns torch.Tensor
        when any of (E, nu, rho) are tensors. The tensor branch keeps c_p
        in the autograd graph if the material parameters require_grad —
        but note that c_p is typically used for the CFL dt computation,
        which is a scalar that does NOT need to be in the gradient path.
        Use float(mat.c_p) or mat.c_p.detach().item() in that context.
        """
        x = (self.lam + 2 * self.mu) / self.rho
        if torch.is_tensor(x):
            return torch.sqrt(x)
        import math
        return math.sqrt(x)

    @property
    def c_s(self):
        """S-wave (shear) speed: c_s = sqrt(mu / rho).

        Same for plane strain and plane stress (shear modulus is
        independent of the plane assumption).

        Returns float or torch.Tensor depending on input types.
        """
        x = self.mu / self.rho
        if torch.is_tensor(x):
            return torch.sqrt(x)
        import math
        return math.sqrt(x)

    @property
    def c_R(self):
        """Rayleigh wave speed (approximate).

        Uses the Viktorov (1967) approximation:
            c_R = c_s * (0.862 + 1.14*nu) / (1 + nu)

        Accurate to <0.5% for all Poisson's ratios. The Rayleigh wave
        is the fastest surface wave and sets the theoretical crack speed
        limit in brittle fracture. Empirical branching onset is ~0.6*c_R
        (Fineberg & Marder 1999).

        Returns float or torch.Tensor depending on input types.
        """
        cs = self.c_s
        nu = self.nu
        factor = (0.862 + 1.14 * nu) / (1 + nu)
        return cs * factor

    def C_matrix(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """2D constitutive matrix (Voigt: [exx, eyy, gxy]).

        Uses plane strain or plane stress depending on material setting.
        Built via torch.stack so it preserves the autograd graph when
        lam and mu are tensors with requires_grad=True.

        Returns
        -------
        C : (3, 3) tensor
        """
        if device is None:
            from ..utils.device import detect_device
            device = detect_device()
        lam, mu = self.lam, self.mu
        # Use torch.stack to keep gradients flowing if lam, mu are tensors.
        # If they're floats, wrap them as 0-d tensors first.
        if not torch.is_tensor(lam):
            lam = torch.tensor(lam, dtype=dtype, device=device)
        if not torch.is_tensor(mu):
            mu = torch.tensor(mu, dtype=dtype, device=device)
        zero = torch.zeros((), dtype=lam.dtype, device=lam.device)
        row0 = torch.stack([lam + 2 * mu, lam,             zero])
        row1 = torch.stack([lam,          lam + 2 * mu,    zero])
        row2 = torch.stack([zero,         zero,            mu])
        return torch.stack([row0, row1, row2])

    def C_plane_strain(self, device=None, dtype=torch.float64) -> torch.Tensor:
        """Constitutive matrix (Voigt: [exx, eyy, gxy]). Legacy alias for C_matrix."""
        return self.C_matrix(device=device, dtype=dtype)

    def degradation(self, d: torch.Tensor) -> torch.Tensor:
        """Degradation function g(d).

        Types:
          'standard': g(d) = (1-d)^2 + eta  (AT2 default)
          'cubic':    Borden cubic family
                      g(d) = (3-s)(1-d)^2 - (2-s)(1-d)^3
          'rational': g(d) = (1-eta) * (1-d)^2 / ((1-d)^2 + a1*d*(1+d)) + eta
                      Wu (2017) rational degradation — less length-scale sensitive.
                      a1 = 4 / (pi * l0), calibrated to match AT2 peak stress.
                      The (1-eta)*R(d) + eta blending matches the 'standard'
                      and 'cubic' forms so g(0) = 1 exactly (closes #279).
        """
        if self.degradation_type == 'cubic':
            omd = 1.0 - d
            ratio = ((3.0 - self.cubic_s) * omd * omd
                     - (2.0 - self.cubic_s) * omd * omd * omd)
            return ratio * (1.0 - self.eta_residual) + self.eta_residual
        elif self.degradation_type == 'rational' or self.pf_model == 'PFCZM':
            if self.pf_model == 'PFCZM':
                return self.pfczm_degradation_derivatives(d)[0]
            # Wu (2017) rational degradation — less length-scale sensitive
            import math
            a1 = 4.0 / (math.pi * self.l0)  # calibrated to match AT2 peak stress
            omd = 1.0 - d
            omd2 = omd * omd
            ratio = omd2 / (omd2 + a1 * d * (1 + d))
            return ratio * (1.0 - self.eta_residual) + self.eta_residual
        return (1.0 - d) ** 2 * (1.0 - self.eta_residual) + self.eta_residual

    def compute_element_l0(self, mesh) -> torch.Tensor:
        """Per-element regularization length ensuring phase-field resolution.

        l0_e = max(l0, min_ratio * h_e) where h_e is the element incircle
        diameter. This ensures l0 >= 2*h on every element, preventing
        under-resolved damage fields on coarse mesh regions.

        Parameters
        ----------
        mesh : FEMMesh

        Returns
        -------
        l0_e : (E,) per-element regularization length
        """
        min_ratio = 2.0  # l0 should be at least 2*h
        return torch.maximum(
            torch.full_like(mesh.elem_h, self.l0),
            min_ratio * mesh.elem_h
        )

    def summary(self) -> str:
        gamma_str = ", gamma-corrected" if self.gamma_correction else ""
        deg_str = f", g(d)={self.degradation_type}" if self.degradation_type != 'standard' else ""
        ps_str = "plane stress" if self.plane_stress else "plane strain"
        return (
            f"Material ({self.pf_model}, {self.energy_split} split, {ps_str}{gamma_str}{deg_str}):\n"
            f"  E={self.E} MPa, nu={self.nu}, rho={self.rho}\n"
            f"  Gc={self.Gc} N/mm, l0={self.l0} mm\n"
            f"  Gc/l0={self.Gc_over_l0:.1f}, Gc*l0={self.Gc_times_l0:.6f}\n"
            f"  kappa={self.kappa:.1f}, mu={self.mu:.1f}, lam={self.lam:.1f}\n"
            f"  c_p={self.c_p:.2f}, c_s={self.c_s:.2f}, c_R={self.c_R:.2f}"
        )

    def __repr__(self):
        return self.summary()


def create_material(preset=None, **overrides) -> Material:
    """Factory for common material presets — or a fully inline material.

    If ``preset`` is ``None`` (or an empty string), the material is built
    purely from the keyword arguments. Otherwise, the named preset is
    loaded and ``overrides`` are layered on top.

    Presets
    -------
    'default' / 'steel_pf' : Steel with AT2 phase-field (Akantu benchmark, Amor split)
    'miehe_tension'         : PhaseFieldX 1711 SENT benchmark (isotropic split)
    'miehe_shear'           : Miehe et al. 2010 SENS benchmark (spectral split)
    'brittle_ceramic'       : Brittle ceramic example
    'pmma'                  : PMMA polymer

    Any keyword argument overrides the preset value. Numeric overrides
    may be supplied either as bare floats (interpreted in the internal
    unit system: MPa, N/mm, mm, tonne/mm^3 — see ``units.py``) or as
    SI-friendly suffixed strings such as ``"32 GPa"``, ``"3 J/m^2"``,
    ``"0.25 mm"``, ``"2450 kg/m^3"``. Suffixed strings are normalised to
    the internal unit before construction; bare floats pass through
    unchanged so legacy configs and callers see bit-identical numerics.

    Unit conventions for each preset are documented in the dict below
    next to the original literature source.
    """
    from ..utils.units import parse_quantity, MATERIAL_OVERRIDE_KINDS
    # Internal units throughout: E [MPa], Gc [N/mm], l0 [mm], rho [tonne/mm^3].
    # SI equivalents are stated alongside each preset so the conversion
    # factor is auditable; literal values are unchanged for bit-identical
    # legacy numerics.
    presets = {
        # Generic Akantu-style steel: E=210 GPa, Gc=2700 J/m^2, l0=5 um, rho=7800 kg/m^3
        'default': dict(
            E=210000.0, nu=0.3, Gc=2.7, l0=0.005, rho=7.8e-9,
        ),
        # Same as 'default' (Akantu benchmark steel).
        'steel_pf': dict(
            E=210000.0, nu=0.3, Gc=2.7, l0=0.005, rho=7.8e-9,
        ),
        # Miehe et al. 2010 — standard SENT/SENS benchmarks
        # lambda=121.15 kN/mm^2, mu=80.77 kN/mm^2 -> E~210 GPa, nu~0.3
        # Gc=2.7e-3 kN/mm = 2.7 N/mm, l0=0.0075 mm (or 2*h)
        # rho is not in the original paper (quasi-static); value here for
        # explicit dynamics compatibility
        # PhaseFieldX-matching presets (Miehe et al. 2010)
        'miehe_tension': dict(
            E=210000.0, nu=0.3, Gc=2.7, l0=0.015, rho=7.8e-9,
            energy_split='isotropic', pf_model='AT2', eta_residual=1e-7,
        ),
        'miehe_shear': dict(
            E=210000.0, nu=0.3, Gc=2.7, l0=0.06, rho=7.8e-9,
            energy_split='spectral', pf_model='AT2', eta_residual=1e-7,
        ),
        # PhaseFieldX Example 1714 — softer material, lower Gc
        'three_point_bending': dict(
            E=20800.0, nu=0.3, Gc=0.5, l0=0.06, rho=1.2e-9,
            energy_split='spectral', pf_model='AT2', eta_residual=1e-7,
        ),
        # L-shaped panel — glass (Rudshaug et al. 2024, Int J Fract)
        'l_shaped_glass': dict(
            E=70000.0, nu=0.23, Gc=0.008, l0=0.4, rho=2.5e-9,
            energy_split='spectral', pf_model='AT2', eta_residual=1e-7,
        ),
        # L-shaped panel — concrete (Winkler 2001, Ambati et al. 2015)
        # Ambati: λ=6.16, μ=10.95 kN/mm², Gc=8.9e-5 kN/mm, ℓ=1.1875 mm
        'l_shaped_concrete': dict(
            E=25850.0, nu=0.18, Gc=0.089, l0=1.1875, rho=2.4e-9,
            energy_split='spectral', pf_model='AT2', eta_residual=1e-7,
        ),
        # Alumina SENT — Kumar & Lopez-Pamies (2020), JMPS
        'alumina_kumar': dict(
            E=335000.0, nu=0.25, Gc=0.0268, l0=0.04, rho=3.9e-9,
            energy_split='star_convex', pf_model='AT2', eta_residual=1e-7,
        ),
        'brittle_ceramic': dict(
            E=370000.0, nu=0.22, Gc=0.042, l0=0.01, rho=3.9e-9,
        ),
        'pmma': dict(
            E=3000.0, nu=0.35, Gc=0.3, l0=0.02, rho=1.18e-9,
        ),
        # Borden et al. (2012) soda-lime glass — dynamic crack branching benchmarks
        # B1 (crack branching), B4 (dynamic SENT), B5 (coalescence)
        # Borden Table: E=32 GPa, nu=0.2, rho=2450 kg/m³, Gc=3 J/m²
        # l0=0.25 mm (Borden primary mesh); AT2, spectral, k=0
        # Units: mm, N, MPa, s, tonne
        'glass_borden': dict(
            E=32000.0, nu=0.20, Gc=3.0e-3, l0=0.25, rho=2.45e-9,
            energy_split='spectral', pf_model='AT2', eta_residual=1e-7,
        ),
        # PMMA for dynamic branching — Bleyer et al. (2017), Int. J. Fract.
        # Bleyer Sec 3.1: E=3.09 GPa, nu=0.35, rho=1180, Gc=300 J/m²=0.3 N/mm
        # AT1 (linear damage, finite elastic threshold), Amor vol-dev split
        # Plane stress (Bleyer Sec 3.1: "analyses performed in 2D plane stress")
        'pmma_bleyer': dict(
            E=3090.0, nu=0.35, Gc=0.3, l0=0.1, rho=1.18e-9,
            energy_split='amor', pf_model='AT1', eta_residual=1e-7,
            plane_stress=True,
        ),
        # Maraging steel for Kalthoff-Winkler — Borden (2012) Sec. 4.3
        # Borden: E=190 GPa, nu=0.3, rho=8000, Gc=22130 J/m²=22.13 N/mm
        # l0=0.195 mm (=1.95e-4 m); AT2 spectral; k=0
        'maraging_steel_kw': dict(
            E=190000.0, nu=0.30, Gc=22.13, l0=0.195, rho=8.0e-9,
            energy_split='spectral', pf_model='AT2', eta_residual=1e-7,
        ),
        # Basalt for dynamic Brazilian disc — arXiv:2411.16393 Sec. 5.1
        # E=20.11 GPa, nu=0.20, rho=2740, Gc=100 J/m²=0.1 N/mm
        # AT1, l0=1.25 mm; sigma_ts=11.31 MPa, sigma_cs=339.3 MPa
        'basalt_brazilian': dict(
            E=20110.0, nu=0.20, Gc=0.1, l0=1.25, rho=2.74e-9,
            energy_split='spectral', pf_model='AT1', eta_residual=1e-7,
            sigma_ts=11.31,  # MPa (enables nucleation enhancement)
        ),
        'soda_lime_glass': dict(
            E=72000.0, nu=0.25, Gc=9.0, l0=0.25, rho=2.44e-9,
            energy_split='spectral', pf_model='AT2', eta_residual=1e-7,
            sigma_ts=30.0,  # MPa (Table 6, arXiv:2411.16393)
        ),
        # (pmma_bleyer defined above — single entry with plane_stress=True)
        # Cement mortar — COMSOL 6.4 Application Library "Brittle Fracture
        # of a Holed Plate" (Geomechanics Module). The phase-field
        # formulation is from Ambati, Gerasimov & De Lorenzis (2015),
        # Comput. Mech. 55, 383-405. Plane stress.
        # E=6 GPa, nu=0.22, Gc=2280 J/m^2 = 2.280 N/mm, l_int=0.25 mm.
        'cement_mortar_ambati': dict(
            E=6000.0, nu=0.22, Gc=2.280, l0=0.25, rho=2.4e-9,
            energy_split='isotropic', pf_model='AT2', eta_residual=1e-6,
            plane_stress=True,
        ),
    }
    if preset in (None, ''):
        # Inline-only material: no preset baseline, dataclass defaults used
        # for any field not provided in overrides.
        params = {}
    elif preset in presets:
        params = presets[preset].copy()
    else:
        # Unknown preset name — preserve legacy behaviour of falling back
        # to 'default' (rather than erroring) so existing configs that
        # accidentally rely on this don't break.
        params = presets['default'].copy()
    # Normalise unit-suffixed string overrides to the internal unit
    # system. Bare floats / ints pass through unchanged (no-op), which
    # preserves bit-identical numerics for every existing call site.
    normalised_overrides = {}
    for key, val in overrides.items():
        if key in MATERIAL_OVERRIDE_KINDS and isinstance(val, str):
            normalised_overrides[key] = parse_quantity(val, MATERIAL_OVERRIDE_KINDS[key])
        else:
            normalised_overrides[key] = val
    params.update(normalised_overrides)
    # Filter to fields the Material dataclass actually accepts. This lets
    # callers pass forward-looking inline keys (e.g. 'kinematics') without
    # blowing up if Material hasn't grown that field yet.
    from dataclasses import fields as _dc_fields
    valid = {f.name for f in _dc_fields(Material)}
    params = {k: v for k, v in params.items() if k in valid}
    return Material(**params)
