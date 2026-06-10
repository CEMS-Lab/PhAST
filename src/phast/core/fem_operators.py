"""
Vectorized FEM operators for 2D plane-strain linear triangles.

All operations are scatter-based (no sparse matrices) for full autograd
compatibility and GPU efficiency. No Python loops over elements.

Provides:
  - Strain computation (element centroids)
  - Stress with isotropic, Amor vol-dev, Miehe spectral, or star-convex split
  - Internal force assembly
  - Strain energy density psi+ (tensile part)
  - Scalar Laplacian matvec (for damage stiffness)
  - Element-to-node projection
"""

import warnings

import torch
from .mesh import FEMMesh
from ..physics.material import Material


class FEMOperators:
    """Vectorized FEM operators for plane-strain linear triangles.

    Parameters
    ----------
    mesh : FEMMesh
        Mesh with precomputed shape function gradients.
    material : Material
        Material properties.
    ctx : DeviceContext or None
        Device context for AMP/profiling. If None, AMP is disabled.
    """

    # Audit T1.4 (W4 SOLVER_BUGS_OPTIMISATION_AUDIT_2026-05-07): the spectral
    # split (``compute_stress_spectral_*`` / ``_psi_plus_spectral*``) decomposes
    # only the *in-plane* strain tensor; under plane stress the out-of-plane
    # strain is not eigen-decomposed, making the +/- energy partition
    # approximate. Several ``Material`` presets activate this combination
    # silently. Class-level one-shot flag emits ``RuntimeWarning`` exactly
    # once per Python session.
    _plane_stress_spectral_warning_emitted = False

    @classmethod
    def _maybe_warn_plane_stress_spectral(cls, where: str) -> None:
        if cls._plane_stress_spectral_warning_emitted:
            return
        cls._plane_stress_spectral_warning_emitted = True
        warnings.warn(
            f"FEMOperators.{where}: spectral energy split is approximate "
            f"under plane stress (out-of-plane strain is not eigen-decomposed). "
            f"For exact tension/compression separation use plane strain or "
            f"another split. Audit T1.4 (W4).",
            RuntimeWarning, stacklevel=3,
        )

    def __init__(self, mesh: FEMMesh, material: Material, ctx=None):
        print(f"[FEMOperators] Initializing ({material.energy_split} split, "
              f"{material.pf_model})...", flush=True)
        self.mesh = mesh
        self.material = material
        self.device = mesh.device
        self.dtype = mesh.dtype
        self._ctx = ctx

        # Constitutive matrix (contiguous for cache efficiency, #26)
        self.C = material.C_plane_strain(mesh.device, mesh.dtype).contiguous()

        # Vector lumped mass (2 DOFs per node, interleaved: [m0,m0,m1,m1,...])
        self.M_vec = material.rho * mesh.M_scalar.unsqueeze(1).expand(-1, 2).flatten()
        self.M_vec_inv = 1.0 / (self.M_vec + 1e-30)

        # CFL timestep
        self.dt_cfl = mesh.h_min / material.c_p
        print(f"[FEMOperators] dt_CFL = {self.dt_cfl:.6e} "
              f"(c_p={material.c_p:.2f})", flush=True)

        # Optional per-element Young's modulus field for spatially heterogeneous
        # elasticity (paper-2 stiff-inclusion / spatial-E inversion). When None,
        # all stiffness paths fall back to the scalar material.E and the solver
        # behaviour is bit-equivalent to the pre-field implementation.
        # Currently supported with energy_split='amor' only; the staggered
        # solver guards the install path. nu, rho, plane_stress remain scalar.
        self.diff_E_field = None
        # Per-element Lame parameter cache — avoids rebuilding (lam_e, mu_e,
        # kappa_e) tensors (and the corresponding autograd graph nodes) on
        # every stress/psi/secant call. Invalidated by identity check on
        # diff_E_field so a direct attribute write or install_diff_E_field
        # both refresh the cache transparently. Saves ~3*n_steps redundant
        # tensor allocations per forward pass.
        self._E_field_cache_id = None
        self._E_field_lam = None
        self._E_field_mu = None
        self._E_field_kappa = None

        # Use mesh's precomputed flat indices (no duplicate allocation)
        self._elem_flat = mesh._elem_flat
        # Precompute areas broadcasted for force assembly
        self._areas_col = mesh.areas.unsqueeze(1)  # (E, 1)

        # Precompute squared gradients for stiffness diagonal caching
        if getattr(mesh, 'element_type', 'T3') == 'Q4':
            self._gp_x_sq = None
            self._gp_y_sq = None
        else:
            self._gp_x_sq = mesh.grad_phi[:, :, 0]**2
            self._gp_y_sq = mesh.grad_phi[:, :, 1]**2

        # Dtype-aware spectral split regularization floor:
        # float32 eps ~6e-8, so 1e-12 vanishes; use 1e-6 instead
        self._spectral_eps = 1e-12 if self.dtype == torch.float64 else 1e-7

    # ------------------------------------------------------------------ #
    # Heterogeneous-elasticity helper
    # ------------------------------------------------------------------ #

    def _resolve_lame(self):
        """Return (lam, mu, kappa) — scalars from `material` when
        ``diff_E_field`` is None, otherwise per-element ``(E,)`` tensors
        derived from the field with scalar nu/rho.

        Uses an identity-keyed cache: the per-element tensors are computed
        once when ``diff_E_field`` first appears (or changes identity), and
        reused on every subsequent stress/psi/secant call until the field
        is replaced. Saves ~3*n_steps redundant tensor allocations and
        autograd graph nodes per forward pass when E_field is set.

        nu and the plane-stress/plane-strain assumption are inherited from
        the underlying ``Material`` (scalar). The returned tensors are kept
        in the autograd graph if ``diff_E_field`` requires_grad, which is
        what allows gradients to flow from a downstream loss back to the
        E-field parameters (e.g. inclusion centre coordinates).
        """
        if self.diff_E_field is None:
            return self.material.lam, self.material.mu, self.material.kappa
        cache_id = id(self.diff_E_field)
        if cache_id != self._E_field_cache_id:
            E_e = self.diff_E_field
            nu = self.material.nu
            mu_e = E_e / (2.0 * (1.0 + nu))
            if self.material.plane_stress:
                lam_e = E_e * nu / (1.0 - nu * nu)
            else:
                lam_e = E_e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            kappa_e = E_e / (3.0 * (1.0 - 2.0 * nu))
            self._E_field_cache_id = cache_id
            self._E_field_lam = lam_e
            self._E_field_mu = mu_e
            self._E_field_kappa = kappa_e
        return self._E_field_lam, self._E_field_mu, self._E_field_kappa

    def recompute_dt_cfl(self):
        """Recompute ``self.dt_cfl`` from the current ``diff_E_field``.

        When a per-element E field is installed, the CFL bound on the
        explicit timestep is set by the *maximum* wave speed in the
        domain, not the bulk material wave speed. With a stiff inclusion
        of contrast alpha the peak modulus is (1+alpha)*E_bulk and the
        peak P-wave speed is sqrt((1+alpha)) faster; running the bulk
        timestep there blows up. Call this after ``diff_E_field`` is
        installed and again whenever the field changes shape (e.g.
        between L-BFGS outer iterations if the inclusion contrast itself
        is being optimised).
        """
        import math
        if self.diff_E_field is None:
            self.dt_cfl = self.mesh.h_min / self.material.c_p
            return
        E_max = float(self.diff_E_field.detach().max().item())
        nu = self.material.nu
        rho = self.material.rho
        mu_max = E_max / (2.0 * (1.0 + nu))
        if self.material.plane_stress:
            lam_max = E_max * nu / (1.0 - nu * nu)
        else:
            lam_max = E_max * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c_p_max = math.sqrt((lam_max + 2.0 * mu_max) / rho)
        self.dt_cfl = self.mesh.h_min / c_p_max
        print(f"[FEMOperators] dt_CFL recomputed for E_field: {self.dt_cfl:.6e} "
              f"(c_p_max={c_p_max:.2f}, E_max={E_max:.3e})", flush=True)

    # ------------------------------------------------------------------ #
    # Strain
    # ------------------------------------------------------------------ #

    def compute_strain(self, u: torch.Tensor):
        """Strain at element centroids.

        Parameters
        ----------
        u : (N, 2) nodal displacements

        Returns
        -------
        eps_xx, eps_yy, gam_xy : each (E,)
        """
        if getattr(self.mesh, 'element_type', 'T3') == 'Q4':
            return self._compute_strain_q4(u)
        gp = self.mesh.grad_phi
        u_e = u[self.mesh.elements]  # (E, 3, 2)
        eps_xx = (gp[:, :, 0] * u_e[:, :, 0]).sum(1)
        eps_yy = (gp[:, :, 1] * u_e[:, :, 1]).sum(1)
        gam_xy = ((gp[:, :, 1] * u_e[:, :, 0]).sum(1) +
                  (gp[:, :, 0] * u_e[:, :, 1]).sum(1))
        return eps_xx, eps_yy, gam_xy

    def _compute_strain_q4(self, u: torch.Tensor):
        """Q4 strain at 2x2 Gauss points, returned as three ``(E, 4)`` tensors."""
        gp = self.mesh.quad_grad_phi
        u_e = u[self.mesh.elements]  # (E, 4, 2)
        eps_xx = torch.einsum('eqa,ea->eq', gp[..., 0], u_e[..., 0])
        eps_yy = torch.einsum('eqa,ea->eq', gp[..., 1], u_e[..., 1])
        gam_xy = (
            torch.einsum('eqa,ea->eq', gp[..., 1], u_e[..., 0])
            + torch.einsum('eqa,ea->eq', gp[..., 0], u_e[..., 1])
        )
        return eps_xx, eps_yy, gam_xy

    # ------------------------------------------------------------------ #
    # Stress (multiple variants)
    # ------------------------------------------------------------------ #

    def compute_stress_linear(self, eps_xx, eps_yy, gam_xy):
        """Undegraded linear elastic stress. Returns sxx, syy, sxy: (E,).

        Uses the per-element Lame parameters from ``_resolve_lame`` when
        ``diff_E_field`` is set, falling back to the cached ``self.C``
        matrix otherwise so the bit-equivalent path is preserved.
        """
        if self.diff_E_field is None:
            C = self.C
            sxx = C[0, 0] * eps_xx + C[0, 1] * eps_yy
            syy = C[1, 0] * eps_xx + C[1, 1] * eps_yy
            sxy = C[2, 2] * gam_xy
            return sxx, syy, sxy
        lam, mu, _ = self._resolve_lame()
        c11 = lam + 2.0 * mu
        sxx = c11 * eps_xx + lam * eps_yy
        syy = lam * eps_xx + c11 * eps_yy
        sxy = mu * gam_xy
        return sxx, syy, sxy

    def compute_stress_isotropic(self, eps_xx, eps_yy, gam_xy, g_d):
        """Isotropic stress — full degradation, no tension/compression split.

        sigma = g(d) * C * eps  (everything is degraded)
        """
        if self.diff_E_field is None:
            C = self.C
            sxx = g_d * (C[0, 0] * eps_xx + C[0, 1] * eps_yy)
            syy = g_d * (C[1, 0] * eps_xx + C[1, 1] * eps_yy)
            sxy = g_d * (C[2, 2] * gam_xy)
            return sxx, syy, sxy
        lam, mu, _ = self._resolve_lame()
        c11 = lam + 2.0 * mu
        sxx = g_d * (c11 * eps_xx + lam * eps_yy)
        syy = g_d * (lam * eps_xx + c11 * eps_yy)
        sxy = g_d * (mu * gam_xy)
        return sxx, syy, sxy

    def compute_stress_amor(self, eps_xx, eps_yy, gam_xy, g_d):
        """Stress with Amor volumetric-deviatoric split.

        sigma = g(d)*sigma_plus + sigma_minus
        sigma_plus = kappa*<tr>_+*I + 2*mu*eps_dev  (tensile, degraded)
        sigma_minus = kappa*<tr>_-*I                 (compressive, intact)

        Uses full 3D trace and deviatoric decomposition:
          Plane strain: eps_zz = 0
          Plane stress: eps_zz = -nu/(1-nu)*(exx+eyy)

        Returns only in-plane (sxx, syy, sxy) needed for 2D force assembly.

        Parameters
        ----------
        g_d : (E,) degradation function values at element centroids.
        """
        _, mu, kappa = self._resolve_lame()
        tr_2d = eps_xx + eps_yy
        if self.material.plane_stress:
            nu = self.material.nu
            ezz = -nu / (1.0 - nu) * tr_2d
        else:
            ezz = 0.0  # plane strain: scalar zero (broadcasts)
        tr = tr_2d + ezz  # 3D trace
        tr_plus = torch.clamp(tr, min=0)
        tr_minus = tr - tr_plus
        dev_xx = eps_xx - tr / 3.0
        dev_yy = eps_yy - tr / 3.0
        sxx = g_d * (kappa * tr_plus + 2 * mu * dev_xx) + kappa * tr_minus
        syy = g_d * (kappa * tr_plus + 2 * mu * dev_yy) + kappa * tr_minus
        sxy = g_d * mu * gam_xy
        return sxx, syy, sxy

    def compute_stress_spectral_algebraic(self, eps_xx, eps_yy, gam_xy, g_d):
        """Spectral split using algebraic projection tensors (default method).

        Uses P1 = (eps - e2*I) / (e1 - e2) instead of atan2/cos/sin.
        Avoids the atan2(0,0)=π/4 singularity that corrupts autograd at
        hydrostatic/zero strain states. ~20% faster on GPU.

        Plane-stress support is absorbed via `material.lam`, which returns
        the effective 2D Lamé parameter `lam_ps = E*nu/(1-nu**2)` when
        `plane_stress=True` (vs the plane-strain `E*nu/((1+nu)*(1-2*nu))`).
        The split then operates on the 2D eigenvalues of the in-plane strain
        tensor and is internally consistent for the reduced 2D problem --
        the out-of-plane strain is not eigen-decomposed (acceptable for 2D
        plane-stress reduction but approximate for true 3D tension/compression
        separation).
        """
        if getattr(self.material, 'plane_stress', False):
            self._maybe_warn_plane_stress_spectral('compute_stress_spectral_algebraic')
        lam, mu, _ = self._resolve_lame()

        exy = gam_xy / 2.0
        trace = eps_xx + eps_yy
        half_diff = (eps_xx - eps_yy) / 2.0
        delta = torch.sqrt(half_diff ** 2 + exy ** 2 + self._spectral_eps)
        e1 = trace / 2.0 + delta
        e2 = trace / 2.0 - delta

        e1_plus = torch.clamp(e1, min=0)
        e2_plus = torch.clamp(e2, min=0)
        tr_plus = torch.clamp(trace, min=0)
        tr_minus = trace - tr_plus

        # Algebraic projection: P1 = (eps - e2*I) / (e1 - e2)
        # delta >= sqrt(spectral_eps), so inv_2delta is bounded — safe
        inv_2delta = 1.0 / (2.0 * delta + 1e-30)
        p1_xx = (half_diff + delta) * inv_2delta
        p1_yy = (-half_diff + delta) * inv_2delta
        p1_xy = exy * inv_2delta

        exx_plus = e1_plus * p1_xx + e2_plus * (1.0 - p1_xx)
        eyy_plus = e1_plus * p1_yy + e2_plus * (1.0 - p1_yy)
        exy_plus = (e1_plus - e2_plus) * p1_xy

        exx_minus = eps_xx - exx_plus
        eyy_minus = eps_yy - eyy_plus
        exy_minus = exy - exy_plus

        sxx = g_d * (lam * tr_plus + 2 * mu * exx_plus) + (
            lam * tr_minus + 2 * mu * exx_minus)
        syy = g_d * (lam * tr_plus + 2 * mu * eyy_plus) + (
            lam * tr_minus + 2 * mu * eyy_minus)
        sxy = g_d * (2 * mu * exy_plus) + (2 * mu * exy_minus)
        return sxx, syy, sxy

    def compute_stress_spectral_stress(self, eps_xx, eps_yy, gam_xy, g_d):
        """Stress-spectral split (Miehe et al. 2010 §3.2).

        Decomposes the un-degraded *stress* tensor into eigenpairs and
        applies ``g(d)`` only to the positive-eigenvalue projection. This
        differs from ``compute_stress_spectral_algebraic`` (Miehe §3.1),
        which decomposes the *strain* tensor. Yu et al. (2021) report a
        10-15% timing-shift between the two formulations on dynamic
        branching benchmarks.

        Algorithm:
          1. σ = λ·tr(ε)·I + 2μ·ε              (linear elastic, undegraded)
          2. Reuse strain eigenvectors n_i (σ and ε share eigenvectors
             for isotropic linear elasticity), with eigenvalues
             σ_i = λ·tr(ε) + 2μ·e_i.
          3. σ⁺ = Σ_i ⟨σ_i⟩₊ n_i⊗n_i,   σ⁻ = σ - σ⁺
          4. σ_damaged = g(d)·σ⁺ + σ⁻

        Notes
        -----
        - Plane-stress is absorbed via ``material.lam`` (effective 2D
          Lamé parameter), matching the strain-spectral path.
        - The resulting σ_damaged is not energy-conjugate to a smooth
          Ψ(ε,d); the solver is consistent only via the autograd-JVP
          route, not the secant linearisation. Calling
          ``freeze_secant_state`` with this split raises
          ``NotImplementedError``.
        - Opt-in: see issue #213 (B7 dynamic branching, COMSOL parity).
        """
        if getattr(self.material, 'plane_stress', False):
            self._maybe_warn_plane_stress_spectral('compute_stress_spectral_stress')
        mu = self.material.mu
        lam = self.material.lam

        exy = gam_xy / 2.0
        trace = eps_xx + eps_yy
        half_diff = (eps_xx - eps_yy) / 2.0
        delta = torch.sqrt(half_diff ** 2 + exy ** 2 + self._spectral_eps)
        e1 = trace / 2.0 + delta
        e2 = trace / 2.0 - delta

        # Principal stresses share eigenvectors with strain
        # (σ = λ·tr·I + 2μ·ε for isotropic linear elasticity).
        s1 = lam * trace + 2.0 * mu * e1
        s2 = lam * trace + 2.0 * mu * e2

        s1_plus = torch.clamp(s1, min=0)
        s2_plus = torch.clamp(s2, min=0)
        s1_minus = s1 - s1_plus
        s2_minus = s2 - s2_plus

        # Algebraic projector P1 = (eps - e2*I)/(e1 - e2) (frozen at strain).
        inv_2delta = 1.0 / (2.0 * delta + 1e-30)
        p1_xx = (half_diff + delta) * inv_2delta
        p1_yy = (-half_diff + delta) * inv_2delta
        p1_xy = exy * inv_2delta

        # Cartesian σ⁺ from sum over positive principal stresses; σ⁻ similarly.
        sxx_plus = s1_plus * p1_xx + s2_plus * (1.0 - p1_xx)
        syy_plus = s1_plus * p1_yy + s2_plus * (1.0 - p1_yy)
        sxy_plus = (s1_plus - s2_plus) * p1_xy

        sxx_minus = s1_minus * p1_xx + s2_minus * (1.0 - p1_xx)
        syy_minus = s1_minus * p1_yy + s2_minus * (1.0 - p1_yy)
        sxy_minus = (s1_minus - s2_minus) * p1_xy

        sxx = g_d * sxx_plus + sxx_minus
        syy = g_d * syy_plus + syy_minus
        sxy = g_d * sxy_plus + sxy_minus
        return sxx, syy, sxy

    def compute_stress_star_convex(self, eps_xx, eps_yy, gam_xy, g_d):
        """Stress with star-convex decomposition (Kumar et al. 2018/2020).

        Element-wise check on volumetric strain:
          - Tension  (tr(eps) >= 0): sigma = g(d) * C * eps  (full degradation)
          - Compress (tr(eps) < 0):  sigma = g(d) * 2*mu*eps_dev + kappa*tr*I
                                     (only deviatoric degraded, volumetric intact)

        Uses the full 3D trace so the volumetric term (kappa * tr) remains
        thermodynamically consistent under both plane strain (eps_zz = 0)
        and plane stress (eps_zz = -nu/(1-nu) * (eps_xx + eps_yy)).

        Ensures star-convexity of the total energy w.r.t. (eps, d), improving
        convergence properties compared to spectral split.

        Reference: Kumar, Bourdin, Francfort & Lopez-Pamies (2020),
        JMPS 142, 104027, doi:10.1016/j.jmps.2020.104027.
        """
        lam, mu, kappa = self._resolve_lame()
        tr_2d = eps_xx + eps_yy
        if self.material.plane_stress:
            nu = self.material.nu
            ezz = -nu / (1.0 - nu) * tr_2d
        else:
            ezz = 0.0  # plane strain: broadcasts as scalar zero
        tr = tr_2d + ezz  # 3D trace
        tension = (tr >= 0)  # (E,) bool

        # Tension branch: full degradation (same as isotropic), but built
        # from per-element Lamé parameters when a differentiable E field is
        # installed.
        sxx_t = g_d * ((lam + 2.0 * mu) * eps_xx + lam * eps_yy)
        syy_t = g_d * (lam * eps_xx + (lam + 2.0 * mu) * eps_yy)
        sxy_t = g_d * (mu * gam_xy)

        # Compression branch: degrade deviatoric only, volumetric intact
        dev_xx = eps_xx - tr / 3.0
        dev_yy = eps_yy - tr / 3.0
        sxx_c = g_d * 2 * mu * dev_xx + kappa * tr
        syy_c = g_d * 2 * mu * dev_yy + kappa * tr
        sxy_c = g_d * mu * gam_xy

        sxx = torch.where(tension, sxx_t, sxx_c)
        syy = torch.where(tension, syy_t, syy_c)
        sxy = torch.where(tension, sxy_t, sxy_c)
        return sxx, syy, sxy

    # ------------------------------------------------------------------ #
    # Internal force
    # ------------------------------------------------------------------ #

    def internal_force(self, u: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Degraded internal force f_int using configured energy split.

        Parameters
        ----------
        u : (N, 2) displacements
        d : (N,) damage field

        Returns
        -------
        f_int : (N, 2) internal force vector
        """
        strain = self.compute_strain(u)
        sxx, syy, sxy = self.compute_stress(u, d, strain=strain)
        return self._assemble_force(sxx, syy, sxy)

    def compute_stress(self, u, d, strain=None):
        """Compute degraded stress for the current energy split.

        Dispatches to the correct split method (spectral/amor/isotropic/star_convex).
        Use this instead of calling individual compute_stress_* methods directly.

        Returns
        -------
        sxx, syy, sxy : (E,) element stresses
        """
        mesh = self.mesh
        if getattr(mesh, 'element_type', 'T3') == 'Q4':
            if self.material.energy_split != 'isotropic':
                raise NotImplementedError(
                    "Native Q4 stress dispatch currently supports "
                    "energy_split='isotropic' only. Use Q4-to-T3 conversion "
                    "or add split-specific Gauss-point Q4 support.")
            d_e = torch.einsum(
                'qa,ea->eq', mesh.quad_N, d[mesh.elements])
        else:
            d_e = d[mesh.elements].mean(1)
        g_d = self.material.degradation(d_e)

        if strain is None:
            strain = self.compute_strain(u)
        eps_xx, eps_yy, gam_xy = strain

        split = self.material.energy_split
        if split == 'spectral':
            return self.compute_stress_spectral_algebraic(eps_xx, eps_yy, gam_xy, g_d)
        elif split == 'spectral_stress':
            return self.compute_stress_spectral_stress(eps_xx, eps_yy, gam_xy, g_d)
        elif split == 'star_convex':
            return self.compute_stress_star_convex(eps_xx, eps_yy, gam_xy, g_d)
        elif split == 'isotropic':
            return self.compute_stress_isotropic(eps_xx, eps_yy, gam_xy, g_d)
        elif split == 'amor':
            return self.compute_stress_amor(eps_xx, eps_yy, gam_xy, g_d)
        raise ValueError(f"Unknown energy_split: {split!r}")

    def internal_force_linear(self, u: torch.Tensor) -> torch.Tensor:
        """Undegraded linear elastic internal force (for initial static solve)."""
        eps_xx, eps_yy, gam_xy = self.compute_strain(u)
        sxx, syy, sxy = self.compute_stress_linear(eps_xx, eps_yy, gam_xy)
        return self._assemble_force(sxx, syy, sxy)

    def _assemble_force(self, sxx, syy, sxy) -> torch.Tensor:
        """Assemble nodal force from element stresses via scatter_add.

        Uses precomputed flat indices and broadcasted areas for speed.
        """
        if getattr(self.mesh, 'element_type', 'T3') == 'Q4':
            return self._assemble_force_q4(sxx, syy, sxy)
        gp = self.mesh.grad_phi
        A = self._areas_col  # (E, 1) precomputed
        fx = A * (sxx.unsqueeze(1) * gp[:, :, 0] +
                  sxy.unsqueeze(1) * gp[:, :, 1])
        fy = A * (sxy.unsqueeze(1) * gp[:, :, 0] +
                  syy.unsqueeze(1) * gp[:, :, 1])

        f = torch.zeros(self.mesh.n_nodes, 2, dtype=self.dtype,
                         device=self.device)
        f[:, 0].scatter_add_(0, self._elem_flat, fx.flatten())
        f[:, 1].scatter_add_(0, self._elem_flat, fy.flatten())
        return f

    def _assemble_force_q4(self, sxx, syy, sxy) -> torch.Tensor:
        """Assemble Q4 nodal forces from stresses at 2x2 Gauss points."""
        gp = self.mesh.quad_grad_phi
        wdet = self.mesh.quad_wdetJ
        fx_q = (
            gp[..., 0] * sxx.unsqueeze(-1)
            + gp[..., 1] * sxy.unsqueeze(-1)
        ) * wdet.unsqueeze(-1)
        fy_q = (
            gp[..., 1] * syy.unsqueeze(-1)
            + gp[..., 0] * sxy.unsqueeze(-1)
        ) * wdet.unsqueeze(-1)
        fx = fx_q.sum(dim=1)
        fy = fy_q.sum(dim=1)
        f = torch.zeros(self.mesh.n_nodes, 2, dtype=self.dtype,
                        device=self.device)
        f[:, 0].scatter_add_(0, self._elem_flat, fx.flatten())
        f[:, 1].scatter_add_(0, self._elem_flat, fy.flatten())
        return f

    # ------------------------------------------------------------------ #
    # Secant linearization (frozen projections for CG)
    # ------------------------------------------------------------------ #

    def freeze_secant_state(self, u: torch.Tensor,
                            d: torch.Tensor) -> dict:
        """Evaluate and freeze tension/compression state for linearized CG.

        Computes eigenvalue signs and eigenvector projections from the
        current displacement u, then caches them so that secant_matvec(p)
        is perfectly linear in p.

        .. warning::
           This is the **secant** stiffness, **not** the consistent
           tangent (Jacobian of `internal_force` w.r.t. u). For the
           spectral split in particular, freezing the eigenvector
           projectors P_i(eps) at u omits the eigvec-rotation term in
           the true tangent. FD-vs-secant probing measured ~33-39%
           relative error on biaxial / uniaxial tension at d=0.3
           (issue #172). SecantCG converges anyway via Newton-secant
           iteration, but downstream consumers that need a *Jacobian*
           (autograd VJP, line-search Wolfe condition, eigenvalue
           analysis, sensitivity analysis) must NOT use this — use
           ``torch.autograd.functional.jacobian`` on
           ``internal_force`` instead. See PR #170 (issue #114) for
           the autograd-JVP path now used by ``QuasiStaticSolver``.

        Parameters
        ----------
        u : (N, 2) current displacement
        d : (N,) current damage field

        Returns
        -------
        state : dict with frozen projection data for secant_matvec
        """
        mesh = self.mesh
        d_e = d[mesh.elements].mean(1)
        g_d = self.material.degradation(d_e)

        eps_xx, eps_yy, gam_xy = self.compute_strain(u)
        split = self.material.energy_split

        state = {'g_d': g_d, 'split': split}

        if split == 'spectral_stress':
            # Stress-spectral split (Miehe §3.2) is opt-in for explicit
            # dynamics (issue #213). The secant linearisation for it is
            # not implemented; QS / SecantCG callers should choose
            # 'spectral' (strain) or use the autograd-JVP path instead.
            raise NotImplementedError(
                "energy_split='spectral_stress' is opt-in for explicit "
                "dynamics only (issue #213); SecantCG / QS path is not "
                "implemented. Use 'spectral' for quasi-static problems.")

        if split == 'spectral':
            # Spectral decomposition is precision-sensitive (eigenvalue signs).
            # On float32 devices (MPS), compute in float64 to avoid sign-flip
            # instability that causes SecantCG divergence.
            # Route through CPU first — MPS cannot convert to float64 directly.
            if self.dtype == torch.float32:
                exy = (gam_xy / 2.0).detach().cpu().to(torch.float64)
                trace = (eps_xx + eps_yy).detach().cpu().to(torch.float64)
                half_diff = ((eps_xx - eps_yy) / 2.0).detach().cpu().to(torch.float64)
                spectral_eps = 1e-12
            else:
                exy = gam_xy / 2.0
                trace = eps_xx + eps_yy
                half_diff = (eps_xx - eps_yy) / 2.0
                spectral_eps = self._spectral_eps

            delta = torch.sqrt(half_diff ** 2 + exy ** 2 + spectral_eps)
            e1 = trace / 2.0 + delta
            e2 = trace / 2.0 - delta

            # Eigenvalue signs and eigenvector projections
            sign1 = (e1 >= 0).to(self.dtype)
            sign2 = (e2 >= 0).to(self.dtype)
            inv_2delta = 1.0 / (2.0 * delta + 1e-30)
            p1_xx = ((half_diff + delta) * inv_2delta).to(self.dtype)
            p1_yy = ((-half_diff + delta) * inv_2delta).to(self.dtype)
            p1_xy = (exy * inv_2delta).to(self.dtype)

            # Move results back to original device (CPU→MPS if needed)
            dev = eps_xx.device
            state['sign1_pos'] = sign1.to(device=dev)
            state['sign2_pos'] = sign2.to(device=dev)
            state['p1_xx'] = p1_xx.to(device=dev)
            state['p1_yy'] = p1_yy.to(device=dev)
            state['p1_xy'] = p1_xy.to(device=dev)

        elif split == 'amor':
            # Mirror compute_stress_amor (lines 213-245): build the 3D trace
            # so the matvec linearisation matches the residual's volumetric
            # term. Plane-strain: eps_zz = 0 ⇒ tr_3d == tr_2d (bit-identical
            # to pre-fix). Plane-stress: eps_zz = -nu/(1-nu)*(exx+eyy)
            # introduces the (1-2nu)/(1-nu) factor on tr (issue #222).
            tr_2d = eps_xx + eps_yy
            if self.material.plane_stress:
                nu = self.material.nu
                ezz = -nu / (1.0 - nu) * tr_2d
                tr = tr_2d + ezz
            else:
                tr = tr_2d  # plane strain: eps_zz = 0
            state['trace_pos'] = (tr >= 0).to(self.dtype)

        elif split == 'star_convex':
            tr_2d = eps_xx + eps_yy
            if self.material.plane_stress:
                nu = self.material.nu
                ezz = -nu / (1.0 - nu) * tr_2d
                tr = tr_2d + ezz
            else:
                tr = tr_2d
            state['tension'] = (tr >= 0)
        # isotropic: already linear, no frozen state needed

        return state

    def secant_matvec(self, p: torch.Tensor, state: dict) -> torch.Tensor:
        """Linearized stiffness action K_secant @ p using frozen projections.

        CG search direction p is projected using frozen eigenvalue signs
        and eigenvectors from the current u.  A(p) is perfectly linear in p,
        guaranteeing CG convergence.

        See ``freeze_secant_state`` for the important warning that this
        is **not** the consistent tangent of ``internal_force``: it
        omits the eigvec-rotation term for the spectral split. Use
        autograd-JVP for any code that needs a true Jacobian (issue
        #172, #114, PR #170).

        Parameters
        ----------
        p : (N, 2) CG search direction
        state : dict from freeze_secant_state()

        Returns
        -------
        Ap : (N, 2) linearized stiffness action
        """
        split = state['split']
        g_d = state['g_d']

        eps_xx, eps_yy, gam_xy = self.compute_strain(p)

        if split == 'isotropic':
            sxx, syy, sxy = self.compute_stress_isotropic(
                eps_xx, eps_yy, gam_xy, g_d)

        elif split == 'amor':
            _, mu, kappa = self._resolve_lame()
            # Use the 3D trace to mirror compute_stress_amor (issue #222).
            # Plane-strain: eps_zz = 0 ⇒ tr_3d == tr_2d (bit-identical
            # to pre-fix). Plane-stress: reconstruct
            # eps_zz = -nu/(1-nu)*(exx+eyy) so the volumetric coefficient
            # picks up the (1-2nu)/(1-nu) factor that the 2D-trace path
            # was missing.
            tr_2d = eps_xx + eps_yy
            if self.material.plane_stress:
                nu = self.material.nu
                ezz = -nu / (1.0 - nu) * tr_2d
                tr = tr_2d + ezz
            else:
                tr = tr_2d  # plane strain: eps_zz = 0
            tr_pos = state['trace_pos']  # frozen (E,) float: 1 or 0
            tr_plus = tr * tr_pos
            tr_minus = tr * (1.0 - tr_pos)
            dev_xx = eps_xx - tr / 3.0
            dev_yy = eps_yy - tr / 3.0
            sxx = g_d * (kappa * tr_plus + 2 * mu * dev_xx) + kappa * tr_minus
            syy = g_d * (kappa * tr_plus + 2 * mu * dev_yy) + kappa * tr_minus
            sxy = g_d * mu * gam_xy

        elif split == 'spectral':
            mu = self.material.mu
            lam = self.material.lam
            exy = gam_xy / 2.0

            # Frozen eigenvector projection tensors
            p1_xx = state['p1_xx']
            p1_yy = state['p1_yy']
            p1_xy = state['p1_xy']

            # Project strain of p onto frozen eigenvectors
            e1_p = p1_xx * eps_xx + 2.0 * p1_xy * exy + p1_yy * eps_yy
            e2_p = (1.0 - p1_xx) * eps_xx - 2.0 * p1_xy * exy + (
                1.0 - p1_yy) * eps_yy

            # Apply frozen eigenvalue signs
            sign1 = state['sign1_pos']
            sign2 = state['sign2_pos']
            e1_p_plus = e1_p * sign1
            e2_p_plus = e2_p * sign2
            e1_p_minus = e1_p * (1.0 - sign1)
            e2_p_minus = e2_p * (1.0 - sign2)
            tr_plus = e1_p_plus + e2_p_plus
            tr_minus = e1_p_minus + e2_p_minus

            # Reconstruct Cartesian eps_plus from frozen projections
            exx_plus = e1_p_plus * p1_xx + e2_p_plus * (1.0 - p1_xx)
            eyy_plus = e1_p_plus * p1_yy + e2_p_plus * (1.0 - p1_yy)
            exy_plus = (e1_p_plus - e2_p_plus) * p1_xy

            exx_minus = eps_xx - exx_plus
            eyy_minus = eps_yy - eyy_plus
            exy_minus = exy - exy_plus

            sxx = g_d * (lam * tr_plus + 2 * mu * exx_plus) + (
                lam * tr_minus + 2 * mu * exx_minus)
            syy = g_d * (lam * tr_plus + 2 * mu * eyy_plus) + (
                lam * tr_minus + 2 * mu * eyy_minus)
            sxy = g_d * (2 * mu * exy_plus) + (2 * mu * exy_minus)

        elif split == 'star_convex':
            lam, mu, kappa = self._resolve_lame()
            tr_2d = eps_xx + eps_yy
            if self.material.plane_stress:
                nu = self.material.nu
                ezz = -nu / (1.0 - nu) * tr_2d
            else:
                ezz = torch.zeros_like(eps_xx)
            tr = tr_2d + ezz
            tension = state['tension']  # frozen bool

            sxx_t = g_d * ((lam + 2.0 * mu) * eps_xx + lam * eps_yy)
            syy_t = g_d * (lam * eps_xx + (lam + 2.0 * mu) * eps_yy)
            sxy_t = g_d * (mu * gam_xy)

            dev_xx = eps_xx - tr / 3.0
            dev_yy = eps_yy - tr / 3.0
            sxx_c = g_d * 2 * mu * dev_xx + kappa * tr
            syy_c = g_d * 2 * mu * dev_yy + kappa * tr
            sxy_c = g_d * mu * gam_xy

            sxx = torch.where(tension, sxx_t, sxx_c)
            syy = torch.where(tension, syy_t, syy_c)
            sxy = torch.where(tension, sxy_t, sxy_c)

        else:
            raise ValueError(f"Unknown split: {split}")

        return self._assemble_force(sxx, syy, sxy)

    # ------------------------------------------------------------------ #
    # Strain energy density
    # ------------------------------------------------------------------ #

    def compute_psi_plus(self, u: torch.Tensor, strain=None) -> torch.Tensor:
        """Tensile strain energy density psi+ at element centroids.

        Returns
        -------
        psi_plus : (E,) tensor
        """
        if strain is None:
            strain = self.compute_strain(u)
        split = self.material.energy_split
        if getattr(self.mesh, 'element_type', 'T3') == 'Q4' and split != 'isotropic':
            raise NotImplementedError(
                "Native Q4 compute_psi_plus currently supports "
                "energy_split='isotropic' only.")
        if split == 'spectral':
            return self._psi_plus_spectral(strain)
        elif split == 'spectral_stress':
            return self._psi_plus_spectral_stress(strain)
        elif split == 'star_convex':
            return self._psi_plus_star_convex(strain)
        elif split == 'isotropic':
            return self._psi_plus_isotropic(strain)
        elif split == 'amor':
            return self._psi_plus_amor(strain)
        raise ValueError(f"Unknown energy_split: {split!r}")

    def _psi_plus_isotropic(self, strain) -> torch.Tensor:
        """psi = 0.5*eps:C:eps  (full strain energy, no split)."""
        lam, mu, _ = self._resolve_lame()
        exx, eyy, gxy = strain
        exy = gxy / 2.0
        tr = exx + eyy
        return 0.5 * lam * tr ** 2 + mu * (exx ** 2 + eyy ** 2 + 2 * exy ** 2)

    def _psi_plus_amor(self, strain) -> torch.Tensor:
        """psi+ = 0.5*kappa*<tr(eps)>+^2 + mu*dev(eps):dev(eps)  (Amor).

        Uses full 3D trace and deviatoric decomposition:
          Plane strain: eps_zz = 0, tr = exx + eyy
          Plane stress: eps_zz = -nu/(1-nu)*(exx+eyy), tr = (1-2nu)/(1-nu)*(exx+eyy)
        kappa is always the 3D bulk modulus E/(3*(1-2*nu)).
        """
        _, mu, kappa = self._resolve_lame()
        exx, eyy, gxy = strain
        tr_2d = exx + eyy
        if self.material.plane_stress:
            nu = self.material.nu
            ezz = -nu / (1.0 - nu) * tr_2d
        else:
            ezz = torch.zeros_like(exx)  # plane strain
        tr = tr_2d + ezz  # 3D trace
        tr_plus = torch.clamp(tr, min=0)
        dev_xx = exx - tr / 3.0
        dev_yy = eyy - tr / 3.0
        dev_zz = ezz - tr / 3.0
        exy = gxy / 2.0
        dev_dot = dev_xx ** 2 + dev_yy ** 2 + dev_zz ** 2 + 2.0 * exy ** 2
        return 0.5 * kappa * tr_plus ** 2 + mu * dev_dot

    def _psi_plus_spectral(self, strain) -> torch.Tensor:
        """psi+ = lam/2*<tr>+^2 + mu*(e1+^2 + e2+^2)  (Miehe spectral)."""
        if getattr(self.material, 'plane_stress', False):
            self._maybe_warn_plane_stress_spectral('_psi_plus_spectral')
        lam, mu, _ = self._resolve_lame()
        exx, eyy, gxy = strain
        exy = gxy / 2.0
        trace = exx + eyy
        delta = torch.sqrt(((exx - eyy) / 2.0) ** 2 + exy ** 2 + self._spectral_eps)
        e1 = trace / 2.0 + delta
        e2 = trace / 2.0 - delta
        e1_plus = torch.clamp(e1, min=0)
        e2_plus = torch.clamp(e2, min=0)
        tr_plus = torch.clamp(trace, min=0)
        return 0.5 * lam * tr_plus ** 2 + mu * (e1_plus ** 2 + e2_plus ** 2)

    def _psi_plus_spectral_stress(self, strain) -> torch.Tensor:
        """psi+ for stress-spectral split (Miehe §3.2).

        Hooke compliance form (plane-stress 2D):
            Ψ⁺ = (1 / (2E)) · ((σ₁⁺)² + (σ₂⁺)² − 2ν · σ₁⁺ · σ₂⁺)

        where σ_i = λ·tr(ε) + 2μ·e_i are the principal stresses computed
        from the un-degraded linear elastic stress σ = λ·tr(ε)·I + 2μ·ε.
        ``σ_i⁺`` are clamped to non-negative values.

        This is the literal compliance form from Miehe 2010 §3.2 / Yu et
        al. 2021 — strictly correct for plane stress. For plane strain
        the Ψ⁺ scalar is approximate, but the *stress field* itself is
        still computed from the proper plane-strain Lamé pair via
        ``compute_stress_spectral_stress``; only the H-field driver
        (this Ψ⁺) carries the plane-stress compliance assumption. In
        practice the H-field is monotonised (max history), so a mild
        scalar mis-scaling shifts the AT1 onset threshold but not the
        crack pattern. Issue #213 (B7 COMSOL parity).
        """
        E = self.material.E
        nu = self.material.nu
        mu = self.material.mu
        lam = self.material.lam
        exx, eyy, gxy = strain
        exy = gxy / 2.0
        trace = exx + eyy
        delta = torch.sqrt(((exx - eyy) / 2.0) ** 2 + exy ** 2 + self._spectral_eps)
        e1 = trace / 2.0 + delta
        e2 = trace / 2.0 - delta
        s1 = lam * trace + 2.0 * mu * e1
        s2 = lam * trace + 2.0 * mu * e2
        s1_plus = torch.clamp(s1, min=0)
        s2_plus = torch.clamp(s2, min=0)
        return (s1_plus ** 2 + s2_plus ** 2 - 2.0 * nu * s1_plus * s2_plus) / (2.0 * E)

    def _psi_plus_star_convex(self, strain) -> torch.Tensor:
        """Star-convex psi+ (Kumar et al. 2020).

        Per-element:
          tr(eps) >= 0: psi+ = 0.5*eps:C:eps  (full energy)
          tr(eps) <  0: psi+ = mu*dev(eps):dev(eps)  (deviatoric only)
        """
        lam, mu, _ = self._resolve_lame()
        exx, eyy, gxy = strain
        exy = gxy / 2.0
        tr_2d = exx + eyy
        if self.material.plane_stress:
            nu = self.material.nu
            ezz = -nu / (1.0 - nu) * tr_2d
        else:
            ezz = torch.zeros_like(exx)
        tr = tr_2d + ezz
        tension = (tr >= 0)

        # Full elastic energy (tension)
        psi_full = (
            0.5 * lam * tr_2d ** 2
            + mu * (exx ** 2 + eyy ** 2 + 2 * exy ** 2)
        )

        # Deviatoric energy only (compression)
        dev_xx = exx - tr / 3.0
        dev_yy = eyy - tr / 3.0
        dev_zz = ezz - tr / 3.0
        psi_dev = mu * (dev_xx ** 2 + dev_yy ** 2 + dev_zz ** 2 + 2.0 * exy ** 2)

        return torch.where(tension, psi_full, psi_dev)

    def _consistent_mass_d1(self, d: torch.Tensor):
        """Integral of linear damage using the consistent T3 mass."""
        if getattr(self.mesh, 'element_type', 'T3') == 'Q4':
            return torch.dot(torch.ones_like(d), self._q4_mass_matvec(d)).item()
        d_e = d[self.mesh.elements]
        return ((self.mesh.areas / 3.0) * d_e.sum(dim=1)).sum().item()

    def _fracture_energy_terms(self, d: torch.Tensor):
        """Return phase-field fracture surface and gradient energies."""
        mat = self.material
        Kd = self.laplacian_matvec(d)
        if getattr(mat, 'pf_model', 'AT2') == 'AT1':
            E_surf = 3.0 * mat.Gc / (8.0 * mat.l0) * self._consistent_mass_d1(d)
            E_grad = 3.0 * mat.Gc * mat.l0 / 8.0 * torch.dot(d, Kd).item()
            return E_surf, E_grad
        E_surf = mat.Gc / (2.0 * mat.l0) * self._consistent_mass_d2(d)
        E_grad = mat.Gc * mat.l0 / 2.0 * torch.dot(d, Kd).item()
        return E_surf, E_grad

    # ------------------------------------------------------------------ #
    # Crack-driving force dispatcher (issue #248)
    # ------------------------------------------------------------------ #

    def compute_driving_force(self, u: torch.Tensor, strain=None) -> torch.Tensor:
        """Crack-driving scalar D fed into the H-update.

        Selected by ``material.driving_force``:

        * ``'strain_energy'`` (default) — return ``compute_psi_plus(u, strain)``
          unchanged. This is a pure pass-through to preserve bit-identical
          behaviour with the legacy code path.
        * ``'principal_stress'`` — Wu (2020)-style D = ⟨σ₁⟩²/(2E) where σ₁ is
          the maximum principal stress of the *un-degraded* linear-elastic
          stress σ = λ·tr(ε)·I + 2μ·ε and ⟨·⟩ = max(·, 0) suppresses the
          compressive branch. For plane stress this is the literal compliance
          form; for plane strain the scalar carries a mild plane-stress
          approximation analogous to ``_psi_plus_spectral_stress`` (see its
          docstring). The H-field is monotonised, so any mis-scaling shifts
          the AT1 onset threshold but not the final crack pattern.

        Returns
        -------
        D : (E,) tensor — non-negative driving force at element centroids.
        """
        mode = getattr(self.material, 'driving_force', 'strain_energy')
        if mode == 'strain_energy':
            return self.compute_psi_plus(u, strain=strain)
        if mode == 'principal_stress':
            if strain is None:
                strain = self.compute_strain(u)
            return self._driving_force_principal_stress(strain)
        raise ValueError(
            f"Unknown driving_force: {mode!r} "
            f"(expected 'strain_energy' or 'principal_stress')")

    def _driving_force_principal_stress(self, strain) -> torch.Tensor:
        """D = ⟨σ₁⟩² / (2 E) at element centroids (Wu 2020-style).

        σ₁ = (σ_xx + σ_yy)/2 + sqrt(((σ_xx − σ_yy)/2)² + σ_xy²) computed
        from the un-degraded elastic stress σ = λ·tr(ε)·I + 2μ·ε. The
        compressive branch (σ₁ < 0) yields zero, mirroring the irreversibility
        / no-compressive-driving conventions of the strain-spectral split.
        """
        lam, mu, _ = self._resolve_lame()
        E = self.diff_E_field if self.diff_E_field is not None else self.material.E
        exx, eyy, gxy = strain
        exy = gxy / 2.0
        tr = exx + eyy
        sxx = lam * tr + 2.0 * mu * exx
        syy = lam * tr + 2.0 * mu * eyy
        sxy = 2.0 * mu * exy
        avg = (sxx + syy) / 2.0
        R = torch.sqrt(((sxx - syy) / 2.0) ** 2 + sxy ** 2 + 1e-30)
        s1 = avg + R
        s1_plus = torch.clamp(s1, min=0)
        return s1_plus ** 2 / (2.0 * E)

    # ------------------------------------------------------------------ #
    # Scalar Laplacian matvec (for damage stiffness K_lap @ d)
    # ------------------------------------------------------------------ #

    def laplacian_matvec(self, d: torch.Tensor) -> torch.Tensor:
        """K_lap @ d where K_lap[i,j] = sum_e area_e * grad_Ni . grad_Nj.

        This is the FEM stiffness matrix action for the scalar Laplacian.
        Uses precomputed flat indices for scatter.
        """
        if getattr(self.mesh, 'element_type', 'T3') == 'Q4':
            return self._q4_laplacian_matvec(d)
        gp = self.mesh.grad_phi
        d_e = d[self.mesh.elements]  # (E, 3)
        gd_x = (gp[:, :, 0] * d_e).sum(1)  # (E,)
        gd_y = (gp[:, :, 1] * d_e).sum(1)

        contrib = self._areas_col * (gp[:, :, 0] * gd_x.unsqueeze(1) +
                                     gp[:, :, 1] * gd_y.unsqueeze(1))

        out = torch.zeros(self.mesh.n_nodes, dtype=self.dtype,
                          device=self.device)
        out.scatter_add_(0, self._elem_flat, contrib.flatten())
        return out

    def _q4_laplacian_matvec(self, d: torch.Tensor) -> torch.Tensor:
        gp = self.mesh.quad_grad_phi
        wdet = self.mesh.quad_wdetJ
        d_e = d[self.mesh.elements]
        gd_x = torch.einsum('eqa,ea->eq', gp[..., 0], d_e)
        gd_y = torch.einsum('eqa,ea->eq', gp[..., 1], d_e)
        contrib = (
            gp[..., 0] * gd_x.unsqueeze(-1)
            + gp[..., 1] * gd_y.unsqueeze(-1)
        ) * wdet.unsqueeze(-1)
        r_e = contrib.sum(dim=1)
        out = torch.zeros(self.mesh.n_nodes, dtype=self.dtype,
                          device=self.device)
        out.scatter_add_(0, self._elem_flat, r_e.flatten())
        return out

    def _q4_mass_matvec(self, d: torch.Tensor) -> torch.Tensor:
        N = self.mesh.quad_N
        wdet = self.mesh.quad_wdetJ
        d_e = d[self.mesh.elements]
        d_q = torch.einsum('qa,ea->eq', N, d_e)
        r_e = torch.einsum('qa,eq,eq->ea', N, d_q, wdet)
        out = torch.zeros(self.mesh.n_nodes, dtype=self.dtype,
                          device=self.device)
        out.scatter_add_(0, self._elem_flat, r_e.flatten())
        return out

    # ------------------------------------------------------------------ #
    # Stiffness diagonal (for Jacobi preconditioner)
    # ------------------------------------------------------------------ #

    def stiffness_diagonal(self, d=None) -> torch.Tensor:
        """Diagonal of the degraded stiffness matrix K(d).

        For node i, component c, the diagonal entry is:
          K_ii_xx = sum_e g(d_e) * area_e * (C00*(dN_i/dx)^2 + C22*(dN_i/dy)^2)
          K_ii_yy = sum_e g(d_e) * area_e * (C11*(dN_i/dy)^2 + C22*(dN_i/dx)^2)

        Used as a Jacobi preconditioner for the mechanics CG solver.

        Parameters
        ----------
        d : (N,) damage field, or None for undamaged (g_d=1)

        Returns
        -------
        diag : (N, 2) diagonal stiffness entries [K_ii_xx, K_ii_yy]
        """
        mesh = self.mesh
        C = self.C
        if getattr(mesh, 'element_type', 'T3') == 'Q4':
            if d is not None:
                d_q = torch.einsum('qa,ea->eq', mesh.quad_N, d[mesh.elements])
                g_q = self.material.degradation(d_q)
            else:
                g_q = torch.ones_like(mesh.quad_wdetJ)
            gp = mesh.quad_grad_phi
            w = g_q * mesh.quad_wdetJ
            kxx_e = (
                w.unsqueeze(-1)
                * (C[0, 0] * gp[..., 0] ** 2
                   + C[2, 2] * gp[..., 1] ** 2)
            ).sum(dim=1)
            kyy_e = (
                w.unsqueeze(-1)
                * (C[1, 1] * gp[..., 1] ** 2
                   + C[2, 2] * gp[..., 0] ** 2)
            ).sum(dim=1)
            diag = torch.zeros(mesh.n_nodes, 2, dtype=self.dtype,
                               device=self.device)
            diag[:, 0].scatter_add_(0, self._elem_flat, kxx_e.flatten())
            diag[:, 1].scatter_add_(0, self._elem_flat, kyy_e.flatten())
            return diag
        gp = mesh.grad_phi  # (E, 3, 2)

        if d is not None:
            d_e = d[mesh.elements].mean(1)  # (E,)
            g_d = self.material.degradation(d_e)  # (E,)
            ga = (g_d * mesh.areas).unsqueeze(1)  # (E, 1)
        else:
            ga = mesh.areas.unsqueeze(1)  # (E, 1) — undamaged, g_d=1

        kxx = ga * (C[0, 0] * self._gp_x_sq + C[2, 2] * self._gp_y_sq)
        kyy = ga * (C[1, 1] * self._gp_y_sq + C[2, 2] * self._gp_x_sq)

        diag = torch.zeros(mesh.n_nodes, 2, dtype=self.dtype,
                           device=self.device)
        diag[:, 0].scatter_add_(0, self._elem_flat, kxx.flatten())
        diag[:, 1].scatter_add_(0, self._elem_flat, kyy.flatten())
        return diag

    # ------------------------------------------------------------------ #
    # Total energy (for energy-based stagger convergence)
    # ------------------------------------------------------------------ #

    def _consistent_mass_d2(self, d: torch.Tensor) -> float:
        """Compute int(d^2 dOmega) using consistent mass matrix.

        Consistent mass for T3: M_e = (area/12) * [2,1,1; 1,2,1; 1,1,2].
        So d^T M d = sum_e (area_e/12) * sum_{a,b} M_ab * d_a * d_b
                    = sum_e (area_e/12) * (2*sum(d_a^2) + 2*sum_{a<b}(d_a*d_b))
                    = sum_e (area_e/12) * (sum(d_a)^2 + sum(d_a^2))

        This matches what PhaseFieldDamageSolver uses in its CG matvec.
        """
        mesh = self.mesh
        if getattr(mesh, 'element_type', 'T3') == 'Q4':
            return torch.dot(d, self._q4_mass_matvec(d)).item()
        d_e = d[mesh.elements]  # (E, 3)
        d_sum = d_e.sum(1)      # (E,)
        d_sq = (d_e ** 2).sum(1)  # (E,)
        return ((mesh.areas / 12.0) * (d_sum ** 2 + d_sq)).sum().item()

    def compute_total_energy(self, u: torch.Tensor, d: torch.Tensor,
                             strain=None, psi_plus=None) -> float:
        """Total energy functional: E_elastic + E_fracture.

        E_elastic = sum_e g(d_e) * psi+(eps_e) * area_e
        E_fracture = Gc/(2*l0) * int(d^2) + Gc*l0/2 * int(|grad(d)|^2)

        Uses consistent mass matrix for the d^2 integral, matching
        what PhaseFieldDamageSolver minimizes in its CG solve.

        Parameters
        ----------
        u : (N, 2) displacement
        d : (N,) damage
        strain : tuple or None — reuse if already computed
        psi_plus : (E,) or None — reuse if already computed

        Returns
        -------
        E_total : float
        """
        if psi_plus is None:
            if strain is None:
                strain = self.compute_strain(u)
            psi_plus = self.compute_psi_plus(u, strain=strain)

        mesh = self.mesh
        mat = self.material

        # Degraded elastic energy
        if getattr(mesh, 'element_type', 'T3') == 'Q4':
            if mat.energy_split != 'isotropic':
                raise NotImplementedError(
                    "Native Q4 energy integration currently supports "
                    "energy_split='isotropic' only.")
            d_q = torch.einsum('qa,ea->eq', mesh.quad_N, d[mesh.elements])
            g_d = mat.degradation(d_q)
            E_elastic = (g_d * psi_plus * mesh.quad_wdetJ).sum().item()
        else:
            d_e = d[mesh.elements].mean(1)
            g_d = mat.degradation(d_e)

            # Total elastic energy comprises degraded tensile + intact compressive parts
            psi_full = self._psi_plus_isotropic(strain)
            psi_minus = psi_full - psi_plus
            elastic_density = (g_d * psi_plus) + psi_minus
            E_elastic = (elastic_density * mesh.areas).sum().item()

        E_surf, E_grad = self._fracture_energy_terms(d)

        return E_elastic + E_surf + E_grad

    def compute_energy_components(self, u: torch.Tensor, d: torch.Tensor,
                                  v: torch.Tensor = None,
                                  strain=None, psi_plus=None) -> dict:
        """Decomposed energy balance: elastic, kinetic, fracture components.

        Parameters
        ----------
        u : (N, 2) displacement
        d : (N,) damage
        v : (N, 2) velocity (optional — needed for kinetic energy)
        strain, psi_plus : reuse if already computed

        Returns
        -------
        dict with keys: 'elastic', 'kinetic', 'fracture', 'total'
        """
        if strain is None:
            strain = self.compute_strain(u)
        if psi_plus is None:
            psi_plus = self.compute_psi_plus(u, strain=strain)

        mesh = self.mesh
        mat = self.material

        # Elastic energy (degraded tensile + intact compressive)
        if getattr(mesh, 'element_type', 'T3') == 'Q4':
            if mat.energy_split != 'isotropic':
                raise NotImplementedError(
                    "Native Q4 energy integration currently supports "
                    "energy_split='isotropic' only.")
            d_q = torch.einsum('qa,ea->eq', mesh.quad_N, d[mesh.elements])
            g_d = mat.degradation(d_q)
            E_elastic = (g_d * psi_plus * mesh.quad_wdetJ).sum().item()
        else:
            d_e = d[mesh.elements].mean(1)
            g_d = mat.degradation(d_e)
            psi_full = self._psi_plus_isotropic(strain)
            psi_minus = psi_full - psi_plus
            E_elastic = ((g_d * psi_plus + psi_minus) * mesh.areas).sum().item()

        E_frac_surf, E_frac_grad = self._fracture_energy_terms(d)
        E_fracture = E_frac_surf + E_frac_grad

        # Kinetic energy: 0.5 * rho * v^2 * M_scalar
        E_kinetic = 0.0
        if v is not None:
            v_sq = (v ** 2).sum(dim=1)  # (N,)
            E_kinetic = 0.5 * mat.rho * (v_sq * mesh.M_scalar).sum().item()

        return {
            'elastic': E_elastic,
            'kinetic': E_kinetic,
            'fracture': E_fracture,
            'total': E_elastic + E_kinetic + E_fracture,
        }

    # ------------------------------------------------------------------ #
    # Reaction forces
    # ------------------------------------------------------------------ #

    def compute_reaction_force(self, u: torch.Tensor, d: torch.Tensor,
                                node_indices: torch.Tensor,
                                component: int = 1,
                                f_ext=None) -> float:
        """Compute reaction force on a set of nodes.

        R = f_int at constrained DOFs (the force needed to maintain BCs).
        When f_ext is provided, the reaction is f_int - f_ext at those DOFs.

        Parameters
        ----------
        u : (N, 2) displacement
        d : (N,) damage
        node_indices : (M,) node indices where reaction is computed
        component : 0=x, 1=y
        f_ext : (N, 2) or None
            External force vector. If provided, reaction = f_int - f_ext
            at the specified nodes.

        Returns
        -------
        total_reaction : float (sum of reaction at specified nodes/component)
        """
        f_int = self.internal_force(u, d)
        if f_ext is not None:
            reaction = f_int[node_indices, component] - f_ext[node_indices, component]
        else:
            reaction = f_int[node_indices, component]
        return reaction.sum().item()

    # ------------------------------------------------------------------ #
    # Principal stress / strain and derived fields
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_principal_stress(sxx, syy, sxy):
        """Principal stresses from Voigt components.

        Parameters
        ----------
        sxx, syy, sxy : (E,) element stress components

        Returns
        -------
        sigma1, sigma2 : (E,) principal stresses (sigma1 >= sigma2)
        """
        avg = (sxx + syy) / 2.0
        R = torch.sqrt(((sxx - syy) / 2.0) ** 2 + sxy ** 2 + 1e-30)
        return avg + R, avg - R

    @staticmethod
    def compute_principal_strain(exx, eyy, gxy):
        """Principal strains from Voigt components.

        Parameters
        ----------
        exx, eyy : (E,) normal strains
        gxy : (E,) engineering shear strain

        Returns
        -------
        eps1, eps2 : (E,) principal strains (eps1 >= eps2)
        """
        exy = gxy / 2.0
        avg = (exx + eyy) / 2.0
        R = torch.sqrt(((exx - eyy) / 2.0) ** 2 + exy ** 2 + 1e-30)
        return avg + R, avg - R

    @staticmethod
    def compute_max_principal_stress(sxx, syy, sxy):
        """Maximum principal stress sigma1.

        More physically relevant than von Mises for brittle fracture.
        """
        avg = (sxx + syy) / 2.0
        R = torch.sqrt(((sxx - syy) / 2.0) ** 2 + sxy ** 2 + 1e-30)
        return avg + R

    def compute_hydrostatic_stress(self, sxx, syy):
        """Hydrostatic (mean) stress for plane strain.

        In plane strain, sigma_zz = nu * (sigma_xx + sigma_yy).
        sigma_h = (sigma_xx + sigma_yy + sigma_zz) / 3
        """
        nu = self.material.nu
        sigma_zz = nu * (sxx + syy)
        return (sxx + syy + sigma_zz) / 3.0

    def compute_stress_triaxiality(self, sxx, syy, sxy):
        """Stress triaxiality eta = sigma_h / sigma_vm.

        Indicator of stress state: eta > 1/3 = tension-dominated,
        eta < 0 = compression-dominated. Important for ductile-brittle
        transition analysis.

        In plane strain, sigma_zz = nu * (sigma_xx + sigma_yy) is nonzero
        and must be included in both hydrostatic and von Mises computations.
        """
        nu = self.material.nu
        sigma_zz = nu * (sxx + syy)
        sigma_h = (sxx + syy + sigma_zz) / 3.0
        sigma_vm = torch.sqrt(
            sxx ** 2 + syy ** 2 + sigma_zz ** 2
            - sxx * syy - sxx * sigma_zz - syy * sigma_zz
            + 3 * sxy ** 2 + 1e-30)
        return sigma_h / (sigma_vm + 1e-30)

    def compute_crack_tip_position(self, d, threshold=0.5):
        """Estimate crack tip position from the damage field.

        Finds the node with d closest to threshold among nodes where
        d is in [threshold-0.2, threshold+0.2], returns its coordinates.

        Parameters
        ----------
        d : (N,) nodal damage
        threshold : float

        Returns
        -------
        x, y : float — crack tip coordinates, or (nan, nan) if no crack
        """
        mask = (d > threshold - 0.2) & (d < threshold + 0.2)
        if not mask.any():
            return float('nan'), float('nan')
        candidates = torch.where(mask)[0]
        dist_to_thresh = (d[candidates] - threshold).abs()
        tip_idx = candidates[dist_to_thresh.argmin()]
        pos = self.mesh.nodes[tip_idx]
        return pos[0].item(), pos[1].item()

    def compute_crack_length(self, d, threshold=0.95):
        """Estimate crack length from the damage field.

        Computes the bounding-box span of nodes with d > threshold.

        Parameters
        ----------
        d : (N,) nodal damage
        threshold : float

        Returns
        -------
        length : float — approximate crack length in mm
        """
        cracked = d > threshold
        if not cracked.any():
            return 0.0
        pts = self.mesh.nodes[cracked]
        span = pts.max(0).values - pts.min(0).values
        return torch.norm(span).item()

    def compute_physics_loss(self, u: torch.Tensor, d: torch.Tensor,
                             H_elem: torch.Tensor, bc_mask=None,
                             f_ext=None) -> torch.Tensor:
        """Differentiable AT2 residual norm for diagnostics.

        Returns ||R_d||^2 + ||R_u||^2 where:
        - R_d = AT2 damage residual (should be zero at equilibrium)
        - R_u = mechanics residual f_int(u,d) - f_ext (should be zero at equilibrium)

        All operations preserve autograd graphs so gradients can flow back
        through ``u`` and ``d`` when the caller needs sensitivity diagnostics.

        Parameters
        ----------
        u : (N, 2) displacement field
        d : (N,) damage field
        H_elem : (E,) driving force history variable
        bc_mask : (N, 2) bool or None
            If provided, constrained DOFs are excluded from the mechanics
            residual (their equilibrium is enforced by BCs, not by R_u=0).
        f_ext : (N, 2) or None
            External force vector. If provided, the mechanics residual is
            R_u = f_int - f_ext instead of R_u = f_int.

        Returns
        -------
        loss : scalar tensor (differentiable)
        """
        mesh = self.mesh
        mat = self.material
        pf_model = getattr(mat, 'pf_model', 'AT2').upper()
        if pf_model != 'AT2':
            raise NotImplementedError(
                "FEMOperators.compute_physics_loss() currently implements "
                f"the AT2 damage residual only; got pf_model={pf_model!r}. "
                "Use AT2 or add a model-specific residual before using this "
                "loss for AT1/PF-CZM training."
            )
        Gc = mat.Gc
        l0 = mat.l0

        # --- Mechanics residual: f_int - f_ext should be zero at equilibrium ---
        f_int = self.internal_force(u, d)
        if f_ext is not None:
            R_u = f_int - f_ext
        else:
            R_u = f_int
        if bc_mask is not None:
            R_u = R_u.clone()
            R_u[bc_mask] = 0.0  # exclude constrained DOFs
        R_u_sq = (R_u ** 2).sum()

        # --- Damage residual: R_d = A*d - b ---
        # AT2 weak form:  Gc*l0*K_lap*d + (2H + Gc/l0)*M*d - 2H*M*1 = 0
        # where M is the consistent mass matrix: area/12 * [2,1,1;1,2,1;1,1,2].
        # This matches PhaseFieldDamageSolver._Ax() exactly.

        # Laplacian stiffness term: Gc*l0 * K_lap @ d
        K_lap_d = self.laplacian_matvec(d)
        stiff_term = Gc * l0 * K_lap_d

        # Reaction (mass) term: (2H + Gc/l0) * M * d
        # Consistent mass: M_e @ d_e = (area/12) * (d_a + d_sum) for node a
        # where d_sum = sum of all 3 nodal d values in the element.
        areas = mesh.areas
        elements = mesh.elements
        elem_flat = self._elem_flat

        rc = (2.0 * H_elem + Gc / l0) * areas / 12.0  # (E,)
        d_e = d[elements]  # (E, 3)
        d_sum = d_e.sum(1)  # (E,)
        # Per-node contribution: rc * (d_a + d_sum)
        react_contrib = rc.unsqueeze(1) * (d_e + d_sum.unsqueeze(1))  # (E, 3)
        react_term = torch.zeros(mesh.n_nodes, dtype=self.dtype,
                                 device=self.device)
        react_term.scatter_add_(0, elem_flat, react_contrib.flatten())

        # RHS: b = 2H * M * 1  (consistent mass applied to ones vector)
        # M*1_a = area/12 * (1 + 3) = area/3 per node (since 1_sum = 3)
        # But with nodal H: b_a = area/6 * (H_a + H_sum) — see damage_solver
        # For element-level H, all 3 nodes get same H_e, so:
        # b_a = 2*H_e * area/12 * (1 + 3) = 2*H_e * area/3
        rhs_coeff = 2.0 * H_elem * areas / 3.0  # (E,)
        rhs_contrib = rhs_coeff.unsqueeze(1).expand(-1, 3).flatten()
        b = torch.zeros(mesh.n_nodes, dtype=self.dtype, device=self.device)
        b.scatter_add_(0, elem_flat, rhs_contrib)

        R_d = stiff_term + react_term - b
        R_d_sq = (R_d ** 2).sum()

        return R_u_sq + R_d_sq

    def compute_crack_front_indicator(self, d: torch.Tensor, d_prev: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute crack front indicator as d * (dd/dt).

        The product of damage and its time derivative is maximal at propagating
        crack tips, providing a more robust crack front detection than simple
        thresholding (d > 0.5). Zero in both undamaged (d=0) and fully cracked
        (dd/dt=0) regions.

        Reference: COMSOL Multiphysics 6.4 Application Library,
        "Phase-Field Modeling of Dynamic Crack Branching".

        Parameters
        ----------
        d : (N,) current damage field
        d_prev : (N,) damage field from previous step
        dt : float, time step size

        Returns
        -------
        indicator : (N,) crack front indicator (peak at propagating crack tips)
        """
        d_dot = (d - d_prev) / max(dt, 1e-30)
        return d * d_dot

    def compute_energies(self, u, d, strain=None):
        """Compute all energy components for diagnostics.

        Uses consistent mass matrix for the d^2 integral, matching
        what PhaseFieldDamageSolver minimizes in its CG solve.

        Returns
        -------
        dict with keys: elastic, fracture_surface, fracture_gradient,
                        fracture_total, total
        """
        if strain is None:
            strain = self.compute_strain(u)
        psi_plus = self.compute_psi_plus(u, strain=strain)
        psi_full = self._psi_plus_isotropic(strain)
        psi_minus = psi_full - psi_plus

        mesh = self.mesh
        mat = self.material
        d_e = d[mesh.elements].mean(1)
        g_d = mat.degradation(d_e)

        E_elastic = ((g_d * psi_plus + psi_minus) * mesh.areas).sum().item()
        E_surf, E_grad = self._fracture_energy_terms(d)

        return {
            'elastic': E_elastic,
            'fracture_surface': E_surf,
            'fracture_gradient': E_grad,
            'fracture_total': E_surf + E_grad,
            'total': E_elastic + E_surf + E_grad,
        }
