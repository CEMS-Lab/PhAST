from __future__ import annotations

"""
AT1/AT2 phase-field damage solver using CG (Conjugate Gradient).

Solves the AT2 weak form:
  int [Gc*l0*grad(d).grad(delta_d) + (2H + Gc/l0)*d*delta_d] dOmega
    = int 2H*delta_d dOmega

Or the AT1 weak form:
  int [3Gc*l0/4*grad(d).grad(delta_d) + 2H*d*delta_d] dOmega
    = int (2H - 3Gc/(8l0))*delta_d dOmega

AT1 has a damage nucleation threshold: d = 0 is optimal until
H > 3Gc/(16l0). Below this threshold, the RHS is negative and
the solver returns d = 0 (via bounds enforcement).

Uses consistent mass matrix: area/12 * [2,1,1; 1,2,1; 1,1,2] to avoid
the lumped-mass artifact that produces diffuse cracks.

Two methods for enforcing box constraints d in [d_prev, 1]:
  - 'post_clamp'   : Solve unconstrained CG, then clamp. Simple baseline.
  - 'projected_cg' : Projected Preconditioned CG (PPCG) with step-limiting
                      and active-set management. Avoids wasting iterations
                      on nodes pinned at bounds.

CG runs in float64 because Gc*l0=0.0135 vs Gc/l0=540 spans 4 orders of
magnitude; float32 gives visibly wrong (too diffuse) results.

GPU strategy:
  - CUDA: CG in float64 natively on GPU (fast)
  - MPS: float64 not supported -> CG on CPU float64, transfer back
  - AMP: always disabled inside CG (accuracy-critical)
"""

import math
import os
from typing import NamedTuple
import torch

# Opt-in deterministic-CG mode for use with torch.utils.checkpoint
# (use_reentrant=False), which strict-checks that the same number of tensors is
# saved during the original forward and the recomputation. Convergence-based
# early breaks make CG iter count data-dependent, which violates that check;
# this flag forces CG to always run a fixed number of iterations and ignores
# the non-SPD recovery break, eliminating the per-step nondeterminism.
# Set TORCH_PF_CG_DETERMINISTIC=1 in the env to enable. Slightly slower but
# correct for gradient-checkpointed inverse-problem demos.
_CG_DETERMINISTIC = os.environ.get('TORCH_PF_CG_DETERMINISTIC', '0') in ('1', 'true', 'TRUE')
from ..core.fem_operators import FEMOperators
from ..physics.material import Material
from ..utils.device import device_supports_float64


class DamageActiveSet(NamedTuple):
    """Boolean masks for the bound-constrained damage projection."""

    lower: torch.Tensor
    upper: torch.Tensor
    fixed: torch.Tensor
    interior: torch.Tensor

    def interior_weight(self, dtype=None, device=None):
        return self.interior.to(dtype=dtype, device=device)

    def lower_weight(self, dtype=None, device=None):
        return self.lower.to(dtype=dtype, device=device)


def classify_damage_active_set(
        d_new: torch.Tensor,
        d_prev: torch.Tensor,
        *,
        upper_bound: float = 1.0,
        lower_atol: float = 1e-12,
        upper_atol: float = 1e-12,
        fixed: torch.Tensor | None = None) -> DamageActiveSet:
    """Classify damage DOFs after projection onto ``[d_prev, upper]``.

    The upper bound and explicit fixed/pinned constraints take precedence over
    the lower-bound identity path. This matters at saturated points where
    ``d_prev == d_new == 1``: the node is upper-active, not lower-active, so
    no upstream gradient should be routed through ``d_prev``.
    """
    if d_new.shape != d_prev.shape:
        raise ValueError(
            f"d_new and d_prev must have the same shape, got "
            f"{tuple(d_new.shape)} and {tuple(d_prev.shape)}")
    upper = d_new >= (upper_bound - upper_atol)
    if fixed is None:
        fixed_mask = torch.zeros_like(upper, dtype=torch.bool)
    else:
        fixed_mask = fixed.to(device=d_new.device, dtype=torch.bool)
        if fixed_mask.shape != d_new.shape:
            raise ValueError(
                f"fixed mask must have shape {tuple(d_new.shape)}, "
                f"got {tuple(fixed_mask.shape)}")
    lower = (d_new <= d_prev + lower_atol) & ~upper & ~fixed_mask
    upper = upper & ~fixed_mask
    interior = ~(lower | upper | fixed_mask)
    return DamageActiveSet(
        lower=lower,
        upper=upper,
        fixed=fixed_mask,
        interior=interior,
    )


def projected_damage_active_mask(
        d: torch.Tensor,
        lower_bound: torch.Tensor,
        residual: torch.Tensor,
        *,
        upper_bound: float = 1.0,
        bound_atol: float = 1e-14,
        fixed: torch.Tensor | None = None) -> torch.Tensor:
    """Return PPCG active DOFs for the box-constrained damage solve."""
    active = (
        ((d <= lower_bound + bound_atol) & (residual < 0))
        | ((d >= upper_bound - bound_atol) & (residual > 0))
    )
    if fixed is not None:
        active = active | fixed.to(device=d.device, dtype=torch.bool)
    return active


def zero_active_entries(values: torch.Tensor,
                        active: torch.Tensor | None) -> torch.Tensor:
    """Clone ``values`` and zero active entries."""
    out = values.clone()
    if active is not None:
        out[active] = 0.0
    return out


def make_gamma_corrected_Gc_field(solver, Gc_scalar):
    """Build a per-element Gc field from a scalar, baking in gamma correction.

    The phase-field solver (when built with ``material.gamma_correction=True``)
    uses an effective ``Gc_eff_e = Gc * gamma_factor_e`` per element, where
    ``gamma_factor_e = 1 / (1 + h_e / (c_w * l0))`` (Bourdin et al. 2000).
    If you want to differentiate the loss with respect to a scalar ``Gc``
    under gamma correction, construct the field tensor with this helper and
    pass it to ``_AdjointDamageSolveField``. Torch autograd handles the
    chain rule through the multiplication automatically:

        dL/dGc_scalar = sum_e ( gamma_factor_e * dL/dGc_field_e )

    Parameters
    ----------
    solver : PhaseFieldDamageSolver
        Must have been constructed with ``gamma_correction=True`` so that
        the element sizes and ``c_w`` coefficient are available.
    Gc_scalar : torch.Tensor
        0-d tensor (optionally ``requires_grad=True``) for the nominal Gc.

    Returns
    -------
    Gc_field : torch.Tensor, shape ``(n_elems,)``
        ``Gc_scalar * gamma_factor_e`` on the solver's compute device/dtype.
        Differentiable wrt ``Gc_scalar`` via autograd.

    Example
    -------
        Gc_scalar = torch.tensor(2.7, requires_grad=True, dtype=torch.float64)
        Gc_field = make_gamma_corrected_Gc_field(solver, Gc_scalar)
        d_new = _AdjointDamageSolveField.apply(
            solver, H_input, d_prev, Gc_field, l0_t)
        loss = ...
        loss.backward()
        # Gc_scalar.grad now holds the chain-ruled scalar gradient.
    """
    if not solver._gamma_correction:
        raise ValueError(
            "Solver was not built with gamma_correction=True. Either enable "
            "it via material.gamma_correction=True, or use "
            "_AdjointDamageSolveScalar directly with your Gc_scalar tensor.")
    c_w = solver._pf_c_alpha()
    elem_h = solver.fem.mesh.elem_h.detach().to(
        dtype=Gc_scalar.dtype, device=Gc_scalar.device)
    gamma_factor_e = 1.0 / (1.0 + elem_h / (c_w * solver._l0))
    return Gc_scalar * gamma_factor_e


def _adjoint_quadratic_forms_perelem(solver, lam, d_sol):
    """Per-element version of _adjoint_quadratic_forms for spatial Gc(x).

    Returns the three adjoint quadratic forms PER ELEMENT (not summed):
        lam_K_d_e[e] = A_e * (∇λ_e · ∇d_sol_e)            (E,)
        lam_M_d_e[e] = (A_e/12) * (λ_sum * d_sum + λ·d)   (E,)
        lam_M_1_e[e] = (A_e/3) * λ_sum                     (E,)

    For spatially-varying Gc_e (per-element field), the gradient is:
        AT2:  dL/dGc_e = -[l0 * lam_K_d_e[e] + (1/l0) * lam_M_d_e[e]]
        AT1:  dL/dGc_e = -3/(8*l0) * lam_M_1_e[e] - 0.75 * l0 * lam_K_d_e[e]
    These match the scalar formulas element-wise and sum to the scalar
    gradient when Gc is uniform.

    Parameters
    ----------
    solver : PhaseFieldDamageSolver
    lam : (N,) adjoint solution
    d_sol : (N,) forward solution (should be d_unc, not d_sol_clamped)

    Returns
    -------
    lam_K_d_e : (E,) per-element contribution to λ^T K d
    lam_M_d_e : (E,) per-element contribution to λ^T M d
    lam_M_1_e : (E,) per-element contribution to λ^T M 1
    """
    elements = solver._cg_elements
    areas = solver._cg_areas
    gp = solver._cg_grad_phi  # (E, 3, 2)

    lam_e = lam[elements]      # (E, 3)
    d_e = d_sol[elements]      # (E, 3)

    g_lam_x = (gp[:, :, 0] * lam_e).sum(1)
    g_lam_y = (gp[:, :, 1] * lam_e).sum(1)
    g_d_x = (gp[:, :, 0] * d_e).sum(1)
    g_d_y = (gp[:, :, 1] * d_e).sum(1)

    lam_K_d_e = areas * (g_lam_x * g_d_x + g_lam_y * g_d_y)

    lam_sum = lam_e.sum(1)
    d_sum = d_e.sum(1)
    lam_dot_d = (lam_e * d_e).sum(1)
    lam_M_d_e = (areas / 12.0) * (lam_sum * d_sum + lam_dot_d)

    lam_M_1_e = (areas / 3.0) * lam_sum

    return lam_K_d_e, lam_M_d_e, lam_M_1_e


def _adjoint_quadratic_forms(solver, lam, d_sol):
    """Compute the three adjoint quadratic forms used in the (Gc, l0) backward.

    For the AT2 (and AT1) damage system on linear triangles:
        λ^T K d_sol  = sum_e A_e * (∇λ_e · ∇d_sol_e)            [Laplacian]
        λ^T M d_sol  = sum_e (A_e/12) * (λ_sum * d_sum + λ·d)   [consistent mass]
        λ^T M 1      = sum_e (A_e/3) * λ_sum                     [consistent mass × 1]

    All three are scalars. They are linear in λ and bilinear (or linear) in
    d_sol, so the gradients of dL/dGc and dL/dl0 are simple linear combinations.

    Parameters
    ----------
    solver : PhaseFieldDamageSolver
        Provides _cg_elements, _cg_areas, _cg_grad_phi (precomputed CG-side
        mesh data).
    lam : (N,) torch.Tensor
        Adjoint solution from A^T λ = grad_output.
    d_sol : (N,) torch.Tensor
        Damage solution from the forward pass.

    Returns
    -------
    lam_K_d : 0-d tensor — λ^T K d_sol
    lam_M_d : 0-d tensor — λ^T M d_sol
    lam_M_1 : 0-d tensor — λ^T M 1
    """
    elements = solver._cg_elements
    areas = solver._cg_areas
    gp = solver._cg_grad_phi  # (E, 3, 2)

    lam_e = lam[elements]      # (E, 3)
    d_e = d_sol[elements]      # (E, 3)

    # Element gradients of lam and d_sol
    g_lam_x = (gp[:, :, 0] * lam_e).sum(1)   # (E,)
    g_lam_y = (gp[:, :, 1] * lam_e).sum(1)
    g_d_x = (gp[:, :, 0] * d_e).sum(1)
    g_d_y = (gp[:, :, 1] * d_e).sum(1)

    # λ^T K d_sol = sum_e A_e * (g_lam · g_d)
    lam_K_d = (areas * (g_lam_x * g_d_x + g_lam_y * g_d_y)).sum()

    # Consistent mass (per-element):
    # M_e[a,b] = (A_e/12) * (1 + δ_ab)
    # λ^T M_e d_e = (A_e/12) * (λ_sum * d_sum + λ_e · d_e)
    lam_sum = lam_e.sum(1)
    d_sum = d_e.sum(1)
    lam_dot_d = (lam_e * d_e).sum(1)
    lam_M_d = ((areas / 12.0) * (lam_sum * d_sum + lam_dot_d)).sum()

    # λ^T M 1 = sum_e (A_e/3) * λ_sum
    lam_M_1 = ((areas / 3.0) * lam_sum).sum()

    return lam_K_d, lam_M_d, lam_M_1


def _prepare_adjoint_pf_dirichlet(solver, mask, values):
    """Move optional phase-field Dirichlet data to the CG device/dtype."""
    if mask is None:
        return None, None
    fixed, vals = solver._prepare_pf_dirichlet(
        mask, values, solver._cg_device, solver._cg_dtype)
    return fixed, vals


def _masked_adjoint_inputs(
    solver, grad_output, d_sol, d_prev, fixed=None, vals=None,
):
    """Split upstream gradient into linear-system and bound/pass-through parts.

    Pinned phase-field Dirichlet DOFs are eliminated variables. They are
    therefore always removed from the adjoint RHS and from the direct
    d_prev pass-through, independent of whether their value also satisfies an
    irreversibility bound.
    """
    fixed_dev = None
    if fixed is not None:
        fixed_dev = fixed.to(device=d_sol.device, dtype=torch.bool)
    active_set = classify_damage_active_set(d_sol, d_prev, fixed=fixed_dev)
    interior_f = active_set.interior.to(d_sol.dtype)
    lower_f = active_set.lower.to(d_sol.dtype)
    g_full = grad_output.detach() * interior_f.to(
        grad_output.device, grad_output.dtype)
    grad_d_prev = grad_output.detach() * lower_f.to(
        grad_output.device, grad_output.dtype)
    return g_full, grad_d_prev


def _pcg_solve_adjoint_system(
    solver, rhs, reaction_coeff, M_inv, fixed=None, vals=None,
    initial=None, rel_tol=1e-22,
):
    """Jacobi-PCG solve used by adjoint recomputation.

    When ``fixed`` is supplied, those DOFs are treated as eliminated:
    their residual and preconditioner entries are zeroed, and the iterate is
    held at ``vals``. This mirrors the forward ``pf_dirichlet`` semantics and
    lets elements touching pinned nodes contribute correctly through the
    free-node adjoint equations.
    """
    if initial is None:
        x = torch.zeros_like(rhs)
    else:
        x = initial.clone()
    M_eff = M_inv
    if fixed is not None:
        if vals is None:
            vals = torch.zeros_like(rhs)
        x = torch.where(fixed, vals, x)
        M_eff = M_inv.clone()
        M_eff[fixed] = 0.0

    r = rhs - solver._Ax(x, reaction_coeff).clone()
    if fixed is not None:
        r[fixed] = 0.0
    z = M_eff * r
    p = z.clone()
    rz = torch.dot(r, z)
    rhs_norm = max(torch.dot(rhs, rhs).item(), 1e-30)
    tol_sq = rel_tol * rhs_norm
    for _ in range(min(solver.max_iter, 1000)):
        Ap = solver._Ax(p, reaction_coeff).clone()
        if fixed is not None:
            Ap[fixed] = 0.0
        pAp = torch.dot(p, Ap)
        if pAp.item() <= 0:
            break
        alpha = rz / (pAp + 1e-30)
        x = x + alpha * p
        if fixed is not None:
            x = torch.where(fixed, vals, x)
        r = r - alpha * Ap
        if fixed is not None:
            r[fixed] = 0.0
        if torch.dot(r, r).item() < tol_sq:
            break
        z = M_eff * r
        rz_new = torch.dot(r, z)
        beta = rz_new / (rz + 1e-30)
        p = z + beta * p
        if fixed is not None:
            p[fixed] = 0.0
        rz = rz_new
    if fixed is not None:
        x = torch.where(fixed, vals, x)
    return x


class _AdjointDamageSolveScalar(torch.autograd.Function):
    """Implicit differentiation through the CG damage solve, w.r.t. (H, Gc, l0).

    Extends the original _AdjointDamageSolve (which only differentiates w.r.t.
    H) by also propagating gradients to scalar Gc and l0. Used by
    PhaseFieldDamageSolver.solve() when the caller passes Gc and/or l0 as 0-d tensors
    with requires_grad=True.

    Forward
    -------
    Updates the solver's cached material parameters from the supplied
    (Gc, l0) tensors, runs the existing CG solver under no_grad
    (unchanged, fast), then restores the original cache. Saves
    (H_input, d_prev, d_sol, Gc_t, l0_t) for backward.

    Backward
    --------
    1. Solve adjoint A^T λ = grad_output (one extra CG, same operator).
    2. Compute dL/dH using the existing analytical formula.
    3. Compute dL/dGc, dL/dl0 using the chain rule via λ^T K d_sol,
       λ^T M d_sol, λ^T M 1 (see _adjoint_quadratic_forms).

    For AT2 (Gc/l0 enters the reaction term and Gc*l0 the Laplacian):
        A    = Gc*l0 * K + (2H + Gc/l0) * M
        b    = 2H * M*1
        ∂A/∂Gc = l0 * K + (1/l0) * M
        ∂A/∂l0 = Gc * K - (Gc/l0**2) * M
        ∂b/∂Gc = ∂b/∂l0 = 0
        dL/dGc = -[l0 * λKd + (1/l0) * λMd]
        dL/dl0 = -[Gc * λKd - (Gc/l0**2) * λMd]

    For AT1 (3Gc*l0/4 in Laplacian, no Gc/l0 in reaction, source term -3Gc/(8l0)):
        A    = 0.75 * Gc * l0 * K + 2H * M
        b    = (2H - 3Gc/(8l0)) * M*1
        ∂A/∂Gc = 0.75 * l0 * K
        ∂A/∂l0 = 0.75 * Gc * K
        ∂b/∂Gc = -3/(8 l0) * M*1
        ∂b/∂l0 =  3 Gc / (8 l0**2) * M*1
        dL/dGc = -[3/(8 l0) * λM1] - 0.75 * l0 * λKd
        dL/dl0 = +[3 Gc / (8 l0**2) * λM1] - 0.75 * Gc * λKd
    """

    @staticmethod
    def forward(ctx, solver, H_input, d_prev, Gc_t, l0_t,
                pf_dirichlet_mask=None, pf_dirichlet_values=None):
        # Update the solver's cached material parameters from the supplied
        # tensors. The forward CG runs in no_grad as usual; the gradient is
        # recovered analytically in backward via implicit differentiation.
        #
        # gamma_correction handling (issue #93): when the solver is built with
        # gamma_correction=True, the assembly uses per-element Gc_eff_e =
        # Gc * gamma_factor_e (gamma_factor_e = 1/(1 + h_e/(c_w*l0))). To stay
        # orthogonal with _AdjointDamageSolveField, we route through the same
        # per-element override path here, then in backward sum-reduce the
        # per-element Gc gradient back to the scalar via gamma_factor_e:
        #     dL/dGc_scalar = sum_e (gamma_factor_e * dL/dGc_eff_e)
        # The l0 gradient under gamma_correction matches the field-adjoint
        # convention (only the direct dependence through Gc_l0 / Gc/l0 / source;
        # the indirect dependence through gamma_factor_e(l0) is treated as a
        # frozen mesh-resolution scaling, same as make_gamma_corrected_Gc_field
        # and _AdjointDamageSolveField). A warning is emitted if l0_t requires
        # grad under gamma_correction.
        if solver._gamma_correction and l0_t.requires_grad:
            import warnings
            warnings.warn(
                "_AdjointDamageSolveScalar with gamma_correction=True returns "
                "the l0 gradient through the *direct* model dependence only "
                "(Gc_l0, Gc/l0, AT1 source). The indirect dependence of "
                "gamma_factor_e on l0 is not chain-ruled — same convention as "
                "_AdjointDamageSolveField + make_gamma_corrected_Gc_field. "
                "If you need the full d/dl0 you must extend the formula. "
                "See issue #93.",
                RuntimeWarning, stacklevel=3,
            )

        Gc_val = float(Gc_t.detach().item())
        l0_val = float(l0_t.detach().item())

        # Save originals so we can restore even if exceptions happen
        _orig_Gc = solver._Gc
        _orig_l0 = solver._l0
        _orig_Gc_l0 = solver._Gc_l0
        _orig_Gc_over_l0 = solver._Gc_over_l0
        _orig_at1_source = solver._at1_source
        _orig_diag_lap = solver._cg_Gc_l0_diag_lap
        _gc = solver._gamma_correction
        # Per-element snapshots only when gamma_correction is on.
        if _gc:
            _orig_Gc_l0_e = solver._Gc_l0_e.clone()
            _orig_Gc_over_l0_e = solver._Gc_over_l0_e.clone()
            _orig_at1_src_e = solver._at1_source_e.clone()
            _orig_diag_lap_e = solver._cg_Gc_l0_e_diag_lap.clone()
            # Build gamma_factor_e at the user-supplied l0 (matches the field
            # adjoint convention: gamma_factor is treated as a frozen mesh
            # scaling for the purposes of the l0 gradient).
            c_w = solver._pf_c_alpha()
            _gamma_factor_e = 1.0 / (
                1.0 + solver.fem.mesh.elem_h.detach().to(
                    dtype=solver._cg_dtype, device=solver._cg_device)
                / (c_w * l0_val))
            _Gc_e_local = Gc_val * _gamma_factor_e  # (E,)
        else:
            _gamma_factor_e = None
            _Gc_e_local = None

        try:
            solver._Gc = Gc_val
            solver._l0 = l0_val
            if solver._pf_model == 'AT1':
                solver._Gc_l0 = 0.75 * Gc_val * l0_val
                solver._Gc_over_l0 = 0.0
                solver._at1_source = 3.0 * Gc_val / (8.0 * l0_val)
            else:  # AT2
                solver._Gc_l0 = Gc_val * l0_val
                solver._Gc_over_l0 = Gc_val / l0_val
                solver._at1_source = 0.0
            solver._cg_Gc_l0_diag_lap = solver._Gc_l0 * solver._cg_diag_lap

            if _gc:
                if solver._pf_model == 'AT1':
                    solver._Gc_l0_e = 0.75 * _Gc_e_local * l0_val
                    solver._Gc_over_l0_e = torch.zeros_like(_Gc_e_local)
                    solver._at1_source_e = 3.0 * _Gc_e_local / (8.0 * l0_val)
                else:  # AT2
                    solver._Gc_l0_e = _Gc_e_local * l0_val
                    solver._Gc_over_l0_e = _Gc_e_local / l0_val
                    solver._at1_source_e = torch.zeros_like(_Gc_e_local)
                solver._cg_Gc_l0_e_diag_lap = (
                    solver._Gc_l0_e.unsqueeze(1) * solver._cg_diag_lap)

            with torch.no_grad():
                d_new = solver._solve_dispatch(
                    H_input, d_prev, pf_dirichlet_mask, pf_dirichlet_values)
        finally:
            # Always restore the original cache so non-differentiable callers
            # are unaffected.
            solver._Gc = _orig_Gc
            solver._l0 = _orig_l0
            solver._Gc_l0 = _orig_Gc_l0
            solver._Gc_over_l0 = _orig_Gc_over_l0
            solver._at1_source = _orig_at1_source
            solver._cg_Gc_l0_diag_lap = _orig_diag_lap
            if _gc:
                solver._Gc_l0_e = _orig_Gc_l0_e
                solver._Gc_over_l0_e = _orig_Gc_over_l0_e
                solver._at1_source_e = _orig_at1_src_e
                solver._cg_Gc_l0_e_diag_lap = _orig_diag_lap_e

        ctx.solver = solver
        # Save the *current values* used in the forward (Python floats) so
        # backward can reconstruct the operator without re-reading the
        # mutable cache.
        ctx.Gc_val = Gc_val
        ctx.l0_val = l0_val
        ctx.pf_model = solver._pf_model
        ctx.gamma_correction = _gc
        ctx.n_inputs = len(ctx.needs_input_grad)
        ctx.has_pf_dirichlet = pf_dirichlet_mask is not None
        if _gc:
            # Save gamma_factor_e for the per-element → scalar reduction in
            # backward.  Stored on the CG device/dtype.
            if pf_dirichlet_mask is not None:
                ctx.save_for_backward(
                    H_input, d_prev, d_new, _gamma_factor_e,
                    pf_dirichlet_mask, pf_dirichlet_values)
            else:
                ctx.save_for_backward(H_input, d_prev, d_new, _gamma_factor_e)
        else:
            if pf_dirichlet_mask is not None:
                ctx.save_for_backward(
                    H_input, d_prev, d_new,
                    pf_dirichlet_mask, pf_dirichlet_values)
            else:
                ctx.save_for_backward(H_input, d_prev, d_new)
        return d_new

    @staticmethod
    def backward(ctx, grad_output):
        solver = ctx.solver
        _gc = getattr(ctx, 'gamma_correction', False)
        has_pf = getattr(ctx, 'has_pf_dirichlet', False)
        if _gc:
            if has_pf:
                (H_input, d_prev, d_sol, gamma_factor_e,
                 pf_mask, pf_vals) = ctx.saved_tensors
            else:
                H_input, d_prev, d_sol, gamma_factor_e = ctx.saved_tensors
                pf_mask = pf_vals = None
        else:
            if has_pf:
                H_input, d_prev, d_sol, pf_mask, pf_vals = ctx.saved_tensors
            else:
                H_input, d_prev, d_sol = ctx.saved_tensors
                pf_mask = pf_vals = None
            gamma_factor_e = None
        Gc_val = ctx.Gc_val
        l0_val = ctx.l0_val
        pf_model = ctx.pf_model

        # Re-set the cache to the forward values for the adjoint solve
        _orig_Gc = solver._Gc
        _orig_l0 = solver._l0
        _orig_Gc_l0 = solver._Gc_l0
        _orig_Gc_over_l0 = solver._Gc_over_l0
        _orig_at1_source = solver._at1_source
        _orig_diag_lap = solver._cg_Gc_l0_diag_lap
        if _gc:
            _orig_Gc_l0_e = solver._Gc_l0_e.clone()
            _orig_Gc_over_l0_e = solver._Gc_over_l0_e.clone()
            _orig_at1_src_e = solver._at1_source_e.clone()
            _orig_diag_lap_e = solver._cg_Gc_l0_e_diag_lap.clone()
            # Per-element effective Gc used by the forward
            Gc_e_np = Gc_val * gamma_factor_e
        try:
            solver._Gc = Gc_val
            solver._l0 = l0_val
            if pf_model == 'AT1':
                solver._Gc_l0 = 0.75 * Gc_val * l0_val
                solver._Gc_over_l0 = 0.0
                solver._at1_source = 3.0 * Gc_val / (8.0 * l0_val)
            else:
                solver._Gc_l0 = Gc_val * l0_val
                solver._Gc_over_l0 = Gc_val / l0_val
                solver._at1_source = 0.0
            solver._cg_Gc_l0_diag_lap = solver._Gc_l0 * solver._cg_diag_lap
            if _gc:
                if pf_model == 'AT1':
                    solver._Gc_l0_e = 0.75 * Gc_e_np * l0_val
                    solver._Gc_over_l0_e = torch.zeros_like(Gc_e_np)
                    solver._at1_source_e = 3.0 * Gc_e_np / (8.0 * l0_val)
                else:
                    solver._Gc_l0_e = Gc_e_np * l0_val
                    solver._Gc_over_l0_e = Gc_e_np / l0_val
                    solver._at1_source_e = torch.zeros_like(Gc_e_np)
                solver._cg_Gc_l0_e_diag_lap = (
                    solver._Gc_l0_e.unsqueeze(1) * solver._cg_diag_lap)

            with torch.no_grad():
                d_init, d_prev_cg, reaction_coeff, b, M_inv, \
                    orig_device, orig_dtype, need_transfer = \
                    solver._prepare_cg(H_input, d_prev)
                fixed, vals = _prepare_adjoint_pf_dirichlet(
                    solver, pf_mask, pf_vals)

                g_full, grad_d_prev = _masked_adjoint_inputs(
                    solver, grad_output, d_sol, d_prev, fixed, vals)
                g = g_full
                if need_transfer:
                    g = g.to(
                        dtype=solver._cg_dtype, device=solver._cg_device)

                # First, recover the linear-system solution before box
                # clamping. With pf_dirichlet, pinned DOFs are eliminated and
                # held at the prescribed values; elements touching those nodes
                # still contribute through the free equations.
                # This is needed because the adjoint formula
                #   dL/dGc = λ^T (∂b/∂Gc - (∂A/∂Gc) d_unc)
                # uses this pre-clamp d_unc, NOT the clamped d_sol. At
                # interior nodes the two coincide; at lower- or upper-active
                # nodes they differ. Cost: one extra CG solve per backward
                # (same as the forward CG cost), comparable to forward solve.
                init = None
                if fixed is not None:
                    init = torch.zeros_like(b)
                    init[fixed] = vals[fixed]
                d_unc = _pcg_solve_adjoint_system(
                    solver, b, reaction_coeff, M_inv, fixed, vals,
                    initial=init, rel_tol=1e-24)

                # Adjoint solve A λ = g_masked (operator is symmetric).
                lam = _pcg_solve_adjoint_system(
                    solver, g, reaction_coeff, M_inv, fixed, None,
                    initial=None, rel_tol=1e-22)

                if _gc:
                    # Per-element quadratic forms: identical math path as
                    # _AdjointDamageSolveField. Then chain-rule reduce to
                    # the scalar Gc via gamma_factor_e (issue #93).
                    lKd_e, lMd_e, lM1_e = _adjoint_quadratic_forms_perelem(
                        solver, lam, d_unc)
                    if pf_model == 'AT1':
                        grad_Gc_e = (-3.0 / (8.0 * l0_val) * lM1_e
                                     - 0.75 * l0_val * lKd_e)
                        # l0 gradient: direct dependence only (matches the
                        # field-adjoint convention); see forward-side warning.
                        grad_l0_val = (3.0 / (8.0 * l0_val ** 2)
                                        * (Gc_e_np * lM1_e).sum()
                                        - 0.75 * (Gc_e_np * lKd_e).sum())
                    else:  # AT2
                        grad_Gc_e = -(l0_val * lKd_e
                                      + (1.0 / l0_val) * lMd_e)
                        grad_l0_val = -((Gc_e_np * lKd_e).sum()
                                        - (Gc_e_np / l0_val ** 2 * lMd_e).sum())
                    # dL/dGc_scalar = sum_e gamma_factor_e * dL/dGc_eff_e
                    grad_Gc_val = (gamma_factor_e * grad_Gc_e).sum()
                else:
                    # Quadratic forms use d_unc, not d_sol — see comment above.
                    lam_K_d, lam_M_d, lam_M_1 = _adjoint_quadratic_forms(
                        solver, lam, d_unc)

                    # Adjoint formulas (see class docstring)
                    if pf_model == 'AT1':
                        grad_Gc_val = (-3.0 / (8.0 * l0_val) * lam_M_1
                                        - 0.75 * l0_val * lam_K_d)
                        grad_l0_val = (3.0 * Gc_val / (8.0 * l0_val ** 2) * lam_M_1
                                        - 0.75 * Gc_val * lam_K_d)
                    else:  # AT2
                        grad_Gc_val = -(l0_val * lam_K_d
                                         + (1.0 / l0_val) * lam_M_d)
                        grad_l0_val = -(Gc_val * lam_K_d
                                         - (Gc_val / l0_val ** 2) * lam_M_d)

                # Wrap as 0-d tensors on the same device/dtype as the input
                Gc_grad_t = grad_Gc_val.to(dtype=orig_dtype, device=orig_device)
                l0_grad_t = grad_l0_val.to(dtype=orig_dtype, device=orig_device)

            # Compute dL/dH using the same analytical formula as
            # _AdjointDamageSolve (element-level only).
            lam_node = lam.to(d_sol.device, d_sol.dtype)
            elements = solver.fem.mesh.elements
            areas = solver.fem.mesh.areas
            d_e = d_sol[elements]
            lam_e = lam_node[elements]
            d_sum = d_e.sum(dim=1, keepdim=True)
            per_node = 2.0 * areas.unsqueeze(1) * (
                1.0 / 3.0 - (d_e + d_sum) / 12.0)
            grad_H = (lam_e * per_node).sum(dim=1)

        finally:
            solver._Gc = _orig_Gc
            solver._l0 = _orig_l0
            solver._Gc_l0 = _orig_Gc_l0
            solver._Gc_over_l0 = _orig_Gc_over_l0
            solver._at1_source = _orig_at1_source
            solver._cg_Gc_l0_diag_lap = _orig_diag_lap
            if _gc:
                solver._Gc_l0_e = _orig_Gc_l0_e
                solver._Gc_over_l0_e = _orig_Gc_over_l0_e
                solver._at1_source_e = _orig_at1_src_e
                solver._cg_Gc_l0_e_diag_lap = _orig_diag_lap_e

        # Return one grad per forward input: (solver, H_input, d_prev, Gc_t, l0_t)
        grads = (None, grad_H, grad_d_prev, Gc_grad_t, l0_grad_t, None, None)
        return grads[:ctx.n_inputs]


class _AdjointDamageSolveField(torch.autograd.Function):
    """Implicit differentiation through the CG damage solve, w.r.t. a
    per-element Gc(x) field (spatially-varying fracture toughness).

    Use case: inverse recovery of Gc(x) from observations. With k = n_elem
    (typically 10^3–10^5), finite differences are infeasible (k+1 forwards
    per gradient), making autograd the only practical option. This is the
    spatial inversion experiment that is a genuine autograd-only demonstration
    that cannot be matched by Powell+FD or grid search.

    Forward
    -------
    Accepts Gc_e_t as (E,) tensor with requires_grad. Overrides the solver's
    per-element Gc arrays (_Gc_l0_e, _Gc_over_l0_e, _at1_source_e), runs the
    existing CG under no_grad, then restores. Saves (H_input, d_prev, d_sol,
    Gc_e, l0_t) for backward. l0 is kept scalar (the common inverse case).

    Backward
    --------
    1. Solve adjoint A^T λ = grad_output (one extra CG).
    2. Recover d_unc via unconstrained CG (same as scalar version).
    3. Compute per-element quadratic forms via _adjoint_quadratic_forms_perelem.
    4. Per-element gradient:
        AT2:  dL/dGc_e = -[l0 * λKd_e + (1/l0) * λMd_e]      (E,)
        AT1:  dL/dGc_e = -3/(8*l0) * λM1_e - 0.75*l0 * λKd_e (E,)
       Sum over elements recovers the scalar case exactly (unit test).

    Note: this function REQUIRES the solver's _gamma_correction machinery
    to be active, because we reuse _Gc_l0_e / _Gc_over_l0_e / _at1_source_e
    arrays. If the solver was built without gamma_correction=True, build a
    fresh solver with that flag before calling this autograd path.
    """

    @staticmethod
    def forward(ctx, solver, H_input, d_prev, Gc_e_t, l0_t,
                pf_dirichlet_mask=None, pf_dirichlet_values=None):
        if not solver._gamma_correction:
            raise RuntimeError(
                "_AdjointDamageSolveField requires solver._gamma_correction=True. "
                "Build the solver with material.gamma_correction=True so the "
                "per-element Gc arrays (_Gc_l0_e, _Gc_over_l0_e, _at1_source_e) "
                "exist.")
        n_elem_expected = solver._cg_elements.shape[0]
        if Gc_e_t.shape != (n_elem_expected,):
            raise ValueError(
                f"Gc_e_t must be ({n_elem_expected},), got {tuple(Gc_e_t.shape)}")

        l0_val = float(l0_t.detach().item())
        Gc_e_np = Gc_e_t.detach().to(
            dtype=solver._cg_dtype, device=solver._cg_device)

        # Save originals for restoration
        _orig_l0 = solver._l0
        _orig_Gc_l0_e = solver._Gc_l0_e.clone()
        _orig_Gc_over_l0_e = solver._Gc_over_l0_e.clone()
        _orig_at1_src_e = solver._at1_source_e.clone() \
            if hasattr(solver, '_at1_source_e') else None
        _orig_diag_lap = solver._cg_Gc_l0_e_diag_lap.clone() \
            if hasattr(solver, '_cg_Gc_l0_e_diag_lap') else None

        try:
            solver._l0 = l0_val
            if solver._pf_model == 'AT1':
                solver._Gc_l0_e = 0.75 * Gc_e_np * l0_val  # (E,)
                solver._Gc_over_l0_e = torch.zeros_like(Gc_e_np)
                solver._at1_source_e = 3.0 * Gc_e_np / (8.0 * l0_val)  # (E,)
            else:  # AT2
                solver._Gc_l0_e = Gc_e_np * l0_val  # (E,)
                solver._Gc_over_l0_e = Gc_e_np / l0_val  # (E,)
                solver._at1_source_e = torch.zeros_like(Gc_e_np)
            solver._cg_Gc_l0_e_diag_lap = \
                solver._Gc_l0_e.unsqueeze(1) * solver._cg_diag_lap

            with torch.no_grad():
                d_new = solver._solve_dispatch(
                    H_input, d_prev, pf_dirichlet_mask, pf_dirichlet_values)
        finally:
            solver._l0 = _orig_l0
            solver._Gc_l0_e = _orig_Gc_l0_e
            solver._Gc_over_l0_e = _orig_Gc_over_l0_e
            if _orig_at1_src_e is not None:
                solver._at1_source_e = _orig_at1_src_e
            if _orig_diag_lap is not None:
                solver._cg_Gc_l0_e_diag_lap = _orig_diag_lap

        ctx.solver = solver
        ctx.l0_val = l0_val
        ctx.pf_model = solver._pf_model
        ctx.n_inputs = len(ctx.needs_input_grad)
        ctx.has_pf_dirichlet = pf_dirichlet_mask is not None
        if pf_dirichlet_mask is not None:
            ctx.save_for_backward(
                H_input, d_prev, d_new, Gc_e_np,
                pf_dirichlet_mask, pf_dirichlet_values)
        else:
            ctx.save_for_backward(H_input, d_prev, d_new, Gc_e_np)
        return d_new

    @staticmethod
    def backward(ctx, grad_output):
        solver = ctx.solver
        if getattr(ctx, 'has_pf_dirichlet', False):
            H_input, d_prev, d_sol, Gc_e_np, pf_mask, pf_vals = ctx.saved_tensors
        else:
            H_input, d_prev, d_sol, Gc_e_np = ctx.saved_tensors
            pf_mask = pf_vals = None
        l0_val = ctx.l0_val
        pf_model = ctx.pf_model

        # Temporarily re-set per-element caches for the adjoint solve
        _orig_l0 = solver._l0
        _orig_Gc_l0_e = solver._Gc_l0_e.clone()
        _orig_Gc_over_l0_e = solver._Gc_over_l0_e.clone()
        _orig_at1_src_e = solver._at1_source_e.clone()
        _orig_diag_lap = solver._cg_Gc_l0_e_diag_lap.clone()
        try:
            solver._l0 = l0_val
            if pf_model == 'AT1':
                solver._Gc_l0_e = 0.75 * Gc_e_np * l0_val
                solver._Gc_over_l0_e = torch.zeros_like(Gc_e_np)
                solver._at1_source_e = 3.0 * Gc_e_np / (8.0 * l0_val)
            else:
                solver._Gc_l0_e = Gc_e_np * l0_val
                solver._Gc_over_l0_e = Gc_e_np / l0_val
                solver._at1_source_e = torch.zeros_like(Gc_e_np)
            solver._cg_Gc_l0_e_diag_lap = \
                solver._Gc_l0_e.unsqueeze(1) * solver._cg_diag_lap

            with torch.no_grad():
                d_init, d_prev_cg, reaction_coeff, b, M_inv, \
                    orig_device, orig_dtype, need_transfer = \
                    solver._prepare_cg(H_input, d_prev)
                fixed, vals = _prepare_adjoint_pf_dirichlet(
                    solver, pf_mask, pf_vals)

                g_full, grad_d_prev = _masked_adjoint_inputs(
                    solver, grad_output, d_sol, d_prev, fixed, vals)
                g = g_full
                if need_transfer:
                    g = g.to(
                        dtype=solver._cg_dtype, device=solver._cg_device)

                # Pre-clamp linear-system solution. With pf_dirichlet, pinned
                # values are held fixed so A_fp v contributes to free-node
                # sensitivities and elements touching pinned nodes retain
                # correct Gc gradients.
                init = None
                if fixed is not None:
                    init = torch.zeros_like(b)
                    init[fixed] = vals[fixed]
                d_unc = _pcg_solve_adjoint_system(
                    solver, b, reaction_coeff, M_inv, fixed, vals,
                    initial=init, rel_tol=1e-24)

                # Adjoint solve A λ = g_masked
                lam = _pcg_solve_adjoint_system(
                    solver, g, reaction_coeff, M_inv, fixed, None,
                    initial=None, rel_tol=1e-22)

                # Per-element quadratic forms — THE key difference from scalar
                lKd_e, lMd_e, lM1_e = _adjoint_quadratic_forms_perelem(
                    solver, lam, d_unc)

                # Per-element Gc gradient (E,)
                if pf_model == 'AT1':
                    grad_Gc_e = (-3.0 / (8.0 * l0_val) * lM1_e
                                  - 0.75 * l0_val * lKd_e)
                else:  # AT2
                    grad_Gc_e = -(l0_val * lKd_e
                                   + (1.0 / l0_val) * lMd_e)

                # l0 gradient (scalar, same math as scalar case but with Gc_e)
                # Sum the per-element contributions using Gc_e weights
                if pf_model == 'AT1':
                    grad_l0_val = (3.0 / (8.0 * l0_val ** 2)
                                    * (Gc_e_np * lM1_e).sum()
                                    - 0.75 * (Gc_e_np * lKd_e).sum())
                else:  # AT2
                    grad_l0_val = -((Gc_e_np * lKd_e).sum()
                                     - (Gc_e_np / l0_val ** 2 * lMd_e).sum())

                Gc_e_grad_t = grad_Gc_e.to(dtype=orig_dtype, device=orig_device)
                l0_grad_t = grad_l0_val.to(dtype=orig_dtype, device=orig_device)

            # H gradient (element-level, same as scalar version)
            lam_node = lam.to(d_sol.device, d_sol.dtype)
            elements = solver.fem.mesh.elements
            areas = solver.fem.mesh.areas
            d_e = d_sol[elements]
            lam_e = lam_node[elements]
            d_sum = d_e.sum(dim=1, keepdim=True)
            per_node = 2.0 * areas.unsqueeze(1) * (
                1.0 / 3.0 - (d_e + d_sum) / 12.0)
            grad_H = (lam_e * per_node).sum(dim=1)

        finally:
            solver._l0 = _orig_l0
            solver._Gc_l0_e = _orig_Gc_l0_e
            solver._Gc_over_l0_e = _orig_Gc_over_l0_e
            solver._at1_source_e = _orig_at1_src_e
            solver._cg_Gc_l0_e_diag_lap = _orig_diag_lap

        # Forward args: (solver, H_input, d_prev, Gc_e_t, l0_t)
        grads = (None, grad_H, grad_d_prev, Gc_e_grad_t, l0_grad_t, None, None)
        return grads[:ctx.n_inputs]


class _AdjointDamageSolve(torch.autograd.Function):
    """Implicit differentiation through the CG damage solve.

    Uses the adjoint method: for Ax=b where A and b depend on parameters,
    the VJP (backward pass) requires solving the adjoint system A^T lambda = g
    (where g = dL/dx is the upstream gradient). Since A is symmetric for the
    phase-field damage equation, A^T = A and the adjoint uses the same operator.

    Forward: run the existing CG solver under no_grad (fast, unchanged).
    Backward: one additional CG solve to get lambda, then propagate gradients
              through b and A's dependence on H_input.

    References:
        Blondel et al. (2022), "Efficient and modular implicit differentiation"
        Bai et al. (2025), "torch-sla: differentiable sparse linear algebra"
    """

    @staticmethod
    def forward(ctx, solver, H_input, d_prev,
                pf_dirichlet_mask=None, pf_dirichlet_values=None):
        # Run the existing solver under no_grad — unchanged, fast
        with torch.no_grad():
            d_new = solver._solve_dispatch(
                H_input, d_prev, pf_dirichlet_mask, pf_dirichlet_values)

        # Save what we need for backward
        ctx.solver = solver
        ctx.n_inputs = len(ctx.needs_input_grad)
        ctx.has_pf_dirichlet = pf_dirichlet_mask is not None
        if pf_dirichlet_mask is not None:
            ctx.save_for_backward(
                H_input, d_prev, d_new, pf_dirichlet_mask, pf_dirichlet_values)
        else:
            ctx.save_for_backward(H_input, d_prev, d_new)
        return d_new

    @staticmethod
    def backward(ctx, grad_output):
        """Adjoint method: solve A lambda = grad_output, then
        dL/dH = (dL/db) * (db/dH) - lambda^T (dA/dH) x
        """
        solver = ctx.solver
        if getattr(ctx, 'has_pf_dirichlet', False):
            H_input, d_prev, d_sol, pf_mask, pf_vals = ctx.saved_tensors
        else:
            H_input, d_prev, d_sol = ctx.saved_tensors
            pf_mask = pf_vals = None

        # We need to solve A lambda = grad_output using the same CG operator.
        # A is defined by reaction_coeff (from H_input) and the Laplacian.
        # Prepare the CG system with the same H_input to get reaction_coeff.
        with torch.no_grad():
            d_init, d_prev_cg, reaction_coeff, b, M_inv, \
                orig_device, orig_dtype, need_transfer = \
                solver._prepare_cg(H_input, d_prev)
            fixed, vals = _prepare_adjoint_pf_dirichlet(
                solver, pf_mask, pf_vals)

            # Adjoint RHS: grad_output (upstream gradient dL/dd_new)
            g, _grad_d_prev_unused = _masked_adjoint_inputs(
                solver, grad_output, d_sol, d_prev, fixed, vals)
            if need_transfer:
                g = g.to(
                    dtype=solver._cg_dtype, device=solver._cg_device)

            # Solve A lambda = g using simple Jacobi-preconditioned CG
            # (reuse the same _Ax operator with same reaction_coeff)
            lam = _pcg_solve_adjoint_system(
                solver, g, reaction_coeff, M_inv, fixed, None,
                initial=None, rel_tol=1e-10)

            if need_transfer:
                lam = lam.to(dtype=orig_dtype, device=orig_device)

        # Compute dL/dH analytically.
        # The system is: A(H) d = b(H), where:
        #   b_i = sum_e 2*H_e * A_e/3  (lumped contribution from elements containing i)
        #   A(H) includes the reaction term (2H + Gc/l0) * consistent_mass
        #
        # From the adjoint: dL/dH_e = lambda^T (db/dH_e - dA/dH_e * d_sol)
        #
        # db/dH_e = 2 * A_e/3 at each node of element e
        # dA/dH_e d_sol = 2 * (consistent_mass_e * d_sol_e) * A_e/12
        #               = 2 * A_e/12 * [(d_i + d_sum) for node i in e]
        #
        # So: dL/dH_e = sum_{i in e} lambda_i * [2*A_e/3 - 2*A_e/12*(d_i + d_sum)]
        #             = 2*A_e * sum_{i in e} lambda_i * [1/3 - (d_i + d_sum)/12]
        #
        # This gives element-level gradients. For element-level H (default),
        # grad_H has shape (E,).

        lam_np = lam.to(d_sol.device, d_sol.dtype)
        elements = solver.fem.mesh.elements
        areas = solver.fem.mesh.areas
        d_e = d_sol[elements]      # (E, 3)
        lam_e = lam_np[elements]   # (E, 3)
        d_sum = d_e.sum(dim=1, keepdim=True)  # (E, 1)

        # db/dH contribution per element: 2 * A/3 per node
        # dA/dH contribution per element: 2 * A/12 * (d_i + d_sum) per node
        # Net per-node: 2*A * [1/3 - (d_i + d_sum)/12]
        # Weighted by lambda_i and summed over nodes in element
        per_node = 2.0 * areas.unsqueeze(1) * (
            1.0/3.0 - (d_e + d_sum) / 12.0)  # (E, 3)
        grad_H = (lam_e * per_node).sum(dim=1)  # (E,)

        # No gradient for solver itself or d_prev
        grads = (None, grad_H, None, None, None)
        return grads[:ctx.n_inputs]


class PhaseFieldDamageSolver:
    """Solver for AT1/AT2/PF-CZM phase-field damage.

    Dispatches on ``material.pf_model`` and supports AT1 (linear damage
    with a finite nucleation barrier), AT2 (quadratic damage with smooth
    onset), and the Wu PF-CZM cohesive phase-field model. PF-CZM uses a
    nonlinear projected residual solve because both ``g'(d)`` and
    ``alpha'(d)`` depend on the current damage state.

    History: this class was originally named ``AT2DamageSolver`` when
    only AT2 was implemented. Renamed on 2026-04-26 after AT1 had been
    the dominant path for several months. No backward-compat alias is
    provided; importers should use ``PhaseFieldDamageSolver`` directly.

    Parameters
    ----------
    fem : FEMOperators
        FEM operators (provides Laplacian matvec, mesh data).
    tol : float
        CG convergence tolerance (solution-based, matches Akantu).
    max_iter : int
        Maximum CG iterations.
    ctx : DeviceContext or None
        Device context for AMP/profiling/compile. If None, uses defaults.
    use_multigrid : bool
        Legacy parameter. Use ``preconditioner`` instead.
        ``True`` maps to ``preconditioner='gmg'``, ``False`` to ``'jacobi'``.
        Ignored when ``preconditioner`` is explicitly set.
    bounds_method : str
        How box constraints [d_prev, 1] are enforced:
          'post_clamp'   — Solve unconstrained CG, then clamp d to
                           [d_prev, 1]. Simple but wastes CG iterations
                           on nodes pinned at their bound.
          'projected_cg' — Projected Preconditioned CG (PPCG). Enforces
                           bounds during the solve via step-limiting and
                           active-set management. After each CG step, the
                           iterate is kept feasible by capping the step
                           length. Nodes at a bound with residual pushing
                           outward form the active set: their residual and
                           search direction are zeroed so CG only operates
                           on free DOFs. When the active set changes, CG
                           restarts to preserve conjugacy.
        Default: 'post_clamp' for backward compatibility.
    preconditioner : str or None
        Preconditioner for CG: 'jacobi', 'spectral' (element-level max
        eigenvalue scaling -- better than Jacobi as it accounts for
        off-diagonal coupling within elements), 'gmg' (2-level geometric
        multigrid), 'amg' (algebraic multigrid, GPU-native V-cycle with
        PyAMG setup), or 'auto' (try amg -> gmg -> jacobi, picking fastest
        available).
        Overrides ``use_multigrid``. Default: None (falls back to ``use_multigrid``).
    """

    def __init__(self, fem: FEMOperators, tol: float = 1e-5,
                 max_iter: int = 5000, ctx=None,
                 use_multigrid: bool = True,
                 bounds_method: str = 'post_clamp',
                 preconditioner: str = None,
                 nodal_H: bool = False):
        if bounds_method not in ('post_clamp', 'projected_cg', 'direct'):
            raise ValueError(
                f"bounds_method must be 'post_clamp', 'projected_cg', or 'direct', "
                f"got '{bounds_method}'")
        pf_tag = getattr(fem.material, 'pf_model', 'AT2')
        # AT1 + nodal_H is not implemented: the nodal-H CG path
        # (`_prepare_cg_nodal`) hard-codes the AT2 RHS `b_a = A/6 * (H_a + S_H)`
        # and does NOT subtract the AT1 elastic-threshold source term
        # `S_H = 3 Gc / (8 l0)`. Routing AT1 configs through this path
        # silently produces AT2 results. Raise eagerly so the user gets a
        # clear error instead of wrong damage fields. See issue #218.
        if nodal_H and pf_tag in ('AT1', 'PFCZM'):
            raise NotImplementedError(
                f"PhaseFieldDamageSolver: pf_model={pf_tag!r} with nodal_H=True "
                "is not implemented. The nodal-H CG path is AT2-specific and "
                "would silently use the wrong source/reaction terms. "
                "Supported combinations: "
                "(AT2, nodal_H=False), (AT2, nodal_H=True), "
                "(AT1, nodal_H=False), (PFCZM, nodal_H=False). Either disable "
                "nodal_H or use pf_model='AT2'."
            )
        h_tag = ', nodal_H' if nodal_H else ''
        print(f"[PhaseFieldDamageSolver] Initializing CG solver (model={pf_tag}, "
              f"tol={tol:.1e}, max_iter={max_iter}, bounds={bounds_method}"
              f"{h_tag})...",
              flush=True)
        degradation_type = getattr(fem.material, 'degradation_type', 'standard')
        if degradation_type != 'standard' and pf_tag != 'PFCZM':
            raise NotImplementedError(
                "PhaseFieldDamageSolver currently supports only "
                "degradation_type='standard'. Non-standard degradation laws "
                f"({degradation_type!r}) require a nonlinear damage residual "
                "consistent with g'(d)."
            )
        self.fem = fem
        self._bounds_method = bounds_method
        self.material = fem.material
        self.mesh = fem.mesh
        self._native_q4_damage = getattr(self.mesh, 'element_type', 'T3') == 'Q4'
        if self._native_q4_damage:
            if pf_tag != 'AT2' or nodal_H:
                raise NotImplementedError(
                    "Native Q4 damage dispatch currently supports only "
                    "pf_model='AT2' with element/Gauss-point H. AT1, PF-CZM, "
                    "and nodal_H Q4 damage require their own quadrature "
                    "residuals.")
        elif getattr(self.mesh, 'element_type', 'T3') != 'T3':
            raise NotImplementedError(
                "PhaseFieldDamageSolver supports native T3 meshes and the "
                "guarded native Q4 AT2 path only.")
        self.tol = tol
        # Deterministic mode: cap max_iter to make CG runtime predictable; this
        # is the count CG always runs to. 50 iters is well-calibrated for the
        # damage problem in practice (typical convergent runs need 5-30).
        if _CG_DETERMINISTIC:
            self.max_iter = int(os.environ.get('TORCH_PF_CG_FIXED_ITERS', '50'))
            # Audit T1.5 (W4): warn once when the deterministic cap silently
            # overrides a larger user-requested ``max_iter``. CG can return
            # non-converged at the capped iteration count and the caller
            # otherwise has no signal that the cap was applied.
            if max_iter > self.max_iter:
                import warnings
                warnings.warn(
                    f"PhaseFieldDamageSolver: TORCH_PF_CG_DETERMINISTIC is set; "
                    f"max_iter capped from {max_iter} to {self.max_iter} "
                    f"(deterministic mode). CG may return non-converged.",
                    RuntimeWarning, stacklevel=2,
                )
            print(f"[PhaseFieldDamageSolver] DETERMINISTIC mode: max_iter forced to {self.max_iter}",
                  flush=True)
        else:
            self.max_iter = max_iter

        # Audit T1.6 (W4): the inner active-set / adjoint PCG loops in the
        # ``_AdjointDamage*`` autograd Functions cap iterations at
        # ``min(self.max_iter, 1000)`` (and ``min(self.max_iter, 500)`` in
        # the simple adjoint path). Warn once at solver init when the
        # user-requested cap exceeds the hard ceiling, so the caller knows
        # the request will be silently truncated for adjoint solves.
        # One-shot per solver instance; keyed on ``_inner_pcg_cap_warned``.
        self._inner_pcg_cap_warned = False
        self._INNER_PCG_HARD_CAP = 1000
        if self.max_iter > self._INNER_PCG_HARD_CAP:
            import warnings
            warnings.warn(
                f"PhaseFieldDamageSolver: requested max_iter={self.max_iter} "
                f"exceeds the inner adjoint/active-set PCG hard cap of "
                f"{self._INNER_PCG_HARD_CAP}; adjoint solves will be capped at "
                f"{self._INNER_PCG_HARD_CAP} iterations and may return "
                f"non-converged.",
                RuntimeWarning, stacklevel=2,
            )
            self._inner_pcg_cap_warned = True
        self._ctx = ctx
        self._nodal_H = nodal_H
        self.last_converged = None

        # Cache material properties to avoid repeated attribute lookups in CG
        self._Gc = fem.material.Gc
        self._l0 = fem.material.l0
        self._pf_model = getattr(fem.material, 'pf_model', 'AT2')
        self._gamma_correction = getattr(fem.material, 'gamma_correction', False)

        # Model-dependent coefficients:
        # AT2: Gc*l0*K_lap + (2H + Gc/l0)*M*d = 2H*M*1
        # AT1: 3Gc*l0/4*K_lap + 2H*M*d = (2H - 3Gc/(8l0))*M*1
        # Allen-Cahn (AT2-like gradient flow): explicit Euler, no CG
        if self._pf_model == 'AT1':
            self._Gc_l0 = 0.75 * fem.material.Gc * fem.material.l0
            self._Gc_over_l0 = 0.0  # no Gc/l0 in AT1 reaction term
            # Honour Material.at1_threshold override (auto -> 3 Gc / (8 l0)).
            self._at1_source = float(fem.material.at1_source)
        elif self._pf_model == 'PFCZM':
            self._Gc_l0 = 2.0 * fem.material.Gc * fem.material.l0 / math.pi
            self._Gc_over_l0 = fem.material.Gc / (math.pi * fem.material.l0)
            self._at1_source = 0.0
        else:  # AT2 or allencahn (uses AT2-like coefficients)
            self._Gc_l0 = fem.material.Gc * fem.material.l0
            self._Gc_over_l0 = fem.material.Gc / fem.material.l0
            self._at1_source = 0.0

        # Allen-Cahn mobility (only used when pf_model == 'allencahn')
        self._mobility = getattr(fem.material, 'mobility', None)
        self._cg_M_lump = None      # lazily filled on first AC step
        self._ac_dt = None          # set by staggered loop when pf_model='allencahn'
        self.damage_viscosity = 0.0
        self.damage_dt = None
        self.damage_viscosity_reference = None
        self._last_viscous_d_prev = None

        # CG device: MPS doesn't support float64, fallback to CPU
        dev = fem.device
        if isinstance(dev, str):
            dev = torch.device(dev)

        if device_supports_float64(dev):
            self._cg_device = dev
        else:
            self._cg_device = torch.device('cpu')

        self._cg_dtype = torch.float64
        print(f"[PhaseFieldDamageSolver] CG device: {self._cg_device}, dtype: float64",
              flush=True)

        # Precompute float64 mesh data on the selected CG device. On MPS hosts
        # this is CPU because MPS does not support float64 damage CG.
        self._cg_areas = fem.mesh.areas.detach().to(
            dtype=self._cg_dtype, device=self._cg_device)
        self._cg_grad_phi = fem.mesh.grad_phi.detach().to(
            dtype=self._cg_dtype, device=self._cg_device)
        self._cg_elements = fem.mesh.elements.detach().to(
            dtype=torch.long, device=self._cg_device)
        self._cg_n_nodes = fem.mesh.n_nodes
        if self._native_q4_damage:
            self._cg_quad_N = fem.mesh.quad_N.detach().to(
                dtype=self._cg_dtype, device=self._cg_device)
            self._cg_quad_grad_phi = fem.mesh.quad_grad_phi.detach().to(
                dtype=self._cg_dtype, device=self._cg_device)
            self._cg_quad_wdetJ = fem.mesh.quad_wdetJ.detach().to(
                dtype=self._cg_dtype, device=self._cg_device)
        else:
            self._cg_quad_N = None
            self._cg_quad_grad_phi = None
            self._cg_quad_wdetJ = None

        # Precompute flattened indices (avoids per-call flatten)
        self._elem_flat = self._cg_elements.flatten()
        self._areas_col = self._cg_areas.unsqueeze(1)
        self._cg_areas_third = self._cg_areas / 3.0

        # Pre-allocate CG work buffers (reused across _Ax calls to avoid
        # per-iteration malloc — saves 2 tensor allocations per CG step)
        self._ax_out = torch.zeros(
            self._cg_n_nodes, dtype=self._cg_dtype, device=self._cg_device)
        self._ax_react = torch.zeros(
            self._cg_n_nodes, dtype=self._cg_dtype, device=self._cg_device)

        # Precompute static preconditioner geometries
        if self._native_q4_damage:
            gp_q = self._cg_quad_grad_phi
            w_q = self._cg_quad_wdetJ
            self._cg_diag_lap = (
                w_q.unsqueeze(2)
                * (gp_q[..., 0] ** 2 + gp_q[..., 1] ** 2)
            ).sum(dim=1)
        else:
            gp = self._cg_grad_phi
            self._cg_diag_lap = self._areas_col * (
                gp[:, :, 0]**2 + gp[:, :, 1]**2)
        self._cg_Gc_l0_diag_lap = self._Gc_l0 * self._cg_diag_lap

        # Gamma-convergence correction (Bourdin et al. 2000):
        # Gc_eff_e = Gc / (1 + h_e / (c_w * l0))
        # Precompute per-element correction factor and Gc-derived quantities
        self._Gc_l0_e = None  # (E,) or None
        self._Gc_over_l0_e = None  # (E,) or None
        if self._gamma_correction:
            c_w = self._pf_c_alpha()
            elem_h = fem.mesh.elem_h.detach().cpu().to(
                dtype=self._cg_dtype).to(device=self._cg_device)
            gamma_factor_e = 1.0 / (1.0 + elem_h / (c_w * self._l0))  # (E,)
            Gc_e = self._Gc * gamma_factor_e  # (E,)
            if self._pf_model == 'AT1':
                self._Gc_l0_e = 0.75 * Gc_e * self._l0  # (E,)
                self._Gc_over_l0_e = torch.zeros_like(Gc_e)
                self._at1_source_e = 3.0 * Gc_e / (8.0 * self._l0)  # (E,)
            elif self._pf_model == 'PFCZM':
                self._Gc_l0_e = 2.0 * Gc_e * self._l0 / math.pi
                self._Gc_over_l0_e = Gc_e / (math.pi * self._l0)
                self._at1_source_e = torch.zeros_like(Gc_e)
            else:  # AT2
                self._Gc_l0_e = Gc_e * self._l0  # (E,)
                self._Gc_over_l0_e = Gc_e / self._l0  # (E,)
                self._at1_source_e = torch.zeros_like(Gc_e)
            # Precompute per-element Gc_l0 * diag_lap for preconditioner
            self._cg_Gc_l0_e_diag_lap = self._Gc_l0_e.unsqueeze(1) * self._cg_diag_lap
            gc_eff_min = Gc_e.min().item()
            gc_eff_max = Gc_e.max().item()
            print(f"[PhaseFieldDamageSolver] Gamma correction enabled: "
                  f"Gc_eff range [{gc_eff_min:.6f}, {gc_eff_max:.6f}] "
                  f"(nominal Gc={self._Gc})", flush=True)

        # Build optional compiled matvec
        self._compiled_Ax = None
        if (ctx and ctx.compile_solvers and self._cg_device.type == 'cuda'
                and not self._native_q4_damage):
            self._try_compile_matvec()

        # Resolve preconditioner: 'preconditioner' param takes priority
        if preconditioner is not None:
            if preconditioner not in ('jacobi', 'spectral', 'gmg', 'amg',
                                     'amgx', 'auto'):
                raise ValueError(
                    f"preconditioner must be 'jacobi', 'spectral', 'gmg', "
                    f"'amg', 'amgx', or 'auto', got '{preconditioner}'")
            requested = preconditioner
        else:
            requested = 'auto' if use_multigrid else 'jacobi'

        if self._native_q4_damage and requested != 'jacobi':
            requested = 'jacobi'
            print("[PhaseFieldDamageSolver] Native Q4 AT2 path: using "
                  "Jacobi preconditioner", flush=True)

        # AT1 + multigrid is unreliable: zero reaction in undamaged regions
        # makes the coarse matrix singular. PF-CZM is nonlinear and uses a
        # diagonal safeguarded descent, so multigrid preconditioners for the
        # linear AT2 operator are not applicable there either.
        if self._pf_model in ('AT1', 'PFCZM') and requested in ('auto', 'amg', 'gmg'):
            requested = 'jacobi'
            print(f"[PhaseFieldDamageSolver] {self._pf_model} model: using "
                  f"Jacobi preconditioner", flush=True)

        # Build preconditioner with automatic fallback chain:
        #   amg → gmg → jacobi (each fallback prints a message)
        self._multigrid = None
        self._preconditioner = self._build_preconditioner(requested)
        self._use_multigrid = self._preconditioner in ('gmg', 'amg')
        self._amg_fallback_active = False
        self._amg_fail_count = 0          # consecutive AMG failures
        self._amg_cooldown = 0            # steps to wait before retrying AMG
        self._amg_retry_interval = 200    # fixed retry interval (steps between AMG retries)

        print(f"[PhaseFieldDamageSolver] Ready (preconditioner={self._preconditioner}).",
              flush=True)

    def _pf_c_alpha(self) -> float:
        if self._pf_model == 'AT2':
            return 2.0
        if self._pf_model == 'AT1':
            return 8.0 / 3.0
        if self._pf_model == 'PFCZM':
            return math.pi
        return 2.0

    def _element_Gc_cg(self) -> torch.Tensor:
        if self._Gc_l0_e is not None:
            if self._pf_model == 'PFCZM':
                return self._Gc_l0_e * (math.pi / (2.0 * self._l0))
            if self._pf_model == 'AT1':
                return self._Gc_l0_e / (0.75 * self._l0)
            return self._Gc_l0_e / self._l0
        return torch.full(
            (self._cg_elements.shape[0],), float(self._Gc),
            dtype=self._cg_dtype, device=self._cg_device)

    def _pfczm_a1_cg(self, Gc_e: torch.Tensor) -> torch.Tensor:
        scale = max(1.0 - float(self.material.eta_residual), 1.0e-30)
        return (
            4.0 * float(self.material.E) * Gc_e
            / (math.pi * self._l0 * float(self.material.sigma_ts) ** 2 * scale)
        )

    def _build_preconditioner(self, requested):
        """Build the best available preconditioner with fallback chain.

        Tries the requested preconditioner first. If it fails (missing
        dependency, runtime error), falls back to the next-best option
        and prints a message.

        For 'auto', queries the device tier to pick the optimal strategy:
          - CUDA (A100/H100/V100, workstation): AmgX → AMG → GMG
          - MPS: GMG (CG runs on CPU anyway, matrix-free is best)
          - CPU: AMG → GMG (no transfer overhead)

        Fallback chain: amgx → amg → gmg → jacobi (each prints a message).

        Returns the name of the preconditioner that was actually built.
        """
        if requested == 'auto':
            from ..utils.device import get_device_tier
            tier = get_device_tier(self._cg_device)
            print(f"[PhaseFieldDamageSolver] Auto-selecting preconditioner for "
                  f"{tier['name']} ({tier['tier']})", flush=True)
            # CUDA: try amgx (fastest) → amg → gmg → jacobi
            # CPU:  try amg → gmg → jacobi
            # MPS:  try gmg → jacobi
            if tier['type'] == 'cuda':
                chain = ('amgx', 'amg', 'gmg', 'jacobi')
            elif tier['type'] == 'cpu':
                chain = ('amg', 'gmg', 'jacobi')
            else:  # mps
                chain = ('gmg', 'jacobi')
            for choice in chain:
                result = self._build_preconditioner(choice)
                if self._multigrid is not None:
                    return result  # successfully built a real preconditioner
                # else: this choice failed, try next in chain
            return 'jacobi'  # all failed, fall back to Jacobi

        if requested == 'amgx':
            try:
                from .multigrid import AmgXPreconditioner
                self._multigrid = AmgXPreconditioner(
                    self.mesh, Gc_l0=self._Gc_l0,
                    device=self._cg_device, dtype=self._cg_dtype)
                return 'amgx'
            except ImportError:
                print("[PhaseFieldDamageSolver] pyamgx not installed",
                      flush=True)
            except Exception as e:
                print(f"[PhaseFieldDamageSolver] AmgX init failed ({e})",
                      flush=True)
            return 'jacobi'

        if requested == 'amg':
            try:
                from .multigrid import AMGPreconditioner
                self._multigrid = AMGPreconditioner(
                    self.mesh, Gc_l0=self._Gc_l0,
                    device=self._cg_device, dtype=self._cg_dtype)
                return 'amg'
            except ImportError:
                print("[PhaseFieldDamageSolver] PyAMG not installed",
                      flush=True)
            except Exception as e:
                print(f"[PhaseFieldDamageSolver] AMG init failed ({e})",
                      flush=True)
            return 'jacobi'

        if requested == 'gmg':
            try:
                from .multigrid import NodeAggregation, ScalarMultigrid
                agg = NodeAggregation(self.mesh, device=self._cg_device)
                self._multigrid = ScalarMultigrid(
                    self.mesh, agg, Gc_l0=self._Gc_l0,
                    device=self._cg_device, dtype=self._cg_dtype)
                return 'gmg'
            except Exception as e:
                print(f"[PhaseFieldDamageSolver] GMG init failed ({e})",
                      flush=True)
            return 'jacobi'

        if requested == 'spectral':
            # Precompute element-level Laplacian matrices (E, 3, 3) for
            # spectral diagonal computation.  These are geometry-only and
            # do not change between solves.
            gp = self._cg_grad_phi
            gp_x = gp[:, :, 0]  # (E, 3)
            gp_y = gp[:, :, 1]  # (E, 3)
            self._K_local = (self._areas_col.unsqueeze(2) *
                             (gp_x.unsqueeze(2) * gp_x.unsqueeze(1) +
                              gp_y.unsqueeze(2) * gp_y.unsqueeze(1)))  # (E,3,3)
            return 'spectral'

        # 'jacobi' — no multigrid object needed
        return 'jacobi'

    def _clear_amg_hierarchy(self, reason, activate_fallback=False):
        """Clear stale AMG coarse operators so vcycle() cannot use them."""
        if self._preconditioner != 'amg' or self._multigrid is None:
            return
        had_hierarchy = getattr(self._multigrid, '_P', None) is not None
        was_active = self._amg_fallback_active
        clear = getattr(self._multigrid, '_clear_hierarchy', None)
        if clear is not None:
            clear()
        if activate_fallback:
            self._amg_fallback_active = True
            self._amg_cooldown = min(self._amg_retry_interval, 5000)
        if had_hierarchy or (activate_fallback and not was_active):
            print(f"[AMG_QS_FALLBACK] {reason}; cleared AMG hierarchy and "
                  f"using Jacobi fallback", flush=True)

    def _try_runtime_gmg_fallback(self, reason, d=None, reaction_coeff=None):
        """Promote a failed runtime AMG rebuild to GMG before using Jacobi."""
        if self._preconditioner != 'amg':
            return False
        if reaction_coeff is not None and not torch.isfinite(reaction_coeff).all():
            return False
        try:
            from .multigrid import NodeAggregation, ScalarMultigrid
            agg = NodeAggregation(self.mesh, device=self._cg_device)
            if d is not None:
                agg.rebuild(d)
            gmg = ScalarMultigrid(
                self.mesh, agg, Gc_l0=self._Gc_l0,
                device=self._cg_device, dtype=self._cg_dtype)
            if reaction_coeff is not None:
                rebuilt = gmg.update(reaction_coeff)
                if rebuilt is False:
                    return False
            self._multigrid = gmg
            self._preconditioner = 'gmg'
            self._use_multigrid = True
            self._amg_fallback_active = False
            self._amg_cooldown = 0
            self._amg_fail_count = 0
            print(f"[AMG_QS_FALLBACK] {reason}; promoted runtime fallback "
                  f"to GMG", flush=True)
            return True
        except Exception as e:
            print(f"[AMG_QS_FALLBACK] {reason}; GMG fallback failed ({e}); "
                  f"using Jacobi fallback", flush=True)
            return False

    def _try_compile_matvec(self):
        """Attempt to torch.compile the CG matvec for CUDA speedup.

        ``Gc_l0`` is passed as an argument (not closed over) so that
        post-compile mutations of ``self._Gc_l0`` (e.g. the
        ``_AdjointDamage*`` autograd Functions in inverse-problem demos
        that overwrite ``solver._Gc_l0`` in-place) are picked up by the
        compiled callable. Audit T1.2 (W4
        SOLVER_BUGS_OPTIMISATION_AUDIT_2026-05-07).
        """
        try:
            gp = self._cg_grad_phi
            areas_col = self._areas_col
            elements = self._cg_elements
            elem_flat = self._elem_flat
            n_nodes = self._cg_n_nodes
            cg_dtype = self._cg_dtype
            cg_device = self._cg_device

            def _Ax_impl(d, reaction_coeff, Gc_l0):
                d_e = d[elements]
                gd_x = (gp[:, :, 0] * d_e).sum(1)
                gd_y = (gp[:, :, 1] * d_e).sum(1)
                lap = areas_col * (gp[:, :, 0] * gd_x.unsqueeze(1) +
                                   gp[:, :, 1] * gd_y.unsqueeze(1))
                out = torch.zeros(n_nodes, dtype=cg_dtype, device=cg_device)
                out.scatter_add_(0, elem_flat, lap.flatten())
                # Consistent mass: area/12 * [2,1,1;1,2,1;1,1,2] @ d
                # = rc * (d_e + d_sum), where rc = (2H+Gc/l0)*area/12
                d_sum = d_e.sum(1)  # (E,)
                mass_contrib = reaction_coeff.unsqueeze(1) * (d_e + d_sum.unsqueeze(1))
                react = torch.zeros(n_nodes, dtype=cg_dtype, device=cg_device)
                react.scatter_add_(0, elem_flat, mass_contrib.flatten())
                return Gc_l0 * out + react

            self._compiled_Ax = torch.compile(
                _Ax_impl, mode='reduce-overhead')
            print("[PhaseFieldDamageSolver] torch.compile enabled for CG matvec",
                  flush=True)
        except Exception as e:
            print(f"[PhaseFieldDamageSolver] torch.compile failed ({e}), "
                  f"using eager mode", flush=True)

    def _Ax(self, d, reaction_coeff):
        """Combined CG matvec: (Gc*l0*K_lap + M_react) @ d.

        When nodal_H is active, ``reaction_coeff`` is actually H_nodal_cg
        and the triple-product mass matrix is used instead.

        When gamma correction is active, Gc*l0 varies per element and is
        applied before scatter (element-wise weighting).
        """
        if self._native_q4_damage:
            return self._Ax_q4(d, reaction_coeff)
        if self._nodal_H:
            return self._Ax_nodal(d, reaction_coeff)  # rc slot carries H_nodal
        if self._compiled_Ax is not None and self._Gc_l0_e is None:
            # Pass ``Gc_l0`` as an argument so the compiled callable
            # observes live mutations of ``self._Gc_l0`` (audit T1.2).
            return self._compiled_Ax(d, reaction_coeff, self._Gc_l0)

        d_e = d[self._cg_elements]
        gp = self._cg_grad_phi
        gd_x = (gp[:, :, 0] * d_e).sum(1)
        gd_y = (gp[:, :, 1] * d_e).sum(1)
        lap_contrib = self._areas_col * (
            gp[:, :, 0] * gd_x.unsqueeze(1) +
            gp[:, :, 1] * gd_y.unsqueeze(1))

        if self._Gc_l0_e is not None:
            # Per-element Gc*l0 weighting before scatter
            lap_contrib = self._Gc_l0_e.unsqueeze(1) * lap_contrib

        out = self._ax_out
        out.zero_()
        out.scatter_add_(0, self._elem_flat, lap_contrib.flatten())

        # Consistent mass: area/12 * [2,1,1;1,2,1;1,1,2] @ d
        # = rc * (d_e + d_sum), where rc = (2H+Gc/l0)*area/12
        d_sum = d_e.sum(1)  # (E,)
        mass_contrib = reaction_coeff.unsqueeze(1) * (d_e + d_sum.unsqueeze(1))
        react = self._ax_react
        react.zero_()
        react.scatter_add_(0, self._elem_flat, mass_contrib.flatten())

        if self._Gc_l0_e is not None:
            out.add_(react)  # Gc_l0 already applied per-element
            # Audit T1.3 (W4): always return a fresh tensor so semantics match
            # the compiled path (which returns a freshly-allocated tensor).
            # Aliasing the persistent buffer ``self._ax_out`` is fragile — any
            # future refactor that interleaves a second ``_Ax(...)`` call (e.g.
            # a preconditioner reusing the matvec) would silently corrupt the
            # caller's cached residual/search direction without raising.
            return out.clone()
        out.mul_(self._Gc_l0).add_(react)
        # Audit T1.3 (W4): always return a fresh tensor; see comment above.
        return out.clone()

    def _viscous_coefficient(self):
        eta = float(getattr(self, 'damage_viscosity', 0.0) or 0.0)
        if eta <= 0.0:
            return 0.0
        dt = getattr(self, 'damage_dt', None)
        if dt is None or float(dt) <= 0.0:
            raise RuntimeError(
                "damage_viscosity requires PhaseFieldDamageSolver.damage_dt "
                "to be a positive pseudo-time/load increment.")
        return eta / float(dt)

    def _add_viscous_element_mass_rhs(self, b, d_prev_cg, coeff):
        if coeff <= 0.0:
            return b
        d_e = d_prev_cg[self._cg_elements]
        d_sum = d_e.sum(1)
        rhs_contrib = (
            coeff * self._cg_areas / 12.0
        ).unsqueeze(1) * (d_e + d_sum.unsqueeze(1))
        b.scatter_add_(0, self._elem_flat, rhs_contrib.flatten())
        return b

    def _viscous_reference(self, d_prev_cg, *, device, dtype):
        ref = getattr(self, 'damage_viscosity_reference', None)
        if ref is None:
            return d_prev_cg
        return ref.detach().to(device=device, dtype=dtype)

    def _normalize_q4_H(self, H_input):
        """Return Q4 AT2 history at Gauss points as an ``(E, 4)`` tensor."""
        n_elem = self._cg_elements.shape[0]
        if H_input.shape == (n_elem,):
            return H_input.unsqueeze(1).expand(-1, self._cg_quad_N.shape[0])
        if H_input.shape == (n_elem, self._cg_quad_N.shape[0]):
            return H_input
        raise ValueError(
            "Native Q4 AT2 damage expects H with shape "
            f"({n_elem},) or ({n_elem}, {self._cg_quad_N.shape[0]}), "
            f"got {tuple(H_input.shape)}")

    def _Ax_q4(self, d, reaction_density):
        """Q4 AT2 matvec using 2x2 Gauss quadrature.

        ``reaction_density`` is ``2H_q + Gc/l0`` at Gauss points. This differs
        from the T3 path, where the area/12 factor is folded into the
        ``reaction_coeff`` argument.
        """
        elements = self._cg_elements
        N = self._cg_quad_N
        gp = self._cg_quad_grad_phi
        wdet = self._cg_quad_wdetJ
        d_e = d[elements]

        gd_x = torch.einsum('eqa,ea->eq', gp[..., 0], d_e)
        gd_y = torch.einsum('eqa,ea->eq', gp[..., 1], d_e)
        lap_contrib = (
            wdet.unsqueeze(2)
            * (
                gp[..., 0] * gd_x.unsqueeze(2)
                + gp[..., 1] * gd_y.unsqueeze(2)
            )
        ).sum(dim=1)

        if self._Gc_l0_e is not None:
            lap_contrib = self._Gc_l0_e.unsqueeze(1) * lap_contrib

        out = self._ax_out
        out.zero_()
        out.scatter_add_(0, self._elem_flat, lap_contrib.flatten())

        d_q = torch.einsum('qa,ea->eq', N, d_e)
        mass_q = reaction_density * d_q * wdet
        mass_contrib = torch.einsum('qa,eq->ea', N, mass_q)
        react = self._ax_react
        react.zero_()
        react.scatter_add_(0, self._elem_flat, mass_contrib.flatten())

        if self._Gc_l0_e is not None:
            out.add_(react)
            return out.clone()
        out.mul_(self._Gc_l0).add_(react)
        return out.clone()

    def _Ax_nodal(self, d, H_nodal_cg):
        """Matvec with nodal H: uses triple-product mass ∫ N_i N_j N_k dx.

        For linear triangles, the triple-product integrals are:
          ∫ N_i N_j N_k dx = A/10  (i=j=k)
          ∫ N_i N_j N_k dx = A/30  (exactly two equal)
          ∫ N_i N_j N_k dx = A/60  (all different)

        The full mass entry M[a,b] = (H_a+H_b+S_H)*A/30 * (1+δ_ab)
                                   + Gc/l0 * A/12 * (1+δ_ab)
        where S_H = sum of nodal H in element.

        The matvec for the mass part decomposes as:
          (M*d)_a = A/30 * [(2*H_a+S_H)*d_a + (H_a+S_H)*S_d + HD]
                  + Gc/l0 * A/12 * (d_a + S_d)
        where S_d = sum(d_e), HD = dot(H_e, d_e).
        """
        elements = self._cg_elements
        areas = self._cg_areas            # (E,)
        gp = self._cg_grad_phi
        Gc_l0_ratio = (self._Gc_over_l0_e if self._Gc_over_l0_e is not None
                       else self._Gc_over_l0)
        visc = self._viscous_coefficient()
        Gc_l0_ratio_eff = Gc_l0_ratio + visc

        d_e = d[elements]                 # (E, 3)
        H_e = H_nodal_cg[elements]        # (E, 3)

        # Laplacian: Gc*l0 * K_lap * d  (unchanged)
        gd_x = (gp[:, :, 0] * d_e).sum(1)
        gd_y = (gp[:, :, 1] * d_e).sum(1)
        lap_contrib = self._areas_col * (
            gp[:, :, 0] * gd_x.unsqueeze(1) +
            gp[:, :, 1] * gd_y.unsqueeze(1))

        if self._Gc_l0_e is not None:
            lap_contrib = self._Gc_l0_e.unsqueeze(1) * lap_contrib

        out = self._ax_out
        out.zero_()
        out.scatter_add_(0, self._elem_flat, lap_contrib.flatten())

        # Triple-product mass contribution from nodal H
        S_H = H_e.sum(dim=1)              # (E,)
        S_d = d_e.sum(dim=1)              # (E,)
        HD = (H_e * d_e).sum(dim=1)       # (E,)

        # H part: A/30 * [(2*H_a + S_H)*d_a + (H_a + S_H)*S_d + HD]
        term_diag = (2.0 * H_e + S_H.unsqueeze(1)) * d_e           # (E, 3)
        term_offdiag = (H_e + S_H.unsqueeze(1)) * S_d.unsqueeze(1) # (E, 3)
        term_dot = HD.unsqueeze(1).expand_as(d_e)                   # (E, 3)
        mass_H = (areas / 30.0).unsqueeze(1) * (term_diag + term_offdiag + term_dot)

        # Gc/l0 part: Gc/l0 * A/12 * (d_a + S_d)  (standard consistent mass)
        if isinstance(Gc_l0_ratio_eff, torch.Tensor) and Gc_l0_ratio_eff.dim() >= 1:
            rc_Gc = (Gc_l0_ratio_eff * areas / 12.0).unsqueeze(1)  # (E, 1)
        else:
            rc_Gc = (Gc_l0_ratio_eff * areas / 12.0).unsqueeze(1)  # (E, 1)
        mass_Gc = rc_Gc * (d_e + S_d.unsqueeze(1))                 # (E, 3)

        mass_total = mass_H + mass_Gc                               # (E, 3)
        react = self._ax_react
        react.zero_()
        react.scatter_add_(0, self._elem_flat, mass_total.flatten())

        if self._Gc_l0_e is not None:
            out.add_(react)
            # Audit T1.3 (W4): always return a fresh tensor (matches eager
            # ``_Ax`` and the compiled path). Aliased-buffer return is fragile.
            return out.clone()
        out.mul_(self._Gc_l0).add_(react)
        # Audit T1.3 (W4): always return a fresh tensor; see comment above.
        return out.clone()

    def _pfczm_residual_energy(self, H_cg, d_cg):
        """Assemble the Wu PF-CZM damage residual and energy.

        Energy density per element:

            H g(d) + Gc/(pi l0) alpha(d) + Gc l0/pi |grad d|^2

        with ``alpha(d)=2d-d^2`` and Wu's rational cohesive degradation.
        The returned residual is ``dE/dd`` using the same consistent mass
        convention as the AT1/AT2 damage equation.
        """
        elements = self._cg_elements
        areas = self._cg_areas
        gp = self._cg_grad_phi
        d_e = d_cg[elements]
        H_e = H_cg
        Gc_e = self._element_Gc_cg()

        a1_e = self._pfczm_a1_cg(Gc_e)
        g, gp_d, _gpp_d = self.material.pfczm_degradation_derivatives(
            d_e, a1=a1_e)
        alpha = self.material.pfczm_alpha(d_e)
        alpha_p = self.material.pfczm_alpha_prime(d_e)

        q = H_e.unsqueeze(1) * gp_d + (
            Gc_e / (math.pi * self._l0)).unsqueeze(1) * alpha_p
        q_sum = q.sum(dim=1)
        mass_contrib = (areas / 12.0).unsqueeze(1) * (
            q + q_sum.unsqueeze(1))

        gd_x = (gp[:, :, 0] * d_e).sum(1)
        gd_y = (gp[:, :, 1] * d_e).sum(1)
        lap_contrib = (2.0 * Gc_e * self._l0 / math.pi).unsqueeze(1) * (
            self._areas_col * (
                gp[:, :, 0] * gd_x.unsqueeze(1)
                + gp[:, :, 1] * gd_y.unsqueeze(1)))

        residual = torch.zeros(
            self._cg_n_nodes, dtype=self._cg_dtype, device=self._cg_device)
        residual.scatter_add_(
            0, self._elem_flat, (mass_contrib + lap_contrib).flatten())

        local_energy = (areas / 3.0) * (
            H_e * g.sum(dim=1)
            + (Gc_e / (math.pi * self._l0)) * alpha.sum(dim=1))
        grad_energy = (
            Gc_e * self._l0 / math.pi
            * areas * (gd_x * gd_x + gd_y * gd_y))
        energy = local_energy.sum() + grad_energy.sum()
        return residual, energy

    def _pfczm_descent_diag(self, H_cg, d_cg):
        elements = self._cg_elements
        areas = self._cg_areas
        gp = self._cg_grad_phi
        d_e = d_cg[elements]
        Gc_e = self._element_Gc_cg()
        a1_e = self._pfczm_a1_cg(Gc_e)
        _g, _gp_d, gpp_d = self.material.pfczm_degradation_derivatives(
            d_e, a1=a1_e)
        alpha_pp = -2.0

        qprime = H_cg.unsqueeze(1) * gpp_d + (
            Gc_e / (math.pi * self._l0)).unsqueeze(1) * alpha_pp
        mass_diag = (areas / 6.0).unsqueeze(1) * qprime
        lap_diag = (2.0 * Gc_e * self._l0 / math.pi).unsqueeze(1) * (
            self._areas_col * (gp[:, :, 0] ** 2 + gp[:, :, 1] ** 2))
        diag_contrib = torch.abs(mass_diag) + lap_diag + 1.0e-18
        diag = torch.zeros(
            self._cg_n_nodes, dtype=self._cg_dtype, device=self._cg_device)
        diag.scatter_add_(0, self._elem_flat, diag_contrib.flatten())
        return diag.clamp_min(1.0e-18)

    @torch.no_grad()
    def compute_residual(self, H_input: torch.Tensor,
                         d: torch.Tensor) -> torch.Tensor:
        """Compute damage PDE residual: R_d = A*d - b.

        Supports both AT2 and AT1 models (coefficients set at init).
        When ``self._nodal_H`` is True, ``H_input`` is (N,) nodal values.

        Parameters
        ----------
        H_input : (E,) or (N,) history variable (element or nodal).
        d : (N,) current damage field.

        Returns
        -------
        residual : (N,) PDE residual vector (on original device/dtype).
        """
        cg_dev = self._cg_device

        # Compute on CG device/dtype for consistency
        need_transfer = (d.device != cg_dev or d.dtype != self._cg_dtype)
        if need_transfer:
            H_cg = H_input.detach().to(
                dtype=self._cg_dtype, device=cg_dev)
            d_cg = d.detach().to(dtype=self._cg_dtype, device=cg_dev)
        else:
            H_cg = H_input
            d_cg = d

        if self._pf_model == 'PFCZM':
            if self._nodal_H:
                raise NotImplementedError("PF-CZM residual supports element-wise H only.")
            residual, _energy = self._pfczm_residual_energy(H_cg, d_cg)
            if need_transfer:
                residual = residual.to(device=d.device, dtype=d.dtype)
            return residual

        if self._native_q4_damage:
            H_q = self._normalize_q4_H(H_cg)
            if self._Gc_over_l0_e is not None:
                Gc_l0_ratio_q = self._Gc_over_l0_e.unsqueeze(1)
            else:
                Gc_l0_ratio_q = self._Gc_over_l0
            reaction_density = 2.0 * H_q + Gc_l0_ratio_q
            rhs_q = 2.0 * H_q * self._cg_quad_wdetJ
            rhs_contrib = torch.einsum('qa,eq->ea', self._cg_quad_N, rhs_q)
            b = torch.zeros(
                self._cg_n_nodes, dtype=self._cg_dtype, device=cg_dev)
            b.scatter_add_(0, self._elem_flat, rhs_contrib.flatten())
            Ad = self._Ax_q4(d_cg, reaction_density)
            residual = Ad - b
            if need_transfer:
                residual = residual.to(device=d.device, dtype=d.dtype)
            return residual

        if self._nodal_H:
            # Nodal-H: RHS = A/6 * (H_a + S_H)
            H_e = H_cg[self._cg_elements]          # (E, 3)
            S_H = H_e.sum(dim=1)                    # (E,)
            rhs_contrib = (self._cg_areas / 6.0).unsqueeze(1) * (
                H_e + S_H.unsqueeze(1))              # (E, 3)
            b = torch.zeros(self._cg_n_nodes, dtype=self._cg_dtype, device=cg_dev)
            b.scatter_add_(0, self._elem_flat, rhs_contrib.flatten())
            d_visc_prev = getattr(self, '_last_viscous_d_prev', None)
            if d_visc_prev is not None:
                b = self._add_viscous_element_mass_rhs(
                    b, d_visc_prev.to(device=cg_dev, dtype=self._cg_dtype),
                    self._viscous_coefficient())
            Ad = self._Ax_nodal(d_cg, H_cg)
        else:
            # Element-H (original)
            if self._Gc_over_l0_e is not None:
                Gc_l0_ratio = self._Gc_over_l0_e
            else:
                Gc_l0_ratio = self._Gc_over_l0
            visc = self._viscous_coefficient()
            reaction_coeff = (
                2.0 * H_cg + Gc_l0_ratio + visc
            ) * self._cg_areas / 12.0
            if (self._Gc_l0_e is not None
                    and hasattr(self, '_at1_source_e')
                    and self._at1_source_e is not None):
                at1_src = self._at1_source_e
            else:
                at1_src = self._at1_source
            rhs_coeff = 2.0 * H_cg - at1_src
            rhs_contrib = (rhs_coeff * self._cg_areas_third).unsqueeze(1).expand(-1, 3).flatten()
            b = torch.zeros(self._cg_n_nodes, dtype=self._cg_dtype, device=cg_dev)
            b.scatter_add_(0, self._elem_flat, rhs_contrib)
            d_visc_prev = getattr(self, '_last_viscous_d_prev', None)
            if d_visc_prev is not None:
                b = self._add_viscous_element_mass_rhs(
                    b, d_visc_prev.to(device=cg_dev, dtype=self._cg_dtype),
                    visc)
            Ad = self._Ax(d_cg, reaction_coeff)

        residual = Ad - b

        if need_transfer:
            residual = residual.to(device=d.device, dtype=d.dtype)
        return residual

    def _compute_preconditioner(self, reaction_coeff):
        """Compute diagonal preconditioner (Jacobi or spectral).

        For 'jacobi': standard diagonal of A.
        For 'spectral': per-element max eigenvalue of the 3x3 local stiffness,
        scattered to nodes.  Accounts for off-diagonal coupling and generally
        gives a tighter bound than Jacobi.
        """
        if self._preconditioner == 'spectral':
            from .multigrid import _scalar_spectral_diag
            # Determine Gc_l0 (scalar or per-element for gamma correction)
            Gc_l0 = self._Gc_l0_e if self._Gc_l0_e is not None else self._Gc_l0
            return _scalar_spectral_diag(
                reaction_coeff, self._K_local, Gc_l0,
                self._elem_flat, self._cg_n_nodes,
                self._cg_dtype, self._cg_device)

        # Standard Jacobi diagonal (consistent mass: diagonal = 2*rc)
        if self._Gc_l0_e is not None:
            diag_contrib = self._cg_Gc_l0_e_diag_lap + 2.0 * reaction_coeff.unsqueeze(1)
        else:
            diag_contrib = self._cg_Gc_l0_diag_lap + 2.0 * reaction_coeff.unsqueeze(1)
        A_diag = torch.zeros(self._cg_n_nodes, dtype=self._cg_dtype, device=self._cg_device)
        A_diag.scatter_add_(0, self._elem_flat, diag_contrib.flatten())
        return A_diag

    def _prepare_cg(self, H_input, d_prev):
        """Shared setup for both CG variants.

        Returns (d, d_prev_cg, reaction_coeff_or_H, b, M_inv, orig_device,
        orig_dtype, need_transfer) — everything needed to start CG.

        When ``self._nodal_H`` is True, ``H_input`` is (N,) nodal values
        and the returned ``reaction_coeff_or_H`` is the nodal H on CG
        device (passed through to ``_Ax`` which dispatches to ``_Ax_nodal``).
        """
        orig_device = d_prev.device
        orig_dtype = d_prev.dtype
        cg_dev = self._cg_device

        # Transfer to CG device/dtype only if needed
        # Route through CPU first — MPS cannot convert to float64 directly
        need_transfer = (orig_device != cg_dev or
                         orig_dtype != self._cg_dtype)
        initial_guess = getattr(self, '_cg_initial_guess', None)
        if need_transfer:
            H_cg = H_input.detach().to(
                dtype=self._cg_dtype, device=cg_dev)
            d_prev_cg = d_prev.detach().to(
                dtype=self._cg_dtype, device=cg_dev)
            if initial_guess is None:
                d = d_prev_cg.clone()
            else:
                d = initial_guess.detach().to(
                    dtype=self._cg_dtype, device=cg_dev)
        else:
            H_cg = H_input
            d_prev_cg = d_prev.clone()
            d = (d_prev_cg.clone() if initial_guess is None
                 else initial_guess.detach().to(
                     device=cg_dev, dtype=self._cg_dtype).clone())

        if self._native_q4_damage:
            return self._prepare_cg_q4(
                H_cg, d, d_prev_cg, orig_device, orig_dtype,
                need_transfer, cg_dev)

        if self._nodal_H:
            return self._prepare_cg_nodal(
                H_cg, d, d_prev_cg, orig_device, orig_dtype,
                need_transfer, cg_dev)

        # ---------- Element-H path (original) ----------
        H_e = H_cg
        if self._Gc_over_l0_e is not None:
            Gc_l0_ratio = self._Gc_over_l0_e
        else:
            Gc_l0_ratio = self._Gc_over_l0

        visc = self._viscous_coefficient()
        reaction_coeff = (
            2.0 * H_e + Gc_l0_ratio + visc
        ) * self._cg_areas / 12.0

        # Update multigrid coarse operator (if enabled).
        # Skip AMG setup when:
        #   1. Damage hasn't changed significantly (cached hierarchy is fine)
        #   2. AMG fallback is active (Jacobi is being used, no point rebuilding)
        #   3. max(d) < 0.1 (reaction-dominated regime where AMG fails anyway)
        # AMG setup costs 100ms-10s on CPU (pyamg) — only worth it for large
        # meshes with significant damage where CG iteration count is high.
        if self._use_multigrid and self._multigrid is not None:
            max_d = d.max().item()

            # Smart AMG management: use AMG when effective, Jacobi when not.
            # After AMG failure, wait _amg_retry_interval steps before retrying.
            # Back off exponentially on consecutive failures.
            if self._amg_cooldown > 0:
                self._amg_cooldown -= 1

            skip_reason = None
            if max_d < 0.1:
                skip_reason = "max(d)<0.1 reaction-dominated damage solve"
            elif max_d > 0.95:
                skip_reason = "max(d)>0.95 near-fully damaged damage solve"
            elif self._amg_fallback_active and self._amg_cooldown > 0:
                skip_reason = "AMG retry cooldown active"
            skip_amg = skip_reason is not None

            if skip_amg and self._preconditioner == 'amg':
                self._clear_amg_hierarchy(
                    skip_reason, activate_fallback=max_d > 0.95)

            need_rebuild = False
            if not skip_amg:
                if not hasattr(self, '_amg_d_snapshot'):
                    need_rebuild = max_d > 0.05
                else:
                    delta_d = (d - self._amg_d_snapshot).abs().max().item()
                    need_rebuild = delta_d > 0.01
                # Also retry if cooldown expired (periodic retry)
                if self._amg_fallback_active and self._amg_cooldown <= 0:
                    need_rebuild = True
            if need_rebuild:
                try:
                    if hasattr(self._multigrid, 'agg'):
                        self._multigrid.agg.rebuild(d)
                    rebuilt = self._multigrid.update(reaction_coeff)
                    if rebuilt:
                        self._amg_d_snapshot = d.clone()
                        self._amg_fallback_active = False
                        self._amg_fail_count = 0
                    else:
                        self._amg_fail_count += 1
                        if self._try_runtime_gmg_fallback(
                                "AMG rebuild returned false",
                                d, reaction_coeff):
                            self._amg_d_snapshot = d.clone()
                        else:
                            self._amg_fallback_active = True
                            self._amg_cooldown = min(
                                self._amg_retry_interval, 5000)
                except Exception:
                    # AMG rebuild failed — back off exponentially
                    self._amg_fail_count += 1
                    if self._try_runtime_gmg_fallback(
                            "AMG rebuild raised an unexpected exception",
                            d, reaction_coeff):
                        self._amg_d_snapshot = d.clone()
                    else:
                        self._clear_amg_hierarchy(
                            "AMG rebuild raised an unexpected exception",
                            activate_fallback=True)

        # RHS: AT2 → 2H*area/3,  AT1 → (2H - 3Gc/(8l0))*area/3
        if (self._Gc_l0_e is not None
                and hasattr(self, '_at1_source_e')
                and self._at1_source_e is not None):
            at1_src = self._at1_source_e
        else:
            at1_src = self._at1_source
        rhs_coeff = 2.0 * H_e - at1_src  # AT2: at1_source=0
        rhs_contrib = (rhs_coeff * self._cg_areas_third).unsqueeze(1).expand(-1, 3).flatten()
        b = torch.zeros(self._cg_n_nodes, dtype=self._cg_dtype, device=cg_dev)
        b.scatter_add_(0, self._elem_flat, rhs_contrib)
        d_visc_prev = self._viscous_reference(
            d_prev_cg, device=cg_dev, dtype=self._cg_dtype)
        b = self._add_viscous_element_mass_rhs(b, d_visc_prev, visc)

        A_diag = self._compute_preconditioner(reaction_coeff)
        M_inv = 1.0 / (A_diag + 1e-30)
        self._last_viscous_d_prev = d_visc_prev.clone()

        return d, d_prev_cg, reaction_coeff, b, M_inv, orig_device, orig_dtype, need_transfer

    def _prepare_cg_q4(self, H_q4_cg, d, d_prev_cg,
                       orig_device, orig_dtype, need_transfer, cg_dev):
        """CG setup for native Q4 AT2 with Gauss-point history."""
        H_q = self._normalize_q4_H(H_q4_cg)
        if self._Gc_over_l0_e is not None:
            Gc_l0_ratio_q = self._Gc_over_l0_e.unsqueeze(1)
        else:
            Gc_l0_ratio_q = self._Gc_over_l0

        reaction_density = 2.0 * H_q + Gc_l0_ratio_q

        N = self._cg_quad_N
        wdet = self._cg_quad_wdetJ
        rhs_q = 2.0 * H_q * wdet
        rhs_contrib = torch.einsum('qa,eq->ea', N, rhs_q)
        b = torch.zeros(self._cg_n_nodes, dtype=self._cg_dtype, device=cg_dev)
        b.scatter_add_(0, self._elem_flat, rhs_contrib.flatten())

        mass_diag = torch.einsum(
            'qa,eq->ea', N * N, reaction_density * wdet)
        if self._Gc_l0_e is not None:
            lap_diag = self._Gc_l0_e.unsqueeze(1) * self._cg_diag_lap
        else:
            lap_diag = self._Gc_l0 * self._cg_diag_lap
        diag_contrib = lap_diag + mass_diag
        A_diag = torch.zeros(
            self._cg_n_nodes, dtype=self._cg_dtype, device=cg_dev)
        A_diag.scatter_add_(0, self._elem_flat, diag_contrib.flatten())
        M_inv = 1.0 / (A_diag + 1e-30)

        return (d, d_prev_cg, reaction_density, b, M_inv,
                orig_device, orig_dtype, need_transfer)

    def _prepare_cg_nodal(self, H_nodal_cg, d, d_prev_cg,
                          orig_device, orig_dtype, need_transfer, cg_dev):
        """CG setup for nodal H (PhaseFieldX convention).

        RHS:  b_a = A/6 * (H_a + S_H)  per node per element
        Preconditioner diagonal:
          A_ii = Gc*l0 * K_diag + (2*H_a + S_H)*A/15 + Gc/l0 * A/6
        """
        elements = self._cg_elements
        areas = self._cg_areas
        H_e = H_nodal_cg[elements]        # (E, 3)
        S_H = H_e.sum(dim=1)              # (E,)

        # RHS: b_a = A/6 * (H_a + S_H)  (consistent mass with nodal H)
        rhs_contrib = (areas / 6.0).unsqueeze(1) * (H_e + S_H.unsqueeze(1))  # (E, 3)
        b = torch.zeros(self._cg_n_nodes, dtype=self._cg_dtype, device=cg_dev)
        b.scatter_add_(0, self._elem_flat, rhs_contrib.flatten())

        # Jacobi preconditioner diagonal:
        # Mass diagonal: (2*H_a + S_H)*A/15 + Gc/l0 * A/6
        Gc_l0_ratio = (self._Gc_over_l0_e if self._Gc_over_l0_e is not None
                       else self._Gc_over_l0)
        visc = self._viscous_coefficient()
        Gc_l0_ratio_eff = Gc_l0_ratio + visc
        mass_diag = (areas / 15.0).unsqueeze(1) * (2.0 * H_e + S_H.unsqueeze(1))
        if isinstance(Gc_l0_ratio_eff, torch.Tensor) and Gc_l0_ratio_eff.dim() >= 1:
            mass_diag = mass_diag + (Gc_l0_ratio_eff * areas / 6.0).unsqueeze(1)
        else:
            mass_diag = mass_diag + (Gc_l0_ratio_eff * areas / 6.0).unsqueeze(1)
        # Laplacian diagonal
        if self._Gc_l0_e is not None:
            lap_diag = self._cg_Gc_l0_e_diag_lap
        else:
            lap_diag = self._cg_Gc_l0_diag_lap
        diag_contrib = lap_diag + mass_diag                         # (E, 3)
        A_diag = torch.zeros(self._cg_n_nodes, dtype=self._cg_dtype, device=cg_dev)
        A_diag.scatter_add_(0, self._elem_flat, diag_contrib.flatten())
        M_inv = 1.0 / (A_diag + 1e-30)
        d_visc_prev = self._viscous_reference(
            d_prev_cg, device=cg_dev, dtype=self._cg_dtype)
        b = self._add_viscous_element_mass_rhs(b, d_visc_prev, visc)
        self._last_viscous_d_prev = d_visc_prev.clone()

        # Return H_nodal_cg in the reaction_coeff slot — _Ax dispatches to _Ax_nodal
        return (d, d_prev_cg, H_nodal_cg, b, M_inv,
                orig_device, orig_dtype, need_transfer)

    def _prepare_pf_dirichlet(self, mask, values, device, dtype):
        """Return phase-field Dirichlet data on the CG solve device."""
        if mask is None:
            return None, None
        fixed = mask.detach().to(device=device, dtype=torch.bool)
        if not bool(fixed.any()):
            return None, None
        if values is None:
            raise ValueError("pf_dirichlet_values must be supplied with mask")
        vals = values.detach().to(device=device, dtype=dtype)
        return fixed, vals

    def _pin_pf_dirichlet_state(self, d, d_prev, fixed, vals):
        if fixed is None:
            return d, d_prev
        return torch.where(fixed, vals, d), torch.where(fixed, vals, d_prev)

    @staticmethod
    def _tensor_needs_grad(value) -> bool:
        return bool(
            torch.is_grad_enabled()
            and isinstance(value, torch.Tensor)
            and value.requires_grad
        )

    def _solve_forward_with_material_overrides(
            self, H_input, d_prev, *,
            Gc=None, l0=None, Gc_field=None,
            pf_dirichlet_mask=None, pf_dirichlet_values=None):
        """Forward-only solve with temporary material overrides.

        This path is used by no-grad dataset generation and live hybrid
        audits. It temporarily installs the per-element material arrays, runs
        the normal constrained solver, and restores the cached scalar or
        gamma-corrected arrays afterwards.
        """
        if Gc is not None and Gc_field is not None:
            raise ValueError("Pass either Gc (scalar) or Gc_field "
                             "(per-element), not both.")
        if self._pf_model == 'PFCZM' and (
                Gc is not None or l0 is not None or Gc_field is not None):
            raise NotImplementedError(
                "PF-CZM material override solves are not yet implemented. "
                "Build a Material with the target Gc/l0/sigma_ts instead of "
                "using solve(..., Gc=..., l0=..., Gc_field=...).")

        _orig_Gc = self._Gc
        _orig_l0 = self._l0
        _orig_Gc_l0 = self._Gc_l0
        _orig_Gc_over_l0 = self._Gc_over_l0
        _orig_at1_source = self._at1_source
        _orig_diag_lap = self._cg_Gc_l0_diag_lap
        _had_Gc_l0_e = hasattr(self, '_Gc_l0_e')
        _had_Gc_over_l0_e = hasattr(self, '_Gc_over_l0_e')
        _had_at1_source_e = hasattr(self, '_at1_source_e')
        _had_diag_lap_e = hasattr(self, '_cg_Gc_l0_e_diag_lap')
        _orig_Gc_l0_e = (
            self._Gc_l0_e.clone()
            if _had_Gc_l0_e and self._Gc_l0_e is not None else None
        )
        _orig_Gc_over_l0_e = (
            self._Gc_over_l0_e.clone()
            if _had_Gc_over_l0_e and self._Gc_over_l0_e is not None else None
        )
        _orig_at1_source_e = (
            self._at1_source_e.clone()
            if _had_at1_source_e and self._at1_source_e is not None else None
        )
        _orig_diag_lap_e = (
            self._cg_Gc_l0_e_diag_lap.clone()
            if _had_diag_lap_e and self._cg_Gc_l0_e_diag_lap is not None
            else None
        )

        try:
            l0_val = float(self._l0 if l0 is None else l0.detach().item())
            self._l0 = l0_val
            if Gc_field is not None:
                n_elem_expected = self._cg_elements.shape[0]
                if Gc_field.shape != (n_elem_expected,):
                    raise ValueError(
                        f"Gc_field must be ({n_elem_expected},), "
                        f"got {tuple(Gc_field.shape)}")
                Gc_e = Gc_field.detach().to(
                    dtype=self._cg_dtype, device=self._cg_device)
                self._Gc = float(Gc_e.mean().item())
                if self._pf_model == 'AT1':
                    self._Gc_l0 = 0.75 * self._Gc * l0_val
                    self._Gc_over_l0 = 0.0
                    self._at1_source = 3.0 * self._Gc / (8.0 * l0_val)
                    self._Gc_l0_e = 0.75 * Gc_e * l0_val
                    self._Gc_over_l0_e = torch.zeros_like(Gc_e)
                    self._at1_source_e = 3.0 * Gc_e / (8.0 * l0_val)
                else:
                    self._Gc_l0 = self._Gc * l0_val
                    self._Gc_over_l0 = self._Gc / l0_val
                    self._at1_source = 0.0
                    self._Gc_l0_e = Gc_e * l0_val
                    self._Gc_over_l0_e = Gc_e / l0_val
                    self._at1_source_e = torch.zeros_like(Gc_e)
                self._cg_Gc_l0_diag_lap = self._Gc_l0 * self._cg_diag_lap
                self._cg_Gc_l0_e_diag_lap = (
                    self._Gc_l0_e.unsqueeze(1) * self._cg_diag_lap)
            elif Gc is not None or l0 is not None:
                Gc_val = float(self._Gc if Gc is None else Gc.detach().item())
                self._Gc = Gc_val
                if self._pf_model == 'AT1':
                    self._Gc_l0 = 0.75 * Gc_val * l0_val
                    self._Gc_over_l0 = 0.0
                    self._at1_source = 3.0 * Gc_val / (8.0 * l0_val)
                else:
                    self._Gc_l0 = Gc_val * l0_val
                    self._Gc_over_l0 = Gc_val / l0_val
                    self._at1_source = 0.0
                self._cg_Gc_l0_diag_lap = self._Gc_l0 * self._cg_diag_lap

            with torch.no_grad():
                return self._solve_dispatch(
                    H_input, d_prev, pf_dirichlet_mask, pf_dirichlet_values)
        finally:
            self._Gc = _orig_Gc
            self._l0 = _orig_l0
            self._Gc_l0 = _orig_Gc_l0
            self._Gc_over_l0 = _orig_Gc_over_l0
            self._at1_source = _orig_at1_source
            self._cg_Gc_l0_diag_lap = _orig_diag_lap
            if _had_Gc_l0_e:
                self._Gc_l0_e = _orig_Gc_l0_e
            elif hasattr(self, '_Gc_l0_e'):
                delattr(self, '_Gc_l0_e')
            if _had_Gc_over_l0_e:
                self._Gc_over_l0_e = _orig_Gc_over_l0_e
            elif hasattr(self, '_Gc_over_l0_e'):
                delattr(self, '_Gc_over_l0_e')
            if _had_at1_source_e:
                self._at1_source_e = _orig_at1_source_e
            elif hasattr(self, '_at1_source_e'):
                delattr(self, '_at1_source_e')
            if _had_diag_lap_e:
                self._cg_Gc_l0_e_diag_lap = _orig_diag_lap_e
            elif hasattr(self, '_cg_Gc_l0_e_diag_lap'):
                delattr(self, '_cg_Gc_l0_e_diag_lap')

    def solve(self, H_input: torch.Tensor,
              d_prev: torch.Tensor,
              Gc: torch.Tensor = None,
              l0: torch.Tensor = None,
              Gc_field: torch.Tensor = None,
              pf_dirichlet_mask: torch.Tensor = None,
              pf_dirichlet_values: torch.Tensor = None,
              initial_guess: torch.Tensor = None) -> torch.Tensor:
        """Solve AT2 damage field given driving force H.

        Dispatches to either unconstrained CG + post-clamp or projected
        preconditioned CG depending on ``self._bounds_method``.

        When ``self.differentiable`` is False (default), the solve runs
        under ``torch.no_grad()`` for performance.

        When True (or when ``Gc``/``l0``/``Gc_field`` are passed as tensors
        with ``requires_grad=True``), the solver uses the adjoint method
        (implicit differentiation) to provide exact gradients through the
        linear solve at the cost of one additional CG solve in the
        backward pass.

        Parameters
        ----------
        H_input : (E,) or (N,) history variable.
            Element-level when ``nodal_H=False`` (default),
            nodal when ``nodal_H=True`` (PhaseFieldX convention).
        d_prev : (N,) damage from previous step (for irreversibility).
        Gc : 0-d tensor, optional
            Override the cached scalar fracture toughness. If a tensor with
            ``requires_grad=True`` is supplied, gradients will flow back to
            it via implicit differentiation. Mutually exclusive with
            ``Gc_field``.
        l0 : 0-d tensor, optional
            Override the cached regularisation length. Same semantics as
            ``Gc``. Can be combined with either ``Gc`` or ``Gc_field``.
        Gc_field : (E,) tensor, optional
            Spatially-varying per-element fracture toughness for
            high-dimensional inverse recovery (k = n_elem unknowns).
            Routes through ``_AdjointDamageSolveField``, which provides
            gradients for the full field in 2 CG solves — independent
            of ``k``. Requires the solver to have been constructed with
            ``material.gamma_correction=True``.
        initial_guess : (N,) tensor, optional
            Initial guess for the forward CG solve. The lower-bound
            irreversibility constraint remains tied to ``d_prev``.
        Returns
        -------
        d_new : (N,) updated damage field on same device/dtype as d_prev.
        """
        self._cg_initial_guess = initial_guess
        try:
            Gc_needs_grad = self._tensor_needs_grad(Gc)
            l0_needs_grad = self._tensor_needs_grad(l0)
            Gc_field_needs_grad = self._tensor_needs_grad(Gc_field)

            if self._native_q4_damage and (
                    getattr(self, 'differentiable', False)
                    or Gc_needs_grad or l0_needs_grad or Gc_field_needs_grad):
                raise NotImplementedError(
                    "Differentiable native Q4 damage solves are not implemented "
                    "yet. Use forward Q4 AT2 solves without gradient-tracked "
                    "material overrides, or convert Q4 cells to T3 for adjoint "
                    "damage workflows.")

            if self._pf_model == 'PFCZM' and (
                    getattr(self, 'differentiable', False)
                    or Gc_needs_grad or l0_needs_grad or Gc_field_needs_grad):
                raise NotImplementedError(
                    "Differentiable PF-CZM damage solves are not implemented yet; "
                    "the current PF-CZM path is a forward nonlinear projected "
                    "solve with finite residual checks.")

            if pf_dirichlet_mask is not None:
                # Differentiable pf_dirichlet is supported by treating pinned
                # damage DOFs as eliminated in both the forward replay and adjoint
                # solve. Keep routing below so the appropriate autograd Function
                # receives the mask/value tensors.
                if not (
                    getattr(self, 'differentiable', False)
                    or Gc_needs_grad or l0_needs_grad or Gc_field_needs_grad
                ):
                    return self._solve_forward_with_material_overrides(
                        H_input, d_prev, Gc=Gc, l0=l0, Gc_field=Gc_field,
                        pf_dirichlet_mask=pf_dirichlet_mask,
                        pf_dirichlet_values=pf_dirichlet_values)

            # Per-element Gc field — high-dimensional spatial inversion
            if Gc_field is not None:
                if Gc is not None:
                    raise ValueError("Pass either Gc (scalar) or Gc_field "
                                      "(per-element), not both.")
                if l0 is None:
                    l0 = torch.tensor(self._l0, dtype=torch.float64)
                if not Gc_field_needs_grad and not l0_needs_grad:
                    return self._solve_forward_with_material_overrides(
                        H_input, d_prev, l0=l0, Gc_field=Gc_field,
                        pf_dirichlet_mask=pf_dirichlet_mask,
                        pf_dirichlet_values=pf_dirichlet_values)
                return _AdjointDamageSolveField.apply(
                    self, H_input, d_prev, Gc_field, l0,
                    pf_dirichlet_mask, pf_dirichlet_values)

            # Differentiable path with (Gc, l0) scalar gradients
            if Gc is not None or l0 is not None:
                if Gc is None:
                    Gc = torch.tensor(self._Gc, dtype=torch.float64)
                if l0 is None:
                    l0 = torch.tensor(self._l0, dtype=torch.float64)
                if not Gc_needs_grad and not l0_needs_grad:
                    return self._solve_forward_with_material_overrides(
                        H_input, d_prev, Gc=Gc, l0=l0,
                        pf_dirichlet_mask=pf_dirichlet_mask,
                        pf_dirichlet_values=pf_dirichlet_values)
                return _AdjointDamageSolveScalar.apply(
                    self, H_input, d_prev, Gc, l0,
                    pf_dirichlet_mask, pf_dirichlet_values)

            if not getattr(self, 'differentiable', False):
                with torch.no_grad():
                    return self._solve_dispatch(
                        H_input, d_prev, pf_dirichlet_mask, pf_dirichlet_values)
            # Differentiable path (H only): use the original adjoint Function
            return _AdjointDamageSolve.apply(
                self, H_input, d_prev, pf_dirichlet_mask, pf_dirichlet_values)
        finally:
            self._cg_initial_guess = None

    def _solve_dispatch(self, H_input, d_prev,
                        pf_dirichlet_mask=None, pf_dirichlet_values=None):
        """Internal dispatch to the appropriate solve method."""
        # Early exit: if d_prev contains NaN/Inf (from a diverged step),
        # return d_prev as-is to avoid propagating garbage through CG.
        if not torch.isfinite(d_prev).all():
            self.last_iter = 0
            self.last_converged = False
            return d_prev.clone()

        if self._pf_model == 'allencahn':
            dt = getattr(self, '_ac_dt', None)
            if dt is None:
                raise RuntimeError(
                    "Allen-Cahn pf_model requires solver._ac_dt to be set "
                    "before solve(); use step_allencahn(H, d_prev, dt) directly "
                    "or have the staggered loop forward its dt.")
            return self.step_allencahn(H_input, d_prev, dt)

        if self._pf_model == 'PFCZM':
            return self._solve_pfczm_projected(
                H_input, d_prev, pf_dirichlet_mask, pf_dirichlet_values)

        if self._native_q4_damage and self._bounds_method == 'direct':
            raise NotImplementedError(
                "Native Q4 AT2 damage supports matrix-free CG bounds methods "
                "('post_clamp' and 'projected_cg'); direct assembled damage "
                "solve remains T3-only.")

        if self._bounds_method == 'projected_cg':
            return self._solve_projected_cg(
                H_input, d_prev, pf_dirichlet_mask, pf_dirichlet_values)
        elif self._bounds_method == 'direct':
            return self._solve_direct(
                H_input, d_prev, pf_dirichlet_mask, pf_dirichlet_values)
        else:
            return self._solve_post_clamp(
                H_input, d_prev, pf_dirichlet_mask, pf_dirichlet_values)

    @torch.no_grad()
    def _solve_pfczm_projected(self, H_elem: torch.Tensor,
                               d_prev: torch.Tensor,
                               pf_dirichlet_mask=None,
                               pf_dirichlet_values=None) -> torch.Tensor:
        """Bounded nonlinear projected solve for Wu PF-CZM.

        The PF-CZM Euler equation is nonlinear in damage because both
        ``g'(d)`` and ``alpha'(d)`` are state dependent. This routine solves
        the variational inequality over ``d_prev <= d <= 1`` using a
        safeguarded diagonal Newton descent with Armijo-style backtracking.
        It is intentionally conservative: every accepted step is feasible and
        non-increasing in the assembled PF-CZM energy.
        """
        if self._nodal_H:
            raise NotImplementedError("PF-CZM supports element-wise H only.")

        orig_device = d_prev.device
        orig_dtype = d_prev.dtype
        cg_dev = self._cg_device
        need_transfer = (
            orig_device != cg_dev
            or orig_dtype != self._cg_dtype
            or H_elem.device != cg_dev
            or H_elem.dtype != self._cg_dtype
        )
        if need_transfer:
            H_cg = H_elem.detach().to(
                dtype=self._cg_dtype, device=cg_dev)
            d_prev_cg = d_prev.detach().to(
                dtype=self._cg_dtype, device=cg_dev)
        else:
            H_cg = H_elem.detach()
            d_prev_cg = d_prev.detach().clone()

        d = torch.clamp(d_prev_cg.clone(), 0.0, 1.0)
        lb = d.clone()
        fixed, vals = self._prepare_pf_dirichlet(
            pf_dirichlet_mask, pf_dirichlet_values,
            self._cg_device, self._cg_dtype)
        if fixed is not None:
            vals = torch.clamp(vals, 0.0, 1.0)
            d = torch.where(fixed, vals, d)
            lb = torch.where(fixed, vals, lb)

        ref_norm = max(math.sqrt(float(self._cg_n_nodes)), 1.0)
        tol = self.tol * ref_norm
        self.last_iter = self.max_iter
        self.last_residual = float('inf')
        self.last_energy = float('nan')
        self.last_converged = False

        for i in range(self.max_iter):
            residual, energy = self._pfczm_residual_energy(H_cg, d)
            if fixed is not None:
                residual[fixed] = 0.0
            active = (
                ((d <= lb + 1e-14) & (residual > 0.0))
                | ((d >= 1.0 - 1e-14) & (residual < 0.0))
            )
            if fixed is not None:
                active = active | fixed
            projected = zero_active_entries(residual, active)
            projected_norm = torch.linalg.vector_norm(projected).item()
            self.last_residual = projected_norm
            self.last_energy = float(energy.item())
            if projected_norm <= tol:
                self.last_iter = i
                self.last_converged = True
                break

            diag = self._pfczm_descent_diag(H_cg, d)
            direction = -projected / diag
            direction[active] = 0.0
            dir_norm = torch.linalg.vector_norm(direction).item()
            if not math.isfinite(dir_norm) or dir_norm <= 1e-30:
                self.last_iter = i
                break

            descent = torch.dot(projected, direction).item()
            if descent >= 0.0:
                direction = -projected
                direction[active] = 0.0
                descent = torch.dot(projected, direction).item()
            if descent >= 0.0:
                self.last_iter = i
                break

            accepted = False
            step = 1.0
            for _ in range(30):
                cand = d + step * direction
                cand = torch.clamp(torch.maximum(cand, lb), 0.0, 1.0)
                if fixed is not None:
                    cand = torch.where(fixed, vals, cand)
                _res_cand, energy_cand = self._pfczm_residual_energy(H_cg, cand)
                if torch.isfinite(energy_cand) and (
                        energy_cand.item()
                        <= energy.item() + 1.0e-12 * max(abs(energy.item()), 1.0)):
                    d = cand
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                self.last_iter = i
                break
        else:
            self.last_iter = self.max_iter

        d = torch.clamp(torch.maximum(d, lb), 0.0, 1.0)
        if fixed is not None:
            d = torch.where(fixed, vals, d)
        if need_transfer:
            return d.to(device=orig_device, dtype=orig_dtype)
        return d

    @torch.no_grad()
    def step_allencahn(self, H_input: torch.Tensor,
                       d_prev: torch.Tensor,
                       dt: float) -> torch.Tensor:
        """Forward-Euler explicit step for the Allen-Cahn (gradient-flow)
        phase-field variant.

        PDE:  ∂d/∂t = -M · δE/δd
                   = -M · [ -Gc·l0·Δd + (Gc/l0 + 2H)·d - 2H ]   (AT2-like)

        Discretisation (lumped mass projection of δE/δd to nodes):
            r_a   = (Gc·l0 · K_lap · d)_a + M_a · [(Gc/l0 + 2H_a) d_a - 2H_a]
            (δE/δd)_a ≈ r_a / M_a_lump
            d_a^{n+1} = clamp(max(d_a^n,
                              d_a^n - dt · M_mob · r_a / M_a_lump), 0, 1)

        Irreversibility is enforced by the elementwise max with d_prev
        (no projection / CG required for AC). Stability bound used by
        callers: dt ≤ 1 / (M_mob · (Gc/l0 + 2 H_max)).

        Parameters
        ----------
        H_input : (E,) element-wise history (H_nodal=False mode only).
        d_prev  : (N,) damage from previous step.
        dt      : float, the explicit time step.

        Returns
        -------
        d_new   : (N,) damage on the same device/dtype as d_prev.
        """
        if self._nodal_H:
            raise NotImplementedError(
                "Allen-Cahn step currently supports element-wise H only.")
        if self._gamma_correction:
            # Smoke-test scope: not wired for per-element Gc gamma correction
            raise NotImplementedError(
                "Allen-Cahn + gamma_correction not implemented in this build.")

        orig_device = d_prev.device
        orig_dtype = d_prev.dtype

        # Move to CG device/dtype (CPU float64 on MPS)
        d = d_prev.detach().to(self._cg_device, self._cg_dtype)
        H = H_input.detach().to(self._cg_device, self._cg_dtype)

        # 1) Laplacian contribution: K_lap @ d, scattered to nodes.
        d_e = d[self._cg_elements]
        gp = self._cg_grad_phi
        gd_x = (gp[:, :, 0] * d_e).sum(1)
        gd_y = (gp[:, :, 1] * d_e).sum(1)
        lap_contrib = self._areas_col * (
            gp[:, :, 0] * gd_x.unsqueeze(1) +
            gp[:, :, 1] * gd_y.unsqueeze(1))  # (E, 3)
        lap_node = torch.zeros(self._cg_n_nodes, dtype=self._cg_dtype,
                               device=self._cg_device)
        lap_node.scatter_add_(0, self._elem_flat, lap_contrib.flatten())

        # 2) Project H_elem -> H_nodal via lumped (area/3) averaging.
        # Build a lumped nodal mass on the CG device (one-time cache).
        if not hasattr(self, '_cg_M_lump') or self._cg_M_lump is None:
            M_lump = torch.zeros(self._cg_n_nodes, dtype=self._cg_dtype,
                                 device=self._cg_device)
            area_third = self._cg_areas_third.unsqueeze(1).expand(-1, 3).flatten()
            M_lump.scatter_add_(0, self._elem_flat, area_third)
            self._cg_M_lump = M_lump

        # Nodal H via area-weighted assembly: H_a = (sum_e A_e/3 * H_e) / M_a_lump
        H_node_num = torch.zeros(self._cg_n_nodes, dtype=self._cg_dtype,
                                 device=self._cg_device)
        weighted = (self._cg_areas_third * H).unsqueeze(1).expand(-1, 3).flatten()
        H_node_num.scatter_add_(0, self._elem_flat, weighted)
        H_node = H_node_num / self._cg_M_lump

        # 3) Assemble lumped residual r_a:
        #    r_a = Gc*l0 * lap_node[a] + M_a * [ (Gc/l0 + 2 H_a) d_a - 2 H_a ]
        Gc_l0 = self._Gc_l0
        Gc_over_l0 = self._Gc_over_l0
        react = self._cg_M_lump * ((Gc_over_l0 + 2.0 * H_node) * d
                                    - 2.0 * H_node)
        r = Gc_l0 * lap_node + react

        # 4) Forward-Euler update with mass projection:
        #    d_new = d - dt * M_mob * r / M_a_lump
        M_mob = float(self._mobility) if self._mobility is not None else 1.0
        d_new = d - dt * M_mob * (r / self._cg_M_lump)

        # 5) Irreversibility (max with d_prev) and box clamp [0, 1].
        d_new = torch.clamp(torch.maximum(d_new, d), 0.0, 1.0)

        self.last_iter = 1
        return d_new.to(device=orig_device, dtype=orig_dtype)

    def allencahn_dt_max(self, H_max: float) -> float:
        """Return CFL-style stability bound for the Allen-Cahn explicit step:

            dt_max = 1 / ( M_mob · (Gc/l0 + 2 H_max) ).
        """
        M_mob = float(self._mobility) if self._mobility is not None else 1.0
        denom = M_mob * (self._Gc_over_l0 + 2.0 * float(H_max))
        if denom <= 0.0:
            return float('inf')
        return 1.0 / denom

    # ------------------------------------------------------------------ #
    # Direct solver: assemble + spsolve (PhaseFieldX-equivalent accuracy)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _solve_direct(self, H_input: torch.Tensor,
                      d_prev: torch.Tensor,
                      pf_dirichlet_mask=None,
                      pf_dirichlet_values=None) -> torch.Tensor:
        """Assemble full damage stiffness matrix and solve via direct LU.

        Uses cupy/cuSPARSE on CUDA, scipy on CPU. Matches PhaseFieldX's
        MUMPS accuracy. Then clamp d to [d_prev, 1].

        When ``self._nodal_H`` is True, ``H_input`` is (N,) nodal values
        and the triple-product mass matrix is used.
        """
        if self._viscous_coefficient() > 0.0:
            raise NotImplementedError(
                "damage_viscosity is currently implemented for CG/projected-CG "
                "damage solves only; direct assembled damage solves would need "
                "the same eta/dt mass term added to the sparse matrix and RHS.")
        import numpy as np
        from .multigrid import _assemble_sparse_cpu, _assemble_sparse_nodal_H
        from .mechanics_solver import _spsolve_auto

        orig_device = d_prev.device
        orig_dtype = d_prev.dtype

        # Transfer to CPU float64 for assembly
        H_cpu = H_input.detach().cpu().to(torch.float64)
        d_prev_cpu = d_prev.detach().cpu().to(torch.float64)

        # Handle gamma correction (per-element Gc/l0 and Gc*l0)
        if self._Gc_over_l0_e is not None:
            Gc_l0_ratio = self._Gc_over_l0_e.detach().cpu().to(torch.float64)
        else:
            Gc_l0_ratio = self._Gc_over_l0

        if hasattr(self, '_Gc_l0_e') and self._Gc_l0_e is not None:
            Gc_l0_coeff = self._Gc_l0_e.detach().cpu().to(torch.float64)
        else:
            Gc_l0_coeff = self._Gc_l0

        areas = self.fem.mesh.areas.detach().cpu().to(torch.float64)
        elements = self.fem.mesh.elements.cpu()
        n_nodes = self.fem.mesh.n_nodes

        # Build element Laplacian matrices: K_e = area * (gp_x^T gp_x + gp_y^T gp_y)
        gp = self.fem.mesh.grad_phi.detach().cpu().to(torch.float64)  # (E, 3, 2)
        areas_col = areas.unsqueeze(1).unsqueeze(2)  # (E, 1, 1)
        gp_x = gp[:, :, 0]  # (E, 3)
        gp_y = gp[:, :, 1]  # (E, 3)
        K_local_cpu = areas_col * (gp_x.unsqueeze(2) * gp_x.unsqueeze(1) +
                                   gp_y.unsqueeze(2) * gp_y.unsqueeze(1))  # (E, 3, 3)

        rows_list, cols_list = [], []
        for i in range(3):
            for j in range(3):
                rows_list.append(elements[:, i].numpy())
                cols_list.append(elements[:, j].numpy())
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)

        elem_flat = elements.flatten()

        if self._nodal_H:
            # ---------- Nodal-H path: triple-product mass ----------
            A_csr = _assemble_sparse_nodal_H(
                K_local_cpu, H_cpu, elements, areas,
                Gc_l0_coeff, Gc_l0_ratio, rows, cols, n_nodes)

            # RHS: b_a = A/6 * (H_a + S_H)
            H_e = H_cpu[elements]                              # (E, 3)
            S_H = H_e.sum(dim=1)                               # (E,)
            rhs_contrib = (areas / 6.0).unsqueeze(1) * (
                H_e + S_H.unsqueeze(1))                        # (E, 3)
            b = torch.zeros(n_nodes, dtype=torch.float64)
            b.scatter_add_(0, elem_flat, rhs_contrib.flatten())
        else:
            # ---------- Element-H path (original) ----------
            H_e = H_cpu
            reaction_coeff = (2.0 * H_e + Gc_l0_ratio) * areas / 12.0
            A_csr = _assemble_sparse_cpu(K_local_cpu, reaction_coeff,
                                         Gc_l0_coeff, rows, cols, n_nodes)

            # RHS: AT2 → 2H*area/3,  AT1 → (2H - 3Gc/(8l0))*area/3
            if hasattr(self, '_at1_source_e') and self._at1_source_e is not None:
                at1_src = self._at1_source_e.detach().cpu().to(torch.float64)
            else:
                at1_src = self._at1_source
            rhs_coeff = 2.0 * H_e - at1_src
            areas_third = areas / 3.0
            rhs_contrib = (rhs_coeff * areas_third).unsqueeze(1).expand(-1, 3).flatten()
            b = torch.zeros(n_nodes, dtype=torch.float64)
            b.scatter_add_(0, elem_flat, rhs_contrib)

        fixed, vals = self._prepare_pf_dirichlet(
            pf_dirichlet_mask, pf_dirichlet_values,
            device=torch.device('cpu'), dtype=torch.float64)
        b_np = b.numpy()
        if fixed is not None:
            fixed_np = fixed.numpy()
            vals_np = vals.numpy()
            fixed_idx = np.nonzero(fixed_np)[0]
            known = np.zeros(n_nodes, dtype=np.float64)
            known[fixed_idx] = vals_np[fixed_idx]
            b_np = b_np - A_csr.dot(known)
            A_lil = A_csr.tolil()
            A_lil[:, fixed_idx] = 0.0
            A_lil[fixed_idx, :] = 0.0
            A_lil[fixed_idx, fixed_idx] = 1.0
            A_csr = A_lil.tocsr()
            b_np[fixed_idx] = vals_np[fixed_idx]

        # Direct solve (cupy on CUDA, scipy on CPU)
        d_np = _spsolve_auto(A_csr, b_np, orig_device)
        d_new = torch.from_numpy(d_np).to(torch.float64)

        # Clamp: irreversibility (d >= d_prev) and physical bound (d <= 1)
        d_new = torch.clamp(d_new, min=0.0)
        d_new = torch.maximum(d_new, d_prev_cpu)
        d_new = torch.clamp(d_new, max=1.0)

        self.last_iter = 1
        return d_new.to(dtype=orig_dtype, device=orig_device)

    # ------------------------------------------------------------------ #
    # Post-clamp CG: unconstrained solve, then enforce bounds
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _solve_post_clamp(self, H_elem: torch.Tensor,
                          d_prev: torch.Tensor,
                          pf_dirichlet_mask=None,
                          pf_dirichlet_values=None) -> torch.Tensor:
        """Unconstrained preconditioned CG, then clamp d to [d_prev, 1].

        This is the original (simpler) approach. CG ignores bounds entirely
        and solves the linear system A d = b. After convergence the result
        is clamped to enforce irreversibility (d >= d_prev) and the physical
        upper bound (d <= 1).
        """
        d, d_prev_cg, reaction_coeff, b, M_inv, \
            orig_device, orig_dtype, need_transfer = \
            self._prepare_cg(H_elem, d_prev)
        fixed, vals = self._prepare_pf_dirichlet(
            pf_dirichlet_mask, pf_dirichlet_values,
            self._cg_device, self._cg_dtype)
        d, d_prev_cg = self._pin_pf_dirichlet_state(d, d_prev_cg, fixed, vals)

        check_every = 5 if self._use_multigrid else 50
        self.last_iter = self.max_iter

        # Initial residual
        r = b - self._Ax(d, reaction_coeff)
        if fixed is not None:
            r[fixed] = 0.0
            M_inv = M_inv.clone()
            M_inv[fixed] = 0.0
        r_norm_sq_0 = torch.dot(r, r).item()

        # Tolerance: relative to max(||b||, ||r0||). The old code used absolute
        # tol when ||b|| < 1, which is unreachable when ||r0|| << tol (tiny RHS
        # at low damage). This caused CG to run all 5000 iterations for no reason.
        b_norm = math.sqrt(torch.dot(b, b).item())
        r0_norm = math.sqrt(r_norm_sq_0)
        ref_norm = max(b_norm, r0_norm, 1e-30)
        tol_sq = (self.tol * ref_norm) ** 2

        # Apply preconditioner. AMG V-cycle can fail for reaction-dominated
        # systems (tiny Laplacian, large Gc/l0 mass term). Detect via rz quality
        # and fall back to Jacobi. Once Jacobi is selected, stick with it until
        # damage changes significantly (AMG rebuild resets the flag).
        use_mg = (self._use_multigrid and self._multigrid is not None
                  and not getattr(self, '_amg_fallback_active', False))
        if use_mg:
            z = self._multigrid.vcycle(r, reaction_coeff)
            rz = torch.dot(r, z)
            # Check quality: rz should be positive and z should reduce residual
            if rz.item() <= 0 or rz.item() < 1e-15 * r_norm_sq_0:
                use_mg = False
                self._amg_fail_count += 1
                self._amg_fallback_active = True
                self._amg_cooldown = min(
                    self._amg_retry_interval, 5000)
                torch.mul(r, M_inv, out=z)
                rz = torch.dot(r, z)
        else:
            z = torch.empty_like(r)
            torch.mul(r, M_inv, out=z)
            rz = torch.dot(r, z)

        p = z.clone()
        _amg_checked = False

        for i in range(self.max_iter):
            total_iter = i + 1

            # Early AMG failure detection: if residual hasn't decreased after
            # check_every iterations, AMG is broken — restart CG with Jacobi.
            if use_mg and not _amg_checked and total_iter == check_every:
                _amg_checked = True
                r_check = torch.dot(r, r).item()
                if r_check > 0.5 * r_norm_sq_0:
                    use_mg = False
                    self._amg_fail_count += 1
                    self._amg_fallback_active = True
                    self._amg_cooldown = min(
                        self._amg_retry_interval, 5000)
                    # Reset CG: recompute true residual from current d
                    r = b - self._Ax(d, reaction_coeff)
                    if fixed is not None:
                        r[fixed] = 0.0
                    r_norm_sq_0 = torch.dot(r, r).item()
                    z = torch.empty_like(r)
                    torch.mul(r, M_inv, out=z)
                    rz = torch.dot(r, z)
                    p = z.clone()
                    continue

            # _Ax returns persistent buffer — consumed before next call
            Ap = self._Ax(p, reaction_coeff)
            pAp_val = torch.dot(p, Ap).item()
            if pAp_val <= 0:
                # Suppress when the residual is already at tolerance --
                # in that case p ~= 0 and pAp == 0 trivially; CG has
                # nothing to do. Warn only when it indicates a real
                # SPD-loss mid-solve.
                if torch.dot(r, r).item() > tol_sq:
                    import warnings
                    warnings.warn(f"CG encountered non-SPD system (pAp={pAp_val:.2e}) at iter {total_iter}. "
                                  "Possible numerical instability.", RuntimeWarning, stacklevel=2)
                self.last_iter = total_iter
                break

            alpha_val = rz.item() / (pAp_val + 1e-30)
            d.add_(p, alpha=alpha_val)
            r.add_(Ap, alpha=-alpha_val)

            # Periodic true-residual recompute to avoid drift
            if total_iter % check_every == 0:
                Ax_d = self._Ax(d, reaction_coeff)
                torch.sub(b, Ax_d, out=r)
                if fixed is not None:
                    r[fixed] = 0.0

            r_norm_sq = torch.dot(r, r).item()
            if r_norm_sq < tol_sq:
                self.last_iter = total_iter
                break
            if r_norm_sq > 1e12 * max(r_norm_sq_0, 1e-30):
                print(f"  [PhaseFieldDamageSolver post_clamp diverged at iter "
                      f"{total_iter}]", flush=True)
                self.last_iter = total_iter
                break

            if use_mg:
                z = self._multigrid.vcycle(r, reaction_coeff)
            else:
                torch.mul(r, M_inv, out=z)

            rz_new = torch.dot(r, z)
            beta_val = (rz_new / (rz + 1e-30)).item()
            p.mul_(beta_val).add_(z)
            rz = rz_new
        else:
            self.last_iter = self.max_iter

        # Enforce bounds post-hoc
        d = torch.clamp(torch.maximum(d, d_prev_cg), 0, 1)
        if fixed is not None:
            d = torch.where(fixed, vals, d)

        if need_transfer:
            return d.to(dtype=orig_dtype, device=orig_device)
        return d

    # ------------------------------------------------------------------ #
    # Projected CG: bounds enforced during the solve
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _solve_projected_cg(self, H_elem: torch.Tensor,
                            d_prev: torch.Tensor,
                            pf_dirichlet_mask=None,
                            pf_dirichlet_values=None) -> torch.Tensor:
        """Projected Preconditioned CG (PPCG) with box constraints.

        Enforces d in [d_prev, 1] *during* the CG solve, not post-hoc.
        This avoids wasting CG iterations on nodes pinned at their bound.

        The algorithm maintains an active set of nodes at their bound:
          - Lower-active: d_i = lb_i and residual pushes below bound (r_i < 0)
          - Upper-active: d_i = ub_i and residual pushes above bound (r_i > 0)
        Residual and search direction are zeroed on active nodes, so CG only
        operates on free DOFs. When the active set changes (a node hits a bound
        or a bound constraint becomes inactive), CG restarts.
        """
        d, d_prev_cg, reaction_coeff, b, M_inv, \
            orig_device, orig_dtype, need_transfer = \
            self._prepare_cg(H_elem, d_prev)
        fixed, vals = self._prepare_pf_dirichlet(
            pf_dirichlet_mask, pf_dirichlet_values,
            self._cg_device, self._cg_dtype)
        d, d_prev_cg = self._pin_pf_dirichlet_state(d, d_prev_cg, fixed, vals)

        # Box constraints: lb = d_prev (irreversibility), ub = 1.0
        lb = d_prev_cg
        ub_val = 1.0

        # Tolerance for bound comparisons: after clamp + arithmetic,
        # d can differ from lb/ub by O(eps_mach). Using exact equality
        # misses nearly-bound nodes, leaving their residuals unzeroed.
        bound_atol = 1e-14

        # Check active set frequently — projected CG needs responsive
        # active set updates. With multigrid: every 5 iters; without: every 10.
        # (Old value of 50 caused active set lag and wasted iterations.)
        check_every = 5 if self._use_multigrid else 10
        self.last_iter = self.max_iter

        # d starts at d_prev (feasible). Compute initial residual.
        r = b - self._Ax(d, reaction_coeff)
        if fixed is not None:
            r[fixed] = 0.0
            M_inv = M_inv.clone()
            M_inv[fixed] = 0.0
        r_norm_sq_0 = torch.dot(r, r).item()

        # Tolerance: relative to max(||b||, ||r0||).
        b_norm = math.sqrt(torch.dot(b, b).item())
        r0_norm = math.sqrt(r_norm_sq_0)
        ref_norm = max(b_norm, r0_norm, 1e-30)
        tol_sq = (self.tol * ref_norm) ** 2

        # Identify initial active set:
        #   lower-active: d_i ~= lb_i AND r_i < 0 (gradient wants to decrease)
        #   upper-active: d_i ~= ub AND r_i > 0 (gradient wants to increase)
        active = projected_damage_active_mask(
            d, lb, r, upper_bound=ub_val, bound_atol=bound_atol,
            fixed=fixed)

        # Zero residual on active nodes
        r_free = zero_active_entries(r, active)

        # Apply preconditioner, zero on active.
        # Use AMG if available and not in cooldown, else Jacobi.
        use_mg = (self._use_multigrid and self._multigrid is not None
                  and not self._amg_fallback_active)
        if use_mg:
            z = self._multigrid.vcycle(r_free, reaction_coeff)
            rz_check = torch.dot(r_free, z)
            if rz_check.item() <= 0:
                use_mg = False
                self._amg_fail_count += 1
                self._clear_amg_hierarchy(
                    "AMG V-cycle produced a non-positive PCG direction",
                    activate_fallback=True)
                z = r_free * M_inv
        else:
            z = r_free * M_inv
        z[active] = 0.0

        p = z.clone()
        rz = torch.dot(r_free, z)

        total_iter = 0
        for i in range(self.max_iter):
            total_iter = i + 1

            Ap = self._Ax(p, reaction_coeff).clone()  # clone: _Ax returns aliased buffer
            pAp = torch.dot(p, Ap)
            if pAp.item() <= 0:
                # Degenerate direction (shouldn't happen for SPD, but guard).
                # In deterministic mode, do not break (would yield iter-count
                # nondeterminism between forward and recomputation); instead
                # treat as a tiny positive value so the step is small.
                if not _CG_DETERMINISTIC:
                    # Suppress when residual already at tolerance -- p ~= 0
                    # implies pAp == 0 trivially; CG has nothing to do. Warn
                    # only when it indicates real SPD-loss mid-solve.
                    if torch.dot(r, r).item() > tol_sq:
                        import warnings
                        warnings.warn(f"CG encountered non-SPD system (pAp={pAp.item():.2e}) at iter {total_iter}. "
                                      "Possible numerical instability.", RuntimeWarning, stacklevel=2)
                    self.last_iter = total_iter
                    break
                # Deterministic fallback: clamp pAp away from zero
                pAp = pAp.clamp(min=1e-30)

            alpha_cg = rz / (pAp + 1e-30)
            alpha_cg_val = alpha_cg.item()

            # --- Step limiting to stay feasible ---
            # Find the max alpha such that lb <= d + alpha*p <= ub for all i.
            # Only need to check free nodes where p != 0.
            alpha_max = alpha_cg_val  # default: full CG step

            # Nodes where p > 0: could hit upper bound
            # alpha_i = (ub - d_i) / p_i, take min
            pos_mask = p > 0
            if pos_mask.any():
                ratios_up = (ub_val - d[pos_mask]) / p[pos_mask]
                alpha_up = ratios_up.min().item()
                if alpha_up < alpha_max:
                    alpha_max = alpha_up

            # Nodes where p < 0: could hit lower bound
            # alpha_i = (lb_i - d_i) / p_i (note: p_i < 0, lb-d <= 0)
            neg_mask = p < 0
            if neg_mask.any():
                ratios_dn = (lb[neg_mask] - d[neg_mask]) / p[neg_mask]
                alpha_dn = ratios_dn.min().item()
                if alpha_dn < alpha_max:
                    alpha_max = alpha_dn

            # If step is limited, we hit a new bound — take the capped step
            # and restart CG with updated active set.
            # alpha_max == 0 means a free node is already at a bound with p
            # pointing outward; take zero step and let active-set rebuild
            # catch it on the restart.
            if alpha_max < alpha_cg_val and alpha_max >= 0:
                d.add_(p, alpha=alpha_max)
                # Snap nodes that are numerically at bounds
                d = torch.clamp(torch.maximum(d, lb), max=ub_val)

                # Recompute full residual (CG restart after active set change)
                r = b - self._Ax(d, reaction_coeff)
                if fixed is not None:
                    r[fixed] = 0.0

                # Rebuild active set
                active = projected_damage_active_mask(
                    d, lb, r, upper_bound=ub_val, bound_atol=bound_atol,
                    fixed=fixed)
                r_free = zero_active_entries(r, active)

                # Check convergence on free residual
                r_free_norm_sq = torch.dot(r_free, r_free).item()
                if r_free_norm_sq < tol_sq and not _CG_DETERMINISTIC:
                    self.last_iter = total_iter
                    break
                if r_free_norm_sq > 1e12 * max(r_norm_sq_0, 1e-30):
                    print(f"  [PhaseFieldDamageSolver diverged at iter {total_iter}]",
                          flush=True)
                    self.last_iter = total_iter
                    break

                # Restart: new preconditioned residual and search direction
                if use_mg:
                    z = self._multigrid.vcycle(r_free, reaction_coeff)
                else:
                    z = r_free * M_inv
                z[active] = 0.0
                p = z.clone()
                rz = torch.dot(r_free, z)
                continue

            # --- Full CG step (no bound hit) ---
            d.add_(p, alpha=alpha_cg_val)
            r.add_(Ap, alpha=-alpha_cg_val)

            # Periodically try to release nodes from active set.
            # A node should be released if the constraint force now points
            # inward (i.e., the unconstrained solution would be feasible).
            if (total_iter) % check_every == 0:
                # Recompute true residual to avoid drift
                r = b - self._Ax(d, reaction_coeff)
                if fixed is not None:
                    r[fixed] = 0.0

                # Release: lower-active nodes where r >= 0, upper-active where r <= 0
                old_active = active.clone()
                active = projected_damage_active_mask(
                    d, lb, r, upper_bound=ub_val, bound_atol=bound_atol,
                    fixed=fixed)
                active_changed = (active != old_active).any()

                r_free = zero_active_entries(r, active)

                r_free_norm_sq = torch.dot(r_free, r_free).item()
                if r_free_norm_sq < tol_sq and not _CG_DETERMINISTIC:
                    self.last_iter = total_iter
                    break
                if r_free_norm_sq > 1e12 * max(r_norm_sq_0, 1e-30):
                    print(f"  [PhaseFieldDamageSolver diverged at iter {total_iter}]",
                          flush=True)
                    self.last_iter = total_iter
                    break

                if active_changed:
                    # Active set changed — restart CG
                    if use_mg:
                        z = self._multigrid.vcycle(r_free, reaction_coeff)
                    else:
                        z = r_free * M_inv
                    z[active] = 0.0
                    p = z.clone()
                    rz = torch.dot(r_free, z)
                    continue
            else:
                # Between checks, keep r_free consistent
                r_free = zero_active_entries(r, active)

            # Standard CG update on free DOFs
            if use_mg:
                z_new = self._multigrid.vcycle(r_free, reaction_coeff)
            else:
                z_new = r_free * M_inv
            z_new[active] = 0.0

            rz_new = torch.dot(r_free, z_new)
            beta_val = (rz_new / (rz + 1e-30)).item()
            p.mul_(beta_val).add_(z_new)
            p[active] = 0.0  # ensure search direction is zero on active set
            rz = rz_new

        else:
            # Loop completed without break — max_iter reached
            self.last_iter = self.max_iter

        # Final safety clamp (should be no-op if PPCG worked correctly,
        # but guards against floating-point drift)
        d = torch.clamp(torch.maximum(d, lb), max=ub_val)
        if fixed is not None:
            d = torch.where(fixed, vals, d)

        if need_transfer:
            return d.to(dtype=orig_dtype, device=orig_device)
        return d
