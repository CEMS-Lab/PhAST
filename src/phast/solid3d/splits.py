"""Degradation, stored energy and damage-driving energy splits (batched).

Every function works on engineering-Voigt strains ``eps`` ``(..., 6)`` in the order ``(xx, yy, zz, yz, xz, xy)`` and broadcasts its other arguments against the
leading dimensions ``...``: stiffnesses are ``(..., 6, 6)``, tangents ``(..., 3)`` and scalar coefficients ``(...)``. Precision is the caller's (float64 in the solver).

The stress is always ``g(d) C eps`` with the degradation ``g``; the damage is driven by one of

* ``full``: ``psi0 = 0.5 eps.C.eps``;
* ``spectral`` (isotropic phases): ``lam/2 <tr eps>+^2 + mu sum_a <eps_a>+^2`` (Miehe, principal strains).

With ``SolverConfig.energy_split_band = 'interface'`` a smeared-laminate interface uses the directional interface split of :func:`interface_energies`: the interface
traction energy ``psi_plus`` drives its damage (``band_stress = 'hybrid'`` keeps the hybrid stress ``g C eps``).

Crack closure. The hybrid stress degrades compression as well as tension, so a fully damaged element has almost no compressive stiffness. With the closure on, the
degraded energy of a damaged element adds back ``(1 - g)`` of a fixed-axis compressive energy ``psi_c`` (:func:`closure_energy`): the cell axes for the isotropic phases
(:func:`closure_isotropic_energy`) and the interface frame for a smeared-laminate interface (:func:`closure_band_energy`).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from phast.core.voigt import (
    engineering_strain_to_tensor,
    engineering_stress_to_tensor,
    tensor_to_engineering_strain,
)
from phast.solid3d.materials import frame_from_tangent


def _matvec(C: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Broadcast ``C (..., m, n) @ v (..., n)``."""

    return torch.matmul(C, v.unsqueeze(-1)).squeeze(-1)


def _as_like(value: Any, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _check_voigt(name: str, value: torch.Tensor) -> None:
    if value.shape[-1] != 6:
        raise ValueError(f"{name} must end in 6 entries, got {tuple(value.shape)}")


# ---------------------------------------------------------------------------
# degradation and stored energy
# ---------------------------------------------------------------------------


def degradation(d: torch.Tensor, kappa: float) -> torch.Tensor:
    """Hybrid AT2 degradation ``g = (1-kappa)(1-d)^2 + kappa``."""

    return (1.0 - kappa) * (1.0 - d).square() + kappa


def degradation_derivative(d: torch.Tensor, kappa: float) -> torch.Tensor:
    """Derivative ``g' = -2(1-kappa)(1-d)``."""

    return -2.0 * (1.0 - kappa) * (1.0 - d)


def full_driving_energy(eps: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Undegraded stored energy density ``psi0 = 0.5 eps.C.eps``."""

    _check_voigt("eps", eps)
    return 0.5 * (eps * _matvec(C, eps)).sum(dim=-1)


def stored_energy_density(eps: torch.Tensor, C: torch.Tensor, g: Any) -> torch.Tensor:
    """Degraded stored energy density ``g * 0.5 eps.C.eps``."""

    return _as_like(g, eps) * full_driving_energy(eps, C)


def degraded_stress(eps: torch.Tensor, C: torch.Tensor, g: Any) -> torch.Tensor:
    """Hybrid AT2 stress ``g C eps`` (engineering Voigt)."""

    _check_voigt("eps", eps)
    return _as_like(g, eps).unsqueeze(-1) * _matvec(C, eps)


# ---------------------------------------------------------------------------
# isotropic spectral split
# ---------------------------------------------------------------------------


def symmetric_eigvalsh_3x3(A: torch.Tensor) -> torch.Tensor:
    """Ascending eigenvalues of symmetric ``(..., 3, 3)`` matrices in closed form.

    Trigonometric solution of the characteristic cubic (Smith 1961), batched
    over the leading dimensions.  Exact for repeated eigenvalues; near a
    repeated pair the relative eigenvalue error is about ``sqrt(eps)``, which
    is immaterial for the tensile-energy split.  Used on CUDA, where the
    batched cuSOLVER ``syev`` path rejects large batches of 3x3 matrices.
    """

    if A.shape[-2:] != (3, 3):
        raise ValueError(f"A must have trailing shape (3, 3), got {tuple(A.shape)}")
    a11, a22, a33 = A[..., 0, 0], A[..., 1, 1], A[..., 2, 2]
    a12, a13, a23 = A[..., 0, 1], A[..., 0, 2], A[..., 1, 2]
    q = (a11 + a22 + a33) / 3.0
    b11, b22, b33 = a11 - q, a22 - q, a33 - q
    p1 = a12.square() + a13.square() + a23.square()
    p = torch.sqrt((b11.square() + b22.square() + b33.square() + 2.0 * p1) / 6.0)
    det_b = (
        b11 * (b22 * b33 - a23.square())
        - a12 * (a12 * b33 - a23 * a13)
        + a13 * (a12 * a23 - b22 * a13)
    )
    tiny = torch.finfo(A.dtype).tiny
    isotropic = p <= tiny
    p_safe = torch.where(isotropic, torch.ones_like(p), p)
    r = (det_b / (2.0 * p_safe.pow(3))).clamp(-1.0, 1.0)
    phi = torch.acos(r) / 3.0
    largest = q + 2.0 * p * torch.cos(phi)
    smallest = q + 2.0 * p * torch.cos(phi + 2.0 * math.pi / 3.0)
    middle = 3.0 * q - largest - smallest
    out = torch.stack((smallest, middle, largest), dim=-1)
    return torch.where(isotropic.unsqueeze(-1), q.unsqueeze(-1).expand_as(out), out)


def principal_values(T: torch.Tensor, *, method: str | None = None) -> torch.Tensor:
    """Ascending eigenvalues of symmetric ``(..., 3, 3)`` tensors.

    ``method`` is ``'eigvalsh'`` (LAPACK/cuSOLVER), ``'analytic'``
    (:func:`symmetric_eigvalsh_3x3`) or ``None``.  ``None`` uses ``'analytic'``
    on CUDA and ``'eigvalsh'`` elsewhere, unless the environment variable
    ``PHAST_SOLID3D_EIG`` names a method.
    """

    if method is None:
        method = os.environ.get("PHAST_SOLID3D_EIG") or ("analytic" if T.is_cuda else "eigvalsh")
    if method == "analytic":
        return symmetric_eigvalsh_3x3(T)
    if method == "eigvalsh":
        return torch.linalg.eigvalsh(T)
    raise ValueError(f"unknown eigenvalue method {method!r}; use 'analytic' or 'eigvalsh'")


def spectral_tensile_energy(eps: torch.Tensor, lam: Any, mu: Any, *, method: str | None = None) -> torch.Tensor:
    """Miehe tensile energy ``lam/2 <tr eps>+^2 + mu sum_a <eps_a>+^2``.

    The engineering shears are halved to form the strain tensor, whose
    principal strains ``eps_a`` come from :func:`principal_values`
    (``eigvalsh`` on CPU, the closed form on CUDA).  ``lam`` and ``mu``
    broadcast against the leading dimensions (per-element values).
    """

    _check_voigt("eps", eps)
    principal = principal_values(engineering_strain_to_tensor(eps), method=method)
    trace_plus = eps[..., :3].sum(dim=-1).clamp(min=0.0)
    return (
        0.5 * _as_like(lam, eps) * trace_plus.square()
        + _as_like(mu, eps) * principal.clamp(min=0.0).square().sum(dim=-1)
    )


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.unsqueeze(-1) * b.unsqueeze(-2)


def complementary_coupling_term(sigma_eff: torch.Tensor, t: Any, E_L: Any, nu_LT: Any) -> torch.Tensor:
    """Axial-transverse coupling ``-(nu_LT/E_L) s11 (tr sigma - s11)``."""

    _check_voigt("sigma_eff", sigma_eff)
    e1 = frame_from_tangent(_as_like(t, sigma_eff))[..., :, 0]
    S = engineering_stress_to_tensor(sigma_eff)
    s11 = (e1 * _matvec(S, e1)).sum(dim=-1)
    trace = sigma_eff[..., :3].sum(dim=-1)
    return -(_as_like(nu_LT, sigma_eff) / _as_like(E_L, sigma_eff)) * s11 * (trace - s11)


# ---------------------------------------------------------------------------
# directional interface driver of the smeared band
# ---------------------------------------------------------------------------

INTERFACE_DIAGONAL_TOLERANCE = 1.0e-10
"""Largest accepted ``|C_nn[i, j]| / max_k C_nn[k, k]`` off the diagonal of the normal block."""

INTERFACE_COMPONENTS: tuple[str, ...] = ("11", "13", "12")
"""Interface traction components in the local frame (local Voigt indices 0, 4 and 5)."""


def interface_traction_vectors(t: Any) -> torch.Tensor:
    """Rows ``m_k`` ``(..., 3, 6)`` with ``t_k = m_k . sigma`` (engineering Voigt stress).

    The local frame is :func:`~phast.solid3d.materials.frame_from_tangent`
    of ``t``: ``n = e1`` is the interface normal and ``e2, e3`` span the
    interface plane.  ``m_k`` is the engineering-strain form
    ``[G11, G22, G33, 2 G23, 2 G13, 2 G12]`` of ``G = sym(a n^T)`` with
    ``a = n, e3, e2`` for the traction components ``sigma_11, sigma_13,
    sigma_12`` (:data:`INTERFACE_COMPONENTS`).  The same rows map the local
    normal-block strains ``(eps_11, gamma_13, gamma_12)`` to the global
    engineering strain, so ``C_nn = M C M^T`` with ``M`` the stacked rows.
    """

    frame = frame_from_tangent(t)
    n, e2, e3 = frame[..., :, 0], frame[..., :, 1], frame[..., :, 2]
    rows = []
    for a in (n, e3, e2):
        G = 0.5 * (_outer(a, n) + _outer(n, a))
        rows.append(tensor_to_engineering_strain(G))
    return torch.stack(rows, dim=-2)


def interface_moduli(C: torch.Tensor, M: torch.Tensor, *, tolerance: float = INTERFACE_DIAGONAL_TOLERANCE) -> torch.Tensor:
    """Diagonal ``(..., 3)`` of the local normal block ``C_nn = M C M^T``.

    Raises ``ValueError`` when ``C_nn`` is not diagonal within ``tolerance``
    (relative to its largest diagonal entry) or not positive: the
    tension-compression split of :func:`interface_energies` needs the normal
    component decoupled from the two interface shears.  A laminate of
    isotropic layers stacked along the normal (the interface) and an isotropic
    phase both have a diagonal ``C_nn``: per isotropic layer it is
    ``diag(lam + 2 mu, mu, mu)``, and the series average
    ``<C_nn^-1>^-1`` of diagonal blocks is diagonal.
    """

    C_nn = M @ C @ M.transpose(-1, -2)
    diagonal = torch.diagonal(C_nn, dim1=-2, dim2=-1)
    off = C_nn - torch.diag_embed(diagonal)
    if diagonal.numel():
        scale = diagonal.abs().amax(dim=-1)
        if not bool((diagonal > 0.0).all()):
            raise ValueError("the interface normal block C_nn must have a positive diagonal")
        defect = float((off.abs().amax(dim=(-2, -1)) / scale).max())
        if not defect <= tolerance:
            raise ValueError(
                f"the interface normal block C_nn is not diagonal in the local frame (relative "
                f"off-diagonal {defect:.3e} > {tolerance:.1e}); the interface driver needs the normal "
                "component decoupled from the interface shears"
            )
    return diagonal


@dataclass(frozen=True, eq=False)
class InterfaceEnergies:
    """Directional interface energies at every point (leading shape of the strain).

    ``traction`` ``(..., 3)`` holds ``(sigma_11, sigma_13, sigma_12)`` of the
    undamaged stress ``sigma_eff = C eps`` in the local frame;
    ``psi0 = psi_interface + psi_rest`` and
    ``psi_interface = psi_plus + psi_minus``.
    """

    psi0: torch.Tensor
    psi_interface: torch.Tensor
    psi_plus: torch.Tensor
    psi_minus: torch.Tensor
    psi_rest: torch.Tensor
    traction: torch.Tensor
    sigma_eff: torch.Tensor


def _opening(traction: torch.Tensor, opening: torch.Tensor | None) -> torch.Tensor:
    if opening is None:
        return traction[..., 0] > 0.0
    if opening.dtype != torch.bool:
        raise TypeError("opening must be a bool tensor")
    return opening


def interface_energies(
    eps: torch.Tensor,
    C: torch.Tensor,
    M: torch.Tensor,
    inv_a: torch.Tensor,
) -> InterfaceEnergies:
    """Directional energy split of the smeared band (Steinke and Kaliske 2019).

    With the interface normal ``n`` (axis 1 of the local frame) the interface
    traction ``t = (sigma_11, sigma_13, sigma_12)`` of ``sigma = C eps``
    is ``t_k = m_k . sigma`` (:func:`interface_traction_vectors`), and with
    ``a = diag(C_nn)`` (:func:`interface_moduli`, ``inv_a = 1/a``)

    * ``psi_interface = 1/2 t^T C_nn^-1 t = sum_k t_k^2 / (2 a_k)``;
    * ``psi_plus = <t_11>+^2/(2 a_11) + t_13^2/(2 a_13) + t_12^2/(2 a_12)``
      (normal opening and both interface shears);
    * ``psi_minus = <t_11>-^2/(2 a_11)`` (normal closure);
    * ``psi_rest = psi0 - psi_interface``.

    ``psi_rest`` is the energy of the in-plane strains with the interface
    tractions relaxed, and it is non-negative.  Proof: in the local frame
    split the engineering strain into the normal block
    ``n = (eps_11, gamma_13, gamma_12)`` and the in-plane block
    ``p = (eps_22, eps_33, gamma_23)``, with the symmetric positive definite
    ``C = [[C_nn, C_np], [C_pn, C_pp]]``.  Then ``t = C_nn n + C_np p``, so
    ``n = C_nn^-1 (t - C_np p)`` and

    ``2 psi0 = n.C_nn n + 2 n.C_np p + p.C_pp p
    = t.C_nn^-1 t + p.(C_pp - C_pn C_nn^-1 C_np) p``.

    The Schur complement ``S = C_pp - C_pn C_nn^-1 C_np`` of an SPD matrix
    is SPD, so ``psi_rest = 1/2 p.S p >= 0``; it depends on the in-plane
    strains only and is the minimum of ``psi0`` over the normal-block strains
    (``t = 0``).  Every strain state with zero interface traction therefore
    has ``psi_plus = psi_minus = 0`` and ``psi_rest = psi0``: stretching
    along the interface plane with the interface tractions relaxed does not
    drive the band.  The degraded energy is
    ``psi = g psi_plus + psi_minus + psi_rest = psi0 - (1 - g) psi_plus``,
    convex and continuously differentiable in ``eps`` (``<x>+^2`` is C1).

    Reference: Steinke, C. and Kaliske, M. (2019), A phase-field crack model
    based on directional stress decomposition, Comput. Mech. 63, 1019-1046,
    doi:10.1007/s00466-018-1635-0.
    """

    _check_voigt("eps", eps)
    sigma = _matvec(C, eps)
    traction = _matvec(M, sigma)
    inv = _as_like(inv_a, eps)
    psi0 = 0.5 * (eps * sigma).sum(dim=-1)
    squares = 0.5 * traction.square() * inv
    psi_interface = squares.sum(dim=-1)
    normal = traction[..., 0]
    psi_minus = 0.5 * normal.clamp(max=0.0).square() * inv[..., 0]
    psi_plus = psi_interface - psi_minus
    return InterfaceEnergies(
        psi0=psi0,
        psi_interface=psi_interface,
        psi_plus=psi_plus,
        psi_minus=psi_minus,
        psi_rest=psi0 - psi_interface,
        traction=traction,
        sigma_eff=sigma,
    )


def interface_driving_energy(eps: torch.Tensor, C: torch.Tensor, M: torch.Tensor, inv_a: torch.Tensor) -> torch.Tensor:
    """Band driving energy ``Y = psi_plus`` of :func:`interface_energies`."""

    return interface_energies(eps, C, M, inv_a).psi_plus


def interface_positive_stress(
    eps: torch.Tensor,
    C: torch.Tensor,
    M: torch.Tensor,
    inv_a: torch.Tensor,
    *,
    opening: torch.Tensor | None = None,
) -> torch.Tensor:
    """``d psi_plus / d eps = C q`` with ``q = sum_k s_k (t_k / a_k) m_k`` (engineering-conjugate).

    ``q`` is the engineering-strain form of ``G = d psi_plus / d sigma``;
    ``s_11 = [t_11 > 0]`` (or ``opening`` when given, which fixes the
    normal-opening pattern and makes the map linear in ``eps``) and
    ``s_13 = s_12 = 1``.
    """

    _check_voigt("eps", eps)
    sigma = _matvec(C, eps)
    traction = _matvec(M, sigma)
    open_ = _opening(traction, opening)
    weights = traction * _as_like(inv_a, eps)
    weights = torch.cat((torch.where(open_, weights[..., 0], torch.zeros_like(weights[..., 0])).unsqueeze(-1), weights[..., 1:]), dim=-1)
    q = (weights.unsqueeze(-1) * M).sum(dim=-2)
    return _matvec(C.transpose(-1, -2), q)


def interface_degraded_stress(
    eps: torch.Tensor,
    C: torch.Tensor,
    M: torch.Tensor,
    inv_a: torch.Tensor,
    g: Any,
    *,
    opening: torch.Tensor | None = None,
) -> torch.Tensor:
    """Band stress ``g d psi_plus/d eps + d(psi_rest + psi_minus)/d eps = C eps - (1 - g) C q``."""

    positive = interface_positive_stress(eps, C, M, inv_a, opening=opening)
    return _matvec(C, eps) - (1.0 - _as_like(g, eps)).unsqueeze(-1) * positive


def interface_degraded_energy(
    eps: torch.Tensor, C: torch.Tensor, M: torch.Tensor, inv_a: torch.Tensor, g: Any
) -> torch.Tensor:
    """Degraded band energy ``g psi_plus + psi_minus + psi_rest = psi0 - (1 - g) psi_plus``."""

    energies = interface_energies(eps, C, M, inv_a)
    return energies.psi0 - (1.0 - _as_like(g, eps)) * energies.psi_plus


# ---------------------------------------------------------------------------
# crack closure of fully damaged material
# ---------------------------------------------------------------------------
#
# Every closure energy of is a sum of terms ``k/2 <v . eps>-^2`` with
# a modulus ``k >= 0`` and a strain functional ``v`` (an engineering-Voigt row,
# ``v . eps`` a linear combination of strain components).  The rows ``(..., K,
# 6)`` and moduli ``(..., K)`` of a phase group define
#
# * the energy ``psi_c = sum_k k_k/2 <v_k . eps>-^2`` (:func:`closure_energy`);
# * its derivative ``sum_k k_k <v_k . eps>- v_k`` (:func:`closure_stress`);
# * for a fixed sign pattern ``s_k = [v_k . eps < 0]`` the tangent
#   ``sum_k s_k k_k v_k v_k^T`` (:func:`closure_stiffness`).
#
# ``<x>-^2`` is convex and continuously differentiable, so ``psi_c`` is convex,
# C1, piecewise quadratic and positively homogeneous of degree two (Euler:
# ``sigma_c . eps = 2 psi_c``).  The degraded energy of an isotropic or band
# element is ``W = g psi0 + (1 - g) psi_c`` with the stress
# ``g C eps + (1 - g) sigma_c``.

CLOSURE_MODULI_TOLERANCE = 1.0e-10
"""Relative tolerance of the set-up checks of the closure moduli."""

CLOSURE_STIFFENING_TOLERANCE = 1.0e-12
"""``lambda_min(C - D) >= -tol * max C_ii``: the no-stiffening check of ."""

ISOTROPIC_CLOSURE_BITS: tuple[str, ...] = ("trace", "xx", "yy", "zz")
"""Pattern bits of the isotropic closure: ``tr eps < 0`` and ``eps_ii < 0`` along the cell axes."""

BAND_CLOSURE_BITS: tuple[str, ...] = ("t11", "p2+p3", "p2", "p3")
"""Pattern bits of the laminate band closure: normal traction and in-plane normal strains."""


def closure_energy(eps: torch.Tensor, rows: torch.Tensor, moduli: torch.Tensor) -> torch.Tensor:
    """``psi_c = sum_k k_k/2 <v_k . eps>-^2`` with rows ``v`` ``(..., K, 6)`` and moduli ``k`` ``(..., K)``."""

    _check_voigt("eps", eps)
    values = torch.matmul(rows, eps.unsqueeze(-1)).squeeze(-1)
    return 0.5 * (_as_like(moduli, eps) * values.clamp(max=0.0).square()).sum(dim=-1)


def _closure_active(values: torch.Tensor, pattern: torch.Tensor | None) -> torch.Tensor:
    if pattern is None:
        return values < 0.0
    if pattern.dtype != torch.bool:
        raise TypeError("pattern must be a bool tensor")
    return pattern


def closure_stress(
    eps: torch.Tensor, rows: torch.Tensor, moduli: torch.Tensor, *, pattern: torch.Tensor | None = None
) -> torch.Tensor:
    """``d psi_c / d eps = sum_k s_k k_k (v_k . eps) v_k`` (engineering-conjugate, ``(..., 6)``).

    ``pattern`` (bool ``(..., K)``) fixes ``s_k``; without it ``s_k = [v_k . eps < 0]``,
    the derivative of :func:`closure_energy`.
    """

    _check_voigt("eps", eps)
    values = torch.matmul(rows, eps.unsqueeze(-1)).squeeze(-1)
    active = _closure_active(values, pattern)
    coefficient = torch.where(active, values, torch.zeros_like(values)) * _as_like(moduli, eps)
    return torch.matmul(coefficient.unsqueeze(-2), rows).squeeze(-2)


def closure_stiffness(
    rows: torch.Tensor, moduli: torch.Tensor, pattern: torch.Tensor | None = None
) -> torch.Tensor:
    """Pattern tangent ``D_s = sum_k s_k k_k v_k v_k^T`` ``(..., 6, 6)``; all bits active without ``pattern``.

    The set-up check of requires ``C - D`` with every bit active
    to be positive semidefinite (``lambda_min(C - D) >= -1e-12 max C_ii``),
    so that the closure never stiffens a phase beyond its intact stiffness.
    """

    weights = _as_like(moduli, rows)
    if pattern is not None:
        if pattern.dtype != torch.bool:
            raise TypeError("pattern must be a bool tensor")
        weights = weights * pattern.to(rows.dtype)
    return torch.matmul(rows.transpose(-1, -2) * weights.unsqueeze(-2), rows)


def no_stiffening_margin(C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """``lambda_min(C - D) / max C_ii`` ``(...)`` in Mandel form (the set-up check value)."""

    from phast.core.voigt import engineering_to_mandel_stiffness

    difference = engineering_to_mandel_stiffness((C - D).detach().cpu())
    difference = 0.5 * (difference + difference.transpose(-1, -2))
    smallest = torch.linalg.eigvalsh(difference)[..., 0]
    scale = torch.diagonal(C.detach().cpu(), dim1=-2, dim2=-1).amax(dim=-1)
    return smallest / scale


# isotropic phases: closure axes are the cell axes --------------------------


def closure_isotropic_rows(dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> torch.Tensor:
    """Rows ``(4, 6)``: ``1 = (1, 1, 1, 0, 0, 0)`` (the trace) and ``e_x, e_y, e_z``."""

    rows = torch.zeros((4, 6), dtype=dtype, device=device)
    rows[0, :3] = 1.0
    rows[1, 0] = rows[2, 1] = rows[3, 2] = 1.0
    return rows


def closure_isotropic_moduli(lam: Any, mu: Any) -> torch.Tensor:
    """Moduli ``(..., 4) = (lam, 2 mu, 2 mu, 2 mu)`` of :func:`closure_isotropic_rows`."""

    lam = torch.as_tensor(lam, dtype=torch.float64) if not isinstance(lam, torch.Tensor) else lam
    mu = _as_like(mu, lam)
    return torch.stack((lam, 2.0 * mu, 2.0 * mu, 2.0 * mu), dim=-1)


def closure_isotropic_energy(eps: torch.Tensor, lam: Any, mu: Any) -> torch.Tensor:
    """Isotropic closure energy ``psi_c = lam/2 <tr eps>-^2 + mu sum_i <eps_ii>-^2`` (cell axes, ).

    Bounds: ``0 <= psi_c <= psi0`` for ``lam >= 0``, because
    ``psi0 = lam/2 (tr eps)^2 + mu |eps|^2 >= lam/2 (tr eps)^2 + mu sum_i eps_ii^2``
    (the shear terms are dropped) and ``<x>-^2 <= x^2``.  For strains
    diagonal in the cell axes the principal strains are ``eps_ii``, so
    ``psi_c`` equals the Miehe compressive energy
    ``psi0 - spectral_tensile_energy``.  The degraded energy
    ``g psi0 + (1 - g) psi_c`` restores the intact stress
    ``C eps`` at any ``d`` for confined uniaxial-strain compression along a
    cell axis and for hydrostatic compression.
    """

    _check_voigt("eps", eps)
    lam = _as_like(lam, eps)
    mu = _as_like(mu, eps)
    trace = eps[..., :3].sum(dim=-1).clamp(max=0.0)
    return 0.5 * lam * trace.square() + mu * eps[..., :3].clamp(max=0.0).square().sum(dim=-1)


def closure_isotropic_stress(
    eps: torch.Tensor, lam: Any, mu: Any, *, pattern: torch.Tensor | None = None
) -> torch.Tensor:
    """``d psi_c / d eps = lam <tr eps>- 1 + 2 mu sum_i <eps_ii>- e_i`` of :func:`closure_isotropic_energy`.

    ``pattern`` (bool ``(..., 4)``, :data:`ISOTROPIC_CLOSURE_BITS`) fixes the
    bits ``[tr < 0], [eps_xx < 0], [eps_yy < 0], [eps_zz < 0]`` and makes
    the map linear; the pattern tangent is
    ``s0 lam 1 1^T + 2 mu sum_i s_i e_i e_i^T``.
    """

    _check_voigt("eps", eps)
    rows = closure_isotropic_rows(eps.dtype, eps.device)
    moduli = closure_isotropic_moduli(_as_like(lam, eps), _as_like(mu, eps))
    return closure_stress(eps, rows, moduli, pattern=pattern)


# frame helpers shared by the interface closure --------------------------------


def normal_strain_vector(t: Any) -> torch.Tensor:
    """``a = [t1^2, t2^2, t3^2, t2 t3, t1 t3, t1 t2]`` ``(..., 6)`` of unit tangents ``t``.

    ``a`` is both the engineering-Voigt contraction vector of a normal strain,
    ``eps_t = t.eps.t = a . eps`` (engineering shears), and the Voigt stress
    form of the tensor ``t (x) t``, so ``d eps_t / d eps = a``.
    """

    t = torch.as_tensor(t)
    if t.shape[-1] != 3:
        raise ValueError(f"t must end in 3 entries, got {tuple(t.shape)}")
    t1, t2, t3 = t[..., 0], t[..., 1], t[..., 2]
    return torch.stack((t1 * t1, t2 * t2, t3 * t3, t2 * t3, t1 * t3, t1 * t2), dim=-1)


def engineering_strain_of_dyad(n: torch.Tensor) -> torch.Tensor:
    """Engineering strain ``[n1^2, n2^2, n3^2, 2 n2 n3, 2 n1 n3, 2 n1 n2]`` of ``n (x) n``.

    Its contraction with a Voigt stress is ``n.sigma.n``; the contraction
    vector of the normal strain ``n.eps.n`` is :func:`normal_strain_vector`
    (shears not doubled).
    """

    n1, n2, n3 = n[..., 0], n[..., 1], n[..., 2]
    return torch.stack((n1 * n1, n2 * n2, n3 * n3, 2.0 * n2 * n3, 2.0 * n1 * n3, 2.0 * n1 * n2), dim=-1)


def local_normal_stiffness(C: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """``C'_ij = s(e_i) . C . s(e_j)`` ``(..., 3, 3)``: the normal block of ``C`` in the frame's axes.

    ``s(e)`` is :func:`engineering_strain_of_dyad`; ``C'_ij`` is the stress
    ``e_i.sigma.e_i`` caused by a unit strain ``e_j (x) e_j``.
    """

    S = torch.stack([engineering_strain_of_dyad(frame[..., :, k]) for k in range(3)], dim=-2)
    return S @ C @ S.transpose(-1, -2)


def determinant_3x3(F: torch.Tensor) -> torch.Tensor:
    """Determinants ``(...)`` of ``(..., 3, 3)`` matrices by the explicit cofactor formula.

    Used for the kinematic guard ``det(I + grad u)`` of on every
    device without a batched LU factorisation.
    """

    if F.shape[-2:] != (3, 3):
        raise ValueError(f"F must have trailing shape (3, 3), got {tuple(F.shape)}")
    a, b, c = F[..., 0, 0], F[..., 0, 1], F[..., 0, 2]
    d, e, f = F[..., 1, 0], F[..., 1, 1], F[..., 1, 2]
    g, h, i = F[..., 2, 0], F[..., 2, 1], F[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


# smeared-laminate band: closure axes are the band frame ---------------------


def closure_band_in_plane_rows(frame: torch.Tensor) -> torch.Tensor:
    """Engineering strains ``(..., 3, 6)`` of unit local in-plane strains ``(eps_22, eps_33, gamma_23)``."""

    e2, e3 = frame[..., :, 1], frame[..., :, 2]
    shear = tensor_to_engineering_strain(0.5 * (_outer(e2, e3) + _outer(e3, e2)))
    return torch.stack((engineering_strain_of_dyad(e2), engineering_strain_of_dyad(e3), shear), dim=-2)


def closure_band_moduli(
    C: torch.Tensor, frame: torch.Tensor, *, tolerance: float = CLOSURE_MODULI_TOLERANCE
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lam_S, mu_S) = (S23, (S22 - S23)/2)`` of the in-plane Schur block of the band.

    ``S = C_pp - C_pn C_nn^-1 C_np`` in the band frame (axis 1 the normal;
    ``n = (eps_11, gamma_13, gamma_12)``, ``p = (eps_22, eps_33, gamma_23)``)
    is the stiffness of the in-plane strains with the interface tractions
    relaxed (:func:`interface_energies`).  Raises ``ValueError`` unless it
    is two-dimensionally isotropic within ``tolerance``: ``S22 = S33``, no
    coupling of the in-plane normal strains with ``gamma_23``, and
    ``S23 >= 0``.
    """

    # set-up only: the batched 3x3 solves run on the CPU (CPU solves)
    device = C.device
    C = C.detach().cpu()
    frame = frame.detach().cpu()
    M = interface_traction_vectors(frame[..., :, 0])
    P = closure_band_in_plane_rows(frame)
    C_nn = M @ C @ M.transpose(-1, -2)
    C_np = M @ C @ P.transpose(-1, -2)
    C_pp = P @ C @ P.transpose(-1, -2)
    S = C_pp - C_np.transpose(-1, -2) @ torch.linalg.solve(C_nn, C_np)
    s22, s33, s23 = S[..., 0, 0], S[..., 1, 1], S[..., 0, 1]
    if s22.numel():
        scale = torch.maximum(s22.abs(), s33.abs())
        defect = float(((s22 - s33).abs() / scale).max())
        coupling = float((torch.maximum(S[..., 0, 2].abs(), S[..., 1, 2].abs()) / scale).max())
        if not (defect <= tolerance and coupling <= tolerance):
            raise ValueError(
                "crack_closure 'axes' needs a band whose in-plane Schur block is two-dimensionally "
                f"isotropic (relative S22 - S33 {defect:.3e}, normal-shear coupling {coupling:.3e}, "
                f"tolerance {tolerance:.1e})"
            )
        if bool((s23 < 0.0).any()):
            raise ValueError("crack_closure 'axes' needs a band Schur block with S23 >= 0 (lam_S >= 0)")
    return s23.to(device), (0.5 * (s22 - s23)).to(device)


def closure_band_rows(C: torch.Tensor, m1: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Rows ``(..., 4, 6) = (C m1, q2 + q3, q2, q3)`` of :data:`BAND_CLOSURE_BITS`.

    ``m1`` is the normal traction row of :func:`interface_traction_vectors`
    (``t11 = m1 . C eps = (C m1) . eps``) and ``q_k = a(e_k)``
    (:func:`normal_strain_vector`), so ``p_k = q_k . eps`` is the in-plane
    normal strain along ``e_k``.
    """

    q2 = normal_strain_vector(frame[..., :, 1])
    q3 = normal_strain_vector(frame[..., :, 2])
    Cm = _matvec(C, m1)
    return torch.stack((Cm, q2 + q3, q2, q3), dim=-2)


def closure_band_row_moduli(inv_a11: Any, lam_S: torch.Tensor, mu_S: torch.Tensor) -> torch.Tensor:
    """Moduli ``(..., 4) = (1/a11, lam_S, 2 mu_S, 2 mu_S)``."""

    inv = _as_like(inv_a11, lam_S)
    return torch.stack((inv, lam_S, 2.0 * mu_S, 2.0 * mu_S), dim=-1)


def closure_band_energy(
    eps: torch.Tensor, rows: torch.Tensor, moduli: torch.Tensor
) -> torch.Tensor:
    """Band closure energy ``<t11>-^2/(2 a11) + lam_S/2 <p2 + p3>-^2 + mu_S (<p2>-^2 + <p3>-^2)``.

    ``rows`` and ``moduli`` come from :func:`closure_band_rows` and
    :func:`closure_band_row_moduli`.  ``psi_c <= psi0``: with the traction
    ``t`` and the in-plane strains ``p`` of the band frame,
    ``psi0 = sum_k t_k^2/(2 a_k) + p.S.p/2`` (proof in
    :func:`interface_energies`), ``<t11>-^2/(2 a11) <= t11^2/(2 a11)``, and
    for a two-dimensionally isotropic ``S`` with ``lam_S >= 0`` (checked by
    :func:`closure_band_moduli`)
    ``p.S.p/2 >= lam_S/2 (p2 + p3)^2 + mu_S (p2^2 + p3^2)``.  The closure
    uses the laminate stiffness, never the isotropic-equivalent ``lam, mu``
    of the band's damage driver.
    """

    return closure_energy(eps, rows, moduli)


def closure_band_stress(
    eps: torch.Tensor, rows: torch.Tensor, moduli: torch.Tensor, *, pattern: torch.Tensor | None = None
) -> torch.Tensor:
    """``d psi_c / d eps = (<t11>-/a11) C m1 + lam_S <p2 + p3>- (q2 + q3) + 2 mu_S (<p2>- q2 + <p3>- q3)``."""

    return closure_stress(eps, rows, moduli, pattern=pattern)


__all__ = [
    "BAND_CLOSURE_BITS",
    "CLOSURE_MODULI_TOLERANCE",
    "CLOSURE_STIFFENING_TOLERANCE",
    "INTERFACE_COMPONENTS",
    "INTERFACE_DIAGONAL_TOLERANCE",
    "ISOTROPIC_CLOSURE_BITS",
    "InterfaceEnergies",
    "closure_band_energy",
    "closure_band_in_plane_rows",
    "closure_band_moduli",
    "closure_band_row_moduli",
    "closure_band_rows",
    "closure_band_stress",
    "closure_energy",
    "closure_isotropic_energy",
    "closure_isotropic_moduli",
    "closure_isotropic_rows",
    "closure_isotropic_stress",
    "closure_stiffness",
    "closure_stress",
    "complementary_coupling_term",
    "degradation",
    "degradation_derivative",
    "degraded_stress",
    "determinant_3x3",
    "engineering_strain_of_dyad",
    "full_driving_energy",
    "interface_degraded_energy",
    "interface_degraded_stress",
    "interface_driving_energy",
    "interface_energies",
    "interface_moduli",
    "interface_positive_stress",
    "interface_traction_vectors",
    "local_normal_stiffness",
    "no_stiffening_margin",
    "principal_values",
    "spectral_tensile_energy",
    "stored_energy_density",
    "symmetric_eigvalsh_3x3",
]
