"""Jacobi-preconditioned conjugate gradients for periodic voxel cells.

Two solvers are provided:

* :func:`pcg` solves ``A x = b`` for a matrix-free symmetric positive
  definite action with a positive Jacobi diagonal;
* :func:`solve_constrained` solves the same system with Dirichlet values on a
  boolean mask by projected CG on the free set: with ``x_c`` the prescribed
  values (zero on the free set) it solves ``A_ff x_f = b_f``,
  ``b_f = mask_f (b - A x_c)``.

Stopping rule (no floor): ``||r|| <= max(rtol ||b||, atol)``, where ``b`` is
the right-hand side of the system actually iterated on (``b_f`` for the
constrained solve) and ``r`` the recursively updated residual.  A zero
right-hand side returns the exact solution after 0 iterations.

Every iteration keeps a few device-side flags (converged, non-positive
curvature, non-finite) instead of synchronising; they are read only every
``check_every`` iterations and at ``maxiter``.  Once the residual meets the
tolerance the iterate is frozen, so the reported iteration count and
residual are those of the first converged iterate even between checks.
Non-positive curvature ``p.A p <= 0`` on a non-converged iterate, a
non-positive diagonal entry on the free set (curvature along a unit vector)
and non-finite values raise :class:`PCGBreakdownError`.  Running out of
iterations returns ``converged=False``; the caller decides what to do.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch


Action = Callable[[torch.Tensor], torch.Tensor]
"""Matrix-free linear action ``v -> A v`` on flat float64 vectors."""


class PCGBreakdownError(RuntimeError):
    """Raised on non-positive curvature, a non-positive diagonal or non-finite values."""


class PCGConvergenceError(RuntimeError):
    """Raised by callers that require convergence when PCG runs out of iterations."""

    def __init__(self, message: str, info: "PCGInfo") -> None:
        super().__init__(message)
        self.info = info


@dataclass(frozen=True)
class PCGInfo:
    """Outcome of one PCG solve.

    ``rhs_norm`` is ``||b||`` of the iterated system (``||b_f||`` for
    :func:`solve_constrained`), ``residual_norm`` the recursive residual norm
    of the returned iterate and ``relative_residual`` their ratio (zero for a
    zero right-hand side).
    """

    iterations: int
    converged: bool
    rhs_norm: float
    residual_norm: float
    relative_residual: float


def _check_vector(name: str, value: torch.Tensor, like: torch.Tensor | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(value.shape)}")
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must be float64, got {value.dtype}")
    if like is not None and (value.shape != like.shape or value.device != like.device):
        raise ValueError(
            f"{name} must have shape {tuple(like.shape)} on {like.device}, got "
            f"{tuple(value.shape)} on {value.device}"
        )
    return value


def _check_settings(rtol: float, atol: float, maxiter: int, check_every: int) -> tuple[float, float]:
    rtol, atol = float(rtol), float(atol)
    if not math.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"rtol must be finite and non-negative, got {rtol!r}")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError(f"atol must be finite and non-negative, got {atol!r}")
    if rtol == 0.0 and atol == 0.0:
        raise ValueError("rtol and atol must not both be zero")
    for name, value in (("maxiter", maxiter), ("check_every", check_every)):
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return rtol, atol


def pcg(
    action: Action,
    rhs: torch.Tensor,
    *,
    diag: torch.Tensor,
    x0: torch.Tensor | None = None,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    maxiter: int = 5000,
    check_every: int = 10,
) -> tuple[torch.Tensor, PCGInfo]:
    """Solve ``A x = rhs`` with Jacobi-preconditioned CG.

    Parameters
    ----------
    action:
        Matrix-free SPD action on ``(n,)`` float64 vectors.
    rhs:
        Right-hand side ``b`` ``(n,)``.
    diag:
        Positive Jacobi diagonal ``(n,)``.
    x0:
        Optional warm start; ignored when ``rhs`` is zero (the solution is 0).
    rtol, atol:
        Stop when ``||r|| <= max(rtol ||b||, atol)``.
    maxiter, check_every:
        Iteration cap and convergence/breakdown check period.

    Returns
    -------
    ``(x, PCGInfo)``.  ``info.converged`` is ``False`` when ``maxiter`` is
    reached; breakdown raises :class:`PCGBreakdownError`.
    """

    rtol, atol = _check_settings(rtol, atol, maxiter, check_every)
    maxiter, check_every = int(maxiter), int(check_every)
    _check_vector("rhs", rhs)
    _check_vector("diag", diag, rhs)
    if x0 is not None:
        _check_vector("x0", x0, rhs)

    stats = torch.stack(
        (
            torch.linalg.vector_norm(rhs),
            torch.isfinite(rhs).all().to(rhs.dtype),
            (diag > 0.0).all().to(rhs.dtype),
            torch.isfinite(diag).all().to(rhs.dtype),
        )
    ).cpu()
    b_norm = float(stats[0])
    if stats[1] < 1.0 or not math.isfinite(b_norm):
        raise PCGBreakdownError("PCG right-hand side contains a non-finite value")
    if stats[3] < 1.0:
        raise PCGBreakdownError("PCG diagonal contains a non-finite value")
    if stats[2] < 1.0:
        raise PCGBreakdownError("PCG diagonal has a non-positive entry (non-positive curvature)")
    if b_norm == 0.0:
        return torch.zeros_like(rhs), PCGInfo(0, True, 0.0, 0.0, 0.0)

    threshold = max(rtol * b_norm, atol)
    if x0 is None:
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        r_norm = b_norm
    else:
        x = x0.clone()
        r = rhs - action(x)
        _check_vector("action(x0)", r, rhs)
        r_norm = float(torch.linalg.vector_norm(r))
        if not math.isfinite(r_norm):
            raise PCGBreakdownError("PCG initial residual is not finite")
    if r_norm <= threshold:
        return x, PCGInfo(0, True, b_norm, r_norm, r_norm / b_norm)

    inv_diag = 1.0 / diag
    z = r * inv_diag
    p = z.clone()
    rz = torch.dot(r, z)
    threshold_sq = torch.tensor(threshold * threshold, dtype=rhs.dtype, device=rhs.device)
    zero = torch.zeros((), dtype=rhs.dtype, device=rhs.device)
    active = torch.ones((), dtype=torch.bool, device=rhs.device)
    breakdown = torch.zeros((), dtype=torch.bool, device=rhs.device)
    nonfinite = torch.zeros((), dtype=torch.bool, device=rhs.device)
    done = torch.zeros((), dtype=torch.long, device=rhs.device)
    rr = torch.dot(r, r)

    for iteration in range(1, maxiter + 1):
        Ap = action(p)
        pAp = torch.dot(p, Ap)
        positive = pAp > 0.0
        breakdown = breakdown | (active & ~positive)
        nonfinite = nonfinite | ~torch.isfinite(pAp)
        step = active & positive
        alpha = torch.where(step, rz / torch.where(positive, pAp, torch.ones_like(pAp)), zero)
        x.add_(alpha * p)
        r.sub_(alpha * Ap)
        done = done + step.to(torch.long)
        z = r * inv_diag
        rz_new = torch.dot(r, z)
        rr = torch.dot(r, r)
        nonfinite = nonfinite | ~torch.isfinite(rz_new) | ~torch.isfinite(rr)
        active = active & (rr > threshold_sq)
        beta = torch.where(active & (rz > 0.0), rz_new / torch.where(rz > 0.0, rz, torch.ones_like(rz)), zero)
        p = z + beta * p
        rz = rz_new
        if iteration % check_every == 0 or iteration == maxiter:
            flags = torch.stack(
                (nonfinite.to(rhs.dtype), breakdown.to(rhs.dtype), active.to(rhs.dtype), rr, done.to(rhs.dtype))
            ).cpu()
            if flags[0] > 0.0:
                raise PCGBreakdownError(f"PCG produced a non-finite value by iteration {iteration}")
            if flags[1] > 0.0:
                raise PCGBreakdownError(
                    f"PCG met non-positive curvature p.A p <= 0 by iteration {iteration}; "
                    "the operator is not positive definite on the free set"
                )
            if flags[2] < 1.0:
                r_norm = math.sqrt(float(flags[3]))
                return x, PCGInfo(int(flags[4]), True, b_norm, r_norm, r_norm / b_norm)
            r_norm = math.sqrt(float(flags[3]))

    return x, PCGInfo(maxiter, False, b_norm, r_norm, r_norm / b_norm)


def solve_constrained(
    action: Action,
    *,
    diag: torch.Tensor,
    fixed_mask: torch.Tensor,
    fixed_values: torch.Tensor,
    rhs: torch.Tensor | None = None,
    x0: torch.Tensor | None = None,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    maxiter: int = 5000,
    check_every: int = 10,
) -> tuple[torch.Tensor, PCGInfo]:
    """Solve ``A x = rhs`` with ``x = fixed_values`` on ``fixed_mask`` (projected CG).

    With ``x_c = where(fixed, fixed_values, 0)`` the free system is
    ``A_ff x_f = b_f``, ``b_f = mask_f (rhs - A x_c)``; ``rhs=None`` means zero.
    CG runs on full-length vectors that vanish on the fixed set, with the
    action projected onto the free set.  ``diag`` must be positive on the
    free set (fixed entries are ignored).  When ``||b_f|| = 0`` the solution
    ``x_c`` is returned after 0 iterations.  The stopping rule is relative to
    ``||b_f||``.

    Returns
    -------
    ``(x, PCGInfo)`` with ``x[fixed] == fixed_values[fixed]`` exactly.
    """

    _check_vector("fixed_values", fixed_values)
    n = fixed_values.shape[0]
    if not isinstance(fixed_mask, torch.Tensor) or fixed_mask.dtype != torch.bool:
        raise TypeError("fixed_mask must be a torch.bool tensor")
    if fixed_mask.shape != (n,) or fixed_mask.device != fixed_values.device:
        raise ValueError(f"fixed_mask must have shape ({n},) on {fixed_values.device}")
    _check_vector("diag", diag, fixed_values)
    if rhs is not None:
        _check_vector("rhs", rhs, fixed_values)
    if x0 is not None:
        _check_vector("x0", x0, fixed_values)
    _check_settings(rtol, atol, maxiter, check_every)

    free = ~fixed_mask
    zero = torch.zeros_like(fixed_values)
    x_c = torch.where(fixed_mask, fixed_values, zero)
    image = action(x_c)
    _check_vector("action(x_c)", image, fixed_values)
    b = -image if rhs is None else rhs - image
    b_f = torch.where(free, b, zero)

    def free_action(v: torch.Tensor) -> torch.Tensor:
        return torch.where(free, action(v), zero)

    diag_f = torch.where(free, diag, torch.ones_like(diag))
    x0_f = None if x0 is None else torch.where(free, x0, zero)
    y, info = pcg(
        free_action,
        b_f,
        diag=diag_f,
        x0=x0_f,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        check_every=check_every,
    )
    return torch.where(free, y, x_c), info


__all__ = [
    "Action",
    "PCGBreakdownError",
    "PCGConvergenceError",
    "PCGInfo",
    "pcg",
    "solve_constrained",
]
