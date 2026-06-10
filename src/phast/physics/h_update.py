"""History-variable update operators for the staggered phase-field solver.

The staggered AT2 update enforces irreversibility through a history variable

    H_{n+1} = max(H_n, psi_+(eps_{n+1}))

The hard ``torch.maximum`` is exact but has a piecewise-constant subgradient:
gradients only flow through the branch that currently holds the max, so a
parameter perturbation that shifts which timestep peaks produces a jump in
the gradient. That breaks L-BFGS / Adam on long dynamic inversions.

This module provides a small dispatcher of smooth-max alternatives so the
solver can swap the operator from configuration without touching the call
site. The default ``hard_max`` reproduces ``torch.maximum`` byte-for-byte
so that all forward benchmarks remain bit-identical when the flag is unset.

Methods
-------
hard_max(a, b)
    Exact maximum; equivalent to ``torch.maximum(a, b)``. Default.
softmax(a, b, beta=1e3)
    Log-sum-exp smoothing: ``(1/beta) * logsumexp(beta * stack([a, b]))``.
    -> hard_max as ``beta -> oo``.
smooth_max(a, b, eps=1e-10)
    Quadratic smoothing: ``0.5 * (a + b + sqrt((a - b)^2 + eps^2))``.
    -> hard_max as ``eps -> 0``.
log_smooth(...)
    Log-domain softplus smoothing with multiplicative bias control.
custom_subgrad(...)
    Forward-exact hard maximum with sigmoid-weighted backward.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch

ALLOWED_METHODS = (
    "hard_max",
    "softmax",
    "smooth_max",
    "log_smooth",
    "custom_subgrad",
)


def hard_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact element-wise maximum. Byte-identical to ``torch.maximum``."""
    return torch.maximum(a, b)


def softmax(
    a: torch.Tensor, b: torch.Tensor, beta: float = 1e3
) -> torch.Tensor:
    """Log-sum-exp smooth maximum.

    ``(1/beta) * logsumexp(beta * stack([a, b]))`` -> ``max(a, b)`` as
    ``beta -> oo``. ``torch.logsumexp`` is numerically stable.
    """
    if beta <= 0:
        raise ValueError(f"softmax beta must be > 0, got {beta}")
    stacked = torch.stack([a, b])
    return (1.0 / beta) * torch.logsumexp(beta * stacked, dim=0)


def smooth_max(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """Quadratic smooth maximum.

    ``0.5 * (a + b + sqrt((a - b)^2 + eps^2))`` -> ``max(a, b)`` as
    ``eps -> 0``. The ``+ eps^2`` regulariser keeps the gradient finite
    at ``a == b`` (the cusp of the hard max).
    """
    if eps < 0:
        raise ValueError(f"smooth_max eps must be >= 0, got {eps}")
    diff = a - b
    return 0.5 * (a + b + torch.sqrt(diff * diff + eps * eps))


def log_smooth(
    a: torch.Tensor, b: torch.Tensor, beta: float = 1.0e6
) -> torch.Tensor:
    """Log-domain softplus smoothing for ``max(a, b)``.

    This computes ``log(exp(beta*a) + exp(beta*b)) / beta`` in a stable form.
    """
    if beta <= 0:
        raise ValueError(f"log_smooth beta must be positive, got {beta}")
    m = torch.maximum(a, b)
    return m + torch.log1p(torch.exp(-beta * torch.abs(a - b))) / beta


def custom_subgrad(
    a: torch.Tensor, b: torch.Tensor, scale: float = 1.0e10
) -> torch.Tensor:
    """Custom-subgradient smooth backward (closes #362).

    Wires the dispatcher to the standalone module
    ``phast.h_update_custom_subgrad.custom_subgrad``
    landed in PR #367. Forward is byte-exact ``torch.maximum``;
    backward is sigmoid-weighted so both operands get nonzero
    gradient near the active-set transition.
    """
    from .h_update_custom_subgrad import custom_subgrad as _impl
    return _impl(a, b, scale=scale)


_DISPATCH: Dict[str, Callable] = {
    "hard_max": hard_max,
    "softmax": softmax,
    "smooth_max": smooth_max,
    "log_smooth": log_smooth,
    "custom_subgrad": custom_subgrad,
}


def dispatch(
    method: str,
    a: torch.Tensor,
    b: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Route to the named H-update operator.

    Parameters
    ----------
    method : str
        One of :data:`ALLOWED_METHODS`.
    a, b : torch.Tensor
        Operands.
    **kwargs
        Forwarded to the underlying operator (e.g. ``beta``, ``eps``).

    Raises
    ------
    ValueError
        If ``method`` is not in :data:`ALLOWED_METHODS`.
    """
    if method not in _DISPATCH:
        raise ValueError(
            f"Unknown H_update_method '{method}'. "
            f"Allowed: {ALLOWED_METHODS}"
        )
    return _DISPATCH[method](a, b, **kwargs)
