"""Advanced loss functions for damage prediction (#51).

Standalone module — not yet wired into a training pipeline. The training
pipeline that consumes these is separate work.

Implements literature quick-wins from issue #51:
- log(E) loss (Manav 2024)            -> log_l2_loss, log_relative_error_loss
- adaptive loss balancing (Kiyani 25) -> adaptive_loss_weights, weighted_loss_sum
- RPROP step factor (Manav 2024)      -> rprop_step_factor
- Dice + Focal (Hamdi 2026)           -> dice_loss, focal_loss, dice_plus_focal

All functions are float64-compatible and autograd-friendly.
"""

from __future__ import annotations

import torch


def log_l2_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """log(L2) loss — bounds gradient magnitude across orders of magnitude.

    Returns: ``log(eps + mean((pred - target)**2))``.
    """
    return torch.log(eps + ((pred - target) ** 2).mean())


def log_relative_error_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """log relative error — invariant to absolute scale of target.

    Returns: ``log(eps + mean(|pred - target|) / (mean(|target|) + eps))``.
    """
    num = (pred - target).abs().mean()
    den = target.abs().mean() + eps
    return torch.log(eps + num / den)


def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    """Dice loss for damage segmentation: ``1 - 2|P intersect T| / (|P| + |T|)``.

    Both ``pred`` and ``target`` should be in [0, 1]. Treats damage as a soft mask.
    """
    p = pred.flatten()
    t = target.flatten()
    intersection = (p * t).sum()
    return 1.0 - (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Binary focal loss — down-weights well-classified examples.

    Both inputs assumed to be probabilities in [0, 1]. Useful for crack
    localisation where most elements have ``d ~ 0`` (easy negatives).
    """
    pred = pred.clamp(eps, 1.0 - eps)
    is_positive = target > 0.5
    pt = torch.where(is_positive, pred, 1.0 - pred)
    alpha_t = torch.where(
        is_positive,
        torch.as_tensor(alpha, dtype=pred.dtype, device=pred.device),
        torch.as_tensor(1.0 - alpha, dtype=pred.dtype, device=pred.device),
    )
    return -(alpha_t * (1.0 - pt) ** gamma * pt.log()).mean()


def dice_plus_focal(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_dice: float = 1.0,
    lambda_focal: float = 1.0,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Combined Dice + Focal — popular for class-imbalanced segmentation."""
    return lambda_dice * dice_loss(pred, target) + lambda_focal * focal_loss(
        pred, target, alpha, gamma
    )


def rprop_step_factor(
    grad: torch.Tensor,
    prev_grad: torch.Tensor,
    eta_plus: float = 1.2,
    eta_minus: float = 0.5,
    step_min: float = 1e-6,
    step_max: float = 50.0,
) -> torch.Tensor:
    """RPROP-style adaptive step factor.

    For each parameter, multiply its step size by ``eta_plus`` when ``grad``
    sign matches ``prev_grad``, ``eta_minus`` otherwise. Returns the
    multiplicative step factor, clamped to ``[step_min, step_max]``.
    """
    sign_change = grad * prev_grad
    factor = torch.where(
        sign_change > 0,
        torch.full_like(grad, eta_plus),
        torch.where(
            sign_change < 0,
            torch.full_like(grad, eta_minus),
            torch.ones_like(grad),
        ),
    )
    return factor.clamp(min=step_min, max=step_max)


def adaptive_loss_weights(
    losses: dict[str, torch.Tensor], eps: float = 1e-8
) -> dict[str, torch.Tensor]:
    """Inverse-magnitude weighting (Kiyani 2025).

    Returns detached tensor weights ``w_i = (1 / (|L_i| + eps)) / sum_j(...)``
    on the same dtype/device as the first loss. The weights do not propagate
    gradients, but they remain tensors so callers can use them directly in a
    torch training loop without losing dtype/device information.
    """
    if not losses:
        raise ValueError("losses must contain at least one term")
    first = next(iter(losses.values()))
    dtype = first.dtype
    device = first.device
    inv_terms: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, loss in losses.items():
            mag = loss.detach().abs().to(dtype=dtype, device=device)
            inv_terms[name] = 1.0 / (mag + eps)
        denom = torch.stack([v.reshape(()) for v in inv_terms.values()]).sum()
        weights = {name: inv / denom for name, inv in inv_terms.items()}
    return weights


def weighted_loss_sum(
    losses: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return ``sum_i w_i L_i`` using adaptive weights by default."""
    if not losses:
        raise ValueError("losses must contain at least one term")
    if weights is None:
        weights = adaptive_loss_weights(losses, eps=eps)
    total = None
    for name, loss in losses.items():
        if name not in weights:
            raise KeyError(f"missing weight for loss term {name!r}")
        term = weights[name].to(dtype=loss.dtype, device=loss.device) * loss
        total = term if total is None else total + term
    assert total is not None
    return total
