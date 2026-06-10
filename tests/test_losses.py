"""Tests for training/losses.py (#51)."""

from __future__ import annotations

import math

import pytest
import torch

from phast.training.losses import (
    adaptive_loss_weights,
    dice_loss,
    dice_plus_focal,
    focal_loss,
    log_l2_loss,
    log_relative_error_loss,
    rprop_step_factor,
    weighted_loss_sum,
)


def test_log_l2_loss_zero_at_match() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    eps = 1e-8
    loss = log_l2_loss(x, x.clone(), eps=eps)
    assert math.isclose(loss.item(), math.log(eps), rel_tol=1e-6)


def test_log_relative_error_loss_finite() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    target = torch.tensor([1.1, 2.1, 3.1], dtype=torch.float64)
    loss = log_relative_error_loss(pred, target)
    assert torch.isfinite(loss)


def test_dice_loss_at_perfect() -> None:
    p = torch.ones(8, dtype=torch.float64)
    loss = dice_loss(p, p.clone(), smooth=1e-8)
    assert loss.item() < 1e-6


def test_dice_loss_at_disjoint() -> None:
    pred = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    target = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    loss = dice_loss(pred, target, smooth=1e-8)
    assert loss.item() > 0.99


def test_focal_loss_balanced_easy() -> None:
    pred = torch.tensor([0.99, 0.01], dtype=torch.float64)
    target = torch.tensor([1.0, 0.0], dtype=torch.float64)
    loss_easy = focal_loss(pred, target)
    pred_hard = torch.tensor([0.5, 0.5], dtype=torch.float64)
    loss_hard = focal_loss(pred_hard, target)
    assert loss_easy < loss_hard


def test_focal_loss_hard_wrong() -> None:
    pred = torch.tensor([0.01, 0.99], dtype=torch.float64)
    target = torch.tensor([1.0, 0.0], dtype=torch.float64)
    loss = focal_loss(pred, target)
    assert loss.item() > 1.0


def test_focal_loss_alpha_weights_classes() -> None:
    pred = torch.tensor([0.5, 0.5], dtype=torch.float64)
    pos = torch.tensor([1.0, 1.0], dtype=torch.float64)
    neg = torch.tensor([0.0, 0.0], dtype=torch.float64)
    alpha = 0.25
    pos_loss = focal_loss(pred, pos, alpha=alpha, gamma=0.0)
    neg_loss = focal_loss(pred, neg, alpha=alpha, gamma=0.0)
    assert math.isclose(
        (pos_loss / neg_loss).item(),
        alpha / (1.0 - alpha),
        rel_tol=1e-12,
    )


def test_dice_plus_focal_combines() -> None:
    pred = torch.rand(16, dtype=torch.float64)
    target = (torch.rand(16, dtype=torch.float64) > 0.7).to(torch.float64)
    loss = dice_plus_focal(pred, target)
    assert torch.isfinite(loss)


def test_rprop_step_sign_change() -> None:
    grad = torch.tensor([1.0, -1.0, 0.0, 1.0], dtype=torch.float64)
    prev = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    f = rprop_step_factor(grad, prev, eta_plus=1.2, eta_minus=0.5)
    # idx 0: same sign -> eta_plus
    assert math.isclose(f[0].item(), 1.2)
    # idx 1: opposite sign -> eta_minus
    assert math.isclose(f[1].item(), 0.5)
    # idx 2: grad zero -> 1.0
    assert math.isclose(f[2].item(), 1.0)
    # idx 3: prev zero -> 1.0
    assert math.isclose(f[3].item(), 1.0)


def test_adaptive_loss_weights_balance() -> None:
    losses = {
        "small": torch.tensor(0.01, dtype=torch.float64),
        "large": torch.tensor(100.0, dtype=torch.float64),
    }
    w = adaptive_loss_weights(losses)
    ratio = w["small"] / w["large"]
    assert math.isclose(ratio.item(), 100.0 / 0.01, rel_tol=1e-3)
    assert math.isclose((w["small"] + w["large"]).item(), 1.0, rel_tol=1e-12)


def test_adaptive_loss_weights_preserve_tensor_dtype_device() -> None:
    losses = {
        "a": torch.tensor(1.0, dtype=torch.float64),
        "b": torch.tensor(2.0, dtype=torch.float64),
    }
    weights = adaptive_loss_weights(losses)
    assert weights["a"].dtype == torch.float64
    assert weights["a"].device == losses["a"].device
    assert not weights["a"].requires_grad


def test_weighted_loss_sum_backpropagates_to_losses() -> None:
    x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    losses = {"quad": x ** 2, "linear": 0.5 * x}
    total = weighted_loss_sum(losses)
    total.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad)


def test_autograd_through_dice() -> None:
    pred = torch.rand(8, dtype=torch.float64, requires_grad=True)
    target = (torch.rand(8, dtype=torch.float64) > 0.5).to(torch.float64)
    loss = dice_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_autograd_through_focal() -> None:
    pred = torch.full((8,), 0.5, dtype=torch.float64, requires_grad=True)
    target = (torch.rand(8, dtype=torch.float64) > 0.5).to(torch.float64)
    loss = focal_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_float64_dtype_preserved() -> None:
    x = torch.ones(4, dtype=torch.float64)
    assert log_l2_loss(x, x).dtype == torch.float64
    assert dice_loss(x, x).dtype == torch.float64
