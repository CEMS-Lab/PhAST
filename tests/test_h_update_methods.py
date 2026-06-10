"""Tests for the H-update dispatcher (issue #360).

Covers:
* byte-identity of the ``hard_max`` default with ``torch.maximum``;
* forward agreement of ``smooth_max`` and ``softmax`` with the hard max
  to within their analytic tolerances;
* finiteness of the ``smooth_max`` gradient at the cusp ``a == b`` where
  ``torch.maximum`` is sub-differentiable;
* ``softmax`` collapsing to ``hard_max`` for large ``beta``;
* ``log_smooth`` and ``custom_subgrad`` dispatching to implemented
  differentiable alternatives;
* dispatcher routing and rejection of unknown methods.
"""
from __future__ import annotations

import pytest
import torch

from phast import h_update as hu
from phast.staggered_solver import SolverConfig, StaggeredSolver


# ----------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------

def _sample_pair(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(64, generator=g, dtype=torch.float64)
    b = torch.randn(64, generator=g, dtype=torch.float64)
    return a, b


# ----------------------------------------------------------------------
# 1. hard_max byte-identity
# ----------------------------------------------------------------------

def test_hard_max_byte_identical_to_torch_maximum():
    a, b = _sample_pair(seed=1)
    out = hu.hard_max(a, b)
    ref = torch.maximum(a, b)
    assert torch.equal(out, ref), "hard_max must be byte-identical to torch.maximum"


def test_dispatch_hard_max_byte_identical():
    a, b = _sample_pair(seed=2)
    out = hu.dispatch("hard_max", a, b)
    ref = torch.maximum(a, b)
    assert torch.equal(out, ref)


# ----------------------------------------------------------------------
# 2. smooth_max forward + gradient
# ----------------------------------------------------------------------

def test_smooth_max_forward_close_to_hard():
    a, b = _sample_pair(seed=3)
    out = hu.smooth_max(a, b, eps=1e-10)
    ref = torch.maximum(a, b)
    err = (out - ref).abs().max().item()
    assert err < 1e-9, f"smooth_max forward error {err} >= 1e-9"


def test_smooth_max_gradient_finite_at_cusp():
    # At a == b, the hard max has subgradient {0.5, 0.5} but autograd
    # picks one branch. smooth_max must give a smooth, finite gradient.
    a = torch.tensor([0.5, 1.0, -2.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([0.5, 1.0, -2.0], dtype=torch.float64, requires_grad=True)
    out = hu.smooth_max(a, b, eps=1e-6).sum()
    out.backward()
    assert a.grad is not None and b.grad is not None
    assert torch.isfinite(a.grad).all(), "smooth_max grad must be finite at a==b"
    assert torch.isfinite(b.grad).all()
    # Symmetry: at a == b each side should carry exactly half the gradient
    assert torch.allclose(a.grad, b.grad, atol=1e-12)
    assert torch.allclose(a.grad, torch.full_like(a.grad, 0.5), atol=1e-6)


# ----------------------------------------------------------------------
# 3. softmax behaviour
# ----------------------------------------------------------------------

def test_softmax_large_beta_approaches_hard_max():
    a, b = _sample_pair(seed=4)
    out = hu.softmax(a, b, beta=1e6)
    ref = torch.maximum(a, b)
    err = (out - ref).abs().max().item()
    # log(2)/beta upper bound on the bias when a == b; here beta = 1e6.
    assert err < 1e-6, f"softmax(1e6) err {err} >= 1e-6"


def test_softmax_rejects_nonpositive_beta():
    a, b = _sample_pair(seed=5)
    with pytest.raises(ValueError):
        hu.softmax(a, b, beta=0.0)
    with pytest.raises(ValueError):
        hu.softmax(a, b, beta=-1.0)


# ----------------------------------------------------------------------
# 4. implemented #361/#362 alternatives
# ----------------------------------------------------------------------

def test_log_smooth_dispatch_implemented_and_differentiable_near_ridge():
    a = torch.ones(4, dtype=torch.float64)
    b = (a * (1.0 - 1e-7)).detach().requires_grad_(True)
    out = hu.dispatch("log_smooth", a, b, beta=1e6)
    assert out.shape == a.shape
    (gb,) = torch.autograd.grad(out.sum(), b)
    assert torch.isfinite(gb).all()
    assert (gb.abs() > 0).all()


def test_custom_subgrad_dispatch_forward_exact_and_splits_cusp_grad():
    a = torch.full((4,), 0.7, dtype=torch.float64, requires_grad=True)
    b = torch.full((4,), 0.7, dtype=torch.float64, requires_grad=True)
    out = hu.dispatch("custom_subgrad", a, b, scale=1e10)
    assert torch.equal(out, torch.maximum(a, b))
    out.sum().backward()
    expected = torch.full_like(a, 0.5)
    assert torch.allclose(a.grad, expected, atol=1e-12)
    assert torch.allclose(b.grad, expected, atol=1e-12)


def test_solver_config_accepts_custom_subgrad_scale():
    cfg = SolverConfig(H_update_method="custom_subgrad", H_update_scale=1.0)
    assert cfg.H_update_method == "custom_subgrad"
    assert cfg.H_update_scale == 1.0


# ----------------------------------------------------------------------
# 5. Dispatcher routing
# ----------------------------------------------------------------------

def test_dispatch_unknown_method_raises():
    a, b = _sample_pair(seed=8)
    with pytest.raises(ValueError, match="Unknown H_update_method"):
        hu.dispatch("does_not_exist", a, b)


@pytest.mark.parametrize(
    "method,expected_max_err",
    [
        ("hard_max", 0.0),
        ("smooth_max", 1e-9),
        ("softmax", 5e-3),  # default beta=1e3 -> log(2)/1e3 ~= 6.9e-4
    ],
)
def test_dispatch_parametrized_implemented_methods(method, expected_max_err):
    a, b = _sample_pair(seed=9)
    out = hu.dispatch(method, a, b)
    ref = torch.maximum(a, b)
    err = (out - ref).abs().max().item()
    assert err <= expected_max_err, (
        f"{method} forward error {err} exceeded {expected_max_err}"
    )


def test_dispatch_forwards_kwargs():
    # At a == b, softmax bias is exactly log(2)/beta. Use a small beta so the
    # default and override branches sit clearly above float64 epsilon and the
    # ordering is unambiguous.
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    b = a.clone()
    out_low = hu.dispatch("softmax", a, b, beta=1.0)
    out_high = hu.dispatch("softmax", a, b, beta=10.0)
    ref = torch.maximum(a, b)
    err_low = (out_low - ref).abs().max().item()
    err_high = (out_high - ref).abs().max().item()
    assert err_high < err_low, (
        f"larger beta should tighten approximation: low={err_low}, high={err_high}"
    )
    # Cross-check: log(2)/beta is the analytical bias at a == b.
    import math
    assert abs(err_low - math.log(2.0) / 1.0) < 1e-12
    assert abs(err_high - math.log(2.0) / 10.0) < 1e-12


def test_solver_h_update_forwards_softmax_beta_for_explicit_method():
    solver = object.__new__(StaggeredSolver)
    solver.config = SolverConfig(H_update_method="softmax", softmax_H_beta=10.0)
    a = torch.tensor([1.0], dtype=torch.float64)
    b = torch.tensor([1.0], dtype=torch.float64)
    out = solver._H_update(a, b)
    ref = hu.softmax(a, b, beta=10.0)
    assert torch.allclose(out, ref)


def test_allowed_methods_constant_matches_dispatch_table():
    # Defensive: dispatcher and the public ALLOWED_METHODS tuple must agree.
    for m in hu.ALLOWED_METHODS:
        if m == "log_smooth":
            a = torch.rand(64, dtype=torch.float64) + 1e-3
            b = torch.rand(64, dtype=torch.float64) + 1e-3
        else:
            a, b = _sample_pair(seed=11)
        hu.dispatch(m, a, b)
