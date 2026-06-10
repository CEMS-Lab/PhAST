"""Tests for NOWS scaffold (#61)."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phast.mixed_precision_cg import cg_mixed_precision  # noqa: E402
from phast.research.nows import NOWSPredictor, make_warm_start  # noqa: E402


def _dummy_K_indices(n: int) -> torch.Tensor:
    # Diagonal sparsity pattern (COO).
    rows = torch.arange(n)
    return torch.stack([rows, rows], dim=0)


def test_zero_fallback():
    """No model → predict returns zeros of correct shape/dtype/device."""
    p = NOWSPredictor()
    n = 7
    b = torch.randn(n, dtype=torch.float64)
    K_idx = _dummy_K_indices(n)
    K_val = torch.ones(n, dtype=torch.float64)
    x0 = p.predict(K_idx, K_val, b, n)
    assert x0.shape == (n,)
    assert x0.dtype == torch.float64
    assert torch.all(x0 == 0)
    assert not p.has_model


def test_callable_model():
    """Lambda model returning b * 0.5 is honoured by predict."""
    model = lambda K_idx, K_val, b: b * 0.5  # noqa: E731
    p = NOWSPredictor(model=model)
    b = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
    n = 3
    K_idx = _dummy_K_indices(n)
    K_val = torch.ones(n, dtype=torch.float64)
    x0 = p.predict(K_idx, K_val, b, n)
    assert torch.allclose(x0, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert p.has_model


def test_cg_with_warm_start():
    """CG converges to the same answer with or without a NOWS warm start."""
    torch.manual_seed(0)
    n = 20
    A_dense = torch.randn(n, n, dtype=torch.float64)
    A_dense = A_dense @ A_dense.T + n * torch.eye(n, dtype=torch.float64)  # SPD
    b = torch.randn(n, dtype=torch.float64)
    x_true = torch.linalg.solve(A_dense, b)

    def matvec(v):
        return A_dense @ v

    # Cold start.
    x_cold, _, conv_cold = cg_mixed_precision(matvec, b, tol=1e-12, max_iter=500)
    # Warm start: feed a "good guess" close to the true solution.
    good_guess = x_true + 1e-3 * torch.randn(n, dtype=torch.float64)
    p = NOWSPredictor(model=lambda Ki, Kv, bb: good_guess.clone())
    K_idx = _dummy_K_indices(n)
    K_val = torch.ones(n, dtype=torch.float64)
    x0 = p.predict(K_idx, K_val, b, n)
    x_warm, _, conv_warm = cg_mixed_precision(matvec, b, x0=x0, tol=1e-12, max_iter=500)

    assert conv_cold and conv_warm
    assert torch.allclose(x_cold, x_true, atol=1e-8)
    assert torch.allclose(x_warm, x_true, atol=1e-8)
    assert torch.allclose(x_cold, x_warm, atol=1e-8)


def test_make_warm_start_from_path():
    """Nonexistent model path falls back gracefully to zeros."""
    p = make_warm_start("/nonexistent/path/to/model.pt")
    assert not p.has_model
    n = 5
    b = torch.ones(n, dtype=torch.float64)
    K_idx = _dummy_K_indices(n)
    K_val = torch.ones(n, dtype=torch.float64)
    x0 = p.predict(K_idx, K_val, b, n)
    assert torch.all(x0 == 0)

    # And the no-arg factory call also yields a zero predictor.
    p2 = make_warm_start()
    assert not p2.has_model
