"""Tests for Volterra time-delayed damage activation primitive (#258)."""

import math

import pytest
import torch

from phast.research.volterra_delay import (
    convolve_history,
    delayed_activation_threshold,
    kernel,
    per_element_delayed_activation,
)


DTYPE = torch.float64
TAU_D = 1.0


def test_exponential_kernel_normalisation():
    # K_exp integrates to 1 over [0, infty); approximate over [0, 20*tau_d].
    n = 20000
    dt = 20.0 * TAU_D / n
    t = torch.arange(n, dtype=DTYPE) * dt
    K = kernel(t, TAU_D, kind="exponential")
    integral = (K.sum() * dt).item()
    assert abs(integral - 1.0) < 1e-3


def test_kernel_at_zero():
    t0 = torch.zeros(1, dtype=DTYPE)
    K = kernel(t0, TAU_D, kind="exponential")
    assert abs(K.item() - 1.0 / TAU_D) < 1e-12


def test_heaviside_kernel():
    t = torch.tensor([-0.1, 0.0, 0.5, 1.0, 1.5], dtype=DTYPE)
    K = kernel(t, TAU_D, kind="heaviside")
    # clamp(min=0) maps -0.1 -> 0 (inside [0, tau_d]), so it returns 1/tau_d.
    expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0], dtype=DTYPE)
    assert torch.allclose(K, expected, atol=1e-12)


def test_convolve_constant_unit_history():
    # history = ones over a long window; exponential kernel.
    # Analytic: int_0^T (1/tau) exp(-(T-tau)/tau) d(tau) = 1 - exp(-T/tau).
    n_steps = 2000
    dt = 0.01
    history = torch.ones(n_steps, dtype=DTYPE)
    val = convolve_history(history, dt, TAU_D, kind="exponential").item()
    T = n_steps * dt
    expected = 1.0 - math.exp(-T / TAU_D)
    assert abs(val - expected) < 1e-2


def test_delayed_activation_below_threshold():
    history = 0.01 * torch.ones(500, dtype=DTYPE)
    out = delayed_activation_threshold(history, dt=0.01, tau_d=TAU_D, threshold=1.0)
    assert out.item() < 1e-6


def test_delayed_activation_above_threshold():
    history = 10.0 * torch.ones(500, dtype=DTYPE)
    out = delayed_activation_threshold(history, dt=0.01, tau_d=TAU_D, threshold=1.0)
    assert out.item() > 1.0 - 1e-6


def test_per_element_activation_shape():
    n_elem, n_steps = 5, 100
    drv = torch.zeros((n_elem, n_steps), dtype=DTYPE)
    drv[0] = 10.0  # element 0 above threshold
    out = per_element_delayed_activation(
        drv, dt=0.01, tau_d=TAU_D, threshold=1.0, kind="exponential"
    )
    assert out.shape == (n_elem,)
    assert out[0].item() > 1.0 - 1e-6
    assert out[1].item() < 1e-6


def test_autograd_through_activation():
    history = torch.linspace(0.0, 5.0, 200, dtype=DTYPE).requires_grad_(True)
    out = delayed_activation_threshold(history, dt=0.01, tau_d=TAU_D, threshold=1.0)
    out.backward()
    assert history.grad is not None
    assert torch.isfinite(history.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
