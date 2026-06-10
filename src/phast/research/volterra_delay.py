"""Time-delayed damage activation via Volterra memory kernel (#258).

Damage activation is delayed by integrating strain history against a memory
kernel K(t - tau). When the convolution integral exceeds a threshold,
damage activates. Three kernel forms supplied:

  exponential:  K(t) = (1/tau_d) * exp(-t / tau_d)
  power_law:    K(t) = (alpha / tau_d) * (t / tau_d)^(alpha - 1) * exp(-(t/tau_d)^alpha)
  heaviside:    K(t) = (1 / tau_d) * 1[0 <= t <= tau_d]   (rectangular kernel)

Scaffold only -- provides primitives. Not wired into damage_solver.py.

References:
  Knauss (1989) -- viscoelastic crack growth
  Voyiadjis & Faghihi (2014) -- time-dependent damage with memory
"""

from __future__ import annotations

from typing import Literal

import torch


KernelType = Literal["exponential", "power_law", "heaviside"]


def kernel(
    t: torch.Tensor,
    tau_d: float,
    kind: KernelType = "exponential",
    alpha: float = 1.0,
) -> torch.Tensor:
    """Evaluate the Volterra memory kernel K(t).

    Args:
        t:      shape (...,) elapsed time(s) since the strain spike
        tau_d:  scalar -- characteristic delay time
        kind:   kernel form
        alpha:  shape parameter for power_law (defaults to 1 = exponential limit)

    Returns:
        Same shape as t.
    """
    t = t.clamp(min=0.0)
    if kind == "exponential":
        return (1.0 / tau_d) * torch.exp(-t / tau_d)
    if kind == "power_law":
        u = (t / tau_d).clamp(min=1e-12)
        return (alpha / tau_d) * (u ** (alpha - 1.0)) * torch.exp(-(u ** alpha))
    if kind == "heaviside":
        out = torch.where(
            (t >= 0.0) & (t <= tau_d),
            torch.full_like(t, 1.0 / tau_d),
            torch.zeros_like(t),
        )
        return out
    raise ValueError(f"unknown kernel kind: {kind!r}")


def convolve_history(
    history: torch.Tensor,
    dt: float,
    tau_d: float,
    kind: KernelType = "exponential",
    alpha: float = 1.0,
) -> torch.Tensor:
    """Compute the Volterra integral int K(t - tau) f(tau) d(tau) at the latest step.

    Args:
        history: (n_steps,) -- driver field (e.g. peak strain) at each prior step
                 history[-1] = current step
                 history[0]  = oldest retained
        dt:      time step
        tau_d:   delay time
        kind, alpha:  kernel parameters

    Returns:
        Scalar tensor -- convolution value at the current step.
    """
    n_steps = history.shape[0]
    times_back = (
        torch.arange(n_steps - 1, -1, -1, dtype=history.dtype, device=history.device)
        * dt
    )
    K = kernel(times_back, tau_d, kind=kind, alpha=alpha)
    return (K * history).sum() * dt


def delayed_activation_threshold(
    history: torch.Tensor,
    dt: float,
    tau_d: float,
    threshold: float,
    kind: KernelType = "exponential",
    alpha: float = 1.0,
) -> torch.Tensor:
    """Returns ~1.0 if Volterra-convolved history exceeds threshold, else ~0.0.

    Smoothed via a sigmoid for autograd-friendliness -- sharp threshold approached
    as the steepness factor (1000.0) increases.
    """
    integral = convolve_history(history, dt, tau_d, kind=kind, alpha=alpha)
    return torch.sigmoid(1000.0 * (integral - threshold))


def per_element_delayed_activation(
    driver_history: torch.Tensor,
    dt: float,
    tau_d: float,
    threshold: float,
    kind: KernelType = "exponential",
    alpha: float = 1.0,
) -> torch.Tensor:
    """Vectorised version: driver_history shape (n_elements, n_steps).

    Returns shape (n_elements,) -- activation factor in [0, 1] per element.
    """
    n_steps = driver_history.shape[-1]
    times_back = (
        torch.arange(
            n_steps - 1, -1, -1, dtype=driver_history.dtype, device=driver_history.device
        )
        * dt
    )
    K = kernel(times_back, tau_d, kind=kind, alpha=alpha)  # (n_steps,)
    integral = (K * driver_history).sum(dim=-1) * dt
    return torch.sigmoid(1000.0 * (integral - threshold))
