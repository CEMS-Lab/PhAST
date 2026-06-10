"""Two-field damage formulation: process-zone field alpha distinct from damage d (#258).

Scaffold only — provides primitives for combined (alpha, d) state and the
alpha-d coupling energy. Not wired into damage_solver.py.

References:
  Wu (2017) — unified PF-CZM
  Wick et al. — staggered two-field schemes
"""

from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class TwoFieldState:
    """Combined (alpha, d) state at element or node centres.

    Attributes:
        alpha: process-zone field, [0, 1] (reversible).
        d:     damage field, [0, 1] (irreversible: d_t >= d_{t-1}).
        d_prev: previous timestep d, for irreversibility check.
    """
    alpha: torch.Tensor
    d: torch.Tensor
    d_prev: torch.Tensor

    def __post_init__(self):
        # Defensive: clamp to bounds
        self.alpha = self.alpha.clamp(0.0, 1.0)
        self.d = torch.maximum(self.d.clamp(0.0, 1.0), self.d_prev)
        # Irreversibility ON by construction


def alpha_to_d_map(alpha: torch.Tensor, mode: str = 'identity') -> torch.Tensor:
    """Map alpha -> candidate d.

    Modes:
      'identity':  d_candidate = alpha
      'quadratic': d_candidate = alpha**2  (smooth onset)
      'tanh':      d_candidate = tanh(3*alpha) (sharp turn-on near alpha=0.3-0.5)
    """
    if mode == 'identity':
        return alpha
    if mode == 'quadratic':
        return alpha ** 2
    if mode == 'tanh':
        return torch.tanh(3.0 * alpha)
    raise ValueError(f"unknown alpha_to_d mode: {mode!r}")


def coupling_energy(alpha: torch.Tensor, d: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Quadratic coupling penalty (alpha - d)^2 with stiffness k.

    Forces alpha and d to track each other; in the limit k -> infinity, two-field
    reduces to single-field. Standard in unified PF-CZM literature.
    """
    return 0.5 * k * ((alpha - d) ** 2).mean()


def two_field_damage_density(alpha: torch.Tensor, d: torch.Tensor,
                              grad_alpha: torch.Tensor,
                              Gc: float, c_w: float, l_alpha: float,
                              k_couple: float = 1.0) -> torch.Tensor:
    """Compute the two-field damage energy density.

    Args:
        alpha:      (n,)
        d:          (n,)
        grad_alpha: (n, dim)
        Gc:         critical energy release rate
        c_w:        AT1/AT2 normalisation (8/3 for AT1, 2 for AT2)
        l_alpha:    process-zone length scale for alpha
        k_couple:   alpha-d coupling stiffness

    Returns: (n,) energy density per element/integration point.
    """
    # AT1 / AT2-style well term on alpha; here using w(alpha) = alpha (AT1)
    well_density = (Gc / c_w) * alpha
    grad_density = (Gc * l_alpha ** 2 / c_w) * (grad_alpha ** 2).sum(dim=-1)
    coupling_density = 0.5 * k_couple * (alpha - d) ** 2
    return well_density + grad_density + coupling_density


def update_d_from_alpha(state: TwoFieldState, mode: str = 'identity') -> TwoFieldState:
    """Update d = max(d_prev, alpha_to_d(alpha))  — enforces irreversibility.

    Returns a NEW state (does not mutate input).
    """
    d_candidate = alpha_to_d_map(state.alpha, mode=mode)
    new_d = torch.maximum(state.d_prev, d_candidate)
    return TwoFieldState(alpha=state.alpha.clone(), d=new_d, d_prev=state.d_prev.clone())
