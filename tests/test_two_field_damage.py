"""Unit tests for two_field_damage scaffold (issue #258)."""

from __future__ import annotations

import math
import pytest
import torch

from phast.research.two_field_damage import (
    TwoFieldState,
    alpha_to_d_map,
    coupling_energy,
    two_field_damage_density,
    update_d_from_alpha,
)


def _t(x):
    return torch.tensor(x, dtype=torch.float64)


def test_state_irreversibility():
    state = TwoFieldState(
        alpha=_t([0.2, 0.4]),
        d=_t([0.1, 0.2]),
        d_prev=_t([0.3, 0.5]),
    )
    assert torch.all(state.d >= state.d_prev)
    assert torch.allclose(state.d, _t([0.3, 0.5]))


def test_alpha_to_d_modes():
    a = _t([0.5])
    assert torch.allclose(alpha_to_d_map(a, 'identity'), _t([0.5]))
    assert torch.allclose(alpha_to_d_map(a, 'quadratic'), _t([0.25]))
    assert torch.allclose(alpha_to_d_map(a, 'tanh'), _t([math.tanh(1.5)]))
    with pytest.raises(ValueError):
        alpha_to_d_map(a, 'bogus')


def test_coupling_energy_zero_at_equal():
    a = _t([0.3, 0.7])
    d = _t([0.3, 0.7])
    assert torch.allclose(coupling_energy(a, d, k=2.0), _t(0.0))


def test_coupling_energy_positive_at_disagree():
    a = _t([1.0])
    d = _t([0.0])
    out = coupling_energy(a, d, k=2.0)
    # 0.5 * 2.0 * (1-0)^2 = 1.0
    assert torch.allclose(out, _t(1.0))


def test_density_reduces_to_single_field_at_high_k():
    a = _t([0.8])
    d = _t([0.2])
    grad = torch.zeros(1, 2, dtype=torch.float64)
    low = two_field_damage_density(a, d, grad, Gc=1.0, c_w=2.0, l_alpha=0.1, k_couple=1.0)
    high = two_field_damage_density(a, d, grad, Gc=1.0, c_w=2.0, l_alpha=0.1, k_couple=1e6)
    # Coupling term dominates at high k; density should grow accordingly
    assert (high > low).all()
    # Coupling part ~ 0.5 * 1e6 * 0.36 = 1.8e5
    assert high.item() > 1e5


def test_update_d_irreversibility():
    state = TwoFieldState(
        alpha=_t([0.3]),
        d=_t([0.5]),
        d_prev=_t([0.5]),
    )
    new_state = update_d_from_alpha(state, mode='identity')
    # alpha=0.3 < d_prev=0.5 -> d stays at 0.5
    assert torch.allclose(new_state.d, _t([0.5]))


def test_autograd_through_density():
    alpha = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([0.3, 0.5], dtype=torch.float64, requires_grad=True)
    grad_alpha = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float64, requires_grad=True)
    density = two_field_damage_density(alpha, d, grad_alpha,
                                        Gc=1.0, c_w=2.0, l_alpha=0.05, k_couple=1.0)
    density.sum().backward()
    for g in (alpha.grad, d.grad, grad_alpha.grad):
        assert g is not None
        assert torch.isfinite(g).all()
