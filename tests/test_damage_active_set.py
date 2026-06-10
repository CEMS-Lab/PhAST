import torch

from phast.damage_solver import (
    classify_damage_active_set,
    projected_damage_active_mask,
    zero_active_entries,
)


def test_damage_active_set_upper_bound_takes_precedence_over_lower():
    d_prev = torch.tensor([0.2, 0.5, 1.0, 0.0], dtype=torch.float64)
    d_new = torch.tensor([0.2, 0.7, 1.0, 1.0], dtype=torch.float64)

    active = classify_damage_active_set(d_new, d_prev)

    assert active.lower.tolist() == [True, False, False, False]
    assert active.upper.tolist() == [False, False, True, True]
    assert active.interior.tolist() == [False, True, False, False]


def test_damage_active_set_fixed_nodes_are_not_lower_or_interior():
    d_prev = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    d_new = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float64)
    fixed = torch.tensor([False, True, False])

    active = classify_damage_active_set(d_new, d_prev, fixed=fixed)

    assert active.fixed.tolist() == [False, True, False]
    assert active.lower.tolist() == [True, False, False]
    assert active.interior.tolist() == [False, False, True]


def test_projected_damage_active_mask_and_zeroing_include_fixed_nodes():
    d = torch.tensor([0.1, 0.5, 1.0, 0.4], dtype=torch.float64)
    lb = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    residual = torch.tensor([-1.0, -1.0, 2.0, 3.0], dtype=torch.float64)
    fixed = torch.tensor([False, False, False, True])

    active = projected_damage_active_mask(d, lb, residual, fixed=fixed)

    assert active.tolist() == [True, False, True, True]
    masked = zero_active_entries(residual, active)
    assert torch.allclose(
        masked,
        torch.tensor([0.0, -1.0, 0.0, 0.0], dtype=torch.float64),
    )
