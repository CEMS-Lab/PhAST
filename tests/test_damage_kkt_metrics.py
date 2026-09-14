"""Tests for the box-constrained KKT audit of a damage field."""
from __future__ import annotations

import pytest
import torch

from phast.solvers.damage_solver import damage_kkt_metrics


def test_exact_solution_reports_no_violation():
    d_prev = torch.tensor([0.0, 0.3, 0.5])
    d = torch.tensor([0.0, 0.6, 0.8])
    # Lower-active node carries a non-negative gradient; interior nodes are
    # stationary, so every projected norm must vanish.
    residual = torch.tensor([0.4, 0.0, 0.0])

    metrics = damage_kkt_metrics(residual, d, d_prev)

    assert metrics["kkt_projected_l2"] == pytest.approx(0.0)
    assert metrics["kkt_projected_linf"] == pytest.approx(0.0)
    assert metrics["kkt_feasible"] is True
    assert metrics["kkt_lower_active_count"] == 1
    assert metrics["kkt_interior_count"] == 2


def test_negative_gradient_at_a_lower_active_node_is_a_violation():
    d_prev = torch.tensor([0.25])
    d = torch.tensor([0.25])
    residual = torch.tensor([-0.75])

    metrics = damage_kkt_metrics(residual, d, d_prev)

    assert metrics["kkt_lower_active_count"] == 1
    assert metrics["kkt_lower_dual_violation_linf"] == pytest.approx(0.75)
    assert metrics["kkt_projected_linf"] == pytest.approx(0.75)


def test_positive_gradient_at_an_upper_active_node_is_a_violation():
    d_prev = torch.tensor([0.4])
    d = torch.tensor([1.0])
    residual = torch.tensor([0.5])

    metrics = damage_kkt_metrics(residual, d, d_prev)

    assert metrics["kkt_upper_active_count"] == 1
    assert metrics["kkt_upper_dual_violation_linf"] == pytest.approx(0.5)


def test_irreversibility_breach_is_reported_as_infeasible():
    d_prev = torch.tensor([0.6])
    d = torch.tensor([0.4])
    residual = torch.tensor([0.0])

    metrics = damage_kkt_metrics(residual, d, d_prev)

    assert metrics["kkt_feasible"] is False
    assert metrics["kkt_lower_feasibility_linf"] == pytest.approx(0.2)


def test_saturated_nodes_are_treated_as_fixed():
    d_prev = torch.tensor([1.0, 0.2])
    d = torch.tensor([1.0, 0.5])
    # The saturated node has coincident bounds, so its residual cannot
    # contribute to the stationarity measure.
    residual = torch.tensor([9.0, 0.0])

    metrics = damage_kkt_metrics(residual, d, d_prev)

    assert metrics["kkt_fixed_count"] == 1
    assert metrics["kkt_projected_linf"] == pytest.approx(0.0)


def test_explicit_dirichlet_nodes_are_excluded():
    d_prev = torch.tensor([0.1, 0.1])
    d = torch.tensor([0.5, 0.5])
    residual = torch.tensor([3.0, 0.0])
    fixed = torch.tensor([True, False])

    metrics = damage_kkt_metrics(residual, d, d_prev, fixed=fixed)

    assert metrics["kkt_fixed_count"] == 1
    assert metrics["kkt_projected_linf"] == pytest.approx(0.0)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="identical shapes"):
        damage_kkt_metrics(
            torch.zeros(3), torch.zeros(3), torch.zeros(2))


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        damage_kkt_metrics(
            torch.zeros(2), torch.zeros(2), torch.zeros(2), bound_atol=-1.0)
