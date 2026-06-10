"""Tests for anisotropic_length_scale.py (#258 scaffold)."""

from __future__ import annotations

import torch

from phast.research.anisotropic_length_scale import (
    anisotropic_L,
    estimate_crack_normal_from_damage_gradient,
    field_anisotropic_L,
    gradient_energy_density,
    isotropic_L,
)


def test_isotropic_reduces_to_scalar():
    L = isotropic_L(0.5)
    expected = 0.25 * torch.eye(2, dtype=torch.float64)
    assert torch.allclose(L, expected, atol=1e-15)

    grad = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    density = gradient_energy_density(grad, L)
    expected_density = (0.5 ** 2) * (3.0 ** 2 + 4.0 ** 2)
    assert torch.allclose(density, torch.tensor([expected_density], dtype=torch.float64), atol=1e-12)


def test_anisotropic_normal_only():
    # l_par = 0 -> L acts only along the normal.
    n = torch.tensor([1.0, 0.0], dtype=torch.float64)
    L = anisotropic_L(l_perp=0.5, l_par=0.0, normal=n)

    # grad parallel to tangent -> density 0.
    grad_t = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    d_t = gradient_energy_density(grad_t, L)
    assert torch.allclose(d_t, torch.zeros(1, dtype=torch.float64), atol=1e-15)

    # grad parallel to normal -> density l_perp^2 * |grad|^2.
    grad_n = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    d_n = gradient_energy_density(grad_n, L)
    assert torch.allclose(d_n, torch.tensor([0.25 * 4.0], dtype=torch.float64), atol=1e-15)


def test_field_anisotropic_per_element():
    normals = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    l_perp = torch.tensor([0.5, 0.3, 0.7], dtype=torch.float64)
    l_par = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64)

    L_field = field_anisotropic_L(l_perp, l_par, normals)
    assert L_field.shape == (3, 2, 2)
    for i in range(3):
        L_i = anisotropic_L(l_perp[i], l_par[i], normals[i])
        assert torch.allclose(L_field[i], L_i, atol=1e-15)


def test_estimate_normal_from_grad():
    grad_d = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    n = estimate_crack_normal_from_damage_gradient(grad_d)
    expected = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0 / 2 ** 0.5, 1.0 / 2 ** 0.5]],
        dtype=torch.float64,
    )
    assert torch.allclose(n, expected, atol=1e-12)
    # All normals must be unit length.
    assert torch.allclose(torch.linalg.norm(n, dim=-1), torch.ones(3, dtype=torch.float64), atol=1e-12)


def test_gradient_energy_isotropic_matches_scalar():
    torch.manual_seed(0)
    l0 = 0.37
    L = isotropic_L(l0)
    grad = torch.randn(50, 2, dtype=torch.float64)
    density = gradient_energy_density(grad, L)
    expected = (l0 ** 2) * (grad ** 2).sum(dim=-1)
    assert torch.allclose(density, expected, atol=1e-12)


def test_autograd_through_L():
    l_perp = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    l_par = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    normal = torch.tensor([1.0, 0.5], dtype=torch.float64)

    L = anisotropic_L(l_perp, l_par, normal)
    grad = torch.tensor([[0.7, -0.3], [0.2, 0.9]], dtype=torch.float64)
    energy = gradient_energy_density(grad, L).sum()
    energy.backward()

    assert l_perp.grad is not None and torch.isfinite(l_perp.grad)
    assert l_par.grad is not None and torch.isfinite(l_par.grad)
    # Both gradients should be non-zero for this generic input.
    assert l_perp.grad.abs() > 0
    assert l_par.grad.abs() > 0
