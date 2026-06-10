"""Anisotropic length-scale tensor for phase-field damage (#258).

Provides L(x) = l_perp^2 * n (x) n + l_par^2 * t (x) t, where (n, t) is an
orthonormal basis with n the local crack normal. Scaffold only --
not wired into damage_solver.py.

The standard phase-field damage gradient term ``l0^2 * grad(d) . grad(d)``
is generalised to the anisotropic form ``(L . grad(d)) . grad(d)`` where
``L`` is a 2x2 SPD tensor that may vary per element.
"""

from __future__ import annotations

import torch


def isotropic_L(l0: float | torch.Tensor) -> torch.Tensor:
    """Return the 2x2 isotropic length-scale tensor ``l0^2 * I``.

    Args:
        l0: scalar (Python float or 0-d tensor) length scale.

    Returns:
        (2, 2) float64 SPD tensor.
    """
    if not torch.is_tensor(l0):
        l0 = torch.tensor(l0, dtype=torch.float64)
    return (l0 ** 2) * torch.eye(2, dtype=torch.float64)


def anisotropic_L(
    l_perp: float | torch.Tensor,
    l_par: float | torch.Tensor,
    normal: torch.Tensor,
) -> torch.Tensor:
    """Build ``L = l_perp^2 * n (x) n + l_par^2 * t (x) t`` for a 2D normal.

    Args:
        l_perp: scalar length scale perpendicular to the crack (along ``n``).
        l_par:  scalar length scale parallel to the crack (along ``t``).
        normal: shape (2,) crack normal (will be normalised internally).

    Returns:
        (2, 2) SPD tensor.
    """
    n = normal / torch.linalg.norm(normal)
    t = torch.stack([-n[1], n[0]])  # 90deg rotation -> tangent
    return (l_perp ** 2) * torch.outer(n, n) + (l_par ** 2) * torch.outer(t, t)


def field_anisotropic_L(
    l_perp_field: torch.Tensor,
    l_par_field: torch.Tensor,
    normal_field: torch.Tensor,
) -> torch.Tensor:
    """Per-element anisotropic ``L``.

    Args:
        l_perp_field: (n_elems,) -- ``l_perp`` at each element.
        l_par_field:  (n_elems,) -- ``l_par`` at each element.
        normal_field: (n_elems, 2) -- crack normal at each element
            (will be normalised internally).

    Returns:
        (n_elems, 2, 2) SPD tensor field.
    """
    norms = torch.linalg.norm(normal_field, dim=-1, keepdim=True).clamp(min=1e-12)
    n = normal_field / norms                              # (n_elems, 2)
    t = torch.stack([-n[..., 1], n[..., 0]], dim=-1)      # (n_elems, 2)
    nn = torch.einsum("ei,ej->eij", n, n)
    tt = torch.einsum("ei,ej->eij", t, t)
    return (l_perp_field ** 2)[:, None, None] * nn + (l_par_field ** 2)[:, None, None] * tt


def estimate_crack_normal_from_damage_gradient(
    grad_d: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Heuristic crack normal: ``n ~ grad(d) / |grad(d)|``.

    Damage rises perpendicular to the crack, so its spatial gradient
    points along the crack normal.

    Args:
        grad_d: (n_elems, 2) damage gradient at each element.
        eps:    floor on |grad(d)| to avoid divide-by-zero.

    Returns:
        (n_elems, 2) unit normal (zero-gradient elements get an
        arbitrary unit direction via the eps floor).
    """
    norms = torch.linalg.norm(grad_d, dim=-1, keepdim=True).clamp(min=eps)
    return grad_d / norms


def gradient_energy_density(
    grad_d: torch.Tensor,
    L: torch.Tensor,
) -> torch.Tensor:
    """Compute ``(grad d) . L . (grad d)`` -- gradient-energy integrand.

    Args:
        grad_d: (n_elems, 2) damage gradient.
        L:      (2, 2) constant tensor or (n_elems, 2, 2) field.

    Returns:
        (n_elems,) scalar density.
    """
    if L.dim() == 2:
        L = L.expand(grad_d.shape[0], -1, -1)
    return torch.einsum("ei,eij,ej->e", grad_d, L, grad_d)
