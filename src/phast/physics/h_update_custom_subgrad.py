"""
Smooth-subgradient custom autograd Function for the H-history update.

Background
----------
The phase-field history variable update,

    H_new = max(H_old, psi_plus),

uses a hard ``torch.maximum``. Forward is exact. Its backward, however,
routes the upstream gradient entirely to whichever operand is the larger
one at each quadrature point. In inverse problems, when ``H_old`` already
dominates ``psi_plus`` everywhere across an entire timestep — which is
typical once damage has been driven by the time-loop's own history —
the gradient pathway into ``psi_plus`` (and therefore into the strain,
the displacement, and ultimately the elastic parameters) is severed.

This module provides ``custom_subgrad(a, b, scale=1e10)``, a custom
``torch.autograd.Function`` that:

- **Forward**: returns ``torch.maximum(a, b)`` *exactly*. There is no
  smoothing of the forward output, so the time-stepping trajectory is
  byte-identical to the production solver.
- **Backward**: routes the upstream gradient through a sigmoid weighting,
  so both branches receive a non-zero contribution. With
  ``w_a = sigmoid((a - b) * scale)``, ``w_b = sigmoid((b - a) * scale)``
  and ``w_a + w_b == 1`` for every entry,

      grad_a = grad_out * w_a
      grad_b = grad_out * w_b

  At ``scale = 1e10`` the weighting matches the hard-max indicator to
  machine precision whenever ``|a - b| > 1e-10``, but at the equality
  ridge ``a == b`` the weights tie to ``0.5/0.5`` instead of dropping to
  zero, and the equality is *smoothly* crossed. This is what unblocks
  gradient flow to ``psi_plus`` for the inverse problem.

This is the headline mechanism for paper 3 contribution C4 (closes #362).
"""

from __future__ import annotations

import torch


class _MaxSmoothBackward(torch.autograd.Function):
    """Custom autograd op: forward = hard max, backward = sigmoid-weighted."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
        # Save the actual operand tensors and the scale (a Python float, not
        # part of the saved-tensor graph). The forward is the *exact* hard
        # maximum — no smoothing, no bias.
        ctx.save_for_backward(a, b)
        ctx.scale = float(scale)
        return torch.maximum(a, b)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        a, b = ctx.saved_tensors
        scale = ctx.scale
        # sigmoid((a - b) * scale) approaches 1 where a > b, 0 where a < b,
        # and 0.5 at the ridge. The two weights sum to 1 by construction,
        # so the backward is a smooth partition of grad_out across the two
        # branches — including a strictly non-zero share to the losing
        # branch, which is the property the H-update inverse problem needs.
        w_a = torch.sigmoid((a - b) * scale)
        w_b = torch.sigmoid((b - a) * scale)
        grad_a = grad_out * w_a
        grad_b = grad_out * w_b
        # No gradient w.r.t. ``scale`` (it is a hyperparameter, not a tensor
        # leaf), hence the trailing ``None``.
        return grad_a, grad_b, None


def custom_subgrad(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1e10,
) -> torch.Tensor:
    """Forward-exact ``maximum(a, b)`` with smooth sigmoid backward.

    Parameters
    ----------
    a, b
        Operands. Must be broadcastable; in practice both are shape
        ``(n_qp,)`` for the H-update.
    scale
        Sharpness of the sigmoid in the backward pass. ``1e10`` (default)
        matches the hard-max indicator to better than 1e-9 wherever
        ``|a - b| > 1e-9``, and only modifies behaviour in a vanishing
        neighbourhood of the equality ridge. Lower values (e.g. ``10.0``)
        are useful for finite-difference ``gradcheck`` tests where the FD
        step would otherwise sit entirely inside the ridge.

    Returns
    -------
    torch.Tensor
        ``torch.maximum(a, b)``, byte-identical to the hard reference.
    """
    return _MaxSmoothBackward.apply(a, b, scale)


__all__ = ["custom_subgrad", "_MaxSmoothBackward"]
