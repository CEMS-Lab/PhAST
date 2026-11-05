"""Wu PF-CZM relations (degradation derivatives, ``a1`` and the admissible length scale) used by the 3D voxel solver."""
import math
import torch


PFCZM_C_ALPHA = math.pi


PFCZM_SOFTENING_LAWS = ('linear', 'exponential')


PFCZM_SOFTENING_EXPONENTS = {'linear': 2, 'exponential': 2.5}


def pfczm_a1(E, Gc, ell, f_t, eta=0.0):
    """Wu PF-CZM parameter ``a1 = 4 E Gc / (pi ell f_t^2 (1 - eta))``.

    ``E``, ``Gc`` and ``f_t`` may be floats or tensors (one value per element);
    ``ell`` and ``eta`` are scalars.  The ``1/(1 - eta)`` factor keeps the
    damage onset at ``H = f_t^2/(2E)`` with the residual-stiffness blend of
    :func:`pfczm_degradation_derivatives` (``g'(0) = -(1 - eta) a1``).
    """
    if isinstance(f_t, torch.Tensor):
        if not bool((f_t > 0.0).all()):
            raise ValueError("PFCZM a1 requires f_t > 0")
    elif not f_t > 0.0:
        raise ValueError("PFCZM a1 requires f_t > 0")
    scale = max(1.0 - float(eta), 1.0e-30)
    return 4.0 * E * Gc / (
        math.pi * ell * f_t ** 2 * scale)


def pfczm_max_strength(E, Gc, ell, bound, eta=0.0):
    """Largest strength with ``pfczm_a1(E, Gc, ell, f_t, eta) >= bound``.

    ``bound`` is :func:`pfczm_onset_convexity_bound`; the result is
    ``sqrt(4 E Gc / (pi ell (1 - eta) bound))``, from :func:`pfczm_a1` at
    ``f_t = 1`` (``a1`` scales with ``f_t^-2``).
    """
    return (pfczm_a1(E, Gc, ell, 1.0, eta) / bound) ** 0.5


def pfczm_max_length_scale(E, Gc, f_t, bound, eta=0.0):
    """Largest length scale with ``pfczm_a1(E, Gc, ell, f_t, eta) >= bound``.

    The result is ``4 E Gc / (pi f_t^2 (1 - eta) bound)``, from
    :func:`pfczm_a1` at ``ell = 1`` (``a1`` scales with ``ell^-1``); with
    linear softening (bound 3/2) it is ``8 l_ch / (3 pi)``,
    ``l_ch = E Gc / f_t^2``.
    """
    return pfczm_a1(E, Gc, 1.0, f_t, eta) / bound


def pfczm_softening_parameters(softening: str) -> tuple[float, float]:
    """Return ``(a2, a3)`` of a Wu softening law (``'linear'`` or ``'exponential'``)."""
    if softening == 'linear':
        return -0.5, 0.0
    if softening == 'exponential':
        # Wu-Nguyen exponential softening parameter with a3=0.
        return 2.0 ** (5.0 / 3.0) - 3.0, 0.0
    raise ValueError(
        f"PF-CZM softening must be one of {PFCZM_SOFTENING_LAWS}, got {softening!r}")


def pfczm_onset_convexity_bound(a2: float, p: float = 2) -> float:
    """Smallest ``a1`` with ``g''(0) >= 0``: ``g''(0) = 2 (1 - eta) a1 (a1 - p - a2)``.

    With linear softening (``p = 2``) the bound is ``a1 >= 3/2``, i.e.
    ``ell <= 8 l_ch / (3 pi)`` with the Irwin length ``l_ch = E Gc / f_t^2``;
    with exponential softening (``p = 2.5``) it is ``2^(5/3) - 1/2 = 2.675``.
    """
    return float(p) + float(a2)


def pfczm_degradation_derivatives(
        d: torch.Tensor,
        a1,
        a2: float,
        a3: float,
        p: float = 2,
        eta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Wu PF-CZM rational degradation g, g' and g''.

    The implementation follows the unified PF-CZM family with
    A=(1-d)^p and Q=a1*d*(1+a2*d+a2*a3*d^2). The residual-stiffness
    blend keeps g(0)=1 and g(1)=eta, matching the other degradation
    families in this solver.  ``a1`` is a float or a tensor; a tensor with
    fewer dimensions than ``d`` is broadcast over the trailing dimensions
    of ``d`` (one value per element, quadrature points last).  ``p >= 2``
    may be fractional (``p = 2.5`` in Wu's exponential softening); an
    integral ``p`` is evaluated with integer powers.
    """
    if isinstance(a1, torch.Tensor):
        a1_eff = a1.to(dtype=d.dtype, device=d.device)
        while a1_eff.ndim < d.ndim:
            a1_eff = a1_eff.unsqueeze(-1)
    else:
        a1_eff = torch.as_tensor(a1, dtype=d.dtype, device=d.device)
    p = int(p) if float(p).is_integer() else float(p)

    omd = torch.clamp(1.0 - d, min=0.0)
    A = omd ** p
    Ap = -float(p) * omd ** (p - 1)
    if p == 2:
        App = torch.full_like(d, 2.0)
    else:
        App = float(p * (p - 1)) * omd ** (p - 2)  # finite at d = 1 for p >= 2

    Q = a1_eff * d * (1.0 + a2 * d + a2 * a3 * d * d)
    Qp = a1_eff * (1.0 + 2.0 * a2 * d + 3.0 * a2 * a3 * d * d)
    Qpp = a1_eff * (2.0 * a2 + 6.0 * a2 * a3 * d)

    S = torch.clamp(A + Q, min=1.0e-30)
    N = Ap * Q - A * Qp
    Np = App * Q - A * Qpp
    ratio = A / S
    ratio_p = N / (S * S)
    ratio_pp = Np / (S * S) - 2.0 * N * (Ap + Qp) / (S * S * S)
    scale = 1.0 - eta
    return (
        scale * ratio + eta,
        scale * ratio_p,
        scale * ratio_pp,
    )
