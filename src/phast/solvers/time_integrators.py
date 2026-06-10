"""Hulbert-Chung generalized-alpha time integrator (issue #102).

Function-style API kept independent of FEMOperators/ExplicitDynamics so the
existing velocity-Verlet path stays bit-exact for current benchmarks. Wiring
into B1/B5 dynamic benchmarks is a separate follow-up.

Reference: Chung & Hulbert, J. Appl. Mech. 60 (1993) 371-375
(https://doi.org/10.1115/1.2900803); Borden 2012 §3.2.
"""

from __future__ import annotations

import torch


def gen_alpha_params(rho_inf: float):
    """Spectral-radius parameterisation (Borden 2012 eqs 95-97).

    rho_inf=0.5 is the Borden 2012 default (high-frequency dissipation);
    rho_inf=1.0 gives alpha_m = alpha_f = 0.5, beta = 1/4, gamma = 1/2
    (zero numerical dissipation, second-order accurate).
    """
    if not (0.0 <= rho_inf <= 1.0):
        raise ValueError(f"rho_inf must lie in [0, 1]; got {rho_inf}")
    alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
    alpha_f = rho_inf / (rho_inf + 1.0)
    beta = 0.25 * (1.0 - alpha_m + alpha_f) ** 2
    gamma = 0.5 - alpha_m + alpha_f
    return alpha_m, alpha_f, beta, gamma


def gen_alpha_step(u, v, a, K, M, f_ext, dt,
                   alpha_m: float, alpha_f: float,
                   beta: float, gamma: float):
    """One Hulbert-Chung generalized-alpha step for M*a + K*u = f_ext.

    Solves the implicit residual::

        (1 - alpha_m) M a_{n+1} + alpha_m M a_n
          + (1 - alpha_f) K u_{n+1} + alpha_f  K u_n
          = f_ext_{n+1-alpha_f}

    together with the Newmark-beta predictor relations for u_{n+1}, v_{n+1}.

    K, M may be dense matrices, vectors (lumped/diagonal), or scalars.
    Autograd-safe: no in-place ops, no .detach().
    """
    def _apply(A, x):
        if A.dim() == 0 or A.dim() == 1:
            return A * x
        return A @ x

    # Newmark predictor (no a_{n+1} contribution yet).
    u_pred = u + dt * v + 0.5 * dt * dt * (1.0 - 2.0 * beta) * a
    v_pred = v + dt * (1.0 - gamma) * a

    # Effective system for a_{n+1}:
    #   [(1 - alpha_m) M + (1 - alpha_f) beta dt^2 K] a_{n+1}
    #   = f_ext - alpha_m M a_n - (1 - alpha_f) K u_pred - alpha_f K u
    rhs = (f_ext
           - alpha_m * _apply(M, a)
           - (1.0 - alpha_f) * _apply(K, u_pred)
           - alpha_f * _apply(K, u))

    if M.dim() <= 1 and K.dim() <= 1:
        lhs_diag = (1.0 - alpha_m) * M + (1.0 - alpha_f) * beta * dt * dt * K
        a_new = rhs / lhs_diag
    else:
        n = rhs.shape[-1]
        eye = torch.eye(n, dtype=rhs.dtype, device=rhs.device)
        M_mat = M if M.dim() == 2 else (M * eye if M.dim() == 1 else M * eye)
        K_mat = K if K.dim() == 2 else (K * eye if K.dim() == 1 else K * eye)
        lhs = (1.0 - alpha_m) * M_mat + (1.0 - alpha_f) * beta * dt * dt * K_mat
        a_new = torch.linalg.solve(lhs, rhs)

    u_new = u_pred + beta * dt * dt * a_new
    v_new = v_pred + gamma * dt * a_new
    return u_new, v_new, a_new


def newmark_step(u, v, a, K, M, f_ext, dt,
                 beta: float = 0.25, gamma: float = 0.5):
    """Implicit Newmark-beta reference (rho_inf=1 limit of gen-alpha)."""
    return gen_alpha_step(u, v, a, K, M, f_ext, dt,
                          alpha_m=0.0, alpha_f=0.0, beta=beta, gamma=gamma)
