"""Tests for Hulbert-Chung generalized-alpha integrator (issue #102)."""
import math
import os
import sys

import torch

import importlib.util as _ilu

_here = os.path.dirname(os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location(
    "_gen_alpha_under_test",
    os.path.join(os.path.dirname(_here), "src", "phast", "time_integrators.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
gen_alpha_params = _mod.gen_alpha_params
gen_alpha_step = _mod.gen_alpha_step


def _scalar_oscillator_step_newmark(u, v, a, K, M, f_ext, dt,
                                    beta=0.25, gamma=0.5):
    """Reference implicit Newmark for a scalar 1-DOF oscillator."""
    u_pred = u + dt * v + 0.5 * dt * dt * (1.0 - 2.0 * beta) * a
    v_pred = v + dt * (1.0 - gamma) * a
    a_new = (f_ext - K * u_pred) / (M + beta * dt * dt * K)
    u_new = u_pred + beta * dt * dt * a_new
    v_new = v_pred + gamma * dt * a_new
    return u_new, v_new, a_new


def test_reduces_to_newmark_at_rho1():
    torch.manual_seed(0)
    M = torch.tensor(1.0, dtype=torch.float64)
    K = torch.tensor(4.0 * math.pi ** 2, dtype=torch.float64)
    dt = 0.01
    u = torch.tensor(1.0, dtype=torch.float64)
    v = torch.tensor(0.0, dtype=torch.float64)
    a = (-K * u) / M
    f_ext = torch.tensor(0.0, dtype=torch.float64)

    # alpha_m=alpha_f=0 -> classical implicit Newmark (beta=1/4, gamma=1/2).
    u_a, v_a, a_a = u.clone(), v.clone(), a.clone()
    u_b, v_b, a_b = u.clone(), v.clone(), a.clone()
    for _ in range(5):
        u_a, v_a, a_a = gen_alpha_step(
            u_a, v_a, a_a, K, M, f_ext, dt,
            alpha_m=0.0, alpha_f=0.0, beta=0.25, gamma=0.5)
        u_b, v_b, a_b = _scalar_oscillator_step_newmark(
            u_b, v_b, a_b, K, M, f_ext, dt)
    assert torch.allclose(u_a, u_b, atol=1e-12, rtol=0)
    assert torch.allclose(v_a, v_b, atol=1e-12, rtol=0)
    assert torch.allclose(a_a, a_b, atol=1e-12, rtol=0)
    # And rho_inf=1.0 spectral params yield zero high-freq dissipation
    # (energy conservation to machine precision on a resolved oscillator).
    am, af, beta, gamma = gen_alpha_params(1.0)
    u1 = torch.tensor(1.0, dtype=torch.float64)
    v1 = torch.tensor(0.0, dtype=torch.float64)
    a1 = (-K * u1) / M
    E0 = 0.5 * M * v1 * v1 + 0.5 * K * u1 * u1
    for _ in range(50):
        u1, v1, a1 = gen_alpha_step(u1, v1, a1, K, M, f_ext, dt,
                                    am, af, beta, gamma)
    E = 0.5 * M * v1 * v1 + 0.5 * K * u1 * u1
    assert abs(float(E / E0) - 1.0) < 1e-3


def _energy_after(omega, rho_inf, n_steps, dt):
    M = torch.tensor(1.0, dtype=torch.float64)
    K = torch.tensor(omega ** 2, dtype=torch.float64)
    u = torch.tensor(1.0, dtype=torch.float64)
    v = torch.tensor(0.0, dtype=torch.float64)
    a = (-K * u) / M
    f_ext = torch.tensor(0.0, dtype=torch.float64)
    am, af, beta, gamma = gen_alpha_params(rho_inf)
    E0 = 0.5 * M * v * v + 0.5 * K * u * u
    for _ in range(n_steps):
        u, v, a = gen_alpha_step(u, v, a, K, M, f_ext, dt, am, af, beta, gamma)
    E = 0.5 * M * v * v + 0.5 * K * u * u
    return float(E / E0)


def test_dissipation_at_rho_half():
    # Resolve low-frequency mode well; dt*omega_high large to be in
    # the "high-frequency" regime that gen-alpha is designed to damp.
    ratio_low = _energy_after(omega=1.0, rho_inf=0.5, n_steps=100, dt=0.01)
    ratio_high = _energy_after(omega=1000.0, rho_inf=0.5, n_steps=100, dt=0.01)
    assert ratio_low > 0.99, f"low-freq energy decayed: {ratio_low}"
    assert ratio_high < 0.5, f"high-freq energy not dissipated: {ratio_high}"


def test_autograd_through_step():
    M = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    K = torch.tensor(4.0, dtype=torch.float64)
    dt = 0.05
    u = torch.tensor(1.0, dtype=torch.float64)
    v = torch.tensor(0.0, dtype=torch.float64)
    a = torch.tensor(-4.0, dtype=torch.float64)
    f_ext = torch.tensor(0.0, dtype=torch.float64)
    am, af, beta, gamma = gen_alpha_params(0.5)
    for _ in range(3):
        u, v, a = gen_alpha_step(u, v, a, K, M, f_ext, dt, am, af, beta, gamma)
    loss = u.sum()
    loss.backward()
    assert M.grad is not None
    assert torch.isfinite(M.grad).all()
    assert M.grad.abs().item() > 0.0


def test_matrix_free_generalized_alpha_matches_dense_linear_step():
    from phast.mechanics_solver import GeneralizedAlphaDynamics

    class LinearFem:
        def __init__(self):
            self.dt_cfl = 1.0
            self.M_vec = torch.ones(2, dtype=torch.float64)
            self.K = torch.tensor([4.0, 9.0], dtype=torch.float64)

        def internal_force(self, u, d):
            return self.K.view(1, 2) * u

        def stiffness_diagonal(self, d=None):
            return self.K.view(1, 2)

    fem = LinearFem()
    dt = 0.05
    u = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    v = torch.zeros_like(u)
    a = -fem.K.view(1, 2) * u
    d = torch.zeros(1, dtype=torch.float64)
    f_ext = torch.zeros_like(u)
    bc_mask = torch.zeros_like(u, dtype=torch.bool)
    bc_vals = torch.zeros_like(u)

    solver = GeneralizedAlphaDynamics(
        fem, dt=dt, rho_inf=1.0, newton_tol=1e-12, cg_tol=1e-12)
    u_g, v_g, a_g = solver.step(u, v, a, d, f_ext, bc_mask, bc_vals)

    am, af, beta, gamma = gen_alpha_params(1.0)
    u_ref, v_ref, a_ref = gen_alpha_step(
        u.reshape(-1), v.reshape(-1), a.reshape(-1),
        fem.K, fem.M_vec, f_ext.reshape(-1), dt, am, af, beta, gamma)

    assert solver.last_converged
    assert torch.allclose(u_g.reshape(-1), u_ref, atol=1e-11, rtol=1e-11)
    assert torch.allclose(v_g.reshape(-1), v_ref, atol=1e-11, rtol=1e-11)
    assert torch.allclose(a_g.reshape(-1), a_ref, atol=1e-11, rtol=1e-11)
