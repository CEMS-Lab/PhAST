"""2-DOF dynamic oscillator demo for the Hulbert-Chung generalized-alpha
integrator (issue #102).

Two decoupled modes: a low-frequency mode (omega_low = 2 pi) we want to
preserve, and a high-frequency mode (omega_high = 2 pi * 1e3) that gen-alpha
with rho_inf = 0.5 should dissipate. With rho_inf = 1.0 both modes are
energy-conserving (Newmark-beta limit).

This is a synthetic 2-DOF demo; wiring gen-alpha into the production
ExplicitDynamicsSolver path (B1-B5 benchmarks) is a separate follow-up
because the existing solver is heavily entangled with the matrix-free PF
operator stack.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from phast.time_integrators import gen_alpha_params, gen_alpha_step


def _energy(u, v, K, M):
    # Diagonal K, M: kinetic + potential per mode.
    ke = 0.5 * M * v * v
    pe = 0.5 * K * u * u
    return ke + pe


def run(rho_inf: float, n_steps: int, dt: float):
    dtype = torch.float64
    # Diagonal (decoupled) 2-DOF: mode 0 = low freq, mode 1 = high freq.
    M = torch.tensor([1.0, 1.0], dtype=dtype)
    omega_lo = 2.0 * math.pi
    omega_hi = 2.0 * math.pi * 1.0e3
    K = torch.tensor([omega_lo ** 2, omega_hi ** 2], dtype=dtype)

    # Equal initial displacement in both modes; zero velocity.
    u = torch.tensor([1.0, 1.0], dtype=dtype)
    v = torch.zeros(2, dtype=dtype)
    a = -K * u / M
    f_ext = torch.zeros(2, dtype=dtype)

    am, af, beta, gamma = gen_alpha_params(rho_inf)

    E_hi = torch.empty(n_steps + 1, dtype=dtype)
    E_lo = torch.empty(n_steps + 1, dtype=dtype)
    E0 = _energy(u, v, K, M)
    E_lo[0] = E0[0]
    E_hi[0] = E0[1]

    for n in range(n_steps):
        u, v, a = gen_alpha_step(u, v, a, K, M, f_ext, dt, am, af, beta, gamma)
        E = _energy(u, v, K, M)
        E_lo[n + 1] = E[0]
        E_hi[n + 1] = E[1]

    return E_lo, E_hi


def main():
    n_steps = 100
    dt = 1.0e-3  # ~ T_low / 1000; ~ T_high * 1; deep in HF regime

    rho_values = [1.0, 0.5]
    results = {rho: run(rho, n_steps, dt) for rho in rho_values}

    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    t = torch.arange(n_steps + 1, dtype=torch.float64) * dt
    styles = {1.0: ("-", r"$\rho_\infty=1.0$ (Newmark, conserving)"),
              0.5: ("--", r"$\rho_\infty=0.5$ (Borden, dissipative)")}
    for rho, (E_lo, E_hi) in results.items():
        ls, lab = styles[rho]
        ax.plot(t.numpy(), (E_hi / E_hi[0]).numpy(), ls, label=lab, lw=1.4)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"high-freq mode energy $E_\mathrm{hi}/E_\mathrm{hi}(0)$")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).with_name("dynamic_oscillator_genalpha.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"plot: {out}  ({out.stat().st_size / 1024:.1f} kB)")
    for rho in rho_values:
        E_lo, E_hi = results[rho]
        r_lo = float(E_lo[-1] / E_lo[0])
        r_hi = float(E_hi[-1] / E_hi[0])
        print(f"rho_inf={rho:.2f}  E_lo_final/E_lo_init={r_lo:.6f}  "
              f"E_hi_final/E_hi_init={r_hi:.6e}")


if __name__ == "__main__":
    main()
