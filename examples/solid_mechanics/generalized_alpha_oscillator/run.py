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
import os
import time
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from phast.time_integrators import gen_alpha_params, gen_alpha_step
from phast.visualization import write_visual_manifest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from _common import (
    copy_thumbnail,
    load_config,
    prepare_output_dir,
    write_diagnostic_setup_preview,
    write_manifest,
)

DEFAULT_CONFIG = {
    "schema_version": 1,
    "example": "solid_mechanics.generalized_alpha_oscillator",
    "time": {"n_steps": 100, "dt": 1.0e-3},
    "integrator": {"rho_inf_values": [1.0, 0.5]},
    "output": {"directory": "outputs"},
}


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


def run_example(config_path: str | Path | None = None):
    started = time.perf_counter()
    cfg = load_config(config_path or Path(__file__).with_name("config.yaml"), DEFAULT_CONFIG)
    out_dir = prepare_output_dir(__file__, cfg)
    n_steps = int(cfg["time"]["n_steps"])
    dt = float(cfg["time"]["dt"])  # ~ T_low / 1000; ~ T_high * 1; deep in HF regime

    rho_values = [float(v) for v in cfg["integrator"]["rho_inf_values"]]
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
    env_out = os.environ.get("PHAST_SOLID_MECH_OUTPUT_DIR")
    if env_out:
        out_dir = Path(env_out)
        out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "response.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    write_diagnostic_setup_preview(
        out_dir,
        title="Generalized-alpha oscillator diagnostic setup",
        config=cfg,
    )
    copy_thumbnail(out_dir)
    write_visual_manifest(
        out_dir,
        ["initial_conditions.png", "response.png", "thumbnail.png"],
        visual_scope="solid_mechanics_diagnostic",
    )

    print(f"plot: {out}  ({out.stat().st_size / 1024:.1f} kB)")
    csv_rows = []
    metrics = {}
    for rho in rho_values:
        E_lo, E_hi = results[rho]
        r_lo = float(E_lo[-1] / E_lo[0])
        r_hi = float(E_hi[-1] / E_hi[0])
        print(f"rho_inf={rho:.2f}  E_lo_final/E_lo_init={r_lo:.6f}  "
              f"E_hi_final/E_hi_init={r_hi:.6e}")
        metrics[f"rho_inf_{rho:g}"] = {
            "E_lo_final_over_initial": r_lo,
            "E_hi_final_over_initial": r_hi,
        }
        t = torch.arange(n_steps + 1, dtype=torch.float64) * dt
        for ti, elo, ehi in zip(t.tolist(), E_lo.tolist(), E_hi.tolist()):
            csv_rows.append([rho, ti, elo / float(E_lo[0]), ehi / float(E_hi[0])])

    csv_path = out_dir / "response.csv"
    np.savetxt(
        csv_path,
        np.array(csv_rows, dtype=float),
        delimiter=",",
        header="rho_inf,time,E_lo_over_initial,E_hi_over_initial",
        comments="",
    )
    write_manifest(
        out_dir,
        example="solid_mechanics.generalized_alpha_oscillator",
        command="python examples/solid_mechanics/generalized_alpha_oscillator/run.py",
        config=cfg,
        metrics=metrics,
        files=[
            "response.csv",
            "response.png",
            "initial_conditions.png",
            "thumbnail.png",
            "visual_manifest.json",
            "run_manifest.json",
        ],
        started_at=started,
    )
    return metrics


def main():
    run_example()


if __name__ == "__main__":
    main()
