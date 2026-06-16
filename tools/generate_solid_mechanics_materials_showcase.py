"""Generate the README solid-mechanics material-kernel showcase panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("assets/solid_mechanics_materials.png")


def linear_elastic_curve(
    *,
    modulus_mpa: float = 210_000.0,
    max_mstrain: float = 4.5,
    points: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Return engineering stress for a small-strain linear elastic bar."""
    strain_mstrain = np.linspace(0.0, max_mstrain, points)
    stress = modulus_mpa * strain_mstrain * 1.0e-3
    return strain_mstrain, stress


def incompressible_neohookean_nominal_curve(
    *,
    max_strain: float = 0.80,
    points: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized nominal stress P/mu and initial tangent 3 epsilon."""
    strain = np.linspace(0.0, max_strain, points)
    stretch = 1.0 + strain
    nominal_over_mu = stretch - stretch ** -2
    initial_tangent_over_mu = 3.0 * strain
    return strain, nominal_over_mu, initial_tangent_over_mu


def j2_isotropic_hardening_curve(
    *,
    modulus_mpa: float = 200_000.0,
    yield_mpa: float = 250.0,
    hardening_mpa: float = 5_000.0,
    max_mstrain: float = 4.5,
    points: int = 240,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a 1D monotonic J2-style elastic-yield-hardening curve."""
    strain = np.linspace(0.0, max_mstrain * 1.0e-3, points)
    yield_strain = yield_mpa / modulus_mpa
    stress = np.where(
        strain <= yield_strain,
        modulus_mpa * strain,
        yield_mpa + hardening_mpa * (strain - yield_strain),
    )
    hardening_reference = yield_mpa + hardening_mpa * (strain - yield_strain)
    return strain * 1.0e3, stress, hardening_reference, yield_strain * 1.0e3


def generate(out: Path = OUT) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 21,
            "axes.labelsize": 17,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=120)
    fig.suptitle("Solid-mechanics material kernels", fontsize=23, y=0.98)

    eps_lin, sigma_lin = linear_elastic_curve()
    ax = axes[0]
    ax.plot(eps_lin, sigma_lin, color="#2563eb", lw=2.8)
    ax.set_title("Linear elastic")
    ax.set_xlabel(r"$\varepsilon_{xx}$ (mstrain)")
    ax.set_ylabel(r"$\sigma_{xx}$ (MPa)")
    ax.grid(True, alpha=0.32)

    eps_nh, p_over_mu, tangent = incompressible_neohookean_nominal_curve()
    ax = axes[1]
    ax.plot(eps_nh, p_over_mu, color="#16a34a", lw=2.8, label="neo-Hookean")
    ax.plot(eps_nh, tangent, color="#6b7280", lw=2.0, ls="--", label="initial tangent")
    ax.fill_between(eps_nh, p_over_mu, tangent, color="#16a34a", alpha=0.10)
    ax.set_title("Incompressible neo-Hookean")
    ax.set_xlabel(r"stretch strain $\lambda - 1$")
    ax.set_ylabel(r"nominal stress $P/\mu$")
    ax.grid(True, alpha=0.32)
    ax.legend(frameon=False, loc="upper left")

    eps_j2, sigma_j2, hardening, yield_mstrain = j2_isotropic_hardening_curve()
    ax = axes[2]
    ax.plot(eps_j2, sigma_j2, color="#dc2626", lw=2.8, label="von Mises")
    active = eps_j2 >= yield_mstrain
    ax.plot(
        eps_j2[active],
        hardening[active],
        color="#111827",
        lw=2.0,
        ls="--",
        label="yield + hardening",
    )
    ax.scatter([0.0], [0.0], s=22, color="#dc2626", zorder=4)
    ax.axvline(yield_mstrain, color="#9ca3af", lw=1.2, ls=":", alpha=0.9)
    ax.set_title("J2 plasticity")
    ax.set_xlabel(r"$\varepsilon_{xx}$ (mstrain)")
    ax.set_ylabel(r"$\sigma_{eq}$ (MPa)")
    ax.grid(True, alpha=0.32)
    ax.legend(frameon=False, loc="lower right")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    generate()
