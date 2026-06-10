"""Standalone J2 plasticity material-point demo (#110/#242).

This is intentionally not a coupled phase-field or global FEM solve. It shows
the customer-facing constitutive kernel that currently exists: rate-independent
von Mises plasticity with linear isotropic hardening, updated by radial return.
"""
from __future__ import annotations

import torch

from phast.material import Material
from phast.plasticity import J2Plasticity, J2State
from phast.plasticity.j2_vonmises import (
    _stress_dev_norm,
    _stress_deviator_voigt6,
)


def von_mises(stress: torch.Tensor) -> torch.Tensor:
    """Return von Mises equivalent stress from Voigt-6 stress."""
    return torch.sqrt(torch.tensor(1.5, dtype=stress.dtype)) * _stress_dev_norm(
        _stress_deviator_voigt6(stress)
    )


def main() -> None:
    torch.set_default_dtype(torch.float64)

    sigma_y0 = 250.0
    hardening = 5_000.0
    mat = Material(
        E=210_000.0,
        nu=0.3,
        plasticity_model="j2_isotropic",
        yield_stress=sigma_y0,
        hardening_modulus=hardening,
        hardening_type="linear_iso",
        plane_stress=True,
    )
    kernel = J2Plasticity(mat, plane_stress=True)

    state = J2State.zeros((1,), dtype=torch.float64)
    strain = torch.zeros((1, 6), dtype=torch.float64)

    rows = []
    for step in range(1, 31):
        strain_next = strain.clone()
        strain_next[..., 0] = step * 1.5e-4

        stress, plastic_strain, eps_p_eq = kernel.step(
            strain,
            strain_next,
            state.stress,
            state.plastic_strain,
            state.eps_p_eq,
        )
        state = J2State(stress, plastic_strain, eps_p_eq)
        strain = strain_next

        vm = float(von_mises(stress)[0])
        eqp = float(eps_p_eq[0])
        yield_current = sigma_y0 + hardening * eqp
        rows.append((step, float(strain[0, 0]), float(stress[0, 0]), vm, eqp,
                     yield_current))

    print("J2 plasticity bar demo: plane-stress uniaxial strain ramp")
    print(f"{'step':>4} {'eps_xx':>10} {'sigma_xx [MPa]':>16} "
          f"{'vm [MPa]':>10} {'eps_p_eq':>11} {'yield [MPa]':>12}")
    for step, eps_xx, sigma_xx, vm, eqp, yield_current in rows[::5]:
        print(f"{step:4d} {eps_xx:10.4e} {sigma_xx:16.3f} "
              f"{vm:10.3f} {eqp:11.4e} {yield_current:12.3f}")

    last = rows[-1]
    plastic_steps = sum(1 for row in rows if row[4] > 0.0)
    vm_error = abs(last[3] - last[5])
    print(f"plastic steps: {plastic_steps}/{len(rows)}")
    print(f"final eps_p_eq: {last[4]:.6e}")
    print(f"final vm-yield residual: {vm_error:.3e} MPa")

    if plastic_steps == 0 or vm_error > 1e-5:
        raise SystemExit("J2 consistency check failed")


if __name__ == "__main__":
    main()
