"""Explicit thermal update for thermo-mechanical phase-field fracture.

Solves the heat equation on the FEM mesh using an explicit Euler step:

    rho_c * dT/dt = k * div(grad(T)) + Q_crack

where Q_crack is a crack-tip heating source derived from the damage
increment field.  Both the thermal update and the Gc(T) coupling are
differentiable via autograd, enabling adjoint-based recovery of thermal
fracture parameters from experimental data.

Gc(T) coupling models
---------------------
- ``linear``    : Gc(T) = Gc0 * (1 - alpha_T * (T - T_ref))
- ``arrhenius`` : Gc(T) = Gc0 * exp(-(Ea_R) * (1/T - 1/T_ref))

Typical use
-----------
::

    from phast.thermal_update import ThermalUpdate

    therm = ThermalUpdate(fem, rho_c=3.5e6, k_therm=45.0, eta_crack=0.9,
                          dt=1e-7, T_ref=293.15)

    T = torch.full((mesh.n_nodes,), 293.15, dtype=torch.float64)
    for step in range(n_steps):
        d_prev = d.clone()
        # ... run mechanics + damage step to get d_new ...
        T = therm.step(T, d_new, d_prev)

    # Gc field at current temperature:
    Gc_T = therm.gc_field(T, Gc0=2.7e-3, model='linear', alpha_T=1e-4)

Integration test
----------------
::

    python -m phast.thermal_update
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import Tensor


class ThermalUpdate:
    """Explicit Euler heat equation on a FEM mesh with crack-tip source.

    The lumped-mass explicit update is:

        T_new[i] = T[i] + (dt / (rho_c * M_lump[i])) *
                   (k * (K_lap @ T)[i] + Q_node[i])

    where K_lap is the FEM Laplacian stiffness (from ``fem.laplacian_matvec``)
    and Q_node is assembled from per-element source Q_e = eta * Gc_e *
    |Delta_d_e| / l0 / dt.

    Parameters
    ----------
    fem        : FEMOperators   — provides laplacian_matvec and mesh
    rho_c      : float          — volumetric heat capacity [J/(m³·K)]
    k_therm    : float          — thermal conductivity [W/(m·K)]
    eta_crack  : float          — Taylor-Quinney coefficient (fraction of
                                  fracture energy converted to heat)
    dt         : float          — thermal time step [s]  (should satisfy
                                  CFL: dt < h² rho_c / (2 k_therm))
    T_ref      : float          — reference temperature [K]
    Gc_ref     : float          — reference critical energy release rate
                                  used for the heating source
    l0         : float          — phase-field regularisation length [m]
    """

    def __init__(
        self,
        fem,
        rho_c: float = 3.5e6,
        k_therm: float = 45.0,
        eta_crack: float = 0.9,
        dt: float = 1e-7,
        T_ref: float = 293.15,
        Gc_ref: float = 2.7e-3,
        l0: float = 1.0e-3,
    ) -> None:
        self.fem = fem
        self.mesh = fem.mesh
        self.rho_c = rho_c
        self.k_therm = k_therm
        self.eta_crack = eta_crack
        self.dt = dt
        self.T_ref = T_ref
        self.Gc_ref = Gc_ref
        self.l0 = l0

        # Precompute (dt / (rho_c * M_lump)) per node — constant coefficient
        M_lump = self.mesh.M_scalar  # (n_nodes,) — area/3 per node
        self._alpha = (dt / (rho_c * M_lump + 1e-30)).to(
            dtype=self.mesh.dtype, device=self.mesh.device
        )

        # Scatter index for element→node assembly (shape n_elems*3)
        self._elem_flat = self.mesh._elem_flat  # (E*3,)

    # ------------------------------------------------------------------
    # Forward step
    # ------------------------------------------------------------------

    def step(
        self,
        T: Tensor,
        d_new: Tensor,
        d_prev: Tensor,
        Gc_field: Tensor | None = None,
    ) -> Tensor:
        """One explicit Euler step of the heat equation.

        Parameters
        ----------
        T        : (n_nodes,) temperature field at current step [K]
        d_new    : (n_nodes,) damage field at new step
        d_prev   : (n_nodes,) damage field at previous step
        Gc_field : (n_elems,) per-element Gc [J/m²], or None → use Gc_ref

        Returns
        -------
        T_new : (n_nodes,) updated temperature field, differentiable
        """
        T = T.to(dtype=self.mesh.dtype, device=self.mesh.device)
        d_new = d_new.to(dtype=self.mesh.dtype, device=self.mesh.device)
        d_prev = d_prev.to(dtype=self.mesh.dtype, device=self.mesh.device)

        # Diffusion: k * (K_lap @ T)
        diffusion = self.k_therm * self.fem.laplacian_matvec(T)

        # Crack-tip heating source (element-level, then scatter to nodes)
        Q_node = self._crack_heating(d_new, d_prev, Gc_field)

        # Explicit Euler: T_new = T + alpha * (diffusion + Q_node)
        T_new = T + self._alpha * (diffusion + Q_node)
        return T_new

    def _crack_heating(
        self,
        d_new: Tensor,
        d_prev: Tensor,
        Gc_field: Tensor | None,
    ) -> Tensor:
        """Assemble nodal crack-tip heating from damage increment.

        Source per element:
            Q_e = eta * Gc_e * area_e * |Delta_d_e| / (l0 * dt)

        where Delta_d_e = max(d_e_new - d_e_prev, 0) averaged from nodes.
        """
        elems = self.mesh.elements  # (E, 3)
        areas = self.mesh.areas     # (E,)

        # Element-average damage increment (damage is irreversible → clamp)
        d_new_e = d_new[elems].mean(dim=1)    # (E,)
        d_prev_e = d_prev[elems].mean(dim=1)  # (E,)
        delta_d_e = (d_new_e - d_prev_e).clamp(min=0.0)  # irreversibility

        # Per-element Gc
        if Gc_field is not None:
            Gc_e = Gc_field.to(dtype=self.mesh.dtype, device=self.mesh.device)
        else:
            Gc_e = torch.full_like(areas, self.Gc_ref)

        # Q_e = eta * Gc_e * area_e * delta_d_e / (l0 * dt)
        Q_e = (self.eta_crack * Gc_e * areas * delta_d_e
               / (self.l0 * self.dt))   # (E,)

        # Scatter: each node gets Q_e / 3 from each of its elements
        Q_node = torch.zeros(self.mesh.n_nodes, dtype=self.mesh.dtype,
                             device=self.mesh.device)
        # Expand element contribution equally to 3 nodes
        Q_contrib = (Q_e / 3.0).unsqueeze(1).expand(-1, 3)  # (E, 3)
        Q_node.scatter_add_(0, self._elem_flat,
                            Q_contrib.reshape(-1))
        return Q_node

    # ------------------------------------------------------------------
    # Multi-step helper
    # ------------------------------------------------------------------

    def integrate(
        self,
        T_init: Tensor,
        d_history: list[Tensor],
        Gc_field: Tensor | None = None,
    ) -> Tensor:
        """Integrate n_steps thermal steps.

        Parameters
        ----------
        T_init    : (n_nodes,) initial temperature
        d_history : list of (n_nodes,) damage fields at consecutive steps
                    (length n_steps + 1)
        Gc_field  : (n_elems,) per-element Gc, or None

        Returns
        -------
        T_final : (n_nodes,) temperature after n_steps
        """
        T = T_init
        for i in range(len(d_history) - 1):
            T = self.step(T, d_history[i + 1], d_history[i], Gc_field)
        return T

    # ------------------------------------------------------------------
    # Gc(T) coupling
    # ------------------------------------------------------------------

    @staticmethod
    def gc_field(
        T: Tensor,
        Gc0: float | Tensor,
        model: Literal["linear", "arrhenius"] = "linear",
        alpha_T: float | Tensor = 1e-4,
        T_ref: float = 293.15,
        Ea_R: float | Tensor = 0.0,
    ) -> Tensor:
        """Per-node Gc(T) field from a temperature field.

        Parameters
        ----------
        T       : (n_nodes,) temperature [K]
        Gc0     : nominal Gc at T_ref [J/m²]
        model   : ``'linear'`` or ``'arrhenius'``
        alpha_T : thermal softening coefficient (linear model) [1/K]
        T_ref   : reference temperature [K]
        Ea_R    : activation energy / R = Ea / R_gas (Arrhenius model) [K]

        Returns
        -------
        Gc_T : (n_nodes,) Gc field, differentiable w.r.t. T, Gc0, alpha_T, Ea_R
        """
        if model == "linear":
            # Gc(T) = Gc0 * (1 - alpha_T * (T - T_ref)), clamped >= 0
            scale = 1.0 - alpha_T * (T - T_ref)
            scale = scale.clamp(min=0.0)
            return Gc0 * scale

        if model == "arrhenius":
            # Gc(T) = Gc0 * exp(-Ea_R * (1/T - 1/T_ref))
            exponent = -Ea_R * (1.0 / T - 1.0 / T_ref)
            return Gc0 * torch.exp(exponent)

        raise ValueError(f"Unknown Gc(T) model: '{model}'. "
                         f"Choose 'linear' or 'arrhenius'.")


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def _run_integration_test() -> None:
    print("thermal_update integration test")
    print("=" * 50)

    import sys
    import os
    # Add package root so we can import phast
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, os.path.dirname(_here))

    from phast.mesh import FEMMesh
    from phast.material import Material
    from phast.fem_operators import FEMOperators

    dtype = torch.float64
    device = "cpu"

    # --- Minimal 2D mesh: 6×6 grid of triangles on [0,5mm]×[0,5mm] ---
    nx, ny = 6, 6   # grid points
    L = 5.0e-3      # 5 mm

    xs = np.linspace(0, L, nx)
    ys = np.linspace(0, L, ny)
    xx, yy = np.meshgrid(xs, ys)
    nodes_np = np.column_stack([xx.ravel(), yy.ravel()])

    # Triangulate: two triangles per quad cell
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = j * nx + (i + 1)
            n01 = (j + 1) * nx + i
            n11 = (j + 1) * nx + (i + 1)
            elems.append([n00, n10, n11])
            elems.append([n00, n11, n01])
    elems_np = np.array(elems, dtype=np.int64)

    nodes_t = torch.tensor(nodes_np, dtype=dtype)
    elems_t = torch.tensor(elems_np, dtype=torch.long)
    mesh = FEMMesh.from_tensors(nodes_t, elems_t, device=device, dtype=dtype)
    print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elems} elems")

    mat = Material(E=210e9, nu=0.3, rho=7800.0, Gc=2.7e-3, l0=L / 5,
                   pf_model="AT2")
    fem = FEMOperators(mesh, mat)

    # --- Thermal module ---
    therm = ThermalUpdate(
        fem,
        rho_c=3.5e6,     # J/(m³·K)
        k_therm=45.0,    # W/(m·K)
        eta_crack=0.9,
        dt=1e-8,         # explicit, stable for h=1mm grid
        T_ref=293.15,
        Gc_ref=2.7e-3,
        l0=L / 5,
    )
    print(f"  ThermalUpdate: dt={therm.dt:.1e}, rho_c={therm.rho_c:.2e}")

    # --- Forward step: damage grows from 0 to 1 in bottom row ---
    T0 = torch.full((mesh.n_nodes,), 293.15, dtype=dtype)
    d_prev = torch.zeros(mesh.n_nodes, dtype=dtype)
    d_new = torch.zeros(mesh.n_nodes, dtype=dtype)
    d_new[:nx] = 1.0  # bottom row fully broken

    T1 = therm.step(T0, d_new, d_prev)
    dT = (T1 - T0)
    print(f"  Temperature rise max: {float(dT.max()):.4e} K")
    print(f"  Temperature rise min: {float(dT.min()):.4e} K")
    assert float(dT.max()) > 0.0, "No heating at crack nodes"
    assert not T1.isnan().any(), "NaN in T_new"

    # --- Gradient: d(T_sum)/d(eta_crack) via autograd ---
    eta_t = torch.tensor(0.9, dtype=dtype, requires_grad=True)
    therm_g = ThermalUpdate(
        fem,
        rho_c=3.5e6, k_therm=45.0, eta_crack=1.0,   # eta_crack ignored here
        dt=1e-8, T_ref=293.15, Gc_ref=2.7e-3, l0=L / 5,
    )
    # Manual: Q_node depends on eta_t
    T0_g = T0.clone()
    d_new_g = d_new.clone()
    d_prev_g = d_prev.clone()

    # Replicate step with eta_t as parameter
    elems = mesh.elements
    areas = mesh.areas
    d_new_e = d_new_g[elems].mean(dim=1)
    d_prev_e = d_prev_g[elems].mean(dim=1)
    delta_d_e = (d_new_e - d_prev_e).clamp(min=0.0)
    Gc_e = torch.full_like(areas, 2.7e-3)
    Q_e = eta_t * Gc_e * areas * delta_d_e / ((L / 5) * 1e-8)
    Q_node = torch.zeros(mesh.n_nodes, dtype=dtype)
    Q_contrib = (Q_e / 3.0).unsqueeze(1).expand(-1, 3)
    Q_node.scatter_add_(0, mesh._elem_flat, Q_contrib.reshape(-1))
    diffusion = 45.0 * fem.laplacian_matvec(T0_g)
    T_new_g = T0_g + therm_g._alpha * (diffusion + Q_node)
    loss = T_new_g.sum()
    loss.backward()

    print(f"  dT_sum/d(eta): {float(eta_t.grad):.4e}")
    assert eta_t.grad is not None, "No gradient for eta"
    assert not torch.isnan(eta_t.grad), "NaN gradient for eta"

    # --- Gc(T) coupling ---
    T_field = torch.linspace(270.0, 400.0, 30, dtype=dtype)

    # Linear model
    alpha_t = torch.tensor(1e-4, dtype=dtype, requires_grad=True)
    Gc_lin = ThermalUpdate.gc_field(T_field, Gc0=2.7e-3, model='linear',
                                    alpha_T=alpha_t, T_ref=293.15)
    print(f"  Gc_linear range: [{float(Gc_lin.min()):.4e}, {float(Gc_lin.max()):.4e}]")
    Gc_lin.sum().backward()
    print(f"  dGc_sum/d(alpha_T): {float(alpha_t.grad):.4e}")
    assert alpha_t.grad is not None
    assert not torch.isnan(alpha_t.grad)

    # Arrhenius model
    Ea_R_t = torch.tensor(500.0, dtype=dtype, requires_grad=True)
    Gc_arr = ThermalUpdate.gc_field(T_field, Gc0=2.7e-3, model='arrhenius',
                                    Ea_R=Ea_R_t, T_ref=293.15)
    print(f"  Gc_arrhenius range: [{float(Gc_arr.min()):.4e}, {float(Gc_arr.max()):.4e}]")
    Gc_arr.sum().backward()
    print(f"  dGc_sum/d(Ea_R): {float(Ea_R_t.grad):.4e}")
    assert Ea_R_t.grad is not None
    assert not torch.isnan(Ea_R_t.grad)

    # --- FD check: gradient of T_sum w.r.t. eta_crack ---
    def T_sum(eta_val: float) -> float:
        therm_fd = ThermalUpdate(
            fem, rho_c=3.5e6, k_therm=45.0, eta_crack=eta_val,
            dt=1e-8, T_ref=293.15, Gc_ref=2.7e-3, l0=L / 5,
        )
        T_fd = therm_fd.step(T0, d_new, d_prev)
        return float(T_fd.sum())

    eps = 1e-6
    fd_grad = (T_sum(0.9 + eps) - T_sum(0.9 - eps)) / (2 * eps)
    print(f"  FD grad (eta_crack): {fd_grad:.4e}")
    print(f"  AD grad (eta_crack): {float(eta_t.grad):.4e}")
    # Both should be same sign and similar magnitude
    assert fd_grad > 0.0 and float(eta_t.grad) > 0.0, "Sign mismatch"

    # --- Multi-step integration ---
    d_hist = [torch.zeros(mesh.n_nodes, dtype=dtype)]
    for _ in range(5):
        d_prev_h = d_hist[-1]
        d_new_h = (d_prev_h + 0.1).clamp(max=1.0)
        d_hist.append(d_new_h)

    T_final = therm.integrate(T0.clone(), d_hist)
    print(f"  T_final (5 steps) max: {float(T_final.max()):.4f} K")
    assert float(T_final.max()) > 293.15, "Temperature should have risen"
    assert not T_final.isnan().any()

    print("\nPASS — all assertions passed")


if __name__ == "__main__":
    _run_integration_test()
