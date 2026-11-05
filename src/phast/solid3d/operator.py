"""Matrix-free periodic voxel operator for hybrid AT2 phase-field fracture.

Mechanics unknowns are ``x = [u~ (3 n_red, node-major), E (6)]``: the
periodic displacement fluctuation on the reduced nodes of
:class:`~phast.solid3d.periodic.PeriodicVoxelMap` and the
engineering macro strain.  The total displacement is
``u = E_tensor (X - origin) + P u~`` and the strain at quadrature point
``q`` of element ``e`` is ``eps_q = B_q u~_e + E``.  The elastic energy
``0.5 sum_e sum_q w_q g_q eps_q.C_e.eps_q = 0.5 x.A(g) x`` defines the
symmetric mechanics operator

``A(g) x = [P^T scatter(sum_q w B_q^T sigma_q), sum_e sum_q w sigma_q]``,
``sigma_q = g_q C_e eps_q``,

whose ``E`` block is ``V`` times the macro stress.  Mixed macro control
fixes the controlled and ``fixed_zero`` components of ``E`` (and the three
dofs of one pinned node); the ``free`` components then carry zero macro
stress at equilibrium.

The damage ``d`` lives on the reduced nodes.  Its element operator is

``Gc_e (M/ell + ell L) + 2(1-kappa) sum_q w_q H_q N_q N_q^T``

with right-hand side ``2(1-kappa) sum_q w_q H_q N_q``, the Galerkin form of
``(Gc/ell + 2(1-kappa)H) d - div(Gc ell grad d) = 2(1-kappa)H`` with a
per-voxel ``Gc`` in both terms.  :meth:`PeriodicVoxelPhaseFieldOperator.damage_system`
scales the system by ``damage_scale = ell/(min Gc * V_voxel)`` for the box-KKT
solver.

Every voxel has the same shape, so one reference voxel serves every element.
Elements are processed in batches of ``cfg.batch_size``, stiffnesses are
gathered from the bank per batch and assembly uses ``scatter_add_``; no batch
loop synchronises with the device.

Interface driver.  Elements with ``DRIVER_INTERFACE`` (a smeared-laminate interface with
``SolverConfig.energy_split_band = 'interface'``) are driven by the interface traction energy ``psi_plus``.  With
``SolverConfig.band_stress = 'hybrid'`` (the default) their stress is the hybrid ``g C eps`` of every other phase, so the mechanics stays linear and the
stress, energy and snapshot arithmetic is exactly that of an isotropic-driver interface; only
:meth:`PeriodicVoxelPhaseFieldOperator.driving_force` differs.  With ``band_stress = 'variational'`` they carry the stress
``sigma_q = C eps_q - (1 - g_q) C q_q`` of :func:`~phast.solid3d.splits.interface_degraded_stress`, the derivative of ``psi0 - (1 - g) psi_plus``.  It is
piecewise linear in ``eps``: for a fixed normal-opening pattern ``s`` (``t_11 > 0`` per quadrature point,
:meth:`PeriodicVoxelPhaseFieldOperator.interface_opening`) it is linear, so :meth:`~PeriodicVoxelPhaseFieldOperator.mech_action` takes the pattern as
``opening`` and stays a symmetric positive definite operator ``A_s(g)``.  The staggered driver re-solves until the pattern of the solution equals the
pattern used.  Every evaluation without ``opening`` (macro stress, energies, snapshots) uses the pattern of its own state.

Crack closure.  Damaged elements add back ``(1 - g)`` of a fixed-axis compressive energy ``psi_c`` (:mod:`.splits`),
``sum_k k_k/2 <v_k . eps>-^2`` over strain functionals ``v_k`` of two groups: the isotropic phases (cell axes, rows ``1, e_x, e_y, e_z``, moduli
``lam, 2 mu, 2 mu, 2 mu``) and the smeared-laminate interface (interface frame: ``C m1, q2 + q3, q2, q3``).  For a fixed sign pattern
(:class:`~phast.solid3d.model.ClosurePattern`, ``closure``) the stress is linear and the action symmetric positive definite; without it each point uses its
own pattern (the internal force of the convex degraded energy).  With the closure off nothing is built and every evaluation follows the plain arithmetic;
passing ``closure`` then raises.

PF-CZM damage law.  The degradation is Wu's rational ``g(d) = (1 - kappa) A/(A + Q) + kappa`` with ``A = (1 - d)^p`` and
``Q = a1 d (1 + a2 d + a2 a3 d^2)``, from :func:`phast.physics.pfczm.pfczm_degradation_derivatives`, with a per-element
``a1 = 4 E_ref Gc / (pi ell f_t^2 (1 - kappa))`` (:func:`phast.physics.pfczm.pfczm_a1`): ``E_ref`` is ``E`` of an isotropic or laminate phase (the normal
modulus ``a_n`` of an interface-driver element) and ``f_t`` is ``PhaseMaterial.tensile_strength``.  Every stress law takes ``g`` as an argument, so the
hybrid stress, the interface stress and the closure add-back ``(1 - g)`` use the PF-CZM ``g`` unchanged.  The damage energy per element is
``sum_q w H_q g(d_q) + Gc/(pi ell) int alpha(d) + Gc ell/pi int |grad d|^2`` with ``alpha(d) = 2d - d^2``; it is not quadratic, and
:meth:`PeriodicVoxelPhaseFieldOperator.pfczm_newton_system` gives the SPD tangent of the projected Newton solve of the staggered driver.  The AT2
damage operators are left as they are.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import math

import numpy as np
import torch

from phast.physics.pfczm import (
    PFCZM_C_ALPHA,
    pfczm_a1,
    pfczm_degradation_derivatives,
    pfczm_max_length_scale,
    pfczm_max_strength,
    pfczm_onset_convexity_bound,
    pfczm_softening_parameters,
)
from phast.solvers.box_kkt_h8 import H8ScalarSPDSystem
from phast.core.voigt import engineering_strain_to_tensor
from phast.solid3d.materials import build_element_arrays, frame_from_tangent
from phast.solid3d.model import (
    ClosurePattern,
    DRIVER_FULL,
    DRIVER_INTERFACE,
    DRIVER_NONE,
    DRIVER_SPECTRAL,
    PHASE_ID,
    PHASE_NAMES,
    ElementMaterialArrays,
    FieldSnapshot,
    LoadCase,
    SolverConfig,
    VoxelModel,
    solver_dtype,
)
from phast.solid3d.periodic import PeriodicVoxelMap, choose_pinned_node, reference_voxel
from phast.solid3d.splits import (
    CLOSURE_MODULI_TOLERANCE,
    CLOSURE_STIFFENING_TOLERANCE,
    closure_band_moduli,
    closure_band_row_moduli,
    closure_band_rows,
    closure_isotropic_moduli,
    closure_isotropic_rows,
    closure_stiffness,
    closure_stress,
    degradation as _degradation,
    determinant_3x3,
    full_driving_energy,
    interface_driving_energy,
    interface_energies,
    interface_moduli,
    interface_positive_stress,
    interface_traction_vectors,
    no_stiffening_margin,
    principal_values,
    spectral_tensile_energy,
)


PFCZM_TANGENT_FLOOR = 0.25
"""Floor ``s`` of the quadrature-point curvature of the PF-CZM Newton tangent, as a fraction of
``2 Gc/(pi ell)``.  The local curvature ``H g''(d_q) + Gc/(pi ell) alpha'' = H g''(d_q) -
2 Gc/(pi ell)`` is used where it exceeds ``s 2 Gc/(pi ell)`` and replaced by that floor elsewhere, so
the subproblem is SPD (the Laplacian alone is singular on a periodic cell) and the iteration is the
exact Newton iteration wherever the local energy is convex enough."""


class PeriodicVoxelPhaseFieldOperator:
    """Matrix-free mechanics and damage operators of one periodic voxel cell.

    Parameters
    ----------
    model:
        The voxel cell (grid, phases, tangents and materials).
    load:
        Mixed macro control; :meth:`mech_fixed_mask` fixes ``load.controlled``
        and ``load.fixed_zero``.
    cfg:
        Solver settings (``kappa``, ``batch_size``, energy splits, device).
    materials:
        Prebuilt per-element arrays; built with
        :func:`~phast.solid3d.materials.build_element_arrays`
        when omitted.  Their device is the operator device.
    pinned_node:
        Reduced node whose fluctuation is fixed to remove rigid translation;
        defaults to :func:`~phast.solid3d.periodic.choose_pinned_node`
        (the first node touching no pore voxel).
    pfczm_strengths:
        Optional research-only main-field tensile-strength overrides, keyed by
        phase names present in the cell. Values must be scalar float64 tensors
        on the operator device; their autograd graph is retained. This is not
        a load-path derivative.
        The mapping is copied, but its tensors are borrowed read-only: rebuild
        the operator for each parameter value/graph rather than mutating them.
        ``pfczm_strength_overrides`` and ``pfczm_phases`` are detached reporting
        snapshots of the effective values; the material table is unchanged.
        Run primal solves with detached parameters or under ``torch.no_grad``;
        rebuild graph-bearing operators for fixed-state residual derivatives.

    Attributes
    ----------
    pmap, ref, mat, device, dtype, n_red, n_mech, volume, pinned_node,
    damage_scale, plus ``model``, ``load``, ``cfg``, ``grid``, ``n_elements``,
    ``ell`` and ``kappa``.
    """

    def __init__(
        self,
        model: VoxelModel,
        load: LoadCase,
        cfg: SolverConfig,
        *,
        materials: ElementMaterialArrays | None = None,
        pinned_node: int | None = None,
        pfczm_strengths: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if not isinstance(model, VoxelModel):
            raise TypeError("model must be a VoxelModel")
        if not isinstance(load, LoadCase):
            raise TypeError("load must be a LoadCase")
        if not isinstance(cfg, SolverConfig):
            raise TypeError("cfg must be a SolverConfig")
        model.validate()
        self.model = model
        self.load = load
        self.cfg = cfg
        self.grid = model.grid
        self.dtype = solver_dtype(cfg.dtype)
        if materials is None:
            materials = build_element_arrays(model, cfg)
        if not isinstance(materials, ElementMaterialArrays):
            raise TypeError("materials must be ElementMaterialArrays")
        if materials.n_elements != self.grid.n_elements:
            raise ValueError(
                f"materials describe {materials.n_elements} elements, the grid has "
                f"{self.grid.n_elements}"
            )
        self.mat = materials
        self.device = materials.device
        self.pfczm_strengths: dict[str, torch.Tensor] = {}
        self.pfczm_strength_overrides: dict[str, Any] = {}
        if pfczm_strengths is not None:
            if cfg.damage_law != "pfczm":
                raise ValueError("pfczm_strengths applies to damage_law 'pfczm' only")
            if not isinstance(pfczm_strengths, Mapping):
                raise TypeError("pfczm_strengths must be a mapping of present phase names to tensors")
            present = {PHASE_NAMES[int(i)] for i in torch.unique(materials.phase).tolist()}
            for name, strength in pfczm_strengths.items():
                if name not in present:
                    raise ValueError(f"pfczm_strengths phase {name!r} is not present in the cell")
                if not isinstance(strength, torch.Tensor):
                    raise TypeError(f"pfczm_strengths[{name!r}] must be a scalar float64 tensor")
                if strength.ndim != 0 or strength.dtype != torch.float64 or strength.layout != torch.strided:
                    raise ValueError(f"pfczm_strengths[{name!r}] must be a scalar float64 tensor")
                if strength.device != self.device:
                    raise ValueError(f"pfczm_strengths[{name!r}] must be on operator device {self.device}")
                if not bool(torch.isfinite(strength.detach())) or not bool(strength.detach() > 0):
                    raise ValueError(f"pfczm_strengths[{name!r}] must be finite and positive")
                self.pfczm_strengths[name] = strength
                self.pfczm_strength_overrides[name] = {
                    "field": "main", "source": "tensor_override", "f_t": float(strength.detach()),
                    "material_f_t": model.materials.phases[name].tensile_strength,
                }
        self.n_elements = self.grid.n_elements
        self.volume = float(self.grid.volume)
        self.ell = float(model.materials.length_scale)
        self.kappa = float(cfg.kappa)
        self.batch_size = int(cfg.batch_size)

        Gc = materials.Gc
        if not bool(torch.isfinite(Gc).all()) or not bool((Gc > 0.0).all()):
            raise ValueError("Gc must be finite and positive at every element")
        self.damage_scale = self.ell / (float(Gc.min()) * float(self.grid.voxel_volume))

        self.pmap = PeriodicVoxelMap.build(self.grid, device=self.device)
        self.ref = reference_voxel(self.grid.h, dtype=self.dtype, device=self.device)
        self.n_red = self.pmap.n_reduced
        self.n_mech = 3 * self.n_red + 6
        if pinned_node is None:
            pinned_node = choose_pinned_node(self.pmap, materials.phase, PHASE_ID["pore"])
        if isinstance(pinned_node, bool) or int(pinned_node) != pinned_node or not 0 <= int(pinned_node) < self.n_red:
            raise ValueError(f"pinned_node must be a reduced node in [0, {self.n_red}), got {pinned_node!r}")
        self.pinned_node = int(pinned_node)

        # Reference-voxel operators shared by every element.
        B = self.ref.B
        self._B_flat = B.permute(2, 0, 1).reshape(24, 48).contiguous()
        self._BB = torch.einsum("qia,qja->qaij", B, B).reshape(8 * 24, 36).contiguous()
        self._N = self.ref.N
        self._NN = self._N.square()
        self._w = self.ref.w
        self._K_damage = (self.ref.M / self.ell + self.ell * self.ref.L).contiguous()
        self._K_damage_diag = torch.diagonal(self._K_damage).clone()
        self._c_history = 2.0 * (1.0 - self.kappa)

        self._driver_groups: list[tuple[int, torch.Tensor]] = []
        known = (DRIVER_FULL, DRIVER_SPECTRAL, DRIVER_INTERFACE)
        for code in known:
            index = torch.nonzero(materials.driver == code, as_tuple=False).reshape(-1)
            if index.numel():
                self._driver_groups.append((code, index))
        codes = set(torch.unique(materials.driver).tolist()) - {DRIVER_NONE, *known}
        if codes:
            raise ValueError(f"unknown element driver codes {sorted(codes)}")
        self._phases_present = model.phases_present()
        self._node_offsets: np.ndarray | None = None

        # Interface-driver elements: element ids, traction rows m_k (n_if, 3, 6) and 1/a_k.
        # They drive the damage in either band stress; the variational band stress
        # (cfg.band_stress 'variational', ) also uses them in the stress pass.
        self._if_index: torch.Tensor | None = None
        self._if_M: torch.Tensor | None = None
        self._if_inv_a: torch.Tensor | None = None
        for code, index in self._driver_groups:
            if code == DRIVER_INTERFACE:
                M = interface_traction_vectors(materials.tangent[index])
                a = interface_moduli(self._stiffness(index), M)
                self._if_index = index
                self._if_M = M.contiguous()
                self._if_inv_a = (1.0 / a).contiguous()
        self._if_variational = self._if_index is not None and cfg.band_stress == "variational"

        for code, index in self._driver_groups:
            pass

        # PF-CZM damage law: nothing is built with damage_law 'at2'.
        self.damage_law = cfg.damage_law
        self._pfczm = cfg.damage_law == "pfczm"
        self._a1: torch.Tensor | None = None
        self._m_damage: torch.Tensor | None = None
        self._K_pfczm: torch.Tensor | None = None
        self.pfczm_bound: float | None = None
        self.pfczm_phases: dict[str, Any] = {}
        if self._pfczm:
            self._build_pfczm(model, materials)

        # Crack closure: nothing is built with crack_closure 'none'.
        self._closure = cfg.crack_closure != "none"
        self._cl_lam: torch.Tensor | None = None
        self._cl_mu: torch.Tensor | None = None
        self._cl_k: torch.Tensor | None = None
        self._cl_iso_mask: torch.Tensor | None = None
        self._cl_rows: torch.Tensor | None = None
        self._cl_BtR: torch.Tensor | None = None
        self._cb_index: torch.Tensor | None = None
        self._cb_rows: torch.Tensor | None = None
        self._cb_k: torch.Tensor | None = None
        if self._closure:
            self._build_closure(model, materials)

    # ------------------------------------------------------------------
    # PF-CZM damage law
    # ------------------------------------------------------------------

    @property
    def is_pfczm(self) -> bool:
        """Whether the damage law is the PF-CZM law (``damage_law = 'pfczm'``, )."""

        return self._pfczm

    def _build_pfczm(self, model: VoxelModel, materials: ElementMaterialArrays) -> None:
        """Per-element ``a1`` of the PF-CZM law and the onset check.

        Raises ``ValueError`` when a phase of the cell has no tensile
        strength, or when ``a1 < p + a2`` (``g''(0) < 0``)
        in an element with a damage driver, i.e. ``ell > 4 E_ref Gc / (pi f_t^2
        (1 - kappa)(p + a2))`` (``8 l_ch / (3 pi)`` for linear softening, with
        ``l_ch = E Gc / f_t^2``).  Elements without a driver (pores) are not
        checked.  :attr:`pfczm_phases` records ``f_t``, ``E_ref``, ``Gc``,
        ``a1``, the bound, the largest admissible strength and length scale
        of every phase.
        """

        cfg = self.cfg
        a2, a3 = pfczm_softening_parameters(cfg.pfczm_softening)
        self._pfczm_a2, self._pfczm_a3, self._pfczm_p = float(a2), float(a3), cfg.pfczm_p
        bound = float(pfczm_onset_convexity_bound(a2, cfg.pfczm_p))
        self.pfczm_bound = bound
        self._m_damage = self.ref.M.sum(dim=1).contiguous()
        self._K_pfczm = ((2.0 / PFCZM_C_ALPHA) * self.ell * self.ref.L).contiguous()
        n_ids = len(PHASE_NAMES)
        nan = float("nan")
        f_by_id = torch.full((n_ids,), nan, dtype=self.dtype, device=self.device)
        E_by_id = torch.full((n_ids,), nan, dtype=self.dtype, device=self.device)
        present = {PHASE_NAMES[int(i)] for i in torch.unique(materials.phase).tolist()}
        missing: list[str] = []
        for name, material in model.materials.phases.items():
            index = PHASE_ID[name]
            E_by_id[index] = float(material.E)
            if name in self.pfczm_strengths:
                # CopySlices preserves the parameter graph; never float() an
                # override on the numerical path into the shared core law.
                f_by_id[index] = self.pfczm_strengths[name]
            elif material.tensile_strength is None:
                if name in present:
                    missing.append(name)
            else:
                f_by_id[index] = float(material.tensile_strength)
        if missing:
            raise ValueError(
                "damage_law 'pfczm' needs a tensile strength for every phase of the cell "
                f"(material.tensile_strength); missing: {', '.join(missing)}"
            )
        phase = materials.phase
        driver = materials.driver
        E_ref = E_by_id[phase]
        if self._if_index is not None:
            # interface-driver band: psi_plus = t_n^2 / (2 a_n) at a pure normal opening, so the onset
            # is at the normal traction t_n = f_t with the normal modulus a_n as reference
            E_ref = E_ref.clone()
            E_ref[self._if_index] = 1.0 / self._if_inv_a[:, 0]
        f_t = f_by_id[phase]
        self._a1 = pfczm_a1(E_ref, materials.Gc, self.ell, f_t, self.kappa).contiguous()
        violations: list[str] = []
        active = driver != DRIVER_NONE
        for name in model.materials.phases:
            mask = phase == PHASE_ID[name]
            if bool(mask.any()):
                self.pfczm_phases[name] = self._pfczm_phase_summary(
                    name, mask, mask & active, self._a1, E_ref, materials.Gc, f_t, bound, violations
                )
        if violations:
            raise ValueError(
                f"damage_law 'pfczm': the onset condition a1 >= p + a2 = {bound:g} (g''(0) >= 0, i.e. "
                "ell <= 4 E Gc / (pi f_t^2 (1 - kappa)(p + a2)), 8 l_ch / (3 pi) for linear softening) "
                f"fails at ell = {self.ell:g} mm in: " + "; ".join(violations) + ""
            )

    def _pfczm_phase_summary(
        self,
        label: str,
        mask: torch.Tensor,
        checked: torch.Tensor,
        a1: torch.Tensor,
        E_ref: torch.Tensor,
        Gc: torch.Tensor,
        f_t: torch.Tensor,
        bound: float,
        violations: list[str],
    ) -> dict[str, Any]:
        """Record of one phase (or field) of the PF-CZM law; appends to ``violations``."""

        weakest = int(torch.argmin(torch.where(mask, a1, torch.full_like(a1, math.inf))))
        values = torch.stack(
            (
                a1[mask].min(), a1[mask].max(), E_ref[mask].min(), E_ref[mask].max(),
                Gc[mask].min(), Gc[mask].max(), f_t[weakest], E_ref[weakest], Gc[weakest],
            )
        ).detach().cpu().numpy()
        a1_min, a1_max, E_min, E_max, Gc_min, Gc_max, f, E_w, Gc_w = (float(v) for v in values)
        # largest admissible strength and length scale, at the element with the smallest a1
        f_max = float(pfczm_max_strength(E_w, Gc_w, self.ell, bound, self.kappa))
        ell_max = float(pfczm_max_length_scale(E_w, Gc_w, f, bound, self.kappa))
        checked_any = bool(checked.any())
        ok = (not checked_any) or float(a1[checked].detach().min()) >= bound
        if not ok:
            violations.append(
                f"{label}: a1 = {float(a1[checked].detach().min()):.4g} < {bound:g} (f_t = {f:g} MPa; largest admissible "
                f"f_t {f_max:.4g} MPa at this ell, largest admissible ell {ell_max:.4g} mm at this f_t)"
            )
        return {
            "f_t": f,
            "E_ref_min": E_min,
            "E_ref_max": E_max,
            "Gc_min": Gc_min,
            "Gc_max": Gc_max,
            "a1_min": a1_min,
            "a1_max": a1_max,
            "a1_bound": bound,
            "f_t_max": f_max,
            "ell_max": ell_max,
            "onset_checked": checked_any,
            "onset_ok": ok,
        }

    def _pfczm_derivatives(self, d_q: torch.Tensor, a1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PF-CZM ``g, g', g''`` at quadrature points ``(nb, 8)`` with the element ``a1`` ``(nb,)``."""

        return pfczm_degradation_derivatives(d_q, a1, self._pfczm_a2, self._pfczm_a3, p=self._pfczm_p, eta=self.kappa)

    def _pfczm_groups(self) -> list[tuple[slice, Any]]:
        """``(rows, elements)`` batches of the damage field (every element)."""

        out = []
        for begin in self._batches():
            rows = slice(begin, min(begin + self.batch_size, self.n_elements))
            out.append((rows, rows))
        return out

    def _pfczm_field(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(Gc, a1)`` per element."""

        if not self._pfczm:
            raise ValueError("the PF-CZM damage operators need damage_law 'pfczm'")
        return self.mat.Gc, self._a1

    def _pfczm_history(self, H: torch.Tensor | None) -> torch.Tensor | None:
        return self._check_qp('H', H)

    def pfczm_damage_energy(self, d: torch.Tensor, H: torch.Tensor | None) -> torch.Tensor:
        """PF-CZM damage energy in mJ (0-dim tensor)

        ``sum_e [sum_q w H_q g(d_q) + Gc/(pi ell)(2 m.d_e - d_e.M d_e) + Gc ell/pi d_e.L d_e]``
        with ``m = M 1`` (the integrals of the shape functions); the 8-point
        quadrature is exact for the two surface terms.  ``H=None`` drops the history term.
        """

        self._check("d", d, (self.n_red,))
        H = self._pfczm_history(H)
        Gc_all, a1_all = self._pfczm_field()
        M, L, m = self.ref.M, self.ref.L, self._m_damage
        nodes_all = self.pmap.element_nodes
        c_local = 1.0 / (PFCZM_C_ALPHA * self.ell)
        c_gradient = self.ell / PFCZM_C_ALPHA
        total = torch.zeros((), dtype=self.dtype, device=self.device)
        for rows, elements in self._pfczm_groups():
            d_e = d[nodes_all[elements]]
            local = 2.0 * (d_e @ m) - ((d_e @ M) * d_e).sum(dim=-1)
            gradient = ((d_e @ L) * d_e).sum(dim=-1)
            total = total + (Gc_all[rows] * (c_local * local + c_gradient * gradient)).sum()
            if H is not None:
                g = self._pfczm_derivatives(d_e @ self._N.T, a1_all[rows])[0]
                total = total + ((H[rows] * self._w) * g).sum()
        return total

    def pfczm_damage_gradient(self, d: torch.Tensor, H: torch.Tensor | None) -> torch.Tensor:
        """Unscaled gradient ``(n_red,)`` of :meth:`pfczm_damage_energy`.

        Per element ``sum_q w H_q g'(d_q) N_q + 2 Gc/(pi ell)(m - M d_e) + 2 Gc ell/pi L d_e``.
        """

        self._check("d", d, (self.n_red,))
        H = self._pfczm_history(H)
        Gc_all, a1_all = self._pfczm_field()
        M, L, m = self.ref.M, self.ref.L, self._m_damage
        out = torch.zeros(self.n_red, dtype=self.dtype, device=self.device)
        nodes_all = self.pmap.element_nodes
        c_local = 2.0 / (PFCZM_C_ALPHA * self.ell)
        c_gradient = 2.0 * self.ell / PFCZM_C_ALPHA
        for rows, elements in self._pfczm_groups():
            nodes = nodes_all[elements]
            d_e = d[nodes]
            y = Gc_all[rows].unsqueeze(-1) * (c_local * (m - d_e @ M) + c_gradient * (d_e @ L))
            if H is not None:
                gp = self._pfczm_derivatives(d_e @ self._N.T, a1_all[rows])[1]
                y = y + ((H[rows] * self._w) * gp) @ self._N
            out.scatter_add_(0, nodes.reshape(-1), y.reshape(-1))
        return out

    def pfczm_tangent_weights(self, d: torch.Tensor, H: torch.Tensor | None) -> torch.Tensor:
        """Quadrature-point curvatures ``(n, 8)`` of the Newton tangent.

        ``max(H g''(d_q) - 2 Gc/(pi ell), s 2 Gc/(pi ell))`` with ``s`` =
        :data:`PFCZM_TANGENT_FLOOR`: the second derivative of the local energy
        ``H g(d) + Gc/(pi ell) alpha(d)`` where it is above the floor, the
        floor elsewhere (``H=None``: no history term).
        """

        self._check("d", d, (self.n_red,))
        H = self._pfczm_history(H)
        Gc_all, a1_all = self._pfczm_field()
        n = self.n_elements
        c_alpha = 2.0 / (PFCZM_C_ALPHA * self.ell)
        out = torch.empty((n, 8), dtype=self.dtype, device=self.device)
        nodes_all = self.pmap.element_nodes
        for rows, elements in self._pfczm_groups():
            concave = (c_alpha * Gc_all[rows]).unsqueeze(-1)
            floor = PFCZM_TANGENT_FLOOR * concave
            if H is None:
                out[rows] = floor.expand(-1, 8)
                continue
            gpp = self._pfczm_derivatives(d[nodes_all[elements]] @ self._N.T, a1_all[rows])[2]
            out[rows] = torch.maximum(H[rows] * gpp - concave, floor)
        return out

    def pfczm_tangent_action(self, field: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Unscaled tangent action ``Gc (2 ell/pi) L f_e + sum_q w c_q N_q N_q^T f_e`` (``c``: ``weights``)."""

        Gc_all, _ = self._pfczm_field()
        out = torch.zeros(self.n_red, dtype=self.dtype, device=self.device)
        nodes_all = self.pmap.element_nodes
        for rows, elements in self._pfczm_groups():
            nodes = nodes_all[elements]
            f_e = field[nodes]
            y = Gc_all[rows].unsqueeze(-1) * (f_e @ self._K_pfczm)
            y = y + ((weights[rows] * self._w) * (f_e @ self._N.T)) @ self._N
            out.scatter_add_(0, nodes.reshape(-1), y.reshape(-1))
        return out

    def pfczm_tangent_diagonal(self, weights: torch.Tensor) -> torch.Tensor:
        """Unscaled Jacobi diagonal of :meth:`pfczm_tangent_action`."""

        Gc_all, _ = self._pfczm_field()
        out = torch.zeros(self.n_red, dtype=self.dtype, device=self.device)
        nodes_all = self.pmap.element_nodes
        K_diag = torch.diagonal(self._K_pfczm)
        for rows, elements in self._pfczm_groups():
            y = Gc_all[rows].unsqueeze(-1) * K_diag + (weights[rows] * self._w) @ self._NN
            out.scatter_add_(0, nodes_all[elements].reshape(-1), y.reshape(-1))
        return out

    def pfczm_newton_system(
        self, d: torch.Tensor, H: torch.Tensor | None, *, scaled: bool = True,
        correction: bool = False,
    ) -> tuple[H8ScalarSPDSystem, torch.Tensor]:
        """SPD Newton subproblem of the PF-CZM damage solve at ``d``, and the gradient there.

        Returns ``(system, gradient)``, both multiplied by ``damage_scale``
        when ``scaled``.
        The tangent is ``K = Gc (2 ell/pi) L + sum_q w c_q N_q N_q^T`` with the
        quadrature-point curvatures ``c_q`` of :meth:`pfczm_tangent_weights`:
        the exact Hessian where the local curvature
        ``H g''(d_q) - 2 Gc/(pi ell)`` is above the floor
        :data:`PFCZM_TANGENT_FLOOR` ``x 2 Gc/(pi ell)``, and that floor
        elsewhere (the consistent mass ``M`` is ``sum_q w N_q N_q^T``, so the
        concave ``alpha'' = -2`` mass term enters per quadrature point).  The right-hand side
        ``K d - gradient`` makes the box-QP ``min 1/2 x.K x - b.x`` the
        model ``min 1/2 (x - d).K (x - d) + gradient.(x - d)``.

        With ``correction=True``, the unknown is ``p = x - d`` and the rhs
        is ``-gradient``. Shift the box bounds by ``-d`` as well. This avoids
        subtracting ``K d`` from nearly equal vectors in the inner KKT audit.
        """

        scale = self.damage_scale if scaled else 1.0
        excluded = None
        weights = self.pfczm_tangent_weights(d, H)
        gradient = scale * self.pfczm_damage_gradient(d, H)

        def action(field: torch.Tensor) -> torch.Tensor:
            out = scale * self.pfczm_tangent_action(field, weights)
            if excluded is not None:
                out = out + excluded * field
            return out

        diagonal = scale * self.pfczm_tangent_diagonal(weights)
        if excluded is not None:
            diagonal = diagonal + excluded
        rhs = -gradient if correction else action(d) - gradient
        system = H8ScalarSPDSystem(action=action, rhs=rhs, diagonal=diagonal)
        return system, gradient

    def _pfczm_fracture_energy(self, d: torch.Tensor) -> dict[str, Any]:
        """PF-CZM surface functional in mJ, ``Gc/(pi ell) int alpha(d) + Gc ell/pi int |grad d|^2``."""

        Gc_all, _ = self._pfczm_field()
        n = self.n_elements
        local = torch.empty(n, dtype=self.dtype, device=self.device)
        gradient = torch.empty_like(local)
        nodes_all = self.pmap.element_nodes
        M, L, m = self.ref.M, self.ref.L, self._m_damage
        for rows, elements in self._pfczm_groups():
            d_e = d[nodes_all[elements]]
            Gc = Gc_all[rows]
            local[rows] = Gc / (PFCZM_C_ALPHA * self.ell) * (2.0 * (d_e @ m) - ((d_e @ M) * d_e).sum(dim=-1))
            gradient[rows] = Gc * self.ell / PFCZM_C_ALPHA * ((d_e @ L) * d_e).sum(dim=-1)
        phase = self.mat.phase
        by_id = torch.zeros(len(PHASE_NAMES), dtype=self.dtype, device=self.device)
        by_id.scatter_add_(0, phase, local + gradient)
        totals = torch.stack((local.sum(), gradient.sum())).cpu().numpy()
        by_id_np = by_id.cpu().numpy()
        return {
            "total": float(totals[0] + totals[1]),
            "local": float(totals[0]),
            "gradient": float(totals[1]),
            "by_phase": {name: float(by_id_np[PHASE_ID[name]]) for name in self._phases_present},
        }

    @property
    def has_interface(self) -> bool:
        """Whether any element uses the directional interface driver (``DRIVER_INTERFACE``)."""

        return self._if_index is not None

    @property
    def n_interface(self) -> int:
        """Number of interface-driver elements."""

        return 0 if self._if_index is None else int(self._if_index.numel())

    @property
    def has_interface_stress(self) -> bool:
        """Whether interface-driver elements carry the variational band stress (``band_stress = 'variational'``).

        Only then is the mechanics piecewise linear in the normal-opening
        pattern; with the hybrid band stress the interface elements
        differ from the others in their driving energy only.
        """

        return self._if_variational

    @property
    def has_closure(self) -> bool:
        """Whether the crack closure of is on (``crack_closure != 'none'``)."""

        return self._closure

    @property
    def n_closure_band(self) -> int:
        """Number of smeared-laminate band elements with the band closure."""

        return 0 if self._cb_index is None else int(self._cb_index.numel())

    # ------------------------------------------------------------------
    # crack closure set-up
    # ------------------------------------------------------------------

    def _build_closure(self, model: VoxelModel, materials: ElementMaterialArrays) -> None:
        """Validate the model for the closure and build the closure groups."""

        phases = model.materials.phases
        driver = materials.driver
        phase = materials.phase
        ids = lambda names: torch.as_tensor(  # noqa: E731
            [PHASE_ID[name] for name in names], dtype=torch.long, device=self.device
        )
        no_add_back = driver == DRIVER_NONE
        if self._if_variational:
            # the variational band stress keeps its own closure energy psi_minus
            no_add_back = no_add_back | (driver == DRIVER_INTERFACE)

        # (b) isotropic phases: closure axes are the cell axes
        iso_names = [name for name, mat in phases.items() if mat.is_isotropic]
        iso = torch.isin(phase, ids(iso_names)) & ~no_add_back if iso_names else torch.zeros_like(no_add_back)
        lam = torch.where(iso, materials.lam, torch.zeros_like(materials.lam))
        mu = torch.where(iso, materials.mu, torch.zeros_like(materials.mu))
        if bool(iso.any()):
            if bool((lam[iso] < 0.0).any()):
                raise ValueError("crack_closure 'axes' needs lam >= 0 in every isotropic phase")
            index = torch.nonzero(iso, as_tuple=False).reshape(-1)
            slots = materials.stiffness_index[index]
            rows = closure_isotropic_rows(self.dtype, self.device)
            for slot in torch.unique(slots).tolist():
                chosen = index[slots == slot]
                lam_s, mu_s = lam[chosen], mu[chosen]
                if float((lam_s - lam_s[0]).abs().max()) > 0.0 or float((mu_s - mu_s[0]).abs().max()) > 0.0:
                    raise ValueError("isotropic elements sharing a stiffness must share lam and mu")
                C = materials.stiffness_bank[slot]
                # iso(lam, mu) = (lam 1 1^T + 2 mu sum_i e_i e_i^T) + mu on the three engineering shears
                D = closure_stiffness(rows, closure_isotropic_moduli(lam_s[0], mu_s[0]))
                shear = torch.zeros((6, 6), dtype=self.dtype, device=self.device)
                shear[3, 3] = shear[4, 4] = shear[5, 5] = mu_s[0]
                defect = float((C - D - shear).abs().max() / C.abs().max())
                if not defect <= CLOSURE_MODULI_TOLERANCE:
                    raise ValueError(
                        "crack_closure 'axes' needs every isotropic stiffness to equal iso(lam, mu) of its "
                        f"phase (relative deviation {defect:.3e} > {CLOSURE_MODULI_TOLERANCE:.0e})"
                    )
                margin = float(no_stiffening_margin(C, D))
                if margin < -CLOSURE_STIFFENING_TOLERANCE:
                    raise ValueError(f"the isotropic closure stiffens the phase (margin {margin:.3e}, )")
            self._cl_lam = lam.contiguous()
            self._cl_mu = mu.contiguous()
            self._cl_k = closure_isotropic_moduli(lam, mu).contiguous()
            self._cl_iso_mask = iso
            self._cl_rows = rows
            self._cl_BtR = torch.einsum("qia,ki->qka", self.ref.B, rows).contiguous()

        # (c) smeared-laminate band: closure axes are the band frame
        laminate_names = [name for name, mat in phases.items() if mat.is_laminate]
        if laminate_names:
            band = torch.isin(phase, ids(laminate_names)) & ~no_add_back
            index = torch.nonzero(band, as_tuple=False).reshape(-1)
            if index.numel():
                tangent = materials.tangent[index]
                frame = frame_from_tangent(tangent)
                C = self._stiffness(index)
                M = interface_traction_vectors(tangent)
                try:
                    a = interface_moduli(C, M)
                except ValueError as exc:
                    raise ValueError(f"crack_closure 'axes' needs a band frame with a diagonal C_nn: {exc}") from exc
                lam_S, mu_S = closure_band_moduli(C, frame)
                rows = closure_band_rows(C, M[..., 0, :], frame)
                moduli = closure_band_row_moduli(1.0 / a[..., 0], lam_S, mu_S)
                self._check_no_stiffening("band", index, rows, moduli)
                self._cb_index = index
                self._cb_rows = rows.contiguous()
                self._cb_k = moduli.contiguous()

    def _check_no_stiffening(self, group: str, index: torch.Tensor, rows: torch.Tensor, moduli: torch.Tensor) -> None:
        """``lambda_min(C - D) >= -1e-12 max C_ii`` with every closure bit of the group active."""

        for begin in range(0, int(index.numel()), self.batch_size):
            part = slice(begin, begin + self.batch_size)
            C = self._stiffness(index[part])
            margin = float(no_stiffening_margin(C, closure_stiffness(rows[part], moduli[part])).min())
            if margin < -CLOSURE_STIFFENING_TOLERANCE:
                raise ValueError(
                    f"the {group} closure stiffens an element beyond its intact stiffness "
                    f"(lambda_min(C - D) / max C_ii = {margin:.3e}, )"
                )

    def _cb_chunks(self) -> list[tuple[slice, torch.Tensor]]:
        """``(rows, element ids)`` batches of the band closure elements."""

        if self._cb_index is None:
            return []
        n = self.n_closure_band
        return [
            (slice(begin, min(begin + self.batch_size, n)), self._cb_index[begin : begin + self.batch_size])
            for begin in range(0, n, self.batch_size)
        ]

    def _check_closure(self, closure: ClosurePattern | None) -> ClosurePattern | None:
        if closure is None:
            return None
        if not self._closure:
            raise ValueError("closure applies with crack_closure 'axes' only; this operator has none")
        if not isinstance(closure, ClosurePattern):
            raise TypeError("closure must be a ClosurePattern")
        expected = (None if self._cl_k is None else (self.n_elements, 8, 4), None if self._cb_index is None else (self.n_closure_band, 8, 4))
        for name, value, shape in zip(("iso", "band"), closure.groups(), expected):
            if shape is None:
                if value is not None:
                    raise ValueError(f"closure.{name} must be None: the model has no {name} closure group")
                continue
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                raise TypeError(f"closure.{name} must be a torch.bool tensor")
            if tuple(value.shape) != shape or value.device != self.device:
                raise ValueError(f"closure.{name} must have shape {shape} on {self.device}")
        return closure

    def _iso_closure_values(self, eps: torch.Tensor) -> torch.Tensor:
        """``(tr eps, eps_xx, eps_yy, eps_zz)`` ``(nb, 8, 4)``."""

        assert self._cl_rows is not None
        return torch.matmul(eps, self._cl_rows.T)

    def _iso_closure_stress(self, eps: torch.Tensor, k: torch.Tensor, pattern: torch.Tensor | None) -> torch.Tensor:
        """``sigma_c`` ``(nb, 8, 6)`` of the isotropic closure with per-element moduli ``k`` ``(nb, 4)``."""

        values = self._iso_closure_values(eps)
        active = values < 0.0 if pattern is None else pattern
        coefficient = torch.where(active, values, torch.zeros_like(values)) * k.unsqueeze(1)
        return torch.matmul(coefficient, self._cl_rows)

    def _closure_values(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Closure functionals ``v . eps`` of the two groups: iso ``(E, 8, 4)`` and band ``(n_cb, 8, 4)``."""

        u, E = self.split_mech(x)
        u_flat = u.reshape(-1)
        dofs_all = self.pmap.element_dofs
        iso = band = None
        if self._cl_k is not None:
            iso = torch.empty((self.n_elements, 8, 4), dtype=self.dtype, device=self.device)
            for begin in self._batches():
                end = min(begin + self.batch_size, self.n_elements)
                iso[begin:end] = self._iso_closure_values(self._qp_strain(u_flat, E, dofs_all[begin:end]))
        if self._cb_index is not None:
            band = torch.empty((self.n_closure_band, 8, 4), dtype=self.dtype, device=self.device)
            for rows, elements in self._cb_chunks():
                eps = self._qp_strain(u_flat, E, dofs_all[elements])
                band[rows] = torch.matmul(self._cb_rows[rows].unsqueeze(1), eps.unsqueeze(-1)).squeeze(-1)
        return iso, band

    def closure_pattern(self, x: torch.Tensor) -> ClosurePattern:
        """Sign pattern ``v . eps < 0`` of every closure term at state ``x``."""

        if not self._closure:
            raise ValueError("closure_pattern needs crack_closure 'axes'")
        iso, band = self._closure_values(x)
        if iso is not None:
            assert self._cl_iso_mask is not None
            iso = (iso < 0.0) & self._cl_iso_mask.view(-1, 1, 1)
        return ClosurePattern(
            iso=iso,
            band=None if band is None else band < 0.0,
        )

    def closure_pattern_change(
        self,
        x: torch.Tensor,
        pattern: ClosurePattern,
        g: torch.Tensor,
        *,
        tolerance: float = 1.0e-8,
    ) -> tuple[ClosurePattern, bool]:
        """New closure pattern of state ``x`` and whether it changed materially.

        A flip of bit ``k`` at a quadrature point changes the stress by
        ``(1 - g) k_k |v_k . eps|``; it counts only where that exceeds
        ``tolerance`` times the largest ``k_k |v_k . eps|`` of its group
        """

        pattern = self._check_closure(pattern)
        assert pattern is not None
        g = self._check_qp("g", g)
        assert g is not None
        iso, band = self._closure_values(x)
        changed = False
        groups = ((iso, pattern.iso, self._cl_k, g, self._cl_iso_mask), (band, pattern.band, self._cb_k, None if self._cb_index is None else g[self._cb_index], None))
        results: list[torch.Tensor | None] = []
        for values, old, moduli, g_group, mask in groups:
            if values is None:
                results.append(None)
                continue
            new = values < 0.0
            if mask is not None:
                new = new & mask.view(-1, 1, 1)
            strength = moduli.unsqueeze(1) * values.abs()
            flipped = new != old
            if not changed and bool(flipped.any()):
                jump = (1.0 - g_group).unsqueeze(-1) * strength
                changed = float(jump[flipped].max()) > tolerance * float(strength.max())
            results.append(new)
        new_iso, new_band = results
        return ClosurePattern(iso=new_iso, band=new_band), changed

    # ------------------------------------------------------------------
    # layout helpers
    # ------------------------------------------------------------------

    def _check(self, name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError(f"{name} must be {self.dtype} on {self.device}")
        return value

    def _check_qp(self, name: str, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return self._check(name, value, (self.n_elements, 8))

    def _batches(self, n: int | None = None) -> range:
        return range(0, self.n_elements if n is None else n, self.batch_size)

    def split_mech(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(u~ (n_red, 3), E (6,))``, views of ``x`` when it is contiguous."""

        self._check("x", x, (self.n_mech,))
        n = 3 * self.n_red
        return x[:n].reshape(self.n_red, 3), x[n:]

    def pack_mech(self, u: torch.Tensor, E: torch.Tensor | Any) -> torch.Tensor:
        """Return ``x = [u~ (3 n_red), E (6)]`` from ``u~`` ``(n_red, 3)`` or ``(3 n_red,)``."""

        u = torch.as_tensor(u, dtype=self.dtype, device=self.device).reshape(-1)
        E = torch.as_tensor(E, dtype=self.dtype, device=self.device).reshape(-1)
        if u.shape != (3 * self.n_red,) or E.shape != (6,):
            raise ValueError(
                f"u must hold {3 * self.n_red} entries and E 6, got {u.numel()} and {E.numel()}"
            )
        return torch.cat((u, E))

    def mech_fixed_mask(self) -> torch.Tensor:
        """Bool ``(n_mech,)``: pinned-node dofs, the controlled and the ``fixed_zero`` E components."""

        mask = torch.zeros(self.n_mech, dtype=torch.bool, device=self.device)
        begin = 3 * self.pinned_node
        mask[begin : begin + 3] = True
        offset = 3 * self.n_red
        mask[offset + self.load.controlled] = True
        for component in self.load.fixed_zero:
            mask[offset + component] = True
        return mask

    def mech_fixed_values(self, e_c: float | torch.Tensor) -> torch.Tensor:
        """``(n_mech,)`` prescribed values: ``e_c`` at the controlled component, zero elsewhere."""

        values = torch.zeros(self.n_mech, dtype=self.dtype, device=self.device)
        values[3 * self.n_red + self.load.controlled] = torch.as_tensor(
            e_c, dtype=self.dtype, device=self.device
        )
        return values

    # ------------------------------------------------------------------
    # mechanics
    # ------------------------------------------------------------------

    def degradation(self, d: torch.Tensor) -> torch.Tensor:
        """Quadrature-point degradation ``g (E, 8)`` with ``d_q = d_e @ N^T``."""

        self._check("d", d, (self.n_red,))
        d_q = d[self.pmap.element_nodes] @ self._N.T
        if self._pfczm:
            return self._pfczm_derivatives(d_q, self._a1)[0]
        return _degradation(d_q, self.kappa)

    def _qp_strain(self, u_flat: torch.Tensor, E: torch.Tensor, dofs: torch.Tensor) -> torch.Tensor:
        """Total QP strain ``(nb, 8, 6)`` of the elements with dofs ``(nb, 24)``."""

        u_e = u_flat[dofs]
        return (u_e @ self._B_flat).view(dofs.shape[0], 8, 6) + E

    def _stiffness(self, elements: slice | torch.Tensor) -> torch.Tensor:
        return self.mat.stiffness_bank[self.mat.stiffness_index[elements]]

    # ------------------------------------------------------------------
    # interface driver helpers
    # ------------------------------------------------------------------

    def _if_chunks(self) -> list[tuple[slice, torch.Tensor]]:
        """``(rows, element ids)`` batches of the interface-driver elements."""

        if self._if_index is None:
            return []
        n = self.n_interface
        return [
            (slice(begin, min(begin + self.batch_size, n)), self._if_index[begin : begin + self.batch_size])
            for begin in range(0, n, self.batch_size)
        ]

    def _if_stress_chunks(self) -> list[tuple[slice, torch.Tensor]]:
        """:meth:`_if_chunks` when the band stress is variational, else no batch."""

        return self._if_chunks() if self._if_variational else []

    def _check_opening(self, opening: torch.Tensor | None) -> torch.Tensor | None:
        if opening is None:
            return None
        if not self._if_variational:
            raise ValueError(
                "opening applies to the variational band stress only (band_stress 'variational' with "
                "interface-driver elements); the hybrid band stress is linear"
            )
        if not isinstance(opening, torch.Tensor) or opening.dtype != torch.bool:
            raise TypeError("opening must be a torch.bool tensor")
        if tuple(opening.shape) != (self.n_interface, 8) or opening.device != self.device:
            raise ValueError(f"opening must have shape ({self.n_interface}, 8) on {self.device}")
        return opening

    def interface_normal_traction(self, x: torch.Tensor) -> torch.Tensor:
        """Undamaged normal interface traction ``t_11 = m_n . C eps`` ``(n_if, 8)`` of the interface elements."""

        u, E = self.split_mech(x)
        u_flat = u.reshape(-1)
        out = torch.empty((self.n_interface, 8), dtype=self.dtype, device=self.device)
        dofs_all = self.pmap.element_dofs
        assert self._if_M is not None or self.n_interface == 0
        for rows, elements in self._if_chunks():
            eps = self._qp_strain(u_flat, E, dofs_all[elements])
            sigma = torch.matmul(eps, self._stiffness(elements).transpose(-1, -2))
            out[rows] = (sigma * self._if_M[rows, 0, :].unsqueeze(1)).sum(dim=-1)
        return out

    def interface_opening(self, x: torch.Tensor) -> torch.Tensor:
        """Normal-opening pattern ``t_11 > 0`` ``(n_if, 8)`` bool of state ``x``."""

        return self.interface_normal_traction(x) > 0.0

    def _interface_correction(
        self,
        eps: torch.Tensor,
        rows: slice,
        elements: torch.Tensor,
        g: torch.Tensor,
        opening: torch.Tensor | None,
    ) -> torch.Tensor:
        """``(1 - g)(C eps - C q)`` ``(nb, 8, 6)``: interface stress minus the hybrid ``g C eps``."""

        C = self._stiffness(elements)
        sigma0 = torch.matmul(eps, C.transpose(-1, -2))
        positive = interface_positive_stress(
            eps,
            C.unsqueeze(1),
            self._if_M[rows].unsqueeze(1),
            self._if_inv_a[rows].unsqueeze(1),
            opening=opening,
        )
        return (1.0 - g).unsqueeze(-1) * (sigma0 - positive)

    def _band_closure_correction(
        self, eps: torch.Tensor, rows: slice, g: torch.Tensor, pattern: torch.Tensor | None
    ) -> torch.Tensor:
        """``(1 - g) sigma_c`` ``(nb, 8, 6)`` of the band closure."""

        sigma_c = closure_stress(eps, self._cb_rows[rows].unsqueeze(1), self._cb_k[rows].unsqueeze(1), pattern=pattern)
        return (1.0 - g).unsqueeze(-1) * sigma_c

    def _stress_pass_closure(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None,
        *,
        assemble: bool,
        opening: torch.Tensor | None,
        closure: ClosurePattern | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """:meth:`_stress_pass` with the crack closure of .

        The batch loop is that of the default pass with the isotropic
        ``(1 - g) sigma_c`` fused in (moduli zero outside the isotropic
        group); then the variational interface loop and the band closure loop.  At
        ``g = None`` (undamaged) every add-back vanishes.
        """

        u, E = self.split_mech(x)
        g = self._check_qp("g", g)
        opening = self._check_opening(opening)
        u_flat = u.reshape(-1)
        r_u = torch.zeros(3 * self.n_red, dtype=self.dtype, device=self.device) if assemble else None
        r_E = torch.zeros(6, dtype=self.dtype, device=self.device)
        energy = torch.zeros((), dtype=self.dtype, device=self.device)
        dofs_all = self.pmap.element_dofs
        iso_k = self._cl_k if g is not None else None
        for begin in self._batches():
            end = min(begin + self.batch_size, self.n_elements)
            dofs = dofs_all[begin:end]
            eps = self._qp_strain(u_flat, E, dofs)
            sigma = torch.matmul(eps, self._stiffness(slice(begin, end)).transpose(-1, -2))
            if g is not None:
                g_b = g[begin:end].unsqueeze(-1)
                sigma = g_b * sigma
                if iso_k is not None:
                    pattern = None if closure is None else closure.iso[begin:end]
                    sigma = sigma + (1.0 - g_b) * self._iso_closure_stress(eps, iso_k[begin:end], pattern)
            weighted = sigma * self._w.view(1, 8, 1)
            r_E += weighted.sum(dim=(0, 1))
            energy += 0.5 * (weighted * eps).sum()
            if r_u is not None:
                local = weighted.reshape(-1, 48) @ self._B_flat.T
                r_u.scatter_add_(0, dofs.reshape(-1), local.reshape(-1))
        corrections: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        if g is not None:
            for rows, elements in self._if_stress_chunks():
                dofs = dofs_all[elements]
                eps = self._qp_strain(u_flat, E, dofs)
                correction = self._interface_correction(
                    eps, rows, elements, g[elements], None if opening is None else opening[rows]
                )
                corrections.append((dofs, eps, correction))
        if g is not None:
            for rows, elements in self._cb_chunks():
                dofs = dofs_all[elements]
                eps = self._qp_strain(u_flat, E, dofs)
                correction = self._band_closure_correction(
                    eps, rows, g[elements], None if closure is None else closure.band[rows]
                )
                corrections.append((dofs, eps, correction))
        for dofs, eps, correction in corrections:
            weighted = correction * self._w.view(1, 8, 1)
            r_E += weighted.sum(dim=(0, 1))
            energy += 0.5 * (weighted * eps).sum()
            if r_u is not None:
                local = weighted.reshape(-1, 48) @ self._B_flat.T
                r_u.scatter_add_(0, dofs.reshape(-1), local.reshape(-1))
        return r_u, r_E, energy

    def _stress_pass(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None,
        *,
        assemble: bool,
        opening: torch.Tensor | None = None,
        closure: ClosurePattern | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Return ``(r_u or None, r_E, elastic energy)`` for state ``x``.

        Interface-driver elements with the variational band stress get the
        correction of :meth:`_interface_correction` (only when ``g`` is given;
        at ``g = 1`` the interface stress is ``C eps``; with the hybrid band
        stress there is no correction) with the normal-opening pattern
        ``opening`` ``(n_if, 8)``, or the pattern of ``x`` itself when
        ``opening`` is ``None``.  With
        the crack closure the pass is :meth:`_stress_pass_closure`
        with the closure pattern ``closure`` (each point's own pattern when
        ``None``); ``closure`` is refused without it.
        """

        closure = self._check_closure(closure)
        if self._closure:
            return self._stress_pass_closure(
                x, g, assemble=assemble, opening=opening, closure=closure
            )
        u, E = self.split_mech(x)
        g = self._check_qp("g", g)
        opening = self._check_opening(opening)
        u_flat = u.reshape(-1)
        r_u = torch.zeros(3 * self.n_red, dtype=self.dtype, device=self.device) if assemble else None
        r_E = torch.zeros(6, dtype=self.dtype, device=self.device)
        energy = torch.zeros((), dtype=self.dtype, device=self.device)
        dofs_all = self.pmap.element_dofs
        for begin in self._batches():
            end = min(begin + self.batch_size, self.n_elements)
            dofs = dofs_all[begin:end]
            eps = self._qp_strain(u_flat, E, dofs)
            sigma = torch.matmul(eps, self._stiffness(slice(begin, end)).transpose(-1, -2))
            if g is not None:
                sigma = g[begin:end].unsqueeze(-1) * sigma
            weighted = sigma * self._w.view(1, 8, 1)
            r_E += weighted.sum(dim=(0, 1))
            energy += 0.5 * (weighted * eps).sum()
            if r_u is not None:
                local = weighted.reshape(-1, 48) @ self._B_flat.T
                r_u.scatter_add_(0, dofs.reshape(-1), local.reshape(-1))
        if g is not None:
            for rows, elements in self._if_stress_chunks():
                dofs = dofs_all[elements]
                eps = self._qp_strain(u_flat, E, dofs)
                correction = self._interface_correction(
                    eps, rows, elements, g[elements], None if opening is None else opening[rows]
                )
                weighted = correction * self._w.view(1, 8, 1)
                r_E += weighted.sum(dim=(0, 1))
                energy += 0.5 * (weighted * eps).sum()
                if r_u is not None:
                    local = weighted.reshape(-1, 48) @ self._B_flat.T
                    r_u.scatter_add_(0, dofs.reshape(-1), local.reshape(-1))
        return r_u, r_E, energy

    def mech_action(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
        *,
        opening: torch.Tensor | None = None,
        closure: ClosurePattern | None = None,
    ) -> torch.Tensor:
        """Apply ``A(g)`` to ``x``; ``g=None`` means undamaged (``g = 1``).

        With interface-driver elements of the variational band stress and
        ``g`` given, ``opening`` ``(n_if, 8)`` fixes their normal-opening
        pattern so that the action is linear in ``x``; ``None`` uses the
        pattern of ``x`` (the consistent internal force, not a linear action).
        With the hybrid band stress the action is linear and
        ``opening`` is refused.  With the crack closure ``closure`` fixes the closure
        pattern (:meth:`closure_pattern`).
        """

        r_u, r_E, _ = self._stress_pass(
            x, g, assemble=True, opening=opening, closure=closure
        )
        assert r_u is not None
        return torch.cat((r_u, r_E))

    def mech_diagonal(
        self,
        g: torch.Tensor | None = None,
        *,
        opening: torch.Tensor | None = None,
        closure: ClosurePattern | None = None,
    ) -> torch.Tensor:
        """Jacobi diagonal ``(n_mech,)`` of ``A(g)``.

        The ``u~`` block scatters the element diagonals
        ``sum_q w g diag(B_q^T C B_q)``; the ``E`` block is
        ``sum_e sum_q w g C_kk``.  On a periodic axis with a single voxel two
        local nodes share a reduced node and their coupling is not on this
        diagonal, so it is then a positive Jacobi estimate rather than the
        exact assembled diagonal.  Interface-driver elements with the
        variational band stress add ``(1 - g) diag(B^T (C - C K_s C) B)`` with
        ``K_s = sum_k s_k m_k m_k^T / a_k`` (``s`` from ``opening``, all
        open when ``None``).  With the crack closure every
        closure term ``k (v . eps)^2/2`` of an active bit adds
        ``sum_q w (1 - g) k (B_q^T v)^2`` (``closure`` fixes the bits, all
        active when ``None``).
        """

        closure = self._check_closure(closure)
        g = self._check_qp("g", g)
        opening = self._check_opening(opening)
        diag_u = torch.zeros(3 * self.n_red, dtype=self.dtype, device=self.device)
        diag_E = torch.zeros(6, dtype=self.dtype, device=self.device)
        dofs_all = self.pmap.element_dofs
        for begin in self._batches():
            end = min(begin + self.batch_size, self.n_elements)
            C = self._stiffness(slice(begin, end))
            nb = end - begin
            gw = self._w.expand(nb, 8) if g is None else g[begin:end] * self._w
            per_qp = (C.reshape(nb, 36) @ self._BB.T).view(nb, 8, 24)
            local = (gw.unsqueeze(-1) * per_qp).sum(dim=1)
            diag_u.scatter_add_(0, dofs_all[begin:end].reshape(-1), local.reshape(-1))
            diag_E += (gw.sum(dim=1).unsqueeze(-1) * torch.diagonal(C, dim1=-2, dim2=-1)).sum(dim=0)
        if g is not None:
            B = self.ref.B
            for rows, elements in self._if_stress_chunks():
                C = self._stiffness(elements)
                nb = C.shape[0]
                CM = torch.matmul(self._if_M[rows], C)  # rows (C m_k)^T, C symmetric
                per_qp = (C.reshape(nb, 36) @ self._BB.T).view(nb, 8, 24)
                BtCm = torch.einsum("qia,eki->eqka", B, CM)
                coef = self._if_inv_a[rows].unsqueeze(1).expand(nb, 8, 3).clone()
                if opening is not None:
                    coef[..., 0] = coef[..., 0] * opening[rows].to(self.dtype)
                reduction = (coef.unsqueeze(-1) * BtCm.square()).sum(dim=2)
                weight = (1.0 - g[elements]) * self._w
                local = (weight.unsqueeze(-1) * (per_qp - reduction)).sum(dim=1)
                diag_u.scatter_add_(0, self.pmap.element_dofs[elements].reshape(-1), local.reshape(-1))
                reduction_E = (coef.unsqueeze(-1) * CM.unsqueeze(1).square()).sum(dim=2)
                diag_C = torch.diagonal(C, dim1=-2, dim2=-1).unsqueeze(1)
                diag_E += (weight.unsqueeze(-1) * (diag_C - reduction_E)).sum(dim=(0, 1))
        if self._closure and g is not None:
            self._closure_diagonal(diag_u, diag_E, g, closure)
        return torch.cat((diag_u, diag_E))

    def _closure_rows_diagonal(
        self,
        diag_u: torch.Tensor,
        diag_E: torch.Tensor,
        elements: torch.Tensor,
        rows: torch.Tensor,
        coefficient: torch.Tensor,
    ) -> None:
        """Add ``sum_q sum_k c_qk (B_q^T v_k)^2`` and ``sum_q sum_k c_qk v_k^2`` for rows ``v`` ``(nb, K, 6)``."""

        B = self.ref.B
        local = torch.zeros((rows.shape[0], 24), dtype=self.dtype, device=self.device)
        for k in range(rows.shape[1]):
            Bv = torch.einsum("qia,ei->eqa", B, rows[:, k])
            local += (coefficient[..., k].unsqueeze(-1) * Bv.square()).sum(dim=1)
        diag_u.scatter_add_(0, self.pmap.element_dofs[elements].reshape(-1), local.reshape(-1))
        diag_E += torch.einsum("ek,eki->i", coefficient.sum(dim=1), rows.square())

    def _closure_diagonal(
        self,
        diag_u: torch.Tensor,
        diag_E: torch.Tensor,
        g: torch.Tensor,
        closure: ClosurePattern | None,
    ) -> None:
        """Closure terms of :meth:`mech_diagonal`."""

        dofs_all = self.pmap.element_dofs
        w = self._w.view(1, 8, 1)
        if self._cl_k is not None:
            BtR2 = self._cl_BtR.square()
            R2 = self._cl_rows.square()
            for begin in self._batches():
                end = min(begin + self.batch_size, self.n_elements)
                coefficient = (1.0 - g[begin:end]).unsqueeze(-1) * self._cl_k[begin:end].unsqueeze(1) * w
                if closure is not None:
                    coefficient = coefficient * closure.iso[begin:end].to(self.dtype)
                local = torch.einsum("eqk,qka->ea", coefficient, BtR2)
                diag_u.scatter_add_(0, dofs_all[begin:end].reshape(-1), local.reshape(-1))
                diag_E += torch.einsum("ek,ki->i", coefficient.sum(dim=1), R2)
        if self._cb_index is not None:
            for rows, elements in self._cb_chunks():
                coefficient = (1.0 - g[elements]).unsqueeze(-1) * self._cb_k[rows].unsqueeze(1) * w
                if closure is not None:
                    coefficient = coefficient * closure.band[rows].to(self.dtype)
                self._closure_rows_diagonal(diag_u, diag_E, elements, self._cb_rows[rows], coefficient)

    def macro_stress(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
        *,
        opening: torch.Tensor | None = None,
        closure: ClosurePattern | None = None,
    ) -> torch.Tensor:
        """Macro stress ``Sigma = r_E / V`` ``(6,)`` (tensor); ``closure`` as in :meth:`mech_action`."""

        _, r_E, _ = self._stress_pass(
            x, g, assemble=False, opening=opening, closure=closure
        )
        return r_E / self.volume

    def elastic_energy(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
        *,
        opening: torch.Tensor | None = None,
        closure: ClosurePattern | None = None,
    ) -> torch.Tensor:
        """Stored elastic energy ``0.5 x.A(g) x`` in mJ (0-dim tensor).

        For interface-driver elements with the variational band stress this is
        ``sum w (psi0 - (1 - g) psi_plus)`` at the pattern of ``x`` (or
        ``opening``; with the hybrid band stress ``sum w g psi0``); the degraded energy is
        positively homogeneous of degree two, so it still equals
        ``0.5 x.A(g) x``.  With the crack
        closure every element adds ``sum w (1 - g) psi_c`` at the pattern of ``x`` or of
        ``closure``.
        """

        _, _, energy = self._stress_pass(
            x, g, assemble=False, opening=opening, closure=closure
        )
        return energy

    def mean_strain(self, x: torch.Tensor) -> torch.Tensor:
        """Volume-average total engineering strain ``<eps>`` ``(6,)`` (tensor).

        ``<eps> = E + sym <grad u~>``.  On a periodic axis the fluctuation
        contributes nothing; on a free axis ``a`` of length ``L_a``,
        ``<du~/dx_a> = (<u~>_top - <u~>_bottom) / L_a`` with the area means of
        the bilinear face interpolant (uniform nodal weights along periodic
        in-plane axes, trapezoid weights along free ones), which is exact for
        trilinear voxels.  For a triply periodic cell ``<eps> = E``.
        """

        u, E = self.split_mech(x)
        rx, ry, rz = self.pmap.reduced_shape
        field = u.reshape(rz, ry, rx, 3)
        weights = []
        for n_red, periodic in zip((rx, ry, rz), self.grid.periodic):
            w = torch.ones(n_red, dtype=self.dtype, device=self.device)
            if not periodic:
                w[0] = 0.5
                w[-1] = 0.5
            weights.append(w / w.sum())
        wx, wy, wz = weights
        lengths = self.grid.lengths
        grad = torch.zeros((3, 3), dtype=self.dtype, device=self.device)  # grad[i, a] = <du_i/dx_a>
        if not self.grid.periodic[0]:
            top = torch.einsum("kj,kjc->c", wz[:, None] * wy[None, :], field[:, :, -1, :])
            bottom = torch.einsum("kj,kjc->c", wz[:, None] * wy[None, :], field[:, :, 0, :])
            grad[:, 0] = (top - bottom) / lengths[0]
        if not self.grid.periodic[1]:
            top = torch.einsum("ki,kic->c", wz[:, None] * wx[None, :], field[:, -1, :, :])
            bottom = torch.einsum("ki,kic->c", wz[:, None] * wx[None, :], field[:, 0, :, :])
            grad[:, 1] = (top - bottom) / lengths[1]
        if not self.grid.periodic[2]:
            top = torch.einsum("ji,jic->c", wy[:, None] * wx[None, :], field[-1])
            bottom = torch.einsum("ji,jic->c", wy[:, None] * wx[None, :], field[0])
            grad[:, 2] = (top - bottom) / lengths[2]
        jump = torch.stack(
            (
                grad[0, 0],
                grad[1, 1],
                grad[2, 2],
                grad[1, 2] + grad[2, 1],
                grad[0, 2] + grad[2, 0],
                grad[0, 1] + grad[1, 0],
            )
        )
        return E + jump

    def hill_mandel(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
        *,
        closure: ClosurePattern | None = None,
    ) -> dict[str, float]:
        """Compare ``<sigma.eps>`` with ``Sigma.E``.

        Returns ``mean_work = 2 E_el/V``, ``macro_work = Sigma.E`` and
        ``relative = |mean - macro| / max(|mean|, |macro|)`` (zero when both
        vanish), as floats.
        """

        _, E = self.split_mech(x)
        if closure is not None:
            (_, r_E, energy) = self._stress_pass(x, g, assemble=False, closure=closure)
        else:
            (_, r_E, energy) = self._stress_pass(x, g, assemble=False)
        mean_work = float(2.0 * energy / self.volume)
        macro_work = float(torch.dot(r_E / self.volume, E))
        scale = max(abs(mean_work), abs(macro_work))
        relative = abs(mean_work - macro_work) / scale if scale > 0.0 else 0.0
        return {"mean_work": mean_work, "macro_work": macro_work, "relative": relative}

    def driving_force(self, x: torch.Tensor) -> torch.Tensor:
        """Undamaged damage-driving energy ``Y (E, 8)``, dispatched on the element driver.

        ``none``: 0; ``full``: ``0.5 eps.C.eps``; ``spectral``: Miehe tensile
        energy with the element ``lam, mu``; ``interface``: the interface
        traction energy ``psi_plus`` of
        :func:`~phast.solid3d.splits.interface_energies`.
        The history maximum is left to the caller.
        """

        u, E = self.split_mech(x)
        u_flat = u.reshape(-1)
        Y = torch.zeros((self.n_elements, 8), dtype=self.dtype, device=self.device)
        mat = self.mat
        dofs_all = self.pmap.element_dofs
        for code, index in self._driver_groups:
            for begin in self._batches(index.numel()):
                chunk = index[begin : begin + self.batch_size]
                eps = self._qp_strain(u_flat, E, dofs_all[chunk])
                if code == DRIVER_FULL:
                    value = full_driving_energy(eps, self._stiffness(chunk).unsqueeze(1))
                elif code == DRIVER_SPECTRAL:
                    value = spectral_tensile_energy(eps, mat.lam[chunk, None], mat.mu[chunk, None])
                elif code == DRIVER_INTERFACE:
                    rows = slice(begin, begin + chunk.numel())
                    value = interface_driving_energy(eps, self._stiffness(chunk).unsqueeze(1), self._if_M[rows].unsqueeze(1), self._if_inv_a[rows].unsqueeze(1))
                else:
                    raise ValueError(f'unknown driver code {code}')
                Y[chunk] = value
        return Y

    # ------------------------------------------------------------------
    # damage
    # ------------------------------------------------------------------

    def damage_action(self, d: torch.Tensor, H: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the unscaled damage operator; ``H=None`` drops the history term."""

        self._check("d", d, (self.n_red,))
        H = self._check_qp("H", H)
        out = torch.zeros(self.n_red, dtype=self.dtype, device=self.device)
        nodes_all = self.pmap.element_nodes
        for begin in self._batches():
            end = min(begin + self.batch_size, self.n_elements)
            nodes = nodes_all[begin:end]
            d_e = d[nodes]
            y = self.mat.Gc[begin:end].unsqueeze(-1) * (d_e @ self._K_damage)
            if H is not None:
                d_q = d_e @ self._N.T
                y = y + self._c_history * ((H[begin:end] * self._w) * d_q) @ self._N
            out.scatter_add_(0, nodes.reshape(-1), y.reshape(-1))
        return out

    def damage_rhs(self, H: torch.Tensor | None) -> torch.Tensor:
        """Unscaled right-hand side ``2(1-kappa) sum_q w H_q N_q`` (zero for ``H=None``)."""

        H = self._check_qp("H", H)
        out = torch.zeros(self.n_red, dtype=self.dtype, device=self.device)
        if H is None:
            return out
        nodes_all = self.pmap.element_nodes
        for begin in self._batches():
            end = min(begin + self.batch_size, self.n_elements)
            y = self._c_history * (H[begin:end] * self._w) @ self._N
            out.scatter_add_(0, nodes_all[begin:end].reshape(-1), y.reshape(-1))
        return out

    def damage_diagonal(self, H: torch.Tensor | None = None) -> torch.Tensor:
        """Unscaled Jacobi diagonal of the damage operator (scattered element diagonals)."""

        H = self._check_qp("H", H)
        out = torch.zeros(self.n_red, dtype=self.dtype, device=self.device)
        nodes_all = self.pmap.element_nodes
        for begin in self._batches():
            end = min(begin + self.batch_size, self.n_elements)
            y = self.mat.Gc[begin:end].unsqueeze(-1) * self._K_damage_diag
            if H is not None:
                y = y + self._c_history * (H[begin:end] * self._w) @ self._NN
            out.scatter_add_(0, nodes_all[begin:end].reshape(-1), y.reshape(-1))
        return out

    def damage_system(self, H: torch.Tensor | None, *, scaled: bool = True) -> H8ScalarSPDSystem:
        """Damage SPD system; action, rhs and diagonal are multiplied by ``damage_scale`` when ``scaled``."""

        H = self._check_qp("H", H)
        scale = self.damage_scale if scaled else 1.0

        def action(field: torch.Tensor) -> torch.Tensor:
            return scale * self.damage_action(field, H)

        return H8ScalarSPDSystem(
            action=action,
            rhs=scale * self.damage_rhs(H),
            diagonal=scale * self.damage_diagonal(H),
        )

    def fracture_energy(self, d: torch.Tensor) -> dict[str, Any]:
        """Fracture energy in mJ as floats, with a per-phase split (pores included).

        Per element ``Gc/(2 ell) d^T M d`` (``'local'``) plus
        ``Gc ell/2 d^T L d`` (``'gradient'``); ``'by_phase'`` maps every
        present phase to its total.  This is the phase-field surface
        functional ``Gamma`` with the element ``Gc``.  With
        the full-energy driver it equals the dissipated energy; the dissipated
        energy is also ``W - E_el``.
        """

        self._check("d", d, (self.n_red,))
        if self._pfczm:
            return self._pfczm_fracture_energy(d)
        local = torch.empty(self.n_elements, dtype=self.dtype, device=self.device)
        gradient = torch.empty_like(local)
        nodes_all = self.pmap.element_nodes
        M, L = self.ref.M, self.ref.L
        for begin in self._batches():
            end = min(begin + self.batch_size, self.n_elements)
            d_e = d[nodes_all[begin:end]]
            Gc = self.mat.Gc[begin:end]
            local[begin:end] = Gc / (2.0 * self.ell) * ((d_e @ M) * d_e).sum(dim=-1)
            gradient[begin:end] = 0.5 * Gc * self.ell * ((d_e @ L) * d_e).sum(dim=-1)
        by_id = torch.zeros(len(PHASE_NAMES), dtype=self.dtype, device=self.device)
        by_id.scatter_add_(0, self.mat.phase, local + gradient)
        totals = torch.stack((local.sum(), gradient.sum())).cpu().numpy()
        by_id_np = by_id.cpu().numpy()
        return {
            "total": float(totals[0] + totals[1]),
            "local": float(totals[0]),
            "gradient": float(totals[1]),
            "by_phase": {name: float(by_id_np[PHASE_ID[name]]) for name in self._phases_present},
        }

    def expand_damage(self, d: torch.Tensor) -> torch.Tensor:
        """Full nodal damage ``P d`` ``(N_full,)``."""

        self._check("d", d, (self.n_red,))
        return self.pmap.expand(d)

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------

    def _relative_node_coordinates(self) -> np.ndarray:
        if self._node_offsets is None:
            self._node_offsets = self.grid.node_coordinates() - np.asarray(self.grid.origin)
        return self._node_offsets

    def field_snapshot(
        self,
        step: int,
        time_value: float,
        x: torch.Tensor,
        d: torch.Tensor,
        H: torch.Tensor | None,
        g: torch.Tensor | None = None,
    ) -> FieldSnapshot:
        """Return the CPU :class:`~phast.solid3d.model.FieldSnapshot` of a state.

        Nodal: total displacement ``E_tensor (X - origin) + P u~``, the
        fluctuation ``P u~`` and ``P d``.  Element quadrature-point means of
        the total strain, the degraded stress ``g C eps``, ``H``, ``g psi0``
        and the driving energy ``Y``; ``history_max`` is the QP maximum of
        ``H``.  ``g`` defaults to :meth:`degradation` of ``d`` and ``H=None``
        to zero.  Interface-driver elements report the driving energy
        ``psi_plus``; with the variational band stress they also report their
        own stress ``C eps - (1 - g) C q`` and degraded energy
        ``psi0 - (1 - g) psi_plus``, with the hybrid band stress ``g C eps`` and ``g psi0`` like every other phase.
        With the
        crack closure the stress and ``psi_elastic`` include the
        closure add-back of each point's own pattern, and the snapshot adds
        ``psi_closure``, ``closure_active`` and ``det_F_min``.
        """

        u, E = self.split_mech(x)
        self._check("d", d, (self.n_red,))
        H = self._check_qp("H", H)
        g = self.degradation(d) if g is None else self._check_qp("g", g)
        assert g is not None
        u_flat = u.reshape(-1)
        strain = torch.empty((self.n_elements, 6), dtype=self.dtype, device=self.device)
        stress = torch.empty_like(strain)
        psi_elastic = torch.empty(self.n_elements, dtype=self.dtype, device=self.device)
        r_E = torch.zeros(6, dtype=self.dtype, device=self.device)
        dofs_all = self.pmap.element_dofs
        for begin in self._batches():
            end = min(begin + self.batch_size, self.n_elements)
            eps = self._qp_strain(u_flat, E, dofs_all[begin:end])
            sigma0 = torch.matmul(eps, self._stiffness(slice(begin, end)).transpose(-1, -2))
            g_b = g[begin:end]
            sigma = g_b.unsqueeze(-1) * sigma0
            strain[begin:end] = eps.mean(dim=1)
            stress[begin:end] = sigma.mean(dim=1)
            psi_elastic[begin:end] = (0.5 * g_b * (eps * sigma0).sum(dim=-1)).mean(dim=1)
            r_E += (sigma * self._w.view(1, 8, 1)).sum(dim=(0, 1))
        for rows, elements in self._if_stress_chunks():
            eps = self._qp_strain(u_flat, E, dofs_all[elements])
            g_e = g[elements]
            correction = self._interface_correction(eps, rows, elements, g_e, None)
            energies = interface_energies(
                eps,
                self._stiffness(elements).unsqueeze(1),
                self._if_M[rows].unsqueeze(1),
                self._if_inv_a[rows].unsqueeze(1),
            )
            stress[elements] = stress[elements] + correction.mean(dim=1)
            psi_elastic[elements] = psi_elastic[elements] + (
                (1.0 - g_e) * (energies.psi0 - energies.psi_plus)
            ).mean(dim=1)
            r_E += (correction * self._w.view(1, 8, 1)).sum(dim=(0, 1))
        closure_fields: dict[str, Any] = {}
        if self._closure:
            closure_fields, extra_energy, extra_stress, extra_r_E = self._closure_snapshot_fields(x, g)
            stress = stress + extra_stress
            psi_elastic = psi_elastic + extra_energy
            r_E = r_E + extra_r_E
        Y = self.driving_force(x)
        if H is None:
            H = torch.zeros((self.n_elements, 8), dtype=self.dtype, device=self.device)

        fluctuation = self.pmap.expand(u).cpu().numpy()
        E_tensor = engineering_strain_to_tensor(E).cpu().numpy()
        displacement = self._relative_node_coordinates() @ E_tensor.T + fluctuation
        return FieldSnapshot(
            step=step,
            time_value=time_value,
            macro_strain=E.cpu().numpy().copy(),
            macro_stress=(r_E / self.volume).cpu().numpy(),
            displacement=displacement,
            fluctuation=fluctuation,
            damage=self.pmap.expand(d).cpu().numpy(),
            strain=strain.cpu().numpy(),
            stress=stress.cpu().numpy(),
            history_mean=H.mean(dim=1).cpu().numpy(),
            history_max=H.amax(dim=1).cpu().numpy(),
            psi_elastic=psi_elastic.cpu().numpy(),
            psi_driving=Y.mean(dim=1).cpu().numpy(),
            **closure_fields,
        )

    def _closure_snapshot_fields(
        self, x: torch.Tensor, g: torch.Tensor
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Closure part of :meth:`field_snapshot`.

        Returns the three snapshot arrays and the element means of the
        isotropic and band add-back energy and stress with their macro
        stress sum.  ``psi_closure`` holds every add-back energy.
        """

        u, E = self.split_mech(x)
        u_flat = u.reshape(-1)
        dofs_all = self.pmap.element_dofs
        w = self._w.view(1, 8, 1)
        energy = torch.zeros(self.n_elements, dtype=self.dtype, device=self.device)
        stress = torch.zeros((self.n_elements, 6), dtype=self.dtype, device=self.device)
        r_E = torch.zeros(6, dtype=self.dtype, device=self.device)
        psi_closure = torch.zeros(self.n_elements, dtype=self.dtype, device=self.device)
        active = torch.zeros(self.n_elements, dtype=self.dtype, device=self.device)
        if self._cl_k is not None:
            assert self._cl_iso_mask is not None
            for begin in self._batches():
                end = min(begin + self.batch_size, self.n_elements)
                eps = self._qp_strain(u_flat, E, dofs_all[begin:end])
                one_minus_g = 1.0 - g[begin:end]
                k = self._cl_k[begin:end]
                correction = one_minus_g.unsqueeze(-1) * self._iso_closure_stress(eps, k, None)
                values = self._iso_closure_values(eps)
                psi_c = 0.5 * (k.unsqueeze(1) * values.clamp(max=0.0).square()).sum(dim=-1)
                stress[begin:end] = correction.mean(dim=1)
                energy[begin:end] = (one_minus_g * psi_c).mean(dim=1)
                r_E += (correction * w).sum(dim=(0, 1))
                bits = (values < 0.0) & self._cl_iso_mask[begin:end].view(-1, 1, 1)
                active[begin:end] = bits.any(dim=-1).to(self.dtype).mean(dim=1)
            psi_closure += energy
        if self._cb_index is not None:
            for rows, elements in self._cb_chunks():
                eps = self._qp_strain(u_flat, E, dofs_all[elements])
                one_minus_g = 1.0 - g[elements]
                correction = self._band_closure_correction(eps, rows, g[elements], None)
                values = torch.matmul(self._cb_rows[rows].unsqueeze(1), eps.unsqueeze(-1)).squeeze(-1)
                psi_c = 0.5 * (self._cb_k[rows].unsqueeze(1) * values.clamp(max=0.0).square()).sum(dim=-1)
                band_energy = (one_minus_g * psi_c).mean(dim=1)
                stress[elements] = stress[elements] + correction.mean(dim=1)
                energy[elements] = energy[elements] + band_energy
                psi_closure[elements] = psi_closure[elements] + band_energy
                r_E += (correction * w).sum(dim=(0, 1))
                active[elements] = (values < 0.0).any(dim=-1).to(self.dtype).mean(dim=1)
        _, det_min = self._element_kinematics(x)
        fields = {
            "psi_closure": psi_closure.cpu().numpy(),
            "closure_active": active.cpu().numpy(),
            "det_F_min": det_min.cpu().numpy(),
        }
        return fields, energy, stress, r_E

    # ------------------------------------------------------------------
    # kinematic guard
    # ------------------------------------------------------------------

    def _element_kinematics(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Element-mean total strain ``(E, 6)`` and the smallest quadrature-point ``det(I + grad u)`` ``(E,)``.

        ``grad u = E_tensor + grad u~`` with ``grad u~ = sum_a u~_a (x) grad N_a``
        at each quadrature point (the total displacement is
        ``E_tensor (X - origin) + P u~``).
        """

        u, E = self.split_mech(x)
        u_flat = u.reshape(-1)
        dofs_all = self.pmap.element_dofs
        F_macro = torch.eye(3, dtype=self.dtype, device=self.device) + engineering_strain_to_tensor(E)
        strain = torch.empty((self.n_elements, 6), dtype=self.dtype, device=self.device)
        det_min = torch.empty(self.n_elements, dtype=self.dtype, device=self.device)
        for begin in self._batches():
            end = min(begin + self.batch_size, self.n_elements)
            dofs = dofs_all[begin:end]
            strain[begin:end] = self._qp_strain(u_flat, E, dofs).mean(dim=1)
            u_e = u_flat[dofs].view(-1, 8, 3)
            grad = torch.einsum("eai,qaj->eqij", u_e, self.ref.grad_N)
            det_min[begin:end] = determinant_3x3(F_macro + grad).amin(dim=1)
        return strain, det_min

    def kinematic_guard(self, x: torch.Tensor, d: torch.Tensor) -> dict[str, float | None]:
        """Smallest normal strain, principal strain and ``det(I + grad u)`` of state ``x``.

        ``min_normal_strain`` is the smallest element-mean ``eps_xx, eps_yy,
        eps_zz``, ``min_principal_strain`` the smallest principal value of the
        element-mean strain tensors and ``min_det_F`` the smallest
        quadrature-point ``det(I + grad u)``; the ``damaged_*`` values are the
        same over the elements whose mean nodal damage exceeds
        ``cfg.damaged_threshold`` (``None`` when there are none).  The values
        are kinematic: they need no closure and are used for closure-off
        baselines as well.
        """

        self._check("d", d, (self.n_red,))
        strain, det_min = self._element_kinematics(x)
        normal = strain[:, :3].amin(dim=1)
        principal = principal_values(engineering_strain_to_tensor(strain))[:, 0]
        damaged = d[self.pmap.element_nodes].mean(dim=1) > self.cfg.damaged_threshold
        out: dict[str, float | None] = {
            "min_normal_strain": float(normal.min()),
            "min_principal_strain": float(principal.min()),
            "min_det_F": float(det_min.min()),
        }
        if bool(damaged.any()):
            out.update(
                {
                    "damaged_min_normal_strain": float(normal[damaged].min()),
                    "damaged_min_principal_strain": float(principal[damaged].min()),
                    "damaged_min_det_F": float(det_min[damaged].min()),
                }
            )
        else:
            out.update({"damaged_min_normal_strain": None, "damaged_min_principal_strain": None, "damaged_min_det_F": None})
        return out


__all__ = ["PeriodicVoxelPhaseFieldOperator"]
