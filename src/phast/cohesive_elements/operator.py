"""Torch-native cohesive interface residual and tangent operators.

This is the first solver-coupled slice for discrete cohesive interfaces. It
works on doubled-node zero-thickness interface records from
``insert_cohesive_layer`` and provides residual, sparse tangent assembly, and
commit/rollback history for a bilinear traction-separation law.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .cohesive_elements import CohesiveElement


@dataclass
class CohesiveState:
    max_delta_eff: torch.Tensor
    damage: torch.Tensor
    dissipated_energy: torch.Tensor

    @classmethod
    def zeros(cls, n_elems: int, n_q: int, *, device=None,
              dtype: torch.dtype = torch.float64) -> "CohesiveState":
        return cls(
            max_delta_eff=torch.zeros(
                (n_elems, n_q), device=device, dtype=dtype),
            damage=torch.zeros((n_elems, n_q), device=device, dtype=dtype),
            dissipated_energy=torch.zeros(
                (n_elems, n_q), device=device, dtype=dtype),
        )

    def clone(self) -> "CohesiveState":
        return CohesiveState(
            max_delta_eff=self.max_delta_eff.clone(),
            damage=self.damage.clone(),
            dissipated_energy=self.dissipated_energy.clone(),
        )


@dataclass(frozen=True)
class BilinearCohesiveLaw:
    """Bilinear mixed-mode cohesive law with scalar damage history.

    ``contact_stiffness`` is an optional normal penalty active only for
    negative normal jumps. The default ``0.0`` preserves traction-free normal
    compression.
    """

    k_n: float
    k_t: float
    sigma_max: float
    delta_c: float
    contact_stiffness: float = 0.0

    @property
    def delta_0(self) -> float:
        return float(self.sigma_max) / float(self.k_n)

    @property
    def fracture_energy(self) -> float:
        """Mode-I bilinear traction-separation area per unit interface length."""

        return 0.5 * float(self.sigma_max) * float(self.delta_c)

    def work_density(self, delta_eff: torch.Tensor) -> torch.Tensor:
        """Integrated scalar bilinear TSL work density up to ``delta_eff``.

        The current cohesive law uses a scalar damage history driven by the
        effective opening. This helper returns the matching bilinear
        traction-separation work for that scalar history. For pure mode-I
        opening it is the exact area under the traction-opening curve.
        """

        delta = torch.clamp(delta_eff, min=0.0)
        delta_0 = torch.as_tensor(
            self.delta_0, device=delta.device, dtype=delta.dtype)
        delta_c = torch.as_tensor(
            self.delta_c, device=delta.device, dtype=delta.dtype)
        k_n = torch.as_tensor(self.k_n, device=delta.device, dtype=delta.dtype)
        fracture_energy = torch.as_tensor(
            self.fracture_energy, device=delta.device, dtype=delta.dtype)

        elastic = 0.5 * k_n * delta * delta
        soft_delta = torch.clamp(delta, max=delta_c)
        onset_work = 0.5 * k_n * delta_0 * delta_0
        sigma_max = torch.as_tensor(
            self.sigma_max, device=delta.device, dtype=delta.dtype)
        softening = sigma_max / (delta_c - delta_0) * (
            delta_c * (soft_delta - delta_0)
            - 0.5 * (soft_delta * soft_delta - delta_0 * delta_0)
        )
        work = torch.where(
            delta <= delta_0,
            elastic,
            torch.where(
                delta < delta_c,
                onset_work + softening,
                fracture_energy,
            ),
        )
        return work

    def dissipated_energy_density(
        self,
        delta_eff: torch.Tensor,
        damage: torch.Tensor,
    ) -> torch.Tensor:
        """Irreversible scalar cohesive dissipation density.

        For pure mode-I monotonic opening this equals accumulated external
        cohesive work minus the recoverable damaged elastic energy and tends to
        ``fracture_energy`` at full separation.
        """

        delta = torch.clamp(delta_eff, min=0.0)
        k_n = torch.as_tensor(self.k_n, device=delta.device, dtype=delta.dtype)
        recoverable = 0.5 * (1.0 - damage) * k_n * delta * delta
        return torch.clamp(
            self.work_density(delta) - recoverable,
            min=0.0,
        )

    def evaluate(
        self,
        jump: torch.Tensor,
        state: CohesiveState,
    ) -> tuple[torch.Tensor, torch.Tensor, CohesiveState]:
        if self.k_n <= 0.0 or self.k_t <= 0.0:
            raise ValueError("cohesive stiffnesses must be positive")
        if self.contact_stiffness < 0.0:
            raise ValueError("contact_stiffness must be non-negative")
        if self.delta_c <= self.delta_0:
            raise ValueError("delta_c must be larger than sigma_max / k_n")

        delta_n_raw = jump[..., 0]
        delta_n = torch.clamp(delta_n_raw, min=0.0)
        delta_t = jump[..., 1]
        eps = torch.as_tensor(1.0e-30, device=jump.device, dtype=jump.dtype)
        delta_eff = torch.sqrt(delta_n * delta_n + delta_t * delta_t + eps)
        max_delta = torch.maximum(state.max_delta_eff, delta_eff)
        max_delta_safe = torch.clamp(
            max_delta,
            min=torch.finfo(jump.dtype).eps,
        )
        damage_raw = (
            self.delta_c
            * (max_delta - self.delta_0)
            / (max_delta_safe * (self.delta_c - self.delta_0))
        )
        damage = torch.clamp(damage_raw, min=0.0, max=1.0)
        dissipated_energy = self.dissipated_energy_density(max_delta, damage)
        k = torch.as_tensor([self.k_n, self.k_t], device=jump.device,
                            dtype=jump.dtype)
        traction = (1.0 - damage).unsqueeze(-1) * k * jump
        contact_normal = self.contact_stiffness * delta_n_raw
        traction = torch.where(
            (delta_n_raw < 0.0)[..., None],
            torch.stack([contact_normal, traction[..., 1]], dim=-1),
            traction,
        )
        tangent = torch.zeros(
            (*jump.shape[:-1], 2, 2), device=jump.device, dtype=jump.dtype)
        tangent[..., 0, 0] = torch.where(
            delta_n_raw < 0.0,
            torch.full_like(damage, float(self.contact_stiffness)),
            (1.0 - damage) * self.k_n,
        )
        tangent[..., 1, 1] = (1.0 - damage) * self.k_t
        loading = delta_eff >= state.max_delta_eff
        active = (
            loading
            & (delta_eff > self.delta_0)
            & (delta_eff < self.delta_c)
        )
        if torch.any(active):
            d_eff_safe = torch.clamp(delta_eff, min=torch.finfo(jump.dtype).eps)
            damage_slope = (
                self.delta_c * self.delta_0
                / ((self.delta_c - self.delta_0) * d_eff_safe * d_eff_safe)
            )
            dd_djump_n = torch.where(
                delta_n_raw > 0.0,
                damage_slope * delta_n / d_eff_safe,
                torch.zeros_like(delta_eff),
            )
            dd_djump_t = damage_slope * delta_t / d_eff_safe
            grad_d = torch.stack([dd_djump_n, dd_djump_t], dim=-1)
            k_jump = torch.stack(
                [self.k_n * jump[..., 0], self.k_t * jump[..., 1]], dim=-1)
            softening = k_jump.unsqueeze(-1) * grad_d.unsqueeze(-2)
            tangent = torch.where(
                active[..., None, None],
                tangent - softening,
                tangent,
            )
            tangent[..., 0, :] = torch.where(
                (delta_n_raw < 0.0)[..., None],
                torch.stack([
                    torch.full_like(damage, float(self.contact_stiffness)),
                    torch.zeros_like(damage),
                ], dim=-1),
                tangent[..., 0, :],
            )
        return traction, tangent, CohesiveState(
            max_delta,
            damage,
            dissipated_energy,
        )


class CohesiveInterfaceOperator:
    """Stateful residual/tangent operator for zero-thickness interfaces."""

    def __init__(
        self,
        cohesives: Sequence[CohesiveElement],
        law: BilinearCohesiveLaw,
        *,
        n_nodes: int,
        device=None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.cohesives = list(cohesives)
        self.law = law
        self.n_nodes = int(n_nodes)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = dtype
        self.gauss_points = torch.tensor(
            [-1.0 / 3.0 ** 0.5, 1.0 / 3.0 ** 0.5],
            device=self.device,
            dtype=dtype,
        )
        self.weights = torch.ones(2, device=self.device, dtype=dtype)
        self.state = CohesiveState.zeros(
            len(self.cohesives), 2, device=self.device, dtype=dtype)
        self._trial_state: CohesiveState | None = None

    def _element_kinematics(self, u: torch.Tensor):
        n_elem = len(self.cohesives)
        B = torch.zeros((n_elem, 2, 2, 8), device=u.device, dtype=u.dtype)
        conn = torch.zeros((n_elem, 4), device=u.device, dtype=torch.long)
        lengths = torch.zeros(n_elem, device=u.device, dtype=u.dtype)
        for e, ce in enumerate(self.cohesives):
            conn[e] = torch.tensor(
                [*ce.nodes_top, *ce.nodes_bottom],
                device=u.device,
                dtype=torch.long,
            )
            lengths[e] = float(ce.length)
            normal = torch.as_tensor(ce.normal, device=u.device, dtype=u.dtype)
            tangent = torch.as_tensor(ce.tangent, device=u.device, dtype=u.dtype)
            for q, xi in enumerate(self.gauss_points.to(device=u.device, dtype=u.dtype)):
                N = torch.stack([(1.0 - xi) * 0.5, (1.0 + xi) * 0.5])
                for a in range(2):
                    B[e, q, 0, 2 * a:2 * a + 2] = N[a] * normal
                    B[e, q, 1, 2 * a:2 * a + 2] = N[a] * tangent
                    b = a + 2
                    B[e, q, 0, 2 * b:2 * b + 2] = -N[a] * normal
                    B[e, q, 1, 2 * b:2 * b + 2] = -N[a] * tangent
        u_e = u[conn].reshape(n_elem, 8)
        jump = torch.einsum("eqai,ei->eqa", B, u_e)
        return conn, B, lengths, jump

    def update_trial(self, u: torch.Tensor) -> CohesiveState:
        _, _, _, jump = self._element_kinematics(u)
        _, _, trial = self.law.evaluate(jump, self.state)
        self._trial_state = trial
        return trial

    def commit(self) -> CohesiveState:
        if self._trial_state is None:
            raise RuntimeError("No cohesive trial state to commit")
        self.state = self._trial_state
        self._trial_state = None
        return self.state

    def rollback(self) -> None:
        self._trial_state = None

    def internal_force(self, u: torch.Tensor,
                       *, state: CohesiveState | None = None) -> torch.Tensor:
        conn, B, lengths, jump = self._element_kinematics(u)
        st = state if state is not None else self.state
        traction, _, trial = self.law.evaluate(jump, st)
        if state is None:
            self._trial_state = trial
        weight = lengths.view(-1, 1) * 0.5 * self.weights.to(u.device, u.dtype)
        f_e = torch.einsum("eqai,eqa,eq->ei", B, traction, weight)
        out = torch.zeros((self.n_nodes, 2), device=u.device, dtype=u.dtype)
        out.scatter_add_(
            0,
            conn.reshape(-1, 1).expand(-1, 2),
            f_e.reshape(-1, 4, 2).reshape(-1, 2),
        )
        return out

    def assemble_tangent(self, u: torch.Tensor,
                         *, state: CohesiveState | None = None) -> torch.Tensor:
        conn, B, lengths, jump = self._element_kinematics(u)
        st = state if state is not None else self.state
        _, D, trial = self.law.evaluate(jump, st)
        if state is None:
            self._trial_state = trial
        weight = lengths.view(-1, 1) * 0.5 * self.weights.to(u.device, u.dtype)
        Ke = torch.einsum("eqai,eqab,eqbj,eq->eij", B, D, B, weight)
        elem_dofs = torch.empty((len(self.cohesives), 8), device=u.device,
                                dtype=torch.long)
        for a in range(4):
            elem_dofs[:, 2 * a] = 2 * conn[:, a]
            elem_dofs[:, 2 * a + 1] = 2 * conn[:, a] + 1
        rows = elem_dofs.repeat_interleave(8, dim=1).reshape(-1)
        cols = elem_dofs.repeat(1, 8).reshape(-1)
        return torch.sparse_coo_tensor(
            torch.stack([rows, cols], dim=0),
            Ke.reshape(-1),
            (2 * self.n_nodes, 2 * self.n_nodes),
            device=u.device,
            dtype=u.dtype,
        ).coalesce()

    def integrated_dissipated_energy(
        self,
        *,
        state: CohesiveState | None = None,
    ) -> torch.Tensor:
        """Integrate scalar cohesive dissipation over all interface elements."""

        st = state if state is not None else self.state
        if len(self.cohesives) == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        lengths = torch.as_tensor(
            [ce.length for ce in self.cohesives],
            device=st.dissipated_energy.device,
            dtype=st.dissipated_energy.dtype,
        )
        weights = self.weights.to(
            device=st.dissipated_energy.device,
            dtype=st.dissipated_energy.dtype,
        )
        quadrature_weight = 0.5 * lengths.view(-1, 1) * weights.view(1, -1)
        return torch.sum(st.dissipated_energy * quadrature_weight)

    def integrated_fracture_energy_capacity(self) -> torch.Tensor:
        """Total mode-I bilinear cohesive energy capacity of the interface."""

        if len(self.cohesives) == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        total_length = sum(float(ce.length) for ce in self.cohesives)
        return torch.as_tensor(
            self.law.fracture_energy * total_length,
            device=self.device,
            dtype=self.dtype,
        )


__all__ = [
    "BilinearCohesiveLaw",
    "CohesiveInterfaceOperator",
    "CohesiveState",
]
