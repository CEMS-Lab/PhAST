"""Public contracts for learned damage updates in staggered FEM simulations.

The classical phase-field damage solver remains the default and authoritative
route. A learned model may be used in one of two explicitly selected modes:

``learned_proposal``
    The prediction is projected to the admissible interval and supplied as the
    initial guess to the classical damage solve.

``learned_replacement``
    The prediction replaces one damage subproblem only when it passes finite
    value, shape, bound, irreversibility, phase-field Dirichlet, and projected
    residual checks. A rejected prediction falls back to the classical solve
    unless fallback is disabled.

This module defines inference contracts. It does not define a network
architecture, training procedure, dataset, or generalization claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import torch


LEARNED_DAMAGE_MODES = {
    "classical",
    "learned_proposal",
    "learned_replacement",
}


@dataclass(frozen=True)
class DamageStepContext:
    """Read-only description of one damage-subproblem call.

    Tensor fields are detached views or clones supplied for inference. A model
    must return one nodal damage value per mesh node.
    """

    step: int
    time: float | None
    load_factor: float
    nodes: torch.Tensor
    elements: torch.Tensor
    displacement: torch.Tensor
    velocity: torch.Tensor
    history_element: torch.Tensor
    history_nodal: torch.Tensor
    damage_previous: torch.Tensor
    material: Mapping[str, Any]
    phase_field_model: str
    energy_split: str
    device: torch.device
    dtype: torch.dtype

    def canonical_node_features(self) -> torch.Tensor:
        """Return ``[x, y, H, d_prev, u_x, u_y]`` node features.

        The helper is intentionally small and architecture-neutral. Models that
        require graph edges, quadrature history, normalization, or additional
        material channels should construct them from the full context.
        """

        coordinates = self.nodes[:, :2].to(
            device=self.device, dtype=self.dtype)
        return torch.cat(
            (
                coordinates,
                self.history_nodal.reshape(-1, 1),
                self.damage_previous.reshape(-1, 1),
                self.displacement[:, :2],
            ),
            dim=1,
        )

    def graph_edge_index(self) -> torch.Tensor:
        """Return directed, duplicate-free mesh edges as ``[2, n_edges]``.

        Plug-ins may ignore this helper, construct node-element incidence
        operators, or add edge features. PhAST does not require PyTorch
        Geometric for the public predictor contract.
        """

        local_nodes = self.elements.shape[1]
        pairs = []
        for local_index in range(local_nodes):
            left = self.elements[:, local_index]
            right = self.elements[:, (local_index + 1) % local_nodes]
            pairs.append(torch.stack((left, right), dim=1))
            pairs.append(torch.stack((right, left), dim=1))
        edges = torch.cat(pairs, dim=0)
        return torch.unique(edges, dim=0).t().contiguous()

    def element_areas(self) -> torch.Tensor:
        """Return the area of every three-node triangle in the mesh."""

        if self.elements.shape[1] != 3:
            raise NotImplementedError(
                "element_areas currently covers three-node triangles; got "
                f"{self.elements.shape[1]} local nodes")
        xy = self.nodes[:, :2][self.elements]
        x0, x1, x2 = xy[:, 0], xy[:, 1], xy[:, 2]
        twice_area = (
            (x1[:, 0] - x0[:, 0]) * (x2[:, 1] - x0[:, 1])
            - (x2[:, 0] - x0[:, 0]) * (x1[:, 1] - x0[:, 1])
        )
        return 0.5 * twice_area.abs()

    def assemble_element_to_nodes(self, values: torch.Tensor) -> torch.Tensor:
        """Assemble ``sum_e int N_i value_e dOmega`` over the mesh.

        This is the finite-element integral of a linear shape function against
        a piecewise-constant element field, without mass inversion. Divide by
        the lumped mass returned for a unit field to obtain a nodal average
        instead.
        """

        values = values.reshape(-1)
        if values.shape[0] != self.elements.shape[0]:
            raise ValueError(
                "values must hold one entry per element; got "
                f"{values.shape[0]} for {self.elements.shape[0]} elements")
        n_local = self.elements.shape[1]
        weighted = (
            values * self.element_areas() / n_local
        ).repeat_interleave(n_local)
        accumulated = torch.zeros(
            self.nodes.shape[0], dtype=weighted.dtype, device=weighted.device)
        accumulated.index_add_(0, self.elements.reshape(-1), weighted)
        return accumulated

    def boundary_node_mask(self) -> torch.Tensor:
        """Flag nodes on the topological boundary of the mesh.

        An element edge shared by exactly one element lies on a boundary. The
        test is topological, so interior voids such as holes and slots are
        detected alongside the outer perimeter.
        """

        n_local = self.elements.shape[1]
        pairs = torch.cat(
            [
                torch.stack(
                    (
                        self.elements[:, local],
                        self.elements[:, (local + 1) % n_local],
                    ),
                    dim=1,
                )
                for local in range(n_local)
            ],
            dim=0,
        )
        undirected, counts = torch.unique(
            pairs.sort(dim=1).values, dim=0, return_counts=True)
        boundary_edges = undirected[counts == 1]
        mask = torch.zeros(
            self.nodes.shape[0], dtype=torch.bool, device=self.nodes.device)
        mask[boundary_edges.reshape(-1)] = True
        return mask

    def edge_lengths(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Return the Euclidean length of every edge in ``edge_index``."""

        delta = (
            self.nodes[:, :2][edge_index[0]] - self.nodes[:, :2][edge_index[1]]
        )
        return torch.linalg.vector_norm(delta, dim=1)


@dataclass(frozen=True)
class DamagePrediction:
    """Prediction returned by a learned damage model."""

    damage: torch.Tensor
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    representation: str = "damage"


@runtime_checkable
class DamagePredictor(Protocol):
    """Protocol implemented by a learned damage predictor."""

    name: str

    def predict(
        self,
        context: DamageStepContext,
    ) -> DamagePrediction | torch.Tensor:
        """Return a nodal damage field or damage increment."""


class DamagePredictionRejected(RuntimeError):
    """Raised when replacement is rejected and fallback is disabled."""


@dataclass(frozen=True)
class DamageDecision:
    """Audited decision for one learned damage prediction."""

    route: str
    candidate: torch.Tensor | None
    accepted_replacement: bool
    reason: str
    residual_relative: float | None = None
    residual_rms: float | None = None
    lower_bound_violation: float | None = None
    upper_bound_violation: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def _predictor_name(predictor: Any) -> str:
    return str(
        getattr(
            predictor,
            "name",
            getattr(predictor, "__name__", predictor.__class__.__name__),
        )
    )


def load_damage_predictor(
    specification: str,
    *,
    checkpoint: str | None,
    device: torch.device,
    options: Mapping[str, Any] | None = None,
) -> DamagePredictor | Callable[[DamageStepContext], Any]:
    """Load a predictor from a public ``module:factory`` specification.

    The factory is called as::

        factory(checkpoint=..., device=..., options={...})

    It must return an object with ``predict(context)`` or a callable accepting
    one :class:`DamageStepContext`. Checkpoint interpretation belongs to the
    user-provided factory so PhAST does not impose one neural architecture.
    """

    if ":" not in specification:
        raise ValueError(
            "damage_predictor must use 'module:factory' notation; "
            f"received {specification!r}"
        )
    module_name, factory_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if factory is None or not callable(factory):
        raise ValueError(
            f"Damage predictor factory {factory_name!r} was not found as a "
            f"callable in module {module_name!r}."
        )
    predictor = factory(
        checkpoint=checkpoint,
        device=device,
        options=dict(options or {}),
    )
    if isinstance(predictor, torch.nn.Module):
        predictor.eval()
    if not hasattr(predictor, "predict") and not callable(predictor):
        raise TypeError(
            "A damage predictor factory must return an object with "
            "predict(context) or a callable accepting DamageStepContext."
        )
    return predictor


class DamageUpdateController:
    """Audit learned predictions and select the damage-update route."""

    def __init__(
        self,
        predictor: DamagePredictor | Callable[[DamageStepContext], Any],
        *,
        mode: str = "learned_proposal",
        residual_rtol: float = 1.0e-3,
        residual_atol: float = 1.0e-8,
        bound_tolerance: float = 1.0e-8,
        fallback: bool = True,
    ) -> None:
        if mode not in LEARNED_DAMAGE_MODES - {"classical"}:
            raise ValueError(
                "Learned damage mode must be 'learned_proposal' or "
                f"'learned_replacement'; received {mode!r}."
            )
        if residual_rtol < 0 or residual_atol < 0 or bound_tolerance < 0:
            raise ValueError("Damage audit tolerances must be non-negative.")
        self.predictor = predictor
        self.mode = mode
        self.residual_rtol = float(residual_rtol)
        self.residual_atol = float(residual_atol)
        self.bound_tolerance = float(bound_tolerance)
        self.fallback = bool(fallback)
        self.calls = 0
        self.proposals = 0
        self.accepted_replacements = 0
        self.fallbacks = 0
        self.failures = 0
        self.last_decision: DamageDecision | None = None

    @property
    def predictor_name(self) -> str:
        return _predictor_name(self.predictor)

    def _predict(self, context: DamageStepContext) -> DamagePrediction:
        with torch.no_grad():
            if hasattr(self.predictor, "predict"):
                output = self.predictor.predict(context)
            else:
                output = self.predictor(context)
        if isinstance(output, DamagePrediction):
            return output
        if isinstance(output, torch.Tensor):
            return DamagePrediction(output)
        raise TypeError(
            "Damage predictor output must be a torch.Tensor or "
            "DamagePrediction."
        )

    @staticmethod
    def _projected_residual(
        residual: torch.Tensor,
        damage: torch.Tensor,
        damage_previous: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        lower_active = damage <= damage_previous + tolerance
        upper_active = damage >= 1.0 - tolerance
        projected = torch.where(
            lower_active,
            torch.minimum(residual, torch.zeros_like(residual)),
            residual,
        )
        projected = torch.where(
            upper_active,
            torch.maximum(projected, torch.zeros_like(projected)),
            projected,
        )
        return projected

    def _reject(
        self,
        reason: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        **metrics: Any,
    ) -> DamageDecision:
        self.failures += 1
        if self.fallback:
            self.fallbacks += 1
            decision = DamageDecision(
                route="classical_fallback",
                candidate=None,
                accepted_replacement=False,
                reason=reason,
                diagnostics=dict(diagnostics or {}),
                **metrics,
            )
            self.last_decision = decision
            return decision
        raise DamagePredictionRejected(reason)

    def decide(
        self,
        context: DamageStepContext,
        *,
        damage_solver: Any,
        phase_field_mask: torch.Tensor | None = None,
        phase_field_values: torch.Tensor | None = None,
    ) -> DamageDecision:
        """Return an audited proposal, replacement, or fallback decision."""

        self.calls += 1
        try:
            prediction = self._predict(context)
        except Exception as exc:
            return self._reject(f"predictor error: {exc}")

        diagnostics = dict(prediction.diagnostics)
        diagnostics.setdefault("representation", prediction.representation)
        raw_prediction = prediction.damage.detach().to(
            device=context.device, dtype=context.dtype)
        if raw_prediction.shape != context.damage_previous.shape:
            return self._reject(
                "prediction shape does not match nodal damage shape",
                diagnostics=diagnostics,
            )
        if prediction.representation == "damage":
            candidate = raw_prediction
        elif prediction.representation == "damage_increment":
            candidate = context.damage_previous + raw_prediction
        else:
            return self._reject(
                "prediction representation must be 'damage' or "
                "'damage_increment'",
                diagnostics=diagnostics,
            )
        if not bool(torch.isfinite(candidate).all()):
            return self._reject(
                "prediction contains NaN or infinite values",
                diagnostics=diagnostics,
            )

        damage_previous = context.damage_previous
        lower_violation = float(
            torch.clamp(damage_previous - candidate, min=0).max().item())
        upper_violation = float(
            torch.clamp(candidate - 1.0, min=0).max().item())

        if self.mode == "learned_proposal":
            candidate = torch.maximum(
                torch.clamp(candidate, min=0.0, max=1.0),
                damage_previous,
            )
            if phase_field_mask is not None and phase_field_values is not None:
                candidate = torch.where(
                    phase_field_mask.to(device=context.device),
                    phase_field_values.to(
                        device=context.device, dtype=context.dtype),
                    candidate,
                )
            self.proposals += 1
            decision = DamageDecision(
                route="learned_proposal_exact_correction",
                candidate=candidate,
                accepted_replacement=False,
                reason="admissible prediction supplied as exact-solver initial guess",
                lower_bound_violation=lower_violation,
                upper_bound_violation=upper_violation,
                diagnostics=diagnostics,
            )
            self.last_decision = decision
            return decision

        if (
            lower_violation > self.bound_tolerance
            or upper_violation > self.bound_tolerance
        ):
            return self._reject(
                "prediction violates damage bounds or irreversibility",
                diagnostics=diagnostics,
                lower_bound_violation=lower_violation,
                upper_bound_violation=upper_violation,
            )

        if phase_field_mask is not None and phase_field_values is not None:
            mask = phase_field_mask.to(device=context.device)
            values = phase_field_values.to(
                device=context.device, dtype=context.dtype)
            if bool(mask.any()):
                bc_error = float(
                    torch.abs(candidate[mask] - values[mask]).max().item())
                if bc_error > self.bound_tolerance:
                    return self._reject(
                        "prediction violates a phase-field Dirichlet condition",
                        diagnostics=diagnostics,
                        lower_bound_violation=lower_violation,
                        upper_bound_violation=upper_violation,
                    )

        try:
            residual = damage_solver.compute_residual(
                context.history_element, candidate)
            reference = damage_solver.compute_residual(
                context.history_element, damage_previous)
        except Exception as exc:
            return self._reject(
                f"damage residual audit failed: {exc}",
                diagnostics=diagnostics,
                lower_bound_violation=lower_violation,
                upper_bound_violation=upper_violation,
            )

        projected = self._projected_residual(
            residual,
            candidate,
            damage_previous,
            self.bound_tolerance,
        )
        if phase_field_mask is not None:
            projected = torch.where(
                phase_field_mask.to(device=projected.device),
                torch.zeros_like(projected),
                projected,
            )
        projected_norm = float(torch.linalg.vector_norm(projected).item())
        reference_norm = float(torch.linalg.vector_norm(reference).item())
        residual_relative = projected_norm / max(
            reference_norm,
            torch.finfo(projected.dtype).eps,
        )
        residual_rms = projected_norm / math.sqrt(max(projected.numel(), 1))
        residual_ok = (
            residual_relative <= self.residual_rtol
            or residual_rms <= self.residual_atol
        )
        if not residual_ok:
            return self._reject(
                "prediction failed the projected damage-residual audit",
                diagnostics=diagnostics,
                residual_relative=residual_relative,
                residual_rms=residual_rms,
                lower_bound_violation=lower_violation,
                upper_bound_violation=upper_violation,
            )

        self.accepted_replacements += 1
        decision = DamageDecision(
            route="learned_replacement",
            candidate=candidate.clone(),
            accepted_replacement=True,
            reason="prediction passed admissibility and residual audits",
            residual_relative=residual_relative,
            residual_rms=residual_rms,
            lower_bound_violation=lower_violation,
            upper_bound_violation=upper_violation,
            diagnostics=diagnostics,
        )
        self.last_decision = decision
        return decision

    def summary(self) -> dict[str, Any]:
        """Return machine-readable controller statistics."""

        return {
            "mode": self.mode,
            "predictor": self.predictor_name,
            "calls": self.calls,
            "proposals": self.proposals,
            "accepted_replacements": self.accepted_replacements,
            "fallbacks": self.fallbacks,
            "failures": self.failures,
            "residual_rtol": self.residual_rtol,
            "residual_atol": self.residual_atol,
            "bound_tolerance": self.bound_tolerance,
            "fallback_enabled": self.fallback,
            "last_route": (
                None if self.last_decision is None
                else self.last_decision.route
            ),
            "last_reason": (
                None if self.last_decision is None
                else self.last_decision.reason
            ),
        }
