"""Staggered alternate-minimisation driver for the periodic hybrid AT2 cells.

One load increment prescribes the controlled macro-strain component and
alternates, for ``k = 1, 2, ...``:

1. ``g = g(d_k)``;
2. mechanics: :func:`~phast.solid3d.linalg.solve_constrained`
   on ``op.mech_action(., g)``, warm-started from the previous iterate (the
   first iterate starts from the previous step scaled to the new target,
   which is the exact mechanics solution at the previous damage); with
   interface-driver elements of the variational band stress the solve repeats over the
   normal-opening pattern until it settles (:meth:`StaggeredSolver._mechanics`);
   with the hybrid band stress (the default) it is one linear solve;
3. ``Y = op.driving_force(x)`` and ``H = max(H_prev_step, Y)``
   (``history='max'``) or ``H = Y`` (``history='current'``), always against
   the previous accepted step, never against the previous stagger iterate;
4. damage: box-KKT solve of ``op.damage_system(H)`` with
   ``d_prev_step <= d <= 1`` (``irreversibility='box_kkt'``), or an
   unconstrained solve whose bound violations are counted
   (``'history_only'``);
5. ``change = max(du_rel, dE_rel, dd_inf)``; converged when
   ``change <= stagger_tol`` and ``k >= min_stagger``.

Steps 1-4 define the damage fixed point ``d = G(d)``.  With
``anderson_depth > 0`` the input of the next
iteration may be a projected Anderson mix of the last iterates
(:class:`ProjectedAnderson`) instead of ``G(d_k)``.  The mix is judged by its
own iteration: when that iteration ends above the lowest total energy
``E_el + Gamma`` of the attempt so far, or with a larger fixed-point residual
``||G(d) - d||_2`` than the iteration that produced the replaced plain iterate
(or fails), the iteration is discarded, the plain iterate is used and the
history restarts (:meth:`StaggeredSolver._attempt`).  The history lives for
one attempt, so it is reset at every increment and cutback and a resumed run
is exact.  The accepted damage is always a plain output of the damage solve.

With the crack closure of (``crack_closure = 'axes'``) the mechanics
solve is an active-set iteration over every sign pattern (the closure pattern
and the opening pattern where present), each pattern solution
reached by the energy line search; the accepted record adds the closure solves
and the kinematic guard of the accepted state.  Anderson mixing is not
available with it.

With the PF-CZM damage law of (``damage_law = 'pfczm'``) step 4
minimises the nonconvex damage energy ``sum H g(d) + Gamma_pfczm(d)`` over
``d_prev_step <= d <= 1`` by a projected Newton iteration
(:meth:`StaggeredSolver._damage_pfczm`): each iteration solves the SPD box-QP
of :meth:`~.operator.PeriodicVoxelPhaseFieldOperator.pfczm_newton_system` with
the box-KKT solver (nodes on a bound with an outward gradient held there),
then an Armijo energy line search along the segment to the QP solution (it
stays in the box), until the projected gradient of the nonlinear problem is at
most ``damage_kkt_tol``.  The records add the Newton iterations of the increment
(``newton_iters_pf``).

The accepted damage is followed by a final mechanics re-solve, and the macro
stress, energies and Hill-Mandel residual are evaluated at that consistent
``(u, d)``.  External work is the trapezoid rule on the controlled component,
``W += V (Sigma_c,n + Sigma_c,n-1)(E_c,n - E_c,n-1)/2``.

A failed attempt (PCG breakdown or non-convergence, a box-KKT failure,
stagger non-convergence or non-finite values) raises
:class:`StaggerConvergenceError`; :meth:`StaggeredSolver.advance_increment`
splits the increment into two halves recursively (``on_stagger_failure =
'cutback'``) up to ``max_cutbacks`` levels and then raises.  No unconverged
state is ever returned.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
import dataclasses
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from phast.solvers.box_kkt_h8 import (
    H8ScalarBoxKKTConvergenceError,
    solve_h8_scalar_box_kkt,
    solve_h8_scalar_dirichlet_pcg,
)
from phast.solid3d.linalg import PCGBreakdownError, PCGInfo, solve_constrained
from phast.solid3d.model import ClosurePattern, FieldSnapshot, IncrementRecord, LoadCase, SolverConfig, resolve_device
from phast.solid3d.operator import PeriodicVoxelPhaseFieldOperator


CHECKPOINT_FORMAT = "phast.solid3d.checkpoint"
CHECKPOINT_VERSION = 1

_TINY = float(np.finfo(np.float64).tiny)
_MIN_LINEAR_TOLERANCE = 4.0 * float(np.finfo(np.float64).eps)

INTERFACE_MAX_SOLVES = 50
"""Most linear mechanics solves of one active-set mechanics solve with interface-driver elements."""

INTERFACE_PATTERN_TOLERANCE = 1.0e-8
"""A normal-opening change counts only where ``|t_11|`` exceeds this fraction of its maximum."""

CLOSURE_PATTERN_TOLERANCE = 1.0e-8
"""A closure-pattern change of bit ``k`` counts only where ``(1 - g) k |v . eps|`` exceeds this
fraction of the largest ``k |v . eps|`` of its group."""

LINE_SEARCH_ARMIJO = 1.0e-4
"""Sufficient-decrease constant of the energy line search of the pattern mechanics solve."""

LINE_SEARCH_ENERGY_RTOL = 1.0e-12
"""Round-off allowance of that line search, relative to the energy at the current iterate."""

LINE_SEARCH_MAX_HALVINGS = 30
"""Most step halvings of that line search before the attempt is cut back."""

PFCZM_ARMIJO = 1.0e-4
"""Sufficient-decrease constant of the energy line search of the PF-CZM Newton damage solve."""

PFCZM_ENERGY_RTOL = 1.0e-12
"""Round-off allowance of that line search, relative to the damage energy at the current iterate."""

PFCZM_MAX_HALVINGS = 30
"""Most step halvings of that line search before the attempt is cut back."""

PFCZM_MAX_DOUBLINGS = 6
"""Most step doublings of the projected extrapolation that follows a full Newton step."""

PFCZM_INNER_KKT_FRACTION = 0.1
"""The box-QP subproblem of a Newton iteration is solved to this fraction of ``damage_kkt_tol``."""

ANDERSON_REGULARISATION = 1.0e-10
"""Tikhonov weight of the Anderson least-squares problem (on unit-norm residual differences)."""

ANDERSON_ENERGY_RTOL = 1.0e-12
"""Round-off allowance of the energy test of a mixed iterate, relative to the reference energy."""

_H8_BREAKDOWN_MESSAGES = (
    "non-positive initial residual product",
    "not positive definite on the free space",
    "non-positive residual product",
)
"""Messages of the numerical PCG breakdowns raised as ``RuntimeError`` by the reused H8 solver."""


def _is_scalar_solver_breakdown(exc: BaseException) -> bool:
    """Whether ``exc`` is one of the H8 PCG breakdowns (a numerical, cut-back-able failure).

    Any other ``RuntimeError`` (a shape or internal error, an asynchronous
    CUDA fault surfacing at a later synchronisation, out of memory) is a
    software or device fault and must reach the failure path.
    """

    if type(exc) is not RuntimeError:
        return False
    message = str(exc)
    return any(text in message for text in _H8_BREAKDOWN_MESSAGES)


# ---------------------------------------------------------------------------
# state and errors
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class SolverState:
    """Accepted solution after ``step`` solves (mutable; tensors on the solver device).

    ``x`` is the mechanics vector ``[u~, E]``, ``d`` the reduced nodal damage,
    ``H`` the ``(E, 8)`` history field used by the damage solve that produced
    ``d``, ``macro_strain`` and ``macro_stress`` the ``(6,)`` macro response,
    ``external_work`` the accumulated work in mJ and ``d_prev_step`` the
    damage of the previous accepted step (the lower bound that produced
    ``d``; zero at the start).  ``increment`` is the number of load-case
    increments completed; it lags ``step`` after cutbacks and tells
    :meth:`StaggeredSolver.run` where to resume.
    """

    step: int
    x: torch.Tensor
    d: torch.Tensor
    H: torch.Tensor
    macro_strain: torch.Tensor
    macro_stress: torch.Tensor
    external_work: float
    d_prev_step: torch.Tensor
    increment: int = 0

    def clone(self) -> "SolverState":
        """Deep copy (tensors cloned)."""

        return SolverState(
            step=int(self.step),
            x=self.x.clone(),
            d=self.d.clone(),
            H=self.H.clone(),
            macro_strain=self.macro_strain.clone(),
            macro_stress=self.macro_stress.clone(),
            external_work=float(self.external_work),
            d_prev_step=self.d_prev_step.clone(),
            increment=int(self.increment),
        )

    def to(self, device: torch.device | str) -> "SolverState":
        """Copy of the state on ``device``."""

        return SolverState(
            step=int(self.step),
            x=self.x.to(device),
            d=self.d.to(device),
            H=self.H.to(device),
            macro_strain=self.macro_strain.to(device),
            macro_stress=self.macro_stress.to(device),
            external_work=float(self.external_work),
            d_prev_step=self.d_prev_step.to(device),
            increment=int(self.increment),
        )


class StaggerConvergenceError(RuntimeError):
    """A load (sub-)increment could not be solved.

    Attributes
    ----------
    state:
        The last accepted :class:`SolverState` (the state to stop or resume
        from).
    target:
        Controlled macro strain of the failed attempt.
    increment:
        Load-case increment (1-based) being solved.
    cutbacks:
        Halving depth of the failed attempt (0 = the full increment).
    reason:
        Why the last attempt failed.
    attempts:
        Attempts made for this load-case increment.
    records:
        Records accepted before the failure (the increment's sub-steps, or
        every record of the run when raised from :meth:`StaggeredSolver.run`).
    """

    def __init__(
        self,
        message: str,
        *,
        state: SolverState,
        target: float,
        increment: int,
        cutbacks: int,
        reason: str,
        attempts: int = 1,
        records: tuple[IncrementRecord, ...] = (),
    ) -> None:
        super().__init__(message)
        self.state = state
        self.target = float(target)
        self.increment = int(increment)
        self.cutbacks = int(cutbacks)
        self.reason = str(reason)
        self.attempts = int(attempts)
        self.records = tuple(records)


class _AttemptFailure(Exception):
    """Internal: one attempt failed for a numerical reason (cut back or raise).

    ``pcg_mech``, ``pcg_pf`` and ``active_set`` carry the Krylov and
    active-set iterations that the failing solve had already spent where they
    are known (a PCG solve that ran out of iterations, the solves of a
    mechanics pattern iteration, a box-KKT solve that did not converge); a
    breakdown inside one Krylov solve reports none.  Only an iteration from a
    mixed iterate adds them to the records, because any other
    failure ends the attempt.
    """

    def __init__(self, message: str, *, pcg_mech: int = 0, pcg_pf: int = 0, active_set: int = 0) -> None:
        super().__init__(message)
        self.pcg_mech = int(pcg_mech)
        self.pcg_pf = int(pcg_pf)
        self.active_set = int(active_set)


# ---------------------------------------------------------------------------
# Anderson mixing of the damage fixed point
# ---------------------------------------------------------------------------


class ProjectedAnderson:
    """Type II Anderson mixing of a fixed point ``d = G(d)`` with a box projection.

    Keeps the last ``depth + 1`` pairs ``(d_i, G(d_i))`` and their residuals
    ``f_i = G(d_i) - d_i``.  :meth:`mix` solves the least-squares problem
    ``min_gamma ||f_k - dF gamma||_2`` over the residual differences
    ``dF = [f_{i+1} - f_i]`` and returns
    ``clamp(G(d_k) - dG gamma, lower, upper)`` with ``dG = [G(d_{i+1}) - G(d_i)]``
    (Walker and Ni 2011, doi:10.1137/10078356X).  This follows
    :class:`phast.solvers.staggered_solver._AndersonAccelerator`, with three
    differences: the columns of ``dF`` are scaled to unit norm and the normal
    equations carry the relative weight :data:`ANDERSON_REGULARISATION`;
    the projection is onto the box ``[lower, upper]`` of the damage solve
    (``d_{n-1} <= d <= 1``); and the safeguard is left to the caller, which
    judges a mixed iterate by the energy and residual of its own staggered
    iteration (:meth:`StaggeredSolver._attempt`).  The history lives for one
    attempt of one increment only.
    """

    def __init__(self, depth: int, *, regularisation: float = ANDERSON_REGULARISATION) -> None:
        if isinstance(depth, bool) or int(depth) != depth or int(depth) < 1:
            raise ValueError(f"depth must be an integer >= 1, got {depth!r}")
        if not (math.isfinite(regularisation) and regularisation >= 0.0):
            raise ValueError(f"regularisation must be finite and >= 0, got {regularisation!r}")
        self.depth = int(depth)
        self.regularisation = float(regularisation)
        self._inputs: deque[torch.Tensor] = deque(maxlen=self.depth + 1)
        self._outputs: deque[torch.Tensor] = deque(maxlen=self.depth + 1)

    def __len__(self) -> int:
        return len(self._inputs)

    def reset(self) -> None:
        """Forget every stored pair."""

        self._inputs.clear()
        self._outputs.clear()

    def push(self, d_in: torch.Tensor, d_out: torch.Tensor) -> None:
        """Store the pair ``(d_in, G(d_in))`` (copies; the oldest pair drops out at ``depth + 1``)."""

        if d_in.shape != d_out.shape or d_in.dim() != 1:
            raise ValueError("d_in and d_out must be vectors of the same length")
        self._inputs.append(d_in.detach().clone())
        self._outputs.append(d_out.detach().clone())

    def mix(self, lower: torch.Tensor | float, upper: torch.Tensor | float = 1.0) -> torch.Tensor | None:
        """Projected mixed iterate, or ``None`` when no mix is defined.

        ``None`` is returned with fewer than two stored pairs, when every
        residual difference vanishes, or when the least-squares solve fails or
        gives a non-finite iterate.
        """

        if len(self._inputs) < 2:
            return None
        inputs, outputs = list(self._inputs), list(self._outputs)
        residuals = [out - inp for inp, out in zip(inputs, outputs)]
        dF = torch.stack([residuals[i + 1] - residuals[i] for i in range(len(residuals) - 1)], dim=1)
        dG = torch.stack([outputs[i + 1] - outputs[i] for i in range(len(outputs) - 1)], dim=1)
        scale = torch.linalg.vector_norm(dF, dim=0)
        keep = scale > 0.0
        if not bool(keep.any()):
            return None
        dF, dG, scale = dF[:, keep], dG[:, keep], scale[keep]
        A = dF / scale
        normal = A.T @ A
        normal.diagonal().add_(self.regularisation)
        try:
            beta = torch.linalg.solve(normal, A.T @ residuals[-1])
        except RuntimeError:
            return None
        gamma = beta / scale
        mixed = outputs[-1] - dG @ gamma
        if not bool(torch.isfinite(mixed).all()):
            return None
        if isinstance(lower, torch.Tensor):
            mixed = torch.maximum(mixed, lower)
        else:
            mixed = torch.clamp(mixed, min=float(lower))
        if isinstance(upper, torch.Tensor):
            mixed = torch.minimum(mixed, upper)
        else:
            mixed = torch.clamp(mixed, max=float(upper))
        return mixed


@dataclass(eq=False)
class _PlainFallback:
    """The plain iterate a pending mixed iterate replaced, and the residual of the iteration that made it."""

    d: torch.Tensor
    x: torch.Tensor
    residual: float


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


class StaggeredSolver:
    """Staggered hybrid AT2 solver of one periodic cell under mixed macro control.

    Parameters
    ----------
    op:
        The :class:`~phast.solid3d.operator.PeriodicVoxelPhaseFieldOperator`.
    load:
        Load case; defaults to ``op.load``.  Its controlled, free and
        ``fixed_zero`` components must match ``op.load``.
    cfg:
        Solver settings; defaults to ``op.cfg``.  ``kappa`` and the energy
        splits must match ``op.cfg`` (they are built into the operator).

    Attributes
    ----------
    op, load, cfg, last_state (the last accepted state), n_attempts,
    n_failed_attempts and ``last_bound_violations`` (``history_only``
    irreversibility: bound-violation counts of the last damage solve).
    """

    def __init__(
        self,
        op: PeriodicVoxelPhaseFieldOperator,
        load: LoadCase | None = None,
        cfg: SolverConfig | None = None,
    ) -> None:
        if not isinstance(op, PeriodicVoxelPhaseFieldOperator):
            raise TypeError("op must be a PeriodicVoxelPhaseFieldOperator")
        load = op.load if load is None else load
        cfg = op.cfg if cfg is None else cfg
        if not isinstance(load, LoadCase):
            raise TypeError("load must be a LoadCase")
        if not isinstance(cfg, SolverConfig):
            raise TypeError("cfg must be a SolverConfig")
        if (load.controlled, set(load.free), set(load.fixed_zero)) != (
            op.load.controlled,
            set(op.load.free),
            set(op.load.fixed_zero),
        ):
            raise ValueError("load must control, free and fix the same components as op.load")
        for key in (
            "kappa",
            "energy_split_isotropic",
            "energy_split_band",
            "band_stress",
            "dtype",
            "crack_closure",
            "damage_law",
            "pfczm_softening",
            "pfczm_p",
        ):
            if getattr(cfg, key) != getattr(op.cfg, key):
                raise ValueError(f"cfg.{key} must match the operator's ({getattr(op.cfg, key)!r})")
        if op.has_closure and cfg.anderson_depth > 0:
            raise ValueError("Anderson mixing (anderson_depth > 0) is not available with crack_closure 'axes'")
        self.op = op
        self.load = load
        self.cfg = cfg
        self.device = op.device
        self.dtype = op.dtype
        self.last_state: SolverState | None = None
        self.n_attempts = 0
        self.n_failed_attempts = 0
        self.last_bound_violations: dict[str, float] = {}
        self._mask = op.mech_fixed_mask()
        self._n_u = 3 * op.n_red
        self._controlled = int(load.controlled)
        self._max_abs_target = max(abs(t) for t in load.targets)
        self._u_floor = min(op.grid.h) * math.sqrt(self._n_u)
        lx, ly, lz = op.grid.lengths
        self._lengths = (lx, ly, lz)
        # applied displacement and reaction force along the controlled normal
        # component c: E_c L_c and Sigma_c times the area of the face normal
        # to c (the two other lengths in axis order, multiplied left to right
        # as before); a shear control keeps the former xx convention
        axis = self._controlled if self._controlled < 3 else 0
        others = tuple(self._lengths[a] for a in range(3) if a != axis)
        self._load_axis = axis
        self._load_length = self._lengths[axis]
        self._face_lengths = (others[0], others[1])
        # optional per-iteration diagnostics (JSON lines); see _log_iteration
        self._stagger_log: str | None = os.environ.get('PHAST_SOLID3D_STAGGER_LOG')
        self.last_change_parts: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_interface_solves = 0
        self.last_line_search_steps = 0
        # optional observer of every staggered iterate of selected steps
        # (``iterate_observer(row)`` for the attempts that produce a step in
        # ``observe_steps``); None leaves the iteration untouched
        self.iterate_observer: Callable[[dict[str, Any]], Any] | None = None
        self.observe_steps: frozenset[int] = frozenset()
        # Newton iterations of the PF-CZM damage solves of the current attempt, and the
        # iteration count and damage energies of the last solve
        self._newton_count = 0
        self.last_newton_iters = 0
        self.last_newton_energies: list[float] = []

    # ------------------------------------------------------------------
    # states
    # ------------------------------------------------------------------

    def initial_state(self) -> SolverState:
        """Undamaged, unloaded state at step 0."""

        op = self.op
        zeros = lambda *shape: torch.zeros(shape, dtype=self.dtype, device=self.device)  # noqa: E731
        return SolverState(
            step=0,
            x=zeros(op.n_mech),
            d=zeros(op.n_red),
            H=zeros(op.n_elements, 8),
            macro_strain=zeros(6),
            macro_stress=zeros(6),
            external_work=0.0,
            d_prev_step=zeros(op.n_red),
            increment=0,
        )

    def initial_record(self) -> IncrementRecord:
        """The step-0 zero row (every present phase listed with zero fracture energy)."""

        return IncrementRecord(
            step=0,
            fracture_by_phase={name: 0.0 for name in self.op.model.phases_present()},
        )

    def snapshot(self, state: SolverState) -> FieldSnapshot:
        """CPU fields of ``state`` (time value: the controlled macro strain)."""

        return self.op.field_snapshot(
            state.step, float(state.macro_strain[self._controlled]), state.x, state.d, state.H
        )

    def _check_state(self, state: SolverState) -> None:
        if not isinstance(state, SolverState):
            raise TypeError("state must be a SolverState")
        op = self.op
        for name, shape in (
            ("x", (op.n_mech,)),
            ("d", (op.n_red,)),
            ("H", (op.n_elements, 8)),
            ("macro_strain", (6,)),
            ("macro_stress", (6,)),
            ("d_prev_step", (op.n_red,)),
        ):
            value = getattr(state, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"state.{name} must be a tensor of shape {shape}")
            if value.dtype != self.dtype or value.device != self.device:
                raise ValueError(f"state.{name} must be {self.dtype} on {self.device}")

    # ------------------------------------------------------------------
    # one attempt
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _linear_mechanics(
        self,
        g: torch.Tensor,
        values: torch.Tensor,
        x0: torch.Tensor,
        opening: torch.Tensor | None,
        closure: ClosurePattern | None = None,
    ) -> tuple[torch.Tensor, PCGInfo]:
        op, cfg = self.op, self.cfg
        if closure is not None:
            action = lambda v: op.mech_action(v, g, opening=opening, closure=closure)
            diag = op.mech_diagonal(g, opening=opening, closure=closure)
        elif opening is None:
            action = lambda v: op.mech_action(v, g)
            diag = op.mech_diagonal(g)
        else:
            action = lambda v: op.mech_action(v, g, opening=opening)
            diag = op.mech_diagonal(g, opening=opening)
        try:
            x, info = solve_constrained(
                action,
                diag=diag,
                fixed_mask=self._mask,
                fixed_values=values,
                x0=x0,
                rtol=cfg.pcg_rtol_mech,
                maxiter=cfg.pcg_maxit_mech,
                check_every=cfg.check_every,
            )
        except PCGBreakdownError as exc:
            raise _AttemptFailure(f"mechanics PCG breakdown: {exc}") from exc
        if not info.converged:
            raise _AttemptFailure(
                f"mechanics PCG did not converge in {cfg.pcg_maxit_mech} iterations "
                f"(relative residual {info.relative_residual:.3e} > {cfg.pcg_rtol_mech:.3e})",
                pcg_mech=info.iterations,
            )
        return x, info

    def _energy_line_search(
        self, x: torch.Tensor, y: torch.Tensor, g: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Damped step from ``x`` towards the pattern solution ``y`` (every mechanics solve with the crack closure).

        The degraded energy ``Pi(z) = elastic_energy(z)`` (each point at its own
        pattern) is the mechanics potential: the fixed dofs are prescribed and
        the free macro strains carry zero stress.  It is convex and
        continuously differentiable, and ``p = y - x`` is a descent direction
        because ``y`` minimises the quadratic of the pattern of ``x``, whose
        gradient at ``x`` is the internal force.  The step length ``alpha`` is
        halved from 1 until ``Pi(x + alpha p) <= Pi(x) + c alpha (grad Pi . p)``
        (:data:`LINE_SEARCH_ARMIJO`, with a round-off allowance
        :data:`LINE_SEARCH_ENERGY_RTOL` of ``|Pi(x)|``); ``alpha = 1`` returns
        ``y`` itself.  This makes the pattern iteration monotone in the energy,
        so it cannot cycle.
        """

        op = self.op
        p = y - x
        energy0 = float(op.elastic_energy(x, g))
        force = op.mech_action(x, g)
        free = ~self._mask
        slope = min(float(torch.dot(force[free], p[free])), 0.0)
        allowance = LINE_SEARCH_ENERGY_RTOL * abs(energy0)
        alpha = 1.0
        for _ in range(LINE_SEARCH_MAX_HALVINGS + 1):
            trial = y if alpha == 1.0 else x + alpha * p
            energy = float(op.elastic_energy(trial, g))
            if not math.isfinite(energy):
                raise _AttemptFailure("mechanics energy is not finite in the line search")
            if energy <= energy0 + LINE_SEARCH_ARMIJO * alpha * slope + allowance:
                return trial, alpha
            alpha *= 0.5
        raise _AttemptFailure(
            f"the energy line search of the {'closure' if op.has_closure else 'interface'} mechanics solve "
            f"found no decrease in {LINE_SEARCH_MAX_HALVINGS} halvings"
        )

    def _mechanics(
        self,
        g: torch.Tensor,
        values: torch.Tensor,
        x0: torch.Tensor,
    ) -> tuple[torch.Tensor, PCGInfo]:
        """Mechanics solve at fixed damage.

        Without interface-driver elements and without the crack closure this is
        one linear solve.  With variational interface elements the band stress is piecewise linear, and the solve is an
        active-set (semismooth Newton)
        iteration on the normal-opening pattern: start from the pattern of
        ``x0``, solve the linear system of that pattern, and repeat with the
        pattern of the solution until no quadrature point changes side by
        more than :data:`INTERFACE_PATTERN_TOLERANCE` times the largest
        normal traction.  Each pattern solution is
        reached by the energy line search of :meth:`_energy_line_search`
        (the plain iteration can cycle between patterns); the iteration ends
        when the pattern settles after a full step.  The degraded energy is
        convex and continuously differentiable, so a settled pattern is its
        minimiser.  The returned ``PCGInfo`` sums the iterations of all
        solves; ``last_interface_solves`` records their number and
        ``last_line_search_steps`` the damped steps (closure only).

        With the crack closure the iteration also covers the
        closure pattern (:meth:`~.operator.PeriodicVoxelPhaseFieldOperator.closure_pattern_change`,
        :data:`CLOSURE_PATTERN_TOLERANCE`), starts from the patterns of
        ``x0`` and always uses the energy line search; the degraded energy
        stays convex and continuously differentiable, so a settled pattern
        after a full step is its minimiser.
        """

        op = self.op
        if not op.has_interface_stress and not op.has_closure:
            return self._linear_mechanics(g, values, x0, None)
        if op.has_closure:
            return self._closure_mechanics(g, values, x0)
        opening = None
        if op.has_interface_stress:
            traction = op.interface_normal_traction(x0)
            opening = traction > 0.0
        iterations = 0
        damped = 0
        for solve in range(1, INTERFACE_MAX_SOLVES + 1):
            alpha = 1.0
            try:
                (x, info) = self._linear_mechanics(g, values, x0, opening)
                iterations += info.iterations
            except _AttemptFailure as exc:
                # the earlier solves of this pattern iteration count too
                exc.pcg_mech += iterations
                raise
            changed_any = False
            if op.has_interface_stress:
                traction = op.interface_normal_traction(x)
                new = traction > 0.0
                scale = traction.abs().max()
                changed = (new != opening) & (traction.abs() > INTERFACE_PATTERN_TOLERANCE * scale)
                changed_any = bool(changed.any())
                opening = new
            if not changed_any and alpha == 1.0:
                self.last_interface_solves = solve
                self.last_line_search_steps = damped
                return x, dataclasses.replace(info, iterations=iterations)
            x0 = x
        raise _AttemptFailure(
            f"the normal-opening pattern of the interface elements did not settle in {INTERFACE_MAX_SOLVES} mechanics solves",
            pcg_mech=iterations,
        )

    def _closure_mechanics(
        self,
        g: torch.Tensor,
        values: torch.Tensor,
        x0: torch.Tensor,
    ) -> tuple[torch.Tensor, PCGInfo]:
        """Line-searched active-set mechanics solve over every pattern (crack closure).

        Starts from the patterns of ``x0``; each linear solve uses the fixed
        closure and opening patterns, the energy line search
        (:meth:`_energy_line_search`, each point at its own pattern) gives
        the next iterate, and the iteration ends when no pattern changes
        materially after a full step, at most :data:`INTERFACE_MAX_SOLVES`
        solves.
        """

        op = self.op
        opening = op.interface_normal_traction(x0) > 0.0 if op.has_interface_stress else None
        closure = op.closure_pattern(x0)
        iterations = 0
        damped = 0
        for solve in range(1, INTERFACE_MAX_SOLVES + 1):
            try:
                y, info = self._linear_mechanics(g, values, x0, opening, closure)
                iterations += info.iterations
                x, alpha = self._energy_line_search(x0, y, g)
                damped += int(alpha < 1.0)
            except _AttemptFailure as exc:
                # the earlier solves of this pattern iteration count too
                exc.pcg_mech += iterations
                raise
            changed_any = False
            if opening is not None:
                traction = op.interface_normal_traction(x)
                new = traction > 0.0
                scale = traction.abs().max()
                changed = (new != opening) & (traction.abs() > INTERFACE_PATTERN_TOLERANCE * scale)
                changed_any = bool(changed.any())
                opening = new
            closure, changed_closure = op.closure_pattern_change(
                x, closure, g, tolerance=CLOSURE_PATTERN_TOLERANCE
            )
            changed_any = changed_any or changed_closure
            if not changed_any and alpha == 1.0:
                self.last_interface_solves = solve
                self.last_line_search_steps = damped
                return x, dataclasses.replace(info, iterations=iterations)
            x0 = x
        raise _AttemptFailure(
            f"the closure pattern (crack_closure 'axes'), with the opening pattern of the interface "
            f"elements where present, did not settle in "
            f"{INTERFACE_MAX_SOLVES} mechanics solves",
            pcg_mech=iterations,
        )

    def _damage(
        self,
        H: torch.Tensor,
        lower: torch.Tensor,
        initial: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, bool]:
        """Return ``(d, pcg iterations, active-set iterations, kkt_ok)``.
        """

        if self.op.is_pfczm:
            return self._damage_pfczm(H, lower, initial)
        cfg = self.cfg
        system = self.op.damage_system(H)
        # The box-KKT PCG stops at tol * max(||r0||, 1) while the KKT audit is
        # absolute, so the linear tolerance is tightened to keep the PCG
        # threshold at or below 0.1 damage_kkt_tol.
        r0 = float(torch.linalg.vector_norm(system.rhs - system.apply(initial)))
        if not math.isfinite(r0):
            raise _AttemptFailure("damage residual is not finite")
        tolerance = min(cfg.damage_linear_tol, 0.1 * cfg.damage_kkt_tol / max(r0, 1.0))
        tolerance = max(tolerance, _MIN_LINEAR_TOLERANCE)
        if cfg.irreversibility == "box_kkt":
            try:
                result = solve_h8_scalar_box_kkt(
                    system,
                    lower_bound=lower,
                    upper_bound=1.0,
                    initial=initial,
                    linear_tolerance=tolerance,
                    kkt_tolerance=cfg.damage_kkt_tol,
                    max_linear_iterations=cfg.damage_maxit,
                    max_active_set_iterations=cfg.damage_max_active_set,
                )
            except H8ScalarBoxKKTConvergenceError as exc:
                label = 'damage'
                raise _AttemptFailure(
                    f"{label} box-KKT solve failed: {exc}",
                    pcg_pf=exc.evidence.total_pcg_iterations,
                    active_set=exc.evidence.active_set_iterations,
                ) from exc
            except RuntimeError as exc:
                if not _is_scalar_solver_breakdown(exc):
                    raise
                raise _AttemptFailure(f"damage PCG breakdown: {exc}") from exc
            return (
                result.solution,
                int(result.total_pcg_iterations),
                int(result.active_set_iterations),
                bool(result.kkt.satisfies_kkt),
            )

        fixed = torch.zeros(self.op.n_red, dtype=torch.bool, device=self.device)
        try:
            solve = solve_h8_scalar_dirichlet_pcg(
                system,
                fixed=fixed,
                values=torch.zeros_like(initial),
                initial=initial,
                tolerance=tolerance,
                max_iterations=cfg.damage_maxit,
            )
        except RuntimeError as exc:
            if not _is_scalar_solver_breakdown(exc):
                raise
            raise _AttemptFailure(f"damage PCG breakdown: {exc}") from exc
        if not solve.converged:
            raise _AttemptFailure(
                f"damage PCG did not converge in {cfg.damage_maxit} iterations",
                pcg_pf=int(solve.krylov.iterations),
            )
        d = solve.solution
        stats = torch.stack(
            (
                (d < lower).sum().to(self.dtype),
                (d < 0.0).sum().to(self.dtype),
                (d > 1.0).sum().to(self.dtype),
                torch.clamp_min(lower - d, 0.0).max(),
                torch.clamp_min(d - 1.0, 0.0).max(),
            )
        ).cpu()
        self.last_bound_violations = {
            "below_previous": int(stats[0]),
            "below_zero": int(stats[1]),
            "above_one": int(stats[2]),
            "max_decrease": float(stats[3]),
            "max_excess_over_one": float(stats[4]),
        }
        kkt_ok = int(stats[0]) == 0 and int(stats[1]) == 0 and int(stats[2]) == 0
        return d, int(solve.krylov.iterations), 0, kkt_ok

    def _damage_pfczm(
        self,
        H: torch.Tensor,
        lower: torch.Tensor,
        initial: torch.Tensor,
        *,
        correction: bool = False,
        trace_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[torch.Tensor, int, int, bool]:
        """Projected Newton solve of the PF-CZM damage energy over ``lower <= d <= 1``.

        Returns ``(d, pcg iterations, active-set iterations, True)`` like
        :meth:`_damage`.  Each iteration ``k``: the scaled gradient ``r`` of
        :meth:`~.operator.PeriodicVoxelPhaseFieldOperator.pfczm_damage_energy`
        at ``d_k``; stop when the projected gradient (``r`` with the outward
        part removed at an active bound) is at most ``damage_kkt_tol`` in the
        maximum norm; otherwise solve the box-QP of
        :meth:`~.operator.PeriodicVoxelPhaseFieldOperator.pfczm_newton_system`
        with the box-KKT solver, holding the nodes that sit on a bound with an
        outward gradient above the tolerance (``d_k = lower`` and ``r > tol``,
        or ``d_k = 1`` and ``r < -tol``), and take the Armijo step along
        ``d_k -> d_QP`` with step halving; after a full step the step is
        doubled along the projected path ``P(d_k + t (d_QP - d_k))`` while the
        energy keeps decreasing (at most :data:`PFCZM_MAX_DOUBLINGS` times),
        unless the accepted full step already meets the nonlinear KKT tolerance,
        because the tangent floor overestimates the curvature of the concave
        softening part and the full step then stops short of the bound
        ``d = 1``. Every iterate is feasible; Armijo permits only the stated
        relative round-off allowance for energy increases
        (:attr:`last_newton_energies`). A subproblem failure, a failed line
        search or ``pfczm_max_newton`` iterations without convergence raise
        :class:`_AttemptFailure`.

        The established absolute-field QP remains the default. Explicit
        ``correction=True`` solves for ``p = d_QP - d_k`` with shifted bounds
        and rhs ``-gradient``, avoiding cancellation in ``K d_QP - b``.
        This coordinate choice is diagnostic only: the full-cell pilot stopped
        earlier, so it is not enabled by the runner. The final stopping test
        remains the nonlinear gradient at physical damage in either form.

        ``trace_callback`` is an optional diagnostic observer. It receives
        scalar Newton/inner-solve evidence and detached tensor views. Observers
        must treat these aliases as read-only and copy retained tensors.
        Extra diagnostic gradient evaluations do not change the acceptance decisions.
        """

        op, cfg = self.op, self.cfg
        label = 'damage'
        tol = cfg.damage_kkt_tol
        inner_tol = PFCZM_INNER_KKT_FRACTION * tol
        scale = op.damage_scale
        assert scale is not None

        def finish(status: str, iteration: int, residual: float | None, reason: str | None = None) -> None:
            if trace_callback is not None:
                trace_callback({"event": "finished", "status": status, "iteration": iteration,
                                "residual": residual, "reason": reason, "d": d.detach(),
                                "correction": correction})

        def projected_residual(field: torch.Tensor) -> float:
            grad = scale * op.pfczm_damage_gradient(field, H)
            projected_grad = torch.where(field <= lower, grad.clamp(max=0.0), grad)
            projected_grad = torch.where(field >= 1.0, projected_grad.clamp(min=0.0), projected_grad)
            return float(projected_grad.abs().max())

        def pcg_record(krylov) -> dict[str, Any]:
            return {name: getattr(krylov, name) for name in (
                "iterations", "converged", "initial_residual_norm", "residual_norm", "relative_residual"
            )}

        d = initial
        energy = float(op.pfczm_damage_energy(d, H))
        if not math.isfinite(energy):
            finish("failed", 0, None, "nonfinite energy")
            raise _AttemptFailure(f"{label} PF-CZM energy is not finite")
        energies = [energy]
        pcg = sets = 0
        ones = torch.ones_like(d)
        for iteration in range(cfg.pfczm_max_newton + 1):
            system, gradient = op.pfczm_newton_system(d, H, correction=correction)
            at_lower = d <= lower
            at_upper = d >= 1.0
            projected = torch.where(at_lower, gradient.clamp(max=0.0), gradient)
            projected = torch.where(at_upper, projected.clamp(min=0.0), projected)
            residual = float(projected.abs().max())
            if trace_callback is not None:
                worst = int(projected.abs().argmax())
                trace_callback({
                    "event": "iterate", "iteration": iteration, "residual": residual,
                    "energy": energy, "worst_index": worst,
                    "gradient_at_worst": float(gradient[worst]), "damage_at_worst": float(d[worst]),
                    "lower_at_worst": float(lower[worst]),
                    "at_lower": bool(at_lower[worst]), "at_upper": bool(at_upper[worst]),
                    "lower_violation": float(torch.clamp_min(lower - d, 0.0).max()),
                    "upper_violation": float(torch.clamp_min(d - 1.0, 0.0).max()),
                    "d": d.detach(), "correction": correction,
                })
            if not math.isfinite(residual):
                finish("failed", iteration, residual, "nonfinite gradient")
                raise _AttemptFailure(f"{label} PF-CZM gradient is not finite", pcg_pf=pcg, active_set=sets)
            if residual <= tol:
                self.last_newton_iters = iteration
                self.last_newton_energies = energies
                self._newton_count += iteration
                finish("converged", iteration, residual)
                return d, pcg, sets, True
            if iteration == cfg.pfczm_max_newton:
                break
            hold_lower = at_lower & (gradient > tol)
            hold_upper = at_upper & (gradient < -tol)
            upper_k = torch.where(hold_lower, lower, ones)
            lower_k = torch.where(hold_upper, ones, lower)
            r0 = float(torch.linalg.vector_norm(gradient))
            tolerance = min(cfg.damage_linear_tol, 0.1 * inner_tol / max(r0, 1.0))
            tolerance = max(tolerance, _MIN_LINEAR_TOLERANCE)
            try:
                result = solve_h8_scalar_box_kkt(
                    system,
                    lower_bound=lower_k - d if correction else lower_k,
                    upper_bound=upper_k - d if correction else upper_k,
                    initial=torch.zeros_like(d) if correction else d,
                    linear_tolerance=tolerance,
                    kkt_tolerance=inner_tol,
                    max_linear_iterations=cfg.damage_maxit,
                    max_active_set_iterations=cfg.damage_max_active_set,
                )
            except H8ScalarBoxKKTConvergenceError as exc:
                if trace_callback is not None:
                    trace_callback({"event": "inner_failure", "iteration": iteration,
                                    "evidence": dataclasses.asdict(exc.evidence),
                                    "inner_pcg": pcg_record(exc.last_linear_solve.krylov),
                                    "inner_solution": exc.last_solution.detach(), "d": d.detach()})
                finish("failed", iteration, residual, str(exc))
                raise _AttemptFailure(
                    f"{label} PF-CZM Newton subproblem failed at Newton iteration {iteration + 1}: {exc}",
                    pcg_pf=pcg + exc.evidence.total_pcg_iterations,
                    active_set=sets + exc.evidence.active_set_iterations,
                ) from exc
            except RuntimeError as exc:
                if not _is_scalar_solver_breakdown(exc):
                    raise
                finish("failed", iteration, residual, str(exc))
                raise _AttemptFailure(f"{label} PF-CZM Newton subproblem PCG breakdown: {exc}") from exc
            pcg += int(result.total_pcg_iterations)
            sets += int(result.active_set_iterations)
            target = result.solution
            if correction:
                target = d + target
                # Set physical Dirichlet bounds exactly after the coordinate translation.
                target = torch.where(result.lower_active | result.prescribed_equal_bound, lower_k, target)
                target = torch.where(result.upper_active, upper_k, target)
                # Free-entry addition can cross a bound by round-off only: the QP was feasible.
                target = torch.minimum(torch.maximum(target, lower_k), upper_k)
            step = target - d
            slope = float((gradient * step).sum()) / scale
            t = 1.0
            if trace_callback is not None:
                armijo_trials = []
            for _ in range(PFCZM_MAX_HALVINGS + 1):
                if t == 1.0:
                    trial = target
                else:
                    # the segment d -> target lies in the box; the clamp removes round-off only
                    trial = torch.minimum(torch.maximum(d + t * step, lower), ones)
                trial_energy = float(op.pfczm_damage_energy(trial, H))
                if trace_callback is not None:
                    armijo_trials.append({"factor": t, "energy": trial_energy,
                                          "threshold": energy + PFCZM_ARMIJO * t * slope
                                          + PFCZM_ENERGY_RTOL * abs(energy)})
                if math.isfinite(trial_energy) and trial_energy <= (
                    energy + PFCZM_ARMIJO * t * slope + PFCZM_ENERGY_RTOL * abs(energy)
                ):
                    break
                t *= 0.5
            else:
                if trace_callback is not None:
                    trace_callback({
                        "event": "line_search_failure", "iteration": iteration, "d": d.detach(),
                        "target": target.detach(), "slope": slope, "energy": energy,
                        "armijo_trials": armijo_trials, "inner_kkt": dataclasses.asdict(result.kkt),
                        "inner_pcg": pcg_record(result.last_linear_solve.krylov),
                        "active_set_trace": [dataclasses.asdict(row) for row in result.active_set_trace],
                    })
                finish("failed", iteration, residual, "line search failed")
                raise _AttemptFailure(
                    f"{label} PF-CZM line search failed at Newton iteration {iteration + 1} "
                    f"(no decrease after {PFCZM_MAX_HALVINGS} halvings)",
                    pcg_pf=pcg,
                    active_set=sets,
                )
            # A full Armijo step can already solve the physical nonlinear KKT.
            # Do not move away from it merely because total-energy round-off
            # makes an extrapolated point appear one ULP lower.
            # Keep the ordinary next-iteration stopping check and accounting.
            target_residual = projected_residual(trial) if t == 1.0 or trace_callback is not None else None
            target_converged = target_residual is not None and target_residual <= tol
            if trace_callback is not None:
                full_target_residual = target_residual if t == 1.0 else projected_residual(target)
                armijo_factor = accepted_factor = t
                target_energy = trial_energy
                doublings = []
            if t == 1.0 and not target_converged:
                # projected extrapolation: where the tangent floor overestimates the curvature (the
                # concave softening part of the local energy) the full step stops short; doubling it
                # along the projected path P(d + t step) reaches the bound d = 1 in a few iterations
                for _ in range(PFCZM_MAX_DOUBLINGS):
                    t *= 2.0
                    further = torch.minimum(torch.maximum(d + t * step, lower), ones)
                    if torch.equal(further, trial):
                        if trace_callback is not None:
                            doublings.append({"factor": t, "accepted": False, "reason": "identical field"})
                        break
                    further_energy = float(op.pfczm_damage_energy(further, H))
                    if trace_callback is not None:
                        accepted = math.isfinite(further_energy) and further_energy < trial_energy
                        doublings.append({"factor": t, "energy": further_energy,
                                          "residual": projected_residual(further), "accepted": accepted})
                    if not (math.isfinite(further_energy) and further_energy < trial_energy):
                        break
                    trial, trial_energy = further, further_energy
                    if trace_callback is not None:
                        accepted_factor = t
            if trace_callback is not None:
                # The quadratic is represented in the selected coordinates, but K is identical.
                curvature = float((step * system.apply(step)).sum())
                trace_callback({
                    "event": "step", "iteration": iteration,
                    "target_residual": target_residual, "full_target_residual": full_target_residual,
                    "final_residual": projected_residual(trial), "armijo_factor": armijo_factor,
                    "target_kkt_satisfied": target_converged,
                    "accepted_factor": accepted_factor, "step_norm": float(step.abs().max()),
                    "step_l2": float(torch.linalg.vector_norm(step)), "energy_before": energy,
                    "target_energy": target_energy, "energy_after": trial_energy, "slope": slope,
                    "armijo_allowance": PFCZM_ENERGY_RTOL * abs(energy),
                    "energy_ulp": math.ulp(energy),
                    "energy_delta_ulps": (trial_energy - energy) / math.ulp(energy),
                    "qp_descent_defect": float((gradient * step).sum()) + curvature,
                    "inner_kkt": dataclasses.asdict(result.kkt),
                    "inner_pcg": pcg_record(result.last_linear_solve.krylov),
                    "active_set_trace": [dataclasses.asdict(row) for row in result.active_set_trace],
                    "doublings": doublings, "armijo_trials": armijo_trials,
                })
            d = trial
            energy = trial_energy
            energies.append(energy)
        self.last_newton_energies = energies
        finish("failed", iteration, residual, "Newton iteration limit")
        raise _AttemptFailure(
            f"{label} PF-CZM Newton did not converge in {cfg.pfczm_max_newton} iterations "
            f"(projected gradient {residual:.3e} > damage_kkt_tol {tol:.3e})",
            pcg_pf=pcg,
            active_set=sets,
        )

    def _change(
        self,
        x_new: torch.Tensor,
        x_old: torch.Tensor,
        d_new: torch.Tensor,
        d_old: torch.Tensor,
        step_size: float,
    ) -> float:
        n = self._n_u
        parts = [
            torch.linalg.vector_norm(x_new[:n] - x_old[:n]),
            torch.linalg.vector_norm(x_new[:n]),
            (x_new[n:] - x_old[n:]).abs().max(),
            x_new[n:].abs().max(),
            (d_new - d_old).abs().max(),
        ]
        values = torch.stack(parts).cpu()
        du, u, dE, E_inf, dd = (float(v) for v in values)
        if not all(math.isfinite(v) for v in (du, u, dE, E_inf, dd)):
            raise _AttemptFailure("staggered iterate is not finite")
        scale_E = max(E_inf, step_size)
        u_ref = max(u, scale_E * self._u_floor)
        du_rel = du / u_ref if u_ref > 0.0 else 0.0
        dE_rel = dE / scale_E if scale_E > 0.0 else 0.0
        self.last_change_parts = (du_rel, dE_rel, dd)
        return max(du_rel, dE_rel, dd)

    def _log_iteration(
        self,
        *,
        increment: int,
        cutbacks: int,
        target: float,
        k: int,
        x: torch.Tensor,
        d: torch.Tensor,
        change: float,
        mech_iters: int,
        damage_iters: int,
        seconds: float,
        energies: tuple[float, float] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one staggered-iteration record to ``PHAST_SOLID3D_STAGGER_LOG`` (diagnostics only).

        Records the three convergence components separately (``du_rel``,
        ``dE_rel`` of the free macro strains, ``dd_inf``), the total energy
        ``E_el + Gamma`` at the current iterate, ``max d``, the six macro
        strains and the Krylov iteration counts.  The extra energy evaluations
        make each logged iteration slightly more expensive.  ``energies``
        passes ``(E_el, Gamma)`` when they are already known; ``extra`` adds
        keys (the Anderson fields of only when mixing is on).
        """

        if energies is None:
            g = self.op.degradation(d)
            elastic = float(self.op.elastic_energy(x, g))
            fracture = float(self.op.fracture_energy(d)['total'])
        else:
            elastic, fracture = (float(v) for v in energies)
        du_rel, dE_rel, dd = self.last_change_parts
        row = {
            "increment": int(increment),
            "cutbacks": int(cutbacks),
            "target": float(target),
            "k": int(k),
            "change": float(change),
            "du_rel": du_rel,
            "dE_rel": dE_rel,
            "dd_inf": dd,
            "elastic": elastic,
            "fracture": fracture,
            "total_energy": elastic + fracture,
            "max_d": float(d.max()),
            "macro_strain": [float(v) for v in x[self._n_u:].cpu()],
            "pcg_mech": int(mech_iters),
            "pcg_damage": int(damage_iters),
            "seconds": float(seconds),
        }
        if extra:
            row.update(extra)
        self._log_row(row)

    def _observe(self, state: SolverState, **row: Any) -> None:
        """Hand one staggered iterate to :attr:`iterate_observer` when its step is observed.

        ``row`` carries ``increment``, ``cutbacks``, ``target``, ``iteration``
        (k), ``x`` (the mechanics solution of the iteration, ``None`` when a
        solve of a mixed iteration broke down), ``g`` (the
        degradation used by that solve), ``d`` (the damage solve output, or the
        mixed input that was discarded), ``change``,
        ``input_mixed`` and ``discarded``; ``step`` (the step the attempt
        produces) and ``attempt`` (:attr:`n_attempts`) are added here.
        """

        step = int(state.step) + 1
        if self.iterate_observer is None or step not in self.observe_steps:
            return
        self.iterate_observer({"step": step, "attempt": int(self.n_attempts), **row})

    def _log_row(self, row: Mapping[str, Any]) -> None:
        with open(self._stagger_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def _total_energy(self, x: torch.Tensor, d: torch.Tensor, g: torch.Tensor | None = None) -> tuple[float, float]:
        """``(E_el(x, g(d)), Gamma(d))`` in mJ; ``g`` is reused when given."""

        g = self.op.degradation(d) if g is None else g
        return float(self.op.elastic_energy(x, g)), float(self.op.fracture_energy(d)["total"])

    def _predictor(self, state: SolverState, previous: float, target: float) -> torch.Tensor:
        if previous != 0.0:
            x = state.x * (target / previous)
        else:
            x = torch.zeros_like(state.x)
        x[self._n_u + self._controlled] = target
        return x

    def advance(
        self,
        state: SolverState,
        target: float,
        *,
        increment: int | None = None,
        cutbacks: int = 0,
    ) -> tuple[SolverState, IncrementRecord]:
        """Solve one (sub-)increment from ``state`` to ``E_c = target`` (a single attempt).

        ``increment`` (1-based, default ``state.increment + 1``) labels the
        record; the returned state's ``increment`` advances only when
        ``target`` is that load-case increment's target.  ``cutbacks`` is the
        halving depth recorded.  Raises :class:`StaggerConvergenceError` (no
        cutback) on failure.
        """

        self._check_state(state)
        target = float(target)
        if not math.isfinite(target):
            raise ValueError(f"target must be finite, got {target!r}")
        c = self._controlled
        previous = float(state.macro_strain[c])
        if target == previous:
            raise ValueError(f"target {target!r} equals the current controlled strain")
        increment = state.increment + 1 if increment is None else int(increment)
        self.n_attempts += 1
        self._sync()
        start = perf_counter()
        try:
            new_state, record = self._attempt(state, target, previous, increment, int(cutbacks), start)
        except _AttemptFailure as exc:
            self.n_failed_attempts += 1
            raise StaggerConvergenceError(
                f"increment {increment} (cutback level {cutbacks}) to E[{c}] = {target:.6e} failed: {exc}",
                state=state,
                target=target,
                increment=increment,
                cutbacks=int(cutbacks),
                reason=str(exc),
            ) from exc
        self.last_state = new_state
        return new_state, record

    def _attempt(
        self,
        state: SolverState,
        target: float,
        previous: float,
        increment: int,
        cutbacks: int,
        start: float,
    ) -> tuple[SolverState, IncrementRecord]:
        op, cfg = self.op, self.cfg
        c = self._controlled
        values = op.mech_fixed_values(target)
        step_size = abs(target - previous)
        x_old = self._predictor(state, previous, target)
        d_old = state.d.clone()
        lower = state.d
        mech_s = damage_s = history_s = 0.0
        pcg_mech = pcg_pf = active_set = 0
        converged = False
        change = math.inf
        kkt_ok = True
        H = state.H
        info: PCGInfo | None = None
        # Anderson mixing: a fresh history for every attempt, so it
        # never crosses an accepted step, an increment or a cutback.
        anderson = ProjectedAnderson(cfg.anderson_depth) if cfg.anderson_depth > 0 else None
        fallback: _PlainFallback | None = None
        best_energy = math.inf
        n_accepted = n_rejected = 0
        closure = op.has_closure
        closure_solves = 0
        self._newton_count = 0

        for k in range(1, cfg.max_stagger + 1):
            t0 = perf_counter()
            try:
                g = op.degradation(d_old)
                (x_new, info) = self._mechanics(g, values, x_old)
                pcg_mech += info.iterations
                if closure:
                    closure_solves += self.last_interface_solves
                t1 = perf_counter()
                Y = op.driving_force(x_new)
                H = torch.maximum(state.H, Y) if cfg.history == "max" else Y
                t2 = perf_counter()
                d_new, iters, sets, kkt_ok = self._damage(H, lower, d_old)
                pcg_pf += iters
                active_set += sets
                t3 = perf_counter()
                change = self._change(x_new, x_old, d_new, d_old, step_size)
            except _AttemptFailure as exc:
                if fallback is None:
                    raise
                # the iteration from a mixed iterate failed: return to the plain iterate; the
                # Krylov and active-set iterations its failing solve had spent count
                n_rejected += 1
                pcg_mech += exc.pcg_mech
                pcg_pf += exc.pcg_pf
                active_set += exc.active_set
                mech_s += perf_counter() - t0
                if self._stagger_log is not None:
                    self._log_row(
                        {
                            "increment": int(increment), "cutbacks": int(cutbacks), "target": float(target),
                            "k": int(k), "input": "mixed", "anderson": "rejected",
                            "anderson_reason": f"failure: {exc}",
                            "anderson_accepted": n_accepted, "anderson_rejected": n_rejected,
                            "seconds": float(perf_counter() - t0),
                        }
                    )
                if self.iterate_observer is not None:
                    self._observe(state, increment=increment, cutbacks=cutbacks, target=target, iteration=k, x=None, g=None, d=d_old, change=float('nan'), input_mixed=True, discarded=True)
                x_old, d_old = fallback.x, fallback.d
                fallback = None
                anderson.reset()  # type: ignore[union-attr]
                continue
            mech_s += t1 - t0
            history_s += t2 - t1
            damage_s += t3 - t2
            extra: dict[str, Any] | None = None
            energies: tuple[float, float] | None = None
            rejected = False
            if anderson is not None:
                residual = float(torch.linalg.vector_norm(d_new - d_old))
                energies = self._total_energy(x_new, d_new)
                extra = {
                    "input": "plain" if fallback is None else "mixed",
                    "fixed_point_residual": residual,
                }
                if fallback is not None:
                    # Judge the mixed input d_old by its own iteration.  The total
                    # energy E_el + Gamma at the end of that iteration must not
                    # exceed the lowest end-of-iteration total energy of the
                    # attempt so far (which includes the iteration that produced
                    # the replaced plain iterate), and its fixed-point residual
                    # must not exceed that of the iteration which produced the
                    # replaced plain iterate.  The energy reference never rises, so
                    # a kept mix cannot recur unless the energies tie to round-off,
                    # and the mixed iteration does not cycle.
                    energy = energies[0] + energies[1]
                    reference = best_energy
                    reasons = []
                    if energy > reference + ANDERSON_ENERGY_RTOL * abs(reference):
                        reasons.append("energy")
                    if residual > fallback.residual:
                        reasons.append("residual")
                    rejected = bool(reasons)
                    if rejected:
                        n_rejected += 1
                    else:
                        n_accepted += 1
                    extra.update({'anderson': 'rejected' if rejected else 'accepted', 'anderson_reason': '+'.join(reasons), 'reference_energy': reference, 'reference_residual': fallback.residual})
                if not rejected:
                    best_energy = min(best_energy, energies[0] + energies[1])
                extra.update(
                    {
                        "lowest_energy": best_energy,
                        "anderson_accepted": n_accepted,
                        "anderson_rejected": n_rejected,
                    }
                )
            if self._stagger_log is not None:
                self._log_iteration(increment=increment, cutbacks=cutbacks, target=target, k=k, x=x_new, d=d_new, change=change, mech_iters=info.iterations, damage_iters=iters, seconds=t3 - t0, energies=energies, extra=extra)
            if self.iterate_observer is not None:
                self._observe(state, increment=increment, cutbacks=cutbacks, target=target, iteration=k, x=x_new, g=g, d=d_new, change=float(change), input_mixed=fallback is not None, discarded=bool(rejected))
            if rejected:
                # discard this iteration and continue from the plain iterate
                x_old, d_old = fallback.x, fallback.d  # type: ignore[union-attr]
                fallback = None
                anderson.reset()  # type: ignore[union-attr]
                continue
            fallback = None
            if change <= cfg.stagger_tol and k >= cfg.min_stagger:
                x_old, d_old = x_new, d_new
                converged = True
                break
            if anderson is not None:
                anderson.push(d_old, d_new)
                mixed = anderson.mix(lower) if k >= cfg.anderson_start else None
                if mixed is not None and not torch.equal(mixed, d_new):
                    fallback = _PlainFallback(d=d_new, x=x_new, residual=residual)
                    x_old, d_old = x_new, mixed
                    continue
            x_old, d_old = x_new, d_new
        if not converged:
            raise _AttemptFailure(
                f"staggered iteration did not converge in {cfg.max_stagger} iterations "
                f"(change {change:.3e} > stagger_tol {cfg.stagger_tol:.3e})"
            )
        stagger_iters = k
        d = d_old

        t0 = perf_counter()
        g = op.degradation(d)
        if cfg.final_mechanics_resolve:
            (x, info) = self._mechanics(g, values, x_old)
            pcg_mech += info.iterations
            if closure:
                closure_solves += self.last_interface_solves
        else:
            x = x_old
        mech_s += perf_counter() - t0
        assert info is not None

        hill = op.hill_mandel(x, g)
        sigma = op.macro_stress(x, g)
        fracture = op.fracture_energy(d)
        elastic = 0.5 * op.volume * hill["mean_work"]
        element_d = d[op.pmap.element_nodes].mean(dim=1)
        _, E = op.split_mech(x)
        scalars = torch.cat(
            (
                E,
                sigma,
                op.mean_strain(x),
                torch.stack(
                    (
                        d.max(),
                        element_d.mean(),
                        (element_d >= cfg.damaged_threshold).to(self.dtype).mean(),
                        H.max(),
                    )
                ),
            )
        ).cpu().numpy()
        if not np.isfinite(scalars).all() or not math.isfinite(elastic) or not math.isfinite(fracture["total"]):
            raise _AttemptFailure("accepted state has non-finite stress, strain or energy")
        macro_strain = tuple(float(v) for v in scalars[:6])
        macro_stress = tuple(float(v) for v in scalars[6:12])
        mean_strain = tuple(float(v) for v in scalars[12:18])
        max_damage, mean_damage, damaged_fraction, max_H = (float(v) for v in scalars[18:22])
        closure_record: dict[str, Any] = {}
        if closure:
            # the closure solves of the increment and the kinematic guard of the accepted state
            guard = op.kinematic_guard(x, d)
            closure_record = {
                "closure_solves": int(closure_solves),
                "min_normal_strain": guard["min_normal_strain"],
                "min_principal_strain": guard["min_principal_strain"],
                "min_det_F": guard["min_det_F"],
            }

        sigma_prev = float(state.macro_stress[c])
        external = state.external_work + op.volume * 0.5 * (macro_stress[c] + sigma_prev) * (target - previous)
        residual = external - (elastic + fracture["total"])
        face_a, face_b = self._face_lengths

        completes = 1 <= increment <= self.load.n_increments and target == self.load.targets[increment - 1]
        new_state = SolverState(
            step=state.step + 1,
            x=x,
            d=d,
            H=H,
            macro_strain=E.clone(),
            macro_stress=sigma,
            external_work=external,
            d_prev_step=state.d,
            increment=increment if completes else state.increment,
        )
        self._sync()
        record = IncrementRecord(
            step=new_state.step,
            increment=increment,
            load_factor=target / self._max_abs_target,
            macro_strain=macro_strain,
            macro_stress=macro_stress,
            mean_strain=mean_strain,
            applied_disp=macro_strain[self._load_axis] * self._load_length,
            reaction_force=macro_stress[self._load_axis] * face_a * face_b,
            elastic=elastic,
            fracture=fracture["total"],
            fracture_by_phase=fracture["by_phase"],
            external=external,
            balance_residual=residual,
            balance_residual_rel=residual / max(abs(external), _TINY),
            max_damage=max_damage,
            mean_damage=mean_damage,
            damaged_volume_fraction=damaged_fraction,
            max_H=max_H,
            stagger_iters=stagger_iters,
            pcg_iters_mech=pcg_mech,
            pcg_iters_pf=pcg_pf,
            active_set_iters=active_set,
            cutbacks=cutbacks,
            converged=True,
            kkt_ok=kkt_ok,
            stagger_change=change,
            mech_relative_residual=info.relative_residual,
            hill_mandel_rel=hill["relative"],
            wall_s=perf_counter() - start,
            mech_s=mech_s,
            damage_s=damage_s,
            history_s=history_s,
            anderson_accepted=n_accepted if anderson is not None else None,
            anderson_rejected=n_rejected if anderson is not None else None,
            newton_iters_pf=self._newton_count if op.is_pfczm else None,
            **closure_record,
        )
        return new_state, record

    # ------------------------------------------------------------------
    # increments with cutback, and the load path
    # ------------------------------------------------------------------

    def advance_increment(
        self,
        state: SolverState,
        target: float,
        *,
        increment: int | None = None,
        on_increment: Callable[[SolverState, IncrementRecord], Any] | None = None,
    ) -> tuple[SolverState, list[IncrementRecord]]:
        """Solve one load-case increment with recursive half-step cutback.

        On a failed attempt with ``on_stagger_failure='cutback'`` the
        remaining interval is split into two halves, each solved the same way,
        down to ``max_cutbacks`` levels; deeper failures (or any failure with
        ``'raise'``) raise :class:`StaggerConvergenceError` whose ``state`` is
        the last accepted (possibly sub-step) state and whose ``records`` are
        the sub-steps accepted so far.  ``on_increment(state, record)`` is
        called after every accepted sub-step.  ``record.cutbacks`` is the
        halving depth of that sub-step (0 when the full increment converged).
        """

        increment = state.increment + 1 if increment is None else int(increment)
        target = float(target)
        c = self._controlled
        accepted: list[IncrementRecord] = []
        latest = [state]
        attempts = [0]

        def solve(current: SolverState, end: float, depth: int) -> SolverState:
            attempts[0] += 1
            try:
                new_state, record = self.advance(current, end, increment=increment, cutbacks=depth)
            except StaggerConvergenceError as exc:
                if self.cfg.on_stagger_failure == "raise" or depth >= self.cfg.max_cutbacks:
                    raise StaggerConvergenceError(
                        f"increment {increment} to E[{c}] = {target:.6e} failed at cutback level "
                        f"{depth} (max_cutbacks={self.cfg.max_cutbacks}, "
                        f"on_stagger_failure={self.cfg.on_stagger_failure!r}): {exc.reason}",
                        state=latest[0],
                        target=end,
                        increment=increment,
                        cutbacks=depth,
                        reason=exc.reason,
                        attempts=attempts[0],
                        records=tuple(accepted),
                    ) from exc
                begin = float(current.macro_strain[c])
                middle = 0.5 * (begin + end)
                current = solve(current, middle, depth + 1)
                return solve(current, end, depth + 1)
            accepted.append(record)
            latest[0] = new_state
            if on_increment is not None:
                on_increment(new_state, record)
            return new_state

        final = solve(state, target, 0)
        if final.increment != increment:
            final = dataclasses.replace(final, increment=increment)
            self.last_state = final
        return final, accepted

    def run(
        self,
        *,
        start_state: SolverState | None = None,
        max_increments: int | None = None,
        on_increment: Callable[[SolverState, IncrementRecord], Any] | None = None,
    ) -> list[IncrementRecord]:
        """Follow the load path from ``start_state`` (default :meth:`initial_state`).

        Solves load-case increments ``start_state.increment + 1, ...`` (at most
        ``max_increments`` of them) and returns every accepted record in order
        (the step-0 row is :meth:`initial_record`, not included).  The final
        state is :attr:`last_state`.  On failure the raised
        :class:`StaggerConvergenceError` carries every record accepted in this
        call and the last accepted state.
        """

        state = self.initial_state() if start_state is None else start_state
        self._check_state(state)
        self.last_state = state
        first = int(state.increment)
        last = self.load.n_increments
        if max_increments is not None:
            if isinstance(max_increments, bool) or int(max_increments) != max_increments or max_increments < 0:
                raise ValueError(f"max_increments must be an integer >= 0, got {max_increments!r}")
            last = min(last, first + int(max_increments))
        records: list[IncrementRecord] = []

        def accepted(new_state: SolverState, record: IncrementRecord) -> None:
            records.append(record)
            if on_increment is not None:
                on_increment(new_state, record)

        for index in range(first, last):
            try:
                state, _ = self.advance_increment(
                    state, self.load.targets[index], increment=index + 1, on_increment=accepted
                )
            except StaggerConvergenceError as exc:
                exc.records = tuple(records)
                self.last_state = exc.state
                raise
        self.last_state = state
        return records


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------


_STATE_TENSORS = ("x", "d", "H", "macro_strain", "macro_stress", "d_prev_step")


def save_checkpoint(path: str | Path, state: SolverState, *, meta: Mapping[str, Any] | None = None) -> Path:
    """Write ``state`` and JSON-serialisable ``meta`` with :func:`torch.save` (atomic replace).

    Tensors are stored on the CPU and ``meta`` as a JSON string, so the file
    loads with ``torch.load(..., weights_only=True)``.
    """

    if not isinstance(state, SolverState):
        raise TypeError("state must be a SolverState")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "state": {
            "step": int(state.step),
            "increment": int(state.increment),
            "external_work": float(state.external_work),
            **{name: getattr(state, name).detach().cpu().clone() for name in _STATE_TENSORS},
        },
        "meta_json": json.dumps(dict(meta or {}), sort_keys=True),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[SolverState, dict[str, Any]]:
    """Return ``(state on device, meta)`` from :func:`save_checkpoint` output."""

    target = resolve_device(device)
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} file")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version {payload.get('version')!r}")
    raw = payload["state"]
    for name in _STATE_TENSORS:
        if not isinstance(raw.get(name), torch.Tensor) or raw[name].dtype != torch.float64:
            raise ValueError(f"checkpoint field {name!r} must be a float64 tensor")
    state = SolverState(
        step=int(raw["step"]),
        x=raw["x"].to(target),
        d=raw["d"].to(target),
        H=raw["H"].to(target),
        macro_strain=raw["macro_strain"].to(target),
        macro_stress=raw["macro_stress"].to(target),
        external_work=float(raw["external_work"]),
        d_prev_step=raw["d_prev_step"].to(target),
        increment=int(raw["increment"]),
    )
    return state, json.loads(payload["meta_json"])


__all__ = [
    "ANDERSON_ENERGY_RTOL",
    "ANDERSON_REGULARISATION",
    "CHECKPOINT_FORMAT",
    "ProjectedAnderson",
    "SolverState",
    "StaggerConvergenceError",
    "StaggeredSolver",
    "load_checkpoint",
    "save_checkpoint",
]
