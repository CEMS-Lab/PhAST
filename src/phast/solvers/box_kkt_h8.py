"""Matrix-free scalar solves on H8 meshes: Dirichlet PCG and the box-constrained (KKT) damage solve with an active set."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import math
from numbers import Real
import torch


H8_SCALAR_BOX_KKT_IDENTIFIER = "h8_scalar_box_kkt_active_set_v1"


DEFAULT_LINEAR_TOLERANCE = 1.0e-10


DEFAULT_KKT_TOLERANCE = 1.0e-9


ScalarAction = Callable[[torch.Tensor], torch.Tensor]


def _finite_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")


def _require_vector(
    value: torch.Tensor,
    *,
    name: str,
    size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Validate one finite scalar nodal vector without coercing it."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1 or value.numel() != size:
        raise ValueError(f"{name} must have shape ({size},), got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains a non-finite value")
    return value


def _max_or_zero(values: torch.Tensor) -> float:
    return 0.0 if values.numel() == 0 else float(values.max().item())


def _min_or_none(values: torch.Tensor) -> float | None:
    return None if values.numel() == 0 else float(values.min().item())


@dataclass(frozen=True)
class H8ScalarSPDSystem:
    """Matrix-free SPD data for one scalar nodal H8 subproblem.

    ``action`` must represent the same SPD system whose ``rhs`` and positive
    Jacobi ``diagonal`` are supplied.  It is deliberately a small protocol,
    rather than a dependency on a particular H8 or AT2 implementation.  The
    action output is checked on every call so malformed matrix-free operators
    fail where they are used rather than being mistaken for a KKT failure.
    """

    action: ScalarAction
    rhs: torch.Tensor
    diagonal: torch.Tensor

    def __post_init__(self) -> None:
        if not callable(self.action):
            raise TypeError("action must be callable")
        if not isinstance(self.rhs, torch.Tensor):
            raise TypeError("rhs must be a torch.Tensor")
        if self.rhs.ndim != 1 or self.rhs.numel() == 0:
            raise ValueError("rhs must be a non-empty one-dimensional tensor")
        if not torch.is_floating_point(self.rhs):
            raise TypeError("rhs must use a floating-point dtype")
        if not bool(torch.isfinite(self.rhs).all()):
            raise ValueError("rhs contains a non-finite value")
        _require_vector(
            self.diagonal,
            name="diagonal",
            size=self.rhs.numel(),
            dtype=self.rhs.dtype,
            device=self.rhs.device,
        )
        if bool((self.diagonal <= 0.0).any()):
            raise ValueError("diagonal must be strictly positive")

    @property
    def n_dofs(self) -> int:
        """Number of scalar nodal degrees of freedom."""

        return self.rhs.numel()

    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """Apply the caller-owned matrix-free operator with shape checks."""

        _require_vector(
            field,
            name="field",
            size=self.n_dofs,
            dtype=self.rhs.dtype,
            device=self.rhs.device,
        )
        image = self.action(field)
        return _require_vector(
            image,
            name="action(field)",
            size=self.n_dofs,
            dtype=self.rhs.dtype,
            device=self.rhs.device,
        )


@dataclass(frozen=True)
class ScalarPCGResult:
    """Outcome of one diagonally preconditioned free-space CG solve."""

    solution: torch.Tensor
    iterations: int
    converged: bool
    initial_residual_norm: float
    residual_norm: float
    relative_residual: float


@dataclass(frozen=True)
class ScalarDirichletResult:
    """Full scalar field reconstructed from an exact-Dirichlet PCG solve."""

    solution: torch.Tensor
    krylov: ScalarPCGResult
    n_free_dofs: int

    @property
    def converged(self) -> bool:
        """Whether the free-space PCG solve satisfied its tolerance."""

        return self.krylov.converged


@dataclass(frozen=True)
class BoxKKTDiagnostics:
    """Primal/dual evidence for one scalar bounded solution.

    The retained gradient is ``r = A d - b``.  At a lower bound its correct
    sign is non-negative; at an upper bound its correct sign is non-positive.
    Equal lower/upper entries are prescribed Dirichlet values and have no
    feasible direction, so no multiplier sign is required there.
    """

    kkt_tolerance: float
    n_dofs: int
    active_lower_dofs: int
    active_upper_dofs: int
    prescribed_equal_bound_dofs: int
    free_dofs: int
    finite_lower_bound_dofs: int
    finite_upper_bound_dofs: int
    bounds_are_finite: bool
    lower_does_not_exceed_upper: bool
    primal_lower_violation_maximum: float
    primal_upper_violation_maximum: float
    primal_lower_strict_violation_dofs: int
    primal_upper_strict_violation_dofs: int
    primal_lower_tolerance_significant_violation_dofs: int
    primal_upper_tolerance_significant_violation_dofs: int
    full_gradient_l2: float
    full_gradient_maximum: float
    free_stationarity_l2: float
    free_stationarity_maximum: float
    lower_reaction_minimum: float | None
    upper_reaction_minimum: float | None
    lower_reaction_sign_violation_dofs: int
    upper_reaction_sign_violation_dofs: int
    projected_gradient_l2: float
    projected_gradient_maximum: float
    complementarity_l2: float
    complementarity_maximum: float
    satisfies_kkt: bool


@dataclass(frozen=True)
class BoxKKTIteration:
    """One exact-Dirichlet/PCG primal active-set iteration trace."""

    iteration: int
    active_lower_dofs: int
    active_upper_dofs: int
    prescribed_equal_bound_dofs: int
    free_dofs: int
    pcg_iterations: int
    pcg_converged: bool
    added_lower_dofs: int
    added_upper_dofs: int
    released_lower_dofs: int
    released_upper_dofs: int


@dataclass(frozen=True)
class H8ScalarBoxKKTResult:
    """A converged H8-0 field with its complete bounded-solve audit."""

    formulation_identifier: str
    solution: torch.Tensor
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor
    lower_active: torch.Tensor
    upper_active: torch.Tensor
    prescribed_equal_bound: torch.Tensor
    last_linear_solve: ScalarDirichletResult
    active_set_iterations: int
    total_pcg_iterations: int
    active_set_trace: tuple[BoxKKTIteration, ...]
    kkt: BoxKKTDiagnostics


@dataclass(frozen=True)
class BoxKKTFailureEvidence:
    """Inspectable evidence retained when the bounded H8-0 solve rejects."""

    reason: str
    active_set_iterations: int
    total_pcg_iterations: int
    active_set_trace: tuple[BoxKKTIteration, ...]
    kkt: BoxKKTDiagnostics


class H8ScalarBoxKKTConvergenceError(RuntimeError):
    """Raised instead of returning a non-converged or non-KKT field."""

    def __init__(
        self,
        reason: str,
        *,
        last_solution: torch.Tensor,
        last_linear_solve: ScalarDirichletResult,
        evidence: BoxKKTFailureEvidence,
    ) -> None:
        super().__init__(reason)
        self.last_solution = last_solution
        self.last_linear_solve = last_linear_solve
        self.evidence = evidence


def _pcg(
    action: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    diagonal: torch.Tensor,
    initial: torch.Tensor | None,
    tolerance: float,
    max_iterations: int,
) -> ScalarPCGResult:
    """Solve one free-space SPD system with diagonally preconditioned CG."""

    _finite_positive("tolerance", tolerance)
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be an integer >= 1")
    if rhs.ndim != 1 or not torch.is_floating_point(rhs):
        raise ValueError("rhs must be a one-dimensional floating-point tensor")
    if not bool(torch.isfinite(rhs).all()):
        raise ValueError("rhs contains a non-finite value")
    if diagonal.shape != rhs.shape or diagonal.dtype != rhs.dtype or diagonal.device != rhs.device:
        raise ValueError("diagonal must share rhs shape, dtype, and device")
    if not bool(torch.isfinite(diagonal).all()) or bool((diagonal <= 0.0).any()):
        raise ValueError("diagonal must be finite and strictly positive")

    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    if x.shape != rhs.shape or x.dtype != rhs.dtype or x.device != rhs.device:
        raise ValueError("initial must share rhs shape, dtype, and device")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("initial contains a non-finite value")

    residual = rhs - action(x)
    if residual.shape != rhs.shape or residual.dtype != rhs.dtype or residual.device != rhs.device:
        raise ValueError("PCG action returned an incompatible vector")
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("PCG action returned a non-finite vector")
    initial_norm = float(torch.linalg.vector_norm(residual).item())
    reference_norm = max(initial_norm, 1.0)
    if initial_norm <= tolerance * reference_norm:
        return ScalarPCGResult(
            solution=x,
            iterations=0,
            converged=True,
            initial_residual_norm=initial_norm,
            residual_norm=initial_norm,
            relative_residual=initial_norm / reference_norm,
        )

    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    residual_dot_preconditioned = torch.dot(residual, preconditioned)
    if not bool(torch.isfinite(residual_dot_preconditioned)) or residual_dot_preconditioned.item() <= 0.0:
        raise RuntimeError("PCG encountered a non-positive initial residual product")

    residual_norm = initial_norm
    for iteration in range(1, max_iterations + 1):
        image = action(direction)
        if image.shape != rhs.shape or image.dtype != rhs.dtype or image.device != rhs.device:
            raise ValueError("PCG action returned an incompatible vector")
        if not bool(torch.isfinite(image).all()):
            raise ValueError("PCG action returned a non-finite vector")
        denominator = torch.dot(direction, image)
        if not bool(torch.isfinite(denominator)) or denominator.item() <= 0.0:
            raise RuntimeError("PCG operator is not positive definite on the free space")
        alpha = residual_dot_preconditioned / denominator
        x = x + alpha * direction
        residual = residual - alpha * image
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        relative_residual = residual_norm / reference_norm
        if residual_norm <= tolerance * reference_norm:
            return ScalarPCGResult(
                solution=x,
                iterations=iteration,
                converged=True,
                initial_residual_norm=initial_norm,
                residual_norm=residual_norm,
                relative_residual=relative_residual,
            )

        next_preconditioned = residual / diagonal
        next_dot = torch.dot(residual, next_preconditioned)
        if not bool(torch.isfinite(next_dot)) or next_dot.item() <= 0.0:
            raise RuntimeError("PCG encountered a non-positive residual product")
        direction = next_preconditioned + (next_dot / residual_dot_preconditioned) * direction
        residual_dot_preconditioned = next_dot

    return ScalarPCGResult(
        solution=x,
        iterations=max_iterations,
        converged=False,
        initial_residual_norm=initial_norm,
        residual_norm=residual_norm,
        relative_residual=residual_norm / reference_norm,
    )


def solve_h8_scalar_dirichlet_pcg(
    system: H8ScalarSPDSystem,
    *,
    fixed: torch.Tensor,
    values: torch.Tensor,
    initial: torch.Tensor | None = None,
    tolerance: float = DEFAULT_LINEAR_TOLERANCE,
    max_iterations: int = 4_000,
) -> ScalarDirichletResult:
    """Solve a scalar system with exact matrix-free Dirichlet elimination.

    The free right hand side is ``b_f - (A d_c)_f``.  The free operator sees
    vectors whose constrained entries are exactly zero, which is the
    matrix-free row/column-elimination counterpart.  ``values`` are copied at
    ``fixed`` entries; no unconstrained solution is clipped or projected.
    """

    if not isinstance(fixed, torch.Tensor):
        raise TypeError("fixed must be a torch.Tensor")
    if fixed.ndim != 1 or fixed.shape != system.rhs.shape:
        raise ValueError("fixed must match rhs shape")
    if fixed.dtype != torch.bool:
        raise TypeError("fixed must use torch.bool")
    if fixed.device != system.rhs.device:
        raise ValueError("fixed must share rhs device")
    _require_vector(
        values,
        name="values",
        size=system.n_dofs,
        dtype=system.rhs.dtype,
        device=system.rhs.device,
    )
    if initial is not None:
        _require_vector(
            initial,
            name="initial",
            size=system.n_dofs,
            dtype=system.rhs.dtype,
            device=system.rhs.device,
        )

    known = torch.where(fixed, values, torch.zeros_like(values))
    free = ~fixed
    n_free = int(free.sum().item())
    if n_free == 0:
        zero = torch.empty(0, dtype=system.rhs.dtype, device=system.rhs.device)
        return ScalarDirichletResult(
            solution=known,
            krylov=ScalarPCGResult(
                solution=zero,
                iterations=0,
                converged=True,
                initial_residual_norm=0.0,
                residual_norm=0.0,
                relative_residual=0.0,
            ),
            n_free_dofs=0,
        )

    reduced_rhs = (system.rhs - system.apply(known))[free]
    reduced_diagonal = system.diagonal[free]

    def reduced_action(vector: torch.Tensor) -> torch.Tensor:
        full = torch.zeros_like(system.rhs)
        full[free] = vector
        return system.apply(full)[free]

    reduced_initial = None if initial is None else initial[free]
    krylov = _pcg(
        reduced_action,
        reduced_rhs,
        diagonal=reduced_diagonal,
        initial=reduced_initial,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    solution = known.clone()
    solution[free] = krylov.solution
    return ScalarDirichletResult(
        solution=solution,
        krylov=krylov,
        n_free_dofs=n_free,
    )


def _bound_vector(
    system: H8ScalarSPDSystem,
    value: torch.Tensor | Real,
    *,
    name: str,
) -> torch.Tensor:
    """Materialise one scalar/nodal bound on the system's vector contract."""

    if isinstance(value, torch.Tensor):
        return _require_vector(
            value,
            name=name,
            size=system.n_dofs,
            dtype=system.rhs.dtype,
            device=system.rhs.device,
        )
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar or a torch.Tensor")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return torch.full_like(system.rhs, scalar)


def _kkt_diagnostics(
    system: H8ScalarSPDSystem,
    solution: torch.Tensor,
    *,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    lower_active: torch.Tensor,
    upper_active: torch.Tensor,
    prescribed_equal_bound: torch.Tensor,
    kkt_tolerance: float,
) -> BoxKKTDiagnostics:
    """Audit a candidate without changing its active set or values."""

    gradient = system.apply(solution) - system.rhs
    fixed = lower_active | upper_active | prescribed_equal_bound
    free = ~fixed

    lower_violation = torch.clamp_min(lower_bound - solution, 0.0)
    upper_violation = torch.clamp_min(solution - upper_bound, 0.0)
    lower_reaction = gradient[lower_active]
    upper_reaction = -gradient[upper_active]
    free_gradient = gradient[free]

    projected_gradient = torch.zeros_like(gradient)
    projected_gradient[free] = free_gradient
    projected_gradient[lower_active] = torch.minimum(
        lower_reaction, torch.zeros_like(lower_reaction)
    )
    projected_gradient[upper_active] = torch.maximum(
        gradient[upper_active], torch.zeros_like(gradient[upper_active])
    )

    lower_multiplier = torch.zeros_like(gradient)
    upper_multiplier = torch.zeros_like(gradient)
    lower_multiplier[lower_active] = torch.clamp_min(lower_reaction, 0.0)
    upper_multiplier[upper_active] = torch.clamp_min(upper_reaction, 0.0)
    complementarity = torch.cat(
        (
            lower_multiplier * (solution - lower_bound),
            upper_multiplier * (upper_bound - solution),
        )
    )

    primal_lower_significant = solution < lower_bound - kkt_tolerance
    primal_upper_significant = solution > upper_bound + kkt_tolerance
    lower_sign_violation = lower_reaction < -kkt_tolerance
    upper_sign_violation = upper_reaction < -kkt_tolerance
    free_stationarity_maximum = _max_or_zero(free_gradient.abs())
    projected_gradient_maximum = _max_or_zero(projected_gradient.abs())
    complementarity_maximum = _max_or_zero(complementarity.abs())
    bounds_are_finite = bool(torch.isfinite(lower_bound).all()) and bool(
        torch.isfinite(upper_bound).all()
    )
    lower_does_not_exceed_upper = bool((lower_bound <= upper_bound).all())
    satisfies_kkt = bool(
        bounds_are_finite
        and lower_does_not_exceed_upper
        and not bool((solution < lower_bound).any())
        and not bool((solution > upper_bound).any())
        and not bool(lower_sign_violation.any())
        and not bool(upper_sign_violation.any())
        and free_stationarity_maximum <= kkt_tolerance
        and projected_gradient_maximum <= kkt_tolerance
        and complementarity_maximum <= kkt_tolerance
    )
    return BoxKKTDiagnostics(
        kkt_tolerance=kkt_tolerance,
        n_dofs=system.n_dofs,
        active_lower_dofs=int(lower_active.sum().item()),
        active_upper_dofs=int(upper_active.sum().item()),
        prescribed_equal_bound_dofs=int(prescribed_equal_bound.sum().item()),
        free_dofs=int(free.sum().item()),
        finite_lower_bound_dofs=int(torch.isfinite(lower_bound).sum().item()),
        finite_upper_bound_dofs=int(torch.isfinite(upper_bound).sum().item()),
        bounds_are_finite=bounds_are_finite,
        lower_does_not_exceed_upper=lower_does_not_exceed_upper,
        primal_lower_violation_maximum=_max_or_zero(lower_violation),
        primal_upper_violation_maximum=_max_or_zero(upper_violation),
        primal_lower_strict_violation_dofs=int((solution < lower_bound).sum().item()),
        primal_upper_strict_violation_dofs=int((solution > upper_bound).sum().item()),
        primal_lower_tolerance_significant_violation_dofs=int(
            primal_lower_significant.sum().item()
        ),
        primal_upper_tolerance_significant_violation_dofs=int(
            primal_upper_significant.sum().item()
        ),
        full_gradient_l2=float(torch.linalg.vector_norm(gradient).item()),
        full_gradient_maximum=_max_or_zero(gradient.abs()),
        free_stationarity_l2=float(torch.linalg.vector_norm(free_gradient).item()),
        free_stationarity_maximum=free_stationarity_maximum,
        lower_reaction_minimum=_min_or_none(lower_reaction),
        upper_reaction_minimum=_min_or_none(upper_reaction),
        lower_reaction_sign_violation_dofs=int(lower_sign_violation.sum().item()),
        upper_reaction_sign_violation_dofs=int(upper_sign_violation.sum().item()),
        projected_gradient_l2=float(torch.linalg.vector_norm(projected_gradient).item()),
        projected_gradient_maximum=projected_gradient_maximum,
        complementarity_l2=float(torch.linalg.vector_norm(complementarity).item()),
        complementarity_maximum=complementarity_maximum,
        satisfies_kkt=satisfies_kkt,
    )


def _failure(
    reason: str,
    *,
    system: H8ScalarSPDSystem,
    solution: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    lower_active: torch.Tensor,
    upper_active: torch.Tensor,
    prescribed_equal_bound: torch.Tensor,
    kkt_tolerance: float,
    last_linear_solve: ScalarDirichletResult,
    active_set_iterations: int,
    total_pcg_iterations: int,
    active_set_trace: list[BoxKKTIteration],
) -> H8ScalarBoxKKTConvergenceError:
    """Create a rejected result with scalar KKT evidence, never a fallback."""

    evidence = BoxKKTFailureEvidence(
        reason=reason,
        active_set_iterations=active_set_iterations,
        total_pcg_iterations=total_pcg_iterations,
        active_set_trace=tuple(active_set_trace),
        kkt=_kkt_diagnostics(
            system,
            solution,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            lower_active=lower_active,
            upper_active=upper_active,
            prescribed_equal_bound=prescribed_equal_bound,
            kkt_tolerance=kkt_tolerance,
        ),
    )
    return H8ScalarBoxKKTConvergenceError(
        reason,
        last_solution=solution,
        last_linear_solve=last_linear_solve,
        evidence=evidence,
    )


def solve_h8_scalar_box_kkt(
    system: H8ScalarSPDSystem,
    *,
    lower_bound: torch.Tensor | Real = 0.0,
    upper_bound: torch.Tensor | Real = 1.0,
    initial: torch.Tensor | None = None,
    linear_tolerance: float = DEFAULT_LINEAR_TOLERANCE,
    kkt_tolerance: float = DEFAULT_KKT_TOLERANCE,
    max_linear_iterations: int = 4_000,
    max_active_set_iterations: int = 64,
) -> H8ScalarBoxKKTResult:
    """Solve a scalar H8 SPD quadratic under exact nodal box constraints.

    ``lower_bound`` may be the previous damage field for irreversibility and
    ``upper_bound`` may be the physical AT2 cap.  Bounds are not post-processed
    by clipping: every final active entry is supplied as a Dirichlet value in
    the next linear solve.  An infeasible ``initial`` is rejected rather than
    modified.  The returned candidate must pass the retained KKT audit;
    otherwise a :class:`H8ScalarBoxKKTConvergenceError` carries the last
    field and diagnostics.
    """

    if not isinstance(system, H8ScalarSPDSystem):
        raise TypeError("system must be an H8ScalarSPDSystem")
    _finite_positive("linear_tolerance", linear_tolerance)
    _finite_nonnegative("kkt_tolerance", kkt_tolerance)
    if not isinstance(max_linear_iterations, int) or max_linear_iterations < 1:
        raise ValueError("max_linear_iterations must be an integer >= 1")
    if not isinstance(max_active_set_iterations, int) or max_active_set_iterations < 1:
        raise ValueError("max_active_set_iterations must be an integer >= 1")

    lower = _bound_vector(system, lower_bound, name="lower_bound")
    upper = _bound_vector(system, upper_bound, name="upper_bound")
    if bool((lower > upper).any()):
        raise ValueError("lower_bound must not exceed upper_bound at any degree of freedom")
    if initial is not None:
        _require_vector(
            initial,
            name="initial",
            size=system.n_dofs,
            dtype=system.rhs.dtype,
            device=system.rhs.device,
        )
        if bool((initial < lower).any()) or bool((initial > upper).any()):
            raise ValueError("initial must already satisfy the requested box bounds; it is never clipped")

    prescribed_equal_bound = lower == upper
    lower_active = torch.zeros_like(prescribed_equal_bound)
    upper_active = torch.zeros_like(prescribed_equal_bound)
    candidate_initial = initial
    active_set_trace: list[BoxKKTIteration] = []
    total_pcg_iterations = 0
    last_linear_solve: ScalarDirichletResult | None = None
    candidate = torch.zeros_like(system.rhs)

    for active_iteration in range(1, max_active_set_iterations + 1):
        fixed = lower_active | upper_active | prescribed_equal_bound
        values = lower.clone()
        values[upper_active] = upper[upper_active]
        linear_solve = solve_h8_scalar_dirichlet_pcg(
            system,
            fixed=fixed,
            values=values,
            initial=candidate_initial,
            tolerance=linear_tolerance,
            max_iterations=max_linear_iterations,
        )
        last_linear_solve = linear_solve
        candidate = linear_solve.solution
        total_pcg_iterations += linear_solve.krylov.iterations
        gradient = system.apply(candidate) - system.rhs
        free = ~fixed

        # Strict primal violations change the active set.  The KKT tolerance
        # belongs only to multiplier/stationarity tests, never to clipping or
        # a hidden primal projection.
        add_lower = free & (candidate < lower)
        add_upper = free & (candidate > upper)
        release_lower = lower_active & (gradient < -kkt_tolerance)
        release_upper = upper_active & (gradient > kkt_tolerance)
        active_set_trace.append(
            BoxKKTIteration(
                iteration=active_iteration,
                active_lower_dofs=int(lower_active.sum().item()),
                active_upper_dofs=int(upper_active.sum().item()),
                prescribed_equal_bound_dofs=int(prescribed_equal_bound.sum().item()),
                free_dofs=int(free.sum().item()),
                pcg_iterations=linear_solve.krylov.iterations,
                pcg_converged=linear_solve.krylov.converged,
                added_lower_dofs=int(add_lower.sum().item()),
                added_upper_dofs=int(add_upper.sum().item()),
                released_lower_dofs=int(release_lower.sum().item()),
                released_upper_dofs=int(release_upper.sum().item()),
            )
        )
        if not linear_solve.krylov.converged:
            raise _failure(
                "H8 scalar box-KKT PCG did not converge",
                system=system,
                solution=candidate,
                lower_bound=lower,
                upper_bound=upper,
                lower_active=lower_active,
                upper_active=upper_active,
                prescribed_equal_bound=prescribed_equal_bound,
                kkt_tolerance=kkt_tolerance,
                last_linear_solve=linear_solve,
                active_set_iterations=active_iteration,
                total_pcg_iterations=total_pcg_iterations,
                active_set_trace=active_set_trace,
            )

        if not bool((add_lower | add_upper | release_lower | release_upper).any()):
            kkt = _kkt_diagnostics(
                system,
                candidate,
                lower_bound=lower,
                upper_bound=upper,
                lower_active=lower_active,
                upper_active=upper_active,
                prescribed_equal_bound=prescribed_equal_bound,
                kkt_tolerance=kkt_tolerance,
            )
            if not kkt.satisfies_kkt:
                raise _failure(
                    "H8 scalar box-KKT active set reached a non-KKT state",
                    system=system,
                    solution=candidate,
                    lower_bound=lower,
                    upper_bound=upper,
                    lower_active=lower_active,
                    upper_active=upper_active,
                    prescribed_equal_bound=prescribed_equal_bound,
                    kkt_tolerance=kkt_tolerance,
                    last_linear_solve=linear_solve,
                    active_set_iterations=active_iteration,
                    total_pcg_iterations=total_pcg_iterations,
                    active_set_trace=active_set_trace,
                )
            return H8ScalarBoxKKTResult(
                formulation_identifier=H8_SCALAR_BOX_KKT_IDENTIFIER,
                solution=candidate,
                lower_bound=lower,
                upper_bound=upper,
                lower_active=lower_active,
                upper_active=upper_active,
                prescribed_equal_bound=prescribed_equal_bound,
                last_linear_solve=linear_solve,
                active_set_iterations=active_iteration,
                total_pcg_iterations=total_pcg_iterations,
                active_set_trace=tuple(active_set_trace),
                kkt=kkt,
            )

        if active_iteration == max_active_set_iterations:
            break
        lower_active = (lower_active | add_lower) & ~release_lower
        upper_active = (upper_active | add_upper) & ~release_upper
        candidate_initial = candidate

    assert last_linear_solve is not None  # Established by max iterations >= 1.
    raise _failure(
        "H8 scalar box-KKT active set did not stabilise",
        system=system,
        solution=candidate,
        lower_bound=lower,
        upper_bound=upper,
        lower_active=lower_active,
        upper_active=upper_active,
        prescribed_equal_bound=prescribed_equal_bound,
        kkt_tolerance=kkt_tolerance,
        last_linear_solve=last_linear_solve,
        active_set_iterations=max_active_set_iterations,
        total_pcg_iterations=total_pcg_iterations,
        active_set_trace=active_set_trace,
    )
