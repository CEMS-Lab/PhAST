"""
Staggered solver: orchestrates mechanics + damage sub-problems.

Supports three solution strategies:
  1. 'explicit'    — Explicit dynamics (Velocity-Verlet) + AT2 CG
  2. 'static'      — Static equilibrium (CG) + AT2 CG (single load step)
  3. 'quasi_static' — NR equilibrium + AT2 CG (incremental loading)

The staggered scheme per timestep:
  1. Advance displacement (mechanics sub-problem)
  2. Compute psi+ (tensile strain energy) and update H = max(H, psi+)
  3. Solve AT2 damage PDE
  4. (Implicit in next step: damage degrades stiffness)

Literature improvements over the original monolithic solver:
  - H capping for out-of-distribution stability (Pantidis 2026)
  - Modular sub-solvers (swappable mechanics + damage)
  - Support for quasi-static NR (not just explicit dynamics)
  - Hooks for neural operator injection (set_damage, get_H_nodal)
"""

import torch
import time
import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, Literal

from ..core.mesh import FEMMesh
from ..physics.material import Material
from ..physics.boundary_conditions import BoundaryConditions
from ..core.fem_operators import FEMOperators
from .damage_solver import PhaseFieldDamageSolver
from .mechanics_solver import (ExplicitDynamics, GeneralizedAlphaDynamics,
                                StaticSolver,
                                QuasiStaticSolver, SecantCGSolver, LBFGSSolver,
                                MonolithicSolver)
from ..utils.io_utils import (write_vtu, write_visualization, write_h5_snapshot,
                        init_h5, CSVHistory)
from ..utils.device import DeviceContext
from ..plasticity import DuctilePhaseFieldCoupling, MeshJ2Elastoplasticity


@dataclass
class SolverConfig:
    """Solver configuration parameters."""
    solver_type: Literal['explicit', 'static', 'quasi_static', 'quasi_static_legacy', 'lbfgs', 'monolithic'] = 'explicit'
    time_integrator: Literal['central_difference', 'verlet', 'newmark',
                             'generalized_alpha', 'gen_alpha'] = 'central_difference'
    rho_inf: float = 0.5
    dt: Optional[float] = None                  # timestep (explicit); None = CFL auto
    dt_safety: float = 1.0                      # CFL safety factor
    num_steps: int = 200
    damage_tol: float = 1e-5                    # AT2 CG convergence tolerance
    damage_max_iter: int = 5000
    static_tol: float = 1e-8                    # static CG / NR tolerance
    static_max_iter: int = 5000
    H_cap: Optional[float] = None               # cap H for stability; None = no cap
    # Smooth-max for H update — enables differentiable inverse problems.
    # Standard torch.maximum(H, psi) has piecewise-discontinuous gradients: the
    # gradient only flows through the single timestep that set the max. When a
    # parameter change shifts which timestep peaks, the gradient jumps. This
    # breaks L-BFGS on long dynamic inversions.
    # softmax_H_beta > 0 replaces torch.maximum with a smooth approximation:
    #     H_new = H + softplus(β*(psi-H)) / β
    # which is smooth everywhere, → torch.maximum as β → ∞.
    # Recommended: 10-100 for inversion, None for forward-only (exact physics).
    softmax_H_beta: Optional[float] = None
    # Issue #360 — H-update dispatcher. Selects the operator used inside
    # ``_H_update``. Default ``'hard_max'`` reproduces ``torch.maximum`` byte
    # for byte (zero-cost wrapper) so all forward benchmarks remain bit
    # identical when the flag is unset. ``'softmax'``, ``'smooth_max'``,
    # ``'log_smooth'`` (#361), and ``'custom_subgrad'`` (#362) are
    # differentiable alternatives.
    # Legacy ``softmax_H_beta`` is retained: when set with the default
    # ``hard_max`` method, the softplus path takes precedence.
    H_update_method: str = 'hard_max'
    H_update_scale: Optional[float] = None      # custom_subgrad backward scale
    stagger_tol: float = 1e-6                   # stagger convergence tolerance
    max_stagger: int = 500
    stagger_criterion: str = 'relative'         # 'relative'|'absolute'|'linf'|'residual'|'am_energy'
    # Norm used inside the 'relative' stagger convergence check.
    # 'l2' (default) preserves historical behaviour; 'linf' switches to
    # max-norm relative change |Delta x|_inf / |x|_inf, the criterion
    # adopted by Bleyer & Roux-Langlois (2017) for anisotropic phase
    # field. Issue #244. Ignored for non-'relative' criteria.
    stagger_norm: str = 'l2'                    # 'l2' | 'linf'
    use_multigrid: bool = True                  # GMG preconditioner for damage CG
    bounds_method: str = 'post_clamp'           # 'post_clamp' or 'projected_cg'
    preconditioner: Optional[str] = None
    adaptive_refine: bool = False               # NVB adaptive mesh refinement
    refine_every: int = 10
    refine_d_threshold: float = 0.5
    refine_grad_d_threshold: float = 0.3
    anderson_depth: int = 0                     # 0=off; 3-5 typical (Storvik 2021)
    adaptive_stagger_tol: bool = False          # loose tol in elastic phase
    differentiable: bool = False
    fail_on_mechanics_nonconvergence: bool = True
    fail_on_stagger_nonconvergence: bool = True
    # Quasistatic damage-rate regularisation.  Zero preserves the rate-
    # independent AT1/AT2 solve.  When positive, the damage equation receives
    # an implicit mass-like term eta * (d - d_n) / dt, which damps abrupt
    # damage jumps in difficult continuation diagnostics.  Use only with an
    # explicit validation note because it changes the solved problem.
    damage_viscosity: float = 0.0
    # Kelvin-Voigt stiffness-proportional damping at omega_max.  See
    # ExplicitDynamics docstring.  Zero = pure Verlet (default).
    damping_ratio_max: float = 0.0
    enable_damage: bool = True                  # False = pure elastic (skip damage solve + H update)
    # Linear-solver backend for the QuasiStaticSolver Newton inner step.
    # Forwarded to QuasiStaticSolver(backend=...) in _build_mechanics_solver.
    # 'auto' picks the best available sparse-direct path; 'scipy', 'mumps',
    # or 'cg' force a specific backend. Required so YAML configs can
    # override the default without editing _build_mechanics_solver. (#196)
    backend: str = 'auto'
    newton_line_search: bool = True             # Residual backtracking for QuasiStaticSolver
    line_search_max_steps: int = 8
    line_search_min_alpha: float = 1e-4
    line_search_c: float = 1e-4
    adaptive_dt: bool = False                   # Adaptive time stepping for explicit dynamics
    adaptive_dt_d_threshold: float = 0.01       # Legacy alias for dt_cutback_threshold
    adaptive_dt_min_factor: float = 0.1         # Legacy: minimum dt as fraction of CFL dt
    adaptive_dt_grow_rate: float = 1.1          # dt growth factor when Δd is quiescent
    # Explicit growth/cutback heuristic: cut dt by 0.5x when delta-d_max in
    # a step exceeds dt_cutback_threshold; grow by adaptive_dt_grow_rate
    # toward the live CFL bound when delta-d_max falls below
    # dt_growth_threshold. dt is hard-floored at dt_floor and capped at the
    # live fem.recompute_dt_cfl() * dt_safety bound.
    dt_growth_threshold: float = 0.001          # Δd_max below which dt grows
    dt_cutback_threshold: float = 0.01          # Δd_max above which dt halves
    dt_floor: float = 1e-12                     # absolute lower bound on dt [s]
    damage_every: int = 3                       # Solve damage every N-th explicit step (subcycling)
                                                # Default 3: damage wave ~0.6*c_R, CFL based on c_p ≈ 3*0.6*c_R
                                                # Use 1 for no subcycling (legacy behavior)
    # Issue #299 — Verlet ordering: corrector damage freshness.
    #   False (default): corrector uses ``d_n`` (lagged damage). This is the
    #     "fully decoupled lagged" explicit scheme matching Borden 2012's
    #     canonical ordering — preserves bit-exact reproducibility against
    #     all existing benchmarks (B1, B5, dataset_benchmark dynamic runs).
    #   True: predictor -> step_solve_damage(d_{n+1}) -> corrector with the
    #     fresh ``d_{n+1}``. Matches PhaFiDyn (Barki 2025) and the typical
    #     "segregated" interpretation of COMSOL's N=2 staggered solver.
    #     This matches PhaFiDyn-style segregated ordering and was retained as
    #     an opt-in dynamic-fracture variant rather than the default Borden
    #     ordering. Subcycling still applies: damage is solved only on the
    #     gated steps, but on those steps the corrector uses the just-updated
    #     d. Costs ~3-5% wall time from the second internal_force assembly
    #     inside the corrector. NOT compatible
    #     with rigid_connector MPC (the unified _step_mpc cannot be split
    #     into predictor + damage + corrector); raises if both are active.
    fresh_d_in_corrector: bool = False
    dump_every: int = 10                        # VTU frequency (0=off)
    h5_every: int = 1                           # H5 frequency (0=off)
    viz_format: str = 'vtu'                     # 'vtu' (ParaView) | 'pv' (pyvista-zstd)
    print_every: int = 10

    def __post_init__(self):
        if (self.preconditioner is None and
                self.solver_type in ('static', 'quasi_static',
                                     'quasi_static_legacy', 'lbfgs',
                                     'monolithic')):
            self.preconditioner = 'jacobi'


class _AndersonAccelerator:
    """Anderson Acceleration (Type II) for the stagger fixed-point on damage.

    Stores m previous (d_in, d_out) pairs where d_out = G(d_in) is the
    damage solver output given mechanics solved with d_in.  Uses
    least-squares on residual differences to extrapolate the next iterate.

    Falls back to the standard iterate if:
      - The least-squares solve is singular
      - The accelerated iterate has larger residual norm (safeguarding)

    References
    ----------
    Anderson (1965), Walker & Ni (2011), Storvik et al. (2021).
    """

    def __init__(self, m: int = 5, reg: float = 1e-10):
        self.m = m
        self.reg = reg
        self._d_hist = deque(maxlen=m + 1)    # d_k inputs
        self._gd_hist = deque(maxlen=m + 1)   # G(d_k) outputs
        self._f_hist = deque(maxlen=m + 1)    # f_k = G(d_k) - d_k residuals

    def reset(self):
        """Clear history (call at each new load step)."""
        self._d_hist.clear()
        self._gd_hist.clear()
        self._f_hist.clear()

    @torch.no_grad()
    def step(self, d_in: torch.Tensor, d_out: torch.Tensor,
             d_prev_step: torch.Tensor = None) -> torch.Tensor:
        """Return accelerated damage field.

        Parameters
        ----------
        d_in : damage field before this stagger iteration
        d_out : damage field after damage solve (= G(d_in))
        d_prev_step : damage at start of load step (for irreversibility)

        Returns
        -------
        d_aa : accelerated damage estimate, clamped to [d_prev_step, 1]
        """
        f = d_out - d_in

        self._d_hist.append(d_in.clone())
        self._gd_hist.append(d_out.clone())
        self._f_hist.append(f.clone())

        m_k = len(self._f_hist) - 1
        if m_k == 0:
            return d_out  # first iteration, no history

        f_k = self._f_hist[-1]

        # DeltaF: columns are f_{i+1} - f_i for i in [0, m_k)
        delta_F = torch.stack([
            self._f_hist[i + 1] - self._f_hist[i]
            for i in range(m_k)
        ], dim=1)  # (N, m_k)

        # Normal equations: (DF^T DF + reg*I) gamma = DF^T f_k
        FtF = delta_F.T @ delta_F
        Ftf = delta_F.T @ f_k
        FtF.diagonal().add_(self.reg)

        try:
            gamma = torch.linalg.solve(FtF, Ftf)
        except RuntimeError:
            return d_out  # singular — fall back

        # DeltaG: columns are g_{i+1} - g_i
        delta_G = torch.stack([
            self._gd_hist[i + 1] - self._gd_hist[i]
            for i in range(m_k)
        ], dim=1)  # (N, m_k)

        d_aa = d_out - delta_G @ gamma

        # Safeguard: if AA increased residual norm, reject
        f_aa_norm = (d_aa - d_in).norm()
        f_std_norm = f.norm()
        if f_aa_norm > 1.5 * f_std_norm:
            return d_out

        if d_prev_step is not None:
            d_aa = torch.maximum(d_aa, d_prev_step)
        else:
            d_aa = d_aa.clamp(min=0.0)
        return d_aa.clamp(max=1.0)


class StaggeredSolver:
    """Orchestrator for staggered phase-field fracture.

    Parameters
    ----------
    mesh : FEMMesh
    material : Material
    bcs : BoundaryConditions
    config : SolverConfig
    """

    def __init__(self, mesh: FEMMesh, material: Material,
                 bcs: BoundaryConditions, config: SolverConfig = None,
                 ctx: DeviceContext = None):
        if config is None:
            config = SolverConfig()

        print(f"[StaggeredSolver] Assembling solver ({config.solver_type})...",
              flush=True)

        self.mesh = mesh
        self.material = material
        self.bcs = bcs
        self.config = config
        self.ctx = ctx
        self.device = mesh.device
        self.dtype = mesh.dtype
        self.steps = 0

        plasticity_model = getattr(material, 'plasticity_model', 'none')
        self.plasticity_operator = None
        self.ductile_coupling = None
        if plasticity_model != 'none':
            self._validate_plasticity_staggered_support(plasticity_model)

        # FEM operators
        self.fem = FEMOperators(mesh, material, ctx=ctx)

        if plasticity_model != 'none':
            self.plasticity_operator = MeshJ2Elastoplasticity(mesh, material)
            self.ductile_coupling = DuctilePhaseFieldCoupling(
                fem=self.fem, plasticity=self.plasticity_operator,
                plastic_work_weight=1.0)

        # Damage solver
        # AT1 requires projected_cg or direct: post_clamp produces d≡0 because
        # AT1 has no Gc/l0 mass term, so elements with H=0 get negative RHS
        # and the unconstrained CG solution is hugely negative everywhere,
        # dragging the crack-tip solution negative via Laplacian coupling.
        bounds = config.bounds_method
        if (getattr(material, 'pf_model', 'AT2') == 'AT1'
                and bounds == 'post_clamp'):
            bounds = 'projected_cg'
            print("[StaggeredSolver] AT1 model detected: switching bounds_method "
                  "from 'post_clamp' to 'projected_cg' (required for AT1)",
                  flush=True)
        self.damage_solver = PhaseFieldDamageSolver(
            self.fem, tol=config.damage_tol,
            max_iter=config.damage_max_iter, ctx=ctx,
            use_multigrid=config.use_multigrid,
            bounds_method=bounds,
            preconditioner=config.preconditioner)
        self.damage_solver.differentiable = config.differentiable
        self.damage_solver.damage_viscosity = float(config.damage_viscosity)

        # Mechanics solver (type-dependent)
        self.mechanics = self._build_mechanics_solver()
        if config.solver_type == 'explicit':
            self.dt = self.mechanics.dt
            ti = config.time_integrator
            if ti in ('verlet', 'newmark'):
                ti = 'central_difference'
            elif ti == 'gen_alpha':
                ti = 'generalized_alpha'
            print(f"[StaggeredSolver] Dynamic({ti}): dt={self.dt:.6e}",
                  flush=True)
            if config.damage_every > 1:
                print(f"[StaggeredSolver] Phase-field subcycling: damage solved every "
                      f"{config.damage_every} steps", flush=True)
        else:
            self.dt = None
            solver_names = {
                'static': 'StaticSolver', 'quasi_static': 'QuasiStaticSolver',
                'quasi_static_legacy': 'SecantCGSolver',
                'lbfgs': 'LBFGSSolver', 'monolithic': 'MonolithicSolver',
            }
            name = solver_names.get(config.solver_type, config.solver_type)
            print(f"[StaggeredSolver] {name} ready", flush=True)

        # State variables
        N = mesh.n_nodes
        E = mesh.n_elems
        self.u = torch.zeros(N, 2, dtype=self.dtype, device=self.device)
        self.v = torch.zeros(N, 2, dtype=self.dtype, device=self.device)
        self.a = torch.zeros(N, 2, dtype=self.dtype, device=self.device)
        self.d = torch.zeros(N, dtype=self.dtype, device=self.device)
        # Apply phase-field Dirichlet pins to the initial damage so the
        # very first step sees ``phi = value`` at the locked nodes
        # (issue #213). No-op if no pf_dirichlet BC is registered.
        self._apply_pf_dirichlet()
        H_shape = (
            (E, mesh.quad_N.shape[0])
            if getattr(mesh, 'element_type', 'T3') == 'Q4'
            else (E,)
        )
        self.H_elem = torch.zeros(H_shape, dtype=self.dtype, device=self.device)
        self.H_nodal = torch.zeros(N, dtype=self.dtype, device=self.device)
        self.f_ext = torch.zeros(N, 2, dtype=self.dtype, device=self.device)

        self._step_count = 0
        # Per-step solver telemetry (issue #300, sub of #298 PF-Hetero-Bench).
        # Pure observability — not consumed by any solver path. Refreshed on
        # every step_full() call. ``_last_stagger_iter`` is set in step_full()
        # below; the residual / dt fields are seeded here so a caller that
        # reads them before the first step gets a deterministic NaN/0.0.
        self._last_stagger_iter = 0
        self._last_residual = float('nan')   # max(u_change, d_change) at convergence
        self._last_residual0 = float('nan')
        self._last_relative_residual = float('nan')
        self._last_mechanics_residual = float('nan')
        self._last_mechanics_residual0 = float('nan')
        self._last_mechanics_relative_residual = float('nan')
        self._last_mechanics_converged = True
        self._last_mechanics_iter = 0
        self._last_dt_used = 0.0             # for adaptive-dt history (forward compat)
        print(f"[StaggeredSolver] State allocated: {N} nodes, {E} elements, "
              f"device={self.device}, dtype={self.dtype}", flush=True)

    def _update_stagger_residual_diagnostics(self, residual_norm: float,
                                             residual0: float | None) -> None:
        """Record absolute and relative stagger residual diagnostics."""
        self._last_residual = float(residual_norm)
        if residual0 is None:
            self._last_residual0 = float('nan')
            self._last_relative_residual = float('nan')
            return
        self._last_residual0 = float(residual0)
        if residual0 > 0.0:
            self._last_relative_residual = float(residual_norm / residual0)
        elif residual_norm == 0.0:
            self._last_relative_residual = 0.0
        else:
            self._last_relative_residual = float('nan')

    def _mirror_mechanics_diagnostics(self) -> None:
        """Mirror the current mechanics-solver diagnostics onto this solver."""
        self._last_mechanics_iter = int(
            getattr(self.mechanics, 'last_iter', 0))
        self._last_mechanics_converged = bool(
            getattr(self.mechanics, 'last_converged', True))
        self._last_mechanics_residual = float(
            getattr(self.mechanics, 'last_residual', float('nan')))
        self._last_mechanics_residual0 = float(
            getattr(self.mechanics, 'last_residual0', float('nan')))
        self._last_mechanics_relative_residual = float(
            getattr(self.mechanics, 'last_relative_residual', float('nan')))

    # ------------------------------------------------------------------ #
    # Mechanics solver construction
    # ------------------------------------------------------------------ #

    def _validate_plasticity_staggered_support(self, plasticity_model: str) -> None:
        """Guard the first supported staggered PF-plasticity slice."""
        if plasticity_model != 'j2_isotropic':
            raise NotImplementedError(
                "Staggered PF-plasticity currently supports only "
                "plasticity_model='j2_isotropic'. "
                f"Got {plasticity_model!r}.")
        if self.config.solver_type != 'quasi_static':
            raise NotImplementedError(
                "Staggered PF-plasticity is currently enabled only for "
                "solver_type='quasi_static'.")
        if getattr(self.mesh, 'element_type', 'T3') != 'T3':
            raise NotImplementedError(
                "Staggered PF-plasticity currently supports T3 meshes only.")
        if getattr(self.material, 'pf_model', 'AT2') != 'AT2':
            raise NotImplementedError(
                "Staggered PF-plasticity currently supports pf_model='AT2' "
                "only.")
        if self.config.backend == 'cg':
            raise NotImplementedError(
                "Staggered PF-plasticity requires sparse-direct mechanics "
                "backend ('auto', 'scipy', or optional mumps/cudss), not "
                "backend='cg'.")
        if getattr(self.bcs, 'rigid_connectors', None):
            raise NotImplementedError(
                "Staggered PF-plasticity does not yet support rigid "
                "connector MPCs.")

    def _build_mechanics_solver(self):
        """Create the appropriate mechanics solver based on config."""
        cfg = self.config
        st = cfg.solver_type
        if st == 'explicit':
            ti = cfg.time_integrator
            if ti in ('verlet', 'newmark'):
                ti = 'central_difference'
            elif ti == 'gen_alpha':
                ti = 'generalized_alpha'
            if ti == 'generalized_alpha':
                return GeneralizedAlphaDynamics(
                    self.fem, dt=cfg.dt, dt_safety=cfg.dt_safety,
                    rho_inf=cfg.rho_inf, newton_tol=cfg.static_tol,
                    newton_max_iter=max(1, min(cfg.static_max_iter, 50)),
                    cg_tol=cfg.static_tol, cg_max_iter=cfg.static_max_iter,
                    differentiable=cfg.differentiable)
            return ExplicitDynamics(
                self.fem, dt=cfg.dt, dt_safety=cfg.dt_safety,
                differentiable=cfg.differentiable,
                damping_ratio_max=cfg.damping_ratio_max)
        elif st == 'static':
            static_backend = cfg.backend if cfg.backend in ('auto', 'scipy', 'cg') else 'scipy'
            return StaticSolver(
                self.fem, tol=cfg.static_tol,
                max_iter=cfg.static_max_iter,
                backend=static_backend)
        elif st == 'quasi_static':
            # 2026-04-29: repointed from SecantCGSolver to QuasiStaticSolver
            # (Newton-Raphson with autograd-enabled sparse-direct inner solve
            # for problems below sparse_dof_threshold; falls back to iterative
            # CG above). Preserves legacy semantics for problems > 200k dofs.
            return QuasiStaticSolver(
                self.fem, tol=cfg.static_tol,
                max_iter=cfg.static_max_iter,
                backend=cfg.backend,
                plasticity_operator=self.plasticity_operator,
                line_search=cfg.newton_line_search,
                line_search_max_steps=cfg.line_search_max_steps,
                line_search_min_alpha=cfg.line_search_min_alpha,
                line_search_c=cfg.line_search_c)
        elif st == 'quasi_static_legacy':
            # Old SecantCG path retained under a new name for any caller
            # that relied on its specific behaviour. Prefer 'quasi_static'.
            return SecantCGSolver(
                self.fem, tol=cfg.static_tol,
                max_iter=cfg.static_max_iter, check_every=50,
                use_multigrid=cfg.use_multigrid)
        elif st == 'lbfgs':
            return LBFGSSolver(
                self.fem, tol=cfg.static_tol,
                max_iter=cfg.static_max_iter)
        elif st == 'monolithic':
            return MonolithicSolver(
                self.fem, tol=cfg.static_tol,
                max_iter=cfg.static_max_iter)
        else:
            raise ValueError(f"Unknown solver_type: {st}")

    # ------------------------------------------------------------------ #
    # Modular step methods
    # ------------------------------------------------------------------ #

    def _adapt_timestep(self, d_prev: torch.Tensor):
        """Adjust dt by Δd-driven growth/cutback (explicit dynamics only).

        Implements the explicit dynamics timestep adaptation heuristic:
            d_max_step = max |d - d_prev|
            if d_max_step > dt_cutback_threshold:
                dt ← max(dt * 0.5, dt_floor)
            elif d_max_step < dt_growth_threshold:
                dt ← min(dt * adaptive_dt_grow_rate, dt_cfl_live)

        The growth bound is recomputed from the FEM operators every call
        (mass/stiffness can drift via degradation or via diff_E_field
        being installed mid-run), then multiplied by ``dt_safety`` so the
        adaptive bound matches the constant-dt initialization.
        """
        cfg = self.config
        d_max_step = (self.d - d_prev).abs().max().item()

        if d_max_step > cfg.dt_cutback_threshold:
            new_dt = max(self.dt * 0.5, cfg.dt_floor)
            if new_dt != self.dt:
                self.dt = new_dt
                self.mechanics.dt = new_dt
        elif d_max_step < cfg.dt_growth_threshold:
            # Live CFL bound — recompute_dt_cfl writes self.fem.dt_cfl in place.
            # Suppress the noisy print used by the diff_E_field path.
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                self.fem.recompute_dt_cfl()
            dt_cfl_live = self.fem.dt_cfl * cfg.dt_safety
            new_dt = min(self.dt * cfg.adaptive_dt_grow_rate, dt_cfl_live)
            if new_dt != self.dt:
                self.dt = new_dt
                self.mechanics.dt = new_dt

    def step_mechanics(self):
        """Advance displacement by one step."""
        bc_mask, bc_vals = self.bcs.get_masks_and_values()

        # Rotation-free rigid-connector MPCs (issue #154 / #165 / #171).
        # The static DirectSolver path consumes them through PR #164, the
        # explicit path through #165/#169, and the iterative-CG
        # SecantCGSolver path through #171 (this PR).
        rcs = self.bcs.get_active_rigid_connectors()

        if self.config.solver_type == 'explicit':
            self.u, self.v, self.a = self.mechanics.step(
                self.u, self.v, self.a, self.d, self.f_ext, bc_mask, bc_vals,
                rigid_connectors=rcs)
            self._mirror_mechanics_diagnostics()
            if (not self._last_mechanics_converged
                    and self.config.fail_on_mechanics_nonconvergence):
                raise RuntimeError(
                    "Dynamic mechanics solve did not converge. For "
                    "generalized_alpha, reduce dt/rho_inf tolerance or set "
                    "fail_on_mechanics_nonconvergence=False for diagnostics.")

        elif self.config.solver_type == 'static':
            self.u = self.mechanics.solve(bc_mask, bc_vals, u_init=self.u)
            self._mirror_mechanics_diagnostics()

        elif self.config.solver_type == 'quasi_static':
            # QuasiStaticSolver.solve signature: (d, f_ext, bc_mask, bc_vals, u_init)
            # Returns (u, converged, n_iter). Supports all energy splits since #114
            # (spectral/amor/star_convex use the secant tangent re-frozen each NR
            # step). 'quasi_static_legacy' kept for backwards compatibility.
            u_new, _conv, _nit = self.mechanics.solve(
                self.d, self.f_ext, bc_mask, bc_vals, u_init=self.u)
            self._mirror_mechanics_diagnostics()
            if not _conv and self.config.fail_on_mechanics_nonconvergence:
                raise RuntimeError(
                    "Quasi-static mechanics solve did not converge "
                    f"after {_nit} Newton iterations. Reduce load step size, "
                    "enable a more robust backend, or explicitly set "
                    "fail_on_mechanics_nonconvergence=False for diagnostic runs."
                )
            self.u = u_new

        elif self.config.solver_type == 'quasi_static_legacy':
            # SecantCGSolver path — supports all energy splits including
            # spectral. Rotation-free rigid_connector MPC is wired through
            # the q-space wrap (issue #171); when ``rcs`` is non-empty the
            # solver runs with multigrid forced off and Jacobi rebuilt
            # against ``diag(T^T K T)`` (see SecantCGSolver._solve_impl_mpc).
            self.u = self.mechanics.solve(
                self.u, self.d, bc_mask, bc_vals, f_ext=self.f_ext,
                rigid_connectors=(rcs or None))
            self._mirror_mechanics_diagnostics()

        elif self.config.solver_type == 'lbfgs':
            self.u, converged, n_iter = self.mechanics.solve(
                self.d, self.f_ext, bc_mask, bc_vals, u_init=self.u)
            self._mirror_mechanics_diagnostics()

    def step_mechanics_arc_length(self, *, lambda_prev: float,
                                  lambda_init: float, ds: float,
                                  alpha: float = 1.0,
                                  u_prev: torch.Tensor = None):
        """Advance QS mechanics with an augmented load-factor equation.

        ``step_mechanics`` solves equilibrium at the currently prescribed
        ``bcs.load_factor``.  This method instead treats the load factor as
        an unknown and enforces a spherical arc-length constraint relative to
        ``(u_prev, lambda_prev)``.  It is intended for displacement-controlled
        quasistatic fracture benchmarks whose Dirichlet values are unit
        patterns scaled by ``BoundaryConditions.load_factor``.
        """
        if self.config.solver_type != 'quasi_static':
            raise NotImplementedError(
                "Arc-length mechanics is currently implemented for "
                "solver_type='quasi_static' only.")
        if u_prev is None:
            u_prev = self.u.detach().clone()

        lf_saved = float(self.bcs.load_factor)
        try:
            self.bcs.load_factor = 1.0
            bc_mask, bc_unit_vals = self.bcs.get_masks_and_values()
            rcs = self.bcs.get_active_rigid_connectors()
            if rcs:
                raise RuntimeError(
                    "arc_length_solver='riks' is not yet supported with "
                    "rotation-free rigid_connector MPC. Use the controller "
                    "arc-length mode or remove rigid_connector constraints "
                    "until the reduced-coordinate augmented Riks block is "
                    "implemented.")
            f_ext_ref = self.f_ext
            if getattr(self.bcs, 'neumann_bcs', None):
                f_ext_ref = self.bcs.get_neumann_forces(self.mesh)
            self._last_arc_f_ext_ref = f_ext_ref.detach().clone()
            u_new, lam_new, conv, nit = (
                self.mechanics.solve_arc_length_dirichlet(
                    self.d, f_ext_ref, bc_mask, bc_unit_vals,
                    u_prev=u_prev, lambda_prev=lambda_prev,
                    lambda_init=lambda_init, ds=ds, alpha=alpha,
                    u_init=self.u, rigid_connectors=(rcs or None)))
            self.bcs.load_factor = float(lam_new)
            self.u = u_new
            self._mirror_mechanics_diagnostics()
            self._last_load_factor = float(lam_new)
            self._last_arc_f_ext_active = (
                float(lam_new) * self._last_arc_f_ext_ref)
            if not conv and self.config.fail_on_mechanics_nonconvergence:
                raise RuntimeError(
                    "Arc-length quasi-static mechanics solve did not "
                    f"converge after {nit} Newton iterations. Reduce "
                    "--arc_length_ds, increase --static_max_iter, or set "
                    "fail_on_mechanics_nonconvergence=False for diagnostics."
                )
            return float(lam_new), bool(conv), int(nit)
        except Exception:
            self.bcs.load_factor = lf_saved
            raise

    def step_full_arc_length(self, *, lambda_prev: float,
                             lambda_init: float, ds: float,
                             alpha: float = 1.0,
                             u_prev: torch.Tensor = None) -> torch.Tensor:
        """Staggered phase-field step with a Riks-style mechanics solve.

        The damage problem remains staggered exactly as in ``step_full``; the
        mechanics subproblem inside each stagger iteration is the augmented
        arc-length solve.  ``self.bcs.load_factor`` is updated to the converged
        load factor before returning.
        """
        if self.config.solver_type != 'quasi_static':
            raise NotImplementedError(
                "step_full_arc_length requires solver_type='quasi_static'.")
        if u_prev is None:
            u_prev = self.u.detach().clone()
        lambda_prev = float(lambda_prev)
        lambda_init = float(lambda_init)
        ds = float(ds)

        _lf_snapshot = self.bcs.load_factor
        base_tol = self.config.stagger_tol
        if self.config.adaptive_stagger_tol:
            max_d = self.d.max().item()
            tol = base_tol * (1.0 + 100.0 * (1.0 - min(max_d, 1.0)) ** 2)
        else:
            tol = base_tol
        max_stag = self.config.max_stagger
        min_stag = 2
        E_old = None
        if self.config.stagger_criterion == 'am_energy':
            strain_init = self.fem.compute_strain(self.u)
            psi_init = self.fem.compute_psi_plus(
                self.u, strain=strain_init, d=self.d)
            E_old = self.fem.compute_total_energy(
                self.u, self.d, strain=strain_init, psi_plus=psi_init)

        aa = None
        d_prev_step = self.d.clone()
        if self.config.damage_viscosity > 0.0:
            self.damage_solver.damage_viscosity_reference = d_prev_step.clone()
        if self.config.anderson_depth > 0:
            if not hasattr(self, '_aa') or self._aa is None:
                self._aa = _AndersonAccelerator(m=self.config.anderson_depth)
            else:
                self._aa.reset()
            aa = self._aa

        residual0 = None
        lam_guess = lambda_init
        for stag in range(max_stag):
            need_convergence = stag >= min_stag - 1
            d_old = self.d.clone() if (aa is not None or need_convergence) else self.d
            u_old = self.u.clone() if need_convergence else self.u

            lam_new, _conv, _nit = self.step_mechanics_arc_length(
                lambda_prev=lambda_prev, lambda_init=lam_guess,
                ds=ds, alpha=alpha, u_prev=u_prev)
            strain = self.fem.compute_strain(self.u)
            self._last_strain = strain
            psi = self.step_compute_driving_force(strain=strain)
            self.step_solve_damage(d_prev_step=d_prev_step)
            lam_guess = lam_new

            if aa is not None:
                self.d = aa.step(d_old, self.d, d_prev_step=d_prev_step)

            if stag >= min_stag - 1:
                crit = self.config.stagger_criterion
                if crit == 'residual':
                    bc_mask, _ = self.bcs.get_masks_and_values()
                    f_ext_active = getattr(
                        self, '_last_arc_f_ext_active', self.f_ext)
                    R_u = self.fem.internal_force(self.u, self.d) - f_ext_active
                    R_u[bc_mask] = 0.0
                    u_change = R_u.norm().item()
                    R_d = self.damage_solver.compute_residual(
                        self.H_elem, self.d)
                    d_change = R_d.norm().item()
                elif crit == 'linf':
                    d_change = (self.d - d_old).abs().max().item()
                    u_change = (self.u - u_old).abs().max().item()
                elif crit == 'am_energy':
                    E_new = self.fem.compute_total_energy(
                        self.u, self.d, strain=strain, psi_plus=psi)
                    if E_old is not None:
                        energy_change = (abs(E_new - E_old)
                                         / max(abs(E_old), 1e-30))
                        u_change = energy_change
                        d_change = energy_change
                    else:
                        u_change = float('inf')
                        d_change = float('inf')
                    E_old = E_new
                elif crit == 'absolute':
                    d_change = (self.d - d_old).norm().item()
                    u_change = (self.u - u_old).norm().item()
                else:
                    if self.config.stagger_norm == 'linf':
                        d_norm = self.d.abs().max().item() + 1e-30
                        u_norm = self.u.abs().max().item() + 1e-30
                        d_change = (self.d - d_old).abs().max().item() / d_norm
                        u_change = (self.u - u_old).abs().max().item() / u_norm
                    else:
                        d_norm = self.d.norm().item() + 1e-30
                        u_norm = self.u.norm().item() + 1e-30
                        d_change = (self.d - d_old).norm().item() / d_norm
                        u_change = (self.u - u_old).norm().item() / u_norm
                residual = float(max(u_change, d_change))
                if residual0 is None:
                    residual0 = residual
                self._update_stagger_residual_diagnostics(residual, residual0)
                if d_change < tol and u_change < tol:
                    break
        else:
            msg = (
                f"Arc-length stagger loop did not converge after {max_stag} "
                f"iterations (criterion='{self.config.stagger_criterion}', "
                f"tol={self.config.stagger_tol})."
            )
            if self.config.fail_on_stagger_nonconvergence:
                raise RuntimeError(
                    msg + " Reduce the continuation step, increase "
                    "--max_stagger, or set "
                    "fail_on_stagger_nonconvergence=False for diagnostics.")
            import warnings
            warnings.warn(
                msg + " Results may be non-equilibrium.",
                RuntimeWarning, stacklevel=2)

        if self.bcs.load_factor != getattr(self, '_last_load_factor', self.bcs.load_factor):
            import warnings
            warnings.warn(
                "bcs.load_factor changed unexpectedly during arc-length "
                "stagger iteration.", RuntimeWarning, stacklevel=2)
        del _lf_snapshot
        self._step_count += 1
        self._last_stagger_iter = stag + 1
        self._last_damage_load_factor = float(self.bcs.load_factor)
        return psi

    def _H_update(self, H_old: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Update history variable H = max(H_old, psi).

        Default: standard `torch.maximum` — exact but piecewise-constant gradient.

        If `config.softmax_H_beta` is set, uses smooth-max:
            H_new = H_old + softplus(β*(psi - H_old)) / β
        which is smooth everywhere and equals torch.maximum in the β → ∞ limit.
        Gives continuously-flowing gradients through the full dynamic loop,
        enabling stable L-BFGS/Adam for inverse problems. Cost: a small bias
        in forward (H_new is ~log(2)/β above max when psi ≈ H_old).

        Recommended β values:
          β = None (default) : exact max, use for forward runs
          β = 100            : inversion with minimal forward drift (~1%)
          β = 10             : inversion with smoother gradients (larger drift)
        """
        # Issue #360 — H_update_method dispatcher takes precedence. The
        # legacy ``softmax_H_beta`` short-circuit ONLY fires when method is
        # the default ``'hard_max'`` (preserves backward-compat for old
        # scripts that set softmax_H_beta without setting H_update_method).
        # When the user explicitly picks a method, the dispatcher wins.
        # Bug 2026-05-09: silent failure where both hard_max and softmax
        # sweep jobs ran as softplus(beta=10) because softmax_H_beta=10
        # was passed unconditionally by a batch script.
        method = self.config.H_update_method
        if method == 'hard_max':
            beta = self.config.softmax_H_beta
            if beta is not None and beta > 0:
                # Legacy softplus-based smooth max — kept for backward compat
                # with checkpoints / scripts that set ``softmax_H_beta``.
                diff = psi - H_old
                return H_old + torch.nn.functional.softplus(beta * diff) / beta
        kwargs = {}
        beta = self.config.softmax_H_beta
        if beta is not None and beta > 0 and method in ('softmax', 'log_smooth'):
            kwargs['beta'] = beta
        scale = self.config.H_update_scale
        if scale is not None and scale > 0 and method == 'custom_subgrad':
            kwargs['scale'] = scale
        from ..physics.h_update import dispatch as _h_dispatch  # local import: avoid cycle
        return _h_dispatch(method, H_old, psi, **kwargs)

    def step_compute_driving_force(self, strain=None) -> torch.Tensor:
        """Compute the crack-driving scalar D and update history variable H.

        The driving scalar is selected by ``material.driving_force`` (issue
        #248): ``'strain_energy'`` returns ψ⁺ (default, legacy behaviour);
        ``'principal_stress'`` returns ⟨σ₁⟩²/(2E). See
        ``FEMOperators.compute_driving_force`` for details.

        If material has sigma_ts (tensile strength), applies nucleation
        enhancement (Kumar et al. 2020, Molnár et al. 2024): scales the
        driving scalar so that the effective AT1 nucleation stress matches
        sigma_ts instead of the (usually higher) threshold 3Gc/(16l0).
        """
        if self.ductile_coupling is not None:
            psi = self.ductile_coupling.driving_force(
                self.u, state=self.plasticity_operator.state)
        else:
            psi = self.fem.compute_driving_force(
                self.u, strain=strain, d=self.d)

        # Nucleation enhancement for AT1 with tensile strength
        sigma_ts = getattr(self.material, 'sigma_ts', None)
        if sigma_ts is not None and sigma_ts > 0 and psi.max().item() > 0:
            # Target: nucleation at H_target = sigma_ts^2 / (2E)
            # AT1 threshold: H_crit = 3Gc / (16*l0)
            # Scale factor: c_e = H_crit / H_target
            H_target = sigma_ts ** 2 / (2.0 * self.material.E)
            H_crit = 3.0 * self.material.Gc / (16.0 * self.material.l0)
            c_e = H_crit / max(H_target, 1e-30)
            if c_e > 1.0:
                psi = psi * c_e

        self.H_elem = self._H_update(self.H_elem, psi)

        # H capping (Pantidis 2026) for stability
        if self.config.H_cap is not None:
            self.H_elem = torch.clamp(self.H_elem, max=self.config.H_cap)

        # Note: elem_to_node performs area-weighted averaging, which smooths
        # peak H values near the crack tip. Neural operators using get_H_nodal()
        # will see a smoother driving force than the element-level H_elem that
        # the damage solver actually uses. For NN training, consider using
        # H_elem directly with element-to-node projection at inference time.
        self.H_nodal = self.mesh.elem_to_node(self.H_elem)
        return psi

    def step_solve_damage(self, d_prev_step=None):
        """Solve AT2 for damage.

        If `self.diff_Gc` and/or `self.diff_l0` are set (0-d tensors,
        typically with requires_grad=True), they are forwarded to the
        damage solver, which engages the implicit-differentiation backward
        path (`_AdjointDamageSolveScalar`). This is the entry point for
        gradient-based inverse problems that need autograd through the
        full dynamic loop with respect to scalar material parameters.

        For SPATIAL Gc(x) recovery (high-dimensional k ~ n_elem inversion),
        set `self.diff_Gc_field` to an (E,) tensor with requires_grad=True.
        This routes through `_AdjointDamageSolveField` which produces the
        full per-element gradient in 2 CG solves, independent of k (vs the
        k+1 forwards that finite differences would need). Requires the
        solver to be built with `material.gamma_correction=True`.
        Mutually exclusive with `diff_Gc`.
        """
        Gc_field = getattr(self, 'diff_Gc_field', None)
        Gc = getattr(self, 'diff_Gc', None) if Gc_field is None else None
        l0 = getattr(self, 'diff_l0', None)
        # Allen-Cahn (gradient-flow) variant needs an explicit dt; forward
        # the current step from the mechanics integrator. Other models
        # ignore this attribute.
        if getattr(self.damage_solver, '_pf_model', None) == 'allencahn':
            self.damage_solver._ac_dt = float(self.dt) if self.dt is not None else None
        if self.config.damage_viscosity > 0.0:
            dt_damage = self._damage_regularization_dt()
            if dt_damage <= 0.0 or not torch.isfinite(
                    torch.tensor(dt_damage, dtype=self.dtype)):
                raise RuntimeError(
                    "damage_viscosity requires a positive damage pseudo-time "
                    "increment. For quasistatic runs, ensure bcs.load_factor "
                    "changes between accepted steps.")
            self.damage_solver.damage_dt = float(dt_damage)
        else:
            self.damage_solver.damage_dt = None
        pf_mask = pf_vals = None
        if getattr(self.bcs, 'pf_dirichlet_bcs', None):
            pf_mask, pf_vals = self.bcs.get_pf_dirichlet_mask_values()
        d_prev_arg = d_prev_step if d_prev_step is not None else self.d
        initial_guess = self.d if d_prev_step is not None else None
        self.d = self.damage_solver.solve(
            self.H_elem, d_prev_arg, Gc=Gc, l0=l0, Gc_field=Gc_field,
            pf_dirichlet_mask=pf_mask, pf_dirichlet_values=pf_vals,
            initial_guess=initial_guess)
        # Phase-field Dirichlet enforcement (issue #213): re-pin the
        # damage value at every step on nodes flagged with the
        # pf_dirichlet BC (matches COMSOL pre-crack convention).
        self._apply_pf_dirichlet()

    def _damage_regularization_dt(self) -> float:
        """Return the pseudo-time/load increment for damage viscosity."""
        if self.dt is not None:
            return float(self.dt)
        load_factor = float(getattr(self.bcs, 'load_factor', 0.0))
        prev = float(getattr(self, '_last_damage_load_factor', 0.0))
        dt = abs(load_factor - prev)
        if dt <= 0.0:
            dt = abs(load_factor)
        return dt

    def _apply_pf_dirichlet(self):
        """Enforce phase-field Dirichlet BCs on ``self.d`` (and ``self.d_prev``).

        For every ``PhaseFieldDirichletBC`` registered on ``self.bcs``,
        reassign ``self.d[idx] = value`` (post-solve, idempotent across
        all damage solver paths — ``post_clamp``, ``projected_cg``,
        ``direct``, ``allencahn``) and also pin ``self.d_prev`` so the
        next step's irreversibility ``d >= d_prev`` constraint is
        consistent. ``self.d_prev`` is only present in the explicit-
        dynamic adaptive-dt path; we update it only when it exists.
        """
        bcs = getattr(self, 'bcs', None)
        if bcs is None or not getattr(bcs, 'pf_dirichlet_bcs', None):
            return
        mask, vals = bcs.get_pf_dirichlet_mask_values()
        if not bool(mask.any()):
            return
        # Cast to the actual damage tensor's device/dtype.
        m = mask.to(self.d.device)
        v = vals.to(device=self.d.device, dtype=self.d.dtype)
        # Hard constraint: at pinned nodes, ``self.d`` is replaced by
        # the prescribed value; elsewhere it's untouched. Autograd
        # passes the upstream gradient through the unpinned branch.
        self.d = torch.where(m, v, self.d)
        d_prev = getattr(self, 'd_prev', None)
        if d_prev is not None:
            self.d_prev = torch.where(
                m.to(d_prev.device),
                v.to(device=d_prev.device, dtype=d_prev.dtype),
                d_prev)

    def advance_step(self):
        """Increment the step counter. Call after manual step_mechanics/step_solve_damage
        sequences that bypass step_full()."""
        self._step_count += 1
        if hasattr(self, 'bcs') and self.bcs is not None:
            self._last_damage_load_factor = float(getattr(self.bcs, 'load_factor', 0.0))

    def _step_full_explicit_fresh_d(self) -> torch.Tensor:
        """Explicit step with fresh d in the corrector (issue #299).

        Reorders the canonical "lagged" Verlet step
            predictor -> corrector(d_n) -> psi+ -> step_solve_damage
        into the PhaFiDyn / segregated ordering
            predictor -> psi+ -> step_solve_damage(d_{n+1}) -> corrector(d_{n+1})

        Subcycling: ``damage_every > 1`` means ``step_solve_damage`` is only
        called on the gated steps (matching the legacy gate). On non-gated
        steps the corrector still uses the most recent ``self.d`` — i.e., d
        is no more stale than under the legacy ordering, but on the steps
        where d IS solved the corrector benefits from the fresh value
        instead of the previous step's.

        Cost: one extra ``internal_force`` assembly per step inside the
        corrector (the predictor's was already implicit in the original
        ``step_mechanics`` -> ``mechanics.step``). Net wall-clock impact
        is dominated by the second internal_force, since the strain /
        psi+ work is identical to the legacy path. Audit estimated 3-5%.

        See ``SolverConfig.fresh_d_in_corrector`` for the public behavior
        contract.
        """
        # Adaptive-dt diagnostic d_prev capture (must be at top, before the
        # damage solve mutates self.d). Mirrors the legacy path.
        if self.config.adaptive_dt:
            self.d_prev = self.d.clone()
        else:
            self.d_prev = None

        # MPC trap (advisor #1): _step_mpc is a unified predictor+corrector
        # interleaved with theta DOF; cannot be split into predictor / damage
        # / corrector cleanly. Fail loudly rather than silently miscompute.
        rcs = self.bcs.get_active_rigid_connectors()
        if rcs:
            raise RuntimeError(
                "fresh_d_in_corrector=True is not compatible with "
                "rigid_connector MPC. The MPC predictor+corrector path "
                "(_step_mpc in mechanics_solver.py) interleaves the per-"
                "connector theta DOF with the master kinematics; it cannot "
                "be split around an in-step damage solve. Either disable "
                "rigid_connector for this run or set "
                "config.solver.fresh_d_in_corrector=False (default).")

        # ----- PREDICTOR -----
        bc_mask, bc_vals = self.bcs.get_masks_and_values()
        u_pred = self.mechanics.predictor(self.u, self.v, self.a,
                                          bc_mask, bc_vals)

        # ----- DRIVING FORCE on the predicted u -----
        strain_pred = self.fem.compute_strain(u_pred)
        self._last_strain = strain_pred

        # Pure-elastic short-circuit (mirrors the legacy enable_damage=False
        # branch). Skip H update + damage solve; corrector still uses self.d
        # (which is whatever it was at start-of-step — no fresh d to inject).
        if not self.config.enable_damage:
            psi = self.fem.compute_psi_plus(
                u_pred, strain=strain_pred, d=self.d)
            v_new, a_new = self.mechanics.corrector(
                u_pred, self.v, self.a, self.d, self.f_ext, bc_mask)
            self.u, self.v, self.a = u_pred, v_new, a_new
            self._step_count += 1
            self._last_stagger_iter = 0
            self._last_residual = float('nan')
            self._last_residual0 = float('nan')
            self._last_relative_residual = float('nan')
            self._last_dt_used = float(self.dt) if self.dt is not None else 0.0
            return psi

        # H = max(H, psi+(u_pred, d_old)). ``step_compute_driving_force``
        # forwards ``strain`` to ``compute_psi_plus``, which uses the
        # passed strain unconditionally; ``self.u`` is consulted only in
        # the fallback ``strain is None`` branch. Hence self.u staying at
        # u_n during the call is benign — the psi+ is evaluated on the
        # u_pred-derived strain we pass explicitly.
        psi = self.step_compute_driving_force(strain=strain_pred)

        # ----- DAMAGE SOLVE (subcycled by damage_every) -----
        if not hasattr(self, '_explicit_step_count'):
            self._explicit_step_count = 0
        self._explicit_step_count += 1

        if (self.config.damage_every <= 1 or
            self._explicit_step_count <= 5 or
            self._explicit_step_count % self.config.damage_every == 0):
            self.step_solve_damage()  # writes self.d in-place; pf_dirichlet
                                      # re-pinned inside step_solve_damage

        # ----- CORRECTOR with the (possibly just-updated) fresh d -----
        v_new, a_new = self.mechanics.corrector(
            u_pred, self.v, self.a, self.d, self.f_ext, bc_mask)

        # Commit u, v, a.
        self.u, self.v, self.a = u_pred, v_new, a_new

        # Adaptive-dt growth/cutback (legacy behaviour preserved).
        if self.config.adaptive_dt and self.d_prev is not None:
            self._adapt_timestep(self.d_prev)

        # Step bookkeeping (parallel to legacy path).
        self._step_count += 1
        self._last_stagger_iter = 1
        self._last_residual = float('nan')
        self._last_residual0 = float('nan')
        self._last_relative_residual = float('nan')
        self._last_dt_used = float(self.dt) if self.dt is not None else 0.0
        return psi

    def step_full(self) -> torch.Tensor:
        """Complete staggered step: mechanics + psi+ + H + damage.

        For explicit dynamics, performs a single pass (no stagger iteration).
        For quasi-static/lbfgs/static solvers, performs inner stagger
        iterations with dual residual convergence (checks both u AND d),
        matching the approach used by PhaseFieldX and Yu et al. (2025).

        Returns psi_plus (E,) for diagnostics.
        Caches strain in self._last_strain for reuse by run() output.
        Caches stagger iteration count in self._last_stagger_iter.
        """
        if self.config.solver_type == 'explicit':
            if self.config.fresh_d_in_corrector:
                return self._step_full_explicit_fresh_d()

            # Cache previous-step damage for adaptive-dt Δd diagnostic.
            # Stored on self so external code (tests, neural-op hooks) can
            # inspect the last increment without recomputing.
            if self.config.adaptive_dt:
                self.d_prev = self.d.clone()
            else:
                self.d_prev = None

            self.step_mechanics()
            strain = self.fem.compute_strain(self.u)
            self._last_strain = strain

            if not self.config.enable_damage:
                self._step_count += 1
                self._last_stagger_iter = 0
                self._last_residual = float('nan')
                self._last_residual0 = float('nan')
                self._last_relative_residual = float('nan')
                self._last_dt_used = float(self.dt) if self.dt is not None else 0.0
                return self.fem.compute_psi_plus(
                    self.u, strain=strain, d=self.d)

            psi = self.step_compute_driving_force(strain=strain)

            if not hasattr(self, '_explicit_step_count'):
                self._explicit_step_count = 0
            self._explicit_step_count += 1

            if (self.config.damage_every <= 1 or
                self._explicit_step_count <= 5 or
                self._explicit_step_count % self.config.damage_every == 0):
                self.step_solve_damage()

            if self.config.adaptive_dt and self.d_prev is not None:
                self._adapt_timestep(self.d_prev)

            self._step_count += 1
            self._last_stagger_iter = 1
            # Explicit dynamics has no stagger residual; the per-step residual
            # is implicit in the CFL stability bound. Use NaN sentinel so the
            # CSV consumer can distinguish "no residual" from a numeric value.
            self._last_residual = float('nan')
            self._last_residual0 = float('nan')
            self._last_relative_residual = float('nan')
            # ``self.dt`` may have been adapted inside _adapt_timestep above.
            self._last_dt_used = float(self.dt) if self.dt is not None else 0.0
            return psi

        if self.config.solver_type == 'monolithic':
            # Coupled (u, d) solve — no stagger loop needed
            bc_mask, bc_vals = self.bcs.get_masks_and_values()
            d_prev = self.d.clone()
            self.u, self.d, converged, n_iter = self.mechanics.solve(
                self.u, self.d, bc_mask, bc_vals,
                f_ext=self.f_ext, d_prev=d_prev)
            self._mirror_mechanics_diagnostics()
            if not converged and self.config.fail_on_mechanics_nonconvergence:
                raise RuntimeError(
                    "Monolithic mechanics/damage solve did not converge "
                    f"after {n_iter} iterations/evaluations. Set "
                    "fail_on_mechanics_nonconvergence=False for diagnostic "
                    "runs only."
                )
            strain = self.fem.compute_strain(self.u)
            self._last_strain = strain
            psi = self.step_compute_driving_force(strain=strain)
            self._step_count += 1
            self._last_stagger_iter = n_iter
            # Monolithic: use the inner solver diagnostics directly.
            self._last_residual = float(
                getattr(self.mechanics, 'last_residual', float('nan')))
            self._last_residual0 = float(
                getattr(self.mechanics, 'last_residual0', float('nan')))
            self._last_relative_residual = float(
                getattr(self.mechanics, 'last_relative_residual', float('nan')))
            self._last_dt_used = float(getattr(self.config, 'dt', 0.0) or 0.0)
            return psi

        # Snapshot load_factor at start of stagger — changing it mid-stagger
        # would make BCs inconsistent between stagger iterations.
        _lf_snapshot = self.bcs.load_factor

        # Stagger iterations for implicit solvers (dual residual check)
        base_tol = self.config.stagger_tol
        if self.config.adaptive_stagger_tol:
            # Scale tolerance: loose when no damage, tight near/at cracking.
            # tol = base * (1 + 100*(1-max_d)^2):  ~100x at d=0, 1x at d=1
            max_d = self.d.max().item()
            tol = base_tol * (1.0 + 100.0 * (1.0 - min(max_d, 1.0)) ** 2)
        else:
            tol = base_tol
        max_stag = self.config.max_stagger
        min_stag = 2
        # Initialize E_old for am_energy criterion to avoid inf on first iteration
        if self.config.stagger_criterion == 'am_energy':
            strain_init = self.fem.compute_strain(self.u)
            psi_init = self.fem.compute_psi_plus(
                self.u, strain=strain_init, d=self.d)
            E_old = self.fem.compute_total_energy(
                self.u, self.d, strain=strain_init, psi_plus=psi_init)
        else:
            E_old = None

        # Anderson Acceleration (Storvik et al. 2021, Walker & Ni 2011)
        # Applied to damage field only (u is solved exactly given d).
        # Reuse a single instance across steps; reset() clears history.
        aa = None
        d_prev_step = self.d.clone()  # irreversibility floor for this load step
        if self.config.damage_viscosity > 0.0:
            self.damage_solver.damage_viscosity_reference = d_prev_step.clone()
        if self.config.anderson_depth > 0:
            if not hasattr(self, '_aa') or self._aa is None:
                self._aa = _AndersonAccelerator(m=self.config.anderson_depth)
            else:
                self._aa.reset()
            aa = self._aa

        residual0 = None
        for stag in range(max_stag):
            # d_old is needed by AA (always) and convergence check (when stag >= min_stag-1).
            # u_old is only needed by the convergence check.
            need_convergence = stag >= min_stag - 1
            if aa is not None or need_convergence:
                d_old = self.d.clone()
            else:
                d_old = self.d
            if need_convergence:
                u_old = self.u.clone()
            else:
                u_old = self.u

            self.step_mechanics()
            strain = self.fem.compute_strain(self.u)
            self._last_strain = strain
            psi = self.step_compute_driving_force(strain=strain)
            self.step_solve_damage(d_prev_step=d_prev_step)

            # Anderson Acceleration: extrapolate d from history
            # d_prev_step enforces irreversibility (d >= d at start of load step)
            if aa is not None:
                self.d = aa.step(d_old, self.d, d_prev_step=d_prev_step)

            # Dual residual: check both u and d convergence
            # Use self.d (the actual iterate, including AA) to measure convergence
            if stag >= min_stag - 1:
                crit = self.config.stagger_criterion
                if crit == 'residual':
                    # PDE residual norms: ||R_u||, ||R_d||
                    bc_mask, _ = self.bcs.get_masks_and_values()
                    R_u = self.fem.internal_force(self.u, self.d) - self.f_ext
                    R_u[bc_mask] = 0.0  # zero out constrained DOFs
                    u_change = R_u.norm().item()
                    R_d = self.damage_solver.compute_residual(
                        self.H_elem, self.d)
                    d_change = R_d.norm().item()
                elif crit == 'linf':
                    d_change = (self.d - d_old).abs().max().item()
                    u_change = (self.u - u_old).abs().max().item()
                elif crit == 'am_energy':
                    E_new = self.fem.compute_total_energy(
                        self.u, self.d, strain=strain, psi_plus=psi)
                    if E_old is not None:
                        energy_change = (abs(E_new - E_old)
                                         / max(abs(E_old), 1e-30))
                        u_change = energy_change
                        d_change = energy_change
                    else:
                        u_change = float('inf')
                        d_change = float('inf')
                    E_old = E_new
                elif crit == 'absolute':
                    d_change = (self.d - d_old).norm().item()
                    u_change = (self.u - u_old).norm().item()
                else:  # 'relative' (default)
                    if self.config.stagger_norm == 'linf':
                        # Bleyer & Roux-Langlois (2017) max-norm criterion.
                        d_norm = self.d.abs().max().item() + 1e-30
                        u_norm = self.u.abs().max().item() + 1e-30
                        d_change = (self.d - d_old).abs().max().item() / d_norm
                        u_change = (self.u - u_old).abs().max().item() / u_norm
                    else:  # 'l2'
                        d_norm = self.d.norm().item() + 1e-30
                        u_norm = self.u.norm().item() + 1e-30
                        d_change = (self.d - d_old).norm().item() / d_norm
                        u_change = (self.u - u_old).norm().item() / u_norm
                # Telemetry (issue #300): record the converged residual so
                # the dataset CSV can carry per-step staggered convergence
                # quality. Pure observability — does not affect the solve.
                residual = float(max(u_change, d_change))
                if residual0 is None:
                    residual0 = residual
                self._update_stagger_residual_diagnostics(residual, residual0)
                if d_change < tol and u_change < tol:
                    break
        else:  # loop completed without break (did not converge)
            msg = (
                f"Stagger loop did not converge after {max_stag} iterations "
                f"(criterion='{self.config.stagger_criterion}', "
                f"tol={self.config.stagger_tol})."
            )
            if self.config.fail_on_stagger_nonconvergence:
                raise RuntimeError(
                    msg + " Reduce the load step, increase --max_stagger, "
                    "or set fail_on_stagger_nonconvergence=False for "
                    "diagnostics.")
            import warnings
            warnings.warn(
                msg + " Results may be non-equilibrium.",
                RuntimeWarning, stacklevel=2)

        # Guard: detect if load_factor was changed during stagger (would corrupt results)
        if self.bcs.load_factor != _lf_snapshot:
            import warnings
            warnings.warn(
                f"bcs.load_factor changed during stagger iteration "
                f"({_lf_snapshot} → {self.bcs.load_factor}). "
                f"Set load_factor BEFORE calling step_full(), not during.",
                RuntimeWarning, stacklevel=2)

        self._step_count += 1
        self._last_stagger_iter = stag + 1
        self._last_damage_load_factor = float(self.bcs.load_factor)
        # ``_last_residual`` was set inside the stagger loop. ``_last_dt_used``
        # is set by the driver (run_config.py) which owns the load-step dt /
        # explicit dt; we only seed it here for callers that bypass the driver.
        return psi

    # ------------------------------------------------------------------ #
    # Neural operator integration hooks
    # ------------------------------------------------------------------ #

    def install_diff_E_field(self, E_field: torch.Tensor):
        """Install a per-element Young's modulus field for spatial-E inversion.

        Mirrors the ``diff_Gc_field`` pattern used by the spatial Gc demo:
        the field is set as an attribute on the underlying FEMOperators,
        the CFL timestep is recomputed from the field maximum (otherwise
        the explicit dynamics blows up inside a stiff inclusion), and the
        mechanics solver is rebuilt so its internal ``self.dt`` picks up
        the new ``fem.dt_cfl``.

        Supported for ``isotropic``, ``amor``, ``spectral`` and
        ``star_convex`` splits. These branches route stress and psi+ through
        ``_resolve_lame`` so the per-element Lamé parameters remain in the
        autograd graph.

        Parameters
        ----------
        E_field : torch.Tensor, shape ``(n_elems,)``
            Per-element Young's modulus. Must be on the solver device and
            dtype. Pass ``requires_grad=True`` (or build it from
            requires_grad parameters via differentiable arithmetic) to
            enable autograd through the field. Pass ``None`` to revert to
            the bulk-material scalar path.
        """
        if E_field is not None:
            assert self.material.energy_split in ('isotropic', 'amor', 'spectral',
                                                  'star_convex'), (
                "install_diff_E_field supports energy_split in "
                "{'isotropic', 'amor', 'spectral', 'star_convex'}. "
                f"Got '{self.material.energy_split}'."
            )
            assert E_field.shape == (self.fem.mesh.n_elems,), (
                f"E_field shape {tuple(E_field.shape)} does not match "
                f"n_elems = {self.fem.mesh.n_elems}"
            )
        self.fem.diff_E_field = E_field
        dt_cfl_prev = self.fem.dt_cfl
        # Suppress the FEMOperators recompute log on subsequent calls (the
        # L-BFGS closure calls install_diff_E_field once per probe, and the
        # max-E barely moves when only (x_h, y_h) change with fixed alpha,
        # sigma — silencing here keeps the inversion log readable).
        self.fem.recompute_dt_cfl = self.fem.recompute_dt_cfl  # bind once
        # Always recompute (cheap), but only rebuild the mechanics solver
        # when dt_cfl moved by more than 0.1% since the last build. Position-
        # only inversions with fixed alpha/sigma keep max(E_field) essentially
        # constant across closure calls; rebuilding ExplicitDynamics 60+ times
        # per L-BFGS run for sub-percent dt drift is wasteful.
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            self.fem.recompute_dt_cfl()
        if self.config.solver_type == 'explicit':
            rel_dt_change = (abs(self.fem.dt_cfl - dt_cfl_prev)
                              / max(dt_cfl_prev, 1e-30))
            if rel_dt_change > 1e-3 or not hasattr(self, '_E_field_first_install'):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.mechanics = self._build_mechanics_solver()
                self.dt = self.mechanics.dt
                self._E_field_first_install = True

    def set_damage(self, d_new: torch.Tensor):
        """Inject external damage field (e.g., from GINO)."""
        self.d = d_new.clone()

    def get_H_nodal(self) -> torch.Tensor:
        """Get current history variable at nodes (for neural operator input)."""
        return self.H_nodal

    def get_state(self) -> dict:
        """Get full solver state as a dict (for checkpointing/rollback).

        Hybrid solver-DL rollouts use this state for path-dependent rollback.
        Keep it broader than the primary variables: rejected learned proposals
        must also restore history diagnostics and installed heterogeneous
        material fields so the trusted correction does not inherit stale
        internal state.
        """
        state = {
            'u': self.u.clone(),
            'v': self.v.clone(),
            'a': self.a.clone(),
            'd': self.d.clone(),
            'H_elem': self.H_elem.clone(),
            'H_nodal': self.H_nodal.clone(),
            'f_ext': self.f_ext.clone(),
            'step': self._step_count,
            'dt': None if self.dt is None else float(self.dt),
            'load_factor': float(getattr(self.bcs, 'load_factor', 1.0)),
            '_explicit_step_count': int(getattr(self, '_explicit_step_count', 0)),
            '_last_stagger_iter': int(getattr(self, '_last_stagger_iter', 0)),
            '_last_residual': float(getattr(self, '_last_residual', float('nan'))),
            '_last_residual0': float(getattr(self, '_last_residual0', float('nan'))),
            '_last_relative_residual': float(getattr(
                self, '_last_relative_residual', float('nan'))),
            '_last_mechanics_residual': float(getattr(
                self, '_last_mechanics_residual', float('nan'))),
            '_last_mechanics_residual0': float(getattr(
                self, '_last_mechanics_residual0', float('nan'))),
            '_last_mechanics_relative_residual': float(getattr(
                self, '_last_mechanics_relative_residual', float('nan'))),
            '_last_mechanics_converged': bool(getattr(
                self, '_last_mechanics_converged', True)),
            '_last_mechanics_iter': int(getattr(self, '_last_mechanics_iter', 0)),
            '_last_dt_used': float(getattr(self, '_last_dt_used', 0.0)),
        }
        def _clone_state_value(value):
            if isinstance(value, torch.Tensor):
                return value.clone()
            if isinstance(value, tuple):
                return tuple(_clone_state_value(v) for v in value)
            if isinstance(value, list):
                return [_clone_state_value(v) for v in value]
            return value

        if hasattr(self, '_last_strain'):
            state['_last_strain'] = _clone_state_value(self._last_strain)
        if getattr(self, 'd_prev', None) is not None:
            state['d_prev'] = self.d_prev.clone()
        if self.plasticity_operator is not None:
            pstate = self.plasticity_operator.state
            state['plasticity_state'] = {
                'strain': pstate.strain.clone(),
                'stress': pstate.stress.clone(),
                'plastic_strain': pstate.plastic_strain.clone(),
                'eps_p_eq': pstate.eps_p_eq.clone(),
                'plastic_work_density': pstate.plastic_work_density.clone(),
            }
        if hasattr(self, 'diff_Gc_field'):
            state['diff_Gc_field'] = self.diff_Gc_field.clone()
        E_field = getattr(self.fem, 'diff_E_field', None)
        if E_field is not None:
            state['diff_E_field'] = E_field.clone()
        return state

    def set_state(self, state: dict):
        """Restore solver state from a dict."""
        self.u = state['u'].to(self.device)
        self.v = state['v'].to(self.device)
        self.a = state['a'].to(self.device)
        self.d = state['d'].to(self.device)
        self.H_elem = state['H_elem'].to(self.device)
        self.H_nodal = state['H_nodal'].to(self.device)
        if 'f_ext' in state:
            self.f_ext = state['f_ext'].to(self.device)
        self._step_count = state.get('step', 0)
        dt_state = state.get('dt', None)
        if 'load_factor' in state and hasattr(self, 'bcs'):
            self.bcs.load_factor = float(state['load_factor'])
        self._explicit_step_count = int(state.get('_explicit_step_count', 0))
        self._last_stagger_iter = int(state.get('_last_stagger_iter', 0))
        def _restore_state_value(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.device)
            if isinstance(value, tuple):
                return tuple(_restore_state_value(v) for v in value)
            if isinstance(value, list):
                return [_restore_state_value(v) for v in value]
            return value

        if '_last_strain' in state:
            self._last_strain = _restore_state_value(state['_last_strain'])
        elif hasattr(self, '_last_strain'):
            delattr(self, '_last_strain')
        self._last_residual = float(state.get('_last_residual', float('nan')))
        self._last_residual0 = float(state.get('_last_residual0', float('nan')))
        self._last_relative_residual = float(state.get(
            '_last_relative_residual', float('nan')))
        self._last_mechanics_residual = float(state.get(
            '_last_mechanics_residual', float('nan')))
        self._last_mechanics_residual0 = float(state.get(
            '_last_mechanics_residual0', float('nan')))
        self._last_mechanics_relative_residual = float(state.get(
            '_last_mechanics_relative_residual', float('nan')))
        self._last_mechanics_converged = bool(state.get(
            '_last_mechanics_converged', True))
        self._last_mechanics_iter = int(state.get('_last_mechanics_iter', 0))
        self._last_dt_used = float(state.get('_last_dt_used', 0.0))
        if 'd_prev' in state:
            self.d_prev = state['d_prev'].to(self.device)
        elif hasattr(self, 'd_prev'):
            self.d_prev = None
        if self.plasticity_operator is not None and 'plasticity_state' in state:
            from ..plasticity import MeshJ2State
            ps = state['plasticity_state']
            restored = MeshJ2State(
                strain=ps['strain'].to(self.device),
                stress=ps['stress'].to(self.device),
                plastic_strain=ps['plastic_strain'].to(self.device),
                eps_p_eq=ps['eps_p_eq'].to(self.device),
                plastic_work_density=ps['plastic_work_density'].to(self.device),
            )
            self.plasticity_operator.state = restored
            self.plasticity_operator._trial_state = None
        if 'diff_Gc_field' in state:
            self.diff_Gc_field = state['diff_Gc_field'].to(self.device)
        elif hasattr(self, 'diff_Gc_field'):
            delattr(self, 'diff_Gc_field')
        if 'diff_E_field' in state:
            self.install_diff_E_field(state['diff_E_field'].to(self.device))
        elif getattr(self.fem, 'diff_E_field', None) is not None:
            self.install_diff_E_field(None)
        if 'dt' in state:
            self.dt = dt_state
            if hasattr(self, 'mechanics') and self.dt is not None:
                self.mechanics.dt = self.dt

    def restore_state(self, state: dict):
        """Apply a state dict (from ``io_utils.load_state_from_h5`` or
        ``get_state``) to the solver tensors.

        Thin wrapper around ``set_state`` that also (a) tolerates the H5
        snapshot schema, which uses ``H`` as an alias for ``H_elem`` and
        does not always carry ``v`` / ``a``, (b) casts to the solver's
        ``device`` + ``dtype``, and (c) recomputes ``H_nodal`` from
        ``H_elem`` via ``mesh.elem_to_node`` so the two stay consistent
        regardless of snapshot precision (snapshots are written float32
        by default; the in-memory state is float64 on CPU).
        """
        def _cast(t):
            if t is None:
                return None
            return t.to(device=self.device, dtype=self.dtype)

        H_elem = state.get('H_elem')
        if H_elem is None:
            H_elem = state.get('H')
        if H_elem is None:
            raise KeyError("state dict missing 'H_elem' / 'H'")

        self.u = _cast(state['u'])
        self.d = _cast(state['d'])
        self.H_elem = _cast(H_elem)
        self.H_nodal = self.mesh.elem_to_node(self.H_elem)

        v = state.get('v')
        a = state.get('a')
        self.v = _cast(v) if v is not None else torch.zeros_like(self.u)
        self.a = _cast(a) if a is not None else torch.zeros_like(self.u)

        self._step_count = int(state.get('step', 0))

    def save_checkpoint(self, path: str):
        """Save solver state to disk for restart.

        Usage::
            solver.save_checkpoint('checkpoint_step_500.pt')
            # later:
            solver.load_checkpoint('checkpoint_step_500.pt')
        """
        torch.save(self.get_state(), path)

    def load_checkpoint(self, path: str):
        """Load solver state from disk.

        Uses weights_only=True for safety (no arbitrary code execution).
        Only loads tensor data saved by save_checkpoint().
        """
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.set_state(state)

    # ------------------------------------------------------------------ #
    # Pre-strain (initial equilibrium)
    # ------------------------------------------------------------------ #

    def pre_strain(self, coupled: bool = False,
                   max_alt_iter: int = 20, alt_tol: float = 1e-4):
        """Initial static equilibrium.

        Two modes:
        - ``coupled=False`` (default, matches Akantu ``solid.solveStep('static')``):
          solves elastic static equilibrium with d=0, seeds H=psi+(u).
        - ``coupled=True`` (matches Bleyer 2017 alternating minimization for
          PMMA branching IC): alternates static mechanics and damage solves
          until d converges. Needed when the pre-strain amplitude is near
          the branching threshold and the initial d=0 assumption delays
          second-lobe nucleation during dynamics.

        Parameters
        ----------
        coupled : bool
            If True, run alternating minimization until damage converges.
        max_alt_iter : int
            Max alternations when coupled=True.
        alt_tol : float
            Convergence tolerance on ||d_new - d_old||_inf (coupled mode).
        """
        if self.d.max().item() > 1e-10:
            import warnings
            warnings.warn("pre_strain() called with existing damage (max(d)={:.4e}). "
                          "The elastic pre-strain solve ignores damage, which may reset "
                          "H_elem inconsistently.".format(self.d.max().item()), RuntimeWarning, stacklevel=2)
        bc_mask, bc_vals = self.bcs.get_masks_and_values()
        static_backend = (self.config.backend if self.config.backend in ('auto', 'scipy', 'cg')
                          else 'scipy')
        static = StaticSolver(self.fem, tol=self.config.static_tol,
                              max_iter=self.config.static_max_iter,
                              backend=static_backend)

        if not coupled:
            # --- Legacy: elastic-only, d=0 ---
            print("[StaggeredSolver] Pre-strain: elastic static equilibrium (d=0)...",
                  flush=True)
            self.u = static.solve(bc_mask, bc_vals)
            psi = self.fem.compute_driving_force(self.u, d=self.d)
            self.H_elem = psi.clone()
            self.H_nodal = self.mesh.elem_to_node(self.H_elem)
        else:
            # --- Coupled (Bleyer 2017): alternating u-d minimization ---
            # First u-solve with d=0 (linear, fast via StaticSolver)
            print(f"[StaggeredSolver] Pre-strain: coupled u-d alternating "
                  f"minimization (max {max_alt_iter} iters, tol {alt_tol:.1e})...",
                  flush=True)
            self.u = static.solve(bc_mask, bc_vals)
            psi = self.fem.compute_driving_force(self.u, d=self.d)
            self.H_elem = psi.clone()
            self.H_nodal = self.mesh.elem_to_node(self.H_elem)
            # Damage-aware static solver: SecantCGSolver handles amor/spectral
            # splits (QuasiStaticSolver rejects them per mechanics_solver:329).
            qs = SecantCGSolver(
                self.fem,
                tol=self.config.static_tol,
                max_iter=self.config.static_max_iter,
                max_newton=5,
                use_multigrid=False)  # match production PMMA run (AT1 + Jacobi)
            f_ext = torch.zeros_like(self.u)
            d_prev_inner = self.d.clone()
            # Damage driver expects H_elem (element-level) when damage solver
            # was built with nodal_H=False (the default), H_nodal otherwise.
            use_nodal_H = getattr(self.damage_solver, 'nodal_H', False)
            for k in range(max_alt_iter):
                # (i) Damage solve with current H
                H_input = self.H_nodal if use_nodal_H else self.H_elem
                self.d = self.damage_solver.solve(H_input, d_prev_inner)
                # Re-pin pf_dirichlet nodes (issue #213).
                self._apply_pf_dirichlet()
                # (ii) Re-solve static mechanics with updated damage field
                self.u = qs.solve(self.u, self.d, bc_mask, bc_vals,
                                   f_ext=f_ext)
                # (iii) History update (monotone max)
                psi = self.fem.compute_driving_force(self.u, d=self.d)
                self.H_elem = self._H_update(self.H_elem, psi)
                self.H_nodal = self.mesh.elem_to_node(self.H_elem)
                delta = (self.d - d_prev_inner).abs().max().item()
                print(f"  [pre_strain alt iter {k+1}] max|Delta d| = "
                      f"{delta:.3e}, max(d) = {self.d.max():.4f}", flush=True)
                if delta < alt_tol:
                    print(f"  [pre_strain] converged after {k+1} alternations",
                          flush=True)
                    break
                d_prev_inner = self.d.clone()
            else:
                print(f"  [pre_strain] WARNING: max alternations reached "
                      f"({max_alt_iter}) without convergence", flush=True)

        self.v = torch.zeros_like(self.u)
        self.a = torch.zeros_like(self.u)
        print(f"[StaggeredSolver] Pre-strain done: max(u_y)={self.u[:, 1].max():.6e}, "
              f"max(psi+)={psi.max():.6e}, max(d)={self.d.max():.4f}", flush=True)
        return psi

    # ------------------------------------------------------------------ #
    # Adaptive mesh refinement
    # ------------------------------------------------------------------ #

    def _try_adaptive_refine(self, psi, step, verbose):
        """Check refinement indicator and refine if elements are marked.

        After refinement, all state fields (u, v, a, d, H_elem, H_nodal,
        f_ext) are interpolated onto the new mesh, and the FEM operators,
        damage solver, and mechanics solver are rebuilt.

        Parameters
        ----------
        psi : (E,) current psi_plus (will be re-interpolated)
        step : int, current step index (for logging)
        verbose : bool

        Returns
        -------
        psi : (E_new,) psi_plus on the (possibly new) mesh
        """
        from .adaptive import (compute_refinement_indicator, refine_mesh,
                               interpolate_field, interpolate_elem_field)

        cfg = self.config
        marked = compute_refinement_indicator(
            self.mesh, self.d,
            grad_d_threshold=cfg.refine_grad_d_threshold,
            d_threshold=cfg.refine_d_threshold)

        if not marked.any():
            return psi

        n_marked = int(marked.sum().item())
        old_n = self.mesh.n_nodes
        old_e = self.mesh.n_elems

        new_mesh, parent_map, child_map = refine_mesh(self.mesh, marked)

        # Interpolate nodal fields
        self.d = interpolate_field(self.mesh, new_mesh, self.d, parent_map)
        self.u = interpolate_field(self.mesh, new_mesh, self.u, parent_map)
        self.v = interpolate_field(self.mesh, new_mesh, self.v, parent_map)
        self.a = interpolate_field(self.mesh, new_mesh, self.a, parent_map)
        self.H_nodal = interpolate_field(
            self.mesh, new_mesh, self.H_nodal, parent_map)
        self.f_ext = interpolate_field(
            self.mesh, new_mesh, self.f_ext, parent_map)

        # Interpolate element fields
        self.H_elem = interpolate_elem_field(
            self.mesh, new_mesh, self.H_elem, child_map)
        psi = interpolate_elem_field(self.mesh, new_mesh, psi, child_map)

        # Swap mesh and rebuild operators / sub-solvers
        self.mesh = new_mesh
        self.fem = FEMOperators(new_mesh, self.material, ctx=self.ctx)
        amr_bounds = cfg.bounds_method
        if (getattr(self.material, 'pf_model', 'AT2') == 'AT1'
                and amr_bounds == 'post_clamp'):
            amr_bounds = 'projected_cg'
        self.damage_solver = PhaseFieldDamageSolver(
            self.fem, tol=cfg.damage_tol,
            max_iter=cfg.damage_max_iter, ctx=self.ctx,
            use_multigrid=cfg.use_multigrid,
            bounds_method=amr_bounds,
            preconditioner=cfg.preconditioner)

        self.mechanics = self._build_mechanics_solver()
        if cfg.solver_type == 'explicit':
            self.dt = self.mechanics.dt

        if verbose:
            print(f"  [AMR] Step {step}: refined {n_marked} elems, "
                  f"{old_n}->{new_mesh.n_nodes} nodes, "
                  f"{old_e}->{new_mesh.n_elems} elems",
                  flush=True)

        return psi

    # ------------------------------------------------------------------ #
    # Full simulation run
    # ------------------------------------------------------------------ #

    def run(self, num_steps: int = None, output_dir: str = None,
            h5_path: str = None, verbose: bool = True):
        """Run the complete simulation.

        Parameters
        ----------
        num_steps : int or None
            Override config.num_steps.
        output_dir : str or None
            Directory for VTU output.
        h5_path : str or None
            Path for H5 training-data output.
        verbose : bool
            Print progress to console.

        Returns
        -------
        history : list of dict with per-step diagnostics.
        """
        if num_steps is None:
            num_steps = self.config.num_steps
        cfg = self.config

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Pre-strain
        psi_init = self.pre_strain()
        if verbose:
            print(f"Pre-strain: max(u_y)={self.u[:, 1].max():.6e}, "
                  f"max(psi+)={psi_init.max():.6e}")

        # H5 output
        h5f = None
        if h5_path:
            h5f = init_h5(h5_path, self.mesh, self.material)

        # CSV history
        csv_hist = None
        if output_dir:
            csv_hist = CSVHistory(os.path.join(output_dir, 'history.csv'))

        if verbose:
            dt_str = f"dt={self.dt:.6e}" if self.dt else "N/A"
            solver_label = cfg.solver_type
            if cfg.solver_type == 'explicit':
                ti = cfg.time_integrator
                if ti in ('verlet', 'newmark'):
                    ti = 'central_difference'
                elif ti == 'gen_alpha':
                    ti = 'generalized_alpha'
                solver_label = f"dynamic/{ti}"
            print(f"\nRunning {num_steps} steps ({solver_label}, {dt_str})")

        history = []
        H_prev = 0.0
        d_prev_max = 0.0

        try:
            for step in range(num_steps):
                t0 = time.time()
                psi = self.step_full()
                elapsed = (time.time() - t0) * 1000

                # --- Adaptive mesh refinement ---
                if (cfg.adaptive_refine
                        and step % cfg.refine_every == 0):
                    psi = self._try_adaptive_refine(psi, step, verbose)

                max_d = self.d.max().item()
                max_H = self.H_nodal.max().item()
                max_psi = psi.max().item()
                delta_H = max_H - H_prev
                delta_d = max_d - d_prev_max

                stag_iter = getattr(self, '_last_stagger_iter', 1)
                cg_iters_d = getattr(self.damage_solver, 'last_iter', 0)
                cg_iters_u = getattr(self.mechanics, 'last_iter', 0)

                # Crack tip tracking (#76) — skip in elastic phase
                if max_d > 0.01:
                    tip_x, tip_y = self.fem.compute_crack_tip_position(self.d)
                    crack_len = self.fem.compute_crack_length(self.d)
                else:
                    tip_x, tip_y, crack_len = 0.0, 0.0, 0.0

                record = {
                    'step': step, 'max_d': max_d, 'max_H': max_H,
                    'max_psi': max_psi, 'delta_H': delta_H,
                    'delta_d': delta_d, 'stagger_iter': stag_iter,
                    'cg_iters_d': cg_iters_d, 'cg_iters_u': cg_iters_u,
                    'crack_tip_x': tip_x, 'crack_tip_y': tip_y,
                    'crack_length': crack_len,
                    'elapsed_ms': elapsed,
                }
                history.append(record)

                if verbose and (step % cfg.print_every == 0 or step < 5):
                    stag_str = (f", stag={stag_iter}"
                                if cfg.solver_type != 'explicit' else "")
                    cg_str = (f", CG(d={cg_iters_d},u={cg_iters_u})"
                              if cg_iters_d or cg_iters_u else "")
                    print(f"  Step {step:4d}: max(d)={max_d:.6f}, "
                          f"max(H)={max_H:.2f}, max(psi+)={max_psi:.2e}, "
                          f"Dd={delta_d:.6e}{stag_str}{cg_str} ({elapsed:.1f}ms)")

                if csv_hist:
                    csv_hist.write_row(step, max_H, max_psi, max_d,
                                       delta_H, delta_d)

                # VTU / H5 output (reuse strain cached by step_full)
                need_vtu = (output_dir and cfg.dump_every > 0
                            and step % cfg.dump_every == 0)
                need_h5 = (h5f and cfg.h5_every > 0
                           and step % cfg.h5_every == 0)
                if need_vtu or need_h5:
                    exx, eyy, gxy = self._last_strain
                if need_vtu:
                    _ext = '.pv' if cfg.viz_format == 'pv' else '.vtu'
                    write_visualization(
                        os.path.join(output_dir, f'step_{step:04d}{_ext}'),
                        self.mesh,
                        point_data={'displacement': self.u, 'damage': self.d,
                                    'H': self.H_nodal},
                        cell_data={'psi_plus': psi, 'H_elem': self.H_elem},
                        format=cfg.viz_format)
                if need_h5:
                    write_h5_snapshot(h5f, step, self.mesh, self.u, self.d,
                                      psi, self.H_elem, exx, eyy, gxy)

                H_prev = max_H
                d_prev_max = max_d
        finally:
            if h5f:
                h5f.close()
            if csv_hist:
                csv_hist.close()

        if verbose:
            print("Done.")
        self.steps = int(num_steps)
        return history


class AIBridge:
    """Residual monitor for neural operator + FEM solver-in-the-loop.

    Monitors the AT2 residual of neural operator damage predictions.
    When ||R_d|| exceeds a threshold, falls back to FEM solver for
    that step, ensuring physical consistency.

    Parameters
    ----------
    solver : StaggeredSolver
    residual_threshold : float
        Max acceptable ||R_d||. Default 1e-3.
    """

    def __init__(self, solver, residual_threshold=1e-3):
        self.solver = solver
        self.threshold = residual_threshold
        self.nn_steps = 0
        self.fallback_steps = 0

    def step_with_nn(self, d_nn):
        """Accept neural operator damage prediction, validate, maybe correct.

        Parameters
        ----------
        d_nn : (N,) damage prediction from neural operator

        Returns
        -------
        d_final : (N,) validated damage field
        used_solver : bool — True if FEM solver was needed
        """
        # Compute residual of NN prediction
        R = self.solver.damage_solver.compute_residual(
            self.solver.H_elem, d_nn)
        r_norm = R.norm().item()

        if r_norm < self.threshold:
            # NN prediction is good — accept it
            self.solver.d = d_nn.clone()
            self.nn_steps += 1
            return d_nn, False
        else:
            # NN prediction violates physics — fall back to FEM
            self.solver.d = self.solver.damage_solver.solve(
                self.solver.H_elem, self.solver.d)
            self.fallback_steps += 1
            return self.solver.d, True

    def stats(self):
        total = self.nn_steps + self.fallback_steps
        if total == 0:
            return "AIBridge: no steps taken"
        pct = self.nn_steps / total * 100
        return (f"AIBridge: {total} steps, "
                f"{self.nn_steps} NN ({pct:.0f}%), "
                f"{self.fallback_steps} fallback ({100-pct:.0f}%)")
