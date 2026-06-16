"""
Mechanics solvers for the displacement sub-problem.

Provides five solvers:
  - SecantCGSolver:    [PRIMARY] Linearized secant CG — used by all quasi-static
                       benchmarks (SENT, SENS, TPB, L-panel). Handles all energy
                       splits correctly via frozen secant linearization.
  - ExplicitDynamics:  [ACTIVE]  Velocity-Verlet (Newmark-Beta central difference)
                       — used by explicit benchmarks. O(N) per step, CFL-limited.
  - StaticSolver:      [INTERNAL] CG linear solve for pre-strain initialization (d=0).
  - QuasiStaticSolver: [AVAILABLE] Newton-Raphson + CG. Superseded by SecantCG for
                       spectral/amor splits (CG conjugacy breaks). Retained for
                       isotropic problems or custom Newton convergence monitoring.
  - LBFGSSolver:       [AVAILABLE] L-BFGS energy minimization. Gradient-only
                       (no matvec). Retained for problems where tangent is unavailable.

GPU strategy:
  - All solvers run on the same device as the mesh (CUDA/MPS/CPU)
  - AMP is disabled inside CG/L-BFGS iterations (accuracy-critical)
  - torch.compile can be applied to the CG matvec on CUDA
"""

import torch
import warnings
import math
from ..core.fem_operators import FEMOperators
from ..utils.device import device_supports_float64
from .time_integrators import gen_alpha_params


class LBFGSPreconditioner:
    """L-BFGS low-rank inverse Hessian preconditioner for CG.

    Stores m (s, y) pairs from previous Newton/stagger steps and applies
    the two-loop recursion as a preconditioner. Starts as identity when
    history is empty and builds up over load steps.
    """

    def __init__(self, m=5):
        self.m = m
        self.s_history = []  # s_k = x_{k+1} - x_k
        self.y_history = []  # y_k = grad_{k+1} - grad_k
        self.rho_history = []

    def update(self, s, y):
        """Store a new (s, y) pair. s = delta_u (flattened), y = delta_grad (flattened)."""
        sy = torch.dot(s, y)
        if sy > 1e-30:  # skip if not positive definite
            self.s_history.append(s)
            self.y_history.append(y)
            self.rho_history.append(1.0 / sy)
            if len(self.s_history) > self.m:
                self.s_history.pop(0)
                self.y_history.pop(0)
                self.rho_history.pop(0)

    def apply(self, r):
        """Apply L-BFGS two-loop recursion: z = H_k @ r."""
        q = r.clone()
        k = len(self.s_history)
        if k == 0:
            return q  # no history, identity preconditioner

        alphas = []
        # Forward loop
        for i in range(k - 1, -1, -1):
            alpha = self.rho_history[i] * torch.dot(self.s_history[i], q)
            q.sub_(alpha * self.y_history[i])
            alphas.append(alpha)
        alphas.reverse()

        # Initial Hessian: H0 = (s^T y / y^T y) * I
        s_last = self.s_history[-1]
        y_last = self.y_history[-1]
        gamma = torch.dot(s_last, y_last) / (torch.dot(y_last, y_last) + 1e-30)
        z = gamma * q

        # Backward loop
        for i in range(k):
            beta = self.rho_history[i] * torch.dot(self.y_history[i], z)
            z.add_((alphas[i] - beta) * self.s_history[i])

        return z


class ExplicitDynamics:
    """[ACTIVE] Velocity-Verlet scheme (Newmark-Beta with alpha=0, beta=0, gamma=1/2).

    Used by: miehe_sent_explicit, datagen_half.
    Equivalent to: explicit Newmark (beta=0, gamma=1/2), central difference.

    Use only when inertial effects are physically relevant (impact, blast,
    wave-driven fracture). For quasi-static benchmarks, use SecantCGSolver.

    Matches Akantu's CentralDifference:
        Predictor:  u_{n+1} = u_n + dt*v_n + 0.5*dt^2*a_n
        Corrector:  a_{n+1} = M^{-1}*(f_ext - f_int(u_{n+1}, d))
                    v_{n+1} = v_n + 0.5*dt*(a_n + a_{n+1})

    Parameters
    ----------
    fem : FEMOperators
    dt : float or None
        Timestep. If None, uses CFL timestep from FEM operators.
    dt_safety : float
        CFL safety factor (default 1.0, matching Akantu).
    """

    def __init__(self, fem: FEMOperators, dt: float = None,
                 dt_safety: float = 1.0, differentiable: bool = False,
                 damping_ratio_max: float = 0.0):
        """
        Parameters
        ----------
        damping_ratio_max : float, default 0.0
            Kelvin-Voigt stiffness-proportional damping ratio AT THE
            HIGHEST MESH MODE (omega_max = 2 / fem.dt_cfl). Sets the
            coefficient beta in the added damping force
                f_damp = beta * K(d) @ v_n,
            reused via fem.internal_force(v_n, d). beta = 2*zeta/omega_max.
            This mirrors the high-frequency damping introduced by
            Borden 2012's generalized-alpha scheme with rho_inf=0.5,
            which is essential to suppress spurious microbranches in
            the dynamic branching benchmark. Zero by default so
            existing benchmarks reproduce bit-exact.
            The CFL limit tightens by (sqrt(1+zeta^2) - zeta), a ~5%
            reduction at zeta=0.1 and ~10% at zeta=0.2.
        """
        self.fem = fem
        self.differentiable = differentiable
        if dt_safety > 1.0:
            import warnings
            warnings.warn(
                f"dt_safety={dt_safety} > 1.0 exceeds CFL limit — "
                f"explicit dynamics may be unstable", stacklevel=2)
        self.damping_ratio_max = float(damping_ratio_max)
        # beta = 2 * zeta / omega_max; omega_max = 2 / dt_cfl_undamped.
        # Hence beta = zeta * dt_cfl_undamped.
        self._damping_beta = self.damping_ratio_max * fem.dt_cfl
        # Damped CFL shrinks by (sqrt(1+zeta^2) - zeta).
        if self.damping_ratio_max > 0.0:
            import math as _math
            zeta = self.damping_ratio_max
            cfl_factor = _math.sqrt(1.0 + zeta * zeta) - zeta
        else:
            cfl_factor = 1.0
        self.dt_cfl = fem.dt_cfl * dt_safety * cfl_factor
        if dt is not None:
            self.dt = dt
        else:
            self.dt = self.dt_cfl

    def predictor(self, u, v, a, bc_mask, bc_vals):
        """Predict displacement: u_{n+1} = u_n + dt*v_n + 0.5*dt^2*a_n."""
        dt = self.dt
        if self.differentiable:
            u_new = u + dt * v + 0.5 * dt * dt * a
            u_new = torch.where(bc_mask, bc_vals, u_new)
        else:
            with torch.no_grad():
                u_new = u.add(v, alpha=dt).add_(a, alpha=0.5 * dt * dt)
                u_new[bc_mask] = bc_vals[bc_mask]
        return u_new

    def corrector(self, u_new, v, a, d, f_ext, bc_mask):
        """Correct velocity and acceleration.

        a_{n+1} = M^{-1}*(f_ext - f_int(u_{n+1}, d) - beta * f_int(v_n, d))
        v_{n+1} = v_n + 0.5*dt*(a_n + a_{n+1})

        The beta*f_int(v_n, d) term is Kelvin-Voigt stiffness-proportional
        damping (beta = zeta*dt_cfl with zeta the damping ratio at
        omega_max). Using v_n (not v_{n+1}) is the standard ABAQUS/
        LS-DYNA explicit-compatible form that avoids a coupled solve
        for a_{n+1}; the tradeoff is a tighter CFL limit, which we
        already applied in __init__.
        """
        dt = self.dt
        beta = self._damping_beta
        if self.differentiable:
            f_int = self.fem.internal_force(u_new, d)
            f_net = f_ext - f_int
            if beta > 0.0:
                f_net = f_net - beta * self.fem.internal_force(v, d)
            M_inv = self.fem.M_vec_inv.view(-1, 2)
            a_new = M_inv * f_net
            v_new = v + 0.5 * dt * (a + a_new)
            bc_zero = torch.zeros_like(a_new)
            a_new = torch.where(bc_mask, bc_zero, a_new)
            v_new = torch.where(bc_mask, bc_zero, v_new)
        else:
            with torch.no_grad():
                f_int = self.fem.internal_force(u_new, d)
                f_net = f_ext - f_int
                if beta > 0.0:
                    f_net = f_net - beta * self.fem.internal_force(v, d)
                M_inv = self.fem.M_vec_inv.view(-1, 2)
                a_new = M_inv * f_net
                v_new = v + 0.5 * dt * (a + a_new)
                a_new[bc_mask] = 0.0
                v_new[bc_mask] = 0.0
        return v_new, a_new

    def step(self, u, v, a, d, f_ext, bc_mask, bc_vals,
             rigid_connectors=None):
        """Full predictor-corrector step.

        Parameters
        ----------
        rigid_connectors : list or None
            Rotation-free rigid-connector multipoint constraints (issue
            #154 / #165). When non-empty, dispatches to the velocity-
            Verlet MPC path (:meth:`_step_mpc`) which evolves a per-
            connector theta DOF in tandem with the master translations,
            then reconstructs slave displacements / velocities /
            accelerations kinematically each step. Master translational
            DOFs must already be Dirichlet-locked (validated by
            :func:`_build_rigid_connector_T`). PR #174 / issue #206.

        Returns
        -------
        u_new, v_new, a_new : updated state tensors
        """
        # Telemetry (issue #300): explicit-dynamics has no iterative solve,
        # but the lumped-mass back-substitution counts as one effective
        # "PCG sweep" so the dataset CSV has a uniform >=1 floor across
        # all mechanics paths. Pure observability — no behaviour change.
        self.last_iter = 1
        if rigid_connectors:
            return self._step_mpc(u, v, a, d, f_ext, bc_mask, bc_vals,
                                  rigid_connectors)
        if not self.differentiable:
            with torch.no_grad():
                u_new = self.predictor(u, v, a, bc_mask, bc_vals)
                v_new, a_new = self.corrector(u_new, v, a, d, f_ext, bc_mask)
                return u_new, v_new, a_new
        u_new = self.predictor(u, v, a, bc_mask, bc_vals)
        v_new, a_new = self.corrector(u_new, v, a, d, f_ext, bc_mask)
        return u_new, v_new, a_new


    # ------------------------------------------------------------------
    # Rotation-free rigid-connector MPC (issue #154 / #165 / #206).
    # ------------------------------------------------------------------
    def _init_mpc(self, rigid_connectors, bc_mask):
        """Lazy-initialise per-connector MPC state.

        Builds, once per rigid_connectors list:
          * Polar mass moment of inertia I_theta about each master
            (sum_slaves M_slave * |X_slave - X_master|^2).
          * Cached lever-arm tensors (Y_s - Y_m, X_s - X_m) for fast
            kinematic reconstruction of slave u / v / a each step.
          * Cached slave-DOF mask used to overwrite predictor /
            corrector output at slave rows (so they evolve via the
            rigid-body kinematics, not via their own nodal mass).
          * Per-connector scalar (theta, dtheta, ddtheta) state.

        Validates that each master node is Dirichlet-locked on at least
        one translational component — mirroring the static-path
        contract enforced in ``_build_rigid_connector_T``.
        """
        # Validate master Dirichlet (re-using the helper for
        # consistency: identical error string + identical contract).
        _build_rigid_connector_T(rigid_connectors, self.fem, bc_mask)

        n_nodes = self.fem.mesh.n_nodes
        device = self.fem.mesh.device
        dtype = self.fem.mesh.dtype

        nodes = self.fem.mesh.nodes  # (N, 2)
        # Per-node lumped mass — M_vec is interleaved (mx==my per node).
        M_node = self.fem.M_vec.view(-1, 2)[:, 0]  # (N,)

        connector_state = []
        slave_dof_mask = torch.zeros(n_nodes, 2, dtype=torch.bool,
                                     device=device)
        for rc in rigid_connectors:
            slaves = rc.slaves_excluding_master().to(device=device)
            m = rc.master_node
            Xm = nodes[m, 0]
            Ym = nodes[m, 1]
            # Lever arms (slave_count,)
            dY = nodes[slaves, 1] - Ym  # Y_s - Y_m
            dX = nodes[slaves, 0] - Xm  # X_s - X_m
            r2 = dX * dX + dY * dY
            I_theta = float((M_node[slaves] * r2).sum().item())
            if I_theta <= 0.0:
                raise ValueError(
                    f"rigid_connector master_node={m}: polar mass "
                    f"moment of inertia I_theta={I_theta:.3e} is "
                    f"non-positive; check that slaves are distinct "
                    f"from the master and have non-zero lumped mass.")
            slave_dof_mask[slaves, 0] = True
            slave_dof_mask[slaves, 1] = True
            connector_state.append({
                'master': int(m),
                'slaves': slaves,
                'dX': dX,        # (n_slave,)
                'dY': dY,        # (n_slave,)
                'I_theta': I_theta,
                'theta': 0.0,
                'dtheta': 0.0,
                'ddtheta': 0.0,
            })

        self._mpc_rcs_id = id(rigid_connectors)
        self._mpc_state = connector_state
        self._mpc_slave_dof_mask = slave_dof_mask
        if not getattr(self, '_mpc_init_logged', False):
            n_rc = len(rigid_connectors)
            inertias = ", ".join(f"{c['I_theta']:.3e}"
                                 for c in connector_state)
            print(
                f"[ExplicitDynamics] rigid_connector MPC active "
                f"({n_rc} connector{'s' if n_rc != 1 else ''}, "
                f"I_theta=[{inertias}]). Slave DOFs reconstructed "
                f"kinematically from master + theta each step.",
                flush=True,
            )
            self._mpc_init_logged = True

    def _reconstruct_slave_field(self, field, master_val_x, master_val_y,
                                 scalar, st):
        """Overwrite slave rows of ``field`` (N,2) via rigid-body kinematics.

        For a single connector, with the linearised constraint about
        theta = 0:
            f_slave_x = master_val_x - scalar * dY
            f_slave_y = master_val_y + scalar * dX
        where ``scalar`` is theta (for u), dtheta (for v), or ddtheta
        (for a). Centripetal / Coriolis terms are dropped to match the
        T-matrix linearization shared with the static and CG paths.
        """
        slaves = st['slaves']
        field[slaves, 0] = master_val_x - scalar * st['dY']
        field[slaves, 1] = master_val_y + scalar * st['dX']

    def _step_mpc(self, u, v, a, d, f_ext, bc_mask, bc_vals,
                  rigid_connectors):
        """Velocity-Verlet step with rotation-free rigid-connector MPC.

        Per-connector theta DOF advances in lockstep with the master
        translations using a *physical* polar mass moment of inertia
        (no artificial epsilon, dt_CFL preserved). Slave displacements,
        velocities, and accelerations are reconstructed each step from
        master + theta — they do not evolve via their own nodal mass.
        """
        # Lazy / re-init when the list identity changes (e.g. user
        # rebuilt BoundaryConditions between solves).
        if (getattr(self, '_mpc_state', None) is None
                or getattr(self, '_mpc_rcs_id', None) != id(rigid_connectors)):
            self._init_mpc(rigid_connectors, bc_mask)

        dt = self.dt
        beta = self._damping_beta
        slave_mask = self._mpc_slave_dof_mask
        states = self._mpc_state

        # Snapshot pre-step master translational kinematics + theta.
        masters = [st['master'] for st in states]
        m_u_old = [(float(u[m, 0]), float(u[m, 1])) for m in masters]
        m_v_old = [(float(v[m, 0]), float(v[m, 1])) for m in masters]
        m_a_old = [(float(a[m, 0]), float(a[m, 1])) for m in masters]

        # ----- PREDICTOR -----
        # 1. Translational predictor on the full (N, 2) tensor. Master
        #    rows are Dirichlet-pinned by bc_mask; slave rows are about
        #    to be overwritten by kinematic reconstruction.
        with torch.no_grad():
            u_new = u.add(v, alpha=dt).add_(a, alpha=0.5 * dt * dt)
            u_new[bc_mask] = bc_vals[bc_mask]

        # 2. Theta predictor + slave-displacement reconstruction.
        for k, st in enumerate(states):
            theta_pred = (st['theta'] + dt * st['dtheta']
                          + 0.5 * dt * dt * st['ddtheta'])
            st['theta_pred'] = theta_pred
            m = st['master']
            self._reconstruct_slave_field(
                u_new, u_new[m, 0], u_new[m, 1], theta_pred, st)

        # ----- CORRECTOR -----
        # 3. Internal force at the reconstructed full-displacement state.
        with torch.no_grad():
            f_int = self.fem.internal_force(u_new, d)
            f_net = f_ext - f_int
            if beta > 0.0:
                f_net = f_net - beta * self.fem.internal_force(v, d)

            # 4. Translational corrector on the full field. Master &
            #    Dirichlet rows are zeroed; slave a/v will be
            #    overwritten via kinematic reconstruction below.
            M_inv = self.fem.M_vec_inv.view(-1, 2)
            a_new = M_inv * f_net
            v_new = v + 0.5 * dt * (a + a_new)
            a_new[bc_mask] = 0.0
            v_new[bc_mask] = 0.0

            # 5. Project slave nodal forces onto each theta DOF and
            #    advance dtheta / ddtheta.
            #    Generalized moment about master k:
            #        M_theta_k = sum_slaves [
            #            (X_s - X_m) * f_y_s - (Y_s - Y_m) * f_x_s ]
            #    where f = f_ext - f_int (- damping). Mass on theta is
            #    the polar inertia I_theta cached at init. Velocity-
            #    Verlet: ddtheta_{n+1} = M_theta / I_theta.
            for k, st in enumerate(states):
                slaves = st['slaves']
                f_sx = f_net[slaves, 0]
                f_sy = f_net[slaves, 1]
                M_theta = float(
                    (st['dX'] * f_sy - st['dY'] * f_sx).sum().item())
                ddtheta_new = M_theta / st['I_theta']
                dtheta_new = (st['dtheta']
                              + 0.5 * dt * (st['ddtheta'] + ddtheta_new))

                # 6. Reconstruct slave velocities / accelerations from
                #    master + theta kinematics (overwrite the values
                #    the per-slave corrector just produced).
                m = st['master']
                self._reconstruct_slave_field(
                    v_new, v_new[m, 0], v_new[m, 1], dtheta_new, st)
                self._reconstruct_slave_field(
                    a_new, a_new[m, 0], a_new[m, 1], ddtheta_new, st)

                # 7. Commit per-connector state.
                st['theta'] = st['theta_pred']
                st['dtheta'] = dtheta_new
                st['ddtheta'] = ddtheta_new
                # Diagnostics retained on the instance (mirrors the
                # static / CG paths' last_theta hook).
                # (Avoid del'ing theta_pred — overwritten next step.)

            # 8. Mirror Dirichlet zero on slave rows isn't needed —
            #    the kinematic reconstruction already produced the
            #    physically correct slave v/a; bc_mask (which excludes
            #    slaves under rotation_free) leaves them untouched.
            del f_int, f_net

        # Diagnostics: last_theta list (parallel to DirectSolver /
        # SecantCGSolver._record_mpc_diagnostics_cg).
        self.last_theta = [st['theta'] for st in states]

        # Suppress unused-variable lint on captured snapshots — kept
        # available for future debugging hooks (e.g. verifying
        # consistency between master a_new and m_a_old + dt update).
        del m_u_old, m_v_old, m_a_old

        return u_new, v_new, a_new


class GeneralizedAlphaDynamics:
    """Forward-only implicit generalized-alpha dynamics.

    This is the COMSOL-style opt-in dynamic path tracked in #570.  It keeps
    the current explicit central-difference solver untouched and solves the
    mechanics update with a matrix-free Newton-PCG effective operator,

        A_eff da = (1-alpha_m) M da
                   + (1-alpha_f) beta dt^2 K_t da.

    The first production slice is intentionally forward-only: it does not
    claim differentiable inverse support, does not support rigid connectors,
    and uses fixed damage during the mechanics solve.  The existing staggered
    phase-field update then advances H and d after this mechanics step.
    """

    def __init__(self, fem: FEMOperators, dt: float = None,
                 dt_safety: float = 1.0, rho_inf: float = 0.5,
                 newton_tol: float = 1e-8, newton_max_iter: int = 20,
                 cg_tol: float = 1e-8, cg_max_iter: int = 500,
                 differentiable: bool = False):
        if differentiable:
            raise NotImplementedError(
                "generalized_alpha is forward-only for now; implicit "
                "adjoint support is tracked in #572.")
        self.fem = fem
        self.dt_cfl = fem.dt_cfl * dt_safety
        self.dt = self.dt_cfl if dt is None else dt
        self.rho_inf = float(rho_inf)
        self.alpha_m, self.alpha_f, self.beta, self.gamma = gen_alpha_params(
            self.rho_inf)
        self.newton_tol = float(newton_tol)
        self.newton_max_iter = int(newton_max_iter)
        self.cg_tol = float(cg_tol)
        self.cg_max_iter = int(cg_max_iter)
        self.last_iter = 0
        self.last_newton_iter = 0
        self.last_residual = float('nan')
        self.last_converged = True

    def _effective_preconditioner(self, d, free_mask_flat):
        M = self.fem.M_vec.reshape(-1)
        K_diag = self.fem.stiffness_diagonal(d).reshape(-1)
        scale_k = (1.0 - self.alpha_f) * self.beta * self.dt * self.dt
        diag = (1.0 - self.alpha_m) * M + scale_k * K_diag
        floor = 1e-12 * diag.abs().max().clamp(min=1e-30)
        return free_mask_flat / diag.abs().clamp(min=floor)

    def _tangent_matvec(self, u_lin, d_lin, direction):
        def _f(uu):
            return self.fem.internal_force(uu, d_lin)

        with torch.enable_grad():
            _, jvp_out = torch.autograd.functional.jvp(
                _f, (u_lin,), (direction,), create_graph=False)
        return jvp_out

    def _pcg(self, matvec, rhs, M_inv, free_mask_flat):
        x = torch.zeros_like(rhs)
        r = rhs * free_mask_flat
        z = M_inv * r
        p = z.clone()
        rz = torch.dot(r, z)
        tol_sq = self.cg_tol * self.cg_tol
        rr0 = torch.dot(r, r).item()
        if rr0 < tol_sq:
            return x, 0
        it = 0
        for it in range(1, self.cg_max_iter + 1):
            Ap = matvec(p.reshape(-1, 2)).reshape(-1) * free_mask_flat
            pAp = torch.dot(p, Ap)
            alpha = rz / (pAp + 1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rr = torch.dot(r, r)
            if rr.item() < tol_sq:
                break
            z = M_inv * r
            rz_new = torch.dot(r, z)
            p = z + (rz_new / (rz + 1e-30)) * p
            rz = rz_new
            if rr.item() > 1e12 * max(rr0, 1e-30):
                break
        return x, it

    def step(self, u, v, a, d, f_ext, bc_mask, bc_vals,
             rigid_connectors=None):
        if rigid_connectors:
            raise NotImplementedError(
                "generalized_alpha does not yet support rigid_connector MPC; "
                "use central_difference/verlet for those runs.")

        dt = self.dt
        am, af, beta, gamma = (
            self.alpha_m, self.alpha_f, self.beta, self.gamma)
        free_mask = (~bc_mask).to(u.dtype)
        free_flat = free_mask.reshape(-1)
        M = self.fem.M_vec.view(-1, 2)

        u_n = u.detach()
        v_n = v.detach()
        a_n = a.detach()
        d_lin = d.detach()
        f_int_n = self.fem.internal_force(u_n, d_lin).detach()

        u_pred = u_n + dt * v_n + 0.5 * dt * dt * (1.0 - 2.0 * beta) * a_n
        v_pred = v_n + dt * (1.0 - gamma) * a_n

        a_new = a_n.clone()
        a_new[bc_mask] = 0.0
        M_inv_eff = self._effective_preconditioner(d_lin, free_flat)
        total_cg = 0
        converged = False

        for nr in range(self.newton_max_iter):
            u_new = u_pred + beta * dt * dt * a_new
            u_new = torch.where(bc_mask, bc_vals, u_new)
            residual = (
                (1.0 - am) * M * a_new
                + am * M * a_n
                + (1.0 - af) * self.fem.internal_force(u_new, d_lin)
                + af * f_int_n
                - f_ext
            ) * free_mask
            res_norm = residual.norm().item()
            self.last_residual = res_norm
            if res_norm < self.newton_tol:
                converged = True
                self.last_newton_iter = nr
                break

            u_lin = u_new.detach().requires_grad_(True)

            def Aeff(direction):
                kt_da = self._tangent_matvec(
                    u_lin, d_lin, beta * dt * dt * direction)
                return ((1.0 - am) * M * direction
                        + (1.0 - af) * kt_da) * free_mask

            da_flat, cg_it = self._pcg(
                Aeff, -residual.reshape(-1), M_inv_eff, free_flat)
            total_cg += cg_it
            a_new = a_new + da_flat.reshape_as(a_new)
            a_new[bc_mask] = 0.0

        u_new = u_pred + beta * dt * dt * a_new
        v_new = v_pred + gamma * dt * a_new
        u_new = torch.where(bc_mask, bc_vals, u_new)
        v_new = torch.where(bc_mask, torch.zeros_like(v_new), v_new)
        a_new = torch.where(bc_mask, torch.zeros_like(a_new), a_new)

        self.last_iter = max(total_cg, 1)
        self.last_converged = converged
        if not converged:
            self.last_newton_iter = self.newton_max_iter
        return u_new, v_new, a_new


class StaticSolver:
    """[INTERNAL] CG solver for linear elastic static equilibrium K @ u = f.

    Used internally by StaggeredSolver for pre-strain initialization (d=0).
    Not intended for direct use in benchmarks.

    Parameters
    ----------
    fem : FEMOperators
    tol : float
        Residual norm tolerance.
    max_iter : int
    """

    def __init__(self, fem: FEMOperators, tol: float = 1e-8,
                 max_iter: int = 5000,
                 backend: str = 'auto',
                 sparse_dof_threshold: int = 200_000):
        self.fem = fem
        self.tol = tol
        self.max_iter = max_iter
        if backend not in ('auto', 'scipy', 'cg'):
            raise ValueError(
                "StaticSolver backend must be 'auto', 'scipy' or 'cg'; "
                f"got {backend!r}")
        self.backend = backend
        self.sparse_dof_threshold = int(sparse_dof_threshold)

    def _resolve_backend(self, n_dof_free: int) -> str:
        if self.backend == 'cg':
            return 'cg'
        try:
            from .sparse_solve import scipy_available
            ok = scipy_available()
        except Exception:
            ok = False
        if self.backend == 'scipy':
            if not ok:
                raise RuntimeError(
                    "StaticSolver(backend='scipy') requires scipy.")
            return 'scipy'
        # auto: scipy when small and available, else CG
        if ok and n_dof_free <= self.sparse_dof_threshold:
            return 'scipy'
        return 'cg'

    @torch.no_grad()
    def solve(self, bc_mask, bc_vals, u_init=None):
        """Solve K @ u = 0 subject to Dirichlet BCs (linear elastic).

        Parameters
        ----------
        bc_mask : (N, 2) bool
        bc_vals : (N, 2) float
        u_init : (N, 2) float or None

        Returns
        -------
        u : (N, 2) equilibrium displacement
        """
        mesh = self.fem.mesh
        n_dof_free = int((~bc_mask).sum().item())
        backend_used = self._resolve_backend(n_dof_free)

        if backend_used == 'scipy':
            return self._solve_scipy(bc_mask, bc_vals, u_init)
        print("[StaticSolver] Starting CG solve...", flush=True)
        if u_init is not None:
            u = u_init.clone()
        else:
            u = torch.zeros(
                mesh.n_nodes, 2, dtype=mesh.dtype, device=mesh.device)
        u[bc_mask] = bc_vals[bc_mask]

        free_mask = (~bc_mask).to(u.dtype)

        # Compute Jacobi preconditioner (diagonal of K, undamaged)
        K_diag = self.fem.stiffness_diagonal(d=None)
        K_diag *= free_mask
        diag_floor = 1e-10 * K_diag.abs().max().clamp(min=1e-30)
        M_inv = free_mask / K_diag.clamp(min=diag_floor)

        r = -self.fem.internal_force_linear(u)
        r *= free_mask
        z = M_inv * r  # preconditioned residual
        p = z.clone()
        rz = (r * z).sum()

        check_every = 50
        tol_sq = self.tol ** 2

        # Check if already converged before entering the loop
        rr = (r * r).sum()
        if rr.item() < tol_sq:
            print(f"[StaticSolver] Already converged, ||r||={rr.sqrt():.2e}",
                  flush=True)
            return u

        r_norm_sq_0 = rr.clone()

        n_iter = 0
        for i in range(self.max_iter):
            Ap = self.fem.internal_force_linear(p)
            Ap *= free_mask
            pAp = (p * Ap).sum()
            if pAp.abs() < 1e-30:
                n_iter = i + 1
                break
            alpha = rz / pAp
            u.add_(alpha * p)
            r.sub_(alpha * Ap)
            rr_new = (r * r).sum()
            z = M_inv * r  # preconditioned residual
            rz_new = (r * z).sum()
            p.mul_(rz_new / (rz + 1e-30)).add_(z)
            rz = rz_new
            rr = rr_new
            n_iter = i + 1

            if n_iter % check_every == 0:
                if rr.item() < tol_sq:
                    break
                if rr.item() > 1e12 * r_norm_sq_0.item():
                    print(f"  [StaticSolver diverged at iter {n_iter}]", flush=True)
                    break

        print(f"[StaticSolver] Converged at iter {n_iter}, ||r||={rr.sqrt():.2e}",
              flush=True)
        return u

    @torch.no_grad()
    def _solve_scipy(self, bc_mask, bc_vals, u_init=None):
        """Sparse-direct linear elastic solve via SciPy SuperLU (#106)."""
        import numpy as np
        from .sparse_solve import SparseSolveAutograd

        mesh = self.fem.mesh
        if u_init is not None:
            u = u_init.clone()
        else:
            u = torch.zeros(
                mesh.n_nodes, 2, dtype=mesh.dtype, device=mesh.device)
        u[bc_mask] = bc_vals[bc_mask]

        # Assemble linear-elastic K with d=0 (no degradation).
        elems = mesh.elements.detach().cpu().numpy()
        n_elem = elems.shape[0]
        elem_dofs = np.zeros((n_elem, 6), dtype=np.int64)
        for i in range(3):
            elem_dofs[:, 2 * i] = 2 * elems[:, i]
            elem_dofs[:, 2 * i + 1] = 2 * elems[:, i] + 1
        rows = np.repeat(elem_dofs, 6, axis=1).flatten()
        cols = np.tile(elem_dofs, (1, 6)).flatten()
        n_dof = 2 * mesh.n_nodes

        gp = mesh.grad_phi.detach().cpu().to(torch.float64).numpy()
        areas = mesh.areas.detach().cpu().to(torch.float64).numpy()
        C = self.fem.C.detach().cpu().to(torch.float64).numpy()
        gpx = gp[:, :, 0]; gpy = gp[:, :, 1]
        B = np.zeros((n_elem, 3, 6), dtype=np.float64)
        for i in range(3):
            B[:, 0, 2 * i] = gpx[:, i]
            B[:, 1, 2 * i + 1] = gpy[:, i]
            B[:, 2, 2 * i] = gpy[:, i]
            B[:, 2, 2 * i + 1] = gpx[:, i]
        CB = np.einsum('ij,ejk->eik', C, B)
        Ke = np.einsum('eji,ejk->eik', B, CB)
        Ke *= areas[:, None, None]
        vals = Ke.flatten()

        # Apply BCs: zero rows/cols of fixed DOFs and put 1 on diag, then move
        # K_fb @ u_b to RHS so the reduced system is K_ff @ u_f = -K_fb @ u_b.
        free_flat = (~bc_mask).detach().cpu().numpy().reshape(-1)  # bool
        u_flat = u.detach().cpu().to(torch.float64).numpy().reshape(-1)

        # Compute RHS = -K @ u_prescribed (with u_free=0 placeholder).
        u_pres = np.where(free_flat, 0.0, u_flat)
        # Sparse matvec K @ u_pres via COO.
        Ku_pres = np.zeros(n_dof, dtype=np.float64)
        np.add.at(Ku_pres, rows, vals * u_pres[cols])
        rhs = -Ku_pres
        rhs[~free_flat] = 0.0  # fixed DOFs: residual irrelevant

        # Zero rows/cols of fixed DOFs in K, and place 1 on diagonal.
        keep = free_flat[rows] & free_flat[cols]
        vals_bc = vals * keep.astype(np.float64)
        fixed_idx = np.where(~free_flat)[0]
        if fixed_idx.size > 0:
            rows_bc = np.concatenate([rows, fixed_idx])
            cols_bc = np.concatenate([cols, fixed_idx])
            vals_bc = np.concatenate(
                [vals_bc, np.ones(fixed_idx.size, dtype=np.float64)])
        else:
            rows_bc = rows; cols_bc = cols

        indices = torch.from_numpy(np.stack([rows_bc, cols_bc], axis=0))
        values = torch.from_numpy(vals_bc)
        rhs_t = torch.from_numpy(rhs)
        du_flat = SparseSolveAutograd.apply(indices, values, rhs_t, n_dof)
        du = du_flat.reshape(-1, 2).to(device=u.device, dtype=u.dtype)
        u_out = u + du
        # Re-impose Dirichlet exactly.
        u_out[bc_mask] = bc_vals[bc_mask]
        return u_out


class QuasiStaticSolver:
    """[AVAILABLE] Newton-Raphson solver for quasi-static equilibrium with degradation.

    Supports all energy splits: 'isotropic', 'amor', 'spectral', 'star_convex'.
    For piecewise-linear splits (spectral/amor/star_convex) the inner CG matvec
    uses an autograd JVP through internal_force at the current Newton iterate u
    -- this is the consistent tangent K(d, u) @ du re-linearised every NR step
    (Miehe et al. 2010 CMAME 199). SecantCGSolver remains the recommended
    path for very large problems thanks to its multigrid/L-BFGS preconditioning;
    QuasiStaticSolver is the simpler default and is preferred when explicit
    Newton convergence monitoring is needed. Spectral support added in PR #170
    (issue #114); see tests/test_quasi_static_spectral.py.

    For implicit quasi-static analysis: finds u such that
        f_int(u, d) = f_ext
    using Newton-Raphson iteration with a CG inner solve.

    Parameters
    ----------
    fem : FEMOperators
    tol : float
        Residual norm tolerance for outer NR loop.
    max_iter : int
        Maximum NR iterations per load step.
    cg_tol : float
        Inner CG tolerance.
    cg_max_iter : int
        Inner CG maximum iterations.
    """

    def __init__(self, fem: FEMOperators, tol: float = 1e-6,
                 max_iter: int = 50, cg_tol: float = 1e-8,
                 cg_max_iter: int = 5000,
                 backend: str = 'auto',
                 sparse_dof_threshold: int = 200_000,
                 inner_solver: str = None,
                 consistent_tangent: bool = None,
                 line_search: bool = True,
                 line_search_max_steps: int = 8,
                 line_search_min_alpha: float = 1e-4,
                 line_search_c: float = 1e-4,
                 tol_rel: float = None,
                 plasticity_operator=None,
                 cohesive_operator=None):
        """Parameters
        ----------
        backend : {'auto', 'scipy', 'mumps', 'cudss', 'cg'}
            Linear-solver backend for the Newton inner step.
            * ``'cg'``    — original matrix-free preconditioned-CG path (default
              behaviour preserved).
            * ``'scipy'`` — assemble K and call
              :class:`~phast.sparse_solve.SparseSolveAutograd`
              (SuperLU).  Raises if scipy is unavailable.
            * ``'mumps'`` — PETSc/MUMPS sparse-direct backend (Phase 2 of
              epic #105).  Raises ``ImportError`` if petsc4py is not
              installed.
            * ``'cudss'`` — cuDSS sparse-direct backend for the sparse J2
              plasticity operator path. Elastic/cohesive mechanics still use
              the SciPy/MUMPS sparse-direct wrappers or CG.
            * ``'auto'``  — pick the best available sparse-direct backend
              (cuDSS on CUDA-backed sparse J2, otherwise mumps > scipy) when
              the free DOF count is ``<= sparse_dof_threshold``; otherwise
              fall back to CG.
        sparse_dof_threshold : int
            DOF cutoff for the auto backend.  Above this size scipy SuperLU is
            disabled and the matrix-free CG path is used instead.
        inner_solver : {'cg', 'direct', None}, optional
            #260 surface-level alias for ``backend``. ``'cg'`` maps to
            ``backend='cg'`` (matrix-free PCG); ``'direct'`` maps to
            ``backend='auto'`` (sparse-direct: mumps > scipy, falling back
            to CG above the DOF threshold). When provided this OVERRIDES
            ``backend``. The full ``backend`` enum (``'scipy'``/``'mumps'``)
            remains available for users who need to pin a specific
            sparse-direct library.
        consistent_tangent : bool, optional
            Tangent-operator selector for piecewise-linear energy splits
            ('spectral'/'amor'/'star_convex'). #260; semantics:
            * ``None`` (default) — preserve PR #170 behaviour: autograd-
              JVP consistent tangent on spectral splits, secant on
              isotropic (where the two coincide exactly).
            * ``True``  — same as None on spectral splits (autograd-JVP),
              no-op on isotropic.
            * ``False`` — opt-in *secant* (frozen-state) tangent fallback
              on spectral splits, mirroring SecantCGSolver's linearization.
              Use this if the autograd-JVP path fails to converge near
              damage saturation; see PR for the regime characterisation.
              No-op on isotropic (secant ≡ consistent there).
            On the isotropic split the tangent is exact regardless of
            this flag, so cantilever / linear-elastic benchmarks see no
            difference.
        """
        self.fem = fem
        if plasticity_operator is not None and cohesive_operator is not None:
            raise NotImplementedError(
                "QuasiStaticSolver does not yet support coupled "
                "plasticity_operator + cohesive_operator mechanics. The "
                "current J2 path and cohesive-interface path have separate "
                "state update/rollback contracts; use one mechanism at a "
                "time until the coupled residual/tangent integration is "
                "implemented.")
        self.tol = tol
        self.max_iter = max_iter
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        if backend not in ('auto', 'scipy', 'mumps', 'cudss', 'cg'):
            raise ValueError(
                "backend must be 'auto', 'scipy', 'mumps', 'cudss' or 'cg'; "
                f"got {backend!r}")
        # inner_solver alias resolution (#260): when supplied, override
        # backend. 'cg' → matrix-free CG; 'direct' → auto (mumps/scipy).
        if inner_solver is not None:
            if inner_solver == 'cg':
                backend = 'cg'
            elif inner_solver == 'direct':
                # MPS/CUDA constraint: SciPy SuperLU is CPU-only, the
                # autograd Function transfers to CPU internally so this
                # works, but warn the user once.
                fem_dev = getattr(fem, 'device', None)
                if fem_dev is not None and not isinstance(fem_dev, str):
                    fem_dev = str(fem_dev)
                if fem_dev and not fem_dev.startswith('cpu'):
                    print(
                        "[QuasiStaticSolver] inner_solver='direct' on "
                        f"device={fem_dev!r}: sparse-direct factor runs "
                        "on CPU float64 (SuperLU/MUMPS); inputs/outputs "
                        "are transferred each Newton step.",
                        flush=True)
                backend = 'auto'
            else:
                raise ValueError(
                    "inner_solver must be 'cg' or 'direct'; "
                    f"got {inner_solver!r}")
        self.backend = backend
        self.inner_solver = inner_solver
        self.consistent_tangent = consistent_tangent
        self.line_search = bool(line_search)
        self.line_search_max_steps = int(line_search_max_steps)
        self.line_search_min_alpha = float(line_search_min_alpha)
        self.line_search_c = float(line_search_c)
        self.sparse_dof_threshold = int(sparse_dof_threshold)
        self.tol_rel = tol_rel
        self.plasticity_operator = plasticity_operator
        self._plasticity_solver = None
        self.cohesive_operator = cohesive_operator
        self._asm_cache = None  # filled lazily once mesh is known
        # MPC diagnostics — populated on rigid_connector solves so
        # downstream postprocess code can read them solver-independently.
        # Mirrors DirectSolver / SecantCGSolver attributes.
        self.last_theta = []
        self.last_master_reaction = []
        self.last_residual = float('nan')
        self.last_residual0 = float('nan')
        self.last_relative_residual = float('nan')
        self.last_line_search_alpha = 1.0
        self.last_line_search_reductions = 0
        self.last_arc_length_residual = float('nan')
        self.last_arc_length_constraint = float('nan')
        self.last_load_factor = float('nan')
        self.last_backend = None
        self.last_failure = None
        self._secant_direct_assembler = None
        self._backend_choice_logged = False

    def _update_residual_diagnostics(self, residual_norm: float,
                                     residual0: float | None) -> None:
        """Record absolute and relative Newton residual diagnostics."""
        self.last_residual = float(residual_norm)
        if residual0 is None:
            self.last_residual0 = float('nan')
            self.last_relative_residual = float('nan')
            return
        self.last_residual0 = float(residual0)
        if residual0 > 0.0:
            self.last_relative_residual = float(residual_norm / residual0)
        elif residual_norm == 0.0:
            self.last_relative_residual = 0.0
        else:
            self.last_relative_residual = float('nan')

    def _resolve_backend(self, n_dof_free: int) -> str:
        """Decide which linear-solve backend to use this Newton step."""
        return self._backend_candidates(n_dof_free, self.backend)[0]

    def _log_backend_choice(self, backend_used: str, energy_split: str,
                            n_dof_free: int, use_autograd_tangent: bool):
        """Emit one unambiguous mechanics-backend line per solver instance."""
        if self._backend_choice_logged:
            return
        if backend_used == 'mumps':
            backend_desc = 'PETSc/MUMPS sparse-direct LU'
        elif backend_used == 'scipy':
            backend_desc = 'SciPy SuperLU sparse-direct LU'
        else:
            backend_desc = 'matrix-free preconditioned CG'
        if backend_used in ('scipy', 'mumps') and energy_split != 'isotropic':
            tangent_desc = 'frozen-state secant sparse tangent'
        elif use_autograd_tangent:
            tangent_desc = 'autograd-JVP consistent tangent'
        else:
            tangent_desc = 'isotropic linear tangent'
        print(
            "[QuasiStaticSolver] mechanics backend: "
            f"{backend_used} ({backend_desc}); tangent={tangent_desc}; "
            f"free_dofs={n_dof_free}",
            flush=True,
        )
        self._backend_choice_logged = True

    def _backend_candidates(self, n_dof_free: int,
                           requested_backend: str) -> list[str]:
        """Return ordered backend candidates for a Newton step."""
        req = str(requested_backend).lower()
        if req not in ('auto', 'scipy', 'mumps', 'cg'):
            raise ValueError(
                "backend must be 'auto', 'scipy', 'mumps' or 'cg'; "
                f"got {requested_backend!r}")
        if req == 'cg':
            return ['cg']
        try:
            # Runtime smoke test (#403): a broken petsc4py install imports
            # cleanly but its solve raises, so import-only checks would route
            # us to a backend that dies mid-Newton. Smoke test result caches.
            from .sparse_solve import scipy_available, _petsc_functional
            scipy_ok = scipy_available()
            mumps_ok = _petsc_functional()
        except Exception:
            scipy_ok = False
            mumps_ok = False

        candidates = []
        if req == 'scipy':
            if scipy_ok:
                candidates.append('scipy')
            candidates.append('cg')
            return candidates
        if req == 'mumps':
            if mumps_ok:
                candidates.append('mumps')
            if scipy_ok and 'scipy' not in candidates:
                candidates.append('scipy')
            candidates.append('cg')
            return candidates

        # auto: prefer mumps > scipy > cg subject to DOF threshold.
        if n_dof_free <= self.sparse_dof_threshold:
            if mumps_ok:
                candidates.append('mumps')
            if scipy_ok and 'scipy' not in candidates:
                candidates.append('scipy')
        if not candidates:
            candidates.append('cg')
        return candidates

    def _has_converged(self, residual_norm: float, residual0: float | None,
                       constraint: float | None = None,
                       tol: float | None = None,
                       tol_rel: float | None = None) -> bool:
        """Shared convergence check for displacement and arc-length solves."""
        if math.isnan(residual_norm):
            return False
        residual_tol = self.tol if tol is None else tol
        if residual_norm <= residual_tol:
            if constraint is None:
                return True
            if abs(constraint) <= residual_tol:
                return True
        rel_tol = self.tol_rel if tol_rel is None else tol_rel
        if rel_tol is not None and residual0 is not None and residual0 > 0:
            if residual_norm <= rel_tol * residual0:
                if constraint is None:
                    return True
                if constraint is not None and abs(constraint) <= residual_tol:
                    return True
        return False

    def _assembly_indices(self):
        """Cache COO row/col indices for element-stiffness blocks.

        Mirrors ``DirectSolver._precompute_assembly_indices`` but lives on
        QuasiStaticSolver to keep that solver self-contained.
        """
        if self._asm_cache is not None:
            return self._asm_cache
        import numpy as np
        elems = self.fem.mesh.elements.detach().cpu().numpy()
        n_local = int(elems.shape[1])
        if n_local not in (3, 4):
            raise NotImplementedError(
                "Sparse-direct mechanics assembly supports T3 and Q4 meshes; "
                f"got {n_local} nodes per element.")
        n_local_dof = 2 * n_local
        n_elem = elems.shape[0]
        elem_dofs = np.zeros((n_elem, n_local_dof), dtype=np.int64)
        for i in range(n_local):
            elem_dofs[:, 2 * i] = 2 * elems[:, i]
            elem_dofs[:, 2 * i + 1] = 2 * elems[:, i] + 1
        rows = np.repeat(elem_dofs, n_local_dof, axis=1).flatten()
        cols = np.tile(elem_dofs, (1, n_local_dof)).flatten()
        n_dof = 2 * self.fem.mesh.n_nodes
        self._asm_cache = (rows, cols, elem_dofs, n_dof)
        return self._asm_cache

    def _assemble_K_isotropic(self, d):
        """Assemble the global tangent stiffness for the isotropic split.

        K = sum_e g(d_e) * area_e * B_e^T C B_e

        Returns ``(indices, values, n_dof)`` torch tensors on CPU/float64
        suitable for :class:`SparseSolveAutograd`.
        """
        import numpy as np
        fem = self.fem
        mesh = fem.mesh
        rows, cols, _, n_dof = self._assembly_indices()
        element_type = getattr(mesh, "element_type", "T3")

        # Keep values in torch so SparseSolveAutograd.backward can propagate
        # through degradation g(d). COO indices remain fixed CPU integers.
        device = d.device
        lam, mu, _kappa = fem._resolve_lame()
        lam_t = torch.as_tensor(lam, device=device, dtype=torch.float64)
        mu_t = torch.as_tensor(mu, device=device, dtype=torch.float64)

        elems = mesh.elements.to(device=device)
        d_e = d.to(dtype=torch.float64)[elems]
        n_elem = elems.shape[0]

        if element_type == "Q4":
            gp = mesh.quad_grad_phi.to(device=device, dtype=torch.float64)
            wdetJ = mesh.quad_wdetJ.to(device=device, dtype=torch.float64)
            quad_N = mesh.quad_N.to(device=device, dtype=torch.float64)
            n_quad = gp.shape[1]
            n_local = gp.shape[2]
            n_local_dof = 2 * n_local

            d_q = torch.einsum('qa,ea->eq', quad_N, d_e)
            g_q = fem.material.degradation(d_q)

            gpx = gp[..., 0]
            gpy = gp[..., 1]
            B = torch.zeros(
                (n_elem, n_quad, 3, n_local_dof),
                dtype=torch.float64,
                device=device,
            )
            for i in range(n_local):
                B[:, :, 0, 2 * i] = gpx[:, :, i]
                B[:, :, 1, 2 * i + 1] = gpy[:, :, i]
                B[:, :, 2, 2 * i] = gpy[:, :, i]
                B[:, :, 2, 2 * i + 1] = gpx[:, :, i]

            if lam_t.ndim == 0:
                C = fem.C.to(device=device, dtype=torch.float64)
                CB = torch.einsum('ij,eqjk->eqik', C, B)
            else:
                C_e = torch.zeros(
                    (n_elem, 3, 3), dtype=torch.float64, device=device)
                C_e[:, 0, 0] = lam_t + 2.0 * mu_t
                C_e[:, 0, 1] = lam_t
                C_e[:, 1, 0] = lam_t
                C_e[:, 1, 1] = lam_t + 2.0 * mu_t
                C_e[:, 2, 2] = mu_t
                CB = torch.einsum('eij,eqjk->eqik', C_e, B)

            Ke_q = torch.einsum('eqji,eqjk->eqik', B, CB)
            Ke = (Ke_q * (g_q * wdetJ).view(n_elem, n_quad, 1, 1)).sum(dim=1)
        else:
            gp = mesh.grad_phi.to(device=device, dtype=torch.float64)
            areas = mesh.areas.to(device=device, dtype=torch.float64)

            # Element-averaged degradation g(d_e).
            d_avg = d_e.mean(dim=1)
            g_e = fem.material.degradation(d_avg)

            gpx = gp[:, :, 0]
            gpy = gp[:, :, 1]
            B = torch.zeros((n_elem, 3, 6), dtype=torch.float64, device=device)
            for i in range(3):
                B[:, 0, 2 * i] = gpx[:, i]
                B[:, 1, 2 * i + 1] = gpy[:, i]
                B[:, 2, 2 * i] = gpy[:, i]
                B[:, 2, 2 * i + 1] = gpx[:, i]

            # Ke = g_e * area_e * B^T C B   (vectorised einsum).
            # When diff_E_field is installed, Lamé parameters are per element.
            # Keep that tensor path alive so sparse-direct QS solves can recover
            # smooth isotropic E(x,y) fields by autograd.
            if lam_t.ndim == 0:
                C = fem.C.to(device=device, dtype=torch.float64)   # (3, 3)
                CB = torch.einsum('ij,ejk->eik', C, B)            # (E, 3, 6)
            else:
                C_e = torch.zeros(
                    (n_elem, 3, 3), dtype=torch.float64, device=device)
                C_e[:, 0, 0] = lam_t + 2.0 * mu_t
                C_e[:, 0, 1] = lam_t
                C_e[:, 1, 0] = lam_t
                C_e[:, 1, 1] = lam_t + 2.0 * mu_t
                C_e[:, 2, 2] = mu_t
                CB = torch.einsum('eij,ejk->eik', C_e, B)         # (E, 3, 6)
            Ke = torch.einsum('eji,ejk->eik', B, CB)              # (E, 6, 6)
            Ke = Ke * (g_e * areas).view(-1, 1, 1)
        vals = Ke.reshape(-1).to(device='cpu', dtype=torch.float64)

        indices = torch.from_numpy(np.stack([rows, cols], axis=0))
        return indices, vals, n_dof

    def _assemble_K_secant(self, u, d):
        """Assemble a frozen-state tangent for non-isotropic splits.

        The matrix-free CG path uses an autograd JVP for the consistent
        tangent of spectral/Amor/star-convex splits. Sparse-direct backends
        need an explicit matrix, so this validation/backend path reuses the
        DirectSolver's frozen secant assembly: freeze the split state at the
        current Newton iterate, assemble the sparse tangent, and factorise it
        with SciPy or PETSc/MUMPS. This mirrors the robust assembled-solver
        route used by reference codes, while the matrix-free JVP path remains
        available through backend='cg'.
        """
        import numpy as np

        if self._secant_direct_assembler is None:
            self._secant_direct_assembler = DirectSolver(
                self.fem, tol=self.tol, max_newton=1,
                rtol=self.tol_rel or 1e-8, log_backend=False)

        state = self.fem.freeze_secant_state(u.detach(), d.detach())
        K = self._secant_direct_assembler._assemble_stiffness(state).tocoo()
        indices = torch.from_numpy(np.vstack((K.row, K.col)).astype(np.int64))
        values = torch.from_numpy(K.data.astype(np.float64, copy=False))
        return indices, values, int(K.shape[0])

    def _scipy_newton_step(self, u, d, residual, free_mask,
                           backend: str = 'scipy', factor_handle=None):
        """One Newton step solved via the sparse-direct autograd wrapper.

        BCs are enforced by zeroing rows/cols of K corresponding to
        constrained DOFs and putting 1.0 on the diagonal -- the residual is
        already masked to be zero at those rows, so du is zero there.

        Parameters
        ----------
        backend : {'scipy', 'mumps'}
            Which sparse-direct backend's autograd Function to dispatch to.
        """
        import numpy as np
        from .sparse_solve import (
            SparseSolveAutograd, _MumpsSparseSolveAutograd)

        split = getattr(self.fem.material, 'energy_split', 'isotropic')
        if split == 'isotropic':
            indices, values, n_dof = self._assemble_K_isotropic(d)
        else:
            indices, values, n_dof = self._assemble_K_secant(u, d)
        # Build a free-DOF mask in flat (n_dof,) form.
        free_flat = free_mask.detach().cpu().to(torch.float64).reshape(-1)  # (N*2,)
        free_np = free_flat.numpy() > 0.5  # bool, True = free

        rows_np = indices[0].numpy()
        cols_np = indices[1].numpy()

        # Zero out fixed rows AND fixed cols.
        keep = free_np[rows_np] & free_np[cols_np]
        keep_t = torch.from_numpy(keep.astype(np.float64))
        values_bc = values * keep_t

        # Add a unit diagonal at fixed DOFs so K is non-singular.
        n = int(n_dof)
        fixed_idx = np.where(~free_np)[0]
        if fixed_idx.size > 0:
            extra_rows = fixed_idx
            extra_cols = fixed_idx
            rows_np = np.concatenate([rows_np, extra_rows])
            cols_np = np.concatenate([cols_np, extra_cols])
            values_bc = torch.cat([
                values_bc,
                torch.ones(fixed_idx.size, dtype=values_bc.dtype),
            ])

        indices_bc = torch.from_numpy(np.stack([rows_np, cols_np], axis=0))
        rhs = residual.to(device='cpu', dtype=torch.float64).reshape(-1).contiguous()

        if backend == 'mumps':
            du_flat = _MumpsSparseSolveAutograd.apply(
                indices_bc, values_bc, rhs, n, factor_handle)
        else:
            du_flat = SparseSolveAutograd.apply(indices_bc, values_bc, rhs, n)
        du = du_flat.reshape(residual.shape)
        # Move back to caller's device/dtype.
        return du.to(device=residual.device, dtype=residual.dtype)

    def _linear_solve(self, u, d, rhs, free_mask, backend_used,
                      use_autograd_tangent, factor_handle=None,
                      n_dof_free: int | None = None):
        """Solve the Newton linear system with backend fallback."""
        n_dof_free = int(free_mask.sum().item()) if n_dof_free is None else int(
            n_dof_free)
        candidates = self._backend_candidates(n_dof_free, backend_used)
        last_error = None
        for i, backend in enumerate(candidates):
            self.last_backend = backend
            try:
                if backend in ('scipy', 'mumps'):
                    du = self._scipy_newton_step(
                        u, d, rhs, free_mask, backend=backend,
                        factor_handle=factor_handle)
                else:
                    du = self._cg_solve(
                        u, d, rhs, free_mask, use_autograd_tangent)
                self.last_failure = None
                return du
            except Exception as exc:
                last_error = exc
                self.last_failure = f"{backend} failed: {exc}"
                if i + 1 >= len(candidates):
                    raise
                next_backend = candidates[i + 1]
                warnings.warn(
                    f"QuasiStaticSolver {backend_used} attempt failed in Newton "
                    f"linear solve ({exc}); retrying with {next_backend}.",
                    RuntimeWarning, stacklevel=2)
        if last_error is None:
            raise RuntimeError("All requested linear solvers failed.")
        raise last_error

    def _tangent_action(self, u, d, direction, free_mask,
                        use_autograd_tangent):
        if not use_autograd_tangent:
            out = self.fem.internal_force(direction, d)
        else:
            u_lin = u.detach()
            d_lin = d.detach()

            def _f(uu):
                return self.fem.internal_force(uu, d_lin)

            _, out = torch.autograd.functional.jvp(_f, (u_lin,), (direction,))
        return out * free_mask

    def solve_arc_length_dirichlet(
        self,
        d,
        f_ext_ref,
        bc_mask,
        bc_unit_vals,
        *,
        u_prev,
        lambda_prev: float,
        lambda_init: float,
        ds: float,
        alpha: float = 1.0,
        u_init=None,
        rigid_connectors=None,
    ):
        """Solve equilibrium with load factor as an unknown.

        This is a Crisfield/Riks-style augmented Newton corrector for the
        load-factor-scaled quasistatic benchmark form used in the standalone
        drivers:

            R(u, lambda) = lambda f_ext_ref - f_int(u, d) = 0
            mean(||u - u_prev||^2) + (alpha (lambda-lambda_prev))^2 = ds^2

        Dirichlet values are scaled by ``lambda`` through ``bc_unit_vals``.
        The derivative of the free-DOF residual with respect to ``lambda`` is
        ``f_ext_ref - K_t u_hat``, where ``u_hat`` is the unit prescribed
        boundary displacement field.  This lets displacement-controlled SENT,
        SENS, and TPB runs pass the peak without prescribing the next load
        factor outside the mechanics solve.

        The implementation intentionally excludes rotation-free rigid
        connectors for now; their reduced q-space tangent needs a separate
        augmented block.
        """
        if rigid_connectors:
            raise NotImplementedError(
                "Arc-length quasistatic solve does not yet support "
                "rotation-free rigid_connector MPC.")
        if ds <= 0.0:
            raise ValueError(f"arc-length radius ds must be > 0, got {ds}")
        if alpha <= 0.0:
            raise ValueError(
                f"arc-length load scaling alpha must be > 0, got {alpha}")

        energy_split = getattr(self.fem.material, 'energy_split', 'isotropic')
        spectral_split = energy_split in (
            'spectral', 'spectral_plane_stress_condensed',
            'amor', 'star_convex')
        use_autograd_tangent = (
            False if (self.consistent_tangent is False and spectral_split)
            else spectral_split
        )

        free_mask = (~bc_mask).to(dtype=d.dtype, device=d.device)
        n_dof_free = int(free_mask.sum().item())
        backend_used = self._resolve_backend(n_dof_free)
        if spectral_split and backend_used in ('scipy', 'mumps'):
            use_autograd_tangent = False
        self._log_backend_choice(
            backend_used, energy_split, n_dof_free, use_autograd_tangent)

        lambda_prev = float(lambda_prev)
        direction = 1.0 if (float(lambda_init) - lambda_prev) >= 0.0 else -1.0
        lam = float(lambda_init)
        if u_init is not None:
            u = u_init.clone()
        else:
            u = u_prev.clone()
        u_seed = u.clone()
        bc_unit_vals = bc_unit_vals.to(device=u.device, dtype=u.dtype)
        bc_mask = bc_mask.to(device=u.device)
        f_ext_ref = f_ext_ref.to(device=u.device, dtype=u.dtype)
        u_prev = u_prev.to(device=u.device, dtype=u.dtype)
        u[bc_mask] = (lam * bc_unit_vals)[bc_mask]

        # Directional field for prescribed displacement changes. It is zero
        # on free DOFs and equals the unit boundary displacement on fixed DOFs.
        u_hat = torch.zeros_like(u)
        u_hat[bc_mask] = bc_unit_vals[bc_mask]

        mumps_handle = None
        if backend_used == 'mumps':
            from .sparse_solve import make_factor_handle
            mumps_handle = make_factor_handle()

        ds = float(ds)
        alpha_l = float(alpha)
        # Use an RMS displacement norm rather than a raw global L2 norm.
        # Otherwise the same physical arc radius shrinks with mesh
        # refinement, which is exactly what a mesh-convergence continuation
        # study must avoid.
        disp_weight = 1.0 / max(1, int(u.numel()))
        predictor_norm = (
            disp_weight * float((u_hat * u_hat).sum().item())
            + alpha_l * alpha_l
        ) ** 0.5
        if predictor_norm > 0.0 and torch.allclose(
                u_seed, u_prev, rtol=0.0, atol=1e-30):
            lam = lambda_prev + direction * ds / predictor_norm
            u[bc_mask] = (lam * bc_unit_vals)[bc_mask]
        converged = False
        last_norm = float('inf')
        self.last_iter = 0
        residual0 = None

        for nr_iter in range(self.max_iter):
            self.last_iter = nr_iter + 1
            f_ext = lam * f_ext_ref
            residual = f_ext - self.fem.internal_force(u, d)
            residual *= free_mask
            res_norm = float(residual.norm().item())
            if residual0 is None:
                residual0 = res_norm
            du_total = u - u_prev
            dlambda = lam - lambda_prev
            constraint = (
                disp_weight * float((du_total * du_total).sum().item())
                + (alpha_l * dlambda) ** 2
                - ds * ds
            )
            aug_norm = (res_norm * res_norm + constraint * constraint) ** 0.5
            self._update_residual_diagnostics(res_norm, residual0)
            self.last_arc_length_residual = res_norm
            self.last_arc_length_constraint = float(constraint)
            self.last_load_factor = float(lam)

            if self._has_converged(
                residual_norm=res_norm,
                residual0=residual0,
                constraint=constraint,
                tol=self.tol,
                tol_rel=self.tol_rel
            ):
                converged = True
                self.last_iter = nr_iter
                self.last_line_search_alpha = 1.0
                self.last_line_search_reductions = 0
                return u, lam, True, nr_iter

            q_lambda = f_ext_ref - self._tangent_action(
                u, d, u_hat, free_mask, use_autograd_tangent)
            q_lambda *= free_mask
            du_bar = self._linear_solve(
                u, d, residual, free_mask, backend_used,
                use_autograd_tangent, factor_handle=mumps_handle)
            du_hat = self._linear_solve(
                u, d, q_lambda, free_mask, backend_used,
                use_autograd_tangent, factor_handle=mumps_handle)
            du_hat_total = du_hat + u_hat

            # Linearized constraint:
            # 2 Δu·(du_bar + dλ (du_hat + u_hat)) + 2 α² Δλ dλ = -g.
            # ``du_hat`` is the free-DOF equilibrium response to lambda;
            # ``u_hat`` is the prescribed-DOF lambda direction that gets
            # imposed after each trial update.
            denom = (
                2.0 * disp_weight * float(
                    (du_total * du_hat_total).sum().item())
                + 2.0 * alpha_l * alpha_l * dlambda
            )
            numer = (
                -constraint
                - 2.0 * disp_weight * float((du_total * du_bar).sum().item())
            )
            if abs(denom) < 1e-30:
                # At the first corrector the linearized arc constraint can
                # be orthogonal to the load direction. Fall back to the
                # predictor sign instead of taking a numerically huge update.
                dlam_corr = 0.0
            else:
                dlam_corr = numer / denom

            step_u = du_bar + dlam_corr * du_hat
            step_lam = float(dlam_corr)

            if not self.line_search:
                u = u + step_u
                lam += step_lam
                u[bc_mask] = (lam * bc_unit_vals)[bc_mask]
                self.last_line_search_alpha = 1.0
                self.last_line_search_reductions = 0
                continue

            # Residual/constraint backtracking. This is not an energy line
            # search; it only damps pathological augmented Newton jumps.
            beta = 1.0
            best = None
            best_norm = float('inf')
            reductions = 0
            for ls_iter in range(max(1, self.line_search_max_steps + 1)):
                u_trial = u + beta * step_u
                lam_trial = lam + beta * step_lam
                u_trial[bc_mask] = (lam_trial * bc_unit_vals)[bc_mask]
                r_trial = lam_trial * f_ext_ref - self.fem.internal_force(
                    u_trial, d)
                r_trial *= free_mask
                g_trial_u = u_trial - u_prev
                g_trial_l = lam_trial - lambda_prev
                g_trial = (
                    disp_weight * float((g_trial_u * g_trial_u).sum().item())
                    + (alpha_l * g_trial_l) ** 2
                    - ds * ds
                )
                trial_res_norm = float(r_trial.norm().item())
                trial_aug_norm = (
                    trial_res_norm * trial_res_norm + g_trial * g_trial
                ) ** 0.5
                if torch.isfinite(r_trial).all() and trial_aug_norm < best_norm:
                    best = (u_trial, lam_trial, trial_res_norm, g_trial)
                    best_norm = trial_aug_norm
                if trial_aug_norm <= max(
                        (1.0 - self.line_search_c * beta) * aug_norm,
                        self.tol):
                    best = (u_trial, lam_trial, trial_res_norm, g_trial)
                    best_norm = trial_aug_norm
                    break
                beta *= 0.5
                reductions = ls_iter + 1
                if beta < self.line_search_min_alpha:
                    break

            if best is None:
                u = u + step_u
                lam += step_lam
                u[bc_mask] = (lam * bc_unit_vals)[bc_mask]
                self.last_line_search_alpha = 1.0
                self.last_line_search_reductions = 0
            else:
                u, lam, res_norm, constraint = best
                self._update_residual_diagnostics(res_norm, residual0)
                self.last_arc_length_residual = float(res_norm)
                self.last_arc_length_constraint = float(constraint)
                self.last_load_factor = float(lam)
                self.last_line_search_alpha = float(beta)
                self.last_line_search_reductions = reductions

            if best_norm > 1e12 * max(last_norm, 1e-30):
                break
            last_norm = min(last_norm, best_norm)

        return u, lam, converged, self.max_iter

    def solve(self, d, f_ext, bc_mask, bc_vals, u_init=None,
              rigid_connectors=None):
        """Solve nonlinear equilibrium: f_int(u, d) = f_ext.

        Uses displacement-controlled NR: apply BCs, iterate on residual.

        Parameters
        ----------
        d : (N,) damage field (fixed during this solve).
        f_ext : (N, 2) external forces.
        bc_mask : (N, 2) bool
        bc_vals : (N, 2) float
        u_init : (N, 2) or None
        rigid_connectors : list of RigidConnector, optional
            Rotation-free rigid connectors enforced via master-slave
            elimination on the Newton iterate (#260, mirrors
            ``SecantCGSolver._solve_impl_mpc`` and
            ``DirectSolver._solve_impl_mpc``). When supplied the inner
            CG solve runs in the reduced q-space (size 2N + n_rc) with
            the matvec wrapped as ``T^T (K (T v))``. The sparse-direct
            backend assembles ``T^T K T`` explicitly. Both paths exit
            after the linear (1-step) solve when ``energy_split`` is
            ``'isotropic'``.

        Returns
        -------
        u : (N, 2) equilibrium displacement
        converged : bool
        n_iter : int
        """
        element_type = str(getattr(self.fem.mesh, "element_type", "T3")).upper()
        if self.plasticity_operator is not None:
            if element_type == "Q4":
                raise NotImplementedError(
                    "Q4 plasticity mechanics is not implemented in the "
                    "current sparse J2 backend; convert to triangles or use "
                    "elastic Q4 mechanics.")
            if rigid_connectors:
                raise NotImplementedError(
                    "J2 plasticity with rigid connectors is not implemented "
                    "yet; remove rigid_connectors or use elastic mechanics.")
            if self.backend == 'cg':
                raise NotImplementedError(
                    "J2 plasticity requires sparse tangent assembly; use "
                    "backend='auto', 'scipy', 'mumps', or 'cudss'.")
            if self._plasticity_solver is None:
                from ..plasticity import SparseJ2QuasiStaticSolver
                self._plasticity_solver = SparseJ2QuasiStaticSolver(
                    self.plasticity_operator,
                    tol=self.tol,
                    tol_rel=self.tol_rel if self.tol_rel is not None else 1e-6,
                    max_iter=self.max_iter,
                    line_search=self.line_search,
                    line_search_max_steps=self.line_search_max_steps,
                    line_search_min_alpha=self.line_search_min_alpha,
                    backend=self.backend,
                )
            u_out, converged, n_iter = self._plasticity_solver.solve(
                bc_mask, bc_vals, f_ext=f_ext, d=d, u_init=u_init)
            self.last_iter = self._plasticity_solver.last_iter
            self.last_residual = self._plasticity_solver.last_residual
            self.last_residual0 = float(getattr(
                self._plasticity_solver, 'last_residual0', float('nan')))
            self.last_relative_residual = float(getattr(
                self._plasticity_solver, 'last_relative_residual',
                float('nan')))
            self.last_line_search_alpha = (
                self._plasticity_solver.last_line_search_alpha)
            self.last_line_search_reductions = (
                self._plasticity_solver.last_line_search_reductions)
            self.last_failure = self._plasticity_solver.last_failure
            self.last_backend = self._plasticity_solver.last_backend
            return u_out, converged, n_iter

        if self.cohesive_operator is not None:
            if element_type == "Q4":
                raise NotImplementedError(
                    "Q4 cohesive mechanics is not implemented in the current "
                    "cohesive sparse backend; convert to triangles before "
                    "using cohesive interfaces.")
            if self.backend == 'cudss':
                raise NotImplementedError(
                    "backend='cudss' is currently supported only for sparse "
                    "J2 plasticity_operator solves. Cohesive sparse mechanics "
                    "supports backend='auto', 'scipy', or 'mumps'.")
            return self._solve_cohesive_sparse(
                d, f_ext, bc_mask, bc_vals, u_init=u_init,
                rigid_connectors=rigid_connectors)

        energy_split = getattr(self.fem.material, 'energy_split', 'isotropic')
        if self.backend == 'cudss':
            raise NotImplementedError(
                "backend='cudss' is currently supported only for sparse J2 "
                "plasticity_operator solves. Elastic phase-field mechanics "
                "supports backend='auto', 'scipy', 'mumps', or 'cg'.")

        mesh = self.fem.mesh
        if u_init is not None:
            u = u_init.clone()
        else:
            u = torch.zeros(
                mesh.n_nodes, 2, dtype=mesh.dtype, device=mesh.device)
        u[bc_mask] = bc_vals[bc_mask]

        free_mask = (~bc_mask).to(u.dtype)

        n_dof_free = int(free_mask.sum().item())
        backend_used = self._resolve_backend(n_dof_free)
        # For 'isotropic' split, sigma(eps) is linear in u, so internal_force
        # itself is the tangent action and we use the cheap forward-only path.
        #
        # For 'spectral'/'amor'/'star_convex', the stress is piecewise linear
        # in u and the secant operator (frozen eigenvalue signs and projectors)
        # is *not* the consistent tangent -- it omits the eigenvector-rotation
        # term that arises when projectors P_i depend on eps. We use an
        # autograd JVP K @ du = d(internal_force)/du . du as the consistent
        # tangent for the inner CG (issue #114, PR #170). The C^1 ridge floor
        # _spectral_eps in the algebraic spectral decomposition keeps the JVP
        # well-defined at coincident eigenvalues.
        spectral_split = energy_split in (
            'spectral', 'spectral_plane_stress_condensed',
            'amor', 'star_convex')
        # consistent_tangent semantics (#260):
        #   None / True  → autograd-JVP consistent tangent on spectral splits
        #                  (preserves PR #170 default behaviour)
        #   False        → opt-in secant (frozen-state) fallback on spectral
        # Isotropic: tangent is exact, the flag is a no-op.
        if self.consistent_tangent is False and spectral_split:
            use_autograd_tangent = False
        else:
            use_autograd_tangent = spectral_split

        if spectral_split and backend_used in ('scipy', 'mumps'):
            use_autograd_tangent = False
        self._log_backend_choice(
            backend_used, energy_split, n_dof_free, use_autograd_tangent)

        # ------------------------------------------------------------
        # MPC dispatch (#260): rotation-free rigid_connector path.
        # Mirrors SecantCGSolver._solve_impl_mpc / DirectSolver MPC.
        # ------------------------------------------------------------
        if rigid_connectors:
            return self._solve_mpc(u, d, f_ext, bc_mask, bc_vals, free_mask,
                                   rigid_connectors, backend_used,
                                   use_autograd_tangent, energy_split)

        # Reuse PETSc Mat/KSP across Newton iterations when the sparsity
        # pattern is fixed. The handle is per solve because active DOFs can
        # change across load steps or boundary-condition updates.
        mumps_handle = None
        if backend_used == 'mumps':
            from .sparse_solve import make_factor_handle
            mumps_handle = make_factor_handle()

        self.last_iter = 0
        residual0 = None
        for nr_iter in range(self.max_iter):
            self.last_iter = nr_iter + 1
            residual = f_ext - self.fem.internal_force(u, d)
            residual *= free_mask
            res_norm = residual.norm().item()
            if residual0 is None:
                residual0 = res_norm
            self._update_residual_diagnostics(res_norm, residual0)

            if self._has_converged(
                residual_norm=float(res_norm),
                residual0=residual0,
                tol=self.tol,
                tol_rel=self.tol_rel
            ):
                self.last_iter = nr_iter
                return u, True, nr_iter

            if not torch.isfinite(residual).all():
                self.last_failure = "residual contains non-finite values"
                return u, False, self.max_iter

            du = self._linear_solve(
                u, d, residual, free_mask, backend_used,
                use_autograd_tangent, factor_handle=mumps_handle,
                n_dof_free=n_dof_free)
            u = self._accept_newton_step(
                u, du, d, f_ext, bc_mask, bc_vals, free_mask, res_norm)

        self.last_iter = self.max_iter
        return u, False, self.max_iter

    def _accept_newton_step(self, u, du, d, f_ext, bc_mask, bc_vals,
                            free_mask, res_norm):
        """Accept a Newton update, with residual backtracking if enabled."""
        if not self.line_search:
            u_new = u + du
            u_new[bc_mask] = bc_vals[bc_mask]
            self.last_line_search_alpha = 1.0
            self.last_line_search_reductions = 0
            return u_new

        alpha = 1.0
        best_u = None
        best_norm = float('inf')
        reductions = 0
        baseline = max(float(res_norm), 1e-30)

        for ls_iter in range(max(1, self.line_search_max_steps + 1)):
            u_trial = u + alpha * du
            u_trial[bc_mask] = bc_vals[bc_mask]
            trial_residual = f_ext - self.fem.internal_force(u_trial, d)
            trial_residual *= free_mask
            trial_norm = float(trial_residual.norm().item())
            if trial_norm < best_norm:
                best_norm = trial_norm
                best_u = u_trial
            sufficient = trial_norm <= (1.0 - self.line_search_c * alpha) * baseline
            if torch.isfinite(trial_residual).all() and sufficient:
                self._update_residual_diagnostics(trial_norm, res_norm)
                self.last_line_search_alpha = float(alpha)
                self.last_line_search_reductions = reductions
                return u_trial
            alpha *= 0.5
            reductions = ls_iter + 1
            if alpha < self.line_search_min_alpha:
                break

        if best_u is not None and best_norm < baseline:
            self._update_residual_diagnostics(best_norm, res_norm)
            self.last_line_search_alpha = float(alpha)
            self.last_line_search_reductions = reductions
            return best_u

        u_full = u + du
        u_full[bc_mask] = bc_vals[bc_mask]
        self.last_line_search_alpha = 1.0
        self.last_line_search_reductions = 0
        return u_full

    def _solve_cohesive_sparse(self, d, f_ext, bc_mask, bc_vals, *,
                               u_init=None, rigid_connectors=None):
        """Sparse Newton solve for elastic bulk plus cohesive interfaces."""

        if rigid_connectors:
            raise NotImplementedError(
                "Cohesive interfaces with rigid connectors are not implemented")
        if getattr(self.fem.material, "energy_split", "isotropic") != "isotropic":
            raise NotImplementedError(
                "Cohesive sparse integration currently supports "
                "energy_split='isotropic' only")
        if self.backend == "cg":
            raise NotImplementedError(
                "Cohesive interfaces currently require sparse-direct backend")

        from .sparse_solve import resolve_sparse_backend, solve as sparse_solve

        mesh = self.fem.mesh
        if u_init is None:
            u = torch.zeros(
                mesh.n_nodes, 2, dtype=mesh.dtype, device=mesh.device)
        else:
            u = u_init.clone()
        u[bc_mask] = bc_vals[bc_mask]
        free_mask = (~bc_mask).to(mesh.dtype)
        free = free_mask.reshape(-1).to(dtype=torch.bool)
        residual0 = None
        self.last_failure = None

        def cohesive_residual(candidate_u, *, mutate_cohesive_state: bool):
            f_int_trial = self.fem.internal_force(candidate_u, d)
            if mutate_cohesive_state:
                f_coh_trial = self.cohesive_operator.internal_force(candidate_u)
            else:
                f_coh_trial = self.cohesive_operator.internal_force(
                    candidate_u, state=self.cohesive_operator.state)
            return (f_ext - f_int_trial - f_coh_trial) * free_mask

        def accept_cohesive_step(current_u, du_step, baseline_norm):
            if not self.line_search:
                u_full = current_u + du_step
                u_full[bc_mask] = bc_vals[bc_mask]
                self.last_line_search_alpha = 1.0
                self.last_line_search_reductions = 0
                return u_full

            alpha = 1.0
            best_u = None
            best_norm = float("inf")
            reductions = 0
            baseline = max(float(baseline_norm), 1.0e-30)

            for ls_iter in range(max(1, self.line_search_max_steps + 1)):
                u_trial = current_u + alpha * du_step
                u_trial[bc_mask] = bc_vals[bc_mask]
                trial_residual = cohesive_residual(
                    u_trial, mutate_cohesive_state=False)
                trial_norm = float(trial_residual.norm().item())
                if torch.isfinite(trial_residual).all() and trial_norm < best_norm:
                    best_norm = trial_norm
                    best_u = u_trial
                sufficient = (
                    trial_norm
                    <= (1.0 - self.line_search_c * alpha) * baseline
                )
                if torch.isfinite(trial_residual).all() and sufficient:
                    self.last_residual = trial_norm
                    self.last_line_search_alpha = float(alpha)
                    self.last_line_search_reductions = reductions
                    if hasattr(self.cohesive_operator, "rollback"):
                        self.cohesive_operator.rollback()
                    return u_trial
                alpha *= 0.5
                reductions = ls_iter + 1
                if alpha < self.line_search_min_alpha:
                    break

            if best_u is not None and best_norm < baseline:
                self.last_residual = best_norm
                self.last_line_search_alpha = float(alpha)
                self.last_line_search_reductions = reductions
                if hasattr(self.cohesive_operator, "rollback"):
                    self.cohesive_operator.rollback()
                return best_u

            u_full = current_u + du_step
            u_full[bc_mask] = bc_vals[bc_mask]
            self.last_line_search_alpha = 1.0
            self.last_line_search_reductions = 0
            if hasattr(self.cohesive_operator, "rollback"):
                self.cohesive_operator.rollback()
            return u_full

        for nr_iter in range(self.max_iter):
            self.last_iter = nr_iter + 1
            residual = cohesive_residual(u, mutate_cohesive_state=True)
            res_norm = float(residual.norm().item())
            if residual0 is None:
                residual0 = res_norm
            self._update_residual_diagnostics(res_norm, residual0)
            if self._has_converged(
                    residual_norm=res_norm, residual0=residual0,
                    tol=self.tol, tol_rel=self.tol_rel):
                if hasattr(self.cohesive_operator, "commit"):
                    self.cohesive_operator.commit()
                self.last_iter = nr_iter
                return u, True, nr_iter
            if not torch.isfinite(residual).all():
                self.last_failure = "cohesive residual contains non-finite values"
                if hasattr(self.cohesive_operator, "rollback"):
                    self.cohesive_operator.rollback()
                return u, False, self.max_iter

            idx, vals, n_dof = self._assemble_K_isotropic(d)
            K_bulk = torch.sparse_coo_tensor(
                idx.to(device=mesh.device),
                vals.to(device=mesh.device, dtype=mesh.dtype),
                (n_dof, n_dof),
                device=mesh.device,
                dtype=mesh.dtype,
            ).coalesce()
            K = (K_bulk + self.cohesive_operator.assemble_tangent(u)).coalesce()
            indices = K.indices()
            values = K.values()
            keep = free.to(device=indices.device)[indices[0]] & free.to(
                device=indices.device)[indices[1]]
            values_bc = values * keep.to(dtype=values.dtype)
            fixed = torch.nonzero(
                ~free.to(device=indices.device), as_tuple=False).reshape(-1)
            if fixed.numel() > 0:
                indices = torch.cat([indices, torch.stack([fixed, fixed])], dim=1)
                values_bc = torch.cat([
                    values_bc,
                    torch.ones(
                        fixed.numel(), device=values.device, dtype=values.dtype),
                ])
            K_bc = torch.sparse_coo_tensor(
                indices, values_bc, (n_dof, n_dof),
                device=mesh.device, dtype=mesh.dtype).coalesce()
            rhs = residual.reshape(-1)
            backend = resolve_sparse_backend(
                self.backend, device_type=K_bc.device.type)
            self.last_backend = backend
            du = sparse_solve(K_bc, rhs, backend=backend).reshape_as(u)
            u = accept_cohesive_step(u, du, res_norm)

        self.last_failure = "maximum iterations reached"
        if hasattr(self.cohesive_operator, "rollback"):
            self.cohesive_operator.rollback()
        return u, False, self.max_iter

    def _cg_solve(self, u, d, rhs, free_mask, use_autograd_tangent=False):
        """Inner CG solve for Newton direction.

        For 'isotropic' split (use_autograd_tangent=False) internal_force
        is linear in u, so K(d) @ du = internal_force(du, d) and CG is exact.

        For 'spectral'/'amor'/'star_convex' (use_autograd_tangent=True)
        the matvec is K @ du = JVP(internal_force(., d), u, du), the true
        consistent tangent at u via autograd. Constant within this CG solve
        (Jacobian is fixed at u), so CG conjugacy is preserved.
        """
        if not use_autograd_tangent:
            def Kd_matvec(du):
                out = self.fem.internal_force(du, d)
                out *= free_mask
                return out
        else:
            # Detach u from any outer graph; JVP creates its own.
            u_lin = u.detach()
            d_lin = d.detach()

            def _f(uu):
                return self.fem.internal_force(uu, d_lin)

            def Kd_matvec(du):
                _, jvp_out = torch.autograd.functional.jvp(
                    _f, (u_lin,), (du,))
                jvp_out = jvp_out * free_mask
                return jvp_out

        du = torch.zeros_like(rhs)
        r = rhs.clone()
        p = r.clone()
        rr = (r * r).sum()

        cg_tol_sq = self.cg_tol ** 2
        check_every = 50
        rr_0 = rr.item()
        self.last_cg_breakdown = False
        
        for i in range(self.cg_max_iter):
            Ap = Kd_matvec(p)
            pAp = (p * Ap).sum()
            if (not torch.isfinite(pAp)) or float(pAp.item()) <= 1e-30:
                self.last_cg_breakdown = True
                return self._krylov_fallback_solve(Kd_matvec, rhs, free_mask)
            alpha = rr / (pAp + 1e-30)
            du.add_(alpha * p)
            r.sub_(alpha * Ap)
            rr_new = (r * r).sum()
            p.mul_(rr_new / (rr + 1e-30)).add_(r)
            rr = rr_new
            
            if (i + 1) % check_every == 0:
                if rr.item() < cg_tol_sq:
                    break
                if rr.item() > 1e12 * rr_0:
                    break

        return du

    def _krylov_fallback_solve(self, matvec, rhs, free_mask):
        """Fallback for indefinite/singular tangents where CG breaks down."""
        try:
            import numpy as np
            from scipy.sparse.linalg import LinearOperator, gmres, minres
        except Exception as exc:
            raise RuntimeError(
                "Matrix-free CG detected an indefinite/non-finite tangent, "
                "but SciPy MINRES/GMRES is unavailable for fallback."
            ) from exc

        shape = rhs.shape
        dtype = rhs.dtype
        device = rhs.device
        rhs_flat = rhs.detach().cpu().to(torch.float64).numpy().reshape(-1)
        free_flat = (
            free_mask.detach().cpu().reshape(-1).to(torch.bool).numpy()
        )
        n_free = int(free_flat.sum())
        if n_free == 0:
            self.last_krylov_solver = 'none'
            self.last_krylov_info = 0
            return torch.zeros_like(rhs)
        rhs_np = rhs_flat[free_flat]

        def _mv(x_np):
            full_np = np.zeros(rhs_flat.shape, dtype=np.float64)
            full_np[free_flat] = np.asarray(x_np, dtype=np.float64)
            x = torch.from_numpy(full_np).to(
                device=device, dtype=dtype).reshape(shape)
            y = matvec(x) * free_mask
            y_np = y.detach().cpu().to(torch.float64).numpy().reshape(-1)
            return y_np[free_flat]

        op = LinearOperator((n_free, n_free), matvec=_mv, dtype=np.float64)
        tol = max(float(self.cg_tol), 1e-10)
        self.last_krylov_solver = 'minres'
        try:
            try:
                sol, info = minres(
                    op, rhs_np, rtol=tol, maxiter=self.cg_max_iter)
            except TypeError:
                sol, info = minres(
                    op, rhs_np, tol=tol, maxiter=self.cg_max_iter)
        except Exception:
            sol, info = None, 1
        if (
            info != 0
            or sol is None
            or not np.all(np.isfinite(np.asarray(sol)))
        ):
            self.last_krylov_solver = 'gmres'
            try:
                sol, info = gmres(
                    op, rhs_np, rtol=tol, atol=0.0,
                    restart=min(200, max(20, n_free)),
                    maxiter=self.cg_max_iter)
            except TypeError:
                sol, info = gmres(
                    op, rhs_np, tol=tol, atol=0.0,
                    restart=min(200, max(20, n_free)),
                    maxiter=self.cg_max_iter)
        if sol is None or not np.all(np.isfinite(np.asarray(sol))):
            raise RuntimeError(
                "Matrix-free CG fallback failed: MINRES/GMRES returned a "
                "non-finite correction.")
        self.last_krylov_info = int(info)
        full_sol = np.zeros(rhs_flat.shape, dtype=np.float64)
        full_sol[free_flat] = np.asarray(sol, dtype=np.float64)
        return torch.from_numpy(full_sol).to(device=device, dtype=dtype).reshape(shape)

    # ------------------------------------------------------------------
    # Rotation-free rigid_connector MPC path (#260)
    # ------------------------------------------------------------------
    def _solve_mpc(self, u, d, f_ext, bc_mask, bc_vals, free_mask,
                   rcs, backend_used, use_autograd_tangent, energy_split):
        """Newton solve with rotation-free rigid_connector master-slave MPC.

        Reduced primary variable q = [u_flat; theta_*] of size 2N + n_rc.
        - CG path: matvec wraps ``T^T (K (T v))`` with K from
          ``internal_force`` (linear in u for isotropic) or autograd-JVP
          (consistent tangent on spectral splits).
        - Sparse-direct path: assemble K, build ``T^T K T`` explicitly,
          drop fixed q-DOFs, factor + back-substitute.

        Returns ``(u, converged, n_iter)`` matching the welded path.
        """
        import numpy as np

        # Build T (master-slave) once per solve.
        T_csr, free_q_mask, free_q_idx, n_dof, n_rc, slave_dof_set = \
            _build_rigid_connector_T(rcs, self.fem, bc_mask)

        torch_dtype = u.dtype
        torch_device = u.device

        # Initial slave reconstruction (theta starts at 0).
        theta_vals = np.zeros(n_rc, dtype=np.float64)
        u_flat = u.detach().cpu().numpy().astype(np.float64).reshape(-1)
        _reconstruct_slaves_from_q(u_flat, rcs, theta_vals, self.fem)
        u = torch.from_numpy(u_flat.reshape(-1, 2)).to(
            dtype=torch_dtype, device=torch_device)

        def _to_full_torch(v_red_np):
            v_full = T_csr @ v_red_np
            return torch.from_numpy(v_full.reshape(-1, 2)).to(
                dtype=torch_dtype, device=torch_device)

        def _to_red_np(v_full_torch):
            v_full = v_full_torch.detach().cpu().numpy().astype(
                np.float64).reshape(-1)
            return T_csr.T @ v_full

        # K-matvec on the FULL space (consistent tangent or linear K).
        if use_autograd_tangent:
            u_lin_holder = [u.detach()]
            d_lin = d.detach()

            def _f(uu):
                return self.fem.internal_force(uu, d_lin)

            def K_full_matvec(p_full):
                _, jvp = torch.autograd.functional.jvp(
                    _f, (u_lin_holder[0],), (p_full,))
                return jvp
        else:
            def K_full_matvec(p_full):
                return self.fem.internal_force(p_full, d)

        def _matvec_red(p_red):
            p_full = _to_full_torch(p_red)
            Kp_full = K_full_matvec(p_full)
            Kp_full = Kp_full.masked_fill(bc_mask, 0.0)
            return _to_red_np(Kp_full)

        def _residual_red(u_cur):
            f_int = self.fem.internal_force(u_cur, d)
            r_full = (f_ext - f_int) if f_ext is not None else -f_int
            r_full = r_full.masked_fill(bc_mask, 0.0)
            r_q = _to_red_np(r_full)
            r_q[~free_q_mask] = 0.0
            return r_q

        residual0 = None
        for nr_iter in range(self.max_iter):
            if use_autograd_tangent:
                u_lin_holder[0] = u.detach()
            r_q = _residual_red(u)
            res_norm = float(np.linalg.norm(r_q[free_q_idx]))
            if residual0 is None:
                residual0 = res_norm
            self._update_residual_diagnostics(res_norm, residual0)
            if self._has_converged(
                residual_norm=float(res_norm),
                residual0=residual0,
                tol=self.tol,
                tol_rel=self.tol_rel
            ):
                self._record_mpc_diagnostics(rcs, theta_vals, u, f_ext, d)
                return u, True, nr_iter

            # ---- Newton step in reduced space ----
            if backend_used in ('scipy', 'mumps') and energy_split == 'isotropic':
                # Assemble K (isotropic only — tangent is exact and
                # constant in u), then form T^T K T explicitly and call
                # the sparse-direct autograd Function.
                try:
                    dq_acc = self._mpc_sparse_direct_step(
                        d, T_csr, free_q_mask, n_dof, n_rc, r_q,
                        backend=backend_used)
                    self.last_failure = None
                except Exception as exc:
                    # One robust fallback from mumps/scipy to MPC-CG.
                    self.last_failure = f"{backend_used} failed: {exc}"
                    warnings.warn(
                        f"QuasiStaticSolver {backend_used} failed in MPC sparse "
                        f"solve ({exc}); retrying MPC-CG.",
                        RuntimeWarning, stacklevel=2)
                    dq_acc = self._mpc_cg_step(
                        _matvec_red, free_q_mask, free_q_idx, n_dof, n_rc,
                        r_q, energy_split, d)
                    self.last_backend = 'cg'
                    self.last_failure = None
            else:
                # CG inner solve: Jacobi preconditioner on diag(T^T K T)
                # (#171/#189 mirror of SecantCG MPC path).
                dq_acc = self._mpc_cg_step(
                    _matvec_red, free_q_mask, free_q_idx, n_dof, n_rc, r_q,
                    energy_split, d)

            # Apply increment: theta + master translation, then reconstruct
            # slaves from master + theta to keep the kinematic constraint.
            for k in range(n_rc):
                theta_vals[k] += float(dq_acc[n_dof + k])
            du_full = T_csr @ dq_acc  # (n_dof,)
            u_flat = u.detach().cpu().numpy().astype(np.float64).reshape(-1)
            u_flat += du_full
            u_new = torch.from_numpy(u_flat.reshape(-1, 2)).to(
                dtype=torch_dtype, device=torch_device)
            u_new[bc_mask] = bc_vals[bc_mask]
            u_flat = u_new.detach().cpu().numpy().astype(
                np.float64).reshape(-1)
            _reconstruct_slaves_from_q(u_flat, rcs, theta_vals, self.fem)
            u = torch.from_numpy(u_flat.reshape(-1, 2)).to(
                dtype=torch_dtype, device=torch_device)

            # For isotropic split the tangent is exact: a single Newton
            # step solves the linear problem to round-off.
            if energy_split == 'isotropic':
                self._record_mpc_diagnostics(rcs, theta_vals, u, f_ext, d)
                return u, True, nr_iter + 1

        self._record_mpc_diagnostics(rcs, theta_vals, u, f_ext, d)
        return u, False, self.max_iter

    def _mpc_cg_step(self, matvec_red, free_q_mask, free_q_idx,
                     n_dof, n_rc, r_q, energy_split, d):
        """One reduced-space PCG solve for the Newton increment dq.

        Jacobi preconditioner on diag(T^T K T): identity-block diagonal is
        equal to the (degraded) K diagonal at non-slave DOFs; theta entries
        computed by one matvec each (n_rc is small). The Jacobi diagonal
        uses the current ``d`` field; this is exact for the isotropic
        split (where K is linear in u). For spectral splits the Jacobi
        ignores the eigenvector-rotation contribution to the consistent
        tangent and is an *approximate* preconditioner — still symmetric
        and positive at well-conditioned DOFs, so CG just takes more
        iterations near damage saturation.
        """
        import numpy as np
        # Assemble Jacobi diagonal using the live damage field. For
        # piecewise-linear splits this is the secant-state diagonal, not
        # the full consistent-tangent diagonal — see docstring.
        K_diag = self.fem.stiffness_diagonal(d).detach().cpu().numpy(
            ).astype(np.float64).reshape(-1)
        diag_q = np.zeros(n_dof + n_rc, dtype=np.float64)
        diag_q[:n_dof] = K_diag
        for k in range(n_rc):
            e_red = np.zeros(n_dof + n_rc, dtype=np.float64)
            e_red[n_dof + k] = 1.0
            Ke = matvec_red(e_red)
            diag_q[n_dof + k] = float(Ke[n_dof + k])
        diag_q[~free_q_mask] = 0.0
        floor = 1e-12 * np.abs(diag_q).max()
        sign = np.where(diag_q != 0.0, np.sign(diag_q), 1.0)
        inv_diag_q = np.where(free_q_mask,
                              1.0 / np.maximum(np.abs(diag_q), floor),
                              0.0) * sign

        r = r_q.copy()
        r[~free_q_mask] = 0.0
        z = inv_diag_q * r
        p = z.copy()
        rz = float((r * z).sum())

        dq = np.zeros_like(r)
        cg_tol_sq = self.cg_tol ** 2
        rr_0 = float((r ** 2).sum())
        check_every = 50
        for i in range(self.cg_max_iter):
            Ap = matvec_red(p)
            Ap[~free_q_mask] = 0.0
            pAp = float((p * Ap).sum())
            alpha = rz / (pAp + 1e-30)
            dq += alpha * p
            r -= alpha * Ap
            z = inv_diag_q * r
            rz_new = float((r * z).sum())
            beta = rz_new / (rz + 1e-30)
            p = beta * p + z
            rz = rz_new
            if (i + 1) % check_every == 0:
                r_sq = float((r[free_q_idx] ** 2).sum())
                if r_sq < cg_tol_sq:
                    break
                if r_sq > 1e12 * max(rr_0, 1e-30):
                    break
        return dq

    def _mpc_sparse_direct_step(self, d, T_csr, free_q_mask, n_dof, n_rc,
                                r_q, backend='scipy'):
        """One Newton step for MPC via sparse-direct on T^T K T.

        Assembles K (isotropic only — tangent is exact and constant in u),
        forms ``Kq = T^T K T``, drops fixed q-DOFs (Dirichlet zero on
        rows/cols + unit diagonal), and calls SparseSolveAutograd /
        MumpsSparseSolveAutograd.
        """
        import numpy as np
        from scipy import sparse as sp
        from .sparse_solve import (
            SparseSolveAutograd, _MumpsSparseSolveAutograd)

        candidates = self._backend_candidates(n_dof + n_rc, backend)
        if not candidates:
            candidates = [backend]
        last_error = None
        for i, backend_item in enumerate(candidates):
            self.last_backend = backend_item
            try:
                indices, values, n_dof_check = self._assemble_K_isotropic(d)
                assert int(n_dof_check) == int(n_dof)
                rows_np = indices[0].numpy()
                cols_np = indices[1].numpy()
                vals_np = values.numpy()
                K_csr = sp.coo_matrix(
                    (vals_np, (rows_np, cols_np)), shape=(n_dof, n_dof)
                ).tocsr()
                # Build Kq = T^T K T (n_q x n_q).
                Kq = (T_csr.T @ K_csr @ T_csr).tocsr()
                # Drop fixed q-DOFs: zero rows & cols, put 1 on diagonal.
                n_q = n_dof + n_rc
                keep = free_q_mask.astype(bool)
                Kq = Kq.tolil()
                # Vectorised row/col drop via mask multiply.
                D_keep = sp.diags(keep.astype(np.float64))
                Kq = (D_keep @ Kq.tocsr() @ D_keep).tolil()
                for i_fix in np.where(~keep)[0]:
                    Kq[i_fix, i_fix] = 1.0
                Kq = Kq.tocoo()
                rhs_q = r_q.copy()
                rhs_q[~keep] = 0.0
                indices_bc = torch.from_numpy(np.stack([Kq.row, Kq.col], axis=0))
                values_bc = torch.from_numpy(Kq.data.astype(np.float64))
                rhs_t = torch.from_numpy(rhs_q.astype(np.float64))
                if backend_item == 'mumps':
                    dq = _MumpsSparseSolveAutograd.apply(
                        indices_bc, values_bc, rhs_t, n_q)
                else:
                    dq = SparseSolveAutograd.apply(
                        indices_bc, values_bc, rhs_t, n_q)
                self.last_failure = None
                return dq.detach().cpu().numpy().astype(np.float64)
            except Exception as exc:
                last_error = exc
                self.last_failure = f"{backend_item} failed: {exc}"
                if i + 1 >= len(candidates):
                    break
                next_backend = candidates[i + 1]
                warnings.warn(
                    f"QuasiStaticSolver {backend} attempt failed in MPC sparse "
                    f"solve ({exc}); retrying with {next_backend}.",
                    RuntimeWarning, stacklevel=2)
                continue

        if last_error is None:
            raise RuntimeError("MPC sparse-direct solve candidates exhausted.")
        raise last_error

    def _record_mpc_diagnostics(self, rcs, theta_vals, u, f_ext, d):
        """Populate last_theta / last_master_reaction (mirrors DirectSolver)."""
        import numpy as np
        f_int = self.fem.internal_force(u, d).detach().cpu().numpy()
        if f_ext is not None:
            f_ext_np = f_ext.detach().cpu().numpy()
        else:
            f_ext_np = np.zeros_like(f_int)
        self.last_theta = list(map(float, theta_vals))
        self.last_master_reaction = []
        for rc in rcs:
            slaves = rc.slaves_excluding_master().detach().cpu().numpy()
            m = rc.master_node
            ids = np.concatenate([np.array([m]), slaves]).astype(np.int64)
            R = (f_int[ids] - f_ext_np[ids]).sum(axis=0)
            self.last_master_reaction.append(
                (int(m), float(R[0]), float(R[1])))


def _try_import_cupy():
    """Try to import cupy and verify CUDA libs work at runtime."""
    try:
        import cupy
        import cupyx.scipy.sparse as cusp
        import cupyx.scipy.sparse.linalg as cusp_linalg
        # Verify cupy can actually use CUDA (catches lib version mismatch)
        cupy.array([1.0])
        return cupy, cusp, cusp_linalg
    except (ImportError, Exception):
        return None, None, None


# Re-export for stable public API; implementation lives in a self-contained
# module so it can be imported without pulling in FEM dependencies (#118).
from .mixed_precision_cg import cg_mixed_precision  # noqa: E402,F401


def _spsolve_auto(A_csr, rhs, device):
    """Sparse direct solve using cupy (CUDA) or scipy (CPU).

    Returns numpy array of the solution. Falls back to scipy if cupy
    fails at runtime (e.g. CUDA version mismatch).
    """
    import numpy as np

    if device is not None and device.type == 'cuda':
        cupy, cusp, cusp_linalg = _try_import_cupy()
        if cupy is not None:
            try:
                dev_idx = device.index or 0
                with cupy.cuda.Device(dev_idx):
                    A_gpu = cusp.csr_matrix(
                        (cupy.array(A_csr.data),
                         cupy.array(A_csr.indices),
                         cupy.array(A_csr.indptr)),
                        shape=A_csr.shape)
                    rhs_gpu = cupy.array(rhs)
                    x_gpu = cusp_linalg.spsolve(A_gpu, rhs_gpu)
                    return cupy.asnumpy(x_gpu)
            except (ImportError, RuntimeError) as e:
                print(f"  [DirectSolver] cupy spsolve failed ({e}), "
                      f"falling back to scipy CPU", flush=True)

    from scipy.sparse.linalg import spsolve
    return spsolve(A_csr, rhs)


# ---------------------------------------------------------------------------
# Rotation-free rigid-connector MPC helpers (issue #154 / #165 / #171).
# Shared by DirectSolver (PR #164) and SecantCGSolver (PR for #171). The
# kinematic constraint is u_full = T @ q where q = [u_flat; theta_*] and T
# is (n_dof, n_dof + n_rc) sparse with identity rows for non-slave DOFs and
# the linearised rigid-body block for slave rows (see RigidConnector.build_T_block).
# ---------------------------------------------------------------------------
def _build_rigid_connector_T(rcs, fem, bc_mask_dev):
    """Assemble the master-slave T matrix and free q-DOF mask.

    Returns
    -------
    T_csr : scipy.sparse.csr_matrix, shape (n_dof, n_dof + n_rc)
    free_q_mask : np.ndarray of bool, shape (n_dof + n_rc,)
    free_q_idx : np.ndarray of int, indices where free_q_mask is True
    n_dof : int
        Original flat DOF count (2 * n_nodes).
    n_rc : int
        Number of rotation-free connectors (= number of theta DOFs).
    slave_dof_set : set[int]
        Flat-DOF indices that are slave-eliminated (kinematic constraint
        rows in T replace identity).
    """
    import numpy as np
    import scipy.sparse as sp

    n_dof = 2 * fem.mesh.n_nodes
    n_rc = len(rcs)
    n_q = n_dof + n_rc

    T_rows = list(range(n_dof))
    T_cols = list(range(n_dof))
    T_vals = [1.0] * n_dof
    slave_dof_set = set()
    bc_flat = bc_mask_dev.flatten().cpu().numpy()
    for rc_idx, rc in enumerate(rcs):
        theta_col = n_dof + rc_idx
        (rrows, rcols, rvals, slave_dofs,
         m_node, m_dofs, _) = rc.build_T_block(fem.mesh, theta_col)
        m_dx, m_dy = m_dofs
        if not (bc_flat[m_dx] or bc_flat[m_dy]):
            raise ValueError(
                "rigid_connector master node "
                f"{m_node} has neither u_x nor u_y under "
                "Dirichlet BC; rotation-free rigid connectors "
                "require the master's translation to be "
                "prescribed (this is what 'prescribe' is for).")
        for s in slave_dofs:
            slave_dof_set.add(int(s))
        T_rows.extend(rrows)
        T_cols.extend(rcols)
        T_vals.extend(rvals)

    T_rows_a = np.asarray(T_rows, dtype=np.int64)
    T_cols_a = np.asarray(T_cols, dtype=np.int64)
    T_vals_a = np.asarray(T_vals, dtype=np.float64)
    # Strip identity rows for slave DOFs (those rows now carry constraint).
    keep = np.ones(len(T_rows_a), dtype=bool)
    seeded_n = n_dof
    seed_in_slave = np.zeros(seeded_n, dtype=bool)
    for s in slave_dof_set:
        seed_in_slave[s] = True
    keep[:seeded_n] = ~seed_in_slave
    T_rows_a = T_rows_a[keep]
    T_cols_a = T_cols_a[keep]
    T_vals_a = T_vals_a[keep]
    T_csr = sp.csr_matrix((T_vals_a, (T_rows_a, T_cols_a)),
                          shape=(n_dof, n_q))

    free_q_mask = np.ones(n_q, dtype=bool)
    free_q_mask[:n_dof] = (~bc_flat) & (~np.isin(np.arange(n_dof),
                                                 list(slave_dof_set)))
    free_q_mask[n_dof:] = True
    free_q_idx = np.where(free_q_mask)[0]
    return T_csr, free_q_mask, free_q_idx, n_dof, n_rc, slave_dof_set


def _reconstruct_slaves_from_q(u_full_flat, rcs, theta_vals, fem):
    """Overwrite slave-node displacements from current master + theta.

    Parameters
    ----------
    u_full_flat : np.ndarray (n_dof,)
        Mutated in place.
    rcs, theta_vals : list of RigidConnector and per-connector theta.
    fem : FEMOperators (provides mesh.nodes for slave coordinates).
    """
    import numpy as np
    nodes_np = fem.mesh.nodes.detach().cpu().numpy()
    Xs = nodes_np[:, 0]
    Ys = nodes_np[:, 1]
    for rc_i, rc in enumerate(rcs):
        m = rc.master_node
        Xm = float(nodes_np[m, 0])
        Ym = float(nodes_np[m, 1])
        u_mx = u_full_flat[2 * m]
        u_my = u_full_flat[2 * m + 1]
        th = float(theta_vals[rc_i])
        slaves = rc.slaves_excluding_master().detach().cpu().numpy()
        for i in slaves:
            i = int(i)
            u_full_flat[2 * i]     = u_mx - th * (Ys[i] - Ym)
            u_full_flat[2 * i + 1] = u_my + th * (Xs[i] - Xm)


class DirectSolver:
    """[VALIDATION] Direct sparse solver for quasi-static equilibrium.

    Assembles the full stiffness matrix and solves via direct LU
    factorisation (equivalent in role to PhaseFieldX's MUMPS). Gives exact
    solutions to machine precision. Use for:
      - Validating SecantCG accuracy
      - Generating reference training data
      - Small-to-medium problems where accuracy > speed

    Backend selection:
      - CUDA device + cupy installed → cupyx.scipy.sparse.linalg.spsolve
        (cuSPARSE on GPU, no CPU transfer)
      - Otherwise → scipy.sparse.linalg.spsolve (CPU)

    For spectral/amor splits, uses Newton iteration with a frozen-state
    secant tangent re-assembled at each step.

    Parameters
    ----------
    fem : FEMOperators
    tol : float
        Newton residual tolerance.
    max_newton : int
        Maximum Newton iterations.
    """

    def __init__(self, fem: FEMOperators, tol: float = 1e-10,
                 max_newton: int = 20, rtol: float = 1e-8,
                 log_backend: bool = True):
        self.fem = fem
        self.tol = tol
        self.rtol = rtol
        self.max_newton = max_newton
        self.last_iter = 0
        self.last_diverged = False

        # Detect backend for spsolve
        dev = fem.device
        if isinstance(dev, str):
            dev = torch.device(dev)
        self._orig_device = dev
        self._use_gpu = (dev.type == 'cuda' and
                         _try_import_cupy()[0] is not None)
        backend = 'cupy/cuSPARSE (GPU)' if self._use_gpu else 'scipy (CPU)'
        if log_backend:
            print(f"[DirectSolver] Backend: {backend}", flush=True)

        # If the native device doesn't support float64 (e.g. MPS), create a
        # CPU float64 FEM copy for internal_force and freeze_secant_state.
        # This matches SecantCGSolver's _solve_cpu_fallback approach and gives
        # the same solution accuracy as running on CPU float64 directly.
        self._needs_cpu_fallback = not device_supports_float64(dev)
        self._cpu_fem = None
        if self._needs_cpu_fallback:
            from ..core.mesh import FEMMesh
            cpu_mesh = FEMMesh.from_tensors(
                fem.mesh.nodes, fem.mesh.elements, fem.mesh.node_sets,
                device='cpu', dtype=torch.float64)
            self._cpu_fem = FEMOperators(cpu_mesh, fem.material, ctx=None)
            print(f"[DirectSolver] MPS detected — FEM calls on CPU float64",
                  flush=True)

        # Precompute element-to-global DOF mapping for stiffness assembly
        self._precompute_assembly_indices()

        # MPC diagnostics — populated by the rotation-free rigid-connector
        # path (issue #171, #206). Mirrors SecantCGSolver.last_theta /
        # SecantCGSolver.last_master_reaction so downstream postprocess
        # code can read them solver-independently.
        self.last_theta = []
        self.last_master_reaction = []

    def _precompute_assembly_indices(self):
        """Build COO index arrays for element stiffness assembly."""
        import numpy as np
        mesh = self.fem.mesh
        elems = mesh.elements.cpu().numpy()  # (E, 3)
        n_elem = elems.shape[0]

        # Each triangle has 3 nodes × 2 DOFs = 6 DOFs per element
        elem_dofs = np.zeros((n_elem, 6), dtype=np.int64)
        for i in range(3):
            elem_dofs[:, 2*i] = 2 * elems[:, i]
            elem_dofs[:, 2*i+1] = 2 * elems[:, i] + 1

        # COO indices for all 6×6 blocks: (E*36,) arrays
        rows = np.repeat(elem_dofs, 6, axis=1).flatten()
        cols = np.tile(elem_dofs, (1, 6)).flatten()
        self._asm_rows = rows
        self._asm_cols = cols
        self._elem_dofs_np = elem_dofs
        self._n_dof = 2 * mesh.n_nodes

    def _compute_element_stiffness(self, state):
        """Compute all element stiffness matrices — fully vectorized.

        Ke[e] = area[e] * B[e]^T @ C[e] @ B[e]

        B matrices are built from grad_phi (vectorized over all elements).
        C matrices are obtained via 3 batched stress calls (one per unit
        strain component) — 3 GPU kernel launches total, not 3 per element.

        Returns (E, 6, 6) numpy array.
        """
        import numpy as np
        fem = self.fem
        mesh = fem.mesh
        dev = fem.device
        dtype = torch.float64
        n_elem = mesh.n_elems

        gp = mesh.grad_phi.detach().cpu().to(dtype).numpy()  # (E, 3, 2)
        areas = mesh.areas.detach().cpu().to(dtype).numpy()    # (E,)
        split = state['split']
        mat = fem.material
        lam, mu = mat.lam, mat.mu

        # --- Vectorized B matrices: (E, 3, 6) ---
        gpx = gp[:, :, 0]  # (E, 3)
        gpy = gp[:, :, 1]  # (E, 3)
        B = np.zeros((n_elem, 3, 6), dtype=np.float64)
        for i in range(3):
            B[:, 0, 2*i] = gpx[:, i]       # exx
            B[:, 1, 2*i+1] = gpy[:, i]     # eyy
            B[:, 2, 2*i] = gpy[:, i]       # gxy
            B[:, 2, 2*i+1] = gpx[:, i]

        # --- Vectorized C matrices: (E, 3, 3) ---
        C = self._compute_all_C(fem, state)

        # --- Ke = area * B^T @ C @ B  (vectorized einsum) ---
        CB = np.einsum('eij,ejk->eik', C, B)           # (E, 3, 6)
        Ke = np.einsum('eji,ejk->eik', B, CB)          # (E, 6, 6)
        Ke *= areas[:, None, None]

        return Ke

    def _compute_all_C(self, fem, state):
        """Compute (E, 3, 3) secant constitutive matrices for all elements.

        Uses frozen projections from freeze_secant_state to correctly
        linearize the spectral/amor/star_convex splits.  For these
        piecewise-linear splits, the tangent depends on which eigenvalue
        or trace sign regime the current strain lies in.  Using the
        frozen projections ensures the C matrix reflects the actual
        strain state, not the unit-probe strain state.

        Probes with 3 unit strain fields (exx=1, eyy=1, gxy=1) and
        applies the frozen linearized stress to fill the 3x3 C matrix.
        """
        import numpy as np
        # Always compute on CPU with float64 — MPS doesn't support float64.
        # All state tensors are moved from their original device to CPU here.
        dev = torch.device('cpu')
        dtype = torch.float64
        n_elem = fem.mesh.n_elems
        split = state['split']
        # Use two-step move (device first, then dtype) to avoid MPS float64 error.
        # PyTorch on MPS will fail if dtype=float64 conversion is attempted
        # on-device; routing through CPU first avoids this.
        g_d = state['g_d'].detach().cpu().to(dtype=dtype)

        def secant_stress_fn(exx, eyy, gxy):
            """Linearized stress using frozen projections from state."""
            if split == 'isotropic':
                # Inline to avoid fem.compute_stress_isotropic which uses fem.C on fem.device
                C_mat = fem.C.detach().cpu().to(dtype=dtype)
                sxx = g_d * (C_mat[0, 0] * exx + C_mat[0, 1] * eyy)
                syy = g_d * (C_mat[1, 0] * exx + C_mat[1, 1] * eyy)
                sxy = g_d * (C_mat[2, 2] * gxy)
                return sxx, syy, sxy

            elif split == 'amor':
                kappa = fem.material.kappa
                mu = fem.material.mu
                # Mirror fem_operators.secant_matvec amor branch (#222 /
                # PR #231): reconstruct the 3D trace so the volumetric
                # term matches compute_stress_amor. Plane-strain:
                # eps_zz = 0 ⇒ tr_3d == tr_2d (bit-identical to pre-fix).
                # Plane-stress: eps_zz = -nu/(1-nu)*(exx+eyy) introduces
                # the (1-2nu)/(1-nu) factor on tr that the 2D-trace path
                # was missing.
                tr_2d = exx + eyy
                if fem.material.plane_stress:
                    nu = fem.material.nu
                    ezz = -nu / (1.0 - nu) * tr_2d
                    tr = tr_2d + ezz
                else:
                    tr = tr_2d  # plane strain: eps_zz = 0
                tr_pos = state['trace_pos'].detach().cpu().to(dtype=dtype)
                tr_plus = tr * tr_pos
                tr_minus = tr * (1.0 - tr_pos)
                dev_xx = exx - tr / 3.0
                dev_yy = eyy - tr / 3.0
                sxx = g_d * (kappa * tr_plus + 2 * mu * dev_xx) + kappa * tr_minus
                syy = g_d * (kappa * tr_plus + 2 * mu * dev_yy) + kappa * tr_minus
                sxy = g_d * mu * gxy
                return sxx, syy, sxy

            elif split == 'spectral':
                mu = fem.material.mu
                lam = fem.material.lam
                exy = gxy / 2.0
                p1_xx = state['p1_xx'].detach().cpu().to(dtype=dtype)
                p1_yy = state['p1_yy'].detach().cpu().to(dtype=dtype)
                p1_xy = state['p1_xy'].detach().cpu().to(dtype=dtype)
                sign1 = state['sign1_pos'].detach().cpu().to(dtype=dtype)
                sign2 = state['sign2_pos'].detach().cpu().to(dtype=dtype)

                # Project test strain onto frozen eigenvectors
                e1_p = p1_xx * exx + 2.0 * p1_xy * exy + p1_yy * eyy
                e2_p = ((1.0 - p1_xx) * exx - 2.0 * p1_xy * exy +
                        (1.0 - p1_yy) * eyy)

                # Apply frozen eigenvalue signs
                e1_p_plus = e1_p * sign1
                e2_p_plus = e2_p * sign2
                e1_p_minus = e1_p * (1.0 - sign1)
                e2_p_minus = e2_p * (1.0 - sign2)
                tr_plus = e1_p_plus + e2_p_plus
                tr_minus = e1_p_minus + e2_p_minus

                # Reconstruct Cartesian eps_plus from frozen projections
                exx_plus = e1_p_plus * p1_xx + e2_p_plus * (1.0 - p1_xx)
                eyy_plus = e1_p_plus * p1_yy + e2_p_plus * (1.0 - p1_yy)
                exy_plus = (e1_p_plus - e2_p_plus) * p1_xy

                exx_minus = exx - exx_plus
                eyy_minus = eyy - eyy_plus
                exy_minus = exy - exy_plus

                sxx = (g_d * (lam * tr_plus + 2 * mu * exx_plus) +
                       (lam * tr_minus + 2 * mu * exx_minus))
                syy = (g_d * (lam * tr_plus + 2 * mu * eyy_plus) +
                       (lam * tr_minus + 2 * mu * eyy_minus))
                sxy = (g_d * (2 * mu * exy_plus) +
                       (2 * mu * exy_minus))
                return sxx, syy, sxy

            elif split == 'star_convex':
                C_mat = fem.C.detach().cpu().to(dtype=dtype)
                kappa = fem.material.kappa
                mu = fem.material.mu
                tr_2d = exx + eyy
                if fem.material.plane_stress:
                    nu = fem.material.nu
                    ezz = -nu / (1.0 - nu) * tr_2d
                    tr = tr_2d + ezz
                else:
                    tr = tr_2d
                tension = state['tension'].detach().cpu()

                sxx_t = g_d * (C_mat[0, 0] * exx + C_mat[0, 1] * eyy)
                syy_t = g_d * (C_mat[1, 0] * exx + C_mat[1, 1] * eyy)
                sxy_t = g_d * (C_mat[2, 2] * gxy)

                dev_xx = exx - tr / 3.0
                dev_yy = eyy - tr / 3.0
                sxx_c = g_d * 2 * mu * dev_xx + kappa * tr
                syy_c = g_d * 2 * mu * dev_yy + kappa * tr
                sxy_c = g_d * mu * gxy

                sxx = torch.where(tension, sxx_t, sxx_c)
                syy = torch.where(tension, syy_t, syy_c)
                sxy = torch.where(tension, sxy_t, sxy_c)
                return sxx, syy, sxy
            else:
                return fem.compute_stress_isotropic(exx, eyy, gxy, g_d)

        C = np.zeros((n_elem, 3, 3), dtype=np.float64)
        ones = torch.ones(n_elem, dtype=dtype, device=dev)
        zeros = torch.zeros(n_elem, dtype=dtype, device=dev)

        # Column 0: unit exx → [sxx, syy, sxy]
        sxx, syy, sxy = secant_stress_fn(ones, zeros, zeros)
        C[:, 0, 0] = sxx.detach().cpu().numpy()
        C[:, 1, 0] = syy.detach().cpu().numpy()
        C[:, 2, 0] = sxy.detach().cpu().numpy()

        # Column 1: unit eyy → [sxx, syy, sxy]
        sxx, syy, sxy = secant_stress_fn(zeros, ones, zeros)
        C[:, 0, 1] = sxx.detach().cpu().numpy()
        C[:, 1, 1] = syy.detach().cpu().numpy()
        C[:, 2, 1] = sxy.detach().cpu().numpy()

        # Column 2: unit gxy → [sxx, syy, sxy]
        sxx, syy, sxy = secant_stress_fn(zeros, zeros, ones)
        C[:, 0, 2] = sxx.detach().cpu().numpy()
        C[:, 1, 2] = syy.detach().cpu().numpy()
        C[:, 2, 2] = sxy.detach().cpu().numpy()

        return C

    def _assemble_stiffness(self, state):
        """Assemble global stiffness from element stiffness matrices."""
        import numpy as np
        import scipy.sparse as sp

        Ke_all = self._compute_element_stiffness(state)  # (E, 6, 6)
        vals = Ke_all.flatten()  # (E*36,)

        K_coo = sp.coo_matrix((vals, (self._asm_rows, self._asm_cols)),
                              shape=(self._n_dof, self._n_dof))
        return K_coo.tocsr()

    @torch.no_grad()
    def solve(self, u, d, bc_mask, bc_vals, f_ext=None,
              rigid_connectors=None):
        """Solve equilibrium using direct sparse factorization.

        Parameters
        ----------
        u : (N, 2) displacement (initial guess)
        d : (N,) damage field (frozen during solve)
        bc_mask : (N, 2) bool
        bc_vals : (N, 2) float
        f_ext : (N, 2) optional external forces
        rigid_connectors : list of RigidConnector, optional
            Rotation-free rigid connectors enforced via master-slave
            elimination on the direct-solve linearisation
            (issue #171 / #206). When supplied, DirectSolver assembles
            the full sparse K, projects to reduced primary variable
            ``q = [u_flat; theta_*]`` of size ``2N + n_rc`` via the
            sparse triple product ``K_q = T^T K T``, and direct-solves
            the reduced free-DOF system at each Newton iteration.
            Slave displacements are reconstructed from master + theta
            after each Newton update (see
            :func:`_reconstruct_slaves_from_q`). Mirrors
            :meth:`SecantCGSolver._solve_impl_mpc` and
            :meth:`ExplicitDynamics.step`'s kwarg signature so callers
            can swap solvers without touching the call site.

        Returns
        -------
        u : (N, 2) equilibrium displacement
        """
        import numpy as np

        if rigid_connectors:
            return self._solve_impl_mpc(u, d, bc_mask, bc_vals, f_ext,
                                        rigid_connectors)

        self.last_diverged = False
        orig_device = u.device
        orig_dtype = u.dtype

        # Route all FEM operations through CPU float64 when the native device
        # doesn't support float64 (e.g. MPS).  This matches SecantCGSolver's
        # _solve_cpu_fallback and gives PhaseFieldX-equivalent accuracy.
        if self._needs_cpu_fallback:
            fem = self._cpu_fem
            u = u.detach().cpu().to(torch.float64)
            d_dev = d.detach().cpu().to(torch.float64)
            bc_mask_dev = bc_mask.cpu()
            bc_vals_dev = bc_vals.detach().cpu().to(torch.float64)
            f_ext_dev = (f_ext.detach().cpu().to(torch.float64)
                         if f_ext is not None else None)
        else:
            fem = self.fem
            u = u.detach().clone()
            d_dev = d.detach()
            bc_mask_dev = bc_mask
            bc_vals_dev = bc_vals.detach()
            f_ext_dev = f_ext.detach() if f_ext is not None else None

        u[bc_mask_dev] = bc_vals_dev[bc_mask_dev]
        free_flat = (~bc_mask_dev).flatten().cpu().numpy()
        free_idx = np.where(free_flat)[0]

        tol_sq = self.tol ** 2
        rtol_sq = self.rtol ** 2
        nr = 0
        prev_r_norm_sq = float('inf')
        r0_norm_sq = None

        for nr in range(max(self.max_newton, 1)):
            # True nonlinear residual
            f_int = fem.internal_force(u, d_dev)
            r = (f_ext_dev - f_int) if f_ext_dev is not None else -f_int
            r[bc_mask_dev] = 0.0

            r_norm_sq = (r * r).sum().item()
            if r0_norm_sq is None:
                r0_norm_sq = max(r_norm_sq, 1e-30)

            # Converged: absolute or relative tolerance
            if r_norm_sq < tol_sq or r_norm_sq < rtol_sq * r0_norm_sq:
                self.last_iter = nr
                return u.to(dtype=orig_dtype, device=orig_device)

            # Stop if residual is no longer decreasing (precision floor reached).
            # On float32 devices (MPS), the residual stalls at ~7e-5 after the
            # first Newton step.  Continuing just adds noise from the float32
            # floor back into u via spsolve, corrupting the displacement field.
            if nr > 0 and r_norm_sq >= prev_r_norm_sq * 0.999:
                self.last_iter = nr
                return u.to(dtype=orig_dtype, device=orig_device)
            prev_r_norm_sq = r_norm_sq

            # Assemble stiffness (re-linearize every Newton step)
            state = fem.freeze_secant_state(u, d_dev)
            K_csr = self._assemble_stiffness(state)

            # Apply BCs: extract free-DOF submatrix
            K_free = K_csr[np.ix_(free_idx, free_idx)]
            rhs = r.flatten().detach().cpu().double().numpy()[free_idx]

            # Direct solve (cupy on CUDA, scipy on CPU)
            du_free = _spsolve_auto(K_free, rhs,
                                    self._orig_device if self._use_gpu else None)

            # Update displacement. Convert du to the same dtype/device as u
            # (CPU float64 in fallback mode, orig_dtype/device otherwise).
            du = np.zeros(self._n_dof, dtype=np.float64)
            du[free_idx] = du_free
            du_t = torch.from_numpy(du).reshape(-1, 2).to(dtype=u.dtype,
                                                           device=u.device)
            u += du_t
            u[bc_mask_dev] = bc_vals_dev[bc_mask_dev]

            # For isotropic split, one Newton step is exact
            if state['split'] == 'isotropic':
                break

        # Check final residual
        f_int_final = fem.internal_force(u, d_dev)
        r_final = (f_ext_dev - f_int_final) if f_ext_dev is not None else -f_int_final
        r_final[bc_mask_dev] = 0.0
        r_final_sq = (r_final * r_final).sum().item()
        self.last_diverged = (r_final_sq > tol_sq and
                              r_final_sq > rtol_sq * r0_norm_sq)
        self.last_iter = nr + 1

        if self.last_diverged:
            print(f"  [DirectSolver] WARNING: did not converge after "
                  f"{self.max_newton} Newton steps, ||r||={r_final_sq**0.5:.2e}"
                  f" (rel={r_final_sq**0.5 / max(r0_norm_sq**0.5, 1e-30):.2e})",
                  flush=True)

        return u.to(dtype=orig_dtype, device=orig_device)

    @torch.no_grad()
    def _solve_impl_mpc(self, u, d, bc_mask, bc_vals, f_ext, rcs):
        """Direct LU solve with rotation-free rigid-connector MPC (#171, #206).

        Mirrors :meth:`solve` but operates in the reduced primary
        variable ``q = [u_flat; theta_*]`` of size ``2N + n_rc``. The
        reduced stiffness ``K_q = T^T K T`` is formed by sparse triple
        product (DirectSolver assembles K explicitly so no probe-build
        is needed, unlike SecantCG); the free-DOF subsystem is direct-
        solved via :func:`_spsolve_auto`. Slave displacements are
        reconstructed from master + theta after each Newton update.

        For isotropic split, one Newton step is exact (matches the
        non-MPC path's early-return guard).
        """
        import numpy as np
        import scipy.sparse as sp

        self.last_diverged = False
        orig_device = u.device
        orig_dtype = u.dtype

        # MPS / float32 fallback — same routing as the welded path.
        if self._needs_cpu_fallback:
            fem = self._cpu_fem
            u = u.detach().cpu().to(torch.float64)
            d_dev = d.detach().cpu().to(torch.float64)
            bc_mask_dev = bc_mask.cpu()
            bc_vals_dev = bc_vals.detach().cpu().to(torch.float64)
            f_ext_dev = (f_ext.detach().cpu().to(torch.float64)
                         if f_ext is not None else None)
        else:
            fem = self.fem
            u = u.detach().clone()
            d_dev = d.detach()
            bc_mask_dev = bc_mask
            bc_vals_dev = bc_vals.detach()
            f_ext_dev = f_ext.detach() if f_ext is not None else None

        # Build T and free-q index. Validates master-Dirichlet contract.
        T_csr, free_q_mask, free_q_idx, n_dof, n_rc, _slave_set = \
            _build_rigid_connector_T(rcs, fem, bc_mask_dev)

        # Apply Dirichlet on master / left edge / etc., then reconstruct
        # slaves from master + initial theta=0 to seed a kinematically
        # admissible u.
        u[bc_mask_dev] = bc_vals_dev[bc_mask_dev]
        theta_vals = np.zeros(n_rc, dtype=np.float64)
        u_flat = u.detach().cpu().numpy().astype(np.float64).reshape(-1)
        _reconstruct_slaves_from_q(u_flat, rcs, theta_vals, fem)
        u = torch.from_numpy(u_flat.reshape(-1, 2)).to(
            dtype=u.dtype, device=u.device)

        tol_sq = self.tol ** 2
        rtol_sq = self.rtol ** 2
        nr = 0
        r0_norm_sq = None

        for nr in range(max(self.max_newton, 1)):
            # True nonlinear residual
            f_int = fem.internal_force(u, d_dev)
            r_full = (f_ext_dev - f_int) if f_ext_dev is not None else -f_int
            r_full[bc_mask_dev] = 0.0

            r_full_np = r_full.detach().cpu().numpy().astype(
                np.float64).reshape(-1)
            r_q = T_csr.T @ r_full_np
            r_q_free = r_q[free_q_idx]
            r_norm_sq = float((r_q_free * r_q_free).sum())

            if r0_norm_sq is None:
                r0_norm_sq = max(r_norm_sq, 1e-30)

            if r_norm_sq < tol_sq or r_norm_sq < rtol_sq * r0_norm_sq:
                self.last_iter = nr
                self._record_mpc_diagnostics(rcs, theta_vals, u, fem,
                                             f_ext_dev, d_dev)
                return u.to(dtype=orig_dtype, device=orig_device)

            # Assemble full K, project to reduced space K_q = T^T K T.
            state = fem.freeze_secant_state(u, d_dev)
            K_csr = self._assemble_stiffness(state)
            K_q = (T_csr.T @ K_csr) @ T_csr  # (n_q, n_q) sparse
            K_qq = K_q.tocsr()[free_q_idx][:, free_q_idx]

            dq_free = _spsolve_auto(
                K_qq, r_q_free,
                self._orig_device if self._use_gpu else None)

            dq_acc = np.zeros_like(r_q)
            dq_acc[free_q_idx] = dq_free

            # Update theta, master and reconstruct slaves
            theta_vals += dq_acc[n_dof:]
            du_full = T_csr @ dq_acc  # (n_dof,)
            u_flat = u.detach().cpu().numpy().astype(
                np.float64).reshape(-1)
            u_flat += du_full
            u_new = torch.from_numpy(u_flat.reshape(-1, 2)).to(
                dtype=u.dtype, device=u.device)
            u_new[bc_mask_dev] = bc_vals_dev[bc_mask_dev]
            u_flat = u_new.detach().cpu().numpy().astype(
                np.float64).reshape(-1)
            _reconstruct_slaves_from_q(u_flat, rcs, theta_vals, fem)
            u = torch.from_numpy(u_flat.reshape(-1, 2)).to(
                dtype=u.dtype, device=u.device)

            # For isotropic split, one Newton step is exact
            if state['split'] == 'isotropic':
                break

        # Final residual check
        f_int_final = fem.internal_force(u, d_dev)
        r_final = ((f_ext_dev - f_int_final)
                   if f_ext_dev is not None else -f_int_final)
        r_final[bc_mask_dev] = 0.0
        r_final_np = r_final.detach().cpu().numpy().astype(
            np.float64).reshape(-1)
        r_q_final = (T_csr.T @ r_final_np)[free_q_idx]
        r_final_sq = float((r_q_final * r_q_final).sum())
        self.last_diverged = (r_final_sq > tol_sq and
                              r_final_sq > rtol_sq * r0_norm_sq)
        self.last_iter = nr + 1

        if self.last_diverged:
            print(f"  [DirectSolver-MPC] WARNING: did not converge after "
                  f"{self.max_newton} Newton steps, "
                  f"||r_q||={r_final_sq**0.5:.2e} "
                  f"(rel={r_final_sq**0.5 / max(r0_norm_sq**0.5, 1e-30):.2e})",
                  flush=True)

        self._record_mpc_diagnostics(rcs, theta_vals, u, fem,
                                     f_ext_dev, d_dev)
        return u.to(dtype=orig_dtype, device=orig_device)

    def _record_mpc_diagnostics(self, rcs, theta_vals, u, fem, f_ext, d):
        """Populate ``last_theta`` / ``last_master_reaction`` for the
        rotation-free MPC path. Mirrors
        :meth:`SecantCGSolver._record_mpc_diagnostics_cg`."""
        import numpy as np
        f_int = fem.internal_force(u, d).detach().cpu().numpy()
        if f_ext is not None:
            f_ext_np = f_ext.detach().cpu().numpy()
        else:
            f_ext_np = np.zeros_like(f_int)
        self.last_theta = list(map(float, theta_vals))
        self.last_master_reaction = []
        for rc in rcs:
            slaves = rc.slaves_excluding_master().detach().cpu().numpy()
            m = rc.master_node
            ids = np.concatenate([np.array([m]), slaves]).astype(np.int64)
            R = (f_int[ids] - f_ext_np[ids]).sum(axis=0)
            self.last_master_reaction.append(
                (int(m), float(R[0]), float(R[1])))


class SecantCGSolver:
    """[PRIMARY] Newton-Secant CG solver for quasi-static equilibrium.

    Recommended solver for all quasi-static benchmarks (SENT, SENS, TPB,
    L-panel). Matches PhaseFieldX's SNES approach used by 17/18 reference
    implementations surveyed. Handles all energy splits correctly.

    Wraps a Newton outer loop around secant-linearized CG inner solves.
    At each Newton step:
      1. Re-freeze the tension/compression state from current u
      2. Compute the TRUE nonlinear residual
      3. Solve the linearized system via preconditioned CG
      4. Update u and check for convergence

    This matches PhaseFieldX's SNES approach: the spectral split makes
    sigma_a(u) nonlinear in u (eigenvalue signs change), so a single
    linearization is insufficient near the critical point.

    Parameters
    ----------
    fem : FEMOperators
    tol : float
        Residual norm tolerance for the outer Newton loop.
    max_iter : int
        Maximum CG iterations per Newton step.
    max_newton : int
        Maximum Newton re-linearization steps (1 = legacy behavior).
    check_every : int
        GPU→CPU sync frequency for CG convergence check.
    use_multigrid : bool
        Replace Jacobi with VectorMultigrid preconditioner. Reduces CG
        iterations at the cost of V-cycle overhead per iteration.
    """

    def __init__(self, fem: FEMOperators, tol: float = 1e-8,
                 max_iter: int = 2000, max_newton: int = 5,
                 check_every: int = 50, use_multigrid: bool = True,
                 line_search_maxiter: int = 6):
        self.fem = fem
        self.tol = tol
        self.max_iter = max_iter
        self.max_newton = max_newton
        self.check_every = check_every
        self.line_search_maxiter = line_search_maxiter

        # MPS float64 fallback: spectral split CG is numerically unstable
        # in float32 (eigenvalue sign flips cause 10^12x residual growth).
        # Create a CPU float64 FEM copy for the solve, matching the pattern
        # used by PhaseFieldDamageSolver.
        dev = fem.device
        if isinstance(dev, str):
            dev = torch.device(dev)
        self._needs_cpu_fallback = not device_supports_float64(dev)
        self._cpu_fem = None
        if self._needs_cpu_fallback:
            from ..core.mesh import FEMMesh
            cpu_mesh = FEMMesh.from_tensors(
                fem.mesh.nodes, fem.mesh.elements, fem.mesh.node_sets,
                device='cpu', dtype=torch.float64)
            self._cpu_fem = FEMOperators(cpu_mesh, fem.material, ctx=None)
            print(f"[SecantCG] MPS detected — CG will run on CPU float64",
                  flush=True)

        # L-BFGS preconditioner for CG (builds up over load steps).
        # The (s,y) history is updated at Newton convergence (outside the CG loop),
        # so the preconditioner is CONSTANT within each inner CG solve — this does
        # NOT break conjugacy. The history reflects previous load steps' curvature,
        # giving a low-rank H^-1 approximation that helps CG on slowly-evolving problems.
        self._lbfgs_precond = LBFGSPreconditioner(m=5)

        # Divergence tracking: set after each solve()
        self.last_diverged = False

        self._use_multigrid = use_multigrid
        self._multigrid = None

        # MPC diagnostics — populated by the rotation-free rigid-connector
        # path (issue #171). Mirrors DirectSolver.last_theta /
        # DirectSolver.last_master_reaction so downstream postprocess code
        # can read them solver-independently.
        self.last_theta = []
        self.last_master_reaction = []

    @torch.no_grad()
    def solve(self, u, d, bc_mask, bc_vals, f_ext=None,
              rigid_connectors=None):
        """Solve equilibrium using Newton-secant CG.

        Parameters
        ----------
        u : (N, 2) displacement (initial guess)
        d : (N,) damage field (frozen during solve)
        bc_mask : (N, 2) bool — True at constrained DOFs
        bc_vals : (N, 2) float — prescribed values
        f_ext : (N, 2) external forces (optional)
        rigid_connectors : list of RigidConnector, optional
            Rotation-free rigid connectors enforced via master-slave
            elimination on the CG iterate (issue #171). When supplied,
            CG runs in the reduced q-space (size 2N + n_rc) with the
            matvec wrapped as ``T^T (K (T v))``. Multigrid is forced off
            for this path (the agg is over original mesh DOFs); Jacobi
            preconditioner is rebuilt against ``diag(T^T K T)``. The
            L-BFGS preconditioner is also bypassed when MPC is active —
            its history is keyed by the original DOF count and would
            need resizing.

        Returns
        -------
        u : (N, 2) equilibrium displacement
        """
        # MPS fallback: route entire solve through CPU float64
        if self._needs_cpu_fallback:
            return self._solve_cpu_fallback(u, d, bc_mask, bc_vals, f_ext,
                                            rigid_connectors)

        if rigid_connectors:
            return self._solve_impl_mpc(self.fem, u, d, bc_mask, bc_vals,
                                        f_ext, rigid_connectors)
        return self._solve_impl(self.fem, u, d, bc_mask, bc_vals, f_ext)

    def _solve_cpu_fallback(self, u, d, bc_mask, bc_vals, f_ext,
                            rigid_connectors=None):
        """Transfer to CPU float64, solve, transfer back."""
        orig_device = u.device
        orig_dtype = u.dtype
        fem_cpu = self._cpu_fem

        # Transfer inputs: MPS→CPU requires .cpu() before .to(float64)
        u_cpu = u.detach().cpu().to(torch.float64)
        d_cpu = d.detach().cpu().to(torch.float64)
        bc_mask_cpu = bc_mask.cpu()
        bc_vals_cpu = bc_vals.detach().cpu().to(torch.float64)
        f_ext_cpu = None
        if f_ext is not None:
            f_ext_cpu = f_ext.detach().cpu().to(torch.float64)

        if rigid_connectors:
            u_result = self._solve_impl_mpc(
                fem_cpu, u_cpu, d_cpu, bc_mask_cpu, bc_vals_cpu, f_ext_cpu,
                rigid_connectors)
        else:
            u_result = self._solve_impl(
                fem_cpu, u_cpu, d_cpu, bc_mask_cpu, bc_vals_cpu, f_ext_cpu)

        # Transfer back: CPU→MPS
        return u_result.to(dtype=orig_dtype, device=orig_device)

    @torch.no_grad()
    def _solve_impl(self, fem, u, d, bc_mask, bc_vals, f_ext):
        """Core Newton-secant CG implementation.

        Architecture (important for code review):
        - OUTER: Newton loop re-linearizes the secant tangent each iteration
        - INNER: CG solves the linearized system with a FIXED preconditioner
        - PRECONDITIONER: L-BFGS builds curvature from previous load steps'
          (s,y) pairs. It is updated ONLY at Newton convergence (lines below)
          and at the end of solve(), never inside the CG loop. Therefore it is
          constant within each CG solve and does NOT break CG conjugacy.
          Falls back to Jacobi when L-BFGS history is empty (first step).
        """
        self.last_diverged = False
        u = u.clone()
        u[bc_mask] = bc_vals[bc_mask]
        free_mask = (~bc_mask).to(u.dtype)

        # --- Jacobi preconditioner (damage-dependent, constant within solve) ---
        M_diag = fem.stiffness_diagonal(d)
        M_diag *= free_mask
        diag_floor = 1e-10 * M_diag.abs().max().clamp(min=1e-30)
        M_inv = free_mask / M_diag.clamp(min=diag_floor)

        # Adaptive check frequency: multigrid converges in fewer iters
        check_every = 5 if self._use_multigrid else self.check_every

        tol_sq = self.tol ** 2
        total_cg_iter = 0
        stale_count = 0  # track consecutive stale/diverged Newton steps

        # Save state for L-BFGS preconditioner update
        u_old = u.clone()
        r_old_store = fem.internal_force(u, d)
        if f_ext is not None:
            r_old_store = f_ext - r_old_store
        else:
            r_old_store = -r_old_store
        r_old_store *= free_mask
        r_old = r_old_store

        for nr in range(self.max_newton):
            # --- Re-linearize: freeze secant state from current u ---
            state = fem.freeze_secant_state(u, d)

            # Update multigrid with new secant state (if enabled)
            if self._use_multigrid and self._multigrid is not None:
                self._multigrid.agg.rebuild(d)
                self._multigrid.update(d, secant_state=state, bc_mask=bc_mask)

            # --- TRUE nonlinear residual ---
            if nr > 0:
                f_int = fem.internal_force(u, d)
                r = (f_ext - f_int) if f_ext is not None else -f_int
                r *= free_mask
            else:
                r = r_old_store.clone()

            # Check Newton convergence
            r_norm_sq = (r * r).sum()
            if r_norm_sq.item() < tol_sq:
                # Update L-BFGS preconditioner before returning
                if self._lbfgs_precond is not None:
                    s = (u - u_old).flatten()
                    y = (r_old - r).flatten()
                    self._lbfgs_precond.update(s, y)
                self.last_iter = total_cg_iter
                return u

            # --- CG inner solve with frozen linearization ---
            r_norm_sq_0 = r_norm_sq
            u_old_nr = u.clone()  # save pre-CG state for staleness rollback
            if self._use_multigrid and self._multigrid is not None:
                z = self._multigrid.vcycle(r)
            elif self._lbfgs_precond is not None and len(self._lbfgs_precond.s_history) > 0:
                z = self._lbfgs_precond.apply(r.flatten()).reshape(r.shape)
            else:
                z = r * M_inv
            p = z.clone()
            rz = (r * z).sum()

            # Budget CG iterations across Newton steps
            cg_budget = self.max_iter // max(self.max_newton, 1)

            for i in range(cg_budget):
                Ap = fem.secant_matvec(p, state)
                Ap *= free_mask

                pAp_val = (p * Ap).sum().item()
                alpha_val = rz.item() / (pAp_val + 1e-30)
                u.add_(p, alpha=alpha_val)
                # Note: BCs are NOT re-enforced here. The search direction p
                # should already be zero at constrained DOFs (since r and z
                # are zero there via free_mask). Re-enforcing BCs between
                # u.add_ and r.add_ would break CG conjugacy.
                r.add_(Ap, alpha=-alpha_val)

                if self._use_multigrid and self._multigrid is not None:
                    z = self._multigrid.vcycle(r)
                elif self._lbfgs_precond is not None and len(self._lbfgs_precond.s_history) > 0:
                    z = self._lbfgs_precond.apply(r.flatten()).reshape(r.shape)
                else:
                    z = r * M_inv
                rz_new = (r * z).sum()
                beta_val = (rz_new / (rz + 1e-30)).item()
                p.mul_(beta_val).add_(z)
                rz = rz_new
                total_cg_iter += 1

                # Note: We do NOT recompute the true residual inside the CG
                # inner loop. The CG solves the LINEARIZED (secant) system,
                # and replacing r with the nonlinear residual would break
                # conjugacy. Round-off drift is acceptable over the CG budget
                # (~400 iters max) and is corrected when the Newton loop
                # recomputes the full nonlinear residual at line 533.

                if (i + 1) % check_every == 0:
                    r_sq = (r * r).sum()
                    if r_sq.item() < tol_sq:
                        break
                    if r_sq.item() > 1e12 * max(r_norm_sq_0.item(), 1e-30):
                        print(f"  [SecantCG diverged at NR={nr} CG={i+1}]",
                              flush=True)
                        break

            # For isotropic split, secant is exact — skip extra Newton steps
            if state['split'] == 'isotropic':
                break

            # Staleness guard with backtracking line search:
            # If the frozen secant linearization became stale (eigenvalue signs
            # flipped during CG for spectral split), the full CG step overshoots.
            # Instead of discarding the entire step (old behavior), find the
            # largest fraction alpha that actually reduces the true nonlinear
            # residual.  This ensures forward progress even when the secant is
            # imperfect — matching SNES behavior where each Newton iteration
            # contributes via line search.
            du = u - u_old_nr  # CG displacement update
            f_int_check = fem.internal_force(u, d)
            r_check = (f_ext - f_int_check) if f_ext is not None else -f_int_check
            r_check *= free_mask
            r_check_sq = (r_check * r_check).sum().item()

            if r_check_sq > r_norm_sq_0.item() * 1.01:
                # Full step made things worse — backtracking line search
                best_alpha = 0.0
                best_r_sq = r_norm_sq_0.item()
                alpha = 0.5
                for _ls in range(self.line_search_maxiter):
                    u_trial = u_old_nr + alpha * du
                    u_trial[bc_mask] = bc_vals[bc_mask]
                    f_int_trial = fem.internal_force(u_trial, d)
                    r_trial = (f_ext - f_int_trial) if f_ext is not None else -f_int_trial
                    r_trial *= free_mask
                    r_trial_sq = (r_trial * r_trial).sum().item()
                    if r_trial_sq < best_r_sq:
                        best_r_sq = r_trial_sq
                        best_alpha = alpha
                    # Bisect: try smaller if still worse than initial,
                    # try larger if we found improvement
                    if r_trial_sq < r_norm_sq_0.item():
                        # This alpha works — try a larger step
                        alpha = min(alpha * 1.5, 1.0)
                        if alpha >= 1.0:
                            break
                    else:
                        alpha *= 0.5

                if best_alpha > 0.0:
                    u.copy_(u_old_nr + best_alpha * du)
                    u[bc_mask] = bc_vals[bc_mask]
                    reduction = best_r_sq / r_norm_sq_0.item()
                    print(f"  [SecantCG] Line search at NR={nr}: "
                          f"alpha={best_alpha:.3f}, "
                          f"residual {reduction:.3f}x", flush=True)
                else:
                    # No alpha helped — full rollback
                    u.copy_(u_old_nr)
                    stale_count += 1
                    print(f"  [SecantCG] Stale at NR={nr}: "
                          f"residual grew {r_check_sq/r_norm_sq_0.item():.1f}x, "
                          f"line search found no improvement, "
                          f"rolling back", flush=True)

        # Flag divergence if ALL Newton iterations were stale/rolled back
        self.last_diverged = (stale_count >= self.max_newton)

        # Update L-BFGS preconditioner with displacement/residual changes
        if self._lbfgs_precond is not None:
            f_int_final = fem.internal_force(u, d)
            r_new = (f_ext - f_int_final) if f_ext is not None else -f_int_final
            r_new *= free_mask
            s = (u - u_old).flatten()
            y = (r_old - r_new).flatten()
            self._lbfgs_precond.update(s, y)

        self.last_iter = total_cg_iter
        return u

    @torch.no_grad()
    def _solve_impl_mpc(self, fem, u, d, bc_mask, bc_vals, f_ext, rcs):
        """Newton-secant CG with rotation-free rigid-connector MPC (issue #171).

        Mirrors :meth:`_solve_impl` but operates in the reduced primary
        variable q = [u_flat; theta_*] of size ``2N + n_rc``. The matvec
        is wrapped as ``T^T (K_secant (T v))``; the residual is
        ``T^T r_full``. Slave displacements are reconstructed from
        master + theta after each Newton update.

        Restrictions vs the welded path:
          * Multigrid preconditioner is forced off (its agg lives over the
            original mesh DOFs and does not transform under T).
          * L-BFGS preconditioner history is bypassed (its (s, y) shapes
            are tied to the original DOF count).
          * Falls back to Jacobi against ``diag(T^T K T)``: identity for
            non-slave q-DOFs (= K diagonal), one secant_matvec per theta
            DOF for the rotation entries.
        """
        import numpy as np

        # One-shot diagnostic (#189): the MPC path silently bypasses both
        # multigrid and the L-BFGS preconditioner history. If either was
        # requested, surface a single info-log per solver instance so a
        # user with `use_multigrid=True` + rigid_connector understands why
        # convergence is slower than the welded path.
        _lbfgs_active = (
            getattr(self, '_lbfgs_precond', None) is not None
            and getattr(self._lbfgs_precond, 'm', 0) > 0
        )
        if (self._use_multigrid or _lbfgs_active) and not getattr(
                self, '_mpc_warning_emitted', False):
            print(
                "[SecantCGSolver][MPC] rigid_connector active -- multigrid "
                "preconditioner\nand L-BFGS history bypassed (#171/#189). "
                "Falling back to Jacobi-only CG.\nThis may slow convergence "
                "on stiff or fine-mesh problems.",
                flush=True,
            )
            self._mpc_warning_emitted = True

        self.last_diverged = False
        u = u.clone()
        u[bc_mask] = bc_vals[bc_mask]
        bc_mask_dev = bc_mask
        bc_vals_dev = bc_vals

        # Build master-slave T matrix and free q-DOF index
        T_csr, free_q_mask, free_q_idx, n_dof, n_rc, slave_dof_set = \
            _build_rigid_connector_T(rcs, fem, bc_mask_dev)

        # Reconstruct slaves from initial master (theta starts at 0)
        theta_vals = np.zeros(n_rc, dtype=np.float64)
        u_flat = u.detach().cpu().numpy().astype(np.float64).reshape(-1)
        _reconstruct_slaves_from_q(u_flat, rcs, theta_vals, fem)
        u = torch.from_numpy(u_flat.reshape(-1, 2)).to(
            dtype=u.dtype, device=u.device)

        torch_dtype = u.dtype
        torch_device = u.device

        def _to_full_torch(v_red_np):
            """T @ v_red  →  (N, 2) torch tensor (matvec input space)."""
            v_full = T_csr @ v_red_np  # (n_dof,)
            return torch.from_numpy(v_full.reshape(-1, 2)).to(
                dtype=torch_dtype, device=torch_device)

        def _to_red_np(v_full_torch):
            """T^T @ v_full  →  (n_q,) numpy."""
            v_full = v_full_torch.detach().cpu().numpy().astype(
                np.float64).reshape(-1)
            return T_csr.T @ v_full

        def _matvec_red(p_red, state):
            """Reduced matvec  K_q v_q = T^T K_secant (T v_q)."""
            p_full = _to_full_torch(p_red)
            Kp_full = fem.secant_matvec(p_full, state)
            # Apply Dirichlet zero on bc_mask before contracting back —
            # identical to the (N,2) path that does ``Ap *= free_mask``
            # to suppress reactions through Dirichlet rows.
            Kp_full = Kp_full.masked_fill(bc_mask_dev, 0.0)
            return _to_red_np(Kp_full)

        def _residual_red(u_cur):
            """r_q = T^T r_full where r_full = f_ext - f_int, zeroed on Dirichlet."""
            f_int = fem.internal_force(u_cur, d)
            r_full = (f_ext - f_int) if f_ext is not None else -f_int
            r_full = r_full.masked_fill(bc_mask_dev, 0.0)
            r_q = _to_red_np(r_full)
            r_q[~free_q_mask] = 0.0
            return r_q, r_full

        tol_sq = self.tol ** 2
        total_cg_iter = 0
        stale_count = 0

        for nr in range(self.max_newton):
            # Re-linearise: freeze secant state at current u
            state = fem.freeze_secant_state(u, d)

            # True nonlinear residual (reduced)
            r_q, _r_full = _residual_red(u)
            r_norm_sq = float((r_q[free_q_idx] ** 2).sum())
            if nr == 0:
                r_norm_sq_0 = max(r_norm_sq, 1e-30)

            if r_norm_sq < tol_sq:
                self.last_iter = total_cg_iter
                self._record_mpc_diagnostics_cg(rcs, theta_vals, u, fem,
                                                f_ext, d)
                return u

            # ---- Build Jacobi preconditioner on diag(T^T K T) ----
            # For non-slave q-DOFs (T is identity column → row), the
            # diagonal equals K_diag at the corresponding flat DOF.
            K_diag_full_t = fem.stiffness_diagonal(d).detach().cpu().numpy(
                ).astype(np.float64).reshape(-1)
            diag_q = np.zeros(n_dof + n_rc, dtype=np.float64)
            diag_q[:n_dof] = K_diag_full_t
            # For theta DOFs: diag entry = e_j^T (T^T K T) e_j. Use one
            # secant_matvec per theta DOF — typically n_rc is 1-few.
            for k in range(n_rc):
                e_red = np.zeros(n_dof + n_rc, dtype=np.float64)
                e_red[n_dof + k] = 1.0
                Ke_red = _matvec_red(e_red, state)
                diag_q[n_dof + k] = float(Ke_red[n_dof + k])
            # Mask: only free q-DOFs participate; floor diagonal
            diag_q[~free_q_mask] = 0.0
            diag_floor = 1e-12 * np.abs(diag_q).max()
            inv_diag_q = np.where(free_q_mask,
                                  1.0 / np.maximum(np.abs(diag_q), diag_floor),
                                  0.0)
            # Restore sign on entries we floored (Jacobi expects diag itself,
            # but we use abs floor only to avoid div-by-zero on ill-conditioned
            # rows — keep the original sign from diag_q).
            sign = np.where(diag_q != 0.0, np.sign(diag_q), 1.0)
            inv_diag_q = inv_diag_q * sign

            # ---- CG inner solve in q-space, solving K_q dq = r_q ----
            r = r_q.copy()
            r[~free_q_mask] = 0.0
            z = inv_diag_q * r
            p = z.copy()
            rz = float((r * z).sum())

            cg_budget = self.max_iter // max(self.max_newton, 1)
            check_every = self.check_every
            dq_acc = np.zeros_like(r)

            for i in range(cg_budget):
                Ap = _matvec_red(p, state)
                Ap[~free_q_mask] = 0.0
                pAp = float((p * Ap).sum())
                alpha_val = rz / (pAp + 1e-30)
                dq_acc += alpha_val * p
                r -= alpha_val * Ap
                z = inv_diag_q * r
                rz_new = float((r * z).sum())
                beta_val = rz_new / (rz + 1e-30)
                p = beta_val * p + z
                rz = rz_new
                total_cg_iter += 1

                if (i + 1) % check_every == 0:
                    r_sq = float((r[free_q_idx] ** 2).sum())
                    if r_sq < tol_sq:
                        break
                    if r_sq > 1e12 * r_norm_sq_0:
                        print(f"  [SecantCG-MPC diverged at NR={nr} "
                              f"CG={i+1}]", flush=True)
                        break

            # ---- Apply increment: update theta, master, reconstruct slaves ----
            for k in range(n_rc):
                theta_vals[k] += dq_acc[n_dof + k]
            du_full = T_csr @ dq_acc  # (n_dof,)
            u_flat = u.detach().cpu().numpy().astype(np.float64).reshape(-1)
            u_flat += du_full
            # Reapply Dirichlet on master translations / left edge etc.
            # Then reconstruct slaves from master + theta to be safe.
            u_new = torch.from_numpy(u_flat.reshape(-1, 2)).to(
                dtype=torch_dtype, device=torch_device)
            u_new[bc_mask_dev] = bc_vals_dev[bc_mask_dev]
            u_flat = u_new.detach().cpu().numpy().astype(
                np.float64).reshape(-1)
            _reconstruct_slaves_from_q(u_flat, rcs, theta_vals, fem)
            u = torch.from_numpy(u_flat.reshape(-1, 2)).to(
                dtype=torch_dtype, device=torch_device)

            # For isotropic split, secant is exact — one Newton step is enough
            if state['split'] == 'isotropic':
                break

            # Staleness guard (line search omitted in MPC path for the
            # initial implementation: rotation-free MPC primarily targets
            # linear-elastic stiff-body coupling where state['split'] is
            # 'isotropic' and the loop already exits above. If a future
            # benchmark exercises spectral split + rotation-free MPC and
            # exhibits staleness, port the line-search block from
            # _solve_impl with reduced-space arithmetic.)

        self.last_diverged = (stale_count >= self.max_newton)
        self.last_iter = total_cg_iter
        self._record_mpc_diagnostics_cg(rcs, theta_vals, u, fem, f_ext, d)
        return u

    def _record_mpc_diagnostics_cg(self, rcs, theta_vals, u, fem, f_ext, d):
        """Mirror DirectSolver._record_mpc_diagnostics for the CG path."""
        import numpy as np
        f_int = fem.internal_force(u, d).detach().cpu().numpy()
        if f_ext is not None:
            f_ext_np = f_ext.detach().cpu().numpy()
        else:
            f_ext_np = np.zeros_like(f_int)
        self.last_theta = list(map(float, theta_vals))
        self.last_master_reaction = []
        for rc in rcs:
            slaves = rc.slaves_excluding_master().detach().cpu().numpy()
            m = rc.master_node
            ids = np.concatenate([np.array([m]), slaves]).astype(np.int64)
            R = (f_int[ids] - f_ext_np[ids]).sum(axis=0)
            self.last_master_reaction.append(
                (int(m), float(R[0]), float(R[1])))


class LBFGSSolver:
    """[AVAILABLE] L-BFGS solver for quasi-static equilibrium.

    Not used by any benchmark. Retained for problems where the tangent
    operator (matvec) is unavailable or too expensive to form. For the
    well-conditioned linear subproblems AT2 produces after secant
    linearization, SecantCGSolver with Jacobi/multigrid preconditioning
    converges faster.

    Minimizes the residual norm ||f_int(u, d) - f_ext||^2
    subject to Dirichlet BCs.

    For the AT2 model where f_int is linear in u (fixed d),
    this is equivalent to solving K(d) u = f_ext.

    L-BFGS builds an approximate inverse Hessian from gradient history.

    Parameters
    ----------
    fem : FEMOperators
    max_iter : int
        Maximum L-BFGS iterations per solve.
    lr : float
        Step size. 1.0 is optimal for quadratic problems.
    history_size : int
        Past iterations for Hessian approximation (10-20 typical).
    tol : float
        Convergence tolerance on residual norm.
    line_search : str
        'strong_wolfe' (default, robust) or None (fixed step).
    """

    def __init__(self, fem: FEMOperators, max_iter: int = 100,
                 lr: float = 1.0, history_size: int = 20,
                 tol: float = 1e-8, line_search: str = 'strong_wolfe'):
        self.fem = fem
        self.max_iter = max_iter
        self.lr = lr
        self.history_size = history_size
        self.tol = tol
        self.line_search = line_search

    def solve(self, d, f_ext, bc_mask, bc_vals, u_init=None):
        """Solve equilibrium via L-BFGS minimization.

        Parameters
        ----------
        d : (N,) damage field (fixed).
        f_ext : (N, 2) external forces.
        bc_mask : (N, 2) bool
        bc_vals : (N, 2) float
        u_init : (N, 2) or None

        Returns
        -------
        u : (N, 2) equilibrium displacement
        converged : bool
        n_iter : int
        """
        import warnings
        split = getattr(self.fem.material, 'energy_split', 'isotropic')
        if split in ('spectral', 'spectral_plane_stress_condensed',
                     'amor', 'star_convex'):
            warnings.warn(
                "LBFGSSolver minimizes ||residual||^2 which is non-convex for "
                f"'{split}' split. Results may converge to saddle points. "
                "Consider using SecantCGSolver instead.", stacklevel=2)

        mesh = self.fem.mesh

        # u needs grad for L-BFGS
        if u_init is not None:
            u = u_init.detach().clone().requires_grad_(True)
        else:
            u = torch.zeros(mesh.n_nodes, 2, dtype=mesh.dtype,
                            device=mesh.device, requires_grad=True)

        with torch.no_grad():
            u.data[bc_mask] = bc_vals[bc_mask]

        optimizer = torch.optim.LBFGS(
            [u],
            lr=self.lr,
            max_iter=self.max_iter,
            max_eval=self.max_iter * 2,
            history_size=self.history_size,
            tolerance_grad=self.tol,
            tolerance_change=1e-14,
            line_search_fn=self.line_search,
        )

        n_eval = [0]
        final_res = [float('inf')]

        def closure():
            optimizer.zero_grad()
            # BCs are re-enforced at the start of each closure evaluation.
            # L-BFGS line search may temporarily violate BCs between closure calls,
            # but they are always restored before loss/gradient computation.
            with torch.no_grad():
                u.data[bc_mask] = bc_vals[bc_mask]

            # f_int(u, d) — needs grad flow through u
            f_int = self.fem.internal_force(u, d)
            residual = f_int - f_ext
            # Zero BC DOFs in residual
            residual = residual.clone()
            residual[bc_mask] = 0.0

            # Minimize ||residual||^2
            loss = 0.5 * (residual * residual).sum()
            loss.backward()

            # Zero grad at BC DOFs
            if u.grad is not None:
                u.grad.data[bc_mask] = 0.0

            n_eval[0] += 1
            final_res[0] = residual.detach().norm().item()
            return loss

        optimizer.step(closure)

        # Final BC enforcement
        with torch.no_grad():
            u.data[bc_mask] = bc_vals[bc_mask]

        converged = final_res[0] < self.tol
        return u.detach(), converged, n_eval[0]


class MonolithicSolver:
    """Monolithic (coupled) solver: minimizes total energy over (u, d) jointly.

    Instead of alternating between mechanics and damage sub-problems
    (staggered), this solver treats displacement and damage as a single
    optimization variable and minimizes the total energy functional:

        E(u, d) = E_elastic(u, d) + E_fracture(d)

    using L-BFGS with post-step clamping and masked gradients for the
    damage irreversibility interval d ∈ [d_prev, 1]. This is an
    experimental surrogate, not a true active-set, variational-inequality,
    or L-BFGS-B bound-constrained damage solve.

    Advantages over staggered:
      - No stagger iterations needed (10x larger load steps possible)
      - Handles "unstable" cracking where staggered schemes stall
      - Experimental handling of irreversibility via clamp/masked-gradient
        projection; validate before using for production fracture results

    References
    ----------
    Gerasimov & De Lorenzis (2019), Wu et al. (2020).

    Parameters
    ----------
    fem : FEMOperators
    max_iter : int
        Maximum L-BFGS iterations per solve.
    tol : float
        Gradient norm tolerance for convergence.
    history_size : int
        L-BFGS history depth.
    """

    def __init__(self, fem: FEMOperators, max_iter: int = 200,
                 tol: float = 1e-8, history_size: int = 20):
        self.fem = fem
        self.max_iter = max_iter
        self.tol = tol
        self.history_size = history_size

    def solve(self, u: torch.Tensor, d: torch.Tensor,
              bc_mask: torch.Tensor, bc_vals: torch.Tensor,
              f_ext: torch.Tensor = None,
              d_prev: torch.Tensor = None) -> tuple:
        """Solve coupled (u, d) by energy minimization.

        Parameters
        ----------
        u : (N, 2) displacement initial guess
        d : (N,) damage initial guess
        bc_mask : (N, 2) bool — constrained displacement DOFs
        bc_vals : (N, 2) float — prescribed displacement values
        f_ext : (N, 2) or None — external forces
        d_prev : (N,) or None — previous damage for irreversibility bound.
                 If None, uses d as lower bound.

        Returns
        -------
        u : (N, 2) equilibrium displacement
        d : (N,) equilibrium damage
        converged : bool
        n_iter : int
        """
        fem = self.fem
        mesh = fem.mesh
        mat = fem.material

        if d_prev is None:
            d_prev = d.clone()

        # Pack (u, d) into single parameter vector
        u_opt = u.detach().clone().requires_grad_(True)
        d_opt = d.detach().clone().requires_grad_(True)

        # Enforce initial BCs
        with torch.no_grad():
            u_opt.data[bc_mask] = bc_vals[bc_mask]
            d_opt.data.copy_(torch.maximum(d_opt.data, d_prev).clamp(max=1.0))

        optimizer = torch.optim.LBFGS(
            [u_opt, d_opt],
            lr=1.0,
            max_iter=self.max_iter,
            max_eval=self.max_iter * 2,
            history_size=self.history_size,
            tolerance_grad=self.tol,
            tolerance_change=1e-14,
            line_search_fn='strong_wolfe',
        )

        n_eval = [0]
        final_grad = [float('inf')]

        Gc = mat.Gc
        l0 = mat.l0

        def closure():
            optimizer.zero_grad()

            # Enforce constraints
            with torch.no_grad():
                u_opt.data[bc_mask] = bc_vals[bc_mask]
                d_opt.data.copy_(torch.maximum(d_opt.data, d_prev).clamp(max=1.0))

            # Elastic energy: sum_e [g(d_e) * psi+(eps_e) + psi-(eps_e)] * A_e
            strain = fem.compute_strain(u_opt)
            psi_plus = fem.compute_psi_plus(u_opt, strain=strain, d=d_opt)

            psi_minus = fem._psi_minus_for_energy(strain, d_opt, psi_plus)

            d_e = d_opt[mesh.elements].mean(1)
            g_d = mat.degradation(d_e)

            E_elastic = ((g_d * psi_plus + psi_minus) * mesh.areas).sum()

            d_e_cm = d_opt[mesh.elements]
            d_sum_cm = d_e_cm.sum(1)
            Kd = fem.laplacian_matvec(d_opt)
            if getattr(mat, 'pf_model', 'AT2') == 'AT1':
                E_surf = (3.0 * Gc / (8.0 * l0)
                          * ((mesh.areas / 3.0) * d_sum_cm).sum())
                E_grad = 3.0 * Gc * l0 / 8.0 * torch.dot(d_opt, Kd)
            else:
                # Consistent mass for T3: area/12 * (d_sum^2 + d_sq),
                # matching PhaseFieldDamageSolver.
                d_sq_cm = (d_e_cm ** 2).sum(1)
                E_surf = (Gc / (2.0 * l0)
                          * ((mesh.areas / 12.0)
                             * (d_sum_cm ** 2 + d_sq_cm)).sum())
                E_grad = Gc * l0 / 2.0 * torch.dot(d_opt, Kd)

            # External work
            E_ext = torch.tensor(0.0, device=u_opt.device, dtype=u_opt.dtype)
            if f_ext is not None:
                E_ext = -(f_ext * u_opt).sum()

            E_total = E_elastic + E_surf + E_grad + E_ext
            E_total.backward()

            # Zero gradients at BC DOFs
            if u_opt.grad is not None:
                u_opt.grad.data[bc_mask] = 0.0

            # Zero gradient for clamped damage DOFs to prevent L-BFGS fighting the clamp
            if d_opt.grad is not None:
                with torch.no_grad():
                    at_lower = d_opt.data <= d_prev + 1e-10
                    at_upper = d_opt.data >= 1.0 - 1e-10
                    d_opt.grad[at_lower & (d_opt.grad < 0)] = 0.0
                    d_opt.grad[at_upper & (d_opt.grad > 0)] = 0.0

            n_eval[0] += 1
            grad_norm_u = u_opt.grad.data.norm().item() if u_opt.grad is not None else 0.0
            grad_norm_d = d_opt.grad.data.norm().item() if d_opt.grad is not None else 0.0
            final_grad[0] = max(grad_norm_u, grad_norm_d)

            return E_total

        optimizer.step(closure)

        # Final constraint enforcement
        with torch.no_grad():
            u_opt.data[bc_mask] = bc_vals[bc_mask]
            d_opt.data.copy_(torch.maximum(d_opt.data, d_prev).clamp(max=1.0))

        converged = final_grad[0] < self.tol
        return u_opt.detach(), d_opt.detach(), converged, n_eval[0]


# -----------------------------------------------------------------------------
# Issue #228: custom-adjoint wrapper for the chunked explicit-dynamics forward.
#
# v1 scope (elastic-only): wrap a deterministic chunk forward
#     (u_in, v_in, a_in, d_in, He_in, Hn_in, E_field) -> chunk_outs
# in a torch.autograd.Function. Forward runs under no_grad and saves the
# input state. Backward recomputes the chunk under grad and uses
# torch.autograd.grad to push cotangents onto the inputs and parameters.
#
# Functionally this is equivalent to torch.utils.checkpoint.checkpoint.
# Its purpose is structural: it isolates the chunked time-loop behind a
# single autograd Function so that v2 (damage-on) can swap the backward
# for a true discrete-time adjoint that composes with the existing
# damage implicit-diff (`_AdjointDamageSolve*` in damage_solver.py).
#
# See docs/adjoint_explicit_dynamics_derivation.md for the derivation.
# -----------------------------------------------------------------------------


class ChunkedExplicitDynamicsAutograd(torch.autograd.Function):
    """Custom-adjoint wrapper around a chunked explicit-dynamics forward.

    The chunk function ``chunk_fn`` is a closure over the staggered solver
    that maps tensor state ``(u, v, a, d, He, Hn)`` plus an ``E_field``
    parameter tensor to a tuple of tensor outputs ``(u', v', a', d',
    He', Hn', n_snaps, *snaps, *loads)``.

    v1 backward path: recompute ``chunk_fn`` with a tape, then call
    ``torch.autograd.grad`` to obtain input/parameter cotangents. This
    is bit-equivalent to ``torch.utils.checkpoint`` on the forward
    state, but it owns its own autograd boundary so v2 can replace the
    backward with a discrete-time adjoint that handles damage saturation
    (issue #228) without an unrolled tape.
    """

    @staticmethod
    def forward(ctx, chunk_fn, n_outs, E_field,
                u_in, v_in, a_in, d_in, He_in, Hn_in):
        # Save closures and the chunk-input state. We rerun under-grad
        # in backward(), so we don't need to retain forward activations.
        ctx.chunk_fn = chunk_fn
        ctx.n_outs = n_outs
        ctx.save_for_backward(E_field, u_in, v_in, a_in, d_in, He_in, Hn_in)
        with torch.no_grad():
            outs = chunk_fn(u_in, v_in, a_in, d_in, He_in, Hn_in, E_field)
        # Detach all tensor outputs (we are inside no_grad already, but
        # be explicit). Non-tensor outputs (n_snaps int) are passed
        # through verbatim.
        out_list = []
        for o in outs:
            if torch.is_tensor(o):
                out_list.append(o.detach())
            else:
                out_list.append(o)
        return tuple(out_list)

    @staticmethod
    def backward(ctx, *grad_outs):
        chunk_fn = ctx.chunk_fn
        E_field, u_in, v_in, a_in, d_in, He_in, Hn_in = ctx.saved_tensors

        # Detach + require_grad on the inputs we want cotangents for.
        # E_field is the parameter; the state inputs are the chunk
        # boundary that the *next* chunk's backward will consume.
        # All inputs get requires_grad=True so torch.autograd.grad accepts
        # them; allow_unused=True handles the case where an input does not
        # actually flow into any output (e.g. d_in in the elastic-only
        # path with damage disabled).
        E_g  = E_field.detach().clone().requires_grad_(True)
        u_g  = u_in.detach().clone().requires_grad_(True)
        v_g  = v_in.detach().clone().requires_grad_(True)
        a_g  = a_in.detach().clone().requires_grad_(True)
        d_g  = d_in.detach().clone().requires_grad_(True)
        He_g = He_in.detach().clone().requires_grad_(True)
        Hn_g = Hn_in.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            outs = chunk_fn(u_g, v_g, a_g, d_g, He_g, Hn_g, E_g)
            # Pair tensor outputs with their cotangents; drop non-tensor
            # outputs (e.g. the snap-count int) which receive no grad.
            tangent_pairs = []
            for o, g in zip(outs, grad_outs):
                if torch.is_tensor(o) and g is not None and o.requires_grad:
                    tangent_pairs.append((o, g))
            if not tangent_pairs:
                grads_in = (None,) * 7
            else:
                tensors  = [p[0] for p in tangent_pairs]
                cotans   = [p[1] for p in tangent_pairs]
                inputs   = [E_g, u_g, v_g, a_g, d_g, He_g, Hn_g]
                grads_in = torch.autograd.grad(
                    tensors, inputs,
                    grad_outputs=cotans,
                    retain_graph=False,
                    allow_unused=True,
                )

        # forward signature was (chunk_fn, n_outs, E_field, u, v, a, d, He, Hn).
        # First two are non-tensor / Python objects -> None gradients.
        return (None, None, *grads_in)
