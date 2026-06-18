"""Mesh-level J2 elastoplastic mechanics and ductile PF coupling helpers.

This module is the first production slice for the beta plasticity validation
gates:

* Gate 1: per-element J2 state, commit/rollback, stress update, internal force,
  and plastic-work accounting on a mesh.
* Gate 2: a documented ductile phase-field driving force that couples elastic
  tensile energy with accumulated plastic work and can drive the bounded
  damage solve used by the validation example.

It intentionally does not replace :class:`StaggeredSolver` yet. The global
staggered Newton/secant tangent integration remains a follow-up; these helpers
provide the validated stateful constitutive layer needed for that work.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from phast.fem_operators import FEMOperators
from phast.material import Material
from phast.mesh import FEMMesh
from phast.plasticity.j2_vonmises import J2Plasticity


@dataclass
class MeshJ2State:
    """Per-element elastoplastic state at one integration point per element."""

    strain: torch.Tensor
    stress: torch.Tensor
    plastic_strain: torch.Tensor
    eps_p_eq: torch.Tensor
    plastic_work_density: torch.Tensor

    @classmethod
    def zeros(cls, n_elems: int, *, device=None,
              dtype: torch.dtype = torch.float64) -> "MeshJ2State":
        return cls(
            strain=torch.zeros((n_elems, 6), device=device, dtype=dtype),
            stress=torch.zeros((n_elems, 6), device=device, dtype=dtype),
            plastic_strain=torch.zeros((n_elems, 6), device=device, dtype=dtype),
            eps_p_eq=torch.zeros(n_elems, device=device, dtype=dtype),
            plastic_work_density=torch.zeros(n_elems, device=device, dtype=dtype),
        )

    def clone(self) -> "MeshJ2State":
        return MeshJ2State(
            strain=self.strain.clone(),
            stress=self.stress.clone(),
            plastic_strain=self.plastic_strain.clone(),
            eps_p_eq=self.eps_p_eq.clone(),
            plastic_work_density=self.plastic_work_density.clone(),
        )


def strain3d_from_mesh(mesh: FEMMesh, u: torch.Tensor) -> torch.Tensor:
    """Return element strains in Voigt-6 format for a 2D mesh displacement."""

    gp = mesh.grad_phi
    u_e = u[mesh.elements]
    eps_xx = (gp[:, :, 0] * u_e[:, :, 0]).sum(1)
    eps_yy = (gp[:, :, 1] * u_e[:, :, 1]).sum(1)
    gam_xy = (
        (gp[:, :, 1] * u_e[:, :, 0]).sum(1)
        + (gp[:, :, 0] * u_e[:, :, 1]).sum(1)
    )
    strain = torch.zeros((mesh.n_elems, 6), device=mesh.device, dtype=mesh.dtype)
    strain[:, 0] = eps_xx
    strain[:, 1] = eps_yy
    strain[:, 3] = gam_xy
    return strain


class MeshJ2Elastoplasticity:
    """Stateful one-point-per-element J2 mechanics layer."""

    def __init__(self, mesh: FEMMesh, material: Material):
        if material.plasticity_model == "none":
            raise ValueError(
                "MeshJ2Elastoplasticity requires material.plasticity_model != 'none'"
            )
        self.mesh = mesh
        self.material = material
        self.kernel = J2Plasticity(
            material, plane_stress=bool(getattr(material, "plane_stress", False)),
            dtype=mesh.dtype,
        )
        self.state = MeshJ2State.zeros(
            mesh.n_elems, device=mesh.device, dtype=mesh.dtype)
        self._trial_state: MeshJ2State | None = None

    def update_trial(self, u: torch.Tensor) -> MeshJ2State:
        """Return and store a trial updated state for displacement ``u``."""

        strain_np1 = strain3d_from_mesh(self.mesh, u)
        stress, plastic_strain, eps_p_eq = self.kernel.step(
            self.state.strain,
            strain_np1,
            self.state.stress,
            self.state.plastic_strain,
            self.state.eps_p_eq,
        )
        d_plastic_strain = plastic_strain - self.state.plastic_strain
        d_work = torch.sum(stress * d_plastic_strain, dim=1).clamp_min(0.0)
        plastic_work_density = self.state.plastic_work_density + d_work
        self._trial_state = MeshJ2State(
            strain=strain_np1,
            stress=stress,
            plastic_strain=plastic_strain,
            eps_p_eq=eps_p_eq,
            plastic_work_density=plastic_work_density,
        )
        return self._trial_state

    def commit(self) -> MeshJ2State:
        """Commit the most recent trial state."""

        if self._trial_state is None:
            raise RuntimeError("No trial state to commit")
        self.state = self._trial_state
        self._trial_state = None
        return self.state

    def rollback(self) -> None:
        """Discard the most recent trial state."""

        self._trial_state = None

    def internal_force(self, d: torch.Tensor | None = None,
                       *, state: MeshJ2State | None = None) -> torch.Tensor:
        """Assemble nodal internal force from the current/trial stress field.

        If ``d`` is supplied, the in-plane stress is degraded with the
        material degradation function. This is the explicit ductile-PF coupling
        used by the Gate 2 helper; compression/contact refinements remain
        formulation work for the global solver integration.
        """

        st = state if state is not None else (self._trial_state or self.state)
        stress = st.stress
        sxx = stress[:, 0]
        syy = stress[:, 1]
        sxy = stress[:, 3]
        if d is not None:
            d_e = d[self.mesh.elements].mean(dim=1)
            g = self.material.degradation(d_e)
            sxx = g * sxx
            syy = g * syy
            sxy = g * sxy

        gp = self.mesh.grad_phi
        f_e = torch.zeros(
            (self.mesh.n_elems, 3, 2), device=self.mesh.device, dtype=self.mesh.dtype)
        f_e[:, :, 0] = (gp[:, :, 0] * sxx.unsqueeze(1)
                        + gp[:, :, 1] * sxy.unsqueeze(1))
        f_e[:, :, 1] = (gp[:, :, 1] * syy.unsqueeze(1)
                        + gp[:, :, 0] * sxy.unsqueeze(1))
        f_e = f_e * self.mesh.areas.view(-1, 1, 1)

        out = torch.zeros(
            (self.mesh.n_nodes, 2), device=self.mesh.device, dtype=self.mesh.dtype)
        out.scatter_add_(
            0,
            self.mesh.elements.reshape(-1, 1).expand(-1, 2),
            f_e.reshape(-1, 2),
        )
        return out

    def inplane_algorithmic_tangent(self, u: torch.Tensor) -> torch.Tensor:
        """Return ``d[sxx, syy, sxy] / d[exx, eyy, gxy]`` per element.

        Plane strain uses the 3x3 slice of the kernel's full 6x6 consistent
        tangent. Plane stress reduces the same 6x6 tangent analytically with a
        Schur complement over the out-of-plane strain component, which is the
        Simo-Taylor plane-stress reduction. This keeps the mesh tangent fully
        vectorized over elements without an additional autograd loop in this
        layer.
        """

        strain_base = strain3d_from_mesh(self.mesh, u).detach()
        strain_n = self.state.strain.detach()
        stress_n = self.state.stress.detach()
        plastic_strain_n = self.state.plastic_strain.detach()
        eps_p_eq_n = self.state.eps_p_eq.detach()

        _, _, _, C6 = self.kernel.step_with_tangent(
            strain_n, strain_base, stress_n, plastic_strain_n, eps_p_eq_n)
        inplane = torch.tensor([0, 1, 3], device=self.mesh.device,
                               dtype=torch.long)
        C_aa = C6.index_select(-2, inplane).index_select(-1, inplane)
        if not self.kernel.plane_stress:
            return C_aa

        # Plane-stress reduction: eliminate epsilon_zz via the converged
        # sigma_zz = 0 constraint. Since the kernel already returns the full
        # consistent 6x6 tangent, the reduced in-plane tangent is the Schur
        # complement over the out-of-plane strain component.
        c_a3 = C6.index_select(-2, inplane).select(-1, 2)
        c_3a = C6.select(-2, 2).index_select(-1, inplane)
        c_33 = C6.select(-1, 2).select(-1, 2)
        c_33 = torch.clamp(c_33, min=1.0e-300)
        return C_aa - c_a3.unsqueeze(-1) * c_3a.unsqueeze(-2) / c_33.view(
            *c_33.shape, 1, 1)

    def element_B_matrices(self) -> torch.Tensor:
        """Return linear T3 strain-displacement matrices, shape ``(E, 3, 6)``."""

        gp = self.mesh.grad_phi
        B = torch.zeros(
            (self.mesh.n_elems, 3, 6), device=self.mesh.device,
            dtype=self.mesh.dtype)
        for i in range(3):
            B[:, 0, 2 * i] = gp[:, i, 0]
            B[:, 1, 2 * i + 1] = gp[:, i, 1]
            B[:, 2, 2 * i] = gp[:, i, 1]
            B[:, 2, 2 * i + 1] = gp[:, i, 0]
        return B

    def assembly_indices(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return global COO rows/cols for all element ``6 x 6`` blocks."""

        elem_dofs = torch.empty(
            (self.mesh.n_elems, 6), device=self.mesh.device, dtype=torch.long)
        elems = self.mesh.elements.to(device=self.mesh.device)
        for i in range(3):
            elem_dofs[:, 2 * i] = 2 * elems[:, i]
            elem_dofs[:, 2 * i + 1] = 2 * elems[:, i] + 1
        rows = elem_dofs.repeat_interleave(6, dim=1).reshape(-1)
        cols = elem_dofs.repeat(1, 6).reshape(-1)
        return rows, cols, 2 * self.mesh.n_nodes

    def assemble_tangent(self, u: torch.Tensor,
                         d: torch.Tensor | None = None) -> torch.Tensor:
        """Assemble sparse global elastoplastic tangent ``d f_int / d u``.

        The tangent is built from the last committed plastic state and the
        trial displacement ``u``; it does not commit or mutate history.
        """

        C = self.inplane_algorithmic_tangent(u)
        B = self.element_B_matrices()
        CB = torch.einsum("eij,ejk->eik", C, B)
        Ke = torch.einsum("eji,ejk->eik", B, CB)
        scale = self.mesh.areas
        if d is not None:
            d_e = d[self.mesh.elements].mean(dim=1)
            scale = scale * self.material.degradation(d_e)
        Ke = Ke * scale.view(-1, 1, 1)

        rows, cols, n_dof = self.assembly_indices()
        indices = torch.stack([rows, cols], dim=0)
        return torch.sparse_coo_tensor(
            indices, Ke.reshape(-1), (n_dof, n_dof),
            device=self.mesh.device, dtype=self.mesh.dtype).coalesce()


class SparseJ2QuasiStaticSolver:
    """Sparse Newton solver for small-strain J2 mechanics.

    The solver assembles a sparse global stiffness from per-element
    algorithmic tangents of the return-mapping kernel, solves the Newton
    system through the package sparse backend abstraction
    (auto/SciPy/PETSc-MUMPS/cuDSS where functional), and commits the J2 state
    exactly once after a converged load step. The implementation is
    intentionally separate from ``StaggeredSolver`` until the damage-coupled
    tangent path has the same validation coverage.
    """

    def __init__(
        self,
        plasticity: MeshJ2Elastoplasticity,
        *,
        tol: float = 1.0e-7,
        tol_rel: float = 1.0e-6,
        max_iter: int = 25,
        line_search: bool = True,
        line_search_max_steps: int = 8,
        line_search_min_alpha: float = 1.0e-4,
        backend: str = "auto",
    ) -> None:
        self.plasticity = plasticity
        self.tol = float(tol)
        self.tol_rel = float(tol_rel)
        self.max_iter = int(max_iter)
        self.line_search = bool(line_search)
        self.line_search_max_steps = int(line_search_max_steps)
        self.line_search_min_alpha = float(line_search_min_alpha)
        self.backend = backend

        self.last_iter = 0
        self.last_residual = float("inf")
        self.last_residual0 = float("inf")
        self.last_line_search_alpha = 1.0
        self.last_line_search_reductions = 0
        self.last_converged = False
        self.last_failure: str | None = None
        self.last_backend: str | None = None

    @property
    def mesh(self) -> FEMMesh:
        return self.plasticity.mesh

    def _residual(
        self,
        u: torch.Tensor,
        d: torch.Tensor | None,
        f_ext: torch.Tensor,
        free_mask: torch.Tensor,
    ) -> torch.Tensor:
        trial = self.plasticity.update_trial(u)
        residual = f_ext - self.plasticity.internal_force(d=d, state=trial)
        return residual * free_mask

    def _has_converged(self, residual_norm: float, residual0: float) -> bool:
        return (
            residual_norm <= self.tol
            or residual_norm <= self.tol_rel * max(residual0, 1.0)
        )

    def _assemble_sparse_tangent(
        self,
        u: torch.Tensor,
        d: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.plasticity.assemble_tangent(u, d=d)

    def _solve_linear(self, K, residual: torch.Tensor,
                      free_mask: torch.Tensor) -> torch.Tensor:
        from phast.sparse_solve import resolve_sparse_backend, solve

        K = K.coalesce()
        indices = K.indices()
        values = K.values()
        n_dof = int(K.shape[0])
        free = free_mask.reshape(-1).to(device=indices.device, dtype=torch.bool)
        keep = free[indices[0]] & free[indices[1]]
        values_bc = values * keep.to(dtype=values.dtype)

        fixed = torch.nonzero(~free, as_tuple=False).reshape(-1)
        if fixed.numel() > 0:
            fixed_idx = torch.stack([fixed, fixed], dim=0)
            indices_bc = torch.cat([indices, fixed_idx], dim=1)
            values_bc = torch.cat([
                values_bc,
                torch.ones(
                    fixed.numel(), device=values.device, dtype=values.dtype),
            ])
        else:
            indices_bc = indices

        K_bc = torch.sparse_coo_tensor(
            indices_bc, values_bc, (n_dof, n_dof),
            device=values.device, dtype=values.dtype).coalesce()
        rhs = residual.reshape(-1)
        backend = resolve_sparse_backend(
            self.backend, device_type=K_bc.device.type)
        self.last_backend = backend
        du_flat = solve(K_bc, rhs, backend=backend)
        return du_flat.reshape_as(residual)

    def solve(
        self,
        bc_mask: torch.Tensor,
        bc_vals: torch.Tensor,
        *,
        f_ext: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        u_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool, int]:
        """Solve a small displacement-controlled J2 equilibrium load step."""

        mesh = self.mesh
        if f_ext is None:
            f_ext = torch.zeros(
                (mesh.n_nodes, 2), device=mesh.device, dtype=mesh.dtype)
        if d is not None and d.shape != (mesh.n_nodes,):
            raise ValueError("d must have shape (n_nodes,)")
        if bc_mask.shape != (mesh.n_nodes, 2):
            raise ValueError("bc_mask must have shape (n_nodes, 2)")
        if bc_vals.shape != (mesh.n_nodes, 2):
            raise ValueError("bc_vals must have shape (n_nodes, 2)")

        if u_init is None:
            u = torch.zeros(
                (mesh.n_nodes, 2), device=mesh.device, dtype=mesh.dtype)
        else:
            u = u_init.clone()
        u[bc_mask] = bc_vals[bc_mask]

        free_mask = (~bc_mask).to(mesh.dtype)
        n_free = int(free_mask.sum().item())
        residual0: float | None = None
        self.last_failure = None
        self.last_converged = False

        if n_free == 0:
            self.plasticity.update_trial(u)
            self.plasticity.commit()
            self.last_iter = 0
            self.last_residual = 0.0
            self.last_residual0 = 0.0
            self.last_converged = True
            return u, True, 0

        for nr_iter in range(self.max_iter):
            self.last_iter = nr_iter + 1
            residual = self._residual(u, d, f_ext, free_mask)
            res_norm = float(residual.norm().item())
            if residual0 is None:
                residual0 = res_norm
                self.last_residual0 = float(residual0)
            self.last_residual = res_norm

            if self._has_converged(res_norm, residual0):
                self.plasticity.update_trial(u)
                self.plasticity.commit()
                self.last_iter = nr_iter
                self.last_converged = True
                return u, True, nr_iter

            if not torch.isfinite(residual).all():
                self.last_failure = "residual contains non-finite values"
                self.plasticity.rollback()
                return u, False, self.max_iter

            K = self._assemble_sparse_tangent(u, d)
            if K._nnz() == 0:
                self.last_failure = "tangent has no nonzero entries"
                self.plasticity.rollback()
                return u, False, self.max_iter
            du = self._solve_linear(K, residual, free_mask)

            alpha = 1.0
            best_u = None
            best_norm = float("inf")
            reductions = 0
            max_trials = (
                self.line_search_max_steps + 1 if self.line_search else 1)
            for ls_iter in range(max_trials):
                u_trial = u + alpha * du
                u_trial[bc_mask] = bc_vals[bc_mask]
                trial_residual = self._residual(u_trial, d, f_ext, free_mask)
                trial_norm = float(trial_residual.norm().item())
                if torch.isfinite(trial_residual).all() and trial_norm < best_norm:
                    best_norm = trial_norm
                    best_u = u_trial
                if not self.line_search or trial_norm < res_norm:
                    self.last_line_search_alpha = float(alpha)
                    self.last_line_search_reductions = reductions
                    u = u_trial
                    break
                alpha *= 0.5
                reductions = ls_iter + 1
                if alpha < self.line_search_min_alpha:
                    break
            else:
                if best_u is None:
                    self.last_failure = "line search produced no finite trial"
                    self.plasticity.rollback()
                    return u, False, self.max_iter
                u = best_u
                self.last_residual = best_norm
                self.last_line_search_alpha = float(alpha)
                self.last_line_search_reductions = reductions

        self.last_failure = "maximum iterations reached"
        self.plasticity.rollback()
        return u, False, self.max_iter


@dataclass
class DuctilePhaseFieldCoupling:
    """Ductile PF driving-force helper.

    The default driving force is

    ``H_ductile = psi_plus_elastic + plastic_work_weight * Wp_accum``.

    This follows the common first-release engineering choice of coupling
    accumulated plastic dissipation into the damage history while leaving
    fully variational gradient-plasticity and consistent global tangents for
    later solver integration.
    """

    fem: FEMOperators
    plasticity: MeshJ2Elastoplasticity
    plastic_work_weight: float = 1.0

    def driving_force(self, u: torch.Tensor,
                      *, state: MeshJ2State | None = None) -> torch.Tensor:
        st = state if state is not None else (
            self.plasticity._trial_state or self.plasticity.state)
        psi_plus = self.fem.compute_psi_plus(u)
        return psi_plus + float(self.plastic_work_weight) * st.plastic_work_density

    def history_update(self, H_old: torch.Tensor, u: torch.Tensor,
                       *, state: MeshJ2State | None = None) -> torch.Tensor:
        return torch.maximum(H_old, self.driving_force(u, state=state))


__all__ = [
    "DuctilePhaseFieldCoupling",
    "MeshJ2Elastoplasticity",
    "MeshJ2State",
    "SparseJ2QuasiStaticSolver",
    "strain3d_from_mesh",
]
