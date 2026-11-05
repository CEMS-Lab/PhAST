"""
2-Level Geometric Multigrid (GMG) Preconditioner for CG Solvers.

Provides V-cycle preconditioning for the AT2 damage solver and (optionally)
the SecantCG mechanics solver. Replaces the diagonal Jacobi preconditioner
with a 2-level multigrid that captures low-frequency error modes, typically
reducing CG iterations by 5-20x compared to Jacobi.

Architecture:
  Fine level:   existing matrix-free scatter operators + Jacobi smoother
  Coarse level: agglomerated node groups + assembled dense matrix + direct solve

The coarse mesh is built by greedy node aggregation on the fine mesh adjacency
graph. Crack-tip aware: nodes with d > threshold form singleton aggregates,
preserving resolution near the crack front.

Usage::

    from phast.multigrid import NodeAggregation, ScalarMultigrid

    # Build aggregation (once, or when crack topology changes)
    agg = NodeAggregation(mesh)

    # Build multigrid preconditioner
    mg = ScalarMultigrid(mesh, agg, Gc_l0=material.Gc * material.l0)

    # Inside CG loop (damage solver):
    mg.update(reaction_coeff)     # rebuild coarse operator when H changes
    z = mg.vcycle(r, reaction_coeff)  # V-cycle preconditioner application
"""

import torch
import numpy as np


# -----------------------------------------------------------------------
# Shared helpers used by ScalarMultigrid, AMGPreconditioner, AmgXPreconditioner
# -----------------------------------------------------------------------

def _scalar_matvec(d, reaction_coeff, grad_phi, areas_col, elements,
                   elem_flat, n_nodes, Gc_l0, dtype, device, react_buf=None):
    """Matrix-free scalar matvec: (Gc_l0*K_lap + M_react) @ d."""
    d_e = d[elements]
    gd_x = (grad_phi[:, :, 0] * d_e).sum(1)
    gd_y = (grad_phi[:, :, 1] * d_e).sum(1)
    lap_contrib = areas_col * (
        grad_phi[:, :, 0] * gd_x.unsqueeze(1) +
        grad_phi[:, :, 1] * gd_y.unsqueeze(1))
    out = torch.zeros(n_nodes, dtype=dtype, device=device)
    out.scatter_add_(0, elem_flat, lap_contrib.flatten())
    d_sum = d_e.sum(1)
    weighted = (reaction_coeff * d_sum).unsqueeze(1).expand(-1, 3).flatten()
    if react_buf is not None:
        react = react_buf
        react.zero_()
    else:
        react = torch.zeros_like(out)
    react.scatter_add_(0, elem_flat, weighted)
    return Gc_l0 * out + react


def _scalar_jacobi_diag(reaction_coeff, Gc_l0_diag_lap, elem_flat,
                        n_nodes, dtype, device):
    """Compute diagonal of A = Gc_l0*K_lap + M_react."""
    diag_contrib = Gc_l0_diag_lap + reaction_coeff.unsqueeze(1)
    A_diag = torch.zeros(n_nodes, dtype=dtype, device=device)
    A_diag.scatter_add_(0, elem_flat, diag_contrib.flatten())
    return A_diag


def _scalar_spectral_diag(reaction_coeff, K_local, Gc_l0, elem_flat,
                           n_nodes, dtype, device):
    """Spectral preconditioner: per-element upper eigenvalue scaling.

    Instead of using only the diagonal of A (standard Jacobi), this computes a
    conservative Gershgorin upper bound for each element's 3x3 local stiffness
    matrix and scatters it to nodes. This avoids a batched eigensolve in large
    meshes while still accounting for off-diagonal coupling within elements.

    Parameters
    ----------
    reaction_coeff : (E,) per-element reaction coefficient (already scaled).
    K_local : (E, 3, 3) per-element Laplacian matrices.
    Gc_l0 : float or (E,) tensor, diffusion coefficient Gc * l0.
    elem_flat : (3*E,) flattened element connectivity.
    n_nodes : int, number of nodes.
    dtype, device : torch dtype/device for output.

    Returns
    -------
    A_diag : (N,) spectral diagonal approximation.
    """
    # Build per-element local matrix with consistent mass:
    # A_e = Gc_l0 * K_local + rc * [[2,1,1],[1,2,1],[1,1,2]]
    # Gc_l0 may be scalar or (E,) tensor (gamma correction)
    if torch.is_tensor(Gc_l0) and Gc_l0.dim() >= 1:
        Gc_l0_bcast = Gc_l0.view(-1, 1, 1)
    else:
        Gc_l0_bcast = Gc_l0
    mass_template = torch.ones(3, 3, dtype=K_local.dtype, device=K_local.device)
    mass_template += torch.eye(3, dtype=K_local.dtype, device=K_local.device)
    mass_local = reaction_coeff.view(-1, 1, 1) * mass_template.unsqueeze(0)
    A_local = Gc_l0_bcast * K_local + mass_local  # (E, 3, 3)
    # Conservative upper bound for max eigenvalue per element. A short power
    # iteration is faster than eigvalsh but can underestimate badly; the
    # Gershgorin row-sum bound stays safe for a diagonal preconditioner.
    lam_max = A_local.abs().sum(dim=2).max(dim=1).values  # (E,)
    # Scatter element bound to nodes (each node gets sum from adjacent elements)
    lam_contrib = lam_max.unsqueeze(1).expand(-1, 3).flatten()
    A_diag = torch.zeros(n_nodes, dtype=dtype, device=device)
    A_diag.scatter_add_(0, elem_flat, lam_contrib)
    return A_diag


def _safe_cholesky(A, dtype, device):
    """Cholesky factorize with regularization. Returns factor or None.

    Falls back to None on MPS (cholesky_solve not supported there).
    """
    dev_type = device.type if hasattr(device, 'type') else str(device).split(':')[0]
    if dev_type == 'mps':
        return None  # cholesky_solve not implemented on MPS
    try:
        n = A.shape[0]
        reg = 1e-8 * A.diag().abs().max().clamp(min=1e-30)
        return torch.linalg.cholesky(
            A + reg * torch.eye(n, dtype=dtype, device=device))
    except (torch.linalg.LinAlgError, NotImplementedError):
        return None


def _coarse_solve(
        r_coarse, A_chol, A_dense, A_sparse=None, A_sparse_solver=None):
    """Coarse solve: sparse → Cholesky → LU → identity fallback.

    Uses scipy sparse direct solve when A_sparse is provided (large coarse
    grids), otherwise dense Cholesky/LU.
    """
    dev = r_coarse.device
    dev_type = dev.type if hasattr(dev, 'type') else str(dev).split(':')[0]

    # Sparse path (large coarse grids, n_c > 5000)
    if A_sparse_solver is not None:
        r_np = r_coarse.detach().to(
            device='cpu', dtype=torch.float64).numpy()
        e_np = A_sparse_solver.solve(r_np)
        return torch.from_numpy(e_np).to(dtype=r_coarse.dtype, device=dev)
    if A_sparse is not None:
        from scipy.sparse.linalg import spsolve
        r_np = r_coarse.detach().to(
            device='cpu', dtype=torch.float64).numpy()
        e_np = spsolve(A_sparse.tocsc(), r_np)
        return torch.from_numpy(e_np).to(dtype=r_coarse.dtype, device=dev)

    # MPS: always solve on CPU in float64 for stability
    if dev_type == 'mps' and A_dense is not None:
        r_cpu = r_coarse.detach().to(device='cpu', dtype=torch.float64)
        A_cpu = A_dense.detach().to(device='cpu', dtype=torch.float64)
        e_cpu = torch.linalg.solve(A_cpu, r_cpu.unsqueeze(1)).squeeze(1)
        return e_cpu.to(dtype=r_coarse.dtype, device=dev)

    if A_chol is not None:
        try:
            return torch.cholesky_solve(
                r_coarse.unsqueeze(1), A_chol).squeeze(1)
        except NotImplementedError:
            pass
    if A_dense is not None:
        try:
            return torch.linalg.solve(
                A_dense, r_coarse.unsqueeze(1)).squeeze(1)
        except torch._C._LinAlgError:
            # Singular coarse matrix (can happen with AT1 zero-reaction regions).
            # Fall back to diagonal scaling rather than crashing.
            diag = A_dense.diagonal()
            diag = torch.clamp(diag.abs(), min=1e-30)
            return r_coarse / diag
    return r_coarse


def _assemble_sparse_cpu(K_local_cpu, reaction_coeff, Gc_l0, rows, cols, n_nodes):
    """Assemble scalar stiffness as scipy CSR on CPU.

    Uses consistent mass matrix: area/12 * [[2,1,1],[1,2,1],[1,1,2]].
    reaction_coeff = (2H + Gc/l0) * area / 12.
    """
    import scipy.sparse as sp
    rc_cpu = reaction_coeff.detach().cpu()
    # Handle per-element Gc_l0 (gamma correction): reshape to (E, 1, 1)
    if isinstance(Gc_l0, torch.Tensor) and Gc_l0.dim() >= 1:
        Gc_l0 = Gc_l0.detach().cpu().view(-1, 1, 1)
    # Consistent mass: [[2,1,1],[1,2,1],[1,1,2]] * rc
    mass_template = torch.ones(3, 3, dtype=K_local_cpu.dtype)
    mass_template += torch.eye(3, dtype=K_local_cpu.dtype)  # [[2,1,1],[1,2,1],[1,1,2]]
    mass_local = rc_cpu.view(-1, 1, 1) * mass_template.unsqueeze(0)  # (E, 3, 3)
    A_local = Gc_l0 * K_local_cpu + mass_local
    # rows/cols are in (i,j)-major outer, e-major inner order (built by the
    # double loop over i,j in _solve_direct).  A_local is (E, 3, 3) = e-major.
    # Permute to (i, j, e) so that flattening matches rows/cols ordering.
    vals = A_local.permute(1, 2, 0).contiguous().numpy().reshape(-1)
    A_coo = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    return A_coo.tocsr()


def _assemble_sparse_nodal_H(K_local_cpu, H_nodal, elements, areas,
                              Gc_l0_coeff, Gc_l0_ratio,
                              rows, cols, n_nodes):
    """Assemble scalar stiffness with nodal H (triple-product mass).

    Uses ∫ N_i N_j N_k dx integrals for the H-dependent mass matrix.
    Element mass entry M[a,b] = (H_a + H_b + S_H)*A/30 * (1+δ_ab)
                               + Gc/l0 * A/12 * (1+δ_ab)
    where S_H = sum of nodal H in element.
    """
    import scipy.sparse as sp
    dtype = K_local_cpu.dtype

    H_e = H_nodal[elements]               # (E, 3)
    S_H = H_e.sum(dim=1)                  # (E,)

    # H-dependent mass: M_H[a,b] = (H_a + H_b + S_H)*A/30 * (1+δ_ab)
    H_sum_ij = H_e.unsqueeze(2) + H_e.unsqueeze(1)       # (E, 3, 3)
    base_H = (H_sum_ij + S_H.view(-1, 1, 1)) * (areas / 30.0).view(-1, 1, 1)
    eye3 = torch.eye(3, dtype=dtype)
    mass_H = base_H * (1.0 + eye3)                        # (E, 3, 3)

    # Gc/l0 consistent mass: Gc/l0 * A/12 * [[2,1,1],[1,2,1],[1,1,2]]
    mass_template = torch.ones(3, 3, dtype=dtype) + eye3
    if isinstance(Gc_l0_ratio, torch.Tensor) and Gc_l0_ratio.dim() >= 1:
        mass_Gc = (Gc_l0_ratio * areas / 12.0).view(-1, 1, 1) * mass_template
    else:
        mass_Gc = (Gc_l0_ratio * areas / 12.0).view(-1, 1, 1) * mass_template

    # Laplacian: Gc*l0 * K_local
    if isinstance(Gc_l0_coeff, torch.Tensor) and Gc_l0_coeff.dim() >= 1:
        lap = Gc_l0_coeff.view(-1, 1, 1) * K_local_cpu
    else:
        lap = Gc_l0_coeff * K_local_cpu

    A_local = lap + mass_H + mass_Gc                       # (E, 3, 3)
    # rows/cols are in (i,j)-major outer, e-major inner order.
    # Permute A_local from (E, 3, 3) to (i, j, e) = (3, 3, E) before flatten.
    vals = A_local.permute(1, 2, 0).contiguous().numpy().reshape(-1)
    A_coo = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    return A_coo.tocsr()


class NodeAggregation:
    """Greedy node aggregation for 2-level geometric multigrid.

    Groups fine-mesh nodes into aggregates of ~target_ratio nodes each.
    Each aggregate maps to one coarse-level DOF.

    Parameters
    ----------
    mesh : FEMMesh
        Fine mesh.
    damage : (N,) tensor or None
        Current damage field. Nodes with d > damage_threshold form singleton
        aggregates to preserve crack-tip resolution.
    damage_threshold : float
        Damage level above which nodes are not coarsened.
    target_ratio : float
        Target coarsening ratio (fine nodes per aggregate). Default 4.
    device : torch.device or None
        Compute device. Defaults to mesh.device.
    """

    def __init__(self, mesh, damage=None, damage_threshold=0.5,
                 target_ratio=4.0, device=None):
        self.mesh = mesh
        self.damage_threshold = damage_threshold
        self.target_ratio = target_ratio
        self._device = device or mesh.device

        # Build adjacency in CSR format from element connectivity
        self._build_adjacency()
        # Run aggregation
        self.rebuild(damage)

    def _build_adjacency(self):
        """Build node adjacency as CSR for fast NumPy-based aggregation."""
        import scipy.sparse as sp
        elems = self.mesh.elements.cpu().numpy()
        n = self.mesh.n_nodes
        # Build symmetric adjacency in CSR format
        rows = np.concatenate([elems[:, 0], elems[:, 0], elems[:, 1],
                               elems[:, 1], elems[:, 2], elems[:, 2]])
        cols = np.concatenate([elems[:, 1], elems[:, 2], elems[:, 0],
                               elems[:, 2], elems[:, 0], elems[:, 1]])
        data = np.ones(len(rows), dtype=np.int8)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        adj.eliminate_zeros()
        self._adj_indptr = adj.indptr
        self._adj_indices = adj.indices

    def rebuild(self, damage=None):
        """Re-run aggregation, optionally with updated damage field.

        Skips the expensive Python-loop BFS if the set of crack-tip nodes
        (d > threshold) hasn't changed since the last call.

        Parameters
        ----------
        damage : (N,) tensor or None
            Current damage. Nodes with d > threshold become singletons.
        """
        n_nodes = self.mesh.n_nodes

        # Determine which nodes should NOT be coarsened (near crack tip)
        if damage is not None:
            d_np = damage.detach().cpu().numpy()
            no_coarsen = d_np > self.damage_threshold
        else:
            no_coarsen = np.zeros(n_nodes, dtype=bool)

        # Skip rebuild if crack-tip node set hasn't changed
        if hasattr(self, '_prev_no_coarsen') and np.array_equal(no_coarsen, self._prev_no_coarsen):
            return
        self._prev_no_coarsen = no_coarsen.copy()

        # Greedy aggregation using CSR adjacency (avoids Python per-node loops
        # for the singleton and orphan passes; the BFS pass is inherently
        # sequential but uses CSR slicing instead of list-of-lists).
        indptr = self._adj_indptr
        indices = self._adj_indices
        target = int(self.target_ratio)

        agg_id = np.full(n_nodes, -1, dtype=np.int64)
        current_agg = 0

        # First pass (vectorized): singleton aggregates for crack-tip nodes
        singleton_mask = no_coarsen
        n_singletons = singleton_mask.sum()
        if n_singletons > 0:
            agg_id[singleton_mask] = np.arange(n_singletons, dtype=np.int64)
            current_agg = int(n_singletons)

        # Second pass: greedy BFS aggregation (sequential but CSR-based)
        for i in range(n_nodes):
            if agg_id[i] >= 0:
                continue
            agg_id[i] = current_agg
            count = 1
            # CSR neighbor lookup
            for j in range(indptr[i], indptr[i + 1]):
                if count >= target:
                    break
                nb = indices[j]
                if agg_id[nb] < 0 and not no_coarsen[nb]:
                    agg_id[nb] = current_agg
                    count += 1
            current_agg += 1

        # Third pass (vectorized): assign orphans to nearest assigned neighbor
        orphans = np.where(agg_id < 0)[0]
        for i in orphans:
            nbs = indices[indptr[i]:indptr[i + 1]]
            assigned = agg_id[nbs]
            valid = assigned >= 0
            if valid.any():
                agg_id[i] = assigned[valid][0]
            else:
                agg_id[i] = current_agg
                current_agg += 1

        self.n_coarse = current_agg
        self.agg_id = torch.from_numpy(agg_id).to(self._device)
        self.coarsening_ratio = n_nodes / max(self.n_coarse, 1)

    def __repr__(self):
        return (f"NodeAggregation(n_fine={self.mesh.n_nodes}, "
                f"n_coarse={self.n_coarse}, "
                f"ratio={self.coarsening_ratio:.1f})")


class ScalarMultigrid:
    """2-level GMG V-cycle preconditioner for scalar fields (damage solver).

    Designed to replace Jacobi preconditioning in ``PhaseFieldDamageSolver``.

    The fine-level operator is ``A = Gc*l0*K_lap + M_react`` where K_lap is the
    scalar Laplacian and M_react is the consistent mass with element-level
    reaction coefficients.

    Parameters
    ----------
    mesh : FEMMesh
    aggregation : NodeAggregation
    Gc_l0 : float
        Product Gc * l0 (material constant).
    n_pre : int
        Pre-smoothing Jacobi sweeps.
    n_post : int
        Post-smoothing Jacobi sweeps.
    omega : float
        Jacobi damping factor (2/3 is standard for Laplacian-type operators).
    device : torch.device
        Compute device (must match CG device in damage_solver).
    dtype : torch.dtype
        Compute dtype (typically float64).
    """

    def __init__(self, mesh, aggregation, Gc_l0,
                 n_pre=2, n_post=2, omega=2.0/3.0,
                 device=None, dtype=torch.float64):
        self.mesh = mesh
        self.agg = aggregation
        self.Gc_l0 = Gc_l0
        self.n_pre = n_pre
        self.n_post = n_post
        self.omega = omega

        dev = device or mesh.device
        self._device = dev
        self._dtype = dtype

        # Cache mesh data on compute device in compute dtype
        self._grad_phi = mesh.grad_phi.detach().to(dtype=dtype, device=dev)
        self._areas = mesh.areas.detach().to(dtype=dtype, device=dev)
        self._areas_col = self._areas.unsqueeze(1)
        self._elements = mesh.elements.detach().to(
            dtype=torch.long, device=dev)
        self._elem_flat = self._elements.flatten()
        self._n_nodes = mesh.n_nodes

        # Precompute Laplacian diagonal for Jacobi
        gp = self._grad_phi
        self._diag_lap = self._areas_col * (gp[:, :, 0]**2 + gp[:, :, 1]**2)
        self._Gc_l0_diag_lap = Gc_l0 * self._diag_lap

        # Precompute element local Laplacian matrices: (E, 3, 3)
        # K_local[e, a, b] = area_e * (grad_phi_a . grad_phi_b)
        gp_x = gp[:, :, 0]  # (E, 3)
        gp_y = gp[:, :, 1]  # (E, 3)
        # Outer product: (E, 3, 1) @ (E, 1, 3) for each component
        self._K_local = (self._areas_col.unsqueeze(2) *
                         (gp_x.unsqueeze(2) * gp_x.unsqueeze(1) +
                          gp_y.unsqueeze(2) * gp_y.unsqueeze(1)))  # (E, 3, 3)

        # Pre-allocated V-cycle buffers
        self._z_buf = torch.zeros(mesh.n_nodes, dtype=dtype, device=dev)
        self._r_coarse_buf = torch.zeros(
            mesh.n_nodes, dtype=dtype, device=dev)
        self._react_buf = torch.zeros(mesh.n_nodes, dtype=dtype, device=dev)

        # Coarse operator state
        self._A_coarse = None
        self._A_coarse_chol = None
        self._A_coarse_sparse = None
        self._A_coarse_sparse_solver = None
        self._A_diag_inv = None

        print(f"[ScalarMultigrid] Initialized: {self.agg}", flush=True)

    def _compute_jacobi_diag(self, reaction_coeff):
        """Compute diagonal of A = Gc_l0*K_lap + M_react for Jacobi."""
        return _scalar_jacobi_diag(reaction_coeff, self._Gc_l0_diag_lap,
                                   self._elem_flat, self._n_nodes,
                                   self._dtype, self._device)

    def _Ax(self, d, reaction_coeff):
        """Fine-level matvec: (Gc_l0*K_lap + M_react) @ d."""
        return _scalar_matvec(d, reaction_coeff, self._grad_phi,
                              self._areas_col, self._elements,
                              self._elem_flat, self._n_nodes,
                              self.Gc_l0, self._dtype, self._device,
                              react_buf=self._react_buf)

    def update(self, reaction_coeff):
        """Rebuild coarse operator and Jacobi diagonal.

        Must be called whenever reaction_coeff changes (i.e., when H_elem
        changes between damage solve calls).

        Parameters
        ----------
        reaction_coeff : (E,) tensor
            Element reaction coefficients: (2*H_e + Gc/l0) * area_e / 12.
        """
        # Jacobi diagonal (for smoothing)
        A_diag = self._compute_jacobi_diag(reaction_coeff)
        self._A_diag_inv = 1.0 / (A_diag + 1e-30)

        # Build coarse operator via rediscretization
        agg_id = self.agg.agg_id
        n_c = self.agg.n_coarse

        # Map fine element connectivity to coarse indices
        coarse_elems = agg_id[self._elements]  # (E, 3)

        # Element local matrices: A_local = Gc_l0 * K_local + reaction_mass
        # Consistent mass: M_local = rc * [2,1,1;1,2,1;1,1,2]
        # where rc = (2H+Gc/l0)*area/12 (passed as reaction_coeff)
        _cm = torch.tensor([[2, 1, 1], [1, 2, 1], [1, 1, 2]],
                           dtype=self._dtype, device=self._device).unsqueeze(0)  # (1,3,3)
        A_local = self.Gc_l0 * self._K_local + reaction_coeff.view(
            -1, 1, 1) * _cm  # (E, 3, 3)

        # Assemble coarse matrix
        ci = coarse_elems.unsqueeze(2).expand(-1, -1, 3).reshape(-1)  # (E*9,)
        cj = coarse_elems.unsqueeze(1).expand(-1, 3, -1).reshape(-1)  # (E*9,)

        if n_c > 5000:
            # Sparse assembly for large coarse grids (avoids O(n_c^2) memory)
            import scipy.sparse as sp
            ci_np = ci.cpu().numpy()
            cj_np = cj.cpu().numpy()
            vals_np = A_local.reshape(-1).cpu().numpy()
            A_csr = sp.csr_matrix(
                (vals_np, (ci_np, cj_np)), shape=(n_c, n_c))
            self._A_coarse_sparse = A_csr
            try:
                from scipy.sparse.linalg import splu
                self._A_coarse_sparse_solver = splu(A_csr.tocsc())
            except Exception:
                self._A_coarse_sparse_solver = None
            self._A_coarse = None
            self._A_coarse_chol = None
        else:
            # Dense assembly + Cholesky for small coarse grids
            flat_idx = ci * n_c + cj
            A_flat = torch.zeros(n_c * n_c, dtype=self._dtype,
                                 device=self._device)
            A_flat.scatter_add_(0, flat_idx, A_local.reshape(-1))
            self._A_coarse = A_flat.reshape(n_c, n_c)
            self._A_coarse_sparse = None
            self._A_coarse_sparse_solver = None
            self._A_coarse_chol = _safe_cholesky(
                self._A_coarse, self._dtype, self._device)
        return True

    @torch.no_grad()
    def vcycle(self, r_fine, reaction_coeff):
        """Apply one V-cycle: z = M_gmg^{-1} @ r_fine.

        Parameters
        ----------
        r_fine : (N,) residual vector on fine level.
        reaction_coeff : (E,) element reaction coefficients.

        Returns
        -------
        z : (N,) preconditioned vector.
        """
        agg_id = self.agg.agg_id
        A_diag_inv = self._A_diag_inv

        # --- Pre-smoothing: n_pre damped Jacobi sweeps ---
        z = self._z_buf
        z.zero_()
        for _ in range(self.n_pre):
            z.add_(A_diag_inv * (r_fine - self._Ax(z, reaction_coeff)))

        # --- Restrict residual to coarse level ---
        r_smooth = r_fine - self._Ax(z, reaction_coeff)
        n_c = self.agg.n_coarse
        r_coarse = self._r_coarse_buf[:n_c]
        r_coarse.zero_()
        r_coarse.scatter_add_(0, agg_id, r_smooth)  # R @ r

        # --- Coarse solve ---
        A_sp = getattr(self, '_A_coarse_sparse', None)
        A_sp_solver = getattr(self, '_A_coarse_sparse_solver', None)
        e_coarse = _coarse_solve(
            r_coarse, self._A_coarse_chol, self._A_coarse,
            A_sp, A_sp_solver)

        # --- Prolongate correction ---
        z.add_(e_coarse[agg_id])  # P @ e_coarse (piecewise constant injection)

        # --- Post-smoothing: n_post damped Jacobi sweeps ---
        for _ in range(self.n_post):
            z.add_(A_diag_inv * (r_fine - self._Ax(z, reaction_coeff)))

        return z


class AmgXPreconditioner:
    """NVIDIA AmgX preconditioner for the damage CG solver (CUDA-only).

    Uses pyamgx to call NVIDIA's AmgX library. AmgX runs its AMG setup and
    solve on the GPU via cuSPARSE/cuSOLVER. The pyamgx Python bindings
    transfer vectors through numpy, so there is a CPU roundtrip per V-cycle
    call. Despite this overhead, AmgX's optimized GPU kernels and deep
    hierarchy (up to 10 levels) typically outperform the torch-based AMG
    on large meshes (50K+ nodes).

    Requires: ``pip install pyamgx`` (CUDA toolkit + AmgX libs must be installed).
    Falls back to AMGPreconditioner if pyamgx is not available.

    Parameters
    ----------
    mesh : FEMMesh
    Gc_l0 : float
    device : torch.device (must be CUDA)
    dtype : torch.dtype
    """

    _instance_count = 0  # Track instances to avoid double pyamgx.finalize()

    def __init__(self, mesh, Gc_l0, device=None, dtype=torch.float64):
        import pyamgx
        if AmgXPreconditioner._instance_count == 0:
            pyamgx.initialize()
        AmgXPreconditioner._instance_count += 1

        self.mesh = mesh
        self.Gc_l0 = Gc_l0
        dev = device or mesh.device
        if dev.type != 'cuda':
            raise RuntimeError("AmgX requires CUDA device")
        self._device = dev
        self._dtype = dtype
        self._n_nodes = mesh.n_nodes

        # AmgX config: aggregation AMG, single V-cycle as preconditioner
        cfg = pyamgx.Config()
        cfg.create_from_dict({
            "config_version": 2,
            "solver": {
                "solver": "AMG",
                "presweeps": 2,
                "postsweeps": 2,
                "max_iters": 1,
                "cycle": "V",
                "smoother": "JACOBI_L1",
                "scope": "amg",
                "algorithm": "AGGREGATION",
                "selector": "SIZE_2",
                "max_levels": 10,
                "coarsest_sweeps": 5,
                "obtain_timings": 0,
            }
        })
        self._cfg = cfg

        # AmgX resources (tied to current CUDA device)
        self._resources = pyamgx.Resources()
        self._resources.create_simple(cfg)
        self._solver = pyamgx.Solver()
        self._solver.create(self._resources, cfg)
        self._matrix = pyamgx.Matrix()
        self._rhs = pyamgx.Vector()
        self._sol = pyamgx.Vector()

        # For sparse assembly (CPU side, one-time precompute)
        gp_cpu = mesh.grad_phi.detach().to(device='cpu', dtype=dtype)
        areas_col_cpu = mesh.areas.detach().to(
            device='cpu', dtype=dtype).unsqueeze(1)
        gp_x, gp_y = gp_cpu[:, :, 0], gp_cpu[:, :, 1]
        self._K_local_cpu = (areas_col_cpu.unsqueeze(2) *
                             (gp_x.unsqueeze(2) * gp_x.unsqueeze(1) +
                              gp_y.unsqueeze(2) * gp_y.unsqueeze(1)))

        si = mesh.sparse_indices.detach().cpu().numpy()
        self._rows = si[0]
        self._cols = si[1]

        self._rhs.create(self._n_nodes)
        self._sol.create(self._n_nodes)

        print(f"[AmgXPreconditioner] CUDA-native initialized: "
              f"{mesh.n_nodes} nodes, device={dev}", flush=True)

    def update(self, reaction_coeff):
        """Upload assembled sparse matrix to AmgX on GPU."""
        A_csr = _assemble_sparse_cpu(self._K_local_cpu, reaction_coeff,
                                     self.Gc_l0, self._rows, self._cols,
                                     self._n_nodes)
        A_csr.sort_indices()

        self._matrix.create_from_csr(
            A_csr.indptr, A_csr.indices, A_csr.data)
        self._solver.setup(self._matrix)

    @torch.no_grad()
    def vcycle(self, r_fine, reaction_coeff=None):
        """Apply AmgX V-cycle: z = M_amgx^{-1} @ r (GPU-native)."""
        r_np = r_fine.detach().to(device='cpu', dtype=torch.float64).numpy()
        self._rhs.upload(r_np)
        self._sol.upload(np.zeros(self._n_nodes))
        self._solver.solve(self._rhs, self._sol)

        z_np = np.empty(self._n_nodes)
        self._sol.download(z_np)
        return torch.from_numpy(z_np).to(
            dtype=self._dtype, device=self._device)

    def __enter__(self):
        """Enter context: deterministic resource management.

        Usage::

            with AmgXPreconditioner(mesh, Gc_l0) as amgx:
                amgx.update(reaction_coeff)
                z = amgx.vcycle(r)
            # destroy() runs automatically on exit, before the pyamgx
            # module can be torn down by Python shutdown.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: release all AmgX resources deterministically."""
        self.destroy()
        return False  # do not suppress exceptions

    def __del__(self):
        """Fallback cleanup only.

        Python's GC provides no ordering guarantee between this instance
        and the `pyamgx` module itself at interpreter shutdown. If pyamgx
        has already been torn down, calling its C destructors can segfault.
        Prefer the context-manager pattern (``__enter__``/``__exit__``) or an
        explicit ``destroy()`` call; this hook only runs as a last resort and
        bails out cleanly during interpreter finalization.
        """
        try:
            import sys
            if sys.is_finalizing():
                return  # interpreter is shutting down; pyamgx may be unsafe
        except Exception:
            return
        self.destroy()

    def destroy(self):
        """Explicitly release AmgX resources."""
        if not hasattr(self, '_destroyed'):
            self._destroyed = True
            try:
                if hasattr(self, '_solver'):
                    self._solver.destroy()
                if hasattr(self, '_rhs'):
                    self._rhs.destroy()
                if hasattr(self, '_sol'):
                    self._sol.destroy()
                if hasattr(self, '_matrix'):
                    self._matrix.destroy()
                if hasattr(self, '_resources'):
                    self._resources.destroy()
                if hasattr(self, '_cfg'):
                    self._cfg.destroy()
                AmgXPreconditioner._instance_count -= 1
                if AmgXPreconditioner._instance_count <= 0:
                    import pyamgx
                    pyamgx.finalize()
                    AmgXPreconditioner._instance_count = 0
            except Exception:
                pass


class AMGPreconditioner:
    """GPU-native Algebraic Multigrid (AMG) preconditioner for the damage CG solver.

    Uses PyAMG's smoothed aggregation for hierarchy **setup only** (one-time per
    update, CPU). The V-cycle itself runs entirely in torch on the compute device
    (GPU/CPU) — no per-iteration CPU roundtrip.

    Architecture:
      Setup (update):  sparse assembly → PyAMG hierarchy → extract P, R, A_c as
                        torch sparse CSR tensors on compute device
      V-cycle:         Chebyshev smoothing (matrix-free scatter ops on device) +
                        torch.sparse restriction/prolongation + dense coarse solve

    Falls back to ScalarMultigrid (GMG) if PyAMG is not installed.

    Parameters
    ----------
    mesh : FEMMesh
    Gc_l0 : float
        Product Gc * l0 (material constant for Laplacian scaling).
    n_pre, n_post : int
        Pre/post smoothing sweeps.
    smoother : str
        'chebyshev' (default, faster convergence) or 'jacobi'.
    device : torch.device
    dtype : torch.dtype
    """

    def __init__(self, mesh, Gc_l0, n_pre=2, n_post=2,
                 smoother='chebyshev', chebyshev_power_iters=5,
                 device=None, dtype=torch.float64):
        self.mesh = mesh
        self.Gc_l0 = Gc_l0
        self.n_pre = n_pre
        self.n_post = n_post
        self._smoother = smoother
        self._chebyshev_power_iters = max(1, int(chebyshev_power_iters))

        dev = device or mesh.device
        self._device = dev
        self._dtype = dtype
        self._n_nodes = mesh.n_nodes

        # GPU-resident mesh data for matrix-free fine-level matvec
        self._grad_phi = mesh.grad_phi.detach().to(dtype=dtype, device=dev)
        self._areas = mesh.areas.detach().to(dtype=dtype, device=dev)
        self._areas_col = self._areas.unsqueeze(1)
        self._elements = mesh.elements.detach().to(
            dtype=torch.long, device=dev)
        self._elem_flat = self._elements.flatten()

        # Jacobi diagonal geometry (precomputed, device-resident)
        gp = self._grad_phi
        self._Gc_l0_diag_lap = Gc_l0 * self._areas_col * (
            gp[:, :, 0]**2 + gp[:, :, 1]**2)

        # CPU-resident data for sparse assembly during update()
        gp_cpu = mesh.grad_phi.detach().to(device='cpu', dtype=dtype)
        areas_col_cpu = mesh.areas.detach().to(
            device='cpu', dtype=dtype).unsqueeze(1)
        gp_x, gp_y = gp_cpu[:, :, 0], gp_cpu[:, :, 1]
        self._K_local_cpu = (areas_col_cpu.unsqueeze(2) *
                             (gp_x.unsqueeze(2) * gp_x.unsqueeze(1) +
                              gp_y.unsqueeze(2) * gp_y.unsqueeze(1)))

        si = mesh.sparse_indices.detach().cpu().numpy()
        self._rows = si[0]
        self._cols = si[1]

        # Pre-allocated react buffer for matvec
        self._react_buf = torch.zeros(mesh.n_nodes, dtype=dtype, device=dev)

        # GPU-resident AMG operators (populated by update())
        self._P = None            # prolongation (torch sparse CSR, on device)
        self._R = None            # restriction  (torch sparse CSR, on device)
        self._A_coarse = None     # dense coarse operator (on device)
        self._A_coarse_chol = None
        self._A_diag_inv = None   # Jacobi diagonal inverse (on device)
        self._n_coarse = 0
        self._cheb_eig_max = None  # estimated max eigenvalue for Chebyshev

        print(f"[AMGPreconditioner] GPU-native initialized: {mesh.n_nodes} nodes, "
              f"{mesh.n_elems} elements, device={dev}, smoother={smoother}",
              flush=True)

    def _clear_hierarchy(self):
        """Drop coarse AMG operators so the next vcycle is Jacobi-only."""
        self._P = None
        self._R = None
        self._A_coarse = None
        self._A_coarse_chol = None
        self._n_coarse = 0
        self._cheb_eig_max = None

    def _compute_jacobi_diag(self, reaction_coeff):
        """Compute diagonal of A for Jacobi smoothing (on device)."""
        return _scalar_jacobi_diag(reaction_coeff, self._Gc_l0_diag_lap,
                                   self._elem_flat, self._n_nodes,
                                   self._dtype, self._device)

    def _Ax(self, d, reaction_coeff):
        """Matrix-free fine-level matvec: (Gc_l0*K_lap + M_react) @ d (on device)."""
        return _scalar_matvec(d, reaction_coeff, self._grad_phi,
                              self._areas_col, self._elements,
                              self._elem_flat, self._n_nodes,
                              self.Gc_l0, self._dtype, self._device,
                              react_buf=self._react_buf)

    def _estimate_spectral_radius(self, reaction_coeff, n_iters=None):
        """Estimate max eigenvalue of D^{-1}A via power iteration (on device).

        Used to set Chebyshev polynomial bounds. Cost: n_iters matvecs.
        Returns
        -------
        float
            Estimated largest eigenvalue of the diagonally scaled operator.
        """
        if n_iters is None:
            n_iters = self._chebyshev_power_iters
        A_diag_inv = self._A_diag_inv
        v = torch.randn(self._n_nodes, dtype=self._dtype, device=self._device)
        v = v / v.norm()
        lam = 1.0
        for _ in range(n_iters):
            w = A_diag_inv * self._Ax(v, reaction_coeff)
            lam = w.norm().item()
            if lam > 0:
                v = w / lam
        return lam

    def _to_torch_sparse_csr(self, sp_coo):
        """Convert scipy COO to torch sparse CSR on device."""
        import scipy.sparse as sp
        csr = sp_coo.tocsr()
        return torch.sparse_csr_tensor(
            torch.tensor(csr.indptr, dtype=torch.int64, device=self._device),
            torch.tensor(csr.indices, dtype=torch.int64, device=self._device),
            torch.tensor(csr.data, dtype=self._dtype, device=self._device),
            size=csr.shape, device=self._device)

    def update(self, reaction_coeff):
        """Rebuild AMG hierarchy and extract operators onto compute device.

        Setup runs on CPU (PyAMG). Extracted P, R, A_coarse are moved to the
        compute device as torch sparse CSR tensors for optimal GPU SpMV.

        Parameters
        ----------
        reaction_coeff : (E,) tensor
            Element reaction coefficients: (2*H_e + Gc/l0) * area_e / 12.
        """
        import pyamg

        # Guard: if reaction_coeff contains non-finite values (from a diverged
        # mechanics solver), do not build an AMG hierarchy from contaminated
        # coefficients. Keep a sanitized Jacobi diagonal so vcycle() remains a
        # finite preconditioner for the caller's fallback path.
        if not torch.isfinite(reaction_coeff).all():
            finite_mask = torch.isfinite(reaction_coeff)
            n_bad = (~finite_mask).sum().item()
            print(f"  [AMG_QS_FALLBACK] {n_bad} non-finite reaction_coeff "
                  f"entries; clearing AMG hierarchy and using Jacobi only",
                  flush=True)
            if finite_mask.any():
                rc_max = reaction_coeff[finite_mask].abs().max()
                if rc_max.item() == 0:
                    rc_max = torch.tensor(1.0, dtype=reaction_coeff.dtype,
                                          device=reaction_coeff.device)
            else:
                rc_max = torch.tensor(1.0, dtype=reaction_coeff.dtype,
                                      device=reaction_coeff.device)
            reaction_coeff = torch.where(
                finite_mask, reaction_coeff,
                torch.full_like(reaction_coeff, rc_max.item()))
            A_diag = self._compute_jacobi_diag(reaction_coeff)
            self._A_diag_inv = 1.0 / (A_diag + 1e-30)
            self._clear_hierarchy()
            return False

        # Jacobi diagonal (on device, for smoothing)
        A_diag = self._compute_jacobi_diag(reaction_coeff)
        self._A_diag_inv = 1.0 / (A_diag + 1e-30)

        # Build into locals and commit only after every object is valid. This
        # prevents a failed rebuild from leaving a mixed stale/new hierarchy.
        try:
            # --- Sparse assembly on CPU ---
            A_csr = _assemble_sparse_cpu(self._K_local_cpu, reaction_coeff,
                                         self.Gc_l0, self._rows, self._cols,
                                         self._n_nodes)

            # --- Build deep AMG hierarchy (CPU) ---
            ml = pyamg.smoothed_aggregation_solver(A_csr)

            # --- Extract prolongation P as torch sparse CSR on device ---
            P_sp = ml.levels[0].P
            P_new = self._to_torch_sparse_csr(P_sp)
            R_new = self._to_torch_sparse_csr(P_sp.T.tocsr())
            n_coarse_new = P_sp.shape[1]

            # --- Coarse operator: Galerkin R*A*P as dense on device ---
            A_c_sp = ml.levels[1].A.toarray()
            A_coarse_new = torch.tensor(
                A_c_sp, dtype=self._dtype, device=self._device)

            A_coarse_chol_new = _safe_cholesky(
                A_coarse_new, self._dtype, self._device)

            # Estimate spectral radius for Chebyshev smoothing. Temporarily use
            # the new objects so failures are caught before commit.
            cheb_eig_max_new = None
            if self._smoother == 'chebyshev':
                old_P, old_R = self._P, self._R
                old_A = self._A_coarse
                old_chol = self._A_coarse_chol
                old_n = self._n_coarse
                old_eig = self._cheb_eig_max
                self._P = P_new
                self._R = R_new
                self._A_coarse = A_coarse_new
                self._A_coarse_chol = A_coarse_chol_new
                self._n_coarse = n_coarse_new
                self._cheb_eig_max = None
                cheb_eig_max_new = self._estimate_spectral_radius(reaction_coeff)
                self._P, self._R = old_P, old_R
                self._A_coarse = old_A
                self._A_coarse_chol = old_chol
                self._n_coarse = old_n
                self._cheb_eig_max = old_eig
        except Exception as e:
            print(f"  [AMG_QS_FALLBACK] pyAMG failed ({e}); clearing AMG "
                  f"hierarchy and using Jacobi only", flush=True)
            self._clear_hierarchy()
            return False

        self._P = P_new
        self._R = R_new
        self._A_coarse = A_coarse_new
        self._A_coarse_chol = A_coarse_chol_new
        self._n_coarse = n_coarse_new
        self._cheb_eig_max = cheb_eig_max_new

        return True

    def _chebyshev_smooth(self, z, r_fine, reaction_coeff, n_sweeps):
        """Chebyshev polynomial smoother (on device, matrix-free).

        Standard 3-term recurrence: faster convergence than Jacobi for the
        same number of sweeps because it damps all eigenvalue components
        within [alpha, beta] optimally.
        """
        eig_max = self._cheb_eig_max or 2.0
        # Chebyshev bounds: [eig_max/30, 1.1*eig_max] for D^{-1}A
        alpha = eig_max / 30.0
        beta = 1.1 * eig_max
        d_coeff = 2.0 / (beta - alpha)
        c_coeff = (beta + alpha) / (beta - alpha)
        D_inv = self._A_diag_inv

        # Sweep 0: initial direction (Jacobi-like seed)
        r = r_fine - self._Ax(z, reaction_coeff)
        p = d_coeff * D_inv * r
        z.add_(p)
        p_older = torch.zeros_like(z)

        # Sweeps 1+: 3-term Chebyshev recurrence
        # The recurrence uses p (direction from 1 iteration ago) and p_older
        # (direction from 2 iterations ago) to achieve optimal polynomial
        # damping over the eigenvalue interval [alpha, beta].
        rho_prev = 1.0
        for k in range(1, n_sweeps):
            r = r_fine - self._Ax(z, reaction_coeff)
            if k == 1:
                rho = 1.0 / (c_coeff - 1.0 / (4.0 * d_coeff))
            else:
                rho = 1.0 / (c_coeff - 0.25 * rho_prev)
            rho = max(min(rho, 2.0), 0.5)  # clamp to prevent blowup
            p_new = rho * (d_coeff * D_inv * r + p) + (1.0 - rho) * p_older
            p_older.copy_(p)
            p = p_new
            rho_prev = rho
            z.add_(p)

        return z

    @torch.no_grad()
    def vcycle(self, r_fine, reaction_coeff):
        """Apply one AMG V-cycle — fully on compute device (GPU/CPU).

        Fine-level smoothing is matrix-free (Chebyshev or Jacobi via scatter
        ops). Restriction and prolongation use torch sparse CSR matmul.
        Coarse solve is dense Cholesky.

        Parameters
        ----------
        r_fine : (N,) residual vector on compute device.
        reaction_coeff : (E,) element reaction coefficients.

        Returns
        -------
        z : (N,) preconditioned vector on compute device.
        """
        if self._P is None:
            return r_fine * self._A_diag_inv if self._A_diag_inv is not None \
                else r_fine.clone()

        # --- Pre-smoothing ---
        z = torch.zeros_like(r_fine)
        if self._smoother == 'chebyshev' and self._cheb_eig_max is not None:
            z = self._chebyshev_smooth(z, r_fine, reaction_coeff, self.n_pre)
        else:
            omega = 2.0 / 3.0
            A_diag_inv_w = omega * self._A_diag_inv
            for _ in range(self.n_pre):
                z.add_(A_diag_inv_w * (r_fine - self._Ax(z, reaction_coeff)))

        # --- Restrict residual: r_c = R @ (r - A@z) ---
        r_smooth = r_fine - self._Ax(z, reaction_coeff)
        r_coarse = torch.sparse.mm(
            self._R, r_smooth.unsqueeze(1)).squeeze(1)

        # --- Coarse solve (dense, on device) ---
        e_coarse = _coarse_solve(r_coarse, self._A_coarse_chol, self._A_coarse)

        # --- Prolongate correction: z += P @ e_c ---
        z.add_(torch.sparse.mm(
            self._P, e_coarse.unsqueeze(1)).squeeze(1))

        # --- Post-smoothing ---
        if self._smoother == 'chebyshev' and self._cheb_eig_max is not None:
            z = self._chebyshev_smooth(z, r_fine, reaction_coeff, self.n_post)
        else:
            omega = 2.0 / 3.0
            A_diag_inv_w = omega * self._A_diag_inv
            for _ in range(self.n_post):
                z.add_(A_diag_inv_w * (r_fine - self._Ax(z, reaction_coeff)))

        return z


class VectorMultigrid:
    """2-level GMG V-cycle preconditioner for vector fields (mechanics solver).

    Handles (N, 2) displacement fields. Uses the same node aggregation as
    ScalarMultigrid but operates on 2*N DOFs.

    Coarse operator is built via **element-level stiffness rediscretization**:
    for each element, compute K_e = area * B^T * D_secant * B (6x6) and scatter
    into the dense coarse matrix. Cost is O(E) — same as one fine-level matvec.
    This replaces the previous O(n_coarse * matvec) probing approach.

    The effective constitutive matrix D_secant (3x3 per element) is computed by
    probing the secant stress formula with 3 unit strain vectors, handling all
    energy splits (isotropic, amor, spectral, star_convex) uniformly.

    .. note::
        Dense coarse matrix scales as (2*n_coarse)^2. For meshes up to ~10K
        nodes (~2500 coarse nodes, ~5000 DOFs), memory is ~100MB. For larger
        meshes, a sparse coarse operator or deeper hierarchy is needed.

    Parameters
    ----------
    fem : FEMOperators
    aggregation : NodeAggregation
    n_pre : int
        Pre-smoothing Jacobi sweeps.
    n_post : int
        Post-smoothing Jacobi sweeps.
    omega : float
        Jacobi damping factor.
    """

    def __init__(self, fem, aggregation, n_pre=2, n_post=2, omega=2.0/3.0):
        self.fem = fem
        self.agg = aggregation
        self.n_pre = n_pre
        self.n_post = n_post
        self.omega = omega

        self._device = fem.device
        self._dtype = fem.dtype

        # Cache mesh data
        mesh = fem.mesh
        self._elements = mesh.elements
        self._areas = mesh.areas
        self._n_nodes = mesh.n_nodes

        # Precompute B matrices (E, 3, 6): strain-displacement
        self._B = self._build_B_matrices(mesh.grad_phi)

        # Coarse state
        self._A_coarse = None
        self._A_coarse_chol = None
        self._M_inv = None
        self._state = None
        self._bc_mask = None

        print(f"[VectorMultigrid] Initialized (rediscretization): {self.agg}",
              flush=True)

    def _build_B_matrices(self, grad_phi):
        """Build strain-displacement B matrices: (E, 3, 6).

        Maps [u0x, u0y, u1x, u1y, u2x, u2y] to [eps_xx, eps_yy, gam_xy].
        """
        E_count = grad_phi.shape[0]
        B = torch.zeros(E_count, 3, 6, dtype=self._dtype,
                        device=self._device)
        B[:, 0, 0::2] = grad_phi[:, :, 0]      # exx from u_ax
        B[:, 1, 1::2] = grad_phi[:, :, 1]      # eyy from u_ay
        B[:, 2, 0::2] = grad_phi[:, :, 1]      # gxy from u_ax
        B[:, 2, 1::2] = grad_phi[:, :, 0]      # gxy from u_ay
        return B

    def _secant_stress(self, exx, eyy, gxy, g_d, state):
        """Compute secant stress from strain components.

        Replicates fem_operators.secant_matvec stress logic for element-local
        constitutive probing. All inputs are (E,) tensors.

        Returns (sxx, syy, sxy) each (E,).
        """
        split = state['split'] if state else 'isotropic'
        mat = self.fem.material

        if split == 'isotropic':
            lam = mat.lam
            mu = mat.mu
            tr = exx + eyy
            sxx = g_d * (lam * tr + 2 * mu * exx)
            syy = g_d * (lam * tr + 2 * mu * eyy)
            sxy = g_d * mu * gxy

        elif split == 'amor':
            kappa = mat.kappa
            mu = mat.mu
            tr = exx + eyy
            tr_pos = state['trace_pos']
            tr_plus = tr * tr_pos
            tr_minus = tr * (1.0 - tr_pos)
            dev_xx = exx - tr / 3.0
            dev_yy = eyy - tr / 3.0
            sxx = g_d * (kappa * tr_plus + 2 * mu * dev_xx) + kappa * tr_minus
            syy = g_d * (kappa * tr_plus + 2 * mu * dev_yy) + kappa * tr_minus
            sxy = g_d * mu * gxy

        elif split == 'spectral':
            lam = mat.lam
            mu = mat.mu
            exy = gxy / 2.0

            p1_xx = state['p1_xx']
            p1_yy = state['p1_yy']
            p1_xy = state['p1_xy']
            sign1 = state['sign1_pos']
            sign2 = state['sign2_pos']

            e1_p = p1_xx * exx + 2.0 * p1_xy * exy + p1_yy * eyy
            e2_p = ((1.0 - p1_xx) * exx - 2.0 * p1_xy * exy +
                    (1.0 - p1_yy) * eyy)

            e1_p_plus = e1_p * sign1
            e2_p_plus = e2_p * sign2
            e1_p_minus = e1_p * (1.0 - sign1)
            e2_p_minus = e2_p * (1.0 - sign2)

            tr_plus = e1_p_plus + e2_p_plus
            tr_minus = e1_p_minus + e2_p_minus

            exx_plus = e1_p_plus * p1_xx + e2_p_plus * (1.0 - p1_xx)
            eyy_plus = e1_p_plus * p1_yy + e2_p_plus * (1.0 - p1_yy)
            exy_plus = (e1_p_plus - e2_p_plus) * p1_xy

            exx_minus = exx - exx_plus
            eyy_minus = eyy - eyy_plus
            exy_minus = exy - exy_plus

            sxx = (g_d * (lam * tr_plus + 2 * mu * exx_plus) +
                   lam * tr_minus + 2 * mu * exx_minus)
            syy = (g_d * (lam * tr_plus + 2 * mu * eyy_plus) +
                   lam * tr_minus + 2 * mu * eyy_minus)
            sxy = g_d * (2 * mu * exy_plus) + 2 * mu * exy_minus

        elif split == 'star_convex':
            C = self.fem.C
            kappa = mat.kappa
            mu = mat.mu
            tr = exx + eyy
            tension = state['tension']

            sxx_t = g_d * (C[0, 0] * exx + C[0, 1] * eyy)
            syy_t = g_d * (C[1, 0] * exx + C[1, 1] * eyy)
            sxy_t = g_d * (C[2, 2] * gxy)

            dev_xx = exx - tr / 3.0
            dev_yy = eyy - tr / 3.0
            sxx_c = g_d * 2 * mu * dev_xx + kappa * tr
            syy_c = g_d * 2 * mu * dev_yy + kappa * tr
            sxy_c = g_d * mu * gxy

            sxx = torch.where(tension, sxx_t, sxx_c)
            syy = torch.where(tension, syy_t, syy_c)
            sxy = torch.where(tension, sxy_t, sxy_c)

        else:
            raise ValueError(f"Unknown split: {split}")

        return sxx, syy, sxy

    def _compute_effective_D(self, g_d, state):
        """Compute per-element effective constitutive matrix (E, 3, 3).

        Probes the secant stress formula with 3 unit strain vectors to build
        D column by column. Handles all energy splits uniformly. Cost: O(3E).
        """
        E_count = len(g_d)
        D = torch.zeros(E_count, 3, 3, dtype=self._dtype,
                        device=self._device)

        ones = torch.ones(E_count, dtype=self._dtype, device=self._device)
        zeros = torch.zeros(E_count, dtype=self._dtype, device=self._device)

        for col, (exx, eyy, gxy) in enumerate([
            (ones, zeros, zeros),   # unit eps_xx
            (zeros, ones, zeros),   # unit eps_yy
            (zeros, zeros, ones),   # unit gam_xy
        ]):
            sxx, syy, sxy = self._secant_stress(exx, eyy, gxy, g_d, state)
            D[:, 0, col] = sxx
            D[:, 1, col] = syy
            D[:, 2, col] = sxy

        return D

    def update(self, d, secant_state=None, bc_mask=None):
        """Rebuild coarse operator via element-level stiffness rediscretization.

        Cost is O(E) — same as one fine-level matvec. Replaces the previous
        O(n_coarse * matvec) probing approach.

        Parameters
        ----------
        d : (N,) damage field.
        secant_state : dict from freeze_secant_state (optional).
        bc_mask : (N, 2) bool — BC mask for free-DOF enforcement.
        """
        # Jacobi diagonal (for smoothing)
        M_diag = self.fem.stiffness_diagonal(d)
        if bc_mask is not None:
            free_mask = (~bc_mask).to(M_diag.dtype)
            M_diag *= free_mask
        diag_floor = 1e-10 * M_diag.abs().max().clamp(min=1e-30)
        if bc_mask is not None:
            self._M_inv = free_mask / M_diag.clamp(min=diag_floor)
        else:
            self._M_inv = 1.0 / M_diag.clamp(min=diag_floor)

        self._state = secant_state
        self._bc_mask = bc_mask

        # Element degradation
        g_d = (secant_state['g_d'] if secant_state and 'g_d' in secant_state
               else self.fem.material.degradation(
                   d[self._elements].mean(1)))

        # Effective constitutive matrix D_secant (E, 3, 3) via probing
        D = self._compute_effective_D(g_d, secant_state)

        # Element stiffness: K_e = area * B^T @ D @ B → (E, 6, 6)
        B = self._B
        DB = torch.bmm(D, B)                        # (E, 3, 6)
        K_e = torch.bmm(B.transpose(1, 2), DB)      # (E, 6, 6)
        K_e *= self._areas.view(-1, 1, 1)

        # Map element DOFs to coarse DOFs
        agg_id = self.agg.agg_id
        n_c = self.agg.n_coarse
        n_c2 = 2 * n_c
        coarse_elems = agg_id[self._elements]  # (E, 3)

        # Coarse DOF indices: [2*c0, 2*c0+1, 2*c1, 2*c1+1, 2*c2, 2*c2+1]
        coarse_dofs = torch.stack([
            2 * coarse_elems[:, 0], 2 * coarse_elems[:, 0] + 1,
            2 * coarse_elems[:, 1], 2 * coarse_elems[:, 1] + 1,
            2 * coarse_elems[:, 2], 2 * coarse_elems[:, 2] + 1,
        ], dim=1)  # (E, 6)

        # Scatter K_e into dense coarse matrix
        ci = coarse_dofs.unsqueeze(2).expand(-1, -1, 6).reshape(-1)  # (E*36,)
        cj = coarse_dofs.unsqueeze(1).expand(-1, 6, -1).reshape(-1)  # (E*36,)
        flat_idx = ci * n_c2 + cj  # (E*36,)

        A_flat = torch.zeros(n_c2 * n_c2, dtype=self._dtype,
                             device=self._device)
        A_flat.scatter_add_(0, flat_idx, K_e.reshape(-1))
        self._A_coarse = A_flat.reshape(n_c2, n_c2)

        # Zero constrained coarse DOFs: if any fine node in an aggregate
        # is constrained, the corresponding coarse DOF is constrained.
        # Without this, the coarse solve moves constrained DOFs and the
        # fine-level smoother has to correct, degrading convergence.
        # Note: If an aggregate contains both constrained and free nodes, the
        # entire coarse DOF is constrained. This is a known limitation of
        # aggregation-based AMG that can degrade convergence near boundaries
        # when aggregate size is comparable to boundary layer thickness.
        if bc_mask is not None:
            for comp in range(2):
                # Scatter bc_mask per component to coarse DOFs
                fine_constrained = bc_mask[:, comp].to(self._dtype)
                coarse_constr = torch.zeros(n_c, dtype=self._dtype,
                                            device=self._device)
                coarse_constr.scatter_add_(0, agg_id, fine_constrained)
                # Any aggregate with >0 constrained nodes is constrained
                constr_dofs = (coarse_constr > 0)
                coarse_dof_idx = torch.arange(n_c, device=self._device)
                cdofs = 2 * coarse_dof_idx[constr_dofs] + comp
                # Zero rows and cols, set diagonal to 1
                self._A_coarse[cdofs, :] = 0
                self._A_coarse[:, cdofs] = 0
                self._A_coarse[cdofs, cdofs] = 1.0

        self._A_coarse_chol = _safe_cholesky(
            self._A_coarse, self._dtype, self._device)

    def _Ax(self, p):
        """Fine-level vector matvec (uses existing FEM operators)."""
        if self._state is not None:
            Ap = self.fem.secant_matvec(p, self._state)
        else:
            Ap = self.fem.internal_force_linear(p)
        if self._bc_mask is not None:
            Ap *= (~self._bc_mask).to(Ap.dtype)
        return Ap

    @torch.no_grad()
    def vcycle(self, r_fine):
        """Apply one V-cycle for vector fields.

        Parameters
        ----------
        r_fine : (N, 2) residual vector.

        Returns
        -------
        z : (N, 2) preconditioned vector.
        """
        agg_id = self.agg.agg_id
        M_inv = self._M_inv
        n_c = self.agg.n_coarse

        # --- Pre-smoothing: damped Jacobi (omega=2/3 for stability) ---
        omega = 2.0 / 3.0
        z = torch.zeros_like(r_fine)
        for _ in range(self.n_pre):
            z.add_(omega * M_inv * (r_fine - self._Ax(z)))

        # --- Restrict residual to coarse level ---
        r_smooth = r_fine - self._Ax(z)
        r_coarse = torch.zeros(2 * n_c, dtype=self._dtype,
                               device=self._device)
        for comp in range(2):
            r_c = torch.zeros(n_c, dtype=self._dtype, device=self._device)
            r_c.scatter_add_(0, agg_id, r_smooth[:, comp])
            r_coarse[comp::2] = r_c

        # --- Coarse solve ---
        e_coarse = _coarse_solve(r_coarse, self._A_coarse_chol, self._A_coarse)

        # --- Prolongate correction ---
        for comp in range(2):
            z[:, comp].add_(e_coarse[comp::2][agg_id])

        # --- Post-smoothing: damped Jacobi (omega=2/3 for stability) ---
        for _ in range(self.n_post):
            z.add_(omega * M_inv * (r_fine - self._Ax(z)))

        return z
