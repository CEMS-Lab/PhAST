"""
Boundary condition specification for the phase-field solver.

Supports:
  - Dirichlet (fixed displacement) on named node sets or coordinate ranges
  - Neumann (traction/pressure) on boundary edges
  - Time-varying (ramp) loading via a load_factor callback
  - Symmetric tension, shear, and custom BC presets
"""

import torch
from typing import Optional, Callable, List, Dict, Tuple

# Forward declaration for type hints in helper functions defined below
# (NeumannBC/RigidConnector use Optional / Dict from typing).


class DirichletBC:
    """A single Dirichlet boundary condition.

    Parameters
    ----------
    node_indices : tensor of int
        Global node indices where this BC applies.
    component : int
        DOF component (0 = x, 1 = y).
    value : float
        Prescribed value at load_factor = 1.0.
    """

    def __init__(self, node_indices: torch.Tensor, component: int,
                 value: float):
        self.node_indices = node_indices
        self.component = component
        self.value = value

    def __repr__(self):
        return (f"DirichletBC(n_nodes={len(self.node_indices)}, "
                f"component={self.component}, value={self.value})")


class NeumannBC:
    """A Neumann (traction) boundary condition on boundary edges.

    Applies a distributed traction vector [tx, ty] on the boundary edges
    connecting the given nodes. The traction is integrated using 1-point
    quadrature (constant over each edge) and assembled into nodal forces.

    Parameters
    ----------
    node_indices : tensor of int
        Boundary node indices (must form connected edges in the mesh).
    traction : list or tensor of 2 floats
        [tx, ty] traction components (force/length in 2D).
    ramp_type : str
        One of ``'constant'``, ``'linear'``, ``'smooth_step'``, ``'cosine'``.
        Selects the time-dependence of the traction magnitude. ``'constant'``
        reproduces the legacy ``neumann`` behaviour (force fully on at t=0).
    t_ramp : float
        Ramp duration in seconds (``s = t / t_ramp`` is the normalized time).
        Ignored for ``'constant'``.
    t_hold : float
        Time at which the traction is held at its ramped value indefinitely.
        For ``ramp_type='cosine'`` this acts as the period: after ``t_hold``
        the load factor stays at 1 (i.e. the cosine ramp ends at t_hold).
        For other ramps, ``t_hold`` defaults to ``t_ramp``.
    """

    def __init__(self, node_indices: torch.Tensor, traction,
                 ramp_type: str = 'constant', t_ramp: float = 0.0,
                 t_hold: Optional[float] = None):
        self.node_indices = node_indices
        self.traction = traction
        self.ramp_type = ramp_type
        self.t_ramp = float(t_ramp)
        self.t_hold = float(t_hold) if t_hold is not None else float(t_ramp)

    def factor_at(self, t: float) -> float:
        """Return the dimensionless ramp factor at time ``t``."""
        return _eval_traction_ramp(t, self.ramp_type, self.t_ramp, self.t_hold)

    def __repr__(self):
        return (f"NeumannBC(n_nodes={len(self.node_indices)}, "
                f"traction={self.traction}, ramp_type={self.ramp_type}, "
                f"t_ramp={self.t_ramp})")


def _eval_traction_ramp(t: float, ramp_type: str,
                        t_ramp: float, t_hold: float) -> float:
    """Evaluate the traction ramp factor at time ``t``.

    Recognised ramp types:

    - ``constant``    : factor = 1 for all t (legacy neumann behaviour).
    - ``linear``      : factor = clamp(t / t_ramp, 0, 1) when t_ramp > 0.
    - ``smooth_step`` : Hermite C1 step ``3s² - 2s³`` with s = t / t_ramp,
      matching ``smooth_step`` / ``smooth_step_tensor`` in this module
      (COMSOL convention; the reference smooth Heaviside used elsewhere).
    - ``cosine``      : ``0.5 * (1 - cos(pi * s))`` with s = t / t_hold,
      i.e. a smooth full-cycle ramp that finishes at t_hold (matches
      ``LoadingConfig.ramp_type='smooth'`` in ``config.py``).
    """
    import math
    rt = (ramp_type or 'constant').lower()
    if rt == 'constant':
        return 1.0
    if rt == 'linear':
        if t_ramp <= 0.0:
            return 1.0
        return max(0.0, min(t / t_ramp, 1.0))
    if rt == 'smooth_step':
        if t_ramp <= 0.0:
            return 1.0
        s = max(0.0, min(t / t_ramp, 1.0))
        return s * s * (3.0 - 2.0 * s)
    if rt == 'cosine':
        period = t_hold if t_hold > 0.0 else t_ramp
        if period <= 0.0:
            return 1.0
        if t >= period:
            return 1.0
        if t <= 0.0:
            return 0.0
        return 0.5 * (1.0 - math.cos(math.pi * t / period))
    raise ValueError(
        f"Unknown traction ramp_type '{ramp_type}'. "
        f"Expected one of: constant, linear, smooth_step, cosine.")


class RigidConnector:
    """Rigid-connector multipoint constraint (master/slave) for 2D.

    Couples a set of slave nodes to a master node such that the slaves
    move as a rigid body with respect to the master. In 2D, the linearised
    constraint about ``theta = 0`` is:

        u_slave_x = u_master_x - theta * (Y_slave - Y_master)
        u_slave_y = u_master_y + theta * (X_slave - X_master)

    Two enforcement modes are supported via ``rotation_free``:

    - ``rotation_free=False`` (the legacy "welded" mode): every slave node
      is locked to the master's prescribed translational displacement on
      the listed ``locked_components``. No rotation freedom; this is what
      :pr:`155` shipped. Existing configs that rely on this need to opt
      in explicitly going forward.

    - ``rotation_free=True`` (default, full MPC): the master gets a free
      rotation DOF ``theta`` (about Z) and every slave node's two DOFs
      are eliminated from the global system via the linearised constraint
      above. Master translational DOFs continue to be Dirichlet-locked
      (driven by ``prescribe``) on whichever components appear in
      ``locked_components``. Implemented by T-matrix master-slave
      elimination across all three solver paths — static
      :class:`mechanics_solver.DirectSolver` (PR #164), explicit-dynamic
      :class:`mechanics_solver.ExplicitDynamics` (PR #174), and
      iterative-CG :class:`mechanics_solver.SecantCGSolver` (PR #182);
      see :meth:`build_T_block`. Tracks issue #154.

    Parameters
    ----------
    master_node : int
        Index of the master/control node. Its coordinates fix the centre
        of rotation; for ``rotation_free=True`` the master must be an
        existing mesh node.
    slave_indices : tensor of int
        Indices of all slave nodes (the master may or may not be in this set;
        it will be removed from the slave set internally).
    locked_components : list of int
        Master translational DOF components to lock (0=x, 1=y). For
        ``rotation_free=False`` these are also the components locked on
        every slave. For ``rotation_free=True`` they are the master DOFs
        that ``prescribe`` drives — slave DOFs are always tied via the
        constraint, regardless of which components are locked.
    prescribe : dict {component: float}
        Per-component prescribed displacement on the master.
        Component values absent from this dict default to 0.
    rotation_free : bool
        If True (default), enable the full master-slave-rotation MPC.
        If False, fall back to the legacy welded behaviour.

    Notes
    -----
    The default flipped from the welded behaviour shipped in :pr:`155`
    (where rotation was implicitly disabled) to the full MPC. To recover
    the legacy behaviour, set ``rotation_free=False`` explicitly in YAML
    or in code.
    """

    def __init__(self, master_node: int, slave_indices: torch.Tensor,
                 locked_components,
                 prescribe: Optional[Dict[int, float]] = None,
                 rotation_free: bool = True):
        self.master_node = int(master_node)
        self.slave_indices = slave_indices
        self.locked_components = list(locked_components)
        self.prescribe = dict(prescribe) if prescribe else {}
        self.rotation_free = bool(rotation_free)

    def slaves_excluding_master(self) -> torch.Tensor:
        """Return slave indices with the master node removed (unique)."""
        flat = self.slave_indices.flatten()
        keep = flat != self.master_node
        return flat[keep].unique()

    def expand_to_dirichlet(self) -> List['DirichletBC']:
        """Expand into per-DOF DirichletBCs.

        - ``rotation_free=False``: legacy welded behaviour. Locks every
          slave + the master on each ``locked_components`` to the
          prescribed value.
        - ``rotation_free=True``: locks only the *master* on each
          ``locked_components`` (slaves are tied through the constraint
          inside the solver, not via Dirichlet).
        """
        bcs = []
        if self.rotation_free:
            master_idx = torch.tensor([self.master_node],
                                      dtype=self.slave_indices.dtype,
                                      device=self.slave_indices.device)
            for c in self.locked_components:
                val = float(self.prescribe.get(c, 0.0))
                bcs.append(DirichletBC(master_idx, c, val))
            return bcs
        # Welded fallback (legacy)
        master_t = torch.tensor([self.master_node],
                                dtype=self.slave_indices.dtype,
                                device=self.slave_indices.device)
        all_idx = torch.cat([master_t, self.slave_indices.flatten()]).unique()
        for c in self.locked_components:
            val = float(self.prescribe.get(c, 0.0))
            bcs.append(DirichletBC(all_idx, c, val))
        return bcs

    def build_T_block(self, mesh, theta_col: int):
        """Build the constraint kinematic block for this connector.

        Returns lists of (row, col, value) triples that populate the
        sparse T matrix for the rotation-free MPC. Each slave node i
        contributes two rows (for u_x and u_y); each row has up to three
        non-zeros (master_x or master_y, theta, and the slave's own
        identity which is *removed*).

        Specifically the linearised constraint is

            u_slave_x = u_master_x - theta * (Y_slave - Y_master)
            u_slave_y = u_master_y + theta * (X_slave - X_master)

        ``theta_col`` is the global column index assigned to this
        connector's rotation DOF (must be unique across all connectors).

        Returns
        -------
        rows, cols, vals : three lists of equal length
            Triples to be added to the global T matrix.
        slave_dof_rows : list of int
            Global flat-DOF row indices that this connector eliminates
            (2*i, 2*i+1 for every slave node).
        master_node : int
        master_dofs : tuple (master_dof_x, master_dof_y) global flat indices.
        theta_col : int
        """
        if not self.rotation_free:
            raise ValueError(
                "build_T_block is only valid for rotation_free=True "
                "rigid connectors.")
        nodes = mesh.nodes.detach().cpu().numpy()
        slaves = self.slaves_excluding_master().detach().cpu().numpy()
        Xm, Ym = float(nodes[self.master_node, 0]), \
            float(nodes[self.master_node, 1])

        master_dof_x = 2 * self.master_node
        master_dof_y = 2 * self.master_node + 1
        rows, cols, vals = [], [], []
        slave_dof_rows = []
        for i in slaves:
            i = int(i)
            Xi, Yi = float(nodes[i, 0]), float(nodes[i, 1])
            row_x = 2 * i
            row_y = 2 * i + 1
            slave_dof_rows.extend([row_x, row_y])
            # u_slave_x row: 1 * u_master_x + (-(Yi - Ym)) * theta
            rows.append(row_x); cols.append(master_dof_x); vals.append(1.0)
            rows.append(row_x); cols.append(theta_col); vals.append(-(Yi - Ym))
            # u_slave_y row: 1 * u_master_y + (Xi - Xm) * theta
            rows.append(row_y); cols.append(master_dof_y); vals.append(1.0)
            rows.append(row_y); cols.append(theta_col); vals.append(Xi - Xm)
        return (rows, cols, vals, slave_dof_rows,
                self.master_node, (master_dof_x, master_dof_y), theta_col)

    def __repr__(self):
        return (f"RigidConnector(master={self.master_node}, "
                f"n_slaves={len(self.slave_indices)}, "
                f"locked={self.locked_components}, "
                f"prescribe={self.prescribe}, "
                f"rotation_free={self.rotation_free})")


class PhaseFieldDirichletBC:
    """Dirichlet BC on the (scalar) phase-field / damage variable.

    Used to lock ``phi = value`` (e.g. ``value = 1.0`` for a sharp
    pre-existing crack, matching the COMSOL pre-crack convention) on a
    set of nodes for the entire simulation. Unlike the displacement
    Dirichlet vocabulary (which is per-component on a 2-DOF vector
    field), the phase field is a scalar so there is no ``component``.

    The constraint is enforced after every damage solve in
    :meth:`StaggeredSolver._apply_pf_dirichlet` (issue #213) — the
    listed nodes have their damage value reassigned to ``value`` and
    the previous-step damage ``d_prev`` is updated in lockstep so
    irreversibility (``d >= d_prev``) stays consistent on the next
    step.
    """

    def __init__(self, node_indices: torch.Tensor, value: float):
        self.node_indices = node_indices
        self.value = float(value)

    def __repr__(self):
        return (f"PhaseFieldDirichletBC(n_nodes={len(self.node_indices)}, "
                f"value={self.value})")


class BoundaryConditions:
    """Collection of Dirichlet and Neumann BCs with time-varying load factor.

    Parameters
    ----------
    n_nodes : int
        Total number of mesh nodes.
    device : str or torch.device
    dtype : torch.dtype
    """

    def __init__(self, n_nodes: int, device=None,
                 dtype: torch.dtype = torch.float64):
        if device is None:
            from ..utils.device import detect_device
            device = detect_device()
        self.n_nodes = n_nodes
        self.device = device
        self.dtype = dtype
        self.bcs: List[DirichletBC] = []
        self.neumann_bcs: List[NeumannBC] = []
        # Rigid connectors with rotation freedom enter the solver via
        # T-matrix elimination (see ``mechanics_solver.DirectSolver``);
        # they are stored separately from the Dirichlet list so the
        # solver can pull them out and build the constraint block.
        self.rigid_connectors: List['RigidConnector'] = []
        # Phase-field Dirichlet BCs (issue #213): scalar damage lock,
        # enforced post-solve in StaggeredSolver._apply_pf_dirichlet.
        self.pf_dirichlet_bcs: List[PhaseFieldDirichletBC] = []
        self._load_factor = 1.0
        self._cached_mask = None
        self._cached_vals = None
        self._cached_lf = None
        self._version = 0

    def add(self, node_indices: torch.Tensor, component: int,
            value: float) -> 'BoundaryConditions':
        """Add a Dirichlet BC. Returns self for chaining."""
        self._cached_mask = None
        self._version += 1
        self.bcs.append(DirichletBC(node_indices, component, value))
        return self

    def fix(self, node_indices: torch.Tensor, component: int) -> 'BoundaryConditions':
        """Fix DOFs to zero (homogeneous Dirichlet)."""
        return self.add(node_indices, component, 0.0)

    def add_neumann(self, node_indices: torch.Tensor, traction,
                    ramp_type: str = 'constant', t_ramp: float = 0.0,
                    t_hold: Optional[float] = None) -> 'BoundaryConditions':
        """Add a Neumann (traction) BC. Returns self for chaining.

        ``ramp_type`` selects a per-BC time-dependence (constant, linear,
        smooth_step, cosine). The legacy ``constant`` value reproduces the
        pre-existing (always-on) behaviour.
        """
        self._cached_mask = None
        self._version += 1
        self.neumann_bcs.append(NeumannBC(node_indices, traction,
                                          ramp_type=ramp_type,
                                          t_ramp=t_ramp, t_hold=t_hold))
        return self

    def add_traction(self, node_indices: torch.Tensor, traction,
                     ramp_type: str = 'constant', t_ramp: float = 0.0,
                     t_hold: Optional[float] = None) -> 'BoundaryConditions':
        """Alias for :meth:`add_neumann` using the new ``traction`` vocabulary."""
        return self.add_neumann(node_indices, traction,
                                ramp_type=ramp_type, t_ramp=t_ramp,
                                t_hold=t_hold)

    def add_symmetry(self, node_indices: torch.Tensor,
                     axis) -> 'BoundaryConditions':
        """Add a symmetry boundary condition on a coordinate-aligned edge.

        Convention (matches the issue spec): ``axis`` names the *normal*
        component to suppress, i.e. the displacement component that the
        symmetry plane prevents.

        - ``axis='y'`` (or 1): edge parallel to the x-axis → fix v=0
          (component 1).
        - ``axis='x'`` (or 0): edge parallel to the y-axis → fix u=0
          (component 0).
        """
        if isinstance(axis, str):
            a = axis.strip().lower()
            if a in ('x', '0'):
                comp = 0
            elif a in ('y', '1'):
                comp = 1
            else:
                raise ValueError(
                    f"symmetry axis must be 'x' or 'y', got {axis!r}")
        else:
            comp = int(axis)
            if comp not in (0, 1):
                raise ValueError(
                    f"symmetry axis component must be 0 or 1, got {comp}")
        return self.fix(node_indices, component=comp)

    def add_rigid_connector(self, master_node: int,
                            slave_indices: torch.Tensor,
                            locked_components,
                            prescribe: Optional[Dict[int, float]] = None,
                            rotation_free: bool = True
                            ) -> 'BoundaryConditions':
        """Add a rigid-connector MPC.

        See :class:`RigidConnector` for full semantics.

        With ``rotation_free=True`` (the default) the master gets a free
        rotation DOF and slave nodes are tied via a master-slave
        elimination performed inside :class:`mechanics_solver.DirectSolver`.
        Only the *master* node is added to the global Dirichlet list (on
        the locked translational components).

        With ``rotation_free=False`` the legacy welded behaviour from PR
        #155 is recovered: every slave + the master is locked on each
        listed component. **Note:** the default flipped to ``True`` in
        PR #154 — configs that depended on the welded behaviour need
        ``rotation_free: false`` set explicitly.
        """
        rc = RigidConnector(master_node, slave_indices,
                            locked_components, prescribe,
                            rotation_free=rotation_free)
        for d in rc.expand_to_dirichlet():
            self._cached_mask = None
            self._version += 1
            self.bcs.append(d)
        if rotation_free:
            self.rigid_connectors.append(rc)
        return self

    def get_active_rigid_connectors(self) -> List['RigidConnector']:
        """Return the rotation-free rigid connectors needing T-matrix MPC."""
        return list(self.rigid_connectors)

    def add_pf_dirichlet(self, node_indices: torch.Tensor,
                         value: float = 1.0) -> 'BoundaryConditions':
        """Add a phase-field Dirichlet BC locking ``phi = value`` on nodes.

        Used to match COMSOL's pre-existing-crack convention (issue
        #213): unlike the IC-only ``preseed_notch_nodesets`` (which
        injects a high-``H`` source at t=0 and lets the bound-clamped
        damage solve drift afterwards), this BC is re-enforced after
        every damage solve so the listed nodes stay pinned for the
        entire simulation. The two mechanisms are independent and may
        coexist (preseed sets the initial elastic state, pf_dirichlet
        locks ``phi = value`` for the duration).
        """
        idx = torch.as_tensor(node_indices, dtype=torch.long,
                              device=self.device).flatten().unique()
        self.pf_dirichlet_bcs.append(PhaseFieldDirichletBC(idx, value))
        return self

    def get_pf_dirichlet_mask_values(
            self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mask, vals)`` for phase-field Dirichlet BCs.

        ``mask`` is ``(N,) bool`` (True where ``phi`` is pinned),
        ``vals`` is ``(N,) self.dtype`` with the prescribed value at
        masked indices and zero elsewhere. When two pf_dirichlet BCs
        target the same node, the *last* one added wins (so the user
        can override defaults by appending later).
        """
        mask = torch.zeros(self.n_nodes, dtype=torch.bool,
                           device=self.device)
        vals = torch.zeros(self.n_nodes, dtype=self.dtype,
                           device=self.device)
        for bc in self.pf_dirichlet_bcs:
            mask[bc.node_indices] = True
            vals[bc.node_indices] = bc.value
        return mask, vals

    def get_neumann_forces(self, mesh, t: Optional[float] = None) -> torch.Tensor:
        """Compute nodal force vector from Neumann BCs.

        Finds boundary edges between nodes in each NeumannBC, computes
        edge lengths, and distributes traction to endpoints using 1-point
        quadrature: f_node = traction * edge_length / 2.

        Parameters
        ----------
        mesh : FEMMesh
            The mesh providing node coordinates and element connectivity.

        Returns
        -------
        f : (N, 2) tensor
            Nodal force contributions from all Neumann BCs.
        """
        f = torch.zeros(self.n_nodes, 2, dtype=self.dtype, device=self.device)
        nodes = mesh.nodes
        elems = mesh.elements  # (E, 3)

        for nbc in self.neumann_bcs:
            idx = nbc.node_indices
            # Per-BC time-dependent ramp factor. If ``t`` is None we keep
            # the legacy behaviour (factor = 1, full traction always on),
            # which is what runs that haven't migrated to time-aware
            # assembly will continue to see.
            if t is not None:
                ramp_factor = nbc.factor_at(t)
            else:
                ramp_factor = 1.0
            t_vec = torch.tensor(nbc.traction, dtype=self.dtype,
                                 device=self.device)

            # Vectorized edge finding: extract all element edges, filter
            # to those where both endpoints are in the Neumann node set
            in_set = torch.zeros(self.n_nodes, dtype=torch.bool,
                                 device=self.device)
            in_set[idx] = True

            # All 3 edges per element: (E,2) pairs for edges 0-1, 1-2, 2-0
            e0, e1, e2 = elems[:, 0], elems[:, 1], elems[:, 2]
            edge_a = torch.cat([e0, e1, e2])  # (3E,)
            edge_b = torch.cat([e1, e2, e0])  # (3E,)

            # Both endpoints in set
            both = in_set[edge_a] & in_set[edge_b]
            ea, eb = edge_a[both], edge_b[both]

            # Deduplicate (keep a < b) — vectorized
            swap = ea > eb
            ea_s = torch.where(swap, eb, ea)
            eb_s = torch.where(swap, ea, eb)
            edge_ids = ea_s * self.n_nodes + eb_s
            unique_ids, inverse, counts = torch.unique(
                edge_ids, return_inverse=True, return_counts=True)
            # Boundary edges are owned by exactly one triangle. Interior edges
            # whose endpoints happen to both lie in the node set belong to two
            # triangles and must not receive traction.
            boundary_mask = counts == 1
            # First occurrence index for each unique edge
            first_occ = torch.zeros(unique_ids.shape[0], dtype=torch.long,
                                    device=edge_ids.device)
            first_occ.scatter_(0, inverse,
                               torch.arange(len(edge_ids), device=edge_ids.device))
            first_occ = first_occ[boundary_mask]
            ea_u, eb_u = ea_s[first_occ], eb_s[first_occ]

            # Compute edge lengths and distribute forces (vectorized)
            if len(ea_u) > 0:
                pa = nodes[ea_u]  # (K, 2)
                pb = nodes[eb_u]  # (K, 2)
                edge_lens = (pa - pb).norm(dim=1)  # (K,)
                force_per_node = (edge_lens / 2.0 * self._load_factor
                                  * ramp_factor
                                  ).unsqueeze(1) * t_vec.unsqueeze(0)  # (K, 2)
                f.scatter_add_(0, ea_u.unsqueeze(1).expand_as(force_per_node),
                               force_per_node)
                f.scatter_add_(0, eb_u.unsqueeze(1).expand_as(force_per_node),
                               force_per_node)
        return f

    @property
    def load_factor(self) -> float:
        return self._load_factor

    @load_factor.setter
    def load_factor(self, value: float):
        self._load_factor = value

    def get_masks_and_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build boolean mask and value arrays for all BCs.

        Returns cached tensors when the load factor and BC list haven't changed.

        Returns
        -------
        bc_mask : (N, 2) bool tensor — True where DOF is constrained
        bc_vals : (N, 2) float tensor — prescribed values (scaled by load_factor)
        """
        # Check if cache is valid (load factor AND BC list version)
        if (self._cached_mask is not None
                and self._cached_lf == self._load_factor
                and getattr(self, '_cached_version', -1) == self._version):
            return self._cached_mask, self._cached_vals

        bc_mask = torch.zeros(
            self.n_nodes, 2, dtype=torch.bool, device=self.device)
        bc_vals = torch.zeros(
            self.n_nodes, 2, dtype=self.dtype, device=self.device)

        for bc in self.bcs:
            idx = bc.node_indices
            c = bc.component
            bc_mask[idx, c] = True
            bc_vals[idx, c] = bc.value * self._load_factor

        self._cached_mask = bc_mask
        self._cached_vals = bc_vals
        self._cached_lf = self._load_factor
        self._cached_version = self._version
        return bc_mask, bc_vals

    def summary(self) -> str:
        lines = [f"BoundaryConditions ({len(self.bcs)} Dirichlet, "
                 f"{len(self.neumann_bcs)} Neumann):"]
        for bc in self.bcs:
            lines.append(f"  {bc}")
        for nbc in self.neumann_bcs:
            lines.append(f"  {nbc}")
        return '\n'.join(lines)

    def __repr__(self):
        return self.summary()


def symmetric_tension_bcs(mesh, disp: float = 0.005) -> BoundaryConditions:
    """Standard symmetric tension test BCs (matches Akantu benchmark).

    - x-displacement fixed on all boundaries
    - y = +disp/2 on top, y = -disp/2 on bottom
    """
    if not mesh.node_sets:
        mesh.identify_boundaries()

    bcs = BoundaryConditions(mesh.n_nodes, mesh.device, mesh.dtype)

    # Note: This over-constrains x-displacement on all boundaries (matching Akantu setup).
    # For standard Miehe benchmark, fix x only on a single node or left boundary.
    for name in ['left', 'right', 'top', 'bottom']:
        if name in mesh.node_sets:
            bcs.fix(mesh.node_sets[name], component=0)

    # Prescribed y-displacement
    if 'top' in mesh.node_sets:
        bcs.add(mesh.node_sets['top'], component=1, value=+disp / 2)
    if 'bottom' in mesh.node_sets:
        bcs.add(mesh.node_sets['bottom'], component=1, value=-disp / 2)

    return bcs


def smooth_step(t: float, t_start: float = 0.0, t_end: float = 1.0) -> float:
    """Smooth Hermite step function for load ramping (COMSOL convention).

    Returns 0 for t <= t_start, 1 for t >= t_end, and a smooth C2 transition
    in between. Avoids spurious high-frequency oscillations from instantaneous
    loading (Borden et al. 2012, COMSOL Application Library).

    Parameters
    ----------
    t : float
        Current time.
    t_start : float
        Start of transition (returns 0 before this).
    t_end : float
        End of transition (returns 1 after this).

    Returns
    -------
    float in [0, 1]

    Reference
    ---------
    COMSOL Multiphysics 6.4, smooth step function with transition zone.
    """
    if t <= t_start:
        return 0.0
    if t >= t_end:
        return 1.0
    s = (t - t_start) / (t_end - t_start)
    # Hermite interpolation: 3s² - 2s³ (C1 smooth)
    return s * s * (3.0 - 2.0 * s)


def smooth_step_tensor(t: 'torch.Tensor', t_start: float = 0.0, t_end: float = 1.0) -> 'torch.Tensor':
    """Vectorized smooth step for tensor inputs."""
    import torch
    s = (t - t_start) / (t_end - t_start)
    s = torch.clamp(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def shear_bcs(mesh, disp: float = 0.005) -> BoundaryConditions:
    """Pure shear test BCs.

    - Bottom fully fixed (x and y)
    - Top: x = +disp, y = 0
    """
    if not mesh.node_sets:
        mesh.identify_boundaries()

    bcs = BoundaryConditions(mesh.n_nodes, mesh.device, mesh.dtype)

    if 'bottom' in mesh.node_sets:
        bcs.fix(mesh.node_sets['bottom'], component=0)
        bcs.fix(mesh.node_sets['bottom'], component=1)

    if 'top' in mesh.node_sets:
        bcs.add(mesh.node_sets['top'], component=0, value=+disp)
        bcs.fix(mesh.node_sets['top'], component=1)

    return bcs
