"""
Mesh loading and FEM precomputation for 2D FEM meshes.

Loads .msh (Gmsh) files via meshio. Precomputes:
  - Element areas, shape function gradients
  - Incircle diameters (for CFL)
  - Lumped scalar mass
  - Node sets (from physical groups or auto-detected boundaries)
"""

import warnings

import torch
import numpy as np
import meshio


class FEMMesh:
    """2D FEM mesh with precomputed geometric quantities.

    Parameters
    ----------
    mesh_path : str
        Path to .msh file (Gmsh format, any version meshio supports).
    device : str or torch.device
        Compute device ('cpu', 'cuda', 'mps'). If None, auto-detects.
    dtype : torch.dtype
        Floating-point precision (default float64 for CG accuracy).
    """

    def __init__(self, mesh_path: str, device=None,
                 dtype: torch.dtype = torch.float64):
        if device is None:
            from ..utils.device import detect_device
            device = detect_device()
        self.device = device
        self.dtype = dtype
        self.mesh_path = mesh_path

        print(f"[FEMMesh] Loading mesh: {mesh_path}", flush=True)
        raw = meshio.read(mesh_path)
        print(f"[FEMMesh] Mesh file read OK", flush=True)

        # --- Nodes (2D) ---
        pts = raw.points[:, :2] if raw.points.shape[1] == 3 else raw.points
        self.nodes = torch.tensor(pts, dtype=dtype, device=device)
        self.n_nodes = self.nodes.shape[0]
        print(f"[FEMMesh] {self.n_nodes} nodes loaded", flush=True)

        # --- 2D cells ---
        self.elements = None
        self.element_type = None
        for cb in raw.cells:
            if cb.type == 'triangle':
                self.elements = torch.tensor(
                    cb.data, dtype=torch.long, device=device)
                self.element_type = 'T3'
                break
            if cb.type in ('quad', 'quadrilateral'):
                self.elements = torch.tensor(
                    cb.data, dtype=torch.long, device=device)
                self.element_type = 'Q4'
                break
        if self.elements is None:
            raise ValueError(f"No supported 2D elements found in {mesh_path}")
        self.n_elems = self.elements.shape[0]
        self.n_elem_nodes = self.elements.shape[1]
        print(f"[FEMMesh] {self.n_elems} {self.element_type} elements loaded",
              flush=True)

        # --- Node sets (physical groups) ---
        self.node_sets = {}
        self._extract_node_sets(raw)
        print(f"[FEMMesh] Node sets: {list(self.node_sets.keys())}", flush=True)

        # --- Precompute FEM quantities ---
        print("[FEMMesh] Precomputing areas, grad_phi, incircle, lumped mass...",
              flush=True)
        self._precompute()
        print(f"[FEMMesh] Precompute done: h_min={self.h_min:.6e}, "
              f"total_area={self.areas.sum():.6f}", flush=True)

    @classmethod
    def from_tensors(cls, nodes: torch.Tensor, elements: torch.Tensor,
                     node_sets: dict = None, device=None,
                     dtype: torch.dtype = torch.float64,
                     element_type: str = 'T3') -> 'FEMMesh':
        """Construct a FEMMesh directly from node/element tensors.

        This bypasses meshio file I/O, which is essential for adaptive
        mesh refinement where the refined mesh is built in-memory.

        Parameters
        ----------
        nodes : torch.Tensor, shape (N, 2)
            Node coordinates.
        elements : torch.Tensor, shape (E, 3) or (E, 4), dtype=torch.long
            T3 triangle or Q4 quadrilateral connectivity.
        node_sets : dict or None
            ``{name: torch.Tensor of node indices}``. If None, empty.
        device : str or torch.device or None
            Compute device. If None, auto-detects.
        dtype : torch.dtype
            Floating-point precision.

        Returns
        -------
        mesh : FEMMesh
            Fully initialised mesh with all precomputed quantities.
        """
        if device is None:
            from ..utils.device import detect_device
            device = detect_device()

        # Bypass __init__ (which requires a file path) by creating a
        # bare instance and populating its attributes directly.
        mesh = object.__new__(cls)
        mesh.device = device
        mesh.dtype = dtype
        mesh.mesh_path = '<from_tensors>'

        # Nodes
        mesh.nodes = nodes.detach().to(dtype=dtype, device=device)
        mesh.n_nodes = mesh.nodes.shape[0]

        element_type = element_type.upper()
        if element_type not in ('T3', 'Q4'):
            raise ValueError(
                f"element_type must be 'T3' or 'Q4', got {element_type!r}")

        # Elements
        mesh.elements = elements.detach().to(dtype=torch.long, device=device)
        expected_nodes = 3 if element_type == 'T3' else 4
        if mesh.elements.ndim != 2 or mesh.elements.shape[1] != expected_nodes:
            raise ValueError(
                f"element_type={element_type!r} expects connectivity shape "
                f"(E, {expected_nodes}), got {tuple(mesh.elements.shape)}")
        mesh.n_elems = mesh.elements.shape[0]
        mesh.element_type = element_type
        mesh.n_elem_nodes = expected_nodes

        # Node sets
        mesh.node_sets = {}
        if node_sets is not None:
            for name, idx in node_sets.items():
                mesh.node_sets[name] = idx.to(dtype=torch.long, device=device)

        # Precompute FEM quantities (areas, grad_phi, incircle, mass, COO)
        mesh._precompute()

        print(f"[FEMMesh.from_tensors] {mesh.n_nodes} nodes, "
              f"{mesh.n_elems} {mesh.element_type} elements, "
              f"h_min={mesh.h_min:.6e}",
              flush=True)
        return mesh

    def _extract_node_sets(self, raw):
        """Extract node sets from meshio point_sets, cell_sets, or gmsh:physical."""
        if hasattr(raw, 'point_sets') and raw.point_sets:
            for name, idx in raw.point_sets.items():
                self.node_sets[name] = torch.tensor(
                    idx, dtype=torch.long, device=self.device)

        if hasattr(raw, 'cell_sets') and raw.cell_sets:
            for name, blocks in raw.cell_sets.items():
                if name in self.node_sets:
                    continue
                # Skip meshio/gmsh internal cell_sets (e.g.
                # ``gmsh:bounding_entities`` emitted by MSH 4.x when
                # ``Physical Point`` groups are present): these are not
                # named node sets and their entries are signed entity
                # tags, not cell indices, which would otherwise raise
                # IndexError when used to index ``cb.data``.
                if name.startswith('gmsh:'):
                    continue
                nodes_in_set = set()
                for i, cb in enumerate(raw.cells):
                    if cb.type in ('line', 'vertex') and i < len(blocks):
                        mask = blocks[i]
                        if isinstance(mask, np.ndarray) and mask.dtype == bool:
                            cell_ids = np.where(mask)[0]
                        elif isinstance(mask, np.ndarray):
                            cell_ids = mask
                        else:
                            continue
                        if len(cell_ids) > 0:
                            nodes_in_set.update(cb.data[cell_ids].flatten().tolist())
                if nodes_in_set:
                    self.node_sets[name] = torch.tensor(
                        sorted(nodes_in_set), dtype=torch.long,
                        device=self.device)

        # Fallback: extract from gmsh:physical cell_data + field_data
        # (MSH 2.2 stores physical group tags per element, not as cell_sets)
        if not self.node_sets and hasattr(raw, 'field_data') and raw.field_data:
            phys_data = raw.cell_data.get('gmsh:physical', [])
            if phys_data:
                # Build reverse map: physical_id -> name (for 0D/1D groups only)
                id_to_name = {}
                for name, info in raw.field_data.items():
                    phys_id, dim = int(info[0]), int(info[1])
                    if dim <= 1:  # points and lines
                        id_to_name[phys_id] = name

                for i, cb in enumerate(raw.cells):
                    if cb.type not in ('line', 'vertex') or i >= len(phys_data):
                        continue
                    tags = phys_data[i]
                    for phys_id, name in id_to_name.items():
                        mask = (tags == phys_id)
                        if mask.any():
                            nodes = set(cb.data[mask].flatten().tolist())
                            if name in self.node_sets:
                                existing = set(self.node_sets[name].tolist())
                                nodes = existing | nodes
                            self.node_sets[name] = torch.tensor(
                                sorted(nodes), dtype=torch.long,
                                device=self.device)

    def _precompute(self):
        if getattr(self, 'element_type', 'T3') == 'Q4':
            self._precompute_q4()
            return
        self._precompute_t3()

    def _precompute_t3(self):
        """Precompute areas, shape function gradients, incircle diameter.

        Optimizations:
          - Reuse edge vectors (v01, v02) for area, grad_phi, and incircle
          - expand+flatten instead of repeat_interleave (avoids large VRAM alloc)
          - Precompute _elem_flat and sparse_indices for assembly ops
        """
        nodes = self.nodes
        elems = self.elements

        p0 = nodes[elems[:, 0]]
        p1 = nodes[elems[:, 1]]
        p2 = nodes[elems[:, 2]]

        # Edge vectors (reused for area, incircle, grad_phi)
        v01 = p1 - p0
        v02 = p2 - p0
        v12 = p2 - p1

        # Signed area via cross product of edge vectors
        signed_2A = v01[:, 0] * v02[:, 1] - v01[:, 1] * v02[:, 0]
        self.areas = 0.5 * torch.abs(signed_2A)
        if (self.areas < 1e-15).any():
            n_degen = (self.areas < 1e-15).sum().item()
            warnings.warn(f"Mesh contains {n_degen} degenerate (near-zero area) elements. "
                          "This may cause numerical instabilities.", stacklevel=2)
        # Regularize while preserving sign (avoids flipping CW elements).
        # For |signed_2A| > 1e-10, use as-is. For near-zero, use sign * 1e-10.
        # For exactly zero (degenerate), default to positive 1e-10.
        sign_2A = torch.sign(signed_2A)
        sign_2A[sign_2A == 0] = 1.0  # degenerate elements default to positive
        inv_2A = 1.0 / torch.where(signed_2A.abs() > 1e-10, signed_2A, sign_2A * 1e-10)

        # Shape function gradients: grad_phi[e, local_node, dim]
        self.grad_phi = torch.zeros(
            self.n_elems, 3, 2, dtype=self.dtype, device=self.device)
        self.grad_phi[:, 0, 0] = (p1[:, 1] - p2[:, 1]) * inv_2A
        self.grad_phi[:, 0, 1] = (p2[:, 0] - p1[:, 0]) * inv_2A
        self.grad_phi[:, 1, 0] = (p2[:, 1] - p0[:, 1]) * inv_2A
        self.grad_phi[:, 1, 1] = (p0[:, 0] - p2[:, 0]) * inv_2A
        self.grad_phi[:, 2, 0] = (p0[:, 1] - p1[:, 1]) * inv_2A
        self.grad_phi[:, 2, 1] = (p1[:, 0] - p0[:, 0]) * inv_2A

        # Incircle diameter: h = 4*Area/perimeter (reuse edge vectors)
        e01 = torch.norm(v01, dim=1)
        e12 = torch.norm(v12, dim=1)
        e20 = torch.norm(v02, dim=1)
        perimeter = e01 + e12 + e20
        self.elem_h = 4.0 * self.areas / (perimeter + 1e-30)
        self.h_min = self.elem_h.min().item()

        # Precompute flattened element indices (used by scatter ops everywhere)
        self._elem_flat = elems.flatten()

        # Lumped scalar mass: expand+flatten avoids repeat_interleave VRAM spike
        self.M_scalar = torch.zeros(
            self.n_nodes, dtype=self.dtype, device=self.device)
        area_third = (self.areas / 3.0).unsqueeze(1).expand(-1, 3).flatten()
        self.M_scalar.scatter_add_(0, self._elem_flat, area_third)

        # Pre-calculate sparse COO indices for optional stiffness assembly
        # i_idx[k], j_idx[k] = (row, col) for the k-th 3x3 element block entry
        i_idx = elems.unsqueeze(2).expand(-1, -1, 3).reshape(-1)  # (E*9,)
        j_idx = elems.unsqueeze(1).expand(-1, 3, -1).reshape(-1)  # (E*9,)
        self.sparse_indices = torch.stack([i_idx, j_idx])  # (2, E*9)

        # SoA layout: (3, E) transpose for coalesced GPU access (#66)
        # When threads process elements in parallel, elements_T[0, :] (all
        # first local nodes) is contiguous — better memory coalescing.
        self.elements_T = self.elements.T.contiguous()

        # Ensure contiguous memory layout for GPU cache efficiency (#26)
        self.areas = self.areas.contiguous()
        self.grad_phi = self.grad_phi.contiguous()
        self.elements = self.elements.contiguous()
        self._elem_flat = self._elem_flat.contiguous()
        self.M_scalar = self.M_scalar.contiguous()

    def _precompute_q4(self):
        """Precompute Q4 quadrature geometry and scalar lumped mass."""
        from .quad_elements import q4_quadrature_geometry

        N_q, gradN_q, wdetJ_q = q4_quadrature_geometry(
            self.nodes, self.elements)
        if (wdetJ_q <= 0).any():
            n_bad = (wdetJ_q <= 0).sum().item()
            warnings.warn(
                f"Mesh contains {n_bad} Q4 quadrature points with non-positive "
                "Jacobian determinant. Check element orientation/quality.",
                stacklevel=2,
            )
        self.quad_N = N_q.to(dtype=self.dtype, device=self.device).contiguous()
        self.quad_grad_phi = gradN_q.to(
            dtype=self.dtype, device=self.device).contiguous()
        self.quad_wdetJ = wdetJ_q.to(
            dtype=self.dtype, device=self.device).contiguous()
        self.areas = self.quad_wdetJ.sum(dim=1).contiguous()
        if (self.areas < 1e-15).any():
            n_degen = (self.areas < 1e-15).sum().item()
            warnings.warn(
                f"Mesh contains {n_degen} degenerate Q4 elements.",
                stacklevel=2,
            )

        coords = self.nodes[self.elements]
        edge_lengths = torch.stack([
            torch.norm(coords[:, 1] - coords[:, 0], dim=1),
            torch.norm(coords[:, 2] - coords[:, 1], dim=1),
            torch.norm(coords[:, 3] - coords[:, 2], dim=1),
            torch.norm(coords[:, 0] - coords[:, 3], dim=1),
        ], dim=1)
        self.elem_h = edge_lengths.min(dim=1).values
        self.h_min = self.elem_h.min().item()

        self._elem_flat = self.elements.flatten()
        self.M_scalar = torch.zeros(
            self.n_nodes, dtype=self.dtype, device=self.device)
        local_mass = torch.einsum(
            'qi,eq->ei', self.quad_N, self.quad_wdetJ)
        self.M_scalar.scatter_add_(0, self._elem_flat, local_mass.flatten())

        elems = self.elements
        i_idx = elems.unsqueeze(2).expand(-1, -1, 4).reshape(-1)
        j_idx = elems.unsqueeze(1).expand(-1, 4, -1).reshape(-1)
        self.sparse_indices = torch.stack([i_idx, j_idx])
        self.elements_T = self.elements.T.contiguous()

        # ``grad_phi`` is constant only for T3. Keep an empty placeholder so
        # unsupported legacy paths fail by element_type guards before use.
        self.grad_phi = torch.empty(
            self.n_elems, 0, 2, dtype=self.dtype, device=self.device)
        self.elements = self.elements.contiguous()
        self._elem_flat = self._elem_flat.contiguous()
        self.M_scalar = self.M_scalar.contiguous()

    def assemble_sparse_matrix(self, elem_values: torch.Tensor) -> torch.Tensor:
        """Assemble a sparse COO matrix from per-element local block values.

        Parameters
        ----------
        elem_values : (E, n_elem_nodes, n_elem_nodes) per-element local matrices

        Returns
        -------
        sparse : (N, N) sparse COO tensor
        """
        values = elem_values.reshape(-1)  # (E*9,)
        return torch.sparse_coo_tensor(
            self.sparse_indices, values,
            size=(self.n_nodes, self.n_nodes),
            dtype=self.dtype, device=self.device,
        ).coalesce()

    def identify_boundaries(self, tol: float = None) -> dict:
        """Auto-detect left/right/top/bottom boundary node sets.

        Only adds boundaries not already present in node_sets.

        Returns
        -------
        dict : name -> tensor of node indices
        """
        if tol is None:
            extent = max(self.nodes.max(0).values - self.nodes.min(0).values)
            tol = 1e-6 * max(extent.item(), 1.0)
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        auto = {
            'left':   torch.where(torch.abs(x - xmin) < tol)[0],
            'right':  torch.where(torch.abs(x - xmax) < tol)[0],
            'bottom': torch.where(torch.abs(y - ymin) < tol)[0],
            'top':    torch.where(torch.abs(y - ymax) < tol)[0],
        }
        for name, idx in auto.items():
            if name not in self.node_sets:
                self.node_sets[name] = idx
        return self.node_sets

    def elem_to_node(self, field_e: torch.Tensor) -> torch.Tensor:
        """Area-weighted projection: element/Gauss-point field -> nodal field.

        T3 accepts ``(E,)`` fields. Q4 accepts either ``(E,)`` element fields
        or ``(E, 4)`` Gauss-point fields and projects with the same 2x2
        quadrature weights used by the native Q4 damage residual.
        """
        if getattr(self, 'element_type', 'T3') == 'Q4':
            if field_e.shape == (self.n_elems,):
                field_q = field_e.unsqueeze(1).expand(-1, self.quad_N.shape[0])
            elif field_e.shape == (self.n_elems, self.quad_N.shape[0]):
                field_q = field_e
            else:
                raise ValueError(
                    "Q4 elem_to_node expects field shape "
                    f"({self.n_elems},) or ({self.n_elems}, {self.quad_N.shape[0]}), "
                    f"got {tuple(field_e.shape)}")
            local = torch.einsum(
                'qa,eq,eq->ea', self.quad_N, field_q, self.quad_wdetJ)
            ns = torch.zeros(self.n_nodes, dtype=self.dtype, device=self.device)
            ns.scatter_add_(0, self._elem_flat, local.flatten())
            return ns / (self.M_scalar + 1e-30)

        w = (field_e * self.areas).unsqueeze(1).expand(-1, 3).flatten()
        ns = torch.zeros(self.n_nodes, dtype=self.dtype, device=self.device)
        ns.scatter_add_(0, self._elem_flat, w)
        return ns / (3.0 * self.M_scalar + 1e-30)

    def compute_field_gradient(self, field: torch.Tensor):
        """Gradient of a nodal scalar field -> (grad_x, grad_y) at nodes.

        Uses element-level gradients with area-weighted L2 projection to nodes.

        NOTE: This performs area-weighted L2 projection (2 scatter passes).
        Cache the result if called repeatedly with the same field — do NOT
        call this inside a Newton-Raphson or CG loop.
        """
        if getattr(self, 'element_type', 'T3') != 'T3':
            raise NotImplementedError(
                "FEMMesh.compute_field_gradient currently supports T3 meshes "
                "only.")
        f_e = field[self.elements]  # (E, 3)
        gx_e = (self.grad_phi[:, :, 0] * f_e).sum(1)  # (E,)
        gy_e = (self.grad_phi[:, :, 1] * f_e).sum(1)
        return self.elem_to_node(gx_e), self.elem_to_node(gy_e)

    def summary(self) -> str:
        lines = [
            f"FEMMesh: {self.mesh_path}",
            f"  {self.n_nodes} nodes, {self.n_elems} elements",
            f"  h_min={self.h_min:.6e}, total_area={self.areas.sum():.6f}",
        ]
        for name, idx in self.node_sets.items():
            lines.append(f"  Node set '{name}': {len(idx)} nodes")
        return '\n'.join(lines)

    def __repr__(self):
        return self.summary()


def build_node_adjacency(elements_np, n_nodes):
    """Build node adjacency lists from triangle element connectivity.

    Uses scipy CSR for vectorized construction, then converts to list-of-lists.

    Parameters
    ----------
    elements_np : (E, 3) numpy array of ints
    n_nodes : int

    Returns
    -------
    neighbors : list of lists, neighbors[i] = sorted unique neighbor node indices
    """
    import scipy.sparse as sp
    import numpy as np
    e0, e1, e2 = elements_np[:, 0], elements_np[:, 1], elements_np[:, 2]
    rows = np.concatenate([e0, e0, e1, e1, e2, e2])
    cols = np.concatenate([e1, e2, e0, e2, e0, e1])
    adj = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                         shape=(n_nodes, n_nodes))
    adj.eliminate_zeros()
    return [adj.indices[adj.indptr[i]:adj.indptr[i+1]].tolist() for i in range(n_nodes)]
