"""Tests for the extended BC vocabulary (issue #138).

Covers:
  - ``symmetry``        — equivalent to ``fix component=<axis>``.
  - ``traction``        — per-BC ramp_type (constant/linear/smooth_step/cosine);
                          regression vs. legacy ``neumann`` for ramp_type=constant.
  - ``rigid_connector`` — simplified master/slave translation MPC: equal
                          displacement on all slave nodes for locked DOFs.
"""

import os
import tempfile
import math
import pytest
import torch


# ---------------------------------------------------------------------------
# symmetry
# ---------------------------------------------------------------------------

class TestSymmetry:
    """`symmetry axis=<a>` is shorthand for `fix component=<a>`."""

    def test_symmetry_y_matches_fix_component_1(self):
        from phast.boundary_conditions import BoundaryConditions

        idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        bcs_sym = BoundaryConditions(n_nodes=10, device='cpu',
                                     dtype=torch.float64)
        bcs_sym.add_symmetry(idx, axis='y')

        bcs_fix = BoundaryConditions(n_nodes=10, device='cpu',
                                     dtype=torch.float64)
        bcs_fix.fix(idx, component=1)

        m_s, v_s = bcs_sym.get_masks_and_values()
        m_f, v_f = bcs_fix.get_masks_and_values()

        assert torch.equal(m_s, m_f)
        assert torch.equal(v_s, v_f)

    def test_symmetry_x_matches_fix_component_0(self):
        from phast.boundary_conditions import BoundaryConditions

        idx = torch.tensor([4, 5, 6], dtype=torch.long)

        bcs_sym = BoundaryConditions(n_nodes=10, device='cpu',
                                     dtype=torch.float64)
        bcs_sym.add_symmetry(idx, axis='x')

        bcs_fix = BoundaryConditions(n_nodes=10, device='cpu',
                                     dtype=torch.float64)
        bcs_fix.fix(idx, component=0)

        m_s, v_s = bcs_sym.get_masks_and_values()
        m_f, v_f = bcs_fix.get_masks_and_values()

        assert torch.equal(m_s, m_f)
        assert torch.equal(v_s, v_f)

    def test_symmetry_invalid_axis_raises(self):
        from phast.boundary_conditions import BoundaryConditions
        bcs = BoundaryConditions(n_nodes=10, device='cpu',
                                 dtype=torch.float64)
        with pytest.raises(ValueError):
            bcs.add_symmetry(torch.tensor([0]), axis='z')


# ---------------------------------------------------------------------------
# traction ramps
# ---------------------------------------------------------------------------

class TestTractionRamp:
    """Per-BC ramp factor evaluation."""

    def test_constant_factor(self):
        from phast.boundary_conditions import _eval_traction_ramp
        for t in (0.0, 1e-9, 1e-3, 1.0):
            assert _eval_traction_ramp(t, 'constant', 0.0, 0.0) == 1.0

    def test_linear_factor(self):
        from phast.boundary_conditions import _eval_traction_ramp
        assert _eval_traction_ramp(0.0, 'linear', 1e-6, 1e-6) == 0.0
        assert _eval_traction_ramp(0.5e-6, 'linear', 1e-6, 1e-6) == pytest.approx(0.5)
        assert _eval_traction_ramp(1e-6, 'linear', 1e-6, 1e-6) == pytest.approx(1.0)
        assert _eval_traction_ramp(2e-6, 'linear', 1e-6, 1e-6) == 1.0

    def test_smooth_step_matches_smooth_step_helper(self):
        """`traction` smooth_step factor must agree with the canonical
        Hermite smooth_step helper exported by the BC module."""
        from phast.boundary_conditions import (
            _eval_traction_ramp, smooth_step,
        )
        t_ramp = 1e-6
        for t in (0.0, 0.1e-6, 0.3e-6, 0.5e-6, 0.7e-6, 0.9e-6, 1.0e-6,
                  1.2e-6):
            f_ramp = _eval_traction_ramp(t, 'smooth_step', t_ramp, t_ramp)
            f_ref = smooth_step(t, 0.0, t_ramp)
            assert f_ramp == pytest.approx(f_ref, abs=1e-12)

    def test_cosine_factor(self):
        from phast.boundary_conditions import _eval_traction_ramp
        T = 1e-6
        assert _eval_traction_ramp(0.0, 'cosine', T, T) == 0.0
        assert _eval_traction_ramp(0.5 * T, 'cosine', T, T) == pytest.approx(0.5)
        assert _eval_traction_ramp(T, 'cosine', T, T) == pytest.approx(1.0)
        assert _eval_traction_ramp(2 * T, 'cosine', T, T) == 1.0

    def test_unknown_ramp_raises(self):
        from phast.boundary_conditions import _eval_traction_ramp
        with pytest.raises(ValueError):
            _eval_traction_ramp(0.0, 'tanh', 1e-6, 1e-6)


# ---------------------------------------------------------------------------
# traction nodal-force regression vs. legacy neumann
# ---------------------------------------------------------------------------

def _make_unit_square_mesh():
    """Two-triangle unit square mesh, in-process (no gmsh)."""
    nodes = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=torch.float64)
    elements = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=torch.long)

    class _M:
        pass
    m = _M()
    m.nodes = nodes
    m.elements = elements
    m.n_nodes = 4
    return m


class TestTractionForcesRegression:
    """`traction ramp_type=constant` must reproduce legacy `neumann`."""

    def test_constant_traction_matches_legacy_neumann(self):
        from phast.boundary_conditions import BoundaryConditions

        mesh = _make_unit_square_mesh()
        top = torch.tensor([2, 3], dtype=torch.long)  # top edge nodes

        bcs_legacy = BoundaryConditions(mesh.n_nodes, device='cpu',
                                        dtype=torch.float64)
        bcs_legacy.add_neumann(top, [0.0, 1.0e6])

        bcs_new = BoundaryConditions(mesh.n_nodes, device='cpu',
                                     dtype=torch.float64)
        bcs_new.add_traction(top, [0.0, 1.0e6], ramp_type='constant')

        # Without time → both behave identically (factor = 1).
        f_legacy = bcs_legacy.get_neumann_forces(mesh)
        f_new = bcs_new.get_neumann_forces(mesh, t=0.0)
        assert torch.allclose(f_new, f_legacy)

        # And with a later time too — constant ramp ignores t.
        f_new_late = bcs_new.get_neumann_forces(mesh, t=1.0)
        assert torch.allclose(f_new_late, f_legacy)

    def test_linear_traction_scales_correctly(self):
        from phast.boundary_conditions import BoundaryConditions

        mesh = _make_unit_square_mesh()
        top = torch.tensor([2, 3], dtype=torch.long)

        bcs = BoundaryConditions(mesh.n_nodes, device='cpu',
                                 dtype=torch.float64)
        bcs.add_traction(top, [0.0, 1.0e6], ramp_type='linear', t_ramp=1e-6)

        f0 = bcs.get_neumann_forces(mesh, t=0.0)
        f_half = bcs.get_neumann_forces(mesh, t=0.5e-6)
        f_end = bcs.get_neumann_forces(mesh, t=1e-6)

        assert torch.allclose(f0, torch.zeros_like(f0))
        assert torch.allclose(f_half, 0.5 * f_end)

    def test_smooth_step_traction_scales_correctly(self):
        from phast.boundary_conditions import (
            BoundaryConditions, smooth_step,
        )

        mesh = _make_unit_square_mesh()
        top = torch.tensor([2, 3], dtype=torch.long)

        t_ramp = 1e-6
        bcs = BoundaryConditions(mesh.n_nodes, device='cpu',
                                 dtype=torch.float64)
        bcs.add_traction(top, [0.0, 1.0e6], ramp_type='smooth_step',
                         t_ramp=t_ramp)

        f_full = bcs.get_neumann_forces(mesh, t=t_ramp)
        for t in (0.25e-6, 0.5e-6, 0.75e-6):
            expected = smooth_step(t, 0.0, t_ramp) * f_full
            got = bcs.get_neumann_forces(mesh, t=t)
            assert torch.allclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# rigid_connector (simplified)
# ---------------------------------------------------------------------------

class TestRigidConnector:
    """Welded (rotation_free=False) rigid_connector: equal prescribed
    displacement on all slaves + master for the locked DOFs.
    Default is rotation_free=True (issue #154 — see TestRigidConnectorRotation)."""

    def test_equal_displacement_on_all_slaves(self):
        from phast.boundary_conditions import BoundaryConditions

        bcs = BoundaryConditions(n_nodes=10, device='cpu',
                                 dtype=torch.float64)
        slaves = torch.tensor([3, 4, 5, 6, 7], dtype=torch.long)
        bcs.add_rigid_connector(master_node=2,
                                slave_indices=slaves,
                                locked_components=[0, 1],
                                prescribe={1: 1.5},  # y = +1.5
                                rotation_free=False)  # welded fallback

        mask, vals = bcs.get_masks_and_values()
        # Master + every slave constrained on both DOFs
        all_idx = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.long)
        for i in all_idx:
            assert mask[i, 0].item() is True
            assert mask[i, 1].item() is True
            assert vals[i, 0].item() == pytest.approx(0.0)
            assert vals[i, 1].item() == pytest.approx(1.5)
        # Other nodes untouched
        for i in (0, 1, 8, 9):
            assert mask[i, 0].item() is False
            assert mask[i, 1].item() is False

    def test_master_included_when_absent_from_slave_set(self):
        from phast.boundary_conditions import RigidConnector

        slaves = torch.tensor([5, 6], dtype=torch.long)
        rc = RigidConnector(master_node=1, slave_indices=slaves,
                            locked_components=[0], prescribe={0: 2.0},
                            rotation_free=False)
        dirichlet = rc.expand_to_dirichlet()
        assert len(dirichlet) == 1  # one component locked → one DirichletBC
        idx = dirichlet[0].node_indices.tolist()
        assert set(idx) == {1, 5, 6}
        assert dirichlet[0].component == 0
        assert dirichlet[0].value == pytest.approx(2.0)


class TestRigidConnectorRotation:
    """Full rotation-free rigid connector (issue #154).

    Default rotation_free=True: master gets a free theta DOF; slaves
    only receive the linearised rigid-body constraint, NOT a Dirichlet
    lock on every component. The T-matrix MPC is exercised by the
    DirectSolver path; here we test the BC layer in isolation.
    """

    def test_default_is_rotation_free(self):
        from phast.boundary_conditions import BoundaryConditions

        bcs = BoundaryConditions(n_nodes=10, device='cpu',
                                 dtype=torch.float64)
        slaves = torch.tensor([3, 4, 5, 6, 7], dtype=torch.long)
        bcs.add_rigid_connector(master_node=2,
                                slave_indices=slaves,
                                locked_components=[0, 1],
                                prescribe={1: 1.5})  # default: rotation_free=True
        # Only the master's translation is in Dirichlet — slaves are
        # tied via the constraint matrix inside the solver.
        mask, vals = bcs.get_masks_and_values()
        assert mask[2, 0].item() is True
        assert mask[2, 1].item() is True
        assert vals[2, 1].item() == pytest.approx(1.5)
        for i in (3, 4, 5, 6, 7):
            assert mask[i, 0].item() is False, \
                f"slave {i} component 0 must NOT be Dirichlet under rotation_free"
            assert mask[i, 1].item() is False, \
                f"slave {i} component 1 must NOT be Dirichlet under rotation_free"
        assert len(bcs.get_active_rigid_connectors()) == 1

    def test_T_block_shape_and_contents(self):
        """build_T_block returns the linearised constraint coefficients."""
        from phast.boundary_conditions import RigidConnector

        # Fake mesh with three nodes: master at (1,2), slave at (3,5),
        # slave at (1,2) (== master position; should be filtered).
        class _M:
            pass
        m = _M()
        m.nodes = torch.tensor([[1.0, 2.0], [3.0, 5.0], [1.0, 2.0]],
                               dtype=torch.float64)
        slaves = torch.tensor([1], dtype=torch.long)  # only one real slave
        rc = RigidConnector(master_node=0, slave_indices=slaves,
                            locked_components=[0, 1], prescribe={1: 0.0},
                            rotation_free=True)
        rows, cols, vals, slave_dofs, m_node, m_dofs, theta_col = \
            rc.build_T_block(m, theta_col=42)
        # Slave 1 contributes two rows: 2 (u_x) and 3 (u_y)
        assert slave_dofs == [2, 3]
        # Row 2: T[2, 0] = 1 (u_master_x), T[2, 42] = -(5-2) = -3
        # Row 3: T[3, 1] = 1 (u_master_y), T[3, 42] = +(3-1) = +2
        triples = list(zip(rows, cols, vals))
        assert (2, 0, 1.0) in triples
        assert (2, 42, -3.0) in triples
        assert (3, 1, 1.0) in triples
        assert (3, 42, 2.0) in triples
        assert m_dofs == (0, 1)

    def test_cantilever_compliance(self):
        """Cantilever beam, plane stress, tip rigid connector.

        Closed-form: u_tip = P L^3 / (3 E I) for a clamped beam loaded
        by a transverse tip force P. With rotation_free=True the tip
        rigid connector should recover this; with rotation_free=False
        the welded BC over-constrains the tip rotation and gives a
        much stiffer answer (smaller |u_tip|).
        """
        import numpy as np
        import torch
        from phast.mesh import FEMMesh
        from phast.material import Material
        from phast.fem_operators import FEMOperators
        from phast.boundary_conditions import BoundaryConditions
        from phast.mechanics_solver import DirectSolver

        # Beam: L=20, h=1, plane stress, E=1, nu=0.0 (Euler–Bernoulli).
        # Build a structured triangle mesh.
        L = 20.0
        h = 1.0
        nx, ny = 80, 6
        xs = np.linspace(0.0, L, nx + 1)
        ys = np.linspace(-h / 2, h / 2, ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        nodes = np.stack([X.ravel(), Y.ravel()], axis=1)
        elems = []
        for j in range(ny):
            for i in range(nx):
                a = j * (nx + 1) + i
                b = a + 1
                c = a + (nx + 1)
                d = c + 1
                elems.append([a, b, d])
                elems.append([a, d, c])
        nodes_t = torch.tensor(nodes, dtype=torch.float64)
        elems_t = torch.tensor(elems, dtype=torch.long)

        # Boundary node sets: clamp = leftmost column; tip_face = rightmost
        left_idx = np.where(np.isclose(nodes[:, 0], 0.0))[0]
        right_idx = np.where(np.isclose(nodes[:, 0], L))[0]
        # Master node at the centroid of the right face: pick the row j=ny//2
        master_node = int(ny // 2 * (nx + 1) + nx)
        node_sets = {
            'left': torch.tensor(left_idx, dtype=torch.long),
            'tip_face': torch.tensor(right_idx, dtype=torch.long),
            'master': torch.tensor([master_node], dtype=torch.long),
        }
        mesh = FEMMesh.from_tensors(nodes_t, elems_t, node_sets,
                                    device='cpu', dtype=torch.float64)

        # Plane stress with E=1, nu=0 → lambda=0, mu = 0.5
        E = 1.0
        nu = 0.0
        mat = Material(E=E, nu=nu, Gc=1.0, l0=1.0,
                       energy_split='isotropic', plane_stress=True)
        fem = FEMOperators(mesh, mat, ctx=None)
        solver = DirectSolver(fem, tol=1e-12, max_newton=4, rtol=1e-12)

        # Apply a pure tip force P on the master via f_ext (no traction
        # ramp — quasi-static one-shot). The connector projects this
        # force back onto the slaves through the constraint.
        P = 1e-3
        I_section = h ** 3 / 12.0  # second moment of area, unit thickness
        u_tip_analytical = P * L ** 3 / (3.0 * E * I_section)

        n = mesh.n_nodes
        d = torch.zeros(n, dtype=torch.float64)

        def _solve(rotation_free):
            bcs = BoundaryConditions(n, device='cpu', dtype=torch.float64)
            bcs.fix(node_sets['left'], component=0)
            bcs.fix(node_sets['left'], component=1)
            # Master must be Dirichlet on x and y for our T-matrix path
            # (master translation is the prescribed-input slot). We
            # prescribe u_master_x = 0 (axial translation pinned) and
            # leave master y free? — no: bc_mask requires y to be
            # constrained too. Instead we apply force via f_ext and
            # *prescribe* the master in the welded comparison only.
            # For rotation_free, we do this differently: Dirichlet on
            # x (no axial), and use a small "soft pull" by prescribing
            # u_y on the master, then read the reaction. To keep the
            # comparison clean we instead enforce both displacements
            # via prescribed values and check the deflection of an
            # interior tip node.
            # Simpler: prescribe u at the master directly and verify
            # by sweeping P -> u relationship (linear elasticity). Use
            # this approach.
            return bcs

        # Strategy: apply a prescribed master u_y and check internal
        # force at the master (master "reaction"). Compare welded vs
        # rotation-free for the same prescribed displacement.
        u_prescribed = 0.05  # small displacement (geometrically linear)

        def _solve_prescribed(rotation_free):
            bcs = BoundaryConditions(n, device='cpu', dtype=torch.float64)
            bcs.fix(node_sets['left'], component=0)
            bcs.fix(node_sets['left'], component=1)
            bcs.add_rigid_connector(
                master_node=master_node,
                slave_indices=node_sets['tip_face'],
                locked_components=[0, 1],
                prescribe={0: 0.0, 1: u_prescribed},
                rotation_free=rotation_free,
            )
            mask, vals = bcs.get_masks_and_values()
            u0 = torch.zeros(n, 2, dtype=torch.float64)
            rcs = bcs.get_active_rigid_connectors() if rotation_free else None
            u_sol = solver.solve(u0, d, mask, vals,
                                 rigid_connectors=rcs)
            f_int = fem.internal_force(u_sol, d)
            # Master reaction = sum of f_int over rigid set, component 1.
            slave_idx = node_sets['tip_face'].numpy()
            ids = np.unique(np.concatenate(
                [np.array([master_node]), slave_idx]))
            R_y = float(f_int[ids, 1].sum().item())
            return R_y, u_sol

        R_free, u_free = _solve_prescribed(rotation_free=True)
        R_weld, u_weld = _solve_prescribed(rotation_free=False)
        # Effective tip stiffness K = R / u
        K_free = R_free / u_prescribed
        K_weld = R_weld / u_prescribed
        # Beam closed-form stiffness: K_analytical = 3 E I / L^3
        K_analytical = 3.0 * E * I_section / L ** 3

        # The rotation-free connector should approach the analytical
        # cantilever stiffness within ~30% (FE shear-locking + slender
        # ratio L/h = 20 means the answer is bracketed but not exact).
        # The welded connector should be substantially stiffer.
        rel_free = abs(K_free - K_analytical) / K_analytical
        assert K_weld > 1.5 * K_free, (
            f"welded should be stiffer than rotation-free; got "
            f"K_weld={K_weld:.4e}, K_free={K_free:.4e}")
        assert rel_free < 0.5, (
            f"rotation-free K should approach analytical PL^3/3EI; "
            f"got K_free={K_free:.4e} vs K_analytical={K_analytical:.4e} "
            f"(rel error {rel_free:.2%})")
        # Stash for visibility in test logs (pytest -s)
        print(f"\n[cantilever] K_analytical = {K_analytical:.4e}")
        print(f"[cantilever] K_rotation_free = {K_free:.4e} "
              f"({rel_free:.2%} from analytical)")
        print(f"[cantilever] K_welded = {K_weld:.4e} "
              f"(ratio welded/free = {K_weld/K_free:.2f})")

    def test_master_must_have_dirichlet_translation(self):
        """rotation-free MPC requires Dirichlet on master translation."""
        import numpy as np
        import torch
        from phast.mesh import FEMMesh
        from phast.material import Material
        from phast.fem_operators import FEMOperators
        from phast.boundary_conditions import BoundaryConditions
        from phast.mechanics_solver import DirectSolver

        # Tiny 2x1 mesh
        nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                              [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                             dtype=torch.float64)
        elems = torch.tensor([[0, 1, 4], [0, 4, 3],
                              [1, 2, 5], [1, 5, 4]], dtype=torch.long)
        node_sets = {
            'left': torch.tensor([0, 3], dtype=torch.long),
            'right': torch.tensor([2, 5], dtype=torch.long),
        }
        mesh = FEMMesh.from_tensors(nodes, elems, node_sets,
                                    device='cpu', dtype=torch.float64)
        mat = Material(E=1.0, nu=0.25, Gc=1.0, l0=1.0,
                       energy_split='isotropic')
        fem = FEMOperators(mesh, mat, ctx=None)
        solver = DirectSolver(fem, tol=1e-10, max_newton=2)
        n = mesh.n_nodes
        d = torch.zeros(n, dtype=torch.float64)

        bcs = BoundaryConditions(n, device='cpu', dtype=torch.float64)
        bcs.fix(node_sets['left'], component=0)
        bcs.fix(node_sets['left'], component=1)
        # Master = node 2; deliberately do NOT include it in Dirichlet
        # (no prescribe and no locked_components → empty Dirichlet for
        # master), so the solver should error.
        bcs.add_rigid_connector(
            master_node=2,
            slave_indices=node_sets['right'],
            locked_components=[],  # no Dirichlet on master
            prescribe={},
            rotation_free=True,
        )
        mask, vals = bcs.get_masks_and_values()
        u0 = torch.zeros(n, 2, dtype=torch.float64)
        rcs = bcs.get_active_rigid_connectors()
        with pytest.raises(ValueError, match="master node"):
            solver.solve(u0, d, mask, vals, rigid_connectors=rcs)


# ---------------------------------------------------------------------------
# YAML dispatch end-to-end (resolve_config wiring)
# ---------------------------------------------------------------------------

YAML_TPL = """
name: BC vocab smoke
geometry:
  type: rectangular_sent
  parameters: {{W: 50.0, H: 20.0, a: 25.0, h_crack: 2.0, h_coarse: 5.0}}
material:
  preset: glass_borden
loading:
  protocol: simple
  num_steps: 1
  t_total: 1.0e-6
solver:
  solver_type: explicit
device:
  device: cpu
boundary_conditions:
{bcs}
"""


def _resolve_yaml(yaml_str):
    from phast.config import load_config, resolve_config
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False, mode='w') as f:
        f.write(yaml_str)
        path = f.name
    try:
        cfg = load_config(path)
        return cfg, resolve_config(cfg)
    finally:
        os.unlink(path)


class TestYAMLDispatch:
    """End-to-end: YAML BC vocab → BoundaryConditions object."""

    def test_yaml_symmetry(self):
        bcs_yaml = "  - {nodes: bottom, type: symmetry, axis: y}\n"
        cfg, objs = _resolve_yaml(YAML_TPL.format(bcs=bcs_yaml))
        bcs = objs['bcs']
        # One DirichletBC produced, on component 1
        assert len(bcs.bcs) == 1
        assert bcs.bcs[0].component == 1
        assert bcs.bcs[0].value == 0.0

    def test_yaml_traction_smooth_step(self):
        bcs_yaml = (
            "  - {nodes: top, type: traction, component: 1, value: 1.0e6, "
            "ramp_type: smooth_step, t_ramp: 1.0e-7}\n"
        )
        cfg, objs = _resolve_yaml(YAML_TPL.format(bcs=bcs_yaml))
        bcs = objs['bcs']
        assert len(bcs.neumann_bcs) == 1
        nbc = bcs.neumann_bcs[0]
        assert nbc.ramp_type == 'smooth_step'
        assert nbc.t_ramp == pytest.approx(1.0e-7)

    def test_yaml_neumann_back_compat(self):
        bcs_yaml = "  - {nodes: top, type: neumann, component: 1, value: 1.0}\n"
        cfg, objs = _resolve_yaml(YAML_TPL.format(bcs=bcs_yaml))
        bcs = objs['bcs']
        assert len(bcs.neumann_bcs) == 1
        # Legacy default → constant ramp
        assert bcs.neumann_bcs[0].ramp_type == 'constant'
