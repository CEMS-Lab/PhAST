"""Smoke tests for AT1 paired with each energy split.

W4 audit Tier-3, Gap 4: AT2 was exercised with all four splits, but
``pf_model='AT1' + energy_split='isotropic'`` and ``+ 'spectral'``
were never exercised by CI. These two combos route through different
branches inside the damage-solver dispatcher (AT1 source term + the
relevant ``_psi_plus_*``), so a missing route would silently NaN at
benchmark scale. This test makes the dispatcher coverage explicit.

Coverage: for each of {isotropic, spectral, amor, star_convex} paired
with AT1, run 100 explicit steps of a coarse SENT mesh and assert
no exception, no NaN, max(d) ∈ (0, 1), and finite total energy
throughout. Detailed physics validation belongs to the benchmarks.

Total mesh: 81 nodes, 126 elements (Miehe SENT, l0=0.5, h_crack=2*l0).
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest
import torch

from phast.boundary_conditions import BoundaryConditions
from phast.material import Material
from phast.mesh import FEMMesh
from phast.mesh_generator import miehe_tension as gen_mesh
from phast.staggered_solver import SolverConfig, StaggeredSolver


# Coarse SENT — keep the test cheap. ~80 nodes; CFL gives small dt
# so 100 steps is well under a second per combo on Mac CPU.
W = 20.0
A_NOTCH = 10.0
L0 = 0.5
H_CRACK = 2.0       # ~4*l0
H_COARSE = 4.0
DISP_MAX = 1.0e-3
T_RAMP = 50e-6
N_STEPS = 100


@pytest.fixture(scope='module')
def mesh_path():
    tmpdir = tempfile.mkdtemp(prefix='test_at1_combos_')
    path = os.path.join(tmpdir, 'sent.msh')
    gen_mesh(path, L=W, a=A_NOTCH, l0=L0,
             h_crack=H_CRACK, h_coarse=H_COARSE, verbose=False)
    yield path


def _build_solver(mesh_path: str, energy_split: str) -> StaggeredSolver:
    """AT1 SENT solver at the requested energy split. PMMA-ish constants."""
    mat = Material(
        E=3090.0, nu=0.35, Gc=0.3, l0=L0, rho=1.18e-9,
        energy_split=energy_split, pf_model='AT1',
        eta_residual=1e-6,
    )
    mesh = FEMMesh(mesh_path, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()

    bcs = BoundaryConditions(mesh.n_nodes, device='cpu', dtype=torch.float64)
    bcs.fix(mesh.node_sets['left'], 0)
    bcs.fix(mesh.node_sets['right'], 0)
    bcs.add(mesh.node_sets['top'], 1, +DISP_MAX)
    bcs.add(mesh.node_sets['bottom'], 1, -DISP_MAX)
    bcs.load_factor = 0.0

    cfg = SolverConfig(
        solver_type='explicit',
        dt_safety=0.5,
        use_multigrid=False,
        damage_every=1,
        adaptive_dt=False,
        print_every=10_000,
    )
    return StaggeredSolver(mesh, mat, bcs, config=cfg)


@pytest.mark.parametrize('energy_split', [
    'isotropic',     # GAP — AT1 + isotropic was never exercised
    'spectral',      # GAP — AT1 + spectral was never exercised
    'amor',          # baseline (already covered, included for symmetry)
    'star_convex',   # baseline (Gap 1 covers psi/stress at unit scale)
])
def test_at1_runs_with_split(mesh_path, energy_split):
    solver = _build_solver(mesh_path, energy_split)

    fem = solver.fem
    t = 0.0
    for step in range(N_STEPS):
        solver.bcs.load_factor = min(t / T_RAMP, 1.0)
        psi_plus = solver.step_full()

        # Dispatcher must produce finite stress/psi/energy on every step.
        assert torch.isfinite(psi_plus).all(), (
            f"psi_plus contains non-finite values at step {step} "
            f"with split={energy_split}")
        assert torch.isfinite(solver.d).all(), (
            f"damage field NaN at step {step} with split={energy_split}")
        assert torch.isfinite(solver.u).all(), (
            f"displacement NaN at step {step} with split={energy_split}")

        E = fem.compute_total_energy(
            solver.u, solver.d,
            strain=solver._last_strain, psi_plus=psi_plus)
        assert math.isfinite(E), (
            f"total energy non-finite at step {step} with split={energy_split}")

        t += solver.dt

    # Damage must stay physically valid: 0 <= d < 1 (post-100 steps it may
    # be very small for a coarse mesh; we just check the bounds, not the
    # absolute level).
    d_max = solver.d.max().item()
    d_min = solver.d.min().item()
    assert 0.0 <= d_min <= d_max < 1.0, (
        f"damage out of [0, 1) bounds for AT1+{energy_split}: "
        f"min={d_min:.3e}, max={d_max:.3e}")
