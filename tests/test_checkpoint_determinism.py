#!/usr/bin/env python
"""Regression test for issue #439 — gradient-checkpoint non-determinism
in the gc_horizontal inverse demo.

Background
----------
Pair B (`gc_horizontal` inverse demo, paper-3 P3-C2) launches L-BFGS at
N=20000 explicit steps with `--checkpoint_chunks 100`, which wraps the
explicit time loop in `torch.utils.checkpoint.checkpoint(use_reentrant=
False)` so each chunk's autograd tape is recomputed at backward time
rather than held in memory. PyTorch strict-checks that the recompute
produces the same number of saved tensors as the original forward;
job 29885 hit the strict mismatch:

    torch.utils.checkpoint.CheckpointError: A different number of
    tensors was saved during the original forward and recomputation.
    Number of tensors saved during forward: 13600
    Number of tensors saved during recomputation: 13400.

Root cause
----------
``demo_inverse_gc_horizontal.run_forward`` installs the
``solver.diff_Gc_field`` attribute before the checkpointed time loop
and **deletes it immediately after the loop returns**, before the
caller invokes ``loss.backward()``. The damage solver routes through
``_AdjointDamageSolveField.apply`` (the autograd Function path that
saves 4 tensors per call) when ``Gc_field`` is set, and through the
plain no-grad ``_solve_dispatch`` (no autograd-saved tensors) when it
is not. Under ``use_reentrant=False`` the per-chunk forward is
re-executed at backward time, but by then the attribute has already
been removed -- so the re-run damage solves do NOT route through the
adjoint Function and the recompute saves a different number of
tensors than the original forward.

Fix
---
Leave ``solver.diff_Gc_field`` installed across the lifetime of the
forward-plus-backward call. It is a plain attribute and the next
forward call overwrites it cleanly. Local 1-line change in the demo
``run_forward``; no solver-level API change.

Reproducer scope
----------------
Reproduces at small scale (5 chunks, 30 steps, tiny mesh) on the
unpatched code. The negative regression test in this file
(`test_premature_del_diff_gc_field_raises_checkpoint_error`)
re-installs the buggy ``del`` pattern explicitly and asserts
``CheckpointError`` -- evidence that the strict-check still catches
this class of bug, so future reintroductions will fail loudly.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, '..', '..')))

from phast.mesh_generator import rectangular_sent
from phast.mesh import FEMMesh
from phast.material import create_material
from phast.boundary_conditions import BoundaryConditions
from phast.staggered_solver import StaggeredSolver, SolverConfig
from phast.config import LoadingConfig, compute_load_factor
from phast.device import DeviceContext


def _build_tiny_solver():
    """Tiny mesh that runs in a few seconds on CPU."""
    mp = os.path.join(tempfile.gettempdir(), "issue439_chkpt_det.msh")
    rectangular_sent(mp, W=4.0, H=2.0, a=0.5, l0=0.2,
                     h_crack=0.4, h_coarse=0.6)
    mesh = FEMMesh(mp, device='cpu', dtype=torch.float64)
    mesh.identify_boundaries()
    mat = create_material(E=3000.0, nu=0.3, rho=1.2e-9, Gc=0.3, l0=0.2,
                          energy_split='amor', pf_model='AT1',
                          plane_stress=True, gamma_correction=True)
    bcs = BoundaryConditions(mesh.n_nodes, device='cpu',
                             dtype=torch.float64)
    bcs.add(mesh.node_sets['top'],    component=1, value=+0.05)
    bcs.add(mesh.node_sets['bottom'], component=1, value=-0.05)
    bcs.fix(mesh.node_sets['left'], component=0)
    cfg = SolverConfig(solver_type='explicit', differentiable=True,
                       damage_every=1, preconditioner='jacobi',
                       damping_ratio_max=0.05, enable_damage=True)
    solver = StaggeredSolver(mesh, mat, bcs, config=cfg,
                             ctx=DeviceContext('cpu', torch.float64))
    return solver


def _make_chunk_closure(solver, loading):
    """Mirror the (post-fix) _chunk function from
    demo_inverse_gc_horizontal so this test does not depend on the
    demo module being importable in CI environments without the
    plotting deps."""
    solver_dt = solver.dt

    def _chunk(u_in, v_in, a_in, d_in, He_in, Hn_in, base_step, k_steps):
        solver.u = u_in
        solver.v = v_in
        solver.a = a_in
        solver.d = d_in
        solver.H_elem = He_in
        solver.H_nodal = Hn_in
        solver._step_count = base_step
        for j in range(k_steps):
            step = base_step + j
            solver.bcs.load_factor = compute_load_factor(
                step, solver_dt, loading)
            solver.step_full()
        return (solver.u, solver.v, solver.a, solver.d,
                solver.H_elem, solver.H_nodal)

    return _chunk


def _seed_state(solver):
    """Reset solver state to a fresh starting point (no damage, zero
    displacement) so test runs are independent."""
    n_nodes = solver.mesh.n_nodes
    n_elem = solver.mesh.n_elems
    dt = solver.dtype
    dev = solver.device
    solver.u = torch.zeros(n_nodes, 2, dtype=dt, device=dev)
    solver.v = torch.zeros(n_nodes, 2, dtype=dt, device=dev)
    solver.a = torch.zeros(n_nodes, 2, dtype=dt, device=dev)
    solver.d = torch.zeros(n_nodes, dtype=dt, device=dev)
    solver.H_elem = torch.zeros(n_elem, dtype=dt, device=dev)
    solver.H_nodal = torch.zeros(n_nodes, dtype=dt, device=dev)
    solver._step_count = 0
    solver._explicit_step_count = 0


def _run_checkpointed(solver, loading, Gc_field, n_steps, n_chunks):
    """Mirror the production checkpointing loop in
    demo_inverse_gc_horizontal.run_forward, post-fix (no premature
    ``del`` of solver.diff_Gc_field)."""
    _seed_state(solver)
    solver.diff_Gc_field = Gc_field
    _chunk = _make_chunk_closure(solver, loading)
    chunk_size = (n_steps + n_chunks - 1) // n_chunks
    u, v, a, d = solver.u, solver.v, solver.a, solver.d
    He, Hn = solver.H_elem, solver.H_nodal
    base = 0
    while base < n_steps:
        k = min(chunk_size, n_steps - base)
        u, v, a, d, He, Hn = torch.utils.checkpoint.checkpoint(
            _chunk, u, v, a, d, He, Hn, base, k,
            use_reentrant=False,
        )
        base += k
    return d


def test_checkpoint_forward_backward_succeeds():
    """The post-fix forward+backward through ``torch.utils.checkpoint``
    completes without raising ``CheckpointError``. Pre-fix this would
    have raised at ``loss.backward()`` because the recompute path
    routed the damage solve through the no-grad branch (``del
    solver.diff_Gc_field`` happened between forward and backward)."""
    n_steps = 30
    n_chunks = 5
    solver = _build_tiny_solver()
    loading = LoadingConfig(ramp_type='smooth', t_ramp=solver.dt * n_steps)
    Gc_field = torch.full((solver.mesh.n_elems,), 0.3,
                          dtype=torch.float64, requires_grad=True)
    d_final = _run_checkpointed(
        solver, loading, Gc_field, n_steps=n_steps, n_chunks=n_chunks)
    loss = (d_final ** 2).sum()
    loss.backward()
    assert Gc_field.grad is not None
    assert torch.isfinite(Gc_field.grad).all()


def test_checkpoint_repeat_runs_match():
    """Two independent forward+backward passes on freshly-built solvers
    produce bit-identical gradients on the leaf parameter. Hedge
    against any future reintroduction of cross-call non-determinism in
    the checkpointed path."""
    n_steps = 30
    n_chunks = 5
    grads = []
    for _run in range(2):
        solver = _build_tiny_solver()
        loading = LoadingConfig(ramp_type='smooth',
                                t_ramp=solver.dt * n_steps)
        Gc_field = torch.full((solver.mesh.n_elems,), 0.3,
                              dtype=torch.float64, requires_grad=True)
        d_final = _run_checkpointed(
            solver, loading, Gc_field, n_steps=n_steps, n_chunks=n_chunks)
        loss = (d_final ** 2).sum()
        loss.backward()
        assert Gc_field.grad is not None
        grads.append(Gc_field.grad.detach().clone())
    assert torch.equal(grads[0], grads[1]), (
        "Repeat runs of the checkpointed forward+backward must yield "
        "identical gradients (issue #439 regression).")


def test_premature_del_diff_gc_field_raises_checkpoint_error():
    """Negative regression: deleting ``solver.diff_Gc_field`` BEFORE
    ``loss.backward()`` (the original buggy pattern in
    ``demo_inverse_gc_horizontal.run_forward``) MUST raise
    ``CheckpointError`` — it routes the recompute through the
    no-grad damage path and saves a different number of tensors than
    the original forward.

    If this test starts FAILING (i.e. the buggy pattern silently
    works), then either PyTorch has relaxed the strict-check (and
    #439 is no longer a risk), or the damage solver no longer
    differentiates between the autograd and no-grad paths in its
    tensor-saving footprint. Either is significant enough to warrant
    updating this test deliberately rather than removing it."""
    n_steps = 30
    n_chunks = 5
    solver = _build_tiny_solver()
    loading = LoadingConfig(ramp_type='smooth', t_ramp=solver.dt * n_steps)
    Gc_field = torch.full((solver.mesh.n_elems,), 0.3,
                          dtype=torch.float64, requires_grad=True)
    _seed_state(solver)
    solver.diff_Gc_field = Gc_field

    _chunk = _make_chunk_closure(solver, loading)
    chunk_size = (n_steps + n_chunks - 1) // n_chunks
    u, v, a, d = solver.u, solver.v, solver.a, solver.d
    He, Hn = solver.H_elem, solver.H_nodal
    base = 0
    while base < n_steps:
        k = min(chunk_size, n_steps - base)
        u, v, a, d, He, Hn = torch.utils.checkpoint.checkpoint(
            _chunk, u, v, a, d, He, Hn, base, k,
            use_reentrant=False,
        )
        base += k
    # Buggy pattern: drop the autograd-routing attribute before
    # backward. Recompute will see a different saved-tensor count and
    # the strict-check should fail.
    del solver.diff_Gc_field
    loss = (d ** 2).sum()
    with pytest.raises(torch.utils.checkpoint.CheckpointError):
        loss.backward()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
