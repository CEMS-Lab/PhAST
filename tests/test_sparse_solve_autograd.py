"""Tests for the SciPy SuperLU autograd wrapper (#106)."""

import sys
import types

import numpy as np
import pytest
import torch

scipy = pytest.importorskip("scipy")

from phast.sparse_solve import (  # noqa: E402
    available_sparse_backends,
    SparseSolveAutograd,
    _MumpsSparseSolveAutograd,
    _cudss_functional,
    _petsc_functional,
    resolve_sparse_backend,
    make_factor_handle,
    solve,
)


def _laplacian_1d(n, dtype=torch.float64):
    """SPD 1D Laplacian as a (indices, values, n) COO tuple."""
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(2.0)
        if i > 0:
            rows.append(i); cols.append(i - 1); vals.append(-1.0)
        if i < n - 1:
            rows.append(i); cols.append(i + 1); vals.append(-1.0)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=dtype)
    return indices, values, n


def _spd_dense_to_coo(K_dense):
    n = K_dense.shape[0]
    idx_r, idx_c = torch.meshgrid(
        torch.arange(n), torch.arange(n), indexing='ij')
    indices = torch.stack(
        [idx_r.reshape(-1), idx_c.reshape(-1)], dim=0).to(torch.long)
    values = K_dense.reshape(-1).to(torch.float64).clone()
    return indices, values, n


def test_roundtrip():
    indices, values, n = _laplacian_1d(10)
    torch.manual_seed(0)
    b = torch.randn(n, dtype=torch.float64)
    x = SparseSolveAutograd.apply(indices, values, b, n)

    K_dense = torch.zeros(n, n, dtype=torch.float64)
    K_dense[indices[0], indices[1]] = values
    Kx = K_dense @ x
    assert torch.allclose(Kx, b, atol=1e-10), (Kx - b).norm().item()


def test_gradcheck_K():
    torch.manual_seed(1)
    n = 4
    A = torch.randn(n, n, dtype=torch.float64)
    K_dense = A @ A.T + n * torch.eye(n, dtype=torch.float64)
    indices, values, _ = _spd_dense_to_coo(K_dense)
    b = torch.randn(n, dtype=torch.float64)

    values = values.detach().clone().requires_grad_(True)

    def f(K_values):
        return SparseSolveAutograd.apply(indices, K_values, b, n)

    assert torch.autograd.gradcheck(
        f, (values,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_b():
    torch.manual_seed(2)
    n = 4
    A = torch.randn(n, n, dtype=torch.float64)
    K_dense = A @ A.T + n * torch.eye(n, dtype=torch.float64)
    indices, values, _ = _spd_dense_to_coo(K_dense)
    b = torch.randn(n, dtype=torch.float64).requires_grad_(True)

    def f(b_):
        return SparseSolveAutograd.apply(indices, values, b_, n)

    assert torch.autograd.gradcheck(
        f, (b,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_vs_dense_solve():
    """100x100 SPD: SuperLU result must match torch.linalg.solve to round-off."""
    torch.manual_seed(3)
    n = 100
    A = torch.randn(n, n, dtype=torch.float64)
    K_dense = A @ A.T + n * torch.eye(n, dtype=torch.float64)
    indices, values, _ = _spd_dense_to_coo(K_dense)
    b = torch.randn(n, dtype=torch.float64)

    x_sparse = SparseSolveAutograd.apply(indices, values, b, n)
    x_dense = torch.linalg.solve(K_dense, b)

    assert torch.allclose(x_sparse, x_dense, atol=1e-10), \
        (x_sparse - x_dense).norm().item()


def test_solve_wrapper_auto_dispatches_to_scipy():
    indices, values, n = _laplacian_1d(8)
    K_torch = torch.sparse_coo_tensor(
        indices, values, (n, n), dtype=torch.float64).coalesce()
    b = torch.randn(n, dtype=torch.float64)
    x = solve(K_torch, b, backend='auto')
    K_dense = K_torch.to_dense()
    assert torch.allclose(K_dense @ x, b, atol=1e-10)


def test_available_sparse_backends_reports_consistent_status():
    status = available_sparse_backends()
    assert isinstance(status.scipy, bool)
    assert isinstance(status.petsc, bool)
    assert isinstance(status.cudss, bool)
    resolved = resolve_sparse_backend("auto", device_type="cpu", status=status)
    assert resolved in {"scipy", "mumps"}
    if status.cudss:
        assert resolve_sparse_backend("auto", device_type="cuda", status=status) in {
            "cudss", "mumps", "scipy"
        }


def test_solve_wrapper_preserves_csr_value_gradients():
    crow = torch.tensor([0, 2, 4], dtype=torch.int64)
    col = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    vals = torch.tensor([2.0, -1.0, -1.0, 2.0],
                        dtype=torch.float64, requires_grad=True)
    K_torch = torch.sparse_csr_tensor(crow, col, vals, size=(2, 2))
    b = torch.tensor([1.0, 0.0], dtype=torch.float64)

    x = solve(K_torch, b, backend='auto')
    assert x.requires_grad
    x.sum().backward()
    assert vals.grad is not None
    assert torch.isfinite(vals.grad).all()


def _install_fake_nvmath(monkeypatch, call_log):
    sparse_mod = types.ModuleType("nvmath.sparse")
    advanced_mod = types.ModuleType("nvmath.sparse.advanced")
    nvmath_mod = types.ModuleType("nvmath")
    nvmath_mod.sparse = sparse_mod
    sparse_mod.advanced = advanced_mod

    class FakeDirectSolver:
        def __init__(self, a, b):
            call_log.append(("init", a.shape, np.asarray(b).shape))
            self.a = a
            self.b = np.asarray(b, dtype=np.float64)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def plan(self):
            call_log.append(("plan",))

        def factorize(self):
            call_log.append(("factorize",))

        def solve(self):
            call_log.append(("solve",))
            return np.linalg.solve(self.a.toarray(), self.b)

    advanced_mod.DirectSolver = FakeDirectSolver
    monkeypatch.setitem(sys.modules, "nvmath", nvmath_mod)
    monkeypatch.setitem(sys.modules, "nvmath.sparse", sparse_mod)
    monkeypatch.setitem(sys.modules, "nvmath.sparse.advanced", advanced_mod)


def test_solve_wrapper_forced_mumps_falls_back_when_petsc_not_functional():
    """A forced MUMPS request should fall back cleanly if PETSc is broken."""
    from phast import sparse_solve as _ss

    indices, values, n = _laplacian_1d(4)
    K_torch = torch.sparse_coo_tensor(
        indices, values, (n, n), dtype=torch.float64).coalesce()
    b = torch.zeros(n, dtype=torch.float64)
    saved_petsc, saved_cudss = _ss._PETSC_FUNCTIONAL, _ss._CUDSS_FUNCTIONAL
    try:
        _ss._PETSC_FUNCTIONAL = True
        _ss._CUDSS_FUNCTIONAL = True
        with pytest.warns(RuntimeWarning, match="Falling back to SciPy"):
            x = solve(K_torch, b, backend='mumps')
        assert torch.isfinite(x).all()
    finally:
        _ss._PETSC_FUNCTIONAL = saved_petsc
        _ss._CUDSS_FUNCTIONAL = saved_cudss


def test_cudss_functional_uses_current_nvmath_api(monkeypatch):
    """Regression for #496: nvmath DirectSolver requires (A, b), not A only."""
    from phast import sparse_solve as _ss

    _ss._reset_backend_cache()
    calls = []
    _install_fake_nvmath(monkeypatch, calls)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    try:
        assert _cudss_functional()
    finally:
        _ss._reset_backend_cache()
    assert calls[:4] == [
        ("init", (2, 2), (2,)),
        ("plan",),
        ("factorize",),
        ("solve",),
    ]


def test_cudss_backend_matches_scipy_with_fake_nvmath(monkeypatch):
    """The cuDSS autograd wrapper follows the same implicit derivative."""
    from phast import sparse_solve as _ss

    calls = []
    _install_fake_nvmath(monkeypatch, calls)
    saved_cudss = _ss._CUDSS_FUNCTIONAL
    try:
        _ss._CUDSS_FUNCTIONAL = True
        indices, values, n = _laplacian_1d(6)
        K_torch = torch.sparse_coo_tensor(
            indices, values, (n, n), dtype=torch.float64).coalesce()
        values = K_torch.values().detach().clone().requires_grad_(True)
        K_torch = torch.sparse_coo_tensor(
            K_torch.indices(), values, (n, n), dtype=torch.float64).coalesce()
        b = torch.randn(n, dtype=torch.float64, requires_grad=True)

        x_cudss = solve(K_torch, b, backend='cudss')
        x_scipy = solve(K_torch.detach(), b.detach(), backend='scipy')
        assert torch.allclose(x_cudss, x_scipy, atol=1e-12)
        loss = (x_cudss ** 2).sum()
        loss.backward()
        assert values.grad is not None
        assert b.grad is not None
    finally:
        _ss._CUDSS_FUNCTIONAL = saved_cudss


@pytest.mark.hpc
@pytest.mark.solver
@pytest.mark.skipif(not _petsc_functional(), reason="PETSc/MUMPS not functional")
def test_petsc_mumps_roundtrip():
    indices, values, n = _laplacian_1d(10)
    torch.manual_seed(0)
    b = torch.randn(n, dtype=torch.float64)
    x = _MumpsSparseSolveAutograd.apply(indices, values, b, n)
    K_dense = torch.zeros(n, n, dtype=torch.float64)
    K_dense[indices[0], indices[1]] = values
    assert torch.allclose(K_dense @ x, b, atol=1e-10)


@pytest.mark.hpc
@pytest.mark.solver
@pytest.mark.skipif(not _petsc_functional(), reason="PETSc/MUMPS not functional")
def test_petsc_mumps_gradcheck():
    torch.manual_seed(1)
    n = 4
    A = torch.randn(n, n, dtype=torch.float64)
    K_dense = A @ A.T + n * torch.eye(n, dtype=torch.float64)
    indices, values, _ = _spd_dense_to_coo(K_dense)
    b = torch.randn(n, dtype=torch.float64).requires_grad_(True)
    values = values.detach().clone().requires_grad_(True)

    def f(K_values, b_):
        return _MumpsSparseSolveAutograd.apply(indices, K_values, b_, n)

    assert torch.autograd.gradcheck(
        f, (values, b), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.hpc
@pytest.mark.solver
@pytest.mark.skipif(not _petsc_functional(), reason="PETSc/MUMPS not functional")
def test_petsc_mumps_matches_scipy():
    torch.manual_seed(3)
    n = 50
    A = torch.randn(n, n, dtype=torch.float64)
    K_dense = A @ A.T + n * torch.eye(n, dtype=torch.float64)
    indices, values, _ = _spd_dense_to_coo(K_dense)
    b = torch.randn(n, dtype=torch.float64)

    x_scipy = SparseSolveAutograd.apply(indices, values, b, n)
    x_mumps = _MumpsSparseSolveAutograd.apply(indices, values, b, n)
    assert torch.allclose(x_scipy, x_mumps, atol=1e-10)


@pytest.mark.hpc
@pytest.mark.solver
@pytest.mark.skipif(not _petsc_functional(), reason="PETSc/MUMPS not functional")
def test_mumps_factor_cache_pattern_hit():
    """Same sparsity pattern, different values -> reuse the PETSc handle."""
    indices, values_a, n = _laplacian_1d(200)
    values_b = values_a * 1.7 + 0.01
    b = torch.randn(n, dtype=torch.float64)

    handle = make_factor_handle()
    assert handle.petsc_ksp is None
    x1 = _MumpsSparseSolveAutograd.apply(indices, values_a, b, n, handle)
    ksp_first = handle.petsc_ksp
    pat_first = handle.pattern_hash
    x2 = _MumpsSparseSolveAutograd.apply(indices, values_b, b, n, handle)

    assert torch.isfinite(x1).all() and torch.isfinite(x2).all()
    assert handle.petsc_ksp is ksp_first
    assert handle.pattern_hash == pat_first


@pytest.mark.hpc
@pytest.mark.solver
@pytest.mark.skipif(not _petsc_functional(), reason="PETSc/MUMPS not functional")
def test_mumps_factor_cache_pattern_miss():
    """Different sparsity pattern -> cache rebuild."""
    indices_a, values_a, n = _laplacian_1d(20)
    b = torch.randn(n, dtype=torch.float64)

    handle = make_factor_handle()
    _MumpsSparseSolveAutograd.apply(indices_a, values_a, b, n, handle)
    ksp_first = handle.petsc_ksp
    pat_first = handle.pattern_hash

    indices_b = torch.cat(
        [indices_a, torch.tensor([[0], [n - 1]], dtype=torch.long)],
        dim=1)
    values_b = torch.cat([values_a, torch.tensor([1e-3], dtype=torch.float64)])
    _MumpsSparseSolveAutograd.apply(indices_b, values_b, b, n, handle)

    assert handle.pattern_hash != pat_first
    assert handle.petsc_ksp is not ksp_first


@pytest.mark.hpc
@pytest.mark.solver
@pytest.mark.skipif(not _petsc_functional(), reason="PETSc/MUMPS not functional")
def test_mumps_factor_cache_correctness():
    indices, values, n = _laplacian_1d(50)
    values2 = values * 2.3
    b = torch.randn(n, dtype=torch.float64)

    x_uncached_a = _MumpsSparseSolveAutograd.apply(indices, values, b, n, None)
    x_uncached_b = _MumpsSparseSolveAutograd.apply(indices, values2, b, n, None)

    handle = make_factor_handle()
    x_cached_a = _MumpsSparseSolveAutograd.apply(indices, values, b, n, handle)
    x_cached_b = _MumpsSparseSolveAutograd.apply(indices, values2, b, n, handle)

    assert torch.allclose(x_uncached_a, x_cached_a, atol=1e-10)
    assert torch.allclose(x_uncached_b, x_cached_b, atol=1e-10)


def test_smoke_test_caches(monkeypatch):
    """`_petsc_functional()` must run the smoke test at most once per process."""
    from phast import sparse_solve as _ss

    _ss._reset_backend_cache()
    counter = {'n': 0}
    real_import = __builtins__['__import__'] if isinstance(
        __builtins__, dict) else __builtins__.__import__

    def counting_import(name, *args, **kwargs):
        if name == 'petsc4py':
            counter['n'] += 1
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', counting_import)
    try:
        _ss._petsc_functional()
        first = counter['n']
        _ss._petsc_functional()
        _ss._petsc_functional()
        assert counter['n'] == first, (
            f"smoke test re-ran: import count {counter['n']} > {first}")
    finally:
        _ss._reset_backend_cache()


def test_explicit_backend_falls_back_gracefully():
    """Forcing `_PETSC_FUNCTIONAL=False`, `solve(..., backend='mumps')` warns
    and returns the SciPy result instead of raising."""
    from phast import sparse_solve as _ss

    indices, values, n = _laplacian_1d(8)
    K_torch = torch.sparse_coo_tensor(
        indices, values, (n, n), dtype=torch.float64).coalesce()
    b = torch.randn(n, dtype=torch.float64)
    saved_petsc, saved_cudss = _ss._PETSC_FUNCTIONAL, _ss._CUDSS_FUNCTIONAL
    try:
        _ss._PETSC_FUNCTIONAL = False
        _ss._CUDSS_FUNCTIONAL = False
        with pytest.warns(RuntimeWarning, match="PETSc/MUMPS"):
            x_mumps = solve(K_torch, b, backend='mumps')
        with pytest.warns(RuntimeWarning, match="cuDSS"):
            x_cudss = solve(K_torch, b, backend='cudss')
        x_scipy = solve(K_torch, b, backend='scipy')
        assert torch.allclose(x_mumps, x_scipy, atol=1e-12)
        assert torch.allclose(x_cudss, x_scipy, atol=1e-12)
    finally:
        _ss._PETSC_FUNCTIONAL = saved_petsc
        _ss._CUDSS_FUNCTIONAL = saved_cudss


@pytest.mark.slow
def test_miehe_smoke():
    """Placeholder for the Miehe SENT end-to-end smoke; runs only with --runslow."""
    pytest.skip("Miehe SENT end-to-end smoke is intentionally not run here.")
