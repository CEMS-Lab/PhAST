"""Tests for mixed-precision Krylov solver (#118)."""
import importlib.util
import os
import time

import pytest
import torch


def _load_cg():
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(here)
    mod_path = os.path.join(pkg_dir, "src", "phast", "mixed_precision_cg.py")
    spec = importlib.util.spec_from_file_location('_mp_cg', mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cg_mixed_precision


cg_mixed_precision = _load_cg()


def _laplacian_1d(n, dtype=torch.float64):
    A = torch.zeros((n, n), dtype=dtype)
    idx = torch.arange(n)
    A[idx, idx] = 2.0
    A[idx[:-1], idx[1:]] = -1.0
    A[idx[1:], idx[:-1]] = -1.0
    return A


def _make_problem(n, seed=0):
    torch.manual_seed(seed)
    A = _laplacian_1d(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)

    def matvec(v):
        return A.to(v.dtype) @ v

    x_ref = torch.linalg.solve(A, b)
    return matvec, b, x_ref


def test_float64_baseline():
    matvec, b, x_ref = _make_problem(100)
    x, iters, conv = cg_mixed_precision(matvec, b, tol=1e-12, max_iter=500,
                                        precision='float64')
    assert conv
    assert torch.linalg.vector_norm(x - x_ref).item() < 1e-9


def test_float32_correctness():
    matvec, b, x_ref = _make_problem(100)
    x, iters, conv = cg_mixed_precision(matvec, b, tol=1e-6, max_iter=2000,
                                        precision='float32')
    err = (torch.linalg.vector_norm(x - x_ref).item()
           / torch.linalg.vector_norm(x_ref).item())
    assert err < 1e-4, f"float32 relative error {err:.2e} too large"


def test_mixed_correctness():
    matvec, b, x_ref = _make_problem(100)
    x, iters, conv = cg_mixed_precision(matvec, b, tol=1e-9, max_iter=2000,
                                        precision='mixed', max_refine=10)
    err = (torch.linalg.vector_norm(x - x_ref).item()
           / torch.linalg.vector_norm(x_ref).item())
    assert err < 1e-7, f"mixed-precision relative error {err:.2e} too large"


def test_mixed_speedup_smoketest():
    n = 2000
    matvec, b, _ = _make_problem(n, seed=1)

    t0 = time.perf_counter()
    x64, _, conv64 = cg_mixed_precision(matvec, b, tol=1e-9, max_iter=20000,
                                        precision='float64')
    t_f64 = time.perf_counter() - t0
    assert conv64

    t0 = time.perf_counter()
    x_mx, _, _ = cg_mixed_precision(matvec, b, tol=1e-7, max_iter=20000,
                                    precision='mixed', max_refine=10)
    t_mx = time.perf_counter() - t0

    err = (torch.linalg.vector_norm(x_mx - x64).item()
           / torch.linalg.vector_norm(x64).item())
    assert err < 1e-5, f"mixed result diverges from float64 by {err:.2e}"
    print(f"\n  n={n}: float64 {t_f64*1e3:.1f} ms, mixed {t_mx*1e3:.1f} ms")
