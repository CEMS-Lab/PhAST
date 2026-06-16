"""Sparse-direct linear solve with autograd support.

Phase 1 of the implicit-solver epic (#105, issue #106). The SciPy SuperLU
backend ships with the project's existing dependencies and serves as the
always-available baseline. Optional PETSc/MUMPS and cuDSS paths are selected
only after runtime smoke tests, so broken optional installs fall back cleanly.

The autograd Function caches the LU factorisation across forward and backward
so the adjoint solve reuses the factor. Forward solves ``K x = b``; backward
solves ``K^T λ = grad_x`` and emits

    grad_K_values[k] = -λ[i_k] * x[j_k]   for (i_k, j_k) in K's COO pattern
    grad_b           = λ

which is the standard implicit-derivative formula for a linear solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import warnings

import numpy as np
import torch


@dataclass
class MumpsFactorCache:
    """Symbolic-stage cache for PETSc/MUMPS across Newton iterations.

    MUMPS factorisation has a symbolic phase that depends on the sparsity
    pattern and a numeric phase that depends on matrix values. In a Newton loop
    with a fixed active-DOF set, the pattern is invariant, so callers can reuse
    the PETSc Mat/KSP and refresh only numeric values.
    """

    pattern_id: int = 0
    pattern_hash: Optional[int] = None
    petsc_ksp: Any = None
    petsc_mat: Any = None
    n: int = 0


def make_factor_handle() -> MumpsFactorCache:
    """Return a fresh MUMPS factor cache handle for one Newton solve."""
    return MumpsFactorCache()


def _pattern_fingerprint(K_indices: torch.Tensor) -> tuple[int, int]:
    """Cheap identity/hash fingerprint of a COO sparsity pattern."""
    arr = K_indices.detach().cpu().numpy()
    return id(K_indices), hash((arr.shape, arr.tobytes()))


_COO_INDEX_NUMPY_CACHE: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}
_COO_INDEX_NUMPY_CACHE_MAX = 64


def _coo_index_numpy(
        K_indices: torch.Tensor,
        *,
        dtype: np.dtype | type | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return cached NumPy COO row/column arrays for fixed sparsity patterns."""
    detached = K_indices.detach()
    target_dtype = np.dtype(dtype) if dtype is not None else None
    key = (
        detached.device.type,
        detached.device.index,
        detached.data_ptr(),
        tuple(detached.shape),
        tuple(detached.stride()),
        detached.dtype,
        getattr(detached, "_version", None),
        target_dtype.str if target_dtype is not None else None,
    )
    cached = _COO_INDEX_NUMPY_CACHE.get(key)
    if cached is not None:
        return cached

    arr = detached.to(device='cpu')
    if target_dtype is not None:
        torch_dtype = torch.int32 if target_dtype == np.dtype(np.int32) else torch.long
        arr = arr.to(dtype=torch_dtype)
    arr_np = arr.numpy()
    rows, cols = arr_np[0], arr_np[1]
    if len(_COO_INDEX_NUMPY_CACHE) >= _COO_INDEX_NUMPY_CACHE_MAX:
        _COO_INDEX_NUMPY_CACHE.pop(next(iter(_COO_INDEX_NUMPY_CACHE)))
    _COO_INDEX_NUMPY_CACHE[key] = (rows, cols)
    return rows, cols


_PETSC_INSTALL_HINT = (
    "petsc4py is not installed. Install via: "
    "`pip install petsc petsc4py` (CPU-only) or build PETSc with MUMPS support. "
    "The MUMPS backend is optional for epic #105 phase 2 (#107)."
)

# Process-local cache for runtime smoke tests. A broken install (import OK,
# solve fails — e.g. PETSc ABI mismatch or nvmath cuDSS API guess — seen in
# HPC validation 2026-05-09) is caught here so callers fall back to SciPy
# instead of dying mid-solve. See #403.
_PETSC_FUNCTIONAL: bool | None = None
_CUDSS_FUNCTIONAL: bool | None = None


@dataclass(frozen=True)
class SparseBackendStatus:
    """Functional-solve availability snapshot for backend dispatch.

    This makes the backend decision explicit and reusable across the
    mechanics solver, damage solver, and future sparse-adjoint seams.
    """

    scipy: bool
    petsc: bool
    cudss: bool

    def preferred(self, device_type: str = "cpu") -> str:
        """Return the best backend available for a device target."""
        if device_type == "cuda" and self.cudss:
            return "cudss"
        if self.petsc:
            return "mumps"
        if self.scipy:
            return "scipy"
        raise RuntimeError(
            "No functional sparse-direct backend found. Install scipy or "
            "an optional PETSc/MUMPS or cuDSS backend.")


def available_sparse_backends() -> SparseBackendStatus:
    """Probe backend support once and return a compact status snapshot."""
    return SparseBackendStatus(
        scipy=scipy_available(),
        petsc=_petsc_functional(),
        cudss=_cudss_functional(),
    )


def resolve_sparse_backend(
    backend: str = "auto",
    *,
    device_type: str = "cpu",
    status: SparseBackendStatus | None = None,
) -> str:
    """Resolve a backend request against the current runtime.

    This centralizes the selection logic so the mechanics solver can make
    the same decision without duplicating availability checks.
    """
    status = status or available_sparse_backends()
    if backend == "auto":
        return status.preferred(device_type=device_type)
    backend = backend.lower()
    if backend in ("petsc", "mumps"):
        backend = "mumps"
    if backend not in ("scipy", "mumps", "cudss"):
        raise ValueError(
            f"Unknown sparse backend '{backend}'. Expected auto/scipy/mumps/cudss."
        )
    if backend == "scipy" and not status.scipy:
        raise RuntimeError("scipy is required for backend='scipy'.")
    if backend == "mumps" and not status.petsc:
        warnings.warn(
            "PETSc/MUMPS backend requested but not functional (import OK but "
            "smoke test failed — likely ABI mismatch, see #403). Falling back "
            "to SciPy SuperLU. Debug with "
            "`python -c 'from petsc4py import PETSc; PETSc.Mat().createAIJ((2,2))'`.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "scipy"
    if backend == "cudss" and not status.cudss:
        warnings.warn(
            "cuDSS backend requested but not functional (nvmath import OK but "
            "DirectSolver smoke test failed — see #403/#379). Falling back to "
            "SciPy SuperLU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "scipy"
    return backend


def scipy_available() -> bool:
    try:
        import scipy.sparse  # noqa: F401
        import scipy.sparse.linalg  # noqa: F401
        return True
    except Exception:
        return False


def _has_petsc4py() -> bool:
    """Import-only fast-path. Use :func:`_petsc_functional` for backend-selection (it detects ABI-broken installs)."""
    try:
        import petsc4py  # noqa: F401
        return True
    except Exception:
        return False


def _has_cudss() -> bool:
    """Import-only fast-path for nvmath. Use :func:`_cudss_functional` for dispatch."""
    try:
        import nvmath  # noqa: F401
        return True
    except Exception:
        return False


def _petsc_functional() -> bool:
    """Return True iff petsc4py imports AND a small SPD MUMPS solve works.

    Cached for the process lifetime. A broken install (ABI mismatch, missing
    MUMPS factor type, etc.) caches False and triggers a SciPy fallback in
    callers — see #403.
    """
    global _PETSC_FUNCTIONAL
    if _PETSC_FUNCTIONAL is not None:
        return _PETSC_FUNCTIONAL
    try:
        import petsc4py
        if not getattr(petsc4py, "_phast_initialised", False):
            petsc4py.init([])
            petsc4py._phast_initialised = True
        from petsc4py import PETSc

        A = PETSc.Mat().createAIJ(size=(2, 2), nnz=2)
        for i, j, v in ((0, 0, 2.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0)):
            A.setValue(i, j, v)
        A.assemble()
        b = PETSc.Vec().createSeq(2)
        b.setValues([0, 1], [1.0, 0.0])
        b.assemble()
        x = b.duplicate()
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        try:
            pc.setFactorSolverType('mumps')
        except Exception:
            # MUMPS factor type unavailable — accept whatever LU PETSc has.
            pass
        ksp.solve(b, x)
        _PETSC_FUNCTIONAL = True
    except Exception:
        _PETSC_FUNCTIONAL = False
    return _PETSC_FUNCTIONAL


def _ensure_petsc_initialised():
    """Initialise petsc4py once with an empty argv."""
    import petsc4py
    if not getattr(petsc4py, "_phast_initialised", False):
        petsc4py.init([])
        petsc4py._phast_initialised = True


def _cudss_functional() -> bool:
    """Return True iff a 2x2 cuDSS direct solve via nvmath works.

    Cached for the process lifetime. Catches the
    `DirectSolver(K)` TypeError seen in HPC validation 2026-05-09 (#379).
    """
    global _CUDSS_FUNCTIONAL
    if _CUDSS_FUNCTIONAL is not None:
        return _CUDSS_FUNCTIONAL
    try:
        if not torch.cuda.is_available():
            _CUDSS_FUNCTIONAL = False
            return _CUDSS_FUNCTIONAL
        from nvmath.sparse.advanced import DirectSolver  # type: ignore
        import scipy.sparse as sp

        K = sp.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
        rhs = np.array([1.0, 0.0])
        with DirectSolver(K, rhs) as solver:
            solver.plan()
            solver.factorize()
            _ = solver.solve()
        _CUDSS_FUNCTIONAL = True
    except Exception:
        _CUDSS_FUNCTIONAL = False
    return _CUDSS_FUNCTIONAL


def _reset_backend_cache() -> None:
    """Clear cached smoke-test results — test-only helper."""
    global _PETSC_FUNCTIONAL, _CUDSS_FUNCTIONAL
    _PETSC_FUNCTIONAL = None
    _CUDSS_FUNCTIONAL = None


class SparseSolveAutograd(torch.autograd.Function):
    """Autograd-aware sparse-direct solve via SciPy SuperLU.

    Inputs
    ------
    K_indices : (2, nnz) int64 COO indices on CPU
    K_values  : (nnz,)  float64 on CPU, may carry requires_grad
    b         : (n,)    float64 on CPU, may carry requires_grad
    n         : int, system size

    Returns
    -------
    x : (n,) float64 with grad_fn pointing back at K_values and b.
    """

    @staticmethod
    def forward(ctx, K_indices, K_values, b, n):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        rows, cols = _coo_index_numpy(K_indices)
        vals = K_values.detach().cpu().numpy().astype(np.float64, copy=False)
        b_np = b.detach().cpu().numpy().astype(np.float64, copy=False)

        K_csc = sp.coo_matrix((vals, (rows, cols)), shape=(int(n), int(n))).tocsc()

        # SuperLU LU factor; reused for the adjoint in backward.
        lu = spla.splu(K_csc)
        x_np = lu.solve(b_np)

        x = torch.from_numpy(x_np).to(dtype=b.dtype, device=b.device)

        ctx.save_for_backward(K_indices, K_values, x)
        ctx.lu = lu
        ctx.n = int(n)
        ctx.K_csc_shape = K_csc.shape
        return x

    @staticmethod
    def backward(ctx, grad_x):
        K_indices, K_values, x = ctx.saved_tensors
        lu = ctx.lu

        # Adjoint: K^T λ = grad_x. SuperLU exposes trans='T' on .solve.
        g_np = grad_x.detach().cpu().numpy().astype(np.float64, copy=False)
        try:
            lam_np = lu.solve(g_np, trans='T')
        except TypeError:
            # Fallback for older SciPy without trans kw: re-factor K^T.
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
            rows, cols = _coo_index_numpy(K_indices)
            vals = K_values.detach().cpu().numpy().astype(np.float64, copy=False)
            KT = sp.coo_matrix(
                (vals, (cols, rows)),
                shape=(ctx.n, ctx.n)).tocsc()
            lam_np = spla.splu(KT).solve(g_np)

        lam = torch.from_numpy(lam_np).to(
            dtype=K_values.dtype, device=K_values.device)
        x_cpu = x.detach().to(dtype=K_values.dtype, device=K_values.device)

        i = K_indices[0].to(torch.long)
        j = K_indices[1].to(torch.long)
        grad_K_values = -lam[i] * x_cpu[j]
        grad_b = lam.to(dtype=grad_x.dtype, device=grad_x.device)

        # Match forward signature: (K_indices, K_values, b, n)
        return None, grad_K_values, grad_b, None


class _MumpsSparseSolveAutograd(torch.autograd.Function):
    """PETSc/MUMPS sparse-direct backend with autograd adjoint."""

    @staticmethod
    def forward(ctx, K_indices, K_values, b, n, factor_handle=None):
        if not _petsc_functional():
            raise RuntimeError(_PETSC_INSTALL_HINT)
        _ensure_petsc_initialised()
        from petsc4py import PETSc
        import scipy.sparse as sp

        rows, cols = _coo_index_numpy(K_indices, dtype=np.int32)
        vals = K_values.detach().cpu().numpy().astype(np.float64, copy=False)
        b_np = b.detach().cpu().numpy().astype(np.float64, copy=False)
        n_int = int(n)

        K_csr = sp.coo_matrix(
            (vals, (rows, cols)), shape=(n_int, n_int)
        ).tocsr()
        K_csr.sum_duplicates()

        cache_hit = False
        if factor_handle is not None and factor_handle.petsc_ksp is not None:
            fp_id, fp_hash = _pattern_fingerprint(K_indices)
            same_pattern = (
                factor_handle.n == n_int
                and (factor_handle.pattern_id == fp_id
                     or factor_handle.pattern_hash == fp_hash)
            )
            if same_pattern:
                A = factor_handle.petsc_mat
                ksp = factor_handle.petsc_ksp
                A.zeroEntries()
                A.setValuesCSR(
                    K_csr.indptr.astype(PETSc.IntType),
                    K_csr.indices.astype(PETSc.IntType),
                    K_csr.data.astype(PETSc.ScalarType),
                )
                A.assemble()
                cache_hit = True

        if not cache_hit:
            A = PETSc.Mat().createAIJ(
                size=(n_int, n_int),
                csr=(K_csr.indptr.astype(PETSc.IntType),
                     K_csr.indices.astype(PETSc.IntType),
                     K_csr.data.astype(PETSc.ScalarType)),
                comm=PETSc.COMM_SELF,
            )
            A.assemble()
            ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
            ksp.setOperators(A)
            ksp.setType('preonly')
            pc = ksp.getPC()
            pc.setType('lu')
            try:
                pc.setFactorSolverType('mumps')
            except Exception:
                # Functional smoke test already proved PETSc LU works. Some
                # local CPU builds expose PETSc LU without an explicit MUMPS
                # factor type; keep the backend usable rather than failing.
                pass
            ksp.setFromOptions()
            if factor_handle is not None:
                fp_id, fp_hash = _pattern_fingerprint(K_indices)
                factor_handle.pattern_id = fp_id
                factor_handle.pattern_hash = fp_hash
                factor_handle.petsc_mat = A
                factor_handle.petsc_ksp = ksp
                factor_handle.n = n_int

        b_vec = PETSc.Vec().createSeq(n_int, comm=PETSc.COMM_SELF)
        b_vec.setArray(b_np.copy())
        x_vec = b_vec.duplicate()
        ksp.solve(b_vec, x_vec)
        x_np = x_vec.getArray(readonly=True).copy()
        x = torch.from_numpy(x_np).to(dtype=b.dtype, device=b.device)

        ctx.save_for_backward(K_indices, K_values, x)
        ctx.ksp = ksp
        ctx.n = n_int
        return x

    @staticmethod
    def backward(ctx, grad_x):
        from petsc4py import PETSc

        K_indices, K_values, x = ctx.saved_tensors
        g_np = grad_x.detach().cpu().numpy().astype(np.float64, copy=False)
        g_vec = PETSc.Vec().createSeq(ctx.n, comm=PETSc.COMM_SELF)
        g_vec.setArray(g_np.copy())
        lam_vec = g_vec.duplicate()
        ctx.ksp.solveTranspose(g_vec, lam_vec)
        lam_np = lam_vec.getArray(readonly=True).copy()

        lam = torch.from_numpy(lam_np).to(
            dtype=K_values.dtype, device=K_values.device)
        x_cpu = x.detach().to(dtype=K_values.dtype, device=K_values.device)
        i = K_indices[0].to(torch.long)
        j = K_indices[1].to(torch.long)
        grad_K_values = -lam[i] * x_cpu[j]
        grad_b = lam.to(dtype=grad_x.dtype, device=grad_x.device)
        return None, grad_K_values, grad_b, None, None


class _CuDSSSparseSolveAutograd(torch.autograd.Function):
    """NVIDIA cuDSS sparse-direct backend with autograd adjoint.

    nvmath's direct-solver API owns both operands at construction time:
    ``DirectSolver(a, b)`` then ``plan()``, ``factorize()``, ``solve()``.
    Backward reuses the implicit derivative formula and solves the transpose
    system by constructing a second direct solve for ``K.T``.
    """

    @staticmethod
    def forward(ctx, K_indices, K_values, b, n):
        if not _cudss_functional():
            raise RuntimeError(
                "cuDSS backend requested but nvmath/cuDSS is not functional.")
        from nvmath.sparse.advanced import DirectSolver  # type: ignore
        import scipy.sparse as sp

        rows, cols = _coo_index_numpy(K_indices)
        vals = K_values.detach().cpu().numpy().astype(np.float64, copy=False)
        b_np = b.detach().cpu().numpy().astype(np.float64, copy=False)
        n_int = int(n)
        K_csr = sp.coo_matrix(
            (vals, (rows, cols)), shape=(n_int, n_int)
        ).tocsr()
        K_csr.sum_duplicates()

        with DirectSolver(K_csr, b_np) as solver:
            solver.plan()
            solver.factorize()
            x_obj = solver.solve()
        x_np = np.asarray(x_obj, dtype=np.float64)
        x = torch.from_numpy(x_np).to(dtype=b.dtype, device=b.device)

        ctx.save_for_backward(K_indices, K_values, x)
        ctx.n = n_int
        return x

    @staticmethod
    def backward(ctx, grad_x):
        from nvmath.sparse.advanced import DirectSolver  # type: ignore
        import scipy.sparse as sp

        K_indices, K_values, x = ctx.saved_tensors
        rows, cols = _coo_index_numpy(K_indices)
        vals = K_values.detach().cpu().numpy().astype(np.float64, copy=False)
        g_np = grad_x.detach().cpu().numpy().astype(np.float64, copy=False)
        KT_csr = sp.coo_matrix(
            (vals, (cols, rows)), shape=(ctx.n, ctx.n)
        ).tocsr()
        KT_csr.sum_duplicates()

        with DirectSolver(KT_csr, g_np) as solver:
            solver.plan()
            solver.factorize()
            lam_obj = solver.solve()
        lam = torch.from_numpy(np.asarray(lam_obj, dtype=np.float64)).to(
            dtype=K_values.dtype, device=K_values.device)
        x_cpu = x.detach().to(dtype=K_values.dtype, device=K_values.device)
        i = K_indices[0].to(torch.long)
        j = K_indices[1].to(torch.long)
        grad_K_values = -lam[i] * x_cpu[j]
        grad_b = lam.to(dtype=grad_x.dtype, device=grad_x.device)
        return None, grad_K_values, grad_b, None


def _coo_from_torch_sparse(K):
    """Return (indices(2,nnz), values(nnz,), n) on CPU float64 from a torch sparse tensor."""
    if K.is_sparse:
        K_coo = K.coalesce() if K.layout == torch.sparse_coo else K.to_sparse_coo().coalesce()
        idx = K_coo.indices().detach().cpu().to(torch.long)
        val = K_coo.values().detach().cpu().to(torch.float64)
        n = int(K_coo.shape[0])
        return idx, val, n
    if K.layout == torch.sparse_csr:
        K_coo = K.to_sparse_coo().coalesce()
        return _coo_from_torch_sparse(K_coo)
    raise TypeError(f"Expected sparse torch tensor, got layout={K.layout}")


def _coo_values_for_autograd(K):
    """Return sparse values in the same coalesced COO order without detaching."""
    if K.layout == torch.sparse_coo:
        return K.coalesce().values().to(torch.float64).cpu()
    if K.layout == torch.sparse_csr:
        return K.to_sparse_coo().coalesce().values().to(torch.float64).cpu()
    return None


def solve(K_torch_sparse, b_torch_dense, *, backend: str = 'auto') -> torch.Tensor:
    """Convenience wrapper around :class:`SparseSolveAutograd`.

    Parameters
    ----------
    K_torch_sparse : torch sparse COO/CSR tensor of shape (n, n)
    b_torch_dense  : torch dense tensor of shape (n,)
    backend        : 'auto' | 'scipy' | 'mumps' | 'cudss'

    Returns
    -------
    x : torch dense tensor of shape (n,)
    """
    backend = resolve_sparse_backend(
        backend,
        device_type=K_torch_sparse.device.type,
    )
    if backend == 'scipy':
        if not scipy_available():
            raise RuntimeError("scipy is required for backend='scipy'.")
        idx, val, n = _coo_from_torch_sparse(K_torch_sparse)
        b = b_torch_dense.detach().cpu().to(torch.float64) \
            if not b_torch_dense.requires_grad else b_torch_dense.to(torch.float64).cpu()
        # Preserve requires_grad through .apply by passing the original tensors
        # when possible. We pass val (may require grad) and b (may require grad).
        val_in = _coo_values_for_autograd(K_torch_sparse)
        if val_in is None:
            val_in = val
        b_in = b_torch_dense.to(torch.float64).cpu() \
            if b_torch_dense.dtype != torch.float64 or b_torch_dense.device.type != 'cpu' \
            else b_torch_dense
        x = SparseSolveAutograd.apply(idx, val_in, b_in, n)
        return x.to(device=b_torch_dense.device, dtype=b_torch_dense.dtype)
    if backend == 'mumps':
        idx, val, n = _coo_from_torch_sparse(K_torch_sparse)
        val_in = _coo_values_for_autograd(K_torch_sparse)
        if val_in is None:
            val_in = val
        b_in = b_torch_dense.to(torch.float64).cpu() \
            if b_torch_dense.dtype != torch.float64 or b_torch_dense.device.type != 'cpu' \
            else b_torch_dense
        x = _MumpsSparseSolveAutograd.apply(idx, val_in, b_in, n)
        return x.to(device=b_torch_dense.device, dtype=b_torch_dense.dtype)
    if backend == 'cudss':
        idx, val, n = _coo_from_torch_sparse(K_torch_sparse)
        val_in = _coo_values_for_autograd(K_torch_sparse)
        if val_in is None:
            val_in = val
        b_in = b_torch_dense.to(torch.float64).cpu() \
            if b_torch_dense.dtype != torch.float64 or b_torch_dense.device.type != 'cpu' \
            else b_torch_dense
        x = _CuDSSSparseSolveAutograd.apply(idx, val_in, b_in, n)
        return x.to(device=b_torch_dense.device, dtype=b_torch_dense.dtype)
    raise ValueError(
        f"backend must be 'auto'|'scipy'|'mumps'|'cudss', got {backend!r}")
