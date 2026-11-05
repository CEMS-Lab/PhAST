"""Mixed-precision conjugate-gradient solver (issue #118).

Provides ``cg_mixed_precision`` with three modes:

  - 'float64' : reference path (default; preserves existing behaviour).
  - 'float32' : full single precision (memory-bandwidth + tensor-core friendly).
  - 'mixed'   : float32 matvec inside CG; float64 residual / convergence test
                with iterative refinement to recover most of float64 accuracy.

The mixed mode follows the three-precision iterative refinement pattern of
Carson & Higham 2018, "Accelerating the solution of linear systems by
iterative refinement in three precisions", SIAM J. Sci. Comput. 40(2).
"""

import torch


def cg_mixed_precision(matvec, rhs, x0=None, tol=1e-8, max_iter=1000,
                        precision='float64', max_refine=5):
    """Conjugate-gradient solve A x = rhs with optional reduced-precision matvec.

    Parameters
    ----------
    matvec : callable v -> A @ v (must accept dtype of v and return same dtype).
    rhs    : torch.Tensor; its dtype defines the "high" working precision.
    x0     : optional initial guess (same dtype as rhs).
    tol    : relative residual tolerance.
    precision : 'float64' | 'float32' | 'mixed'.
    max_refine : iterative-refinement steps for 'mixed' mode.

    Returns
    -------
    (x, iters, converged)
    """
    if precision not in ('float64', 'float32', 'mixed'):
        raise ValueError(
            f"precision must be float64|float32|mixed, got {precision!r}")

    hi_dtype = rhs.dtype
    lo_dtype = torch.float32

    def _cg_inner(matvec_fn, b, x_init, work_dtype, tol_val, n_iter):
        if x_init is not None:
            x = x_init.to(work_dtype).clone()
            r = b.to(work_dtype) - matvec_fn(x)
        else:
            x = torch.zeros_like(b, dtype=work_dtype)
            r = b.to(work_dtype).clone()
        p = r.clone()
        rr = (r * r).sum()
        rr0 = max(rr.item(), 1.0)
        tol_sq = tol_val * tol_val
        for i in range(n_iter):
            Ap = matvec_fn(p)
            pAp = (p * Ap).sum()
            alpha = rr / (pAp + 1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = (r * r).sum()
            p = p * (rr_new / (rr + 1e-30)) + r
            rr = rr_new
            if rr.item() < tol_sq * rr0:
                return x, i + 1, True
        return x, n_iter, False

    if precision == 'float64':
        return _cg_inner(matvec, rhs, x0, hi_dtype, tol, max_iter)

    if precision == 'float32':
        def mv32(v):
            return matvec(v.to(lo_dtype)).to(lo_dtype)
        b32 = rhs.to(lo_dtype)
        x0_32 = x0.to(lo_dtype) if x0 is not None else None
        x, iters, conv = _cg_inner(mv32, b32, x0_32, lo_dtype, tol, max_iter)
        return x.to(hi_dtype), iters, conv

    # 'mixed': low-precision inner solve + float64 iterative refinement.
    def mv_lo(v):
        return matvec(v.to(lo_dtype)).to(lo_dtype)

    x_hi = (x0.clone().to(hi_dtype) if x0 is not None
            else torch.zeros_like(rhs, dtype=hi_dtype))
    total_iters = 0
    converged = False
    b_norm = max(torch.linalg.vector_norm(rhs).item(), 1.0)
    for _ in range(max_refine):
        r_hi = rhs - matvec(x_hi.to(hi_dtype)).to(hi_dtype)
        if torch.linalg.vector_norm(r_hi).item() <= tol * b_norm:
            converged = True
            break
        dx_lo, n_in, _ = _cg_inner(mv_lo, r_hi.to(lo_dtype), None,
                                   lo_dtype, tol, max_iter)
        total_iters += n_in
        x_hi = x_hi + dx_lo.to(hi_dtype)
    return x_hi, total_iters, converged
