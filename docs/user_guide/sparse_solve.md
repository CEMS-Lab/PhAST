# Sparse linear solver (autograd-enabled)

Backend selection, worked examples, and how the autograd-aware sparse solve
fits into the implicit-solver path.


This makes the code:
- **Readable**: Any PyTorch user can follow the FEM assembly
- **Debuggable**: Standard PyTorch profiling and debugging tools work
- **Portable**: Runs on CUDA, MPS, and CPU without platform-specific sparse libraries
- **Testable**: Easy to verify against analytical solutions

### Where Matrix-Free Hurts Us

We are honest about the trade-offs. Matrix-free has real disadvantages:

#### 1. Preconditioner Quality

Historically our biggest gap vs reference codes. The best preconditioners (AMG,
ILU, Cholesky) require an assembled sparse matrix.

**What we have now (v0.9.0):**

- **Jacobi** (diagonal): 3-10x iteration reduction. Always available.
- **2-level Geometric Multigrid** (v0.9.0, `multigrid.py`): 5-10x iteration
  reduction on top of Jacobi. Uses our scatter infrastructure for the fine level
  and assembles a dense coarse operator via node aggregation — **no global sparse
  matrix required.** See the **Multigrid Preconditioner** section below for
  details.

**What the reference codes have:**
- **MUMPS** (direct LU): Zero iterations — one factorization, exact solve. But
  O(N^1.5) memory and not GPU-friendly.
- **AMG** (algebraic multigrid): 50-100x iteration reduction. Requires sparse K.
- **hypre**: GPU-accelerated AMG. Requires sparse K.

**Current state:** With 2-level GMG, our damage CG converges in ~5-10 iterations
(vs ~50 with Jacobi alone). This is competitive with AMG on moderately-sized
meshes (<50K nodes). For very large meshes (>100K nodes), AMG's ability to build
deeper hierarchies gives it an edge.

**Remaining plan (see issue #67):**
- Medium-term: Assemble sparse K only for the preconditioner setup, keep the
  matvec matrix-free. Feed K to PyAMG for a proper AMG hierarchy. Best of both
  worlds.
- The method `mesh.assemble_sparse_matrix()` already exists for this purpose.

#### 2. Direct Solvers

For small problems (<10K DOFs), direct solvers (MUMPS, SuperLU) are faster than
CG — one factorisation, no iteration. We can't use direct solvers without a
sparse matrix. For our target use case (training data generation on GPU with
10K-1M DOFs), iterative solvers are appropriate.

#### 3. Condition Number Estimation

AMG setup automatically estimates the condition number and adapts the hierarchy.
Our Jacobi preconditioner is blind to conditioning. We compensate with:
- Convergence checks every 50 iterations (not every iteration)
- CG divergence detection (`||r|| > 1e6 * ||r_0||`)
- H-capping to prevent condition number explosion near d→1

### The Middle Ground: Selective Sparse Assembly

We are moving toward a hybrid approach where the **matvec stays matrix-free** but
we **assemble K once** for preconditioner construction:

```python

