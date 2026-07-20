# Sparse linear solver

PhAST is organized around a matrix-free primary solve path, but several public
workflows benefit from an explicit sparse operator during preconditioner setup,
validation, or small auxiliary solves. This page explains when that route is
used, what it provides, and how to choose the backend.

## Purpose

The sparse pathway exists for three reasons:

- to build stronger preconditioners for the damage and mechanics solves;
- to support direct inspection of assembled operators when debugging a run;
- to provide a conventional sparse representation for workflows that need it.

The main forward solve still follows the matrix-free path. Sparse assembly is a
supporting capability rather than the default execution model.

## When to use the sparse path

Use a sparse representation when one of the following is true:

- convergence is poor with a purely diagonal preconditioner;
- a run must be compared against a direct or assembled reference solver;
- a workflow needs a conventional matrix for analysis or external tooling;
- a small problem is better served by a direct solve than by a Krylov method.

For large fracture calculations, the sparse matrix is typically most useful as
an internal preconditioning aid. For small verification problems, a direct
sparse solve may be acceptable.

## Backend selection

Backend choice depends on the hardware and on whether the solve is direct or
iterative.

| Platform | Preferred option | Notes |
|---|---|---|
| CUDA GPU | `amg` or `gmg` | `amg` is appropriate when a sparse setup is worthwhile; `gmg` is the portable fallback. |
| CPU | `amg` or direct sparse solve | Use a direct solver for small problems and AMG for larger ones. |
| Apple MPS | `gmg` | Portable and numerically robust for the current PhAST workflows. |

The exact solver selection is controlled through the public configuration
surface and the corresponding example deck.

## Preconditioner quality

The main practical benefit of sparse assembly is improved preconditioning. A
simple diagonal preconditioner is often sufficient for small verification
problems, but fracture simulations with sharp stiffness contrasts usually
benefit from a multilevel preconditioner.

Common options are:

- **Jacobi**: the simplest and most portable choice.
- **Geometric multigrid**: suitable when a coarse hierarchy can be built from
  the mesh and the physics remain well conditioned.
- **Algebraic multigrid**: useful when an assembled sparse matrix is available
  and a stronger hierarchy is needed.

## Practical guidance

1. Prefer the matrix-free forward path for the main nonlinear solve.
2. Assemble a sparse operator only when it is needed for preconditioning or
   debugging.
3. Use the smallest backend that still converges reliably for the target
   problem.
4. Verify a new configuration on a small mesh before scaling to a larger run.
5. If a sparse path is used for a public example, record the exact backend and
   solver settings in the run manifest.

## Relation to the rest of PhAST

Sparse solving is one component of the broader workflow:

1. define the geometry and regions;
2. assign materials and boundary conditions;
3. choose the analysis step and solver settings;
4. execute the run;
5. inspect the stored histories and visual artifacts.

The sparse solver supports step 3, but it does not replace the workflow
definitions or the result-inspection APIs.
