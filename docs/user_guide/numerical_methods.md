# Numerical methods

PhAST evaluates supported continuum models through finite-element operators and solver-specific update routes. The following description separates the physical statement, its weak form, its discrete form, and the iterations used to obtain a state.

## Strong, weak, and discrete forms

The **strong form** states the differential balance law in the domain, constitutive relations, initial data, and boundary conditions. The **weak form** multiplies the balance by admissible test functions and integrates by parts, moving derivatives and natural boundary terms into an integral statement. The **discrete form** selects a mesh, finite-element shape functions, quadrature, and degrees of freedom. Assembly then produces residual vectors and, where required, tangent or secant operators for the finite set of unknowns.

These terms describe successive mathematical representations, not three independent solver modes. A public configuration selects an implemented route; it does not provide a general weak-form compiler.

## Nonlinear and staggered levels

An analysis step may contain several levels of iteration:

- A load or time increment advances the prescribed control.
- A staggered iteration alternates the coupled mechanics and phase-field updates for quasi-static fracture.
- A Newton iteration linearizes a nonlinear subproblem around the current iterate when that solver route uses Newton updates.
- A Krylov or conjugate-gradient iteration solves the resulting linear system approximately; a direct sparse factorization instead performs a factor and triangular solve.
- A local constitutive iteration, such as a return mapping, updates an integration-point state where the material route requires it.

Iteration counts in telemetry must therefore be read with their level labels. A linear-solver iteration count is not a staggered convergence count, and none of these counts alone establishes physical accuracy.

## Tangents, secants, and JVPs

A **tangent** is the local derivative of the discrete residual or constitutive response with respect to an increment. An algorithmic or consistent tangent matches the implemented update, including local internal-variable treatment, where available. A **secant** is a finite change quotient between two states; it describes an interval rather than the derivative at one state. A **JVP** (Jacobian-vector product) applies a derivative to a direction without forming the full Jacobian. It is useful for matrix-free linearization and differentiable workflows.

The sparse autograd interface uses an adjoint transpose solve for the reverse derivative of `K x = b`; this should not be conflated with a constitutive tangent or a finite-difference secant.

## Operator, solver, preconditioner, backend, device, fallback

An **operator** evaluates or represents a residual/Jacobian action. A **linear solver** computes an update from that operator. A **preconditioner** changes the conditioning of an iterative solve. A **backend** is the implementation used for a requested operation, such as SciPy or PETSc/MUMPS sparse direct solve. A **device** is the execution target, such as CPU, CUDA, or MPS. A **fallback** is an explicit alternate route selected when a requested optional implementation is unavailable or non-functional; it is not evidence that the requested backend ran.

The [sparse-solve guide](sparse_solve.md) gives the low-level dispatch details.
