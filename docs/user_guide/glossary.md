# Glossary

**Analysis step**  A named solution stage containing an analysis kind, loading controls, and active boundary conditions.

**Backend**  The implementation selected for an operation, such as SciPy, PETSc/MUMPS, or cuDSS for an assembled sparse solve.

**Device**  The compute target on which tensors and supported operations run, such as CPU, CUDA, or MPS.

**Discrete form**  The finite-dimensional mesh-and-basis representation of a weak problem, including assembled degrees of freedom and operators.

**Fallback**  An explicit alternate implementation selected when a requested route is unavailable or non-functional.

**History**  A stored scalar or tabular record indexed by step or time, such as reaction force, energy, or solver telemetry.

**JVP**  Jacobian-vector product: application of a derivative to a direction without necessarily materializing the full Jacobian.

**Operator**  A residual or linearized mapping evaluated in assembled or matrix-free form.

**Preconditioner**  An approximate inverse or transformation used to improve an iterative linear solve; it is not itself the physical model.

**Result**  A read-only handle for artifacts and stored state in a completed run directory.

**Secant**  A finite change quotient between two states, rather than a local derivative at one state.

**Strong form**  The pointwise differential statement of the governing equations and boundary/initial data.

**Tangent**  A local derivative of a residual or constitutive update with respect to an increment.

**Trajectory**  A stored sequence of field snapshots, in the preferred Zarr or legacy H5 format when enabled.

**Weak form**  An integral statement obtained by testing the strong form and integrating by parts.
