# Student notebooks

These notebooks complement the canonical
[`Getting Started`](https://cems-lab.github.io/PhAST/getting-started.html)
route. They distinguish executable workflow instruction from retained numerical
evidence.

| Sequence | Notebook | Scope |
|---|---|---|
| 01 | [`docs/tutorial/problem_setup_walkthrough.ipynb`](../../docs/tutorial/problem_setup_walkthrough.ipynb) | SENT geometry, mesh, boundary conditions, material, solver, and output setup; the default two-step CPU run is a workflow check rather than crack-propagation evidence. |
| 02 | [`02_mesh_resolution_diagnostic.ipynb`](02_mesh_resolution_diagnostic.ipynb) | Analytical AT2 profile sampling and an $h/\ell_0$ resolution diagnostic; not a solved convergence study. |
| 03 | [`03_miehe_retained_results.ipynb`](03_miehe_retained_results.ipynb) | Inspection of the checked-in Miehe SENT reference calculation, including load-displacement and damage evolution. |

The current public CLI does not expose a one-command VTU/PVD export, and
`Result.export()` does not generate new artifacts. Lower-level VTU writing APIs
exist for advanced workflows, but a ParaView time-series contract is not part
of these newcomer notebooks. The repository does not currently present
asymmetric three-point bending or L-shaped panel notebooks as public validated
benchmarks. Those require benchmark-specific numerical evidence before they can
be added to this sequence.
