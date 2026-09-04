# Student notebooks

These notebooks complement the canonical [Getting Started](https://cems-lab.github.io/PhAST/getting-started.html) route. They use repository-relative paths and CPU-oriented instructions; run them from the repository root or open the documentation copies in Colab after selecting the intended immutable source revision.

| Sequence | Notebook | Scope |
|---|---|---|
| 01 | [`docs/tutorial/problem_setup_walkthrough.ipynb`](../../docs/tutorial/problem_setup_walkthrough.ipynb) | Build and inspect a SENT problem: geometry, mesh groups, supports, loading, classical solver settings, and retained outputs. |
| 02 | [`02_mesh_resolution_diagnostic.ipynb`](02_mesh_resolution_diagnostic.ipynb) | Analytical AT2 profile sampling at fixed `l0`; not an FEM convergence study. Documentation copy: [`docs/tutorial/notebook_mesh_resolution.ipynb`](../../docs/tutorial/notebook_mesh_resolution.ipynb). |
| 03 | [`03_miehe_retained_results.ipynb`](03_miehe_retained_results.ipynb) | Inspect checked-in Miehe SENT histories and evidence metadata without rerunning the retained calculation. Documentation copy: [`docs/tutorial/notebook_retained_results.ipynb`](../../docs/tutorial/notebook_retained_results.ipynb). |

The public CLI does not expose a one-command VTU/PVD export, and `Result.export()` does not generate new artifacts. These notebooks do not promise a ParaView time-series workflow or add new retained results.
