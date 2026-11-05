# Citing PhAST

If PhAST contributes to your research, please cite the journal article
(doi:10.1016/j.cma.2026.119547) and the software repository. The preprint is arXiv:2606.23458; its title
differs from the journal title.

GitHub reads [`CITATION.cff`](../CITATION.cff) and shows a **Cite this
repository** button on the repository sidebar. Use that button for the most
current machine-readable citation metadata.

## Article BibTeX

```bibtex
@article{ani2026phast,
  title={PhAST: A differentiable, matrix-free finite-element solver for dynamic phase-field fracture: Formulation, benchmarks, and inverse analysis},
  author={Ani, Allamaprabhu and Molinari, Jean-François and Subhash, Ghatu and Ponnusami, Sathiskumar A.},
  journal={Computer Methods in Applied Mechanics and Engineering},
  year={2026},
  note={In press, accepted 27 October 2026},
  doi={10.1016/j.cma.2026.119547},
  eprint={2606.23458},
  archivePrefix={arXiv},
  primaryClass={cs.CE}
}
```

## Repository BibTeX

```bibtex
@software{ani_phast_2026,
  author       = {Allamaprabhu Ani and Jean-Francois Molinari and
                  Ghatu Subhash and Sathiskumar A. Ponnusami},
  title        = {{PhAST}: Phase-field Autograd Solver in Torch},
  year         = {2026},
  version      = {0.18.0},
  license      = {MIT},
  url          = {https://github.com/CEMS-Lab/PhAST},
  note         = {Official code repository for doi:10.1016/j.cma.2026.119547.}
}
```

## What To Cite

| Use case | Citation guidance |
|---|---|
| You used the released PhAST software | Cite the repository using `CITATION.cff` or the BibTeX entry above. |
| You used a specific public example | Cite the repository and mention the example folder, configuration file, and commit hash. |
| You used a timing or benchmark comparison | Cite the repository and report the exact command, hardware, PyTorch version, and generated manifests. |
| You are discussing the formulation | Cite the PhAST manuscript, plus the underlying phase-field references listed in [Physics, Units, and Formulation](user_guide/physics.md). |

For reproducibility, include the PhAST commit hash and the path to the
configuration you ran, for example:

```text
PhAST commit: <git-commit>
Configuration: examples/dynamic/B2_kalthoff_winkler/config.yaml
Command: python -m phast run examples/dynamic/B2_kalthoff_winkler/config.yaml --output_dir runs/B2_kalthoff_winkler
```
