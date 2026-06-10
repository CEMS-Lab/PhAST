# CANONICAL Kalthoff half-plate runs

**This is the authoritative source for the paper's Kalthoff-Winkler benchmark (B2).**

## Geometry

- **100 × 100 mm half-plate**, single notch at y = 25 mm, a = 50 mm
- Symmetry BC: `u_y = 0` at y = 0 (lower edge)
- Impact zone: `x = 0`, `y ∈ [0, 24.875]` mm, prescribed velocity `v_x(t)` ramping to 16.5 m/s
- Material: AT2, spectral split, plane strain. E = 190 GPa, ν = 0.3, ρ = 8000 kg/m³, Gc = 22.13 N/mm, ℓ₀ = 0.195 mm

This matches **paper §3.2 problem-setup paragraph and Table 1 (B2 row) and Table 4 convergence study** exactly.

## Source chain

1. **Mesh generator + config:** `examples/dynamic/kalthoff/timing_comparison/`
   - `make_mesh_half.py` builds the half-plate mesh, parameterised by `h_crack`
   - `config.yaml` defines material (E, ν, ρ, Gc, ℓ₀), AT2/spectral/plane-strain, impact BC, 100 μs total time
   - The mesh file `kalthoff_half_h0.25.msh` is regenerated in place per mesh in the sweep
2. **Driver:** `scripts/hpc_kalthoff_halfplate.slurm` (HPC job 19148, 2026-04-22, single A100)
3. **Outputs:** in `examples/dynamic/kalthoff/reference_runs/B2_kalthoff_mesh{1..5}/` directories
   (gitignored; regenerated per HPC sweep). Each contains `config.yaml`,
   `crack_tip.csv`, `history.csv`, `energy.csv`, `damage_final.png`,
   `run_metadata.json`. Mesh 1 also retains `training_data.h5` locally;
   mesh 2–5 H5 files live only on HPC scratch.

Five meshes:

| Dir | h_crack (mm) | Nodes | n_steps |
|---|---|---|---|
| `B2_kalthoff_mesh1/` | 1.00  | 3,015   | 3,759  |
| `B2_kalthoff_mesh2/` | 0.50  | 9,735   | 4,296  |
| `B2_kalthoff_mesh3/` | 0.25  | 35,487  | 11,775 |
| `B2_kalthoff_mesh4/` | 0.15  | 94,878  | 22,826 |
| `B2_kalthoff_mesh5/` | 0.10  | 213,045 | 31,194 |

Mesh 3 is the **headline run** for Fig 7 (`kalthoff_m3_energy.png`). It ran clean with no AMG warnings.

Note: every `config.yaml` in the five dirs references the same nominal
template path `kalthoff_half_h0.25.msh`. The actual mesh in the run
came from `examples/dynamic/kalthoff/timing_comparison/make_mesh_half.py`
regenerating the `.msh` in place per `h_crack` value before each
launch — confirmed by `run_metadata.json:mesh.n_nodes` differing
across the five dirs (3015 / 9735 / 35487 / 94878 / 213045).

## Authoritative figure mapping (as of 2026-05-09, issue #234)

- Fig 7 `kalthoff_m3_energy.png` ← `B2_kalthoff_mesh3/energy.csv` via `scripts/build_pmma_fig8.py:fig_kalthoff_energy()`
- Table 4 *Initiation* column ← VERIFIED match against
  `B2_kalthoff_mesh{1..5}/crack_tip.csv` (first row with `x > 50` mm
  per the table caption). The numeric initiation-time sequence is intentionally omitted from the public provenance note; regenerate it from `crack_tip.csv` before quoting Table 4.
- Table 4 *Crack-angle* column ← UNVERIFIED for canonical data.
  Mesh 1 H5 extraction (about 62°–68° at 25–75th percentile) does NOT
  match Table 4's 70°–73° entry. Mesh 2–5 H5 files are HPC-only;
  see `papers/paper/audit_claims/PAPER_1_TODOS.md` Section J for the
  follow-up rsync action.

## Known issues

- **Mesh 4 and mesh 5 historical runs emitted pyAMG NaN warnings** during damage initiation (~step 2900–7400). Those old outputs remain questionable because they may have reused a stale AMG hierarchy. The current solver tags this failure mode as `AMG_QS_FALLBACK`, clears stale AMG state, and falls back to Jacobi; rerun mesh 4/5 before using them for headline figures. Mesh 1, 2, 3 are clean.
