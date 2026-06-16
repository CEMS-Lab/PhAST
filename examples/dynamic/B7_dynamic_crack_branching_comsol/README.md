# B7 Dynamic Crack Branching COMSOL Cross-Check

Accepted dynamic crack branching comparison package from HPC job 47961. This
folder includes the runnable YAML, curated PNG/CSV/report metadata, and excludes
the private COMSOL binary model and vendor PDF.

Run:

```bash
python -m phast run examples/dynamic/B7_dynamic_crack_branching_comsol/config.yaml
```

Use `initial_conditions.png`, `thumbnail.png`, `damage_final.png`,
`energy.png`, `compare.png`, and `compare_report.txt` as the public evidence
set. The raw 98 GB Zarr trajectory and internal branching montage remain
HPC/private.
