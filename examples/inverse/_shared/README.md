# Shared Inverse Modules

- `forward.py`: dynamic AT2 forward model of the SENT plate and the fixed-position observation operators.
- `recovery.py`: bounded parameter recovery, gradient checks, source checksums and output writers.
- `plotting.py`: field and optimisation-history plots of a completed run.
- `current_forward.py`, `current_recovery.py` and `current_plotting.py`: the cross-mesh recipe, built on the three modules above and called by `cross_mesh_toughness/run.py` and `plot.py`.
