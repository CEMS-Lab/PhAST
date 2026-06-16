# Result API

Use `phast.load_result(path)` to inspect an existing run directory without
rerunning a solver:

```python
import phast

result = phast.load_result("runs/notched_plate")
metadata = result.metadata()
manifest = result.manifest()
mesh = result.mesh()
histories = result.history_names()
response = result.history("reaction_force")
visuals = result.visuals()
fields = result.field_names()
has_damage = result.has_field("damage")
damage = result.field("damage", step=-1)
```

The current API is read-only and supports existing output formats:

| Method | Purpose |
|---|---|
| `metadata()` | Return `run_metadata.json` content when present. |
| `manifest()` | Return `run_manifest.json`, falling back to metadata for older runs. |
| `mesh()` | Return mesh metadata/provenance from metadata, manifest config, or trajectory mesh groups. |
| `history_names()` | List CSV-backed histories and supported aliases. |
| `history(name)` | Return rows for a CSV-backed history. |
| `visuals()` | Return visual manifest rows or discovered media artifacts. |
| `field_names()` | Discover canonical stored field names from Zarr/H5 trajectory stores. |
| `has_field(name)` | Check a canonical field name or supported alias. |
| `field(name, step=-1)` | Load a directly stored raw Zarr/H5 field as a NumPy array. |

## Reserved postprocess methods

These method names are public, but they are explicit boundaries rather than
automatic artifact generators:

```python
result.plot("damage")      # clear ResultLoadError until postprocess wiring lands
result.animate("damage")   # clear ResultLoadError until postprocess wiring lands
result.export("vtu")       # clear ResultLoadError until export wiring lands
```

Use `result.visuals()` to inspect artifacts already written by a run, or call
`python -m phast postprocess <run_dir>` explicitly when postprocessing
generation is needed.

| Method | Boundary |
|---|---|
| `plot(field, step=-1)` | Reserved public postprocess method; raises a clear `ResultLoadError` and points to existing `visuals()` / `python -m phast postprocess`. |
| `animate(field)` | Reserved public animation method; raises a clear `ResultLoadError` and points to existing `visuals()` / `python -m phast postprocess`. |
| `export(format)` | Reserved public export method; raises a clear `ResultLoadError` until explicit export wiring lands. |

Canonical history aliases include `response`, `reaction_force`,
`load_displacement`, `max_damage`, `energy`, `solver_telemetry`, and
`timing_per_step` when backed by existing CSV files or columns.

Canonical field aliases include `damage`, `displacement`, `history_field`,
`history_field_nodal`, `stress`, `strain`, `velocity`, and `acceleration` when
stored in an existing trajectory store. Field loading returns NumPy arrays
because Zarr/H5 readers are NumPy-native; training code can use
`torch.as_tensor(result.field("damage"))` when it needs a tensor view.

Derived fields such as von Mises stress or displacement magnitude are not
silently computed by `field()`. They require explicit postprocessing support
and currently raise a clear `ResultLoadError` unless the requested raw dataset
is stored directly.

`Result.plot()`, `Result.animate()`, and `Result.export()` are intentionally
present as stable public method names but do not create new artifacts from a
read-only result yet.
