# Results and visualization

## Result anatomy

A run directory can contain:

| Artifact | Role |
|---|---|
| `config.yaml` | Saved input or resolved configuration. |
| `run_metadata.json` | Run, mesh, device, and provenance metadata when written. |
| `run_manifest.json` | Artifact-oriented manifest; older runs may use metadata. |
| CSV histories | Scalar response, energy, telemetry, or timing rows. |
| Zarr trajectory | Preferred reloadable field snapshots when requested. |
| H5 trajectory | Legacy-compatible trajectory format when requested. |
| PNG/VTU/MP4 and visual manifest | Human-facing visual outputs and their descriptions. |

The exact contents depend on the configuration and execution route. An output request is not a guarantee that an unsupported field or postprocessor will be created.

## Reload numerical state

```python
import phast

result = phast.load_result("runs/notched_plate")
print(result.field_names())
print(result.history_names())
if result.has_field("damage"):
    final_damage = result.field("damage", step=-1)
```

`field()` reads a stored raw field from the trajectory store and returns a NumPy array. It does not derive stress, strain, or other quantities that were not stored. `history()` reads CSV-backed history rows. `metadata()`, `manifest()`, and `mesh()` provide run context and provenance.

## Visualization

Set supported output options such as `plots`, `gif`, or trajectory/VTU output in YAML, or invoke `python -m phast postprocess <run_dir>` for an existing run where the postprocessor has the required source data. `visuals()` lists files already present; it does not generate them.

Visualization is an inspection product, not reloadable state. A plot can show a field without preserving the mesh coordinates, field ordering, configuration, or all time steps needed to reproduce numerical analysis. Conversely, a trajectory can be reloadable without a rendered image.
