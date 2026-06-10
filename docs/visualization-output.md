# Visualisation output: VTU vs `.pv`

`phast` writes per-step visualisation snapshots through
`io_utils.write_visualization(...)`, which dispatches to one of two
backends:

| Backend | Extension | Library | Speed (writeup numbers) | ParaView native? |
|---|---|---|---|---|
| **VTU (default)** | `.vtu`, `.pvd` | `meshio` + VTK XML, zlib compression | baseline | ✅ yes |
| **PyVista-zstd** | `.pv` | PyVista 0.48 reader/writer registry + `pyvista-zstd` | ~43–90× faster on multi-million-cell meshes | ❌ no — read via PyVista |

VTU stays the default because ParaView opens it natively. `.pv` is an
opt-in fast path for users with heavy per-step output (3D extension,
high-snapshot-density runs, dozens of fields per step) who don't mind
running a one-line conversion before ParaView.

## When to use `.pv`

Switch to `.pv` only when **all** of these are true:

- Each snapshot is large (≥ 1M cells, or many fields per step), so the
  90× write speedup is real wall-time
- You don't mind reading via PyVista (Python) instead of ParaView
- You can install the optional `viz-fast` extras

For a typical 235k-node Bleyer-SENT run at `vtu_every=10`, VTU writes
take a fraction of a second per step — the speedup is real but
absolute time saved is small. Stick with VTU until output volume
becomes the bottleneck.

## Install

```bash
pip install -e ".[viz-fast]"
```

This pulls `pyvista>=0.48` and `pyvista-zstd>=0.2`. PyVista has VTK as
a transitive dependency (~50 MB), so it's behind extras to keep core
installs lean.

## Enable

### YAML config

```yaml
output:
  vtu: true
  vtu_every: 10
  viz_format: pv      # default is "vtu"
```

### Python (direct `SolverConfig`)

```python
from phast.staggered_solver import SolverConfig

cfg = SolverConfig(
    solver_type='explicit',
    num_steps=20000,
    dump_every=10,
    viz_format='pv',
)
```

### Failure modes

| What happens | Why | What you see |
|---|---|---|
| You set `viz_format='pv'` but didn't `pip install -e ".[viz-fast]"` | `pyvista` not importable | One-time `RuntimeWarning`; the writer rewrites the path extension `.pv` → `.vtu` and falls back to VTU. Run continues. |
| You call `write_pv(...)` directly with `pyvista` missing | Direct call, no fallback | `ImportError` with the install hint |

## Viewing `.pv` files

`.pv` is **not openable in ParaView directly.** Two routes:

### Route 1 — Read in PyVista (Python, best for interactive 3D)

```python
import pyvista as pv

mesh = pv.read('step_0050.pv')          # one-line load
print(mesh.point_data.keys())            # damage, displacement, H, ...
print(mesh.cell_data.keys())             # psi_plus, H_elem, ...

# Plot interactively
mesh.plot(scalars='damage', show_edges=False, cmap='inferno')
```

PyVista's plotter is a full Qt/Trame window with the VTK pipeline
under the hood — same renderer family ParaView uses, scriptable in
Python.

### Route 2 — Convert to `.vtu` and open in ParaView

```python
import pyvista as pv
mesh = pv.read('step_0050.pv')
mesh.save('step_0050.vtu')               # zlib-compressed VTU
```

Or batch the conversion of an output directory:

```python
import pyvista as pv
from pathlib import Path

for src in sorted(Path('runs/my_run').glob('step_*.pv')):
    pv.read(src).save(src.with_suffix('.vtu'))
```

After conversion, open in ParaView normally. You don't lose
information — `.pv` and `.vtu` carry the same point/cell data.

### Route 3 — Selective-field load (when each snapshot has many fields)

The `pyvista-zstd` reader supports decompressing only the arrays you
ask for. Useful when post-processing scripts touch one or two fields
out of dozens:

```python
import pyvista as pv
mesh = pv.read('step_0050.pv', point_arrays=['damage'])
```

This skips decompression of `displacement`, `H`, etc. — meaningful
speedup when each snapshot carries 30+ fields.

## How does `.pv` relate to VTK?

PyVista is built on top of VTK — the in-memory data structures are
the same `vtkUnstructuredGrid` objects ParaView uses. The `.pv`
format change is **only the on-disk encoding**: zstd-compressed
binary instead of zlib-wrapped XML. The mesh, fields, cell types,
and topology are all VTK-native; round-tripping `.pv` → in-memory
PyVista object → `.vtu` is lossless.

## Repository touch points

| File | Symbol | Purpose |
|---|---|---|
| `io_utils.py` | `write_vtu` | VTU writer (default, ParaView-native, unchanged) |
| `io_utils.py` | `write_pv` | `.pv` writer (lazy `pyvista` import, raises `ImportError` if missing) |
| `io_utils.py` | `write_visualization(..., format=...)` | Dispatcher with fail-soft fallback to VTU |
| `config.py` | `OutputConfig.viz_format` | YAML config hook (`'vtu'` or `'pv'`) |
| `staggered_solver.py` | `SolverConfig.viz_format` | Threaded into the per-step dump call site |
| `pyproject.toml` | `viz-fast` extras | Pulls `pyvista>=0.48`, `pyvista-zstd>=0.2` |

The PVD time-series collection writer (`write_pvd`) is VTU-only —
it's the ParaView convention. For `.pv`-mode runs, load each
snapshot directly via `pv.read('step_NNNN.pv')` or use PyVista's
`MultiBlock` containers.
