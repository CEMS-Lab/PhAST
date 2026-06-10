"""
Output utilities: VTU (ParaView), Zarr trajectory stores, CSV (history),
and legacy H5 compatibility helpers.

Includes helpers for GNO (Graph Neural Operator) training data export:
- ``compute_edge_index`` converts triangle connectivity to PyG edge_index.
- ``write_profiler_csv`` exports solver profiling data.
- ``generate_run_tag`` creates informative output directory names.
- ``save_run_metadata`` writes a JSON with full run context.
"""

import torch
import numpy as np
import os
import json
import platform
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict


def _telemetry_value(row, key, index, default=0.0):
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[index]
    except IndexError:
        return default


def write_solver_telemetry_csv(path: str, rows):
    """Write per-step solver convergence telemetry.

    Schema is shared by YAML runs and standalone benchmark drivers:
    step, time/load value, stagger iterations, mechanics PCG iterations,
    phase-field PCG iterations, absolute/relative residuals, and
    load/time increment.
    """
    with open(path, 'w') as fh:
        fh.write("step,time,newton_iters,pcg_iters_mech,pcg_iters_pf,"
                 "residual,relative_residual,mechanics_residual,"
                 "mechanics_relative_residual,dt,line_search_alpha,"
                 "line_search_reductions,continuation_mode,"
                 "arc_length_residual,arc_length_constraint,load_factor\n")
        for row in rows:
            step = int(_telemetry_value(row, 'step', 0))
            load_time = float(_telemetry_value(row, 'time', 1))
            newton_iters = int(_telemetry_value(row, 'newton_iters', 2))
            pcg_mech = int(_telemetry_value(row, 'pcg_iters_mech', 3))
            pcg_pf = int(_telemetry_value(row, 'pcg_iters_pf', 4))
            residual = float(_telemetry_value(row, 'residual', 5, float('nan')))
            relative_residual = float(_telemetry_value(
                row, 'relative_residual', 6, float('nan')))
            mechanics_residual = float(_telemetry_value(
                row, 'mechanics_residual', 7, float('nan')))
            mechanics_relative_residual = float(_telemetry_value(
                row, 'mechanics_relative_residual', 8, float('nan')))
            dt = float(_telemetry_value(row, 'dt', 9))
            ls_alpha = float(_telemetry_value(
                row, 'line_search_alpha', 10, 1.0))
            ls_reductions = int(_telemetry_value(
                row, 'line_search_reductions', 11, 0))
            cont_mode = str(_telemetry_value(
                row, 'continuation_mode', 12, ''))
            arc_res = float(_telemetry_value(
                row, 'arc_length_residual', 13, float('nan')))
            arc_constraint = float(_telemetry_value(
                row, 'arc_length_constraint', 14, float('nan')))
            load_factor = float(_telemetry_value(
                row, 'load_factor', 15, load_time))
            fh.write(f"{step},{load_time:.9e},{newton_iters},{pcg_mech},"
                     f"{pcg_pf},{residual:.9e},{relative_residual:.9e},"
                     f"{mechanics_residual:.9e},"
                     f"{mechanics_relative_residual:.9e},{dt:.9e},"
                     f"{ls_alpha:.9e},{ls_reductions},{cont_mode},"
                     f"{arc_res:.9e},{arc_constraint:.9e},"
                     f"{load_factor:.9e}\n")


def write_energy_csv(path: str, rows):
    """Write per-step energy history with a shared forward/QS schema."""
    with open(path, 'w') as fh:
        fh.write("step,time,elastic,fracture,kinetic,external,total\n")
        for row in rows:
            step = int(row.get('step', 0))
            time_value = float(row.get('time', row.get('disp', 0.0)))
            elastic = float(row.get('elastic', 0.0))
            fracture = float(row.get('fracture', 0.0))
            kinetic = float(row.get('kinetic', 0.0))
            external = float(row.get('external', 0.0))
            total = float(row.get('total', elastic + fracture + kinetic - external))
            fh.write(
                f"{step},{time_value:.9e},{elastic:.9e},{fracture:.9e},"
                f"{kinetic:.9e},{external:.9e},{total:.9e}\n"
            )


def plot_energy_history(rows, path: str, xlabel: str = "Step") -> None:
    """Render elastic/fracture/kinetic/total energy history to a PNG."""
    if not rows:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = [float(row.get('time', row.get('step', i))) for i, row in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [float(r.get('elastic', 0.0)) for r in rows],
            label='Elastic', lw=1.8)
    ax.plot(x, [float(r.get('fracture', 0.0)) for r in rows],
            label='Fracture', lw=1.8)
    if any(abs(float(r.get('kinetic', 0.0))) > 0.0 for r in rows):
        ax.plot(x, [float(r.get('kinetic', 0.0)) for r in rows],
                label='Kinetic', lw=1.5)
    ax.plot(x, [float(r.get('total', 0.0)) for r in rows],
            label='Total', lw=2.0, ls='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Energy [N mm]')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_edge_index(elements):
    """Compute PyG-style edge_index from triangle connectivity.

    For each triangle (i, j, k), creates bidirectional edges:
    (i,j), (j,i), (j,k), (k,j), (i,k), (k,i).
    Duplicates from shared edges are removed and the result is sorted.

    Parameters
    ----------
    elements : (E, 3) array-like
        Triangle node indices. Accepts numpy array or torch tensor.

    Returns
    -------
    edge_index : (2, num_edges) numpy array, dtype int64
        Bidirectional, duplicate-free, sorted by (src, dst).
    """
    if isinstance(elements, torch.Tensor):
        elems = elements.cpu().numpy()
    else:
        elems = np.asarray(elements)

    # Extract the three vertex columns
    i, j, k = elems[:, 0], elems[:, 1], elems[:, 2]

    # Build all six directed edges per triangle
    src = np.concatenate([i, j, j, k, i, k])
    dst = np.concatenate([j, i, k, j, k, i])

    # Stack, remove duplicates, sort
    edges = np.stack([src, dst], axis=0).T          # (6E, 2)
    edges = np.unique(edges, axis=0)                # sorted, no dups
    edge_index = edges.T.astype(np.int64)           # (2, num_edges)
    return edge_index


def write_vtu(path: str, mesh, point_data: Optional[Dict] = None,
              cell_data: Optional[Dict] = None):
    """Write a VTU file for ParaView visualization.

    Parameters
    ----------
    path : str
        Output .vtu file path.
    mesh : FEMMesh
    point_data : dict of str -> tensor (N,...) — nodal fields
    cell_data : dict of str -> tensor (E,...) — element fields
    """
    import meshio

    pts = mesh.nodes.cpu().numpy()
    pts3d = np.column_stack([pts, np.zeros(len(pts))])
    cells = [("triangle", mesh.elements.cpu().numpy())]

    pd = {}
    if point_data:
        for k, v in point_data.items():
            a = v.detach().cpu().numpy()
            if a.ndim == 2 and a.shape[1] == 2:
                a = np.column_stack([a, np.zeros(len(a))])
            pd[k] = a

    cd = {}
    if cell_data:
        for k, v in cell_data.items():
            a = v.detach().cpu().numpy()
            cd[k] = [a]

    m = meshio.Mesh(pts3d, cells, point_data=pd, cell_data=cd)
    # Use binary XML with zlib compression for 60-70% smaller files
    m.write(path, binary=True, compression="zlib")


def write_pv(path: str, mesh, point_data: Optional[Dict] = None,
             cell_data: Optional[Dict] = None):
    """Write a multi-threaded zstd-compressed mesh file (.pv format).

    Same signature as ``write_vtu``. The ``.pv`` extension is dispatched
    by PyVista's reader/writer registry (PyVista >= 0.48) to the
    pyvista-zstd backend, which reports up to 43-90x faster writes than
    VTK XML on multi-million-cell meshes.

    Requires the optional viz-fast dependencies::

        pip install phast[viz-fast]

    For ParaView native compatibility, prefer ``write_vtu``: ParaView
    cannot open ``.pv`` directly (read via PyVista or convert).
    """
    try:
        import pyvista as pv  # noqa: F401  (pulled for the registry side-effect)
    except ImportError as e:
        raise ImportError(
            "write_pv requires PyVista >= 0.48 and pyvista-zstd. "
            "Install via: pip install phast[viz-fast]"
        ) from e

    pts = mesh.nodes.cpu().numpy()
    pts3d = np.column_stack([pts, np.zeros(len(pts))])
    elems = mesh.elements.cpu().numpy()
    n_cells = len(elems)

    # PyVista flat cell array: [3, n0, n1, n2, 3, n0, n1, n2, ...]
    cells = np.column_stack(
        [np.full(n_cells, 3, dtype=np.int64), elems]
    ).ravel()
    cell_types = np.full(n_cells, pv.CellType.TRIANGLE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, pts3d)

    if point_data:
        for k, v in point_data.items():
            a = v.detach().cpu().numpy()
            if a.ndim == 2 and a.shape[1] == 2:
                a = np.column_stack([a, np.zeros(len(a))])
            grid.point_data[k] = a
    if cell_data:
        for k, v in cell_data.items():
            grid.cell_data[k] = v.detach().cpu().numpy()

    grid.save(path)


def write_visualization(path: str, mesh, point_data: Optional[Dict] = None,
                        cell_data: Optional[Dict] = None,
                        format: str = 'vtu'):
    """Dispatch to ``write_vtu`` (default) or ``write_pv``.

    ``format='pv'`` falls back to ``vtu`` with a one-time warning if
    pyvista-zstd is unavailable.
    """
    if format == 'vtu':
        return write_vtu(path, mesh, point_data, cell_data)
    if format == 'pv':
        try:
            return write_pv(path, mesh, point_data, cell_data)
        except ImportError:
            if not getattr(write_visualization, '_pv_warned', False):
                import warnings
                warnings.warn(
                    "format='pv' requested but pyvista-zstd is not "
                    "installed; falling back to VTU. "
                    "pip install phast[viz-fast] to enable.",
                    RuntimeWarning, stacklevel=2,
                )
                write_visualization._pv_warned = True
            # Swap extension if caller passed .pv
            if path.endswith('.pv'):
                path = path[:-3] + '.vtu'
            return write_vtu(path, mesh, point_data, cell_data)
    raise ValueError(f"Unknown viz format {format!r}; expected 'vtu' or 'pv'.")


def write_pvd(pvd_path: str, vtu_files: list, times: list = None):
    """Write a PVD collection file for ParaView time-series loading.

    ParaView opens the .pvd and loads the full time series with one click.

    Parameters
    ----------
    pvd_path : str — output .pvd file path
    vtu_files : list of str — paths to .vtu files (relative to pvd_path)
    times : list of float or None — simulation times (uses step index if None)
    """
    import xml.etree.ElementTree as ET
    root = ET.Element('VTKFile', type='Collection', version='0.1')
    coll = ET.SubElement(root, 'Collection')
    pvd_dir = os.path.dirname(pvd_path) or '.'
    for i, vtu in enumerate(vtu_files):
        t = times[i] if times is not None else float(i)
        rel = os.path.relpath(vtu, pvd_dir)
        ET.SubElement(coll, 'DataSet', timestep=f'{t:.6e}', file=rel)
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(pvd_path, xml_declaration=True, encoding='utf-8')


def _zarr_write_array(group, name: str, data):
    """Write/replace one array with zarr-v2/v3-compatible calls."""
    arr = np.asarray(data)
    if name in group:
        del group[name]
    try:
        return group.create_array(name, data=arr)
    except AttributeError:
        return group.create_dataset(name, data=arr, shape=arr.shape,
                                    dtype=arr.dtype)
    except TypeError:
        return group.create_dataset(name, data=arr, shape=arr.shape,
                                    dtype=arr.dtype)


def _zarr_create_resizable_array(group, name: str, sample):
    """Create a step-major resizable array for fast trajectory scans."""
    arr = np.asarray(sample)
    shape = (0,) + arr.shape
    chunks = (1,) + arr.shape
    if name in group:
        return group[name]
    try:
        return group.create_array(
            name, shape=shape, chunks=chunks, dtype=arr.dtype)
    except AttributeError:
        return group.create_dataset(
            name, shape=shape, chunks=chunks, dtype=arr.dtype)
    except TypeError:
        return group.create_dataset(
            name, shape=shape, chunks=chunks, dtype=arr.dtype)


def _zarr_append_dense_snapshot(zarr_root, step: int, arrays: dict,
                                attrs: dict) -> None:
    """Append a columnar snapshot alongside legacy step-group output.

    The legacy ``simulation_data/steps/step_####`` hierarchy is kept for
    backward compatibility. This dense layout makes common readers scan one
    array per field instead of thousands of tiny groups.
    """
    sim = _zarr_require_group(zarr_root, 'simulation_data')
    traj = _zarr_require_group(sim, 'trajectory')
    count = int(traj.attrs.get('count', 0))
    step_index = count
    if 'step' not in traj:
        try:
            traj.create_array('step', shape=(0,), chunks=(1024,),
                              dtype=np.dtype('int64'))
            traj.create_array('time_s', shape=(0,), chunks=(1024,),
                              dtype=np.dtype('float64'))
            traj.create_array('applied_disp', shape=(0,), chunks=(1024,),
                              dtype=np.dtype('float64'))
            traj.create_array('reaction_force', shape=(0,), chunks=(1024,),
                              dtype=np.dtype('float64'))
        except AttributeError:
            traj.create_dataset('step', shape=(0,), chunks=(1024,),
                                dtype=np.dtype('int64'))
            traj.create_dataset('time_s', shape=(0,), chunks=(1024,),
                                dtype=np.dtype('float64'))
            traj.create_dataset('applied_disp', shape=(0,), chunks=(1024,),
                                dtype=np.dtype('float64'))
            traj.create_dataset('reaction_force', shape=(0,), chunks=(1024,),
                                dtype=np.dtype('float64'))
        except TypeError:
            traj.create_dataset('step', shape=(0,), chunks=(1024,),
                                dtype=np.dtype('int64'))
            traj.create_dataset('time_s', shape=(0,), chunks=(1024,),
                                dtype=np.dtype('float64'))
            traj.create_dataset('applied_disp', shape=(0,), chunks=(1024,),
                                dtype=np.dtype('float64'))
            traj.create_dataset('reaction_force', shape=(0,), chunks=(1024,),
                                dtype=np.dtype('float64'))

    if count > 0:
        existing_steps = np.asarray(traj['step'][:count])
        matches = np.where(existing_steps == int(step))[0]
        if matches.size:
            step_index = int(matches[-1])

    for key, arr in arrays.items():
        ds = _zarr_create_resizable_array(traj, key, arr)
        if step_index >= ds.shape[0]:
            ds.resize((step_index + 1,) + ds.shape[1:])
        ds[step_index] = arr

    for key in ('step', 'time_s', 'applied_disp', 'reaction_force'):
        ds = traj[key]
        if step_index >= ds.shape[0]:
            ds.resize((step_index + 1,))
    traj['step'][step_index] = int(step)
    def _attr_float(name):
        value = attrs.get(name, np.nan)
        return np.nan if value is None else float(value)
    traj['time_s'][step_index] = _attr_float('time_s')
    traj['applied_disp'][step_index] = _attr_float('applied_disp')
    traj['reaction_force'][step_index] = _attr_float('reaction_force')
    traj.attrs['count'] = max(count, step_index + 1)
    traj.attrs['layout'] = 'dense_step_major_v1'


def _zarr_require_group(group, name: str):
    if name in group:
        return group[name]
    try:
        return group.require_group(name)
    except AttributeError:
        return group.create_group(name)


def _snapshot_arrays(mesh, u, d, psi_plus_e, H_e, eps_xx=None, eps_yy=None,
                     gam_xy=None, sxx=None, syy=None, sxy=None,
                     H_nodal=None, velocity=None, acceleration=None,
                     precision='float32') -> dict:
    dtype = np.dtype(precision)
    arrays = {
        'damage_nodal': d.detach().cpu().numpy().astype(dtype, copy=False),
        'displacement': u.detach().cpu().numpy().astype(dtype, copy=False),
        'psi_plus': psi_plus_e.detach().cpu().numpy().astype(dtype, copy=False),
        'H_elem': H_e.detach().cpu().numpy().astype(dtype, copy=False),
    }
    if eps_xx is not None and eps_yy is not None and gam_xy is not None:
        strain = torch.stack([eps_xx, eps_yy, gam_xy], dim=1)
        arrays['strain'] = strain.detach().cpu().numpy().astype(dtype, copy=False)
    if sxx is not None and syy is not None and sxy is not None:
        stress = torch.stack([sxx, syy, sxy], dim=1)
        arrays['stress'] = stress.detach().cpu().numpy().astype(dtype, copy=False)
    if H_nodal is not None:
        arrays['H_nodal'] = H_nodal.detach().cpu().numpy().astype(dtype, copy=False)
    if velocity is not None:
        arrays['velocity'] = velocity.detach().cpu().numpy().astype(dtype, copy=False)
    if acceleration is not None:
        arrays['acceleration'] = acceleration.detach().cpu().numpy().astype(dtype, copy=False)
    return arrays


def _write_snapshot_attrs(group, mesh, d, reaction_force=None, energies=None,
                          applied_disp=None, time_s=None) -> None:
    if reaction_force is not None:
        group.attrs['reaction_force'] = float(reaction_force)
    if applied_disp is not None:
        group.attrs['applied_disp'] = float(applied_disp)
    if energies is not None:
        for k, v in energies.items():
            group.attrs[f'energy_{k}'] = float(v)
    if time_s is not None:
        group.attrs['time_s'] = float(time_s)

    try:
        from ..core.fem_operators import compute_damage_scalars
        dmg = compute_damage_scalars(d, mesh)
        for k, v in dmg.items():
            group.attrs[k] = float(v)
    except Exception:
        # Postprocessors must never block trajectory output.
        pass


def init_zarr(zarr_path: str, mesh, material):
    """Create and initialize a Zarr trajectory store.

    The group/dataset hierarchy mirrors legacy ``training_data.h5`` so
    existing field contracts stay stable while new runs use a chunked,
    directory-backed store.
    """
    import zarr

    path = Path(zarr_path)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    root = zarr.open_group(str(path), mode='w')
    sim = _zarr_require_group(root, 'simulation_data')
    mesh_grp = _zarr_require_group(sim, 'mesh')
    _zarr_write_array(mesh_grp, 'node_coordinates',
                      mesh.nodes.detach().cpu().numpy())
    _zarr_write_array(mesh_grp, 'element_connectivity',
                      mesh.elements.detach().cpu().numpy())
    _zarr_write_array(mesh_grp, 'edge_index', compute_edge_index(mesh.elements))

    mesh_grp.attrs['n_nodes'] = int(mesh.nodes.shape[0])
    mesh_grp.attrs['n_elements'] = int(mesh.elements.shape[0])

    node_sets = getattr(mesh, 'node_sets', None) or {}
    if node_sets:
        ns_grp = _zarr_require_group(mesh_grp, 'node_sets')
        for name, idx in node_sets.items():
            try:
                arr = idx.cpu().numpy() if hasattr(idx, 'cpu') else np.asarray(idx)
                _zarr_write_array(ns_grp, name, arr.astype('int64'))
            except Exception:
                pass

    meta = _zarr_require_group(sim, 'metadata')
    meta.attrs['Gc'] = material.Gc
    meta.attrs['l0'] = material.l0
    meta.attrs['E'] = material.E
    meta.attrs['nu'] = material.nu
    meta.attrs['rho'] = material.rho
    meta.attrs['energy_split'] = material.energy_split
    meta.attrs['pf_model'] = material.pf_model
    meta.attrs['plane_stress'] = bool(material.plane_stress)
    root.attrs['format'] = 'phast.trajectory.zarr'
    root.attrs['writer'] = 'phast.io_utils.init_zarr'
    root.attrs['layouts'] = 'legacy_step_groups,dense_step_major_v1'
    return root


def write_zarr_snapshot(zarr_root, step: int, mesh, u, d, psi_plus_e, H_e,
                        eps_xx=None, eps_yy=None, gam_xy=None,
                        sxx=None, syy=None, sxy=None,
                        H_nodal=None, reaction_force=None,
                        energies=None, applied_disp=None,
                        velocity=None, acceleration=None, time_s=None,
                        precision='float32'):
    """Write one timestep snapshot to a Zarr trajectory store."""
    sim = _zarr_require_group(zarr_root, 'simulation_data')
    steps = _zarr_require_group(sim, 'steps')
    name = f'step_{step:04d}'
    if name in steps:
        del steps[name]
    grp = steps.create_group(name)
    arrays = _snapshot_arrays(
        mesh, u, d, psi_plus_e, H_e,
        eps_xx=eps_xx, eps_yy=eps_yy, gam_xy=gam_xy,
        sxx=sxx, syy=syy, sxy=sxy, H_nodal=H_nodal,
        velocity=velocity, acceleration=acceleration,
        precision=precision)
    for key, arr in arrays.items():
        _zarr_write_array(grp, key, arr)
    _write_snapshot_attrs(
        grp, mesh, d, reaction_force=reaction_force, energies=energies,
        applied_disp=applied_disp, time_s=time_s)
    _zarr_append_dense_snapshot(
        zarr_root, step, arrays,
        {
            'reaction_force': reaction_force,
            'applied_disp': applied_disp,
            'time_s': time_s,
        })


def write_h5_snapshot(h5f, step: int, mesh, u, d, psi_plus_e, H_e,
                      eps_xx=None, eps_yy=None, gam_xy=None,
                      sxx=None, syy=None, sxy=None,
                      H_nodal=None, reaction_force=None,
                      energies=None, applied_disp=None,
                      velocity=None, acceleration=None, time_s=None,
                      precision='float32'):
    """Write one timestep snapshot to an open H5 file.

    Parameters
    ----------
    h5f : h5py.File (open for writing)
    step : int
    mesh, u, d, psi_plus_e, H_e : solver state tensors
    H_nodal : (N,) tensor, optional
    reaction_force : float, optional
    energies : dict, optional
    applied_disp : float, optional
    velocity : (N, 2) tensor, optional — nodal velocity (dynamic only)
    acceleration : (N, 2) tensor, optional — nodal acceleration (dynamic only)
    time_s : float, optional — simulation time in seconds
    precision : str
        'float32' (default, ~10× smaller with gzip) or 'float64' (lossless).
    """
    grp = h5f.create_group(f'simulation_data/steps/step_{step:04d}')

    # float32 + gzip: NNs train in float32 anyway, damage fields (mostly zeros)
    # compress 70-80%. Use precision='float64' for lossless storage.
    _kw = dict(dtype=precision, compression='gzip', compression_opts=4)

    for key, arr in _snapshot_arrays(
            mesh, u, d, psi_plus_e, H_e,
            eps_xx=eps_xx, eps_yy=eps_yy, gam_xy=gam_xy,
            sxx=sxx, syy=syy, sxy=sxy, H_nodal=H_nodal,
            velocity=velocity, acceleration=acceleration,
            precision=precision).items():
        grp.create_dataset(key, data=arr, **_kw)
    _write_snapshot_attrs(
        grp, mesh, d, reaction_force=reaction_force, energies=energies,
        applied_disp=applied_disp, time_s=time_s)


def load_state_from_h5(h5_path: str) -> Dict:
    """Load the latest per-step snapshot from a training_data.h5 file.

    Walks ``simulation_data/steps/`` and returns the state stored under the
    group with the highest step index. Used to resume a preempted HPC run
    via the ``--restart-from`` CLI flag (AsFem-style restart).

    Returns
    -------
    dict with keys
        ``u``, ``v``, ``a``       : nodal displacement / velocity / accel
        ``d``                     : nodal damage
        ``H_elem``, ``H_nodal``   : history variable (element + nodal)
        ``H``                     : alias of ``H_elem`` (caller-friendly)
        ``step``                  : integer step index of the snapshot
        ``time_s``                : simulation time at that step (or None)

    Velocity / acceleration are returned as zero-filled arrays if the
    snapshot was written from a quasi-static run that did not record them.
    """
    import h5py
    with h5py.File(h5_path, 'r') as h5f:
        if 'simulation_data/steps' not in h5f:
            raise RuntimeError(
                f"H5 file {h5_path} has no 'simulation_data/steps' group; "
                "cannot restart from it.")
        steps_grp = h5f['simulation_data/steps']
        # Snapshot keys are written as ``step_{N:04d}`` (see
        # ``write_h5_snapshot``); newer/longer runs may extend the width
        # past 4 digits, so do not assume a fixed pad. Just take the
        # numerically largest suffix.
        step_keys = [k for k in steps_grp.keys() if k.startswith('step_')]
        if not step_keys:
            raise RuntimeError(
                f"H5 file {h5_path} contains no step_* groups.")
        step_keys.sort(key=lambda k: int(k.split('_')[-1]))
        latest_key = step_keys[-1]
        latest = steps_grp[latest_key]
        step_idx = int(latest_key.split('_')[-1])

        def _get(name):
            if name not in latest:
                return None
            return torch.from_numpy(latest[name][...])

        u = _get('displacement')
        d = _get('damage_nodal')
        H_elem = _get('H_elem')
        H_nodal = _get('H_nodal')
        v = _get('velocity')
        a = _get('acceleration')
        if u is None or d is None or H_elem is None:
            raise RuntimeError(
                f"Snapshot {latest_key} is missing required fields "
                "(displacement / damage_nodal / H_elem).")
        if v is None:
            v = torch.zeros_like(u)
        if a is None:
            a = torch.zeros_like(u)
        time_s = float(latest.attrs['time_s']) if 'time_s' in latest.attrs else None

    return {
        'u': u,
        'v': v,
        'a': a,
        'd': d,
        'H_elem': H_elem,
        'H_nodal': H_nodal,
        'H': H_elem,            # convenience alias requested by spec
        'step': step_idx,
        'time_s': time_s,
    }


def load_state_from_zarr(zarr_path: str, step: Optional[int] = None) -> Dict:
    """Load a per-step snapshot from a ``training_data.zarr`` store.

    Parameters
    ----------
    zarr_path:
        Path to the Zarr trajectory store.
    step:
        Optional solver step index to load. When omitted, the latest stored
        snapshot is returned. The dense ``simulation_data/trajectory`` layout
        is preferred and legacy ``simulation_data/steps/step_####`` groups are
        used as a fallback.
    """
    import zarr

    root = zarr.open(str(zarr_path), mode='r')
    sim = root['simulation_data']
    step_idx = None
    time_s = None

    if 'trajectory' in sim and 'step' in sim['trajectory']:
        traj = sim['trajectory']
        count = int(traj.attrs.get('count', len(traj['step'])))
        if count <= 0:
            raise RuntimeError(f"Zarr store {zarr_path} contains no snapshots.")
        steps = np.asarray(traj['step'][:count], dtype=np.int64)
        if step is None:
            i = count - 1
        else:
            matches = np.where(steps == int(step))[0]
            if matches.size == 0:
                raise KeyError(
                    f"Zarr store {zarr_path} has no dense snapshot for "
                    f"step {int(step)}.")
            i = int(matches[-1])
        step_idx = int(np.asarray(traj['step'][i]))

        def _get_dense(name):
            if name not in traj:
                return None
            return torch.from_numpy(np.asarray(traj[name][i]))

        u = _get_dense('displacement')
        d = _get_dense('damage_nodal')
        H_elem = _get_dense('H_elem')
        H_nodal = _get_dense('H_nodal')
        v = _get_dense('velocity')
        a = _get_dense('acceleration')
        if 'time_s' in traj:
            t_val = float(np.asarray(traj['time_s'][i]))
            time_s = None if np.isnan(t_val) else t_val
    else:
        if 'steps' not in sim:
            raise RuntimeError(
                f"Zarr store {zarr_path} has no trajectory or steps group.")
        steps_grp = sim['steps']
        step_keys = [k for k in steps_grp.keys() if k.startswith('step_')]
        if not step_keys:
            raise RuntimeError(f"Zarr store {zarr_path} contains no step_* groups.")
        step_keys.sort(key=lambda k: int(k.split('_')[-1]))
        if step is None:
            latest_key = step_keys[-1]
        else:
            latest_key = f"step_{int(step):04d}"
            if latest_key not in steps_grp:
                # Historical stores may have wider step padding; match by
                # numeric suffix rather than assuming a fixed string width.
                matches = [
                    k for k in step_keys if int(k.split('_')[-1]) == int(step)
                ]
                if not matches:
                    raise KeyError(
                        f"Zarr store {zarr_path} has no step group for "
                        f"step {int(step)}.")
                latest_key = matches[-1]
        latest = steps_grp[latest_key]
        step_idx = int(latest_key.split('_')[-1])

        def _get_step_array(name):
            if name not in latest:
                return None
            return torch.from_numpy(np.asarray(latest[name]))

        u = _get_step_array('displacement')
        d = _get_step_array('damage_nodal')
        H_elem = _get_step_array('H_elem')
        H_nodal = _get_step_array('H_nodal')
        v = _get_step_array('velocity')
        a = _get_step_array('acceleration')
        time_s = float(latest.attrs['time_s']) if 'time_s' in latest.attrs else None

    if u is None or d is None or H_elem is None:
        raise RuntimeError(
            "Latest Zarr snapshot is missing required fields "
            "(displacement / damage_nodal / H_elem).")
    if v is None:
        v = torch.zeros_like(u)
    if a is None:
        a = torch.zeros_like(u)
    return {
        'u': u,
        'v': v,
        'a': a,
        'd': d,
        'H_elem': H_elem,
        'H_nodal': H_nodal,
        'H': H_elem,
        'step': step_idx,
        'time_s': time_s,
    }


def init_h5(h5_path: str, mesh, material):
    """Create and initialize an H5 file with mesh and metadata.

    Stores node coordinates, element connectivity, PyG-compatible
    ``edge_index``, and mesh size attributes (``n_nodes``, ``n_elements``).

    Returns
    -------
    h5f : h5py.File (open, caller must close)
    """
    import h5py
    h5f = h5py.File(h5_path, 'w')
    try:
        mesh_grp = h5f.create_group('simulation_data/mesh')
        mesh_grp.create_dataset('node_coordinates',
                                data=mesh.nodes.cpu().numpy())
        mesh_grp.create_dataset('element_connectivity',
                                data=mesh.elements.cpu().numpy())

        # PyG-compatible edge_index for GNO training
        edge_index = compute_edge_index(mesh.elements)
        mesh_grp.create_dataset('edge_index', data=edge_index)

        # Store mesh size as attributes for quick access
        mesh_grp.attrs['n_nodes'] = int(mesh.nodes.shape[0])
        mesh_grp.attrs['n_elements'] = int(mesh.elements.shape[0])

        # Persist nodeset memberships so post-processing can identify
        # named regions (e.g. ``notch_upper``) in the H5 alone, without
        # re-loading the mesh. Each nodeset is stored as an int64
        # dataset of node indices under ``simulation_data/mesh/node_sets``.
        # Issue #213: the B7 initiation detector needs the preseed
        # notch nodes to exclude them from max(d).
        node_sets = getattr(mesh, 'node_sets', None) or {}
        if node_sets:
            ns_grp = mesh_grp.create_group('node_sets')
            for name, idx in node_sets.items():
                try:
                    arr = idx.cpu().numpy() if hasattr(idx, 'cpu') else \
                        np.asarray(idx)
                    ns_grp.create_dataset(name, data=arr.astype('int64'))
                except Exception:
                    # Best-effort: skip nodesets that fail to serialise.
                    pass

        meta = h5f.create_group('simulation_data/metadata')
        meta.attrs['Gc'] = material.Gc
        meta.attrs['l0'] = material.l0
        meta.attrs['E'] = material.E
        meta.attrs['nu'] = material.nu
        meta.attrs['rho'] = material.rho
        meta.attrs['energy_split'] = material.energy_split
        meta.attrs['pf_model'] = material.pf_model
        meta.attrs['plane_stress'] = bool(material.plane_stress)
        return h5f
    except Exception:
        h5f.close()
        raise


class CSVHistory:
    """CSV writer for simulation history (load-displacement, max damage, etc.)."""

    def __init__(self, path: str):
        import csv
        self._file = open(path, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            'step', 'max_H_nodal', 'max_psi_plus', 'max_damage',
            'delta_H', 'delta_damage', 'reaction_force', 'applied_disp',
        ])

    def write_row(self, step, max_H, max_psi, max_d, delta_H, delta_d,
                  reaction_force=None, applied_disp=None):
        """Write one history row.

        Parameters
        ----------
        step, max_H, max_psi, max_d, delta_H, delta_d : existing fields
        reaction_force : float, optional
            Reaction force at this step (for load-displacement curves).
        applied_disp : float, optional
            Applied displacement at this step.
        """
        rf_str = f'{reaction_force:.6f}' if reaction_force is not None else ''
        ad_str = f'{applied_disp:.8f}' if applied_disp is not None else ''
        self._writer.writerow([
            step, f'{max_H:.4f}', f'{max_psi:.4f}', f'{max_d:.6f}',
            f'{delta_H:.4f}', f'{delta_d:.6f}', rf_str, ad_str,
        ])
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_profiler_csv(path: str, profiler):
    """Export profiler timing data to CSV.

    Parameters
    ----------
    path : str
        Output CSV file path.
    profiler : Profiler
        A ``phast.device.Profiler`` instance (or any object
        with a ``_timings`` dict mapping ``name -> [total_time, count]``).

    The CSV columns are::

        region,total_seconds,calls,avg_ms,percent
    """
    import csv

    timings = profiler._timings
    if not timings:
        # Write header-only file so downstream readers don't break
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'region', 'total_seconds', 'calls', 'avg_ms', 'percent',
            ])
        return

    total = sum(v[0] for v in timings.values())

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'region', 'total_seconds', 'calls', 'avg_ms', 'percent',
        ])
        for name, (t, n) in sorted(timings.items(), key=lambda x: -x[1][0]):
            avg_ms = (t / n * 1000) if n > 0 else 0.0
            pct = (t / total * 100) if total > 0 else 0.0
            writer.writerow([
                name, f'{t:.6f}', n, f'{avg_ms:.4f}', f'{pct:.2f}',
            ])


def _platform_tag() -> str:
    """Short platform tag: 'mac', 'linux', or 'win'."""
    s = platform.system().lower()
    if s == 'darwin':
        return 'mac'
    if s == 'windows':
        return 'win'
    return s  # 'linux'


def _device_tag(device: torch.device) -> str:
    """Short device tag: 'cpu', 'mps', 'cuda0', 'cuda1', etc."""
    if device.type == 'cuda':
        idx = device.index if device.index is not None else 0
        return f'cuda{idx}'
    return device.type  # 'mps' or 'cpu'


def generate_run_tag(device: torch.device,
                     prefix: str = 'run',
                     extra: str = '') -> str:
    """Generate an informative output directory name.

    Format: ``{prefix}_{platform}_{device}_{YYYYMMDD}_{HHMMSS}``

    Examples::

        run_mac_mps_20260313_143022
        run_linux_cuda0_20260313_143022
        run_glass_mac_cpu_20260313_143022

    Parameters
    ----------
    device : torch.device
    prefix : str
        Leading tag (e.g. 'run', 'validation').
    extra : str
        Optional extra tag inserted after prefix (e.g. 'glass', 'tension').
    """
    parts = [prefix]
    if extra:
        parts.append(extra)
    parts.append(_platform_tag())
    parts.append(_device_tag(device))
    parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
    return '_'.join(parts)


def get_peak_memory_stats(device: Optional[torch.device] = None) -> Dict:
    """Return peak memory usage observed so far.

    Fields (all in bytes, converted to MiB in the returned dict):

        cpu_rss_peak_MiB         — process resident set size peak
                                    (``resource.RUSAGE_SELF.ru_maxrss``).
                                    On macOS this is in bytes; on Linux
                                    it is in kilobytes — we normalise.
        gpu_allocated_peak_MiB   — ``torch.cuda.max_memory_allocated``
                                    at the moment of the call.
        gpu_reserved_peak_MiB    — ``torch.cuda.max_memory_reserved``.

    GPU fields are absent when CUDA is unavailable or ``device`` is not
    a cuda device.  Meant to be called at the END of a run (after all
    significant work) so the "peak" reflects the full simulation.
    """
    out: Dict = {}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            rss_MiB = rss / (1024 * 1024)       # bytes on macOS
        else:
            rss_MiB = rss / 1024                # KiB on Linux
        out['cpu_rss_peak_MiB'] = round(rss_MiB, 1)
    except Exception:
        pass

    if torch.cuda.is_available() and device is not None \
            and getattr(device, 'type', None) == 'cuda':
        try:
            alloc = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            out['gpu_allocated_peak_MiB'] = round(alloc / (1024 ** 2), 1)
            out['gpu_reserved_peak_MiB'] = round(reserved / (1024 ** 2), 1)
        except Exception:
            pass
    return out


def reset_peak_memory_stats(device: Optional[torch.device] = None) -> None:
    """Reset CUDA peak-memory counters so the next recording captures
    only the work that follows this call. CPU RSS cannot be reset."""
    if torch.cuda.is_available() and device is not None \
            and getattr(device, 'type', None) == 'cuda':
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass


def save_run_metadata(output_dir: str, *,
                      problem_name: str = '',
                      device: Optional[torch.device] = None,
                      material=None,
                      mesh=None,
                      solver_config: Optional[Dict] = None,
                      extra: Optional[Dict] = None,
                      include_memory: bool = True):
    """Write a run_metadata.json with full context for reproducibility.

    Saved fields: platform, device, PyTorch version, Python version,
    material parameters, mesh statistics, solver configuration,
    peak CPU/GPU memory observed during the run (when
    ``include_memory=True``, which is the default), and any
    user-supplied extras.
    """
    meta = {
        'timestamp': datetime.now().isoformat(),
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python': sys.version.split()[0],
            'pytorch': torch.__version__,
        },
        'problem': problem_name,
    }

    if device is not None:
        dev_info = {'device': str(device), 'dtype_default': 'float64'}
        if device.type == 'cuda':
            props = torch.cuda.get_device_properties(device)
            dev_info['gpu_name'] = props.name
            dev_info['vram_gb'] = round(props.total_memory / (1024**3), 1)
        elif device.type == 'mps':
            dev_info['dtype_default'] = 'float32'
            dev_info['note'] = 'CPU float64 fallback for damage CG solver'
        meta['device'] = dev_info

    if material is not None:
        meta['material'] = {
            'E': material.E, 'nu': material.nu,
            'Gc': material.Gc, 'l0': material.l0,
            'energy_split': material.energy_split,
            'pf_model': material.pf_model,
            'eta_residual': material.eta_residual,
            'plane_stress': bool(material.plane_stress),
        }
        if material.rho > 0:
            meta['material']['rho'] = material.rho

    if mesh is not None:
        meta['mesh'] = {
            'n_nodes': mesh.n_nodes,
            'n_elements': mesh.n_elems,
            'h_min': float(mesh.h_min),
        }

    if solver_config:
        meta['solver'] = solver_config

    # Git hash (best-effort)
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=3,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            meta['git_hash'] = result.stdout.strip()
    except Exception:
        pass

    if extra:
        for k, v in extra.items():
            if k in meta:
                meta[f'user_{k}'] = v  # prefix to avoid collision
            else:
                meta[k] = v

    if include_memory:
        mem = get_peak_memory_stats(device)
        if mem:
            meta['memory'] = mem

    path = os.path.join(output_dir, 'run_metadata.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    return path
