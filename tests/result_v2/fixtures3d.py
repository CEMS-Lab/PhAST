"""Layout-2.0 files with 3D unstructured meshes, written by the independent reference builder.

``tests/result_v2`` has no unstructured 3D file; the views, the figures of 3D meshes and the tensor-order check
need one. Values encode their indices (as in ``tests/result_v2/fixtures.py``) and the symmetric tensors carry
distinct ``xy``, ``yz`` and ``xz`` values, so that a reordered tensor changes its principal values.
"""
from __future__ import annotations

import numpy as np

from result_v2 import fixtures as _fx
from result_v2.fixtures import ISO, F, U, coded, history, per_point, qp_block, write_fixture

VTK_H8, VTK_TET4 = 12, 10
_fx.VTK.setdefault("TET4", VTK_TET4)       # the reference builder lists only the types of its own fixtures
# layout order xx, yy, zz, yz, xz, xy; distinct shear values
S0 = np.array([[10.0, 3.0, 2.0], [3.0, 20.0, 1.0], [2.0, 1.0, 30.0]])
S0_LAYOUT = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
E0 = np.array([[1.0, 0.6, 0.5], [0.6, 2.0, 0.4], [0.5, 0.4, 3.0]]) * 1e-3
E0_ENGINEERING = np.array([1.0, 2.0, 3.0, 0.8, 1.0, 1.2]) * 1e-3          # gamma = 2 * epsilon for the shears
COMPONENTS_3D = "xx,yy,zz,yz,xz,xy"


def hex_mesh(nx: int, ny: int, nz: int, lx: float = 4.0, ly: float = 3.0, lz: float = 2.0):
    xs, ys, zs = np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1), np.linspace(0, lz, nz + 1)
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    p = lambda i, j, k: i + (nx + 1) * (j + (ny + 1) * k)
    cells = [[p(i, j, k), p(i + 1, j, k), p(i + 1, j + 1, k), p(i, j + 1, k),
              p(i, j, k + 1), p(i + 1, j, k + 1), p(i + 1, j + 1, k + 1), p(i, j + 1, k + 1)]
             for k in range(nz) for j in range(ny) for i in range(nx)]
    return pts, np.array(cells, np.int64)


def split_tets(pts: np.ndarray, hexes: np.ndarray) -> np.ndarray:
    tets = []
    for h in hexes:
        for a, b, c, d in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)):
            t = [h[a], h[b], h[c], h[d]]
            vol = np.linalg.det(np.stack([pts[t[1]] - pts[t[0]], pts[t[2]] - pts[t[0]], pts[t[3]] - pts[t[0]]]))
            tets.append(t if vol > 0 else [t[0], t[2], t[1], t[3]])
    return np.array(tets, np.int64)


def _scaled(ns: int, e: int, base: np.ndarray, n_qp: int | None = None) -> np.ndarray:
    """Tensor ``(1 + step + 0.01 cell [+ 0.001 point]) * base`` for every step and cell (and point)."""
    s = np.arange(ns)[:, None]
    c = np.arange(e)[None, :]
    f = 1.0 + s + 0.01 * c
    if n_qp is None:
        return f[..., None] * base
    q = np.arange(n_qp)[None, None, :]
    return (f[..., None] + 0.001 * q)[..., None] * base


def unstructured_3d(kind: str = "h8", interface: bool = False):
    """Description of a 3D unstructured run: ``kind`` is ``h8`` (24 hexahedra, 8 points each) or ``tet4``
    (144 tetrahedra, one point each); ``interface`` adds a COH4 interface block on the plane z = lz / 2."""
    nx, ny, nz = 4, 3, 2
    pts, hexes = hex_mesh(nx, ny, nz)
    ns = 3
    n = len(pts)
    if kind == "h8":
        cells, vtk, block = hexes, VTK_H8, "H8"
    else:
        cells, vtk, block = split_tets(pts, hexes), VTK_TET4, "TET4"
    e = len(cells)
    if kind == "h8":
        qp = qp_block("H8", e)
        H = per_point(ns, e, 8, base=0.5)
        cell_fields = {"H": (H, F("quadrature_point", ",".join(f"qp{q}" for q in range(8)), "MPa", "history"), float)}
    else:
        qp = dict(cell_type="TET4", range=[0, e], qp_points=[[0.25, 0.25, 0.25]], qp_weights=[1 / 6], rule="centroid",
                  point_order="single point")
        H = coded(ns, e, base=0.5)
        cell_fields = {"H": (H, F("cell", "H", "MPa", "history"), float)}
    method = "qp_mean_volume_weighted" if kind == "h8" else "exact"
    stress = _scaled(ns, e, S0_LAYOUT)
    strain = _scaled(ns, e, E0_ENGINEERING)
    cell_fields["stress"] = (stress, F("cell", COMPONENTS_3D, "MPa", "derived", derived_method=method), np.float64)
    cell_fields["strain"] = (strain, F("cell", COMPONENTS_3D, "1", "derived", derived_method=method, shear="engineering"),
                             np.float64)
    extra = {}
    if kind == "h8":
        extra["quadrature/H8/stress"] = (_scaled(ns, e, S0_LAYOUT, 8), F("quadrature_point", COMPONENTS_3D, "MPa", "derived"))
    extra_attrs = {}
    interface_rows = 0
    if interface:
        mid = [i + (nx + 1) * (j + (ny + 1) * 1) for j in range(ny) for i in range(nx)]
        lower = np.array([[m, m + 1, m + 1 + (nx + 1), m + (nx + 1)] for m in mid], np.int64)
        coh = np.concatenate([lower, lower], axis=1)          # lower side then upper side (the same nodes in this file)
        interface_rows = len(coh)
        extra["interfaces/crack/connectivity"] = (coh, {"cell_type": "COH4"})
        pts_q = np.array([[-0.5774, -0.5774], [0.5774, -0.5774], [-0.5774, 0.5774], [0.5774, 0.5774]])
        extra["interfaces/crack/quadrature/points"] = (pts_q, {})
        extra["interfaces/crack/cohesive_damage"] = (per_point(ns, len(coh), 4) / 1e4,
                                                     F("interface_point", "d", "1", "history"))
        extra_attrs["interfaces/crack"] = dict(law="exponential", strength=30.0, Gc=0.5,
                                               units='{"strength": "MPa", "Gc": "N/mm"}')
    top = np.nonzero(np.isclose(pts[:, 2], 2.0))[0]
    bottom = np.nonzero(np.isclose(pts[:, 2], 0.0))[0]
    desc = dict(kind="ug", points=pts, cells=cells, types=[vtk] * e,
                root=dict(profile="run", dim=3, analysis="3d", run_id=f"views-{kind}{'-coh' if interface else ''}"),
                point_fields={"displacement": (U(ns, n, 3), F("node", "x,y,z", "mm", "state"), float),
                              "damage": (coded(ns, n, base=0.25) / 1e4, F("node", "d", "1", "state"), float)},
                cell_fields=cell_fields, cell_blocks={block: qp},
                node_sets={"top": top, "bottom": bottom}, phases={0: ISO}, extra=extra, extra_attrs=extra_attrs,
                bc={"top": dict(set="top", kind="displacement", components="z", value="load_parameter"),
                    "bottom": dict(set="bottom", kind="displacement", components="x,y,z", value=0.0)},
                state_links={"displacement": "/VTKHDF/PointData/displacement", "damage": "/VTKHDF/PointData/damage",
                             "H": "/VTKHDF/CellData/H"},
                history=history(ns, [0, 1, 2]), step_times=[0.01, 0.02, 0.03])
    info = dict(n_points=n, n_cells=e, n_interface=interface_rows, kind=kind, block=block, steps=ns)
    return desc, info


def voxel_tensor():
    """ImageData run (6 x 5 x 4 voxels) with a 3D stress and an engineering-shear strain."""
    nx, ny, nz = 6, 5, 4
    ns = 3
    e = nx * ny * nz
    factors = (1.0 + np.arange(ns)[:, None, None, None] + 0.01 * np.arange(e).reshape(nz, ny, nx)[None])
    stress = factors[..., None] * S0_LAYOUT
    strain = factors[..., None] * E0_ENGINEERING
    k, j, i = np.meshgrid(np.arange(nz + 1), np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
    damage = np.stack([1e6 * s + 1e4 * i + 1e2 * j + k for s in range(ns)]).astype(float)
    desc = dict(kind="image", grid=(nx, ny, nz), spacing=(0.5, 0.5, 0.5),
                root=dict(profile="run", dim=3, analysis="3d", run_id="views-voxel-tensor"),
                static_cell={"phase_id": ((np.arange(e).reshape(nz, ny, nx) % 3).astype(np.uint8), F("cell", "id", "1", "state"))},
                point_fields={"displacement": (np.repeat(damage[..., None], 3, axis=-1) * 1e-6, F("node", "x,y,z", "mm", "state"), float),
                              "damage": (damage / 1e7, F("node", "d", "1", "state"), float)},
                cell_fields={"stress": (stress, F("cell", COMPONENTS_3D, "MPa", "derived", derived_method="exact"), np.float64),
                             "strain": (strain, F("cell", COMPONENTS_3D, "1", "derived", derived_method="exact", shear="engineering"), np.float64)},
                cell_blocks={"H8": qp_block("H8", e)}, phases={0: ISO, 1: dict(ISO, name="b"), 2: dict(ISO, name="c")},
                state_links={"displacement": "/VTKHDF/PointData/displacement", "damage": "/VTKHDF/PointData/damage"},
                history=history(ns, [0, 1, 2]), step_times=[0.01, 0.02, 0.03])
    return desc, dict(n_points=(nx + 1) * (ny + 1) * (nz + 1), n_cells=e, steps=ns)


BUILDERS = {"h8_block": lambda: unstructured_3d("h8"), "tet4_block": lambda: unstructured_3d("tet4"),
            "h8_cohesive": lambda: unstructured_3d("h8", interface=True), "voxel_tensor": voxel_tensor}


def build_all(out_dir):
    """Write the 3D files into ``out_dir``; returns ``{name: (path, info)}``."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, build in BUILDERS.items():
        desc, info = build()
        write_fixture(out / f"{name}.vtkhdf", desc)
        result[name] = (out / f"{name}.vtkhdf", info)
    return result
