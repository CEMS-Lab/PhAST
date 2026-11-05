#!/usr/bin/env python3
"""Spherical pore in a periodic cube.

A 24 um cube (48 x 48 x 48 voxels of 0.5 um, 110,592 hexahedral elements), periodic in x, y and z, with a spherical pore of diameter 0.4 L at its centre,
pulled in tension along x. PF-CZM damage, spectral energy split, crack closure.

Run
    python examples/three_dimensional/pore_in_cube/run.py --output-dir runs/pore_in_cube                       # full deck
    python examples/three_dimensional/pore_in_cube/run.py --output-dir runs/pore_in_cube_quick --max-increments 4   # setup check, about half a minute

The run writes one ``result.vtkhdf`` (float64) that ``phast.load_result`` reads and ParaView opens. The output folder must not exist and
must lie outside a cloud-synchronised folder.
"""
import argparse

from phast import solid3d


def build_problem() -> solid3d.Problem:
    cell = solid3d.PeriodicCell(size_um=(24, 24, 24), voxel_um=0.5)
    cell.add_sphere_pore(diameter_fraction=0.4)                                              # pore diameter 0.4 L at the cell centre
    matrix = solid3d.Material(E=345e3, nu=0.2, strength=600.0, Gc=25e-3)                    # MPa, -, MPa, N/mm
    return solid3d.Problem(
        cell, materials={"matrix": matrix},
        load=solid3d.MacroStrain("xx", to=3e-3, increments=150),            # E_xx to 0.30 %; the other macro stresses are zero
        damage=solid3d.PFCZM(length_scale_um=1.5),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True, help="run folder to create (must not exist)")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: the PyTorch default)")
    ap.add_argument("--max-increments", type=int, default=None, help="stop after this many load increments (default: all 150)")
    ap.add_argument("--store-every", type=int, default=5, help="store the fields every N increments (history is stored at every increment)")
    a = ap.parse_args(argv)
    summary = build_problem().run(a.output_dir, device=a.device, threads=a.threads, max_increments=a.max_increments, store_every=a.store_every)
    print(f"{summary.status}: {summary.increments} increments, peak stress {summary.peak_stress:.1f} MPa at {100 * summary.peak_strain:.3f} % strain -> {summary.path}")
    return 0 if summary.status == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
