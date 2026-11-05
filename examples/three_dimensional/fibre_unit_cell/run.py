#!/usr/bin/env python3
"""Nine-fibre unit cell under transverse tension.

A 51.5 x 51.5 x 1 um cell (103 x 103 x 2 voxels of 0.5 um, 21,218 hexahedral elements), periodic in x, y and z, with nine parallel fibres and an interface band around
each. The cell is uniform along the fibre axis z, so the response is the transverse plane problem solved with 3D elements. The microstructure (phase of every voxel and the
interface normal) is read from microstructure.npz; any voxel image can be imported the same way with PeriodicCell.from_phase_ids.

Run
    python examples/three_dimensional/fibre_unit_cell/run.py --output-dir runs/fibre_unit_cell                       # full deck
    python examples/three_dimensional/fibre_unit_cell/run.py --output-dir runs/fibre_unit_cell_quick --max-increments 4   # setup check, about half a minute

The run writes one ``result.vtkhdf`` (float64) that ``phast.load_result`` reads and ParaView opens. The output folder must not exist and
must lie outside a cloud-synchronised folder.
"""
import argparse
from pathlib import Path

import numpy as np

from phast import solid3d


def build_problem() -> solid3d.Problem:
    image = np.load(Path(__file__).with_name("microstructure.npz"))                          # phase_id (nz, ny, nx) and the interface normal of every voxel
    cell = solid3d.PeriodicCell.from_phase_ids(image["phase_id"], float(image["voxel_um"]), image["tangent"])
    matrix = solid3d.Material(E=345e3, nu=0.2, strength=600.0, Gc=25e-3)                    # MPa, -, MPa, N/mm
    fibre = solid3d.Material(E=270e3, nu=0.2, strength=3136.0, Gc=0.5180310650205762)
    interface = solid3d.Interface(
        E=119683.28713197137, nu=0.2, strength=63.0, Gc=0.0006463399858678095,
        layers=((0.10823084088511756, 20000.0, 0.2), (0.4458845795574412, 270000.0, 0.2), (0.4458845795574412, 345000.0, 0.2)))
    return solid3d.Problem(
        cell, materials={"matrix": matrix, "inclusion": fibre, "interface": interface},
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
