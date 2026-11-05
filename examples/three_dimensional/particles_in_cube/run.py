#!/usr/bin/env python3
"""Spherical particles with a weak interface shell in a periodic cube.

A 24 um cube (110,592 hexahedral elements), periodic in x, y and z, with four spherical particles of radius 5 um placed by random sequential addition. Each particle
is wrapped in an interface shell about 1.9 um thick: an oriented laminate whose stiffness, strength (63 MPa) and fracture energy are lower than the matrix.

Run
    python examples/three_dimensional/particles_in_cube/run.py --output-dir runs/particles_in_cube                       # full deck
    python examples/three_dimensional/particles_in_cube/run.py --output-dir runs/particles_in_cube_quick --max-increments 4   # setup check, about half a minute

The run writes one ``result.vtkhdf`` (float64) that ``phast.load_result`` reads and ParaView opens. The output folder must not exist and
must lie outside a cloud-synchronised folder.
"""
import argparse

from phast import solid3d


def build_problem() -> solid3d.Problem:
    cell = solid3d.PeriodicCell(size_um=(24, 24, 24), voxel_um=0.5)
    cell.add_particles(n=4, radius_um=5.0, interface_um=1.9235993121898142, gap_um=1.0, seed=20261023)
    matrix = solid3d.Material(E=345e3, nu=0.2, strength=600.0, Gc=25e-3)                    # MPa, -, MPa, N/mm
    particle = solid3d.Material(E=270e3, nu=0.2, strength=3136.0, Gc=0.5180310650205762)
    interface = solid3d.Interface(                                                            # laminate of (fraction, E, nu) layers across the shell
        E=122601.74043706217, nu=0.2, strength=63.0, Gc=0.00064633998586781,
        layers=((0.10397175686880507, 20000.0, 0.2), (0.44801412156559745, 270000.0, 0.2), (0.44801412156559745, 345000.0, 0.2)))
    return solid3d.Problem(
        cell, materials={"matrix": matrix, "inclusion": particle, "interface": interface},
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
