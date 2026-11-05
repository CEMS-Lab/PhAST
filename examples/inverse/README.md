# Inverse Examples

This folder contains one worked inverse calculation: the
[cross-mesh toughness recovery](cross_mesh_toughness/README.md).
It estimates the fracture toughness $G_c$ from synthetic displacement
readings that were generated on a finer mesh.
The folder holds the recorded results of the recovery by automatic
differentiation (AD) and by Brent's scalar search, and a script that repeats
the AD recovery with the current solver.

| Paper calculation | Folder | Recorded toughness error |
| --- | --- | --- |
| Cross-mesh toughness recovery ($h = 0.1875$ mm inverse mesh) | [cross_mesh_toughness](cross_mesh_toughness/README.md) | 2.22 % (AD), 1.65 % (Brent) |

The observations are synthetic, and the recovery starts from a stated initial
value.
New runs write to a directory under `runs/`.

The glass and alumina scalar drivers and recorded histories are also included
under [reproduction/cmame_2026](../../reproduction/cmame_2026/README.md).
The larger `PhAST-paper-reproduction.zip` holds additional study records,
including the joint toughness and length-scale recovery. It is separate
from this software ZIP.

The folder [`_shared`](_shared/README.md) holds the forward model, recovery
and plotting modules that the cross-mesh scripts import.

If a command fails, open a GitHub issue with the command, the configuration,
the operating system, the Python and PhAST versions and the complete error
output.
