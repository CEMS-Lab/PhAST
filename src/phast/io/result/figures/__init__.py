"""Standard figures of any stored run: ``phast figures <run>`` (result file layout, section 7).

Draws, from a layout-2.x result file read by :func:`phast.load_result`, with no
run-specific code:

- a setup sheet: mesh or voxel grid with the phases, node sets and boundary
  conditions, materials and loading; voxel runs add an isometric material view
  with a cut-away corner and mid-plane sections;
- field sheets at chosen stored steps (default: first, the step nearest the
  largest load-like history value, last): displacement magnitude, damage,
  history variable (per-point values as cell means) and other scalar fields;
  2D maps on the true mesh, voxel runs as mid-plane sections and an isometric
  view; a component sheet for stress, strain and other multi-component fields;
  a sheet of damage across the chosen steps;
- an evolution sheet from the history columns: load and response, energies
  (with their method attributes), damage measures and residuals, when those
  columns exist.

PNG files stay below 2000 px on both axes and are listed in ``manifest.json``
beside them with their kind, steps, fields and the SHA-256 of the source
files. The package lives beside the reader because it is one of the views of
the result file; it is never imported by :mod:`phast.io.result` itself, so the
reader keeps depending on h5py and NumPy only.
"""
from __future__ import annotations

from .cli import draw_all, main

__all__ = ["draw_all", "main"]
