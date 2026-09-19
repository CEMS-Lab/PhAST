# Experimental TET4 geometry

The `phast.core.tet4_geometry` module prepares geometric quantities for
four-node linear tetrahedra. It supports geometry experiments independently
of the existing two-dimensional solver workflows.

## Geometric quantities

- `Tet4Geometry.from_tensors(...)` validates `(N, 3)` floating-point node
  coordinates and `(E, 4)` integer TET4 connectivity.
- Negative-orientation tetrahedra are repaired by local node reordering.
- Degenerate tetrahedra are rejected using element-scale volume criteria.
- The geometry object stores positive element volumes and constant gradients
  of the four TET4 shape functions. Gradients are computed from element edge
  vectors to avoid subtraction through an absolute-coordinate inverse.
- `structured_tet4_block(...)` creates a small rectangular block split into six
  tetrahedra per cell for tests and experimental setup checks.
- `nodes_on_plane(...)` selects mesh-aligned node sets by coordinate.

## A rectangular block

This example constructs a unit block and selects the nodes on its left face.
Coordinates use a consistent length unit chosen by the caller. Volumes have
the cube of that unit and shape gradients have its inverse.

```python
import torch
from phast.core.tet4_geometry import structured_tet4_block

geometry = structured_tet4_block(
    length=1.0, height=1.0, thickness=1.0, nx=2, ny=2, nz=2,
)
left_nodes = geometry.nodes_on_plane(axis=0, coordinate=0.0)
assert geometry.n_elements == 48
assert len(left_nodes) == 9
torch.testing.assert_close(geometry.volumes.sum(), torch.tensor(1.0, dtype=torch.float64))
```

The geometry stores precomputed volumes and gradients. Rebuild it after any
coordinate or connectivity change. In-place changes to its tensors, or to
shared input node storage, leave the precomputed quantities stale.

Plane selection uses a tolerance based on the selected coordinate extent and
floating-point spacing. An explicit `tolerance` is available. Coordinate
differences must remain representable in the selected dtype.

## Scope and verification

Tests cover orientation, volume, shape gradients, affine fields, rigid
translation and mesh-aligned plane selection in CPU float32 and float64.
This experimental module provides geometry only. Three-dimensional mechanics,
fracture, material assignment, boundary-condition objects and composite models
require separate solver implementation and verification. The existing
two-dimensional mesh and solver routes are unchanged.
