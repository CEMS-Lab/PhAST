"""phast.cohesive_elements — discrete cohesive zone primitives.

Topology helpers plus the first torch-native residual/tangent/energy-state
operator are available. Quasi-static mechanics coupling is implemented through
``QuasiStaticSolver(cohesive_operator=...)``. A bounded brittle PF+cohesive
validation smoke exists in the examples; calibrated structural PF+cohesive
benchmarks and the full traction-separation-law suite — PPR, Camanho,
mixed-mode unloading — remain tracked under #259/#350.

Design context: see issue #261 (this issue) and the PF-CZM design doc landed in
commit ``ae39668`` for the contrast vs the diffuse PF-CZM track (#247).

Public API
----------
- :class:`CohesiveElement` — per-interface-element side-data record.
- :func:`build_cohesive_strip` — given a mesh and a Physical Line id, returns
  the list of duplicated nodes + interface elements (does NOT mutate the mesh).
- :func:`cohesive_traction` — legacy exponential TSL helper.
- :class:`CohesiveInterfaceOperator` — torch residual/tangent/state operator.
"""

from .cohesive_elements import (
    CohesiveElement,
    build_cohesive_strip,
    cohesive_traction,
)
from .mesh_integration import (
    CohesiveInsertionResult,
    MeshIOCohesiveInsertionResult,
    insert_cohesive_layer,
    insert_cohesive_layer_meshio,
    insert_cohesive_layer_with_metadata,
)
from .operator import (
    BilinearCohesiveLaw,
    CohesiveInterfaceOperator,
    CohesiveState,
)

__all__ = [
    "BilinearCohesiveLaw",
    "CohesiveElement",
    "CohesiveInsertionResult",
    "CohesiveInterfaceOperator",
    "CohesiveState",
    "MeshIOCohesiveInsertionResult",
    "build_cohesive_strip",
    "cohesive_traction",
    "insert_cohesive_layer",
    "insert_cohesive_layer_meshio",
    "insert_cohesive_layer_with_metadata",
]
