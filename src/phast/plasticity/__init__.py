"""
phast.plasticity — standalone elastoplastic kernels.

This package is the plasticity track foundation. It exposes a material-point
return-mapping integrator plus the first mesh-level J2 state/coupling layer.
The mesh layer validates per-element state, commit/rollback, internal-force
assembly, plastic-work accounting, and ductile phase-field driving-force
coupling. The beta validation examples solve bounded AT2 damage on that
ductile history; full global ``StaggeredSolver`` PF-plasticity benchmark
integration is still a separate hardening step.

Currently provided
------------------
- :class:`J2Plasticity` — rate-independent J2 (von Mises) flow with
  optional linear isotropic / Voce / Swift hardening. Plane-strain
  and plane-stress paths supported (plane stress uses a nested
  Newton iteration on the through-thickness elastic strain).
- :class:`MeshJ2Elastoplasticity` — one-point-per-element stateful J2 layer.
- :class:`SparseJ2QuasiStaticSolver` — sparse J2 Newton solver with
  element algorithmic tangents for validation and production hardening.
- :class:`DuctilePhaseFieldCoupling` — elastic-plus-plastic-work phase-field
  driving-force helper for ductile damage validation examples.

Public API
----------
>>> from phast.material import Material
>>> from phast.plasticity import J2Plasticity
>>> mat = Material(E=210000.0, nu=0.3, plasticity_model='j2_isotropic',
...                yield_stress=250.0, hardening_modulus=1000.0,
...                hardening_type='linear_iso')
>>> kernel = J2Plasticity(mat)

Units
-----
Consistent mm-tonne-N-s (MPa) — same as the rest of the solver. See
``phast/units.py``.
"""

from .j2_vonmises import J2Plasticity, J2State
from .mesh_j2 import (
    DuctilePhaseFieldCoupling,
    MeshJ2Elastoplasticity,
    MeshJ2State,
    SparseJ2QuasiStaticSolver,
    strain3d_from_mesh,
)

__all__ = [
    'DuctilePhaseFieldCoupling',
    'J2Plasticity',
    'J2State',
    'MeshJ2Elastoplasticity',
    'MeshJ2State',
    'SparseJ2QuasiStaticSolver',
    'strain3d_from_mesh',
]
