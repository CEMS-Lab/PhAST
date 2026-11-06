"""3D voxel phase-field fracture on periodic unit cells (beta).

``from phast import solid3d`` gives ``PeriodicCell``, ``Material``, ``Interface``, ``MacroStrain``, ``PFCZM``, ``SolverSettings`` and ``Problem``;
see :mod:`phast.solid3d.api` for a complete example. The solver core (``model``, ``operator``, ``periodic``, ``staggered``, ``splits``, ``materials``,
``linalg``) is the matrix-free eight-node hexahedral engine; ``results`` writes ``result.vtkhdf``.
"""
# PhAST | Allamaprabhu Ani | https://allamaprabhuani.github.io/

from phast.solid3d.api import Interface, MacroStrain, Material, PFCZM, PeriodicCell, Problem, RunSummary, SolverSettings

__all__ = ["Interface", "MacroStrain", "Material", "PFCZM", "PeriodicCell", "Problem", "RunSummary", "SolverSettings"]
