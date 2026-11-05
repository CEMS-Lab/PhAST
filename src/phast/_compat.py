"""Legacy import paths.

Earlier releases exposed flat module names (``phast.mesh``, ``phast.damage_solver``) and subpackages that have since been renamed
(``phast.config``, ``phast.utils``, ``phast.plasticity``). The code now lives in one place per topic:

    phast.api       Problem, Result: what users import
    phast.core      mesh, elements, FEM operators, material-independent kernels, device handling
    phast.solvers   mechanics, damage, staggered coupling, linear solvers, time integrators
    phast.physics   material laws, boundary and initial conditions, plasticity, cohesive elements
    phast.io        configuration, result files, trajectory outputs, provenance
    phast.cli       doctor, scaffolder, setup preview
    phast.viz       figure style, plots, post-processing
    phast.workflow  fluent problem specification and the example runners
    phast.solid3d   3D voxel fracture (beta)

This module keeps every old import path working. ``import phast.mesh`` and ``from phast.utils.visualization import plot_damage`` return the
module that now lives at the new path (the same object, so patching either name patches both), and ``python -m phast.precheck`` still runs.
New code should import the new paths; the table below lists the old names that are served.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from types import ModuleType

# legacy module name -> current module name
MODULE_ALIASES: dict[str, str] = {
    "phast.adaptive": "phast.solvers.adaptive",
    "phast.bar": "phast.core.bar",
    "phast.boundary_conditions": "phast.physics.boundary_conditions",
    "phast.config_schema": "phast.io.config.config_schema",
    "phast.config_validation": "phast.io.config.config_validation",
    "phast.core.problem": "phast.api.problem",
    "phast.core.scaffolder": "phast.cli.scaffolder",
    "phast.damage_solver": "phast.solvers.damage_solver",
    "phast.device": "phast.core.device",
    "phast.doctor": "phast.cli.doctor",
    "phast.explain_config": "phast.io.config.explain_config",
    "phast.fem_operators": "phast.core.fem_operators",
    "phast.fracture_mechanics": "phast.physics.fracture_mechanics",
    "phast.geometry_compiler": "phast.core.geometry_compiler",
    "phast.geometry_dsl": "phast.core.geometry_dsl",
    "phast.h_update": "phast.physics.h_update",
    "phast.h_update_custom_subgrad": "phast.physics.h_update_custom_subgrad",
    "phast.initial_conditions": "phast.physics.initial_conditions",
    "phast.io_utils": "phast.io.outputs",
    "phast.line_mesh": "phast.core.line_mesh",
    "phast.material": "phast.physics.material",
    "phast.mechanics_solver": "phast.solvers.mechanics_solver",
    "phast.mesh": "phast.core.mesh",
    "phast.mesh_generator": "phast.core.mesh_generator",
    "phast.mesh_inspection": "phast.core.mesh_inspection",
    "phast.metrics": "phast.core.metrics",
    "phast.mixed_precision_cg": "phast.solvers.mixed_precision_cg",
    "phast.multigrid": "phast.solvers.multigrid",
    "phast.p2_elements": "phast.core.p2_elements",
    "phast.paper_figure_style": "phast.viz.figure_style",
    "phast.plasticity_interface_runner": "phast.workflow.plasticity_interface_runner",
    "phast.postprocess_hdf5": "phast.io.postprocess_hdf5",
    "phast.postprocess_paper": "phast.viz.postprocess",
    "phast.precheck": "phast.io.config.precheck",
    "phast.problem": "phast.api.problem",
    "phast.provenance": "phast.io.provenance",
    "phast.quad_elements": "phast.core.quad_elements",
    "phast.region_resolution": "phast.core.region_resolution",
    "phast.result": "phast.api.result",
    "phast.run_config": "phast.io.config.run_config",
    "phast.scaffolder": "phast.cli.scaffolder",
    "phast.setup_preview": "phast.cli.setup_preview",
    "phast.solid_mechanics_runner": "phast.workflow.solid_mechanics_runner",
    "phast.sparse_solve": "phast.solvers.sparse_solve",
    "phast.staggered_solver": "phast.solvers.staggered_solver",
    "phast.thermal_update": "phast.physics.thermal_update",
    "phast.time_integrators": "phast.solvers.time_integrators",
    "phast.units": "phast.core.units",
    "phast.utils.device": "phast.core.device",
    "phast.utils.doctor": "phast.cli.doctor",
    "phast.utils.io_utils": "phast.io.outputs",
    "phast.utils.metrics": "phast.core.metrics",
    "phast.utils.paper_figure_style": "phast.viz.figure_style",
    "phast.utils.postprocess_hdf5": "phast.io.postprocess_hdf5",
    "phast.utils.postprocess_paper": "phast.viz.postprocess",
    "phast.utils.provenance": "phast.io.provenance",
    "phast.utils.units": "phast.core.units",
    "phast.utils.visualization": "phast.viz.visualization",
    "phast.visualization": "phast.viz.visualization",
}

# legacy package name -> current package name; every submodule follows (phast.config.run_config -> phast.io.config.run_config)
PACKAGE_ALIASES: dict[str, str] = {
    "phast.cohesive_elements": "phast.physics.cohesive_elements",
    "phast.config": "phast.io.config",
    "phast.plasticity": "phast.physics.plasticity",
    "phast.solid_mechanics_runners": "phast.workflow.runners",
}

# packages that no longer exist; their names resolve to a namespace that forwards to the modules listed in MODULE_ALIASES
REMOVED_PACKAGES = ("phast.utils",)


def resolve(name: str) -> tuple[str, bool] | None:
    """Return ``(current_name, is_package)`` for a legacy module name, or ``None``."""
    if name in MODULE_ALIASES:
        return MODULE_ALIASES[name], False
    for old, new in PACKAGE_ALIASES.items():
        if name == old:
            return new, True
        if name.startswith(old + "."):
            return new + name[len(old):], False
    return None


class _Loader(importlib.abc.Loader):
    def __init__(self, target: str, is_package: bool):
        self._target, self._is_package = target, is_package

    def create_module(self, spec):
        return importlib.import_module(self._target)          # the alias is the same module object

    def exec_module(self, module):
        return None

    def is_package(self, fullname):
        return self._is_package

    def get_code(self, fullname):                              # lets ``python -m phast.precheck`` run the current module
        spec = importlib.util.find_spec(self._target)
        return spec.loader.get_code(self._target) if spec and spec.loader else None

    def get_source(self, fullname):
        spec = importlib.util.find_spec(self._target)
        return spec.loader.get_source(self._target) if spec and spec.loader else None


class _NamespaceLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = ModuleType(spec.name)
        module.__path__ = []                                     # a package, so ``from phast.utils import visualization`` resolves

        def __getattr__(attr, _prefix=spec.name):
            target = MODULE_ALIASES.get(f"{_prefix}.{attr}")
            if target is None:
                raise AttributeError(f"module {_prefix!r} has no attribute {attr!r}")
            return importlib.import_module(target)
        module.__getattr__ = __getattr__
        return module

    def exec_module(self, module):
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("phast."):
            return None
        if fullname in REMOVED_PACKAGES:
            return importlib.util.spec_from_loader(fullname, _NamespaceLoader(), is_package=True)
        hit = resolve(fullname)
        if hit is None:
            return None
        current, is_package = hit
        return importlib.util.spec_from_loader(fullname, _Loader(current, is_package), is_package=is_package)


def install() -> None:
    """Register the legacy-path finder once."""
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())


def alias_module(alias: str, target: str) -> ModuleType:
    """Expose ``target`` under the module name ``alias`` (kept for code that registered its own aliases)."""
    module = importlib.import_module(target)
    sys.modules[alias] = module
    return module
