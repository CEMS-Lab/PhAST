"""Public API of the 3D voxel solver: describe a periodic cell, its materials and the load, then run.

    from phast import solid3d

    cell = solid3d.PeriodicCell(size_um=(24, 24, 24), voxel_um=0.5)
    cell.add_sphere_pore(diameter_fraction=0.4)
    matrix = solid3d.Material(E=345e3, nu=0.2, strength=600.0, Gc=25e-3)          # MPa, -, MPa, N/mm
    problem = solid3d.Problem(cell, materials={"matrix": matrix}, load=solid3d.MacroStrain("xx", to=3e-3, increments=150))
    summary = problem.run("runs/pore_in_cube", device="cpu")

The cell is periodic in x, y and z. One macro-strain component is prescribed; the other macro stress components are zero. Damage follows the
PF-CZM law with a spectral energy split and crack closure. Units: mm for the solver, micrometres in the cell description, MPa, N/mm.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

from phast.solid3d import geometry as geo
from phast.solid3d.materials import build_element_arrays
from phast.solid3d.model import (
    LoadCase, MaterialTable, PhaseMaterial, SolverConfig, VOIGT, VoxelGrid, VoxelModel, configure_threads,
)
from phast.solid3d.operator import PeriodicVoxelPhaseFieldOperator
from phast.solid3d.results import ResultFile
from phast.solid3d.staggered import StaggeredSolver, StaggerConvergenceError

PORE_STIFFNESS_RATIO = 1.0e-6
ROLE_PHASE = {"matrix": "mi_matrix", "inclusion": "fibre", "interface": "bn_band", "pore": "pore"}


@dataclass(frozen=True)
class Material:
    """Isotropic elastic phase with a PF-CZM fracture law: modulus ``E`` and strength in MPa, fracture energy ``Gc`` in N/mm."""

    E: float
    nu: float
    strength: float
    Gc: float

    def phase(self, role: str) -> PhaseMaterial:
        return PhaseMaterial(name=ROLE_PHASE[role], kind="isotropic", Gc=self.Gc, driver="spectral", E=self.E, nu=self.nu, tensile_strength=self.strength)


@dataclass(frozen=True)
class Interface(Material):
    """Smeared interface: a laminate of isotropic ``layers`` ``(fraction, E, nu)`` stacked along the interface normal.

    ``E`` and ``nu`` are the isotropic equivalent used only by the damage driver; the stiffness is that of the laminate."""

    layers: tuple[tuple[float, float, float], ...] = ()

    def phase(self, role: str) -> PhaseMaterial:
        return PhaseMaterial(name=ROLE_PHASE[role], kind="smeared_laminate", Gc=self.Gc, driver="spectral", E=self.E, nu=self.nu,
                             layers=tuple(tuple(layer) for layer in self.layers), tensile_strength=self.strength)


@dataclass(frozen=True)
class MacroStrain:
    """Prescribed macro strain: ``component`` (xx, yy, zz, yz, xz or xy) rises linearly to ``to`` in ``increments`` equal steps; the other macro stress components are zero."""

    component: str
    to: float
    increments: int

    def load_case(self) -> LoadCase:
        controlled = VOIGT.index(self.component)
        free = [i for i in range(6) if i != controlled]
        return LoadCase.from_schedule(f"macro_{self.component}", controlled=controlled, schedule=[(self.to, self.to / self.increments)], free=free, fixed_zero=[])


@dataclass(frozen=True)
class PFCZM:
    """Phase-field cohesive-zone damage law (Wu) with length scale ``length_scale_um`` and ``softening`` 'linear' or 'exponential'."""

    length_scale_um: float
    softening: str = "linear"
    p: float = 2


@dataclass(frozen=True)
class SolverSettings:
    """Staggered-solve tolerances. The defaults are the settings of the shipped examples."""

    stagger_tol: float = 1.0e-4
    max_stagger: int = 2000
    pcg_rtol_mech: float = 1.0e-8
    damage_kkt_tol: float = 1.0e-8
    max_cutbacks: int = 5
    crack_closure: str = "axes"
    batch_size: int = 16384


@dataclass
class PeriodicCell:
    """A regular voxel grid, periodic in x, y and z, with a phase label and an interface normal in every voxel."""

    size_um: Sequence[float]
    voxel_um: float
    phase_id: np.ndarray | None = None
    tangent: np.ndarray | None = None
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        h = self.voxel_um / 1000.0
        shape = []
        for length in self.size_um:
            n = length / self.voxel_um
            if abs(n - round(n)) > 1.0e-9:
                raise ValueError(f"the cell size {length} um is not a multiple of the voxel {self.voxel_um} um")
            shape.append(int(round(n)))
        self.grid = VoxelGrid(shape=tuple(shape), h=(h, h, h))
        if self.phase_id is None:
            self.phase_id = np.full(self.grid.array_shape, geo.MATRIX, dtype=np.uint8)
        if self.tangent is None:
            self.tangent = np.zeros((*self.grid.array_shape, 3))
            self.tangent[..., 2] = 1.0

    @classmethod
    def from_phase_ids(cls, phase_id: np.ndarray, voxel_um: float, tangent: np.ndarray | None = None) -> "PeriodicCell":
        """A cell from an imported microstructure: ``phase_id`` (nz, ny, nx) with 3 matrix, 0 inclusion, 1 interface, 4 pore."""
        nz, ny, nx = phase_id.shape
        return cls((nx * voxel_um, ny * voxel_um, nz * voxel_um), voxel_um, np.asarray(phase_id, np.uint8), tangent)

    def add_sphere_pore(self, diameter_fraction: float) -> "PeriodicCell":
        self.phase_id = geo.sphere_pore(self.grid, diameter_fraction)
        self.notes["pore_diameter_fraction"] = diameter_fraction
        return self

    def add_particles(self, n: int, radius_um: float, interface_um: float, gap_um: float = 1.0, seed: int = 0) -> "PeriodicCell":
        self.phase_id, self.tangent, centres = geo.particles(self.grid, n, radius_um, interface_um, gap_um, seed)
        self.notes.update(particle_centres_mm=centres.tolist(), particle_radius_um=radius_um, interface_um=interface_um, seed=seed)
        return self

    @property
    def roles_present(self) -> list[str]:
        ids = set(np.unique(self.phase_id).tolist())
        return [role for role, pid in geo.ROLES.items() if pid in ids]


@dataclass
class RunSummary:
    path: Path
    status: str
    increments: int
    stored_steps: list
    peak_stress: float
    peak_strain: float
    seconds: float


class Problem:
    """A periodic cell, its materials, the macro load and the damage law."""

    def __init__(self, cell: PeriodicCell, materials: Mapping[str, Material], load: MacroStrain, damage: PFCZM = PFCZM(1.5), solver: SolverSettings = SolverSettings()):
        missing = [r for r in cell.roles_present if r != "pore" and r not in materials]
        if missing:
            raise ValueError(f"no material for {missing}; the cell contains {cell.roles_present}")
        self.cell, self.materials, self.load, self.damage, self.solver = cell, dict(materials), load, damage, solver

    def model(self) -> VoxelModel:
        phases = {}
        for role in self.cell.roles_present:
            if role == "pore":
                m = self.materials["matrix"]
                phases["pore"] = PhaseMaterial(name="pore", kind="isotropic", Gc=m.Gc, driver="spectral", E=m.E * PORE_STIFFNESS_RATIO, nu=m.nu, tensile_strength=m.strength)
            else:
                phases[ROLE_PHASE[role]] = self.materials[role].phase(role)
        table = MaterialTable(phases=phases, length_scale=self.damage.length_scale_um / 1000.0, scale="micro")
        return VoxelModel(grid=self.cell.grid, phase_id=self.cell.phase_id, tangent=self.cell.tangent, materials=table)

    def solver_config(self, device: str, threads: int | None) -> SolverConfig:
        s = self.solver
        return SolverConfig(device=device, dtype="float64", num_threads=threads, batch_size=s.batch_size, kappa=1.0e-6, stagger_tol=s.stagger_tol, max_stagger=s.max_stagger,
                            pcg_rtol_mech=s.pcg_rtol_mech, damage_kkt_tol=s.damage_kkt_tol, max_cutbacks=s.max_cutbacks, crack_closure=s.crack_closure,
                            energy_split_isotropic="spectral", energy_split_band="isotropic", damage_law="pfczm", pfczm_softening=self.damage.softening, pfczm_p=self.damage.p,
                            on_stagger_failure="cutback", final_mechanics_resolve=True)

    def description(self) -> dict:
        """The problem as plain data (also written into the result file)."""
        return dict(cell=dict(size_um=list(self.cell.size_um), voxel_um=self.cell.voxel_um, shape=list(self.cell.grid.shape), roles=self.cell.roles_present, **self.cell.notes),
                    materials={k: {f: (list(map(list, v)) if f == "layers" else v) for f, v in vars(m).items()} for k, m in self.materials.items()},
                    load=vars(self.load), damage=vars(self.damage), solver=vars(self.solver))

    def run(self, output: str | Path, *, device: str = "cpu", threads: int | None = None, max_increments: int | None = None, store_every: int = 5,
            on_increment: Callable | None = None, quiet: bool = False) -> RunSummary:
        """Solve the load path and write ``<output>/result.vtkhdf``. The output folder must not exist; keep it outside a synchronised folder."""
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"{output} exists and is not empty")
        output.mkdir(parents=True, exist_ok=True)
        configure_threads(threads)
        started = time.perf_counter()
        model = self.model()
        cfg = self.solver_config(device, threads)
        load = self.load.load_case()
        op = PeriodicVoxelPhaseFieldOperator(model, load, cfg, materials=build_element_arrays(model, cfg, device=device))
        solver = StaggeredSolver(op)
        n_total = len(load.targets) if max_increments is None else min(len(load.targets), int(max_increments))
        stored_every = max(1, int(store_every))
        rf = ResultFile(output, op, cfg, load, run_id=output.name or "run", expected_steps=n_total // stored_every + 3,
                        config_text=yaml.safe_dump(json.loads(json.dumps(self.description(), default=str)), sort_keys=False))
        state0 = solver.initial_state()
        rf.add(solver.initial_record(), snapshot=solver.snapshot(state0), state=state0)
        peak = [0.0, 0.0]
        status, reason = "complete", None

        def accepted(state, rec):
            store = rec.step % stored_every == 0 or rec.increment >= n_total
            rf.add(rec, snapshot=solver.snapshot(state) if store else None, state=state)
            stress = float(rec.macro_stress[load.controlled])
            if stress > peak[0]:
                peak[:] = [stress, float(rec.macro_strain[load.controlled])]
            if on_increment is not None:
                on_increment(state, rec)
            if not quiet and (rec.increment % 10 == 0 or rec.increment >= n_total):
                print(f"[solid3d] increment {rec.increment}/{n_total}  E_{self.load.component} = {100 * rec.macro_strain[load.controlled]:.3f} %  stress = {stress:.1f} MPa  max d = {rec.max_damage:.2f}", flush=True)

        try:
            solver.run(max_increments=n_total, on_increment=accepted)
        except StaggerConvergenceError as exc:
            status, reason = "failed", str(exc)
            raise
        finally:
            path = rf.close(status, reason)
        return RunSummary(path=path, status=status, increments=n_total, stored_steps=rf.stored_steps, peak_stress=peak[0], peak_strain=peak[1],
                          seconds=time.perf_counter() - started)
