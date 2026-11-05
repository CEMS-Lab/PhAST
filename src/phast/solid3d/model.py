"""Shared constants, dataclasses and device helpers for periodic voxel cells.

Conventions (frozen):

* units are mm, MPa, N/mm (fracture energy), N and mJ = N*mm;
* voxel arrays are ``(nz, ny, nx)`` in C order with element id
  ``e = i + nx*(j + ny*k)``;
* full node arrays are ``(nz+1, ny+1, nx+1)`` with node id
  ``i + (nx+1)*(j + (ny+1)*k)``, identical to
  :meth:`phast.core.hex8.H8Mesh.structured_block`;
* Voigt order is ``(xx, yy, zz, yz, xz, xy)`` with engineering shear, so a
  stiffness maps engineering strain to stress;
* the hybrid AT2 degradation is ``g = (1-kappa)(1-d)^2 + kappa``
  (``phast_standard_quadratic_v1``); with ``SolverConfig.damage_law =
  'pfczm'`` it is Wu's rational PF-CZM degradation with the same
  residual blend, ``g = (1-kappa) A/(A + Q) + kappa``;
* precision is float64 only, on CPU or CUDA.  MPS and float32 are refused.

Every dataclass is frozen except :class:`VoxelModel`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phast.physics.pfczm import PFCZM_SOFTENING_EXPONENTS
from phast.core.hex8 import H8Mesh


VOIGT: tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy")
"""Engineering Voigt component order."""

PHASE_NAMES: tuple[str, ...] = (
    "fibre",
    "bn_band",
    "cvi_sic",
    "mi_matrix",
    "pore",
)
"""Phase labels: inclusion (``fibre``), interface (``bn_band``), ``cvi_sic``, ``mi_matrix`` and ``pore``."""

PHASE_ID: dict[str, int] = {name: index for index, name in enumerate(PHASE_NAMES)}
"""Phase name to integer id stored in :attr:`VoxelModel.phase_id`."""

DRIVER_NONE, DRIVER_FULL, DRIVER_SPECTRAL = 0, 1, 2
"""Per-element damage-driver codes used by :class:`ElementMaterialArrays`."""

DRIVER_INTERFACE = 3
"""Directional interface driver of the smeared BN band (``SolverConfig.energy_split_band =
'interface'``, ): the band is driven by the interface traction energy only.  Its
stress is the hybrid ``g C eps`` with ``SolverConfig.band_stress = 'hybrid'`` (the default,
) or the derivative of the degraded energy ``psi0 - (1 - g) psi_plus`` with
``band_stress = 'variational'`` (see :mod:`.splits`)."""

DRIVER_NAMES: tuple[str, ...] = ("none", "full", "spectral", "interface")
"""Driver names indexed by driver code."""

BAND_SPLITS: tuple[str, ...] = ("isotropic", "interface")
"""Values of :attr:`SolverConfig.energy_split_band`."""

BAND_STRESSES: tuple[str, ...] = ("hybrid", "variational")
"""Values of :attr:`SolverConfig.band_stress`."""

CRACK_CLOSURES: tuple[str, ...] = ("none", "axes")
"""Values of :attr:`SolverConfig.crack_closure`."""

_DEFERRED_CRACK_CLOSURES: dict[str, str] = {
    "onset": "the frozen-frame closure ('onset') is deferred to a later amendment",
    "spectral": "the Miehe spectral closure reference ('spectral') is deferred to a later amendment",
}

DRIVER_CODE: dict[str, int] = {name: code for code, name in enumerate(DRIVER_NAMES)}
"""Driver name to driver code."""

DAMAGE_LAWS: tuple[str, ...] = ("at2", "pfczm")
"""Values of :attr:`SolverConfig.damage_law` (``'pfczm'``: )."""

PFCZM_SOFTENINGS: tuple[str, ...] = ("linear", "exponential")
"""Values of :attr:`SolverConfig.pfczm_softening`."""

PFCZM_SETTINGS: tuple[str, ...] = ("pfczm_softening", "pfczm_p", "pfczm_max_newton")
""":class:`SolverConfig` settings that apply with ``damage_law = 'pfczm'`` only."""

DEGRADATION = "phast_standard_quadratic_v1"
"""Degradation law ``g = (1-kappa)(1-d)^2 + kappa`` (so ``g(0) = 1``)."""

PFCZM_DEGRADATION = "wu_pfczm_rational_v1"
"""Degradation label of ``damage_law = 'pfczm'``: Wu's rational
``g = (1-kappa) A/(A + Q) + kappa`` with ``A = (1-d)^p`` and
``Q = a1 d (1 + a2 d + a2 a3 d^2)``.  The resolved parameters of a PF-CZM run
record it as ``phase_field.degradation_convention`` in place of the inputs
convention :data:`DEGRADATION`, which they keep as
``phase_field.inputs_degradation_convention``."""

MATERIAL_KINDS: tuple[str, ...] = ("isotropic", "smeared_laminate")
"""Allowed :attr:`PhaseMaterial.kind` values."""

MATERIAL_TABLE_COLUMNS: tuple[str, ...] = (
    "phase",
    "phase_id",
    "kind",
    "E",
    "nu",
    "Gc",
    "driver",
    "sources",
)
"""Frozen ``material_table.csv`` columns returned by :meth:`MaterialTable.rows`."""

MATERIAL_TABLE_STRENGTH_COLUMNS: tuple[str, ...] = ("f_t",)
"""Columns appended to ``material_table.csv`` when a phase has a tensile strength (PF-CZM, )."""


# ---------------------------------------------------------------------------
# small validators
# ---------------------------------------------------------------------------


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _positive(name: str, value: float) -> float:
    value = _finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def _non_negative(name: str, value: float) -> float:
    value = _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return value


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return int(value)


def _choice(name: str, value: str, allowed: Sequence[str]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(allowed)!r}, got {value!r}")
    return value


def _triple(name: str, value: Sequence[Any], cast: Any) -> tuple[Any, Any, Any]:
    items = tuple(value)
    if len(items) != 3:
        raise ValueError(f"{name} must have three entries, got {value!r}")
    return tuple(cast(item) for item in items)  # type: ignore[return-value]


def _vector6(name: str, value: Sequence[float]) -> tuple[float, ...]:
    items = tuple(float(item) for item in value)
    if len(items) != 6:
        raise ValueError(f"{name} must have six entries, got {value!r}")
    return items


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoxelGrid:
    """Structured voxel grid with optional periodicity per axis.

    ``shape`` is ``(nx, ny, nz)`` and ``h`` the voxel size per axis in mm.
    Voxel arrays use :attr:`array_shape` ``(nz, ny, nx)`` and node arrays
    :attr:`node_array_shape` ``(nz+1, ny+1, nx+1)``.
    """

    shape: tuple[int, int, int]
    h: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    periodic: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        shape = tuple(
            _positive_int(f"shape[{axis}]", value)
            for axis, value in enumerate(_triple("shape", self.shape, lambda v: v))
        )
        h = tuple(
            _positive(f"h[{axis}]", value)
            for axis, value in enumerate(_triple("h", self.h, float))
        )
        origin = tuple(
            _finite(f"origin[{axis}]", value)
            for axis, value in enumerate(_triple("origin", self.origin, float))
        )
        periodic = _triple("periodic", self.periodic, lambda v: v)
        for axis, flag in enumerate(periodic):
            if not isinstance(flag, (bool, np.bool_)):
                raise TypeError(f"periodic[{axis}] must be a bool, got {flag!r}")
        _set(self, "shape", shape)
        _set(self, "h", h)
        _set(self, "origin", origin)
        _set(self, "periodic", tuple(bool(flag) for flag in periodic))

    @property
    def nx(self) -> int:
        return self.shape[0]

    @property
    def ny(self) -> int:
        return self.shape[1]

    @property
    def nz(self) -> int:
        return self.shape[2]

    @property
    def n_elements(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def n_nodes_full(self) -> int:
        return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    @property
    def lengths(self) -> tuple[float, float, float]:
        """Cell edge lengths ``(Lx, Ly, Lz) = (nx hx, ny hy, nz hz)``."""

        return (self.nx * self.h[0], self.ny * self.h[1], self.nz * self.h[2])

    @property
    def volume(self) -> float:
        lx, ly, lz = self.lengths
        return lx * ly * lz

    @property
    def voxel_volume(self) -> float:
        return self.h[0] * self.h[1] * self.h[2]

    @property
    def array_shape(self) -> tuple[int, int, int]:
        """Voxel-array shape ``(nz, ny, nx)``."""

        return (self.nz, self.ny, self.nx)

    @property
    def node_array_shape(self) -> tuple[int, int, int]:
        """Full node-array shape ``(nz+1, ny+1, nx+1)``."""

        return (self.nz + 1, self.ny + 1, self.nx + 1)

    def h8_mesh(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> H8Mesh:
        """Return the matching :class:`H8Mesh` with nodes shifted by ``origin``."""

        lx, ly, lz = self.lengths
        mesh = H8Mesh.structured_block(
            length=lx,
            height=ly,
            thickness=lz,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dtype=dtype,
            device=device,
        )
        offset = torch.tensor(self.origin, dtype=dtype, device=mesh.nodes.device)
        return H8Mesh(
            nodes=mesh.nodes + offset,
            elements=mesh.elements,
            structured_shape=mesh.structured_shape,
            dimensions=mesh.dimensions,
        )

    def element_centres(self) -> np.ndarray:
        """Voxel centres ``(E, 3)`` in element-id order."""

        k, j, i = np.meshgrid(
            np.arange(self.nz, dtype=np.float64),
            np.arange(self.ny, dtype=np.float64),
            np.arange(self.nx, dtype=np.float64),
            indexing="ij",
        )
        centres = np.stack(
            (
                self.origin[0] + (i + 0.5) * self.h[0],
                self.origin[1] + (j + 0.5) * self.h[1],
                self.origin[2] + (k + 0.5) * self.h[2],
            ),
            axis=-1,
        )
        return centres.reshape(-1, 3)

    def node_coordinates(self) -> np.ndarray:
        """Full node coordinates ``(N, 3)``, equal to ``H8Mesh`` nodes plus ``origin``."""

        return self.h8_mesh(dtype=torch.float64, device="cpu").nodes.numpy().copy()

    def connectivity(self) -> np.ndarray:
        """Element connectivity ``(E, 8)`` int64 in the ``H8Mesh`` local order."""

        nx, ny, nz = self.shape
        k, j, i = np.meshgrid(
            np.arange(nz, dtype=np.int64),
            np.arange(ny, dtype=np.int64),
            np.arange(nx, dtype=np.int64),
            indexing="ij",
        )
        stride_x = nx + 1
        stride_z = (ny + 1) * stride_x
        v0 = (k * (ny + 1) + j) * stride_x + i
        v1 = v0 + 1
        v3 = v0 + stride_x
        v2 = v3 + 1
        v4 = v0 + stride_z
        v5 = v4 + 1
        v7 = v4 + stride_x
        v6 = v7 + 1
        return np.stack((v0, v1, v2, v3, v4, v5, v6, v7), axis=-1).reshape(-1, 8)


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseMaterial:
    """Constitutive and fracture data of one phase.

    ``kind='isotropic'`` needs ``E`` and ``nu``.
    ``kind='smeared_laminate'`` (the smeared interface) needs ``layers``, a tuple of
    ``(fraction, E, nu)`` isotropic layers stacked along local axis 1 (the
    voxel tangent, i.e. the interface normal), whose exact laminate stiffness
    is the phase stiffness, and ``E`` and ``nu``, an isotropic equivalent used
    only for the damage driver (Lame parameters of the spectral split).  ``Gc`` is
    the phase-field fracture energy in N/mm.  ``driver`` is the nominal damage
    driver recorded in the material table; the solver resolves it per element
    from :class:`SolverConfig` (``'none'`` is always honoured).
    ``tensile_strength`` (MPa) is the tensile strength ``f_t`` of
    the PF-CZM damage law (``SolverConfig.damage_law = 'pfczm'``); it is ``None`` with the AT2
    law.  The ``'interface'`` driver (recorded for the band when
    ``SolverConfig.energy_split_band`` is ``'interface'``) needs an isotropic or
    smeared-laminate phase.  ``sources`` maps each
    property to the inputs path or derivation it came from.
    """

    name: str
    kind: str
    Gc: float
    driver: str
    E: float | None = None
    nu: float | None = None
    sources: Mapping[str, str] = field(default_factory=dict)
    layers: tuple[tuple[float, float, float], ...] | None = None
    tensile_strength: float | None = None

    def __post_init__(self) -> None:
        _choice("phase name", self.name, PHASE_NAMES)
        if self.tensile_strength is not None:
            _set(self, "tensile_strength", _positive(f"{self.name}: tensile_strength", self.tensile_strength))
        _choice(f"{self.name}: kind", self.kind, MATERIAL_KINDS)
        _choice(f"{self.name}: driver", self.driver, DRIVER_NAMES)
        _set(self, "Gc", _positive(f"{self.name}: Gc", self.Gc))
        if self.kind == "smeared_laminate":
            if self.layers is None or len(self.layers) < 1:
                raise ValueError(f"{self.name}: smeared_laminate phases need layers")
            layers = []
            for index, layer in enumerate(self.layers):
                if len(layer) != 3:
                    raise ValueError(f"{self.name}: layer {index} must be (fraction, E, nu)")
                fraction = _positive(f"{self.name}: layer {index} fraction", layer[0])
                E_layer = _positive(f"{self.name}: layer {index} E", layer[1])
                nu_layer = _finite(f"{self.name}: layer {index} nu", layer[2])
                if not -1.0 < nu_layer < 0.5:
                    raise ValueError(f"{self.name}: layer {index} nu must lie in (-1, 0.5)")
                layers.append((fraction, E_layer, nu_layer))
            if abs(sum(layer[0] for layer in layers) - 1.0) > 1.0e-12:
                raise ValueError(f"{self.name}: layer fractions must sum to 1")
            _set(self, "layers", tuple(layers))
        elif self.layers is not None:
            raise ValueError(f"{self.name}: layers apply to smeared_laminate phases only")
        if self.E is None or self.nu is None:
            raise ValueError(f"{self.name}: {self.kind} phases need E and nu")
        _set(self, "E", _positive(f"{self.name}: E", self.E))
        nu = _finite(f"{self.name}: nu", self.nu)
        if not -1.0 < nu < 0.5:
            raise ValueError(f"{self.name}: nu must lie in (-1, 0.5), got {nu!r}")
        _set(self, "nu", nu)
        _set(self, "sources", {str(k): str(v) for k, v in dict(self.sources).items()})

    @property
    def phase_id(self) -> int:
        return PHASE_ID[self.name]

    @property
    def is_isotropic(self) -> bool:
        return self.kind == "isotropic"

    @property
    def is_laminate(self) -> bool:
        return self.kind == "smeared_laminate"



@dataclass(frozen=True)
class MaterialTable:
    """Phase materials of one scale with its uniform length scale ``ell`` (mm)."""

    phases: Mapping[str, PhaseMaterial]
    length_scale: float
    scale: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        phases = dict(self.phases)
        if not phases:
            raise ValueError("MaterialTable needs at least one phase")
        for key, material in phases.items():
            if not isinstance(material, PhaseMaterial):
                raise TypeError(f"phase {key!r} must be a PhaseMaterial")
            if key != material.name:
                raise ValueError(f"phase key {key!r} differs from material name {material.name!r}")
        ordered = {name: phases[name] for name in PHASE_NAMES if name in phases}
        _set(self, "phases", ordered)
        _set(self, "length_scale", _positive("length_scale", self.length_scale))
        _choice("scale", self.scale, ("micro",))
        _set(self, "notes", tuple(str(note) for note in self.notes))

    @property
    def has_strengths(self) -> bool:
        """Whether any phase has a PF-CZM tensile strength."""

        return any(material.tensile_strength is not None for material in self.phases.values())

    def columns(self) -> tuple[str, ...]:
        """``material_table.csv`` columns: :data:`MATERIAL_TABLE_COLUMNS`, plus
        :data:`MATERIAL_TABLE_STRENGTH_COLUMNS` when a phase has a tensile strength."""

        if self.has_strengths:
            return MATERIAL_TABLE_COLUMNS + MATERIAL_TABLE_STRENGTH_COLUMNS
        return MATERIAL_TABLE_COLUMNS

    def rows(self) -> list[dict[str, Any]]:
        """Rows with the columns of :meth:`columns`, in phase-id order.

        Without tensile strengths the columns are the frozen
        :data:`MATERIAL_TABLE_COLUMNS`, as before .
        """

        strengths = self.has_strengths
        rows: list[dict[str, Any]] = []
        for name, material in self.phases.items():
            rows.append(
                {
                    "phase": name,
                    "phase_id": PHASE_ID[name],
                    "kind": material.kind,
                    "E": material.E,
                    "nu": material.nu,
                    "Gc": material.Gc,
                    "driver": material.driver,
                    "sources": "; ".join(
                        f"{key}={value}" for key, value in material.sources.items()
                    ),
                }
            )
            if strengths:
                rows[-1]["f_t"] = material.tensile_strength
        return rows


# ---------------------------------------------------------------------------
# voxel model
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class VoxelModel:
    """Phase-labelled voxel cell (mutable).

    ``phase_id`` is ``(nz, ny, nx)`` uint8 with ids from :data:`PHASE_ID`.
    ``tangent`` is ``(nz, ny, nx, 3)`` float64; it is a unit vector in every
    smeared-laminate voxel (``(0, 0, 1)``
    elsewhere; the interface normal in interface voxels).  ``aux['interface_nodes']`` is an optional
    ``(N_full,)`` bool node mask.
    """

    grid: VoxelGrid
    phase_id: np.ndarray
    tangent: np.ndarray
    materials: MaterialTable
    metadata: dict[str, Any] = field(default_factory=dict)
    aux: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check shapes, dtypes, labels, materials and tangents; coerce arrays."""

        if not isinstance(self.grid, VoxelGrid):
            raise TypeError("grid must be a VoxelGrid")
        if not isinstance(self.materials, MaterialTable):
            raise TypeError("materials must be a MaterialTable")
        shape = self.grid.array_shape
        phase = np.asarray(self.phase_id)
        if phase.shape != shape:
            raise ValueError(f"phase_id must have shape {shape}, got {phase.shape}")
        if not np.issubdtype(phase.dtype, np.integer):
            raise TypeError(f"phase_id must be an integer array, got {phase.dtype}")
        if phase.size and (int(phase.min()) < 0 or int(phase.max()) >= len(PHASE_NAMES)):
            raise ValueError("phase_id contains an unknown phase id")
        self.phase_id = np.ascontiguousarray(phase, dtype=np.uint8)

        tangent = np.asarray(self.tangent, dtype=np.float64)
        if tangent.shape != (*shape, 3):
            raise ValueError(f"tangent must have shape {(*shape, 3)}, got {tangent.shape}")
        if not np.isfinite(tangent).all():
            raise ValueError("tangent contains non-finite values")
        self.tangent = np.ascontiguousarray(tangent)

        present = self.phases_present()
        missing = [name for name in present if name not in self.materials.phases]
        if missing:
            raise ValueError(f"phases {missing} are present but have no material")
        for name in present:
            if self.materials.phases[name].is_isotropic:
                continue
            mask = self.phase_id == PHASE_ID[name]
            norms = np.linalg.norm(self.tangent[mask], axis=-1)
            if norms.size and float(np.abs(norms - 1.0).max()) > 1.0e-8:
                raise ValueError(f"phase {name!r}: tangents must be unit vectors")

        if "interface_nodes" in self.aux:
            mask = np.asarray(self.aux["interface_nodes"])
            if mask.shape != (self.grid.n_nodes_full,) or mask.dtype != np.bool_:
                raise ValueError(
                    "aux['interface_nodes'] must be a bool array of shape "
                    f"({self.grid.n_nodes_full},)"
                )

    def phases_present(self) -> tuple[str, ...]:
        """Names of the phases present, in phase-id order."""

        ids = np.unique(np.asarray(self.phase_id))
        return tuple(PHASE_NAMES[int(index)] for index in ids)

    def volume_fractions(self) -> dict[str, float]:
        """Voxel volume fraction of every present phase."""

        counts = np.bincount(
            np.asarray(self.phase_id).reshape(-1), minlength=len(PHASE_NAMES)
        )
        total = float(counts.sum())
        return {
            name: float(counts[PHASE_ID[name]]) / total for name in self.phases_present()
        }


# ---------------------------------------------------------------------------
# geometry parameters
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# load case
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadCase:
    """Mixed macro-strain control.

    ``targets[n-1]`` is the prescribed ``E[controlled]`` after increment ``n``
    (the load path starts from zero; unloading is allowed but consecutive
    values must differ).  Components in ``free`` carry zero macro stress and
    components in ``fixed_zero`` are held at zero strain.  ``{controlled}``,
    ``free`` and ``fixed_zero`` partition ``{0, ..., 5}``.
    """

    name: str
    controlled: int
    targets: tuple[float, ...]
    free: tuple[int, ...]
    fixed_zero: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.controlled, bool) or int(self.controlled) != self.controlled:
            raise ValueError(f"controlled must be an integer, got {self.controlled!r}")
        controlled = int(self.controlled)
        free = tuple(int(v) for v in self.free)
        fixed_zero = tuple(int(v) for v in self.fixed_zero)
        components = [controlled, *free, *fixed_zero]
        if sorted(components) != list(range(6)):
            raise ValueError(
                "{controlled}, free and fixed_zero must partition {0,...,5}; got "
                f"controlled={controlled}, free={free}, fixed_zero={fixed_zero}"
            )
        targets = tuple(float(v) for v in self.targets)
        if not targets:
            raise ValueError("a LoadCase needs at least one target")
        previous = 0.0
        for index, target in enumerate(targets):
            if not math.isfinite(target):
                raise ValueError(f"target {index} is not finite: {target!r}")
            if target == previous:
                raise ValueError(f"target {index} repeats the previous value {target!r}")
            previous = target
        _set(self, "name", str(self.name))
        _set(self, "controlled", controlled)
        _set(self, "targets", targets)
        _set(self, "free", free)
        _set(self, "fixed_zero", fixed_zero)

    @classmethod
    def from_schedule(
        cls,
        name: str,
        *,
        controlled: int,
        schedule: Sequence[tuple[float, float]],
        free: Sequence[int],
        fixed_zero: Sequence[int],
    ) -> "LoadCase":
        """Build targets from ``[(upto, inc), ...]`` segments starting at zero.

        Each segment is split into ``round(|upto - start| / |inc|)`` equal steps
        and lands exactly on ``upto``.
        """

        targets: list[float] = []
        start = 0.0
        for index, segment in enumerate(schedule):
            upto, inc = (float(v) for v in segment)
            if not (math.isfinite(upto) and math.isfinite(inc)) or inc == 0.0:
                raise ValueError(f"schedule segment {index} is invalid: {segment!r}")
            n_steps = int(round(abs(upto - start) / abs(inc)))
            if n_steps < 1:
                raise ValueError(
                    f"schedule segment {index} ({segment!r}) has no step from {start!r}"
                )
            span = upto - start
            for step in range(1, n_steps):
                targets.append(start + span * (step / n_steps))
            targets.append(upto)
            start = upto
        return cls(
            name=name,
            controlled=controlled,
            targets=tuple(targets),
            free=tuple(free),
            fixed_zero=tuple(fixed_zero),
        )

    @property
    def n_increments(self) -> int:
        return len(self.targets)

    def increments(self) -> tuple[float, ...]:
        """Per-increment changes ``targets[n] - targets[n-1]`` (from zero)."""

        previous = (0.0, *self.targets[:-1])
        return tuple(target - start for start, target in zip(previous, self.targets))


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------


def _parse_device(name: str | torch.device) -> torch.device:
    try:
        return torch.device(name)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"device must be cpu or cuda, got {name!r}") from exc


def _check_device_name(name: str) -> str:
    device = _parse_device(name)
    if device.type == "mps":
        raise ValueError("MPS is refused: the 3D voxel solver runs in float64 on CPU or CUDA")
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"device must be cpu or cuda, got {name!r}")
    return str(name)


@dataclass(frozen=True)
class SolverConfig:
    """Numerical settings of the staggered phase-field solve.

    ``kappa`` is the residual stiffness of the degradation law.  The default
    is a numerical placeholder; the runner sets it from the inputs file
    (``micro_phase_field.residual_stiffness`` or
    ``meso_phase_field.residual_stiffness``).  ``damage_kkt_tol`` applies to
    the scaled damage system.

    ``energy_split_band`` selects the damage driver of the ``bn_band`` phase:
    ``'isotropic'`` (default) follows ``energy_split_isotropic`` (the
    isotropic-equivalent spectral or full driver, hybrid stress ``g C eps``);
    ``'interface'`` uses the directional interface driver
    (:data:`DRIVER_INTERFACE`, ): the band is driven by its interface
    traction energy ``psi_plus`` only.

    ``band_stress`` selects the stress of the interface-driven
    band: ``'hybrid'`` (default) is ``g C eps``, the formulation of every
    other phase, and ``'variational'`` the derivative of
    ``psi0 - (1 - g) psi_plus`` (kept for comparison).
    It applies only with ``energy_split_band = 'interface'``;
    ``'variational'`` is refused otherwise.  Earlier versions always used
    the variational stress in the interface band.  The variational band never degrades
    the in-plane energy ``psi_rest``, so it keeps carrying stress along the
    interface plane at any damage and acts as an unbreakable in-plane
    ligament.  The hybrid band degrades its whole stress by
    ``g(d)``: in-plane stretching does not drive band damage, but band damage
    caused by its neighbours (through the gradient term) removes the in-plane
    stiffness, the normal closure and the shear stiffness together.  It is
    not variationally consistent, so the energy balance is a diagnostic as
    for the other phases, and a crack crossing the band costs only the band
    ``Gc``.

    ``anderson_depth`` switches on Anderson mixing of the
    staggered damage fixed point with that many residual differences
    (0, the default, keeps plain alternate minimisation);
    ``anderson_start`` is the number of plain staggered iterations of each
    attempt before the first mixed iterate (at least 2, because a mix needs
    two residuals).

    ``crack_closure`` restores the compressive stiffness of
    damaged material: ``'none'`` (default) keeps the former stress laws;
    ``'axes'`` adds back ``(1 - g)`` of a fixed-axis compressive energy (cell
    axes in the isotropic phases and the band frame in the smeared-laminate
    interface, see :mod:`.splits`).  It is refused with Anderson mixing.

    ``damage_law`` selects the damage law: ``'at2'`` (default,
    the hybrid AT2 law of every earlier run) or ``'pfczm'``, Wu's
    phase-field cohesive zone model (Wu 2017, J. Mech. Phys. Solids 103,
    72-99, doi 10.1016/j.jmps.2017.03.015) with the crack geometric function
    ``alpha(d) = 2d - d^2``, ``c_alpha = pi`` and the rational degradation of
    :func:`phast.physics.material.pfczm_degradation_derivatives`, so that the
    tensile strength ``f_t`` of each phase (``PhaseMaterial.tensile_strength``)
    is an input and the peak stress no longer depends on ``ell``.
    ``pfczm_softening`` (``'linear'`` default, ``'exponential'``),
    ``pfczm_p`` and ``pfczm_max_newton`` (the most projected Newton
    iterations of one damage solve) apply with ``'pfczm'`` only and must
    keep their defaults with ``'at2'``.  ``pfczm_p`` is the exponent of Wu's
    softening law and must match it: 2 (default) for linear softening, 2.5
    for exponential softening (Wu and Nguyen 2018, J. Mech. Phys. Solids 119,
    20-42, doi 10.1016/j.jmps.2018.06.006).  ``'pfczm'`` is refused with
    ``irreversibility = 'history_only'`` (``alpha`` is unbounded below for
    ``d < 0``).  Anderson mixing is available with ``'pfczm'`` where it is
    available with AT2 (not with crack closure).
    """

    device: str = "cpu"
    dtype: str = "float64"
    num_threads: int | None = None
    batch_size: int = 65536
    kappa: float = 1.0e-6
    stagger_tol: float = 1.0e-4
    max_stagger: int = 2000
    min_stagger: int = 2
    pcg_rtol_mech: float = 1.0e-8
    pcg_maxit_mech: int = 20000
    preconditioner: str = "jacobi"
    check_every: int = 10
    damage_linear_tol: float = 1.0e-10
    damage_kkt_tol: float = 1.0e-8
    damage_maxit: int = 4000
    damage_max_active_set: int = 64
    irreversibility: str = "box_kkt"
    history: str = "max"
    energy_split_isotropic: str = "spectral"
    on_stagger_failure: str = "cutback"
    max_cutbacks: int = 1
    final_mechanics_resolve: bool = True
    damaged_threshold: float = 0.95
    energy_split_band: str = "isotropic"
    band_stress: str = "hybrid"
    anderson_depth: int = 0
    anderson_start: int = 2
    crack_closure: str = "none"
    damage_law: str = "at2"
    pfczm_softening: str = "linear"
    pfczm_p: float = 2
    pfczm_max_newton: int = 100

    def __post_init__(self) -> None:
        _check_device_name(self.device)
        solver_dtype(self.dtype)
        if self.num_threads is not None:
            _positive_int("num_threads", self.num_threads)
        for key in (
            "batch_size",
            "max_stagger",
            "pcg_maxit_mech",
            "check_every",
            "damage_maxit",
            "damage_max_active_set",
        ):
            _set(self, key, _positive_int(key, getattr(self, key)))
        if isinstance(self.min_stagger, bool) or int(self.min_stagger) != self.min_stagger or self.min_stagger < 0:
            raise ValueError(f"min_stagger must be an integer >= 0, got {self.min_stagger!r}")
        if self.min_stagger > self.max_stagger:
            raise ValueError("min_stagger must not exceed max_stagger")
        if isinstance(self.max_cutbacks, bool) or int(self.max_cutbacks) != self.max_cutbacks or self.max_cutbacks < 0:
            raise ValueError(f"max_cutbacks must be an integer >= 0, got {self.max_cutbacks!r}")
        kappa = _finite("kappa", self.kappa)
        if not 0.0 <= kappa < 1.0:
            raise ValueError(f"kappa must lie in [0, 1), got {kappa!r}")
        _set(self, "kappa", kappa)
        for key in ("stagger_tol", "pcg_rtol_mech", "damage_linear_tol", "damage_kkt_tol"):
            _set(self, key, _positive(key, getattr(self, key)))
        threshold = _finite("damaged_threshold", self.damaged_threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError("damaged_threshold must lie in (0, 1]")
        _set(self, "damaged_threshold", threshold)
        _choice("preconditioner", self.preconditioner, ("jacobi",))
        _choice("irreversibility", self.irreversibility, ("box_kkt", "history_only"))
        _choice("history", self.history, ("max", "current"))
        _choice("energy_split_isotropic", self.energy_split_isotropic, ("spectral", "full"))
        _choice("energy_split_band", self.energy_split_band, BAND_SPLITS)
        _choice("band_stress", self.band_stress, BAND_STRESSES)
        if self.band_stress == "variational" and self.energy_split_band != "interface":
            raise ValueError(
                "band_stress 'variational' needs energy_split_band 'interface' (the variational band "
                "stress is the derivative of the interface-split energy, )"
            )
        _choice("on_stagger_failure", self.on_stagger_failure, ("cutback", "raise"))
        _set(self, "final_mechanics_resolve", bool(self.final_mechanics_resolve))
        depth = self.anderson_depth
        if isinstance(depth, bool) or int(depth) != depth or int(depth) < 0:
            raise ValueError(f"anderson_depth must be an integer >= 0, got {depth!r}")
        _set(self, "anderson_depth", int(depth))
        start = self.anderson_start
        if isinstance(start, bool) or int(start) != start or int(start) < 2:
            raise ValueError(
                f"anderson_start must be an integer >= 2 (a mix needs two staggered residuals), got {start!r}"
            )
        _set(self, "anderson_start", int(start))
        if self.crack_closure in _DEFERRED_CRACK_CLOSURES:
            raise ValueError(
                f"crack_closure must be one of {CRACK_CLOSURES}: "
                f"{_DEFERRED_CRACK_CLOSURES[self.crack_closure]}"
            )
        _choice("crack_closure", self.crack_closure, CRACK_CLOSURES)
        if self.crack_closure != "none" and self.anderson_depth > 0:
            raise ValueError(
                "crack_closure 'axes' is not available with Anderson mixing (anderson_depth > 0): the "
                "closure makes the mechanics an active-set solve whose energy line search the mixing "
                "safeguard does not cover"
            )
        _choice("damage_law", self.damage_law, DAMAGE_LAWS)
        _choice("pfczm_softening", self.pfczm_softening, PFCZM_SOFTENINGS)
        p = self.pfczm_p
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 2:
            raise ValueError(f"pfczm_p must be a number >= 2, got {p!r}")
        # an integral p is stored as an int, so a linear-softening record reads 2 (not 2.0)
        p = int(p) if float(p).is_integer() else float(p)
        _set(self, "pfczm_p", p)
        _set(self, "pfczm_max_newton", _positive_int("pfczm_max_newton", self.pfczm_max_newton))
        if self.damage_law == "at2":
            changed = [key for key in PFCZM_SETTINGS if getattr(self, key) != getattr(SolverConfig, key)]
            if changed:
                raise ValueError(f"{', '.join(changed)} apply to damage_law 'pfczm' only")
        else:
            if self.irreversibility != "box_kkt":
                raise ValueError(
                    "damage_law 'pfczm' needs irreversibility 'box_kkt': the PF-CZM crack function "
                    "alpha(d) = 2d - d^2 is unbounded below for d < 0, so the damage solve needs the box"
                )
            wu_p = PFCZM_SOFTENING_EXPONENTS[self.pfczm_softening]
            if p != wu_p:
                raise ValueError(
                    f"pfczm_softening {self.pfczm_softening!r} is Wu's law with pfczm_p = {wu_p}, got {p!r}: "
                    "the exponent belongs to the softening law (linear: p = 2, a2 = -1/2; exponential: "
                    "p = 2.5, a2 = 2^(5/3) - 3), and another p gives a different softening curve"
                )


OPT_IN_SOLVER_KEYS: tuple[str, ...] = (
    "energy_split_band",
    "band_stress",
    "anderson_depth",
    "anderson_start",
    "crack_closure",
    "damage_law",
    *PFCZM_SETTINGS,
)
""":class:`SolverConfig` options added after the first runs.

Their defaults keep the former behaviour, so records list them only when they
are set: ``energy_split_band`` when it is not ``'isotropic'``, ``band_stress``
whenever ``energy_split_band`` is ``'interface'`` (both values, so that every
interface record names its band stress; a record written before with
``energy_split_band: interface`` and no ``band_stress`` ran the variational
band stress), the two Anderson settings when either differs from its
default, ``crack_closure`` when it is not ``'none'``, and ``damage_law`` with
the PF-CZM settings when ``damage_law`` is not ``'at2'`` (all of them then).
"""


def solver_settings_record(solver: SolverConfig) -> dict[str, Any]:
    """``dataclasses.asdict(solver)`` without the opt-in settings left at their defaults.

    Used for the resolved parameters and the Zarr configuration, so that runs
    with the defaults record the settings exactly as before the options
    existed.
    """

    record = asdict(solver)
    if solver.energy_split_band == SolverConfig.energy_split_band:
        record.pop("energy_split_band")
    if solver.energy_split_band != "interface":
        record.pop("band_stress")
    if (solver.anderson_depth, solver.anderson_start) == (SolverConfig.anderson_depth, SolverConfig.anderson_start):
        record.pop("anderson_depth")
        record.pop("anderson_start")
    if solver.crack_closure == SolverConfig.crack_closure:
        record.pop("crack_closure")
    if solver.damage_law == SolverConfig.damage_law:
        for key in ("damage_law", *PFCZM_SETTINGS):
            record.pop(key)
    return record


@dataclass(frozen=True)
class OutputConfig:
    """Output cadence and format settings for one run directory."""

    run_id: str
    run_dir: Path
    vtu_every: int = 5
    zarr_every: int = 5
    frame_every: int = 2
    checkpoint_every: int = 10
    write_vtu: bool = True
    # Zarr is the legacy field store: new runs write ``result.vtkhdf`` (the result-file launcher) or HDF5, and the
    # storage policy refuses Zarr output.  A deck that needs it for a retained run sets ``write_zarr: true``.
    write_zarr: bool = False
    vtu_compression: str | None = "zlib"
    isovalue: float = 0.95
    figure_dpi: int = 200
    gif_fps: int = 10
    mp4_fps: int = 12
    gif_max_bytes: int = 10_000_000
    allow_cloud_path: bool = False
    # further outputs, all off by default so that default runs write
    # exactly the former files (--extra-outputs on switches the first four on)
    derived_fields: bool = False
    phase_history: bool = False
    crack_surfaces: bool = False
    adaptive_snapshots: bool = False
    adaptive_drop: float = 0.02
    adaptive_damage_rise: float = 0.2
    dump_stagger_iterates_at: tuple[int, ...] = ()
    # Number of rolling checkpoints kept (1 keeps one ``checkpoint.pt``).
    checkpoint_keep: int = 1
    # Checkpoint schedule: ``none`` keeps the rolling file only; ``equidistant_sqrtN`` retains a checkpoint every ceil(sqrt(N)) increments of the N planned ones, and the last;
    # ``at_steps`` retains exactly ``checkpoint_steps``.
    checkpoint_schedule: str = "none"
    checkpoint_steps: tuple[int, ...] = ()
    # Cadence of the derived cell arrays in the result file: 1 (default) writes stress, strain and the energy densities at every stored step; N > 1 writes them at every N-th increment only.
    derived_every: int = 1

    def __post_init__(self) -> None:
        if not str(self.run_id):
            raise ValueError("run_id must be a non-empty string")
        _set(self, "run_id", str(self.run_id))
        _set(self, "run_dir", Path(self.run_dir).expanduser())
        for key in (
            "vtu_every",
            "zarr_every",
            "frame_every",
            "checkpoint_every",
            "checkpoint_keep",
            "derived_every",
            "figure_dpi",
            "gif_fps",
            "mp4_fps",
            "gif_max_bytes",
        ):
            _set(self, key, _positive_int(key, getattr(self, key)))
        if self.vtu_compression not in ("zlib", "lzma", None):
            raise ValueError("vtu_compression must be 'zlib', 'lzma' or None")
        if self.checkpoint_schedule not in CHECKPOINT_SCHEDULES:
            raise ValueError(f"checkpoint_schedule must be one of {CHECKPOINT_SCHEDULES}, got {self.checkpoint_schedule!r}")
        steps = tuple(sorted({_positive_int("checkpoint_steps", v) for v in self.checkpoint_steps}))
        if self.checkpoint_schedule == "at_steps" and not steps:
            raise ValueError("checkpoint_schedule at_steps needs a non-empty checkpoint_steps list")
        if steps and self.checkpoint_schedule != "at_steps":
            raise ValueError("checkpoint_steps is used with checkpoint_schedule at_steps only")
        _set(self, "checkpoint_steps", steps)
        isovalue = _finite("isovalue", self.isovalue)
        if not 0.0 < isovalue < 1.0:
            raise ValueError("isovalue must lie in (0, 1)")
        _set(self, "isovalue", isovalue)
        for key in (
            "write_vtu",
            "write_zarr",
            "allow_cloud_path",
            "derived_fields",
            "phase_history",
            "crack_surfaces",
            "adaptive_snapshots",
        ):
            _set(self, key, bool(getattr(self, key)))
        drop = _finite("adaptive_drop", self.adaptive_drop)
        if not 0.0 < drop < 1.0:
            raise ValueError("adaptive_drop must lie in (0, 1) (a relative fall of the controlled macro stress)")
        _set(self, "adaptive_drop", drop)
        rise = _finite("adaptive_damage_rise", self.adaptive_damage_rise)
        if not 0.0 < rise <= 1.0:
            raise ValueError("adaptive_damage_rise must lie in (0, 1] (a rise of the largest nodal damage)")
        _set(self, "adaptive_damage_rise", rise)
        steps = self.dump_stagger_iterates_at
        if steps is None:
            steps = ()
        if isinstance(steps, (str, bytes)) or not isinstance(steps, (list, tuple, set, frozenset)):
            raise ValueError("dump_stagger_iterates_at must be a list of step numbers")
        checked = []
        for value in steps:
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"dump_stagger_iterates_at entries must be integers >= 1, got {value!r}")
            checked.append(int(value))
        _set(self, "dump_stagger_iterates_at", tuple(sorted(set(checked))))


OPT_IN_OUTPUT_KEYS: tuple[str, ...] = (
    "derived_fields",
    "phase_history",
    "crack_surfaces",
    "adaptive_snapshots",
    "adaptive_drop",
    "adaptive_damage_rise",
    "dump_stagger_iterates_at",
)
""":class:`OutputConfig` options added after the first runs.

Their defaults write exactly the former outputs, so records list them only
when one of them differs from its default (:func:`output_settings_record`).
"""

CHECKPOINT_SCHEDULES: tuple[str, ...] = ("none", "equidistant_sqrtN", "at_steps")
"""Values of :attr:`OutputConfig.checkpoint_schedule`."""


def planned_increments(n_increments: int, max_increments: int | None) -> int:
    """The increments a run will reach: the load path's count, cut by ``max_increments`` when that is set."""

    n = int(n_increments)
    return n if max_increments is None else min(n, int(max_increments))


def scheduled_checkpoint_steps(output: "OutputConfig", n_increments: int) -> frozenset[int]:
    """The increments whose accepted state the run saves and retains as a checkpoint under the schedule.

    ``equidistant_sqrtN``: every ``ceil(sqrt(N))`` increments and the last, N = ``n_increments`` (pass
    :func:`planned_increments` so that a run cut by ``max_increments`` keeps its real last increment).  ``at_steps``: the
    listed steps (a step beyond N is kept in the set and simply never reached).  ``none``: the empty set.
    """

    if output.checkpoint_schedule == "none":
        return frozenset()
    if output.checkpoint_schedule == "at_steps":
        return frozenset(int(v) for v in output.checkpoint_steps)
    n = int(n_increments)
    if n < 1:
        return frozenset()
    stride = math.isqrt(n)
    if stride * stride < n:
        stride += 1
    return frozenset(range(stride, n + 1, stride)) | {n}


EXAMPLE_OUTPUT_KEYS: tuple[str, ...] = ("derived_fields", "phase_history", "crack_surfaces", "adaptive_snapshots")
"""Opt-in output options that ``--extra-outputs on`` switches on together; off by default."""


def output_settings_record(output: "OutputConfig") -> dict[str, Any]:
    """``dataclasses.asdict(output)`` without the opt-in options while all of them are at their defaults.

    Used for the resolved parameters and the Zarr configuration, so that runs
    with the default outputs record the settings exactly as before the
    options existed.  As soon as one option is set, all of them are listed.
    """

    record = asdict(output)
    if output.checkpoint_keep == OutputConfig.checkpoint_keep:
        record.pop("checkpoint_keep")  # its own check, not in OPT_IN_OUTPUT_KEYS: setting another opt-in key must not list it
    if output.derived_every == OutputConfig.derived_every:
        record.pop("derived_every")
    if output.checkpoint_schedule == OutputConfig.checkpoint_schedule:
        record.pop("checkpoint_schedule"); record.pop("checkpoint_steps")   # listed only when a schedule is set
    else:
        record["checkpoint_steps"] = list(output.checkpoint_steps)
    if all(getattr(output, key) == getattr(OutputConfig, key) for key in OPT_IN_OUTPUT_KEYS):
        for key in OPT_IN_OUTPUT_KEYS:
            record.pop(key)
    else:
        record["dump_stagger_iterates_at"] = list(output.dump_stagger_iterates_at)
    return record


# ---------------------------------------------------------------------------
# solver data containers
# ---------------------------------------------------------------------------


_ELEMENT_FIELDS: tuple[str, ...] = (
    "Gc",
    "driver",
    "lam",
    "mu",
    "phase",
)


@dataclass(frozen=True, eq=False)
class ElementMaterialArrays:
    """Per-element material data as torch tensors on the solver device.

    ``stiffness_bank`` ``(K, 6, 6)`` holds global-frame engineering
    stiffnesses; ``stiffness_index`` ``(E,)`` long selects one per element.
    Isotropic entries (``lam``, ``mu``) are zero where they do not apply.
    """

    stiffness_bank: torch.Tensor
    stiffness_index: torch.Tensor
    Gc: torch.Tensor
    driver: torch.Tensor
    lam: torch.Tensor
    mu: torch.Tensor
    phase: torch.Tensor
    tangent: torch.Tensor

    def __post_init__(self) -> None:
        bank = self.stiffness_bank
        if bank.ndim != 3 or bank.shape[1:] != (6, 6):
            raise ValueError(f"stiffness_bank must be (K, 6, 6), got {tuple(bank.shape)}")
        if bank.dtype != torch.float64:
            raise TypeError("stiffness_bank must be float64")
        n_elements = int(self.stiffness_index.shape[0])
        if self.stiffness_index.ndim != 1 or self.stiffness_index.dtype != torch.long:
            raise ValueError("stiffness_index must be a 1D long tensor")
        if n_elements and (
            int(self.stiffness_index.min()) < 0 or int(self.stiffness_index.max()) >= bank.shape[0]
        ):
            raise ValueError("stiffness_index points outside the stiffness bank")
        for name in _ELEMENT_FIELDS:
            value = getattr(self, name)
            if tuple(value.shape) != (n_elements,):
                raise ValueError(f"{name} must have shape ({n_elements},), got {tuple(value.shape)}")
            if value.device != bank.device:
                raise ValueError(f"{name} must live on {bank.device}")
        if self.driver.dtype != torch.int8:
            raise TypeError("driver must be int8")
        if self.phase.dtype != torch.long:
            raise TypeError("phase must be long")
        if tuple(self.tangent.shape) != (n_elements, 3):
            raise ValueError(f"tangent must have shape ({n_elements}, 3)")

    @property
    def n_elements(self) -> int:
        return int(self.stiffness_index.shape[0])

    @property
    def n_bank(self) -> int:
        return int(self.stiffness_bank.shape[0])

    @property
    def device(self) -> torch.device:
        return self.stiffness_bank.device

    def numpy(self) -> dict[str, np.ndarray]:
        """All fields as CPU numpy arrays."""

        names = ("stiffness_bank", "stiffness_index", *_ELEMENT_FIELDS, "tangent")
        return {name: getattr(self, name).detach().cpu().numpy() for name in names}


@dataclass(frozen=True, eq=False)
class ClosurePattern:
    """Sign pattern of the crack-closure terms at every quadrature point.

    ``iso`` ``(E, 8, 4)`` bool holds the bits ``[tr eps < 0], [eps_xx < 0], [eps_yy < 0], [eps_zz < 0]`` of the isotropic closure (rows of
    elements without it are False); ``band`` ``(n_cb, 8, 4)`` the bits ``[t11 < 0], [p2 + p3 < 0], [p2 < 0], [p3 < 0]`` of the
    smeared-laminate interface elements. A group that the model does not have is ``None``. For a fixed pattern the closure stress is linear
    in the strain.
    """

    iso: torch.Tensor | None = None
    band: torch.Tensor | None = None

    def groups(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return (self.iso, self.band)

    def equals(self, other: "ClosurePattern") -> bool:
        """Whether both patterns hold the same bits in every group."""

        for mine, theirs in zip(self.groups(), other.groups()):
            if (mine is None) != (theirs is None):
                return False
            if mine is not None and not torch.equal(mine, theirs):
                return False
        return True


CLOSURE_RECORD_FIELDS: tuple[str, ...] = ("closure_solves", "min_normal_strain", "min_principal_strain", "min_det_F")
""":class:`IncrementRecord` fields recorded with ``crack_closure = 'axes'``: the linear
mechanics solves of the closure pattern iterations of the increment and the kinematic guard of the
accepted state (smallest element-mean normal and principal strain, smallest quadrature-point
``det(I + grad u)``)."""


@dataclass(frozen=True)
class IncrementRecord:
    """Scalar record of one accepted load increment (``step = 0`` is the start).

    Energies are in mJ, stresses in MPa, ``applied_disp`` in mm and
    ``reaction_force`` in N.  ``step`` counts accepted solutions and
    ``increment`` the load-case increment the step belongs to.
    ``macro_strain`` holds the affine macro unknowns ``E``; ``mean_strain`` is
    the volume-average total strain ``<eps>`` (it differs from ``E`` in the
    components carried by the fluctuation across free faces, e.g. ``zz``,
    ``yz`` and ``xz`` with free z faces) and defaults to ``macro_strain``,
    which is exact for a triply periodic cell.

    ``anderson_accepted`` and ``anderson_rejected`` count the mixed damage
    iterates of the increment that the safeguard kept or replaced by the
    plain iterate; they are ``None`` when Anderson mixing is off.

    With ``crack_closure = 'axes'`` ``closure_solves`` counts the
    linear mechanics solves of the closure pattern iterations of the
    increment, and ``min_normal_strain`` (smallest element-mean ``eps_xx,
    eps_yy, eps_zz``), ``min_principal_strain`` (smallest principal value of
    the element-mean strains) and ``min_det_F`` (smallest quadrature-point
    ``det(I + grad u)``) describe the accepted state; they are ``None``
    otherwise (:data:`CLOSURE_RECORD_FIELDS`).

    With ``damage_law = 'pfczm'`` ``newton_iters_pf`` counts the
    projected Newton iterations of every damage solve of the increment; ``pcg_iters_pf`` and
    ``active_set_iters`` then count the Krylov and active-set iterations of
    their box-constrained subproblems.  It is ``None`` with the AT2 law.
    """

    step: int
    increment: int = 0
    load_factor: float = 0.0
    macro_strain: tuple[float, ...] = (0.0,) * 6
    macro_stress: tuple[float, ...] = (0.0,) * 6
    applied_disp: float = 0.0
    reaction_force: float = 0.0
    elastic: float = 0.0
    fracture: float = 0.0
    fracture_by_phase: Mapping[str, float] = field(default_factory=dict)
    external: float = 0.0
    balance_residual: float = 0.0
    balance_residual_rel: float = 0.0
    max_damage: float = 0.0
    mean_damage: float = 0.0
    damaged_volume_fraction: float = 0.0
    max_H: float = 0.0
    stagger_iters: int = 0
    pcg_iters_mech: int = 0
    pcg_iters_pf: int = 0
    active_set_iters: int = 0
    cutbacks: int = 0
    converged: bool = True
    kkt_ok: bool = True
    stagger_change: float = 0.0
    mech_relative_residual: float = 0.0
    hill_mandel_rel: float = 0.0
    wall_s: float = 0.0
    mech_s: float = 0.0
    damage_s: float = 0.0
    history_s: float = 0.0
    mean_strain: tuple[float, ...] | None = None
    anderson_accepted: int | None = None
    anderson_rejected: int | None = None
    closure_solves: int | None = None
    min_normal_strain: float | None = None
    min_principal_strain: float | None = None
    min_det_F: float | None = None
    newton_iters_pf: int | None = None

    def __post_init__(self) -> None:
        _set(self, "step", int(self.step))
        if self.newton_iters_pf is not None:
            _set(self, "newton_iters_pf", int(self.newton_iters_pf))
        _set(self, "increment", int(self.increment))
        if self.closure_solves is not None:
            _set(self, "closure_solves", int(self.closure_solves))
        for name in CLOSURE_RECORD_FIELDS[1:]:
            if getattr(self, name) is not None:
                _set(self, name, float(getattr(self, name)))
        for name in ("anderson_accepted", "anderson_rejected"):
            if getattr(self, name) is not None:
                _set(self, name, int(getattr(self, name)))
        _set(self, "macro_strain", _vector6("macro_strain", self.macro_strain))
        _set(self, "macro_stress", _vector6("macro_stress", self.macro_stress))
        mean = self.macro_strain if self.mean_strain is None else self.mean_strain
        _set(self, "mean_strain", _vector6("mean_strain", mean))
        by_phase = {str(k): float(v) for k, v in dict(self.fracture_by_phase).items()}
        unknown = [name for name in by_phase if name not in PHASE_ID]
        if unknown:
            raise ValueError(f"fracture_by_phase has unknown phases {unknown}")
        _set(self, "fracture_by_phase", by_phase)


def _as_float_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


SNAPSHOT_CLOSURE_FIELDS: tuple[str, ...] = ("psi_closure", "closure_active", "det_F_min")
"""Optional :class:`FieldSnapshot` element fields of the crack closure."""


@dataclass(frozen=True, eq=False)
class FieldSnapshot:
    """CPU numpy fields at one step.

    Nodal ``(N, ...)``: total displacement ``u = E_tensor (x - origin) + P u~``,
    the periodic ``fluctuation`` and ``damage``.  Element ``(E, ...)``
    quadrature-point means: ``strain``, ``stress``, ``history_mean``,
    ``psi_elastic`` (mean of ``g psi0``) and ``psi_driving`` (mean of ``Y``);
    ``history_max`` is the quadrature-point maximum.

    With ``crack_closure = 'axes'`` ``psi_closure`` is the
    quadrature-point mean of the closure add-back energy ``(1 - g) psi_c``
    so that it is
    part of ``psi_elastic``; ``closure_active`` the fraction of the
    quadrature points at which a closure term carries stress; and
    ``det_F_min`` the smallest quadrature-point ``det(I + grad u)``.  All
    three are ``None`` otherwise.
    """

    step: int
    time_value: float
    macro_strain: np.ndarray
    macro_stress: np.ndarray
    displacement: np.ndarray
    fluctuation: np.ndarray
    damage: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    history_mean: np.ndarray
    history_max: np.ndarray
    psi_elastic: np.ndarray
    psi_driving: np.ndarray
    psi_closure: np.ndarray | None = None
    closure_active: np.ndarray | None = None
    det_F_min: np.ndarray | None = None

    def __post_init__(self) -> None:
        _set(self, "step", int(self.step))
        _set(self, "time_value", float(self.time_value))
        _set(self, "macro_strain", _as_float_array("macro_strain", self.macro_strain, (6,)))
        _set(self, "macro_stress", _as_float_array("macro_stress", self.macro_stress, (6,)))
        damage = np.asarray(self.damage, dtype=np.float64)
        if damage.ndim != 1:
            raise ValueError("damage must be a nodal (N,) array")
        n_nodes = damage.shape[0]
        _set(self, "damage", damage)
        for name in ("displacement", "fluctuation"):
            _set(self, name, _as_float_array(name, getattr(self, name), (n_nodes, 3)))
        strain = np.asarray(self.strain, dtype=np.float64)
        if strain.ndim != 2 or strain.shape[1] != 6:
            raise ValueError("strain must be an element (E, 6) array")
        n_elements = strain.shape[0]
        _set(self, "strain", strain)
        _set(self, "stress", _as_float_array("stress", self.stress, (n_elements, 6)))
        for name in ("history_mean", "history_max", "psi_elastic", "psi_driving"):
            _set(self, name, _as_float_array(name, getattr(self, name), (n_elements,)))
        closure = [getattr(self, name) is None for name in SNAPSHOT_CLOSURE_FIELDS]
        if any(closure) and not all(closure):
            raise ValueError(f"the closure fields {SNAPSHOT_CLOSURE_FIELDS} are given together or not at all")
        if self.psi_closure is not None:
            for name in SNAPSHOT_CLOSURE_FIELDS:
                _set(self, name, _as_float_array(name, getattr(self, name), (n_elements,)))

    @property
    def has_closure(self) -> bool:
        """Whether the snapshot carries the crack-closure fields."""

        return self.psi_closure is not None

    @property
    def n_nodes(self) -> int:
        return int(self.damage.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.strain.shape[0])


ANIMATION_SLICE_KEYS: tuple[str, ...] = ("xy", "xz", "yz")
"""Nodal mid-plane slices: 'xy' at k = nz//2, 'xz' at j = ny//2, 'yz' at i = nx//2."""


# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------


def resolve_device(name: str | torch.device) -> torch.device:
    """Return a CPU or CUDA device; refuse MPS and unavailable CUDA."""

    device = _parse_device(name)
    if device.type == "mps":
        raise ValueError("MPS is refused: the 3D voxel solver runs in float64 on CPU or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {str(name)!r} requested but CUDA is not available")
        return device
    if device.type != "cpu":
        raise ValueError(f"device must be cpu or cuda, got {str(name)!r}")
    return device


def solver_dtype(name: str | torch.dtype) -> torch.dtype:
    """Return ``torch.float64``; every other dtype is refused."""

    if name == "float64" or name is torch.float64:
        return torch.float64
    raise ValueError(f"only float64 is supported, got {name!r}")


def configure_threads(n: int | None) -> int:
    """Set the torch intra-op thread count when ``n`` is given; return the count."""

    if n is not None:
        torch.set_num_threads(_positive_int("num_threads", n))
    return int(torch.get_num_threads())
