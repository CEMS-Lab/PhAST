"""Stiffness, rotation, bounds and fracture-energy formulas for periodic voxel cells.

All stiffnesses are 6x6 engineering-Voigt matrices in the order
``(xx, yy, zz, yz, xz, xy)``: they map engineering strain to stress.  Mandel
form is used only for eigenvalue and bound checks.  Smeared
laminates (the interface) stack isotropic layers along local axis 1, the
interface normal carried by the voxel tangent.

Fracture-energy helpers use mm, MPa and N/mm.  The AT2 relations are

* ``sigma_c = sqrt(27 E Gc / (256 ell))`` (homogeneous-state strength);
* ``Gc = 256 sigma_c^2 ell / (27 E)``;
* ``ell = 27 E Gc / (256 sigma_c^2)``.

The smeared BN band relation (1D variational AT2, ``Gc(x)`` inside the
surface-energy integral, crack centred in a band of width ``w`` between two
neighbours ``j``) is

``G_eff = (Gc_b/2) sum_j (T Gc_b + Gc_j) / (Gc_b + T Gc_j)``,
``T = tanh(w / (2 ell))``,

which tends to ``Gc_b`` for ``w -> inf`` and to the neighbour mean for
``w -> 0``.  It is a starting value only; the band energy is calibrated
numerically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
import torch

from phast.core.voigt import (
    engineering_strain_to_tensor,
    engineering_stress_to_tensor,
    engineering_to_mandel_stiffness,
    tensor_to_engineering_strain,
    tensor_to_engineering_stress,
)
from phast.core.hex8 import isotropic_stiffness_3d
from phast.solid3d.model import (
    DRIVER_FULL,
    DRIVER_INTERFACE,
    DRIVER_NONE,
    DRIVER_SPECTRAL,
    PHASE_ID,
    PHASE_NAMES,
    ElementMaterialArrays,
    PhaseMaterial,
    SolverConfig,
    VoxelModel,
    resolve_device,
    solver_dtype,
)


_DTYPE = torch.float64

_AT2_NUM = 27.0
_AT2_DEN = 256.0
"""AT2 homogeneous-state strength fractions ``27/256``."""

_MPA_M_TO_N_PER_MM = 1.0e3
"""Unit conversion: ``KIc^2/E`` in MPa m equals ``1e3`` N/mm."""

# ---------------------------------------------------------------------------
# elastic stiffness
# ---------------------------------------------------------------------------


def isotropic_stiffness(
    E: float,
    nu: float,
    *,
    dtype: torch.dtype = _DTYPE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the ``(6, 6)`` isotropic engineering stiffness."""

    return isotropic_stiffness_3d(float(E), float(nu), dtype=dtype, device=device)


def _check_spd(name: str, stiffness: torch.Tensor) -> None:
    eigenvalues = torch.linalg.eigvalsh(engineering_to_mandel_stiffness(stiffness).detach().cpu())
    if not bool(torch.isfinite(eigenvalues).all()) or float(eigenvalues.min()) <= 0.0:
        raise ValueError(f"{name} is not symmetric positive definite")


_LAMINATE_NORMAL = (0, 4, 5)
"""Voigt components continuous across layers stacked along local axis 1 (stress ``xx, xz, xy``)."""

_LAMINATE_IN_PLANE = (1, 2, 3)
"""Voigt components continuous along the layers (strain ``yy, zz, yz``)."""


def laminate_stiffness(
    layers: Sequence[Sequence[float]],
    *,
    dtype: torch.dtype = _DTYPE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Exact stiffness of isotropic layers ``(fraction, E, nu)`` stacked along local axis 1.

    The normal components ``n = (xx, xz, xy)`` carry a uniform stress and the
    in-plane components ``p = (yy, zz, yz)`` a uniform strain, so
    ``C_nn = <S_nn>^-1`` with ``S_nn = C_nn^-1`` per layer,
    ``C_np = C_nn <C_nn^-1 C_np>``, and
    ``C_pp = <C_pp - C_pn C_nn^-1 C_np> + <C_pn C_nn^-1> C_nn <C_nn^-1 C_np>``.
    The result is transversely isotropic about axis 1: series coupled
    normal to the layers and parallel coupled along them.
    """

    rows = [tuple(float(v) for v in layer) for layer in layers]
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError("layers must be a non-empty sequence of (fraction, E, nu)")
    fractions = torch.tensor([row[0] for row in rows], dtype=dtype, device=device)
    if bool((fractions <= 0.0).any()) or abs(float(fractions.sum()) - 1.0) > 1.0e-12:
        raise ValueError("layer fractions must be positive and sum to 1")
    C = torch.stack([isotropic_stiffness(row[1], row[2], dtype=dtype, device=device) for row in rows])
    n = torch.tensor(_LAMINATE_NORMAL, device=C.device)
    p = torch.tensor(_LAMINATE_IN_PLANE, device=C.device)
    C_nn = C[:, n][:, :, n]
    C_np = C[:, n][:, :, p]
    C_pn = C[:, p][:, :, n]
    C_pp = C[:, p][:, :, p]
    C_nn_inv = torch.linalg.inv(C_nn)

    def mean(values: torch.Tensor) -> torch.Tensor:
        return torch.einsum("l,lij->ij", fractions, values)

    A = torch.linalg.inv(mean(C_nn_inv))
    B = mean(C_nn_inv @ C_np)
    P = mean(C_pn @ C_nn_inv)
    out = torch.zeros((6, 6), dtype=dtype, device=C.device)
    out[n.unsqueeze(1), n.unsqueeze(0)] = A
    out[n.unsqueeze(1), p.unsqueeze(0)] = A @ B
    out[p.unsqueeze(1), n.unsqueeze(0)] = P @ A
    out[p.unsqueeze(1), p.unsqueeze(0)] = mean(C_pp - C_pn @ C_nn_inv @ C_np) + P @ A @ B
    out = 0.5 * (out + out.transpose(-1, -2))
    _check_spd("laminate stiffness", out)
    return out


def phase_local_stiffness(
    material: PhaseMaterial,
    *,
    dtype: torch.dtype = _DTYPE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a phase's stiffness in its material frame (isotropic or laminate)."""

    if material.is_isotropic:
        return isotropic_stiffness(material.E, material.nu, dtype=dtype, device=device)
    if material.is_laminate:
        return laminate_stiffness(material.layers, dtype=dtype, device=device)
    raise ValueError(f"unsupported phase kind {material.kind!r}")


def lame(E: Any, nu: Any) -> tuple[Any, Any]:
    """Return the Lame parameters ``(lambda, mu)`` for scalars or arrays."""

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


# ---------------------------------------------------------------------------
# frames and rotation
# ---------------------------------------------------------------------------


def frame_from_tangent(t: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return right-handed material frames ``(..., 3, 3)`` with columns ``e1, e2, e3``.

    ``e1 = t/|t|``; ``e2 = normalise(a x e1)`` with ``a`` the global axis
    least aligned with ``t`` (lowest index on ties); ``e3 = e1 x e2``.  The
    columns therefore map local to global coordinates.  A zero tangent is
    treated as ``t = (1, 0, 0)`` (no device synchronisation is needed).
    Non-tensor input (lists, numpy arrays of any dtype) is converted to
    float64; a tensor keeps its floating dtype, and a non-floating tensor is
    converted to float64.
    """

    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(np.asarray(t, dtype=np.float64), dtype=_DTYPE)
    elif not torch.is_floating_point(t):
        t = t.to(_DTYPE)
    if t.shape[-1] != 3:
        raise ValueError(f"tangent must end in 3 entries, got {tuple(t.shape)}")
    norm = torch.linalg.vector_norm(t, dim=-1, keepdim=True)
    valid = norm > 0.0
    unit_x = torch.zeros_like(t)
    unit_x[..., 0] = 1.0
    e1 = torch.where(valid, t / torch.where(valid, norm, torch.ones_like(norm)), unit_x)
    least = torch.argmin(torch.abs(e1), dim=-1)
    axis = torch.nn.functional.one_hot(least, num_classes=3).to(dtype=e1.dtype)
    e2 = torch.linalg.cross(axis, e1, dim=-1)
    e2 = e2 / torch.linalg.vector_norm(e2, dim=-1, keepdim=True)
    e3 = torch.linalg.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)


def _basis_tensors(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    eye = torch.eye(6, dtype=dtype, device=device)
    return engineering_strain_to_tensor(eye), engineering_stress_to_tensor(eye)


def rotate_stiffness_batch(C_local: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate engineering stiffnesses by tensor action.

    ``R`` ``(n, 3, 3)`` maps material to global coordinates (its columns are
    the material axes).  ``C_local`` is ``(6, 6)`` (broadcast) or
    ``(n, 6, 6)``.  Each item equals
    :func:`phast.core.voigt.rotate_stiffness_engineering`.
    """

    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"R must have shape (n, 3, 3), got {tuple(R.shape)}")
    if C_local.shape[-2:] != (6, 6) or C_local.ndim not in (2, 3):
        raise ValueError(f"C_local must be (6, 6) or (n, 6, 6), got {tuple(C_local.shape)}")
    if C_local.ndim == 3 and C_local.shape[0] != R.shape[0]:
        raise ValueError("C_local and R must have the same batch size")
    if C_local.dtype != R.dtype or C_local.device != R.device:
        raise TypeError("C_local and R must share dtype and device")
    identity = torch.eye(3, dtype=R.dtype, device=R.device)
    defect = (R.transpose(-1, -2) @ R - identity).abs()
    if R.shape[0] and float(defect.max()) > 1.0e-10:
        raise ValueError("every R must be orthogonal")
    strain_basis, stress_basis = _basis_tensors(R.dtype, R.device)
    local_strain = torch.einsum("nji,cjk,nkl->ncil", R, strain_basis, R)
    T_strain = tensor_to_engineering_strain(local_strain).transpose(1, 2)
    global_stress = torch.einsum("nij,cjk,nlk->ncil", R, stress_basis, R)
    T_stress = tensor_to_engineering_stress(global_stress).transpose(1, 2)
    return T_stress @ C_local @ T_strain


# ---------------------------------------------------------------------------
# constants extraction
# ---------------------------------------------------------------------------


_SHEAR_INDEX = {frozenset((1, 2)): 3, frozenset((0, 2)): 4, frozenset((0, 1)): 5}


def _as_numpy_6x6(C: Any) -> np.ndarray:
    if isinstance(C, torch.Tensor):
        array = C.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        array = np.asarray(C, dtype=np.float64)
    if array.shape != (6, 6):
        raise ValueError(f"stiffness must have shape (6, 6), got {array.shape}")
    return 0.5 * (array + array.T)


def engineering_constants(C: Any, *, axis_order: Sequence[int] = (0, 1, 2)) -> dict[str, float]:
    """Return engineering constants of ``C`` in the axis order given.

    Keys: ``E1, E2, E3, nu12, nu13, nu23, nu21, nu31, nu32, G23, G13, G12``,
    where axis ``1`` is ``axis_order[0]`` and so on, and ``nu_ij`` is the
    contraction along ``j`` under uniaxial stress along ``i``.
    """

    order = tuple(int(a) for a in axis_order)
    if sorted(order) != [0, 1, 2]:
        raise ValueError(f"axis_order must be a permutation of (0, 1, 2), got {axis_order!r}")
    S = np.linalg.inv(_as_numpy_6x6(C))
    a = order
    out: dict[str, float] = {}
    for i in range(3):
        out[f"E{i + 1}"] = float(1.0 / S[a[i], a[i]])
    for i in range(3):
        for j in range(3):
            if i != j:
                out[f"nu{i + 1}{j + 1}"] = float(-S[a[i], a[j]] / S[a[i], a[i]])
    out["G23"] = float(1.0 / S[_SHEAR_INDEX[frozenset((a[1], a[2]))], _SHEAR_INDEX[frozenset((a[1], a[2]))]])
    out["G13"] = float(1.0 / S[_SHEAR_INDEX[frozenset((a[0], a[2]))], _SHEAR_INDEX[frozenset((a[0], a[2]))]])
    out["G12"] = float(1.0 / S[_SHEAR_INDEX[frozenset((a[0], a[1]))], _SHEAR_INDEX[frozenset((a[0], a[1]))]])
    return out


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------


def _stack_phases(C_by_phase: Any, fractions: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(C_by_phase, Mapping):
        if not isinstance(fractions, Mapping):
            raise TypeError("fractions must be a mapping when C_by_phase is a mapping")
        keys = list(C_by_phase)
        if set(keys) != set(fractions):
            raise ValueError("C_by_phase and fractions must have the same keys")
        stiffness = torch.stack([torch.as_tensor(C_by_phase[k], dtype=_DTYPE) for k in keys])
        weights = torch.tensor([float(fractions[k]) for k in keys], dtype=_DTYPE)
    else:
        if isinstance(C_by_phase, torch.Tensor):
            stiffness = C_by_phase.to(_DTYPE)
        else:
            stiffness = torch.stack([torch.as_tensor(c, dtype=_DTYPE) for c in C_by_phase])
        weights = torch.as_tensor(fractions, dtype=_DTYPE, device=stiffness.device).reshape(-1)
    if stiffness.ndim != 3 or stiffness.shape[1:] != (6, 6):
        raise ValueError(f"stiffness must stack to (P, 6, 6), got {tuple(stiffness.shape)}")
    weights = weights.to(stiffness.device)
    if weights.shape[0] != stiffness.shape[0]:
        raise ValueError("one fraction per phase is required")
    if bool((weights < 0.0).any()) or abs(float(weights.sum()) - 1.0) > 1.0e-9:
        raise ValueError("fractions must be non-negative and sum to 1")
    return stiffness, weights


def voigt_reuss_bounds(C_by_phase: Any, fractions: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Voigt ``sum f C`` and Reuss ``inv(sum f inv(C))`` stiffnesses.

    ``C_by_phase`` is a mapping (with matching ``fractions`` mapping) or a
    ``(P, 6, 6)`` stack with ``(P,)`` fractions; engineering form throughout.
    """

    stiffness, weights = _stack_phases(C_by_phase, fractions)
    C_V = torch.einsum("p,pij->ij", weights, stiffness)
    compliance = torch.einsum("p,pij->ij", weights, torch.linalg.inv(stiffness))
    C_R = torch.linalg.inv(compliance)
    return 0.5 * (C_V + C_V.T), 0.5 * (C_R + C_R.T)


def loewner_min_eig(C_upper: Any, C_lower: Any) -> float:
    """Relative minimum eigenvalue of ``Mandel(C_upper - C_lower)``.

    The eigenvalue is divided by the larger spectral norm of the two Mandel
    stiffnesses, so ``>= 0`` means ``C_lower <= C_upper`` in the Loewner order.
    """

    upper = torch.as_tensor(_as_numpy_6x6(C_upper), dtype=_DTYPE)
    lower = torch.as_tensor(_as_numpy_6x6(C_lower), dtype=_DTYPE)
    difference = engineering_to_mandel_stiffness(upper - lower)
    minimum = float(torch.linalg.eigvalsh(difference.detach().cpu()).min())
    scale = max(
        float(torch.linalg.eigvalsh(engineering_to_mandel_stiffness(upper).detach().cpu()).abs().max()),
        float(torch.linalg.eigvalsh(engineering_to_mandel_stiffness(lower).detach().cpu()).abs().max()),
    )
    if scale <= 0.0:
        raise ValueError("stiffnesses must not both vanish")
    return minimum / scale


# ---------------------------------------------------------------------------
# fracture-energy formulas
# ---------------------------------------------------------------------------


def kic_to_gc_plane_strain(KIc: Any, E: Any, nu: Any) -> Any:
    """Plane-strain ``Gc = 1e3 KIc^2 (1 - nu^2)/E`` (KIc in MPa m^0.5, E in MPa, Gc in N/mm)."""

    return _MPA_M_TO_N_PER_MM * KIc**2 * (1.0 - nu**2) / E


def at2_strength(E: Any, Gc: Any, ell: Any) -> Any:
    """AT2 homogeneous-state strength ``sqrt(27 E Gc/(256 ell))``."""

    return np.sqrt(_AT2_NUM * E * Gc / (_AT2_DEN * ell))


def at2_gc_from_strength(sigma: Any, E: Any, ell: Any) -> Any:
    """Strength-matched AT2 fracture energy ``256 sigma^2 ell/(27 E)``."""

    return _AT2_DEN * sigma**2 * ell / (_AT2_NUM * E)


def at2_length(E: Any, Gc: Any, sigma: Any) -> Any:
    """AT2 length ``27 E Gc/(256 sigma^2)`` matching a strength."""

    return _AT2_NUM * E * Gc / (_AT2_DEN * sigma**2)


def smeared_band_modulus(w: float, t_bn: float, E_bn: float, E_in: float, E_out: float) -> float:
    """Series (Reuss) modulus of a band of width ``w`` holding a BN layer ``t_bn``.

    ``E_band = w / (t_bn/E_bn + 0.5 (w - t_bn)/E_in + 0.5 (w - t_bn)/E_out)``,
    with the remaining width split equally between the inner and outer phases.
    """

    w, t_bn = float(w), float(t_bn)
    if not (math.isfinite(w) and math.isfinite(t_bn)) or t_bn <= 0.0 or w < t_bn:
        raise ValueError(f"need 0 < t_bn <= w, got w={w!r}, t_bn={t_bn!r}")
    for name, value in (("E_bn", E_bn), ("E_in", E_in), ("E_out", E_out)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    rest = 0.5 * (w - t_bn)
    return w / (t_bn / float(E_bn) + rest / float(E_in) + rest / float(E_out))


def band_effective_toughness_1d(Gc_b: Any, Gc_l: Any, Gc_r: Any, w: Any, ell: Any) -> Any:
    """Analytic 1D dissipation per area of a crack centred in a band.

    ``G = (Gc_b/2) sum_j (T Gc_b + Gc_j)/(Gc_b + T Gc_j)``, ``T = tanh(w/(2 ell))``,
    for neighbours ``j`` = left and right.
    """

    T = np.tanh(w / (2.0 * ell))
    left = (T * Gc_b + Gc_l) / (Gc_b + T * Gc_l)
    right = (T * Gc_b + Gc_r) / (Gc_b + T * Gc_r)
    return 0.5 * Gc_b * (left + right)


def band_gc_start(target: float, Gc_l: float, Gc_r: float, w: float, ell: float) -> float:
    """Invert :func:`band_effective_toughness_1d` for ``Gc_b`` by log-space bisection.

    ``G(Gc_b)`` increases monotonically from 0 to infinity when ``w > 0``.
    """

    target, Gc_l, Gc_r, w, ell = (float(v) for v in (target, Gc_l, Gc_r, w, ell))
    for name, value in (("target", target), ("Gc_l", Gc_l), ("Gc_r", Gc_r), ("w", w), ("ell", ell)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")

    def g(value: float) -> float:
        return float(band_effective_toughness_1d(value, Gc_l, Gc_r, w, ell))

    lower = upper = target
    for _ in range(400):
        if g(lower) <= target:
            break
        lower *= 0.1
    else:
        raise ValueError("could not bracket the band fracture energy from below")
    for _ in range(400):
        if g(upper) >= target:
            break
        upper *= 10.0
    else:
        raise ValueError("could not bracket the band fracture energy from above")
    for _ in range(200):
        middle = math.sqrt(lower * upper)
        if g(middle) < target:
            lower = middle
        else:
            upper = middle
        if upper / lower - 1.0 <= 4.0 * np.finfo(np.float64).eps:
            break
    return math.sqrt(lower * upper)


# ---------------------------------------------------------------------------
# element arrays
# ---------------------------------------------------------------------------


def resolve_driver(material: PhaseMaterial, cfg: SolverConfig) -> int:
    """Return the element driver code for a phase under ``cfg``.

    Pores and phases whose nominal driver is ``'none'`` get ``DRIVER_NONE``.
    The ``bn_band`` phase gets the directional interface driver
    (``DRIVER_INTERFACE``) when ``cfg.energy_split_band`` is ``'interface'``;
    otherwise (``'isotropic'``, the default) it is treated like the other
    isotropic and smeared-laminate phases, which get spectral or full from
    ``cfg.energy_split_isotropic`` (a laminate drives damage with the Lame
    parameters of its isotropic equivalent ``E, nu``)  The nominal driver
    ``'interface'`` is otherwise treated like any driver other than ``'none'``.
    """

    if material.name == "pore" or material.driver == "none":
        return DRIVER_NONE
    if material.name == "bn_band" and cfg.energy_split_band == "interface":
        if not (material.is_isotropic or material.is_laminate):
            raise ValueError("the interface driver needs an isotropic or smeared_laminate bn_band phase")
        return DRIVER_INTERFACE
    if material.is_isotropic or material.is_laminate:
        return DRIVER_SPECTRAL if cfg.energy_split_isotropic == "spectral" else DRIVER_FULL
    return DRIVER_FULL


def build_element_arrays(
    model: VoxelModel,
    cfg: SolverConfig,
    *,
    device: torch.device | str | None = None,
) -> ElementMaterialArrays:
    """Build the per-element material arrays on ``device`` (default ``cfg.device``).

    Bank layout: one entry per present isotropic phase (phase-id order),
    followed by one entry per smeared-laminate
    voxel in element order, rotated from its tangent with
    :func:`frame_from_tangent`.  ``lam`` and ``mu`` are set for isotropic and
    smeared-laminate phases (the laminate's isotropic equivalent).
    """

    model.validate()
    target = resolve_device(cfg.device if device is None else device)
    dtype = solver_dtype(cfg.dtype)
    phase_flat = model.phase_id.reshape(-1).astype(np.int64)
    n_elements = phase_flat.shape[0]
    materials = model.materials.phases
    present = model.phases_present()
    iso_names = [name for name in present if materials[name].is_isotropic]
    framed_names = [name for name in present if not materials[name].is_isotropic]

    stiffness_index = np.empty(n_elements, dtype=np.int64)
    bank: list[torch.Tensor] = []
    for slot, name in enumerate(iso_names):
        material = materials[name]
        bank.append(isotropic_stiffness(material.E, material.nu, dtype=dtype, device=target).unsqueeze(0))
        stiffness_index[phase_flat == PHASE_ID[name]] = slot

    n_ids = len(PHASE_NAMES)
    by_id = {
        key: np.zeros(n_ids, dtype=np.float64)
        for key in ("Gc", "lam", "mu")
    }
    driver_by_id = np.zeros(n_ids, dtype=np.int8)
    for name in present:
        material = materials[name]
        pid = PHASE_ID[name]
        by_id["Gc"][pid] = material.Gc
        driver_by_id[pid] = resolve_driver(material, cfg)
        lam, mu = lame(material.E, material.nu)
        by_id["lam"][pid] = lam
        by_id["mu"][pid] = mu

    tangent_flat = model.tangent.reshape(-1, 3)
    if framed_names:
        framed_ids = np.array([PHASE_ID[name] for name in framed_names], dtype=np.int64)
        framed_elements = np.nonzero(np.isin(phase_flat, framed_ids))[0]
        local = torch.stack(
            [phase_local_stiffness(materials[name], dtype=dtype, device=target) for name in framed_names]
        )
        slot_of_id = np.full(n_ids, -1, dtype=np.int64)
        slot_of_id[framed_ids] = np.arange(len(framed_names))
        which = torch.as_tensor(slot_of_id[phase_flat[framed_elements]], dtype=torch.long, device=target)
        tangents = torch.as_tensor(tangent_flat[framed_elements], dtype=dtype, device=target)
        rotated = rotate_stiffness_batch(local[which], frame_from_tangent(tangents))
        bank.append(rotated)
        stiffness_index[framed_elements] = len(iso_names) + np.arange(framed_elements.shape[0])

    def element(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values[phase_flat], dtype=dtype, device=target)

    return ElementMaterialArrays(
        stiffness_bank=torch.cat(bank, dim=0),
        stiffness_index=torch.as_tensor(stiffness_index, dtype=torch.long, device=target),
        Gc=element(by_id["Gc"]),
        driver=torch.as_tensor(driver_by_id[phase_flat], dtype=torch.int8, device=target),
        lam=element(by_id["lam"]),
        mu=element(by_id["mu"]),
        phase=torch.as_tensor(phase_flat, dtype=torch.long, device=target),
        tangent=torch.as_tensor(tangent_flat, dtype=dtype, device=target),
    )
