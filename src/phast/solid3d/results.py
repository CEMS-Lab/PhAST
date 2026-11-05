"""Write the result file (``result.vtkhdf``) of a 3D voxel run through :mod:`phast.io.result`.

One file per run: the voxel grid as VTK ImageData, the state fields of the stored increments (displacement, damage, H at the eight Gauss
points, stress, strain and the energy densities), a history row for every increment, and the material, boundary-condition and periodicity
description that the layout asks for. Read it with ``phast.load_result``, validate it with ``python -m phast result validate``, open it in ParaView.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from phast.core.hex8 import h8_gauss_points
from phast.io.result import Field, ImageGrid, ResultWriter
from phast.solid3d.materials import frame_from_tangent, phase_local_stiffness
from phast.solid3d.model import PHASE_ID, PHASE_NAMES, VOIGT

RESULT_NAME = "result.vtkhdf"
QP_ORDER = "zeta outer, eta, xi inner, minus then plus"
QP_COMPONENTS = ",".join(f"qp{q}" for q in range(8))
TENSOR = ",".join(VOIGT)


def image_grid(grid) -> ImageGrid:
    """``ImageGrid`` of a ``VoxelGrid`` with the H8 2 x 2 x 2 Gauss rule of the solver."""
    pts, wts = h8_gauss_points(order=2, dtype=torch.float64)
    quad = {"H8": (pts.numpy(), wts.numpy(), "gauss 2x2x2", QP_ORDER)}
    return ImageGrid(tuple(grid.shape), tuple(float(h) for h in grid.h), tuple(float(o) for o in grid.origin), cell_type="H8", quadrature=quad)


def reduced_of_full(grid) -> np.ndarray:
    """Periodic reduced node id of every full grid node (index ``i`` maps to ``i % nx`` on a periodic axis)."""
    nx, ny, nz = grid.shape
    px, py, pz = grid.periodic
    nxr, nyr = (nx if px else nx + 1), (ny if py else ny + 1)
    k, j, i = np.meshgrid(np.arange(nz + 1, dtype=np.int64), np.arange(ny + 1, dtype=np.int64), np.arange(nx + 1, dtype=np.int64), indexing="ij")
    ir = i % nx if px else i
    jr = j % ny if py else j
    kr = k % nz if pz else k
    return (ir + nxr * (jr + nyr * kr)).reshape(-1)


def phase_materials(model, cfg) -> dict:
    """``materials/phases/<id>`` attributes (and the 6 x 6 local stiffness of laminate phases)."""
    law = "PFCZM" if cfg.damage_law == "pfczm" else "AT2"
    out = {}
    for name, m in model.materials.phases.items():
        a = dict(name=name, fracture_law=law, Gc=float(m.Gc), ell=float(model.materials.length_scale), driver=m.driver, engine_kind=m.kind)
        units = {"Gc": "N/mm", "ell": "mm"}
        if m.is_isotropic:
            a.update(elastic_model="isotropic", E=float(m.E), nu=float(m.nu))
            units["E"] = "MPa"
        elif m.is_laminate:
            a.update(elastic_model="stiffness", layers=json.dumps([list(map(float, layer)) for layer in m.layers]),
                     E_driver=float(m.E), nu_driver=float(m.nu),
                     stiffness=phase_local_stiffness(m, dtype=torch.float64).cpu().numpy(),
                     stiffness_frame="local axes (orientation rows); axis 1 = interface normal; engineering Voigt xx, yy, zz, yz, xz, xy",
                     layers_description="(fraction, E, nu) isotropic layers stacked along local axis 1; E_driver and nu_driver are the isotropic equivalent used only for the damage driver")
            units.update(stiffness="MPa", E_driver="MPa")
        if m.tensile_strength is not None:
            a["strength"] = float(m.tensile_strength)
            units["strength"] = "MPa"
        if law == "PFCZM":
            a.update(pfczm_softening=str(cfg.pfczm_softening), pfczm_p=float(cfg.pfczm_p))
        a["units"] = units
        out[PHASE_ID[name]] = a
    return out


def loading(load, grid) -> tuple[dict, dict]:
    """``loading/periodic`` and ``loading/bc`` attributes of the mixed macro-strain load case."""
    names = lambda idx: ",".join(VOIGT[i] for i in idx)          # noqa: E731
    periodic = dict(axes=",".join(a for a, p in zip("xyz", grid.periodic) if p), controlled=VOIGT[load.controlled], free=names(load.free),
                    fixed_zero=names(load.fixed_zero), load_case=str(load.name), n_increments=float(len(load.targets)),
                    description="u = E (X - origin) + u~, u~ periodic on the periodic axes; E[controlled] follows history/load_parameter, "
                                "E[free] carries zero macro stress, E[fixed_zero] = 0")
    bcs = {"macro_controlled": dict(set="cell", kind="periodic", components=VOIGT[load.controlled], value="load_parameter")}
    if load.free:
        bcs["macro_free"] = dict(set="cell", kind="traction", components=names(load.free), value=0.0, description="zero macro stress in these components")
    if load.fixed_zero:
        bcs["macro_fixed_zero"] = dict(set="cell", kind="periodic", components=names(load.fixed_zero), value=0.0)
    return periodic, bcs


HISTORY_UNITS = {"time": "1", "dt": "1", "load_parameter": "1", "load_case_increment": "1", "residual": "1", "pcg_iterations_mechanics": "1",
                 "pcg_iterations_damage": "1", "active_set_iterations": "1", "converged": "1", "kkt_ok": "1", "mechanics_relative_residual": "1",
                 "hill_mandel_relative": "1", "applied_displacement": "mm", "reaction_force_controlled": "N", "energy/balance_relative": "1",
                 "max_damage": "1", "mean_damage": "1", "damaged_volume_fraction": "1", "max_H": "MPa", "closure_solves": "1",
                 "min_normal_strain": "1", "min_principal_strain": "1", "min_det_F": "1", "newton_iterations_damage": "1",
                 "seconds_wall": "s", "seconds_mechanics": "s", "seconds_damage": "s", "seconds_history": "s", "seconds_output": "s"}


def history_attrs(controlled: int) -> dict:
    quad = "phase-field surface functional Gamma(d) of the configured damage law, element Gc, exact H8 element matrices"
    attrs = {"increment": dict(description="accepted solution counter (step 0 = initial state)"),
             "time": dict(description="pseudo-time: accumulated |change of E[controlled]|; quasi-static, no physical time"),
             "dt": dict(description="increment of time"),
             "load_parameter": dict(description=f"controlled macro strain E_{VOIGT[controlled]}"),
             "load_case_increment": dict(description="load-case increment the step belongs to (repeats under cutbacks)"),
             "residual": dict(description="stagger change max(du_rel, dE_rel, |dd|_inf) at acceptance"),
             "reaction_force_controlled": dict(description="macro stress of the controlled component times the cell face area normal to its axis"),
             "energy/elastic": dict(method="0.5 V <sigma : eps> (Hill-Mandel mean work at the accepted state)"),
             "energy/fracture": dict(method=quad),
             "energy/external_work": dict(method="trapezoidal rule over the controlled macro strain: W += V (Sigma_c + Sigma_c,prev)/2 dE_c"),
             "energy/dissipation": dict(method="external_work - elastic"),
             "energy/balance_residual": dict(method="external_work - (elastic + fracture)"),
             "energy/balance_relative": dict(method="balance_residual / max(|external_work|, tiny)")}
    for name in PHASE_NAMES:
        attrs[f"energy/fracture_{name}"] = dict(method=quad + f", elements of phase {name}")
    return attrs


class ResultFile:
    """The result file of one run. ``add`` is called for every accepted increment; ``close`` publishes the file."""

    def __init__(self, run_dir, op, cfg, load, *, run_id: str, expected_steps: int, config_text: str = ""):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / RESULT_NAME
        if self.path.exists() or Path(str(self.path) + ".partial").exists():
            raise FileExistsError(f"{self.path} exists; a result file is never overwritten")
        self.model, self.cfg, self.load, self.op = op.model, cfg, load, op
        self.controlled = int(load.controlled)
        self.closure = cfg.crack_closure != "none"
        self.pfczm = cfg.damage_law == "pfczm"
        self.phases = tuple(self.model.phases_present())
        self._t_acc, self._last_load = 0.0, 0.0
        self.rows, self.stored_steps = 0, []
        fields = self._declare()
        periodic, bcs = loading(load, self.model.grid)
        meta = dict(command_line=" ".join(os.path.basename(a) for a in sys.argv[:1]), software="phast solid3d (beta)", solver_dtype=str(cfg.dtype),
                    tolerances=json.dumps(dict(stagger_tol=cfg.stagger_tol, pcg_rtol_mech=cfg.pcg_rtol_mech, damage_kkt_tol=cfg.damage_kkt_tol,
                                               damage_linear_tol=cfg.damage_linear_tol)),
                    claim_boundary="periodic voxel unit cell; prescribed macro strain; not a general-geometry 3D specimen; not validated against experiment",
                    time_definition="quasi-static: time = accumulated |change of the controlled macro strain|; no physical time")
        self._rw = ResultWriter(self.path, image_grid(self.model.grid), fields, dim=3, analysis="3d", run_id=run_id, history_coupling="fresh",
                                history_update="hard_max", materials=phase_materials(self.model, cfg), bcs=bcs, periodic=periodic,
                                history_units=dict(HISTORY_UNITS), history_attrs=history_attrs(self.controlled), config=config_text, meta=meta,
                                expected_steps=expected_steps, sidecars=False)
        try:
            self._rw.write_array("loading/periodic/reduced_of_full", reduced_of_full(self.model.grid), dict(description="reduced (periodic) node of every full grid node"))
            mat = op.mat
            elem = {"Gc": (np.asarray(mat.Gc.cpu(), np.float64), "N/mm", "element Gc used by the solver"),
                    "stiffness_index": (np.asarray(mat.stiffness_index.cpu(), np.int64), "1", "row of stiffness_bank used by each element"),
                    "stiffness_bank": (np.asarray(mat.stiffness_bank.cpu(), np.float64), "MPa", "global-frame 6x6 stiffness, engineering Voigt xx, yy, zz, yz, xz, xy")}
            for name, (arr, units, desc) in elem.items():
                self._rw.write_array(f"materials/element/{name}", arr, dict(units=units, description=desc))
        except Exception:
            self._rw.close("failed")
            raise

    # ------------------------------------------------------------------ declaration
    def _declare(self):
        phase_flat = np.asarray(self.model.phase_id).reshape(-1).astype(np.uint8)
        tangent = torch.as_tensor(np.asarray(self.model.tangent).reshape(-1, 3), dtype=torch.float64)
        tensor_attrs = dict(derived_method="qp_mean_volume_weighted", description="mean over the 8 Gauss points (equal weights on the voxel) of the accepted state; float64")
        f = [Field("phase_id", "cell", "id", "1", "state", static=True, data=phase_flat, attrs=dict(phase_names=json.dumps(list(PHASE_NAMES)))),
             Field("orientation", "cell", "e1x,e1y,e1z,e2x,e2y,e2z,e3x,e3y,e3z", "1", "state", static=True,
                   data=frame_from_tangent(tangent).transpose(-1, -2).reshape(-1, 9).numpy(),
                   attrs=dict(description="local axes as rows (e1, e2, e3); e1 = interface normal for laminate phases; isotropic phases ignore it")),
             Field("displacement", "node", "x,y,z", "mm", "state", attrs=dict(description="total displacement u = E (X - origin) + u~ of the accepted state")),
             Field("displacement_fluctuation", "node", "x,y,z", "mm", "state", attrs=dict(description="periodic fluctuation u~ (the solver unknown, expanded to every node)")),
             Field("damage", "node", "d", "1", "state", attrs=dict(description="nodal phase-field damage d")),
             Field("H", "quadrature_point", QP_COMPONENTS, "MPa", "history",
                   attrs=dict(block="H8", point_order=QP_ORDER, description="history variable H at each Gauss point after the increment (solver state)")),
             Field("stress", "cell", TENSOR, "MPa", "derived", attrs=dict(tensor_attrs)),
             Field("strain", "cell", TENSOR, "1", "derived", attrs=dict(tensor_attrs, shear="engineering")),
             Field("psi_driving", "cell", "psi", "MPa", "derived", attrs=dict(tensor_attrs, description="mean over the 8 Gauss points of the damage driving energy Y")),
             Field("psi_elastic", "cell", "psi", "MPa", "derived", attrs=dict(tensor_attrs, description="mean over the 8 Gauss points of the degraded elastic energy density"))]
        if self.closure:
            f += [Field("psi_closure", "cell", "psi", "MPa", "derived", attrs=dict(tensor_attrs, description="elastic energy added back by the crack-closure treatment")),
                  Field("closure_active", "cell", "fraction", "1", "derived", attrs=dict(derived_method="recomputed", description="fraction of the 8 Gauss points at which a closure term carries stress")),
                  Field("det_F_min", "cell", "det", "1", "derived", attrs=dict(derived_method="recomputed", description="smallest Jacobian determinant of the deformation over the Gauss points"))]
        return f

    # ------------------------------------------------------------------ rows and fields
    def _row(self, rec, output_s: float) -> dict:
        load = float(rec.macro_strain[self.controlled])
        dt = abs(load - self._last_load) if rec.step > 0 else 0.0
        self._t_acc += dt
        self._last_load = load
        row = dict(increment=int(rec.step), time=self._t_acc, dt=dt, load_parameter=load, load_factor=float(rec.load_factor), load_case_increment=float(rec.increment),
                   damage_solved=0 if rec.step == 0 else 1, cutbacks=int(rec.cutbacks), stagger_iterations=int(rec.stagger_iters), residual=float(rec.stagger_change),
                   pcg_iterations_mechanics=float(rec.pcg_iters_mech), pcg_iterations_damage=float(rec.pcg_iters_pf), active_set_iterations=float(rec.active_set_iters),
                   converged=float(bool(rec.converged)), kkt_ok=float(bool(rec.kkt_ok)), mechanics_relative_residual=float(rec.mech_relative_residual),
                   hill_mandel_relative=float(rec.hill_mandel_rel), applied_displacement=float(rec.applied_disp), reaction_force_controlled=float(rec.reaction_force))
        for i, c in enumerate(VOIGT):
            row[f"macro_stress/{c}"] = float(rec.macro_stress[i])
            row[f"macro_strain/{c}"] = float(rec.macro_strain[i])
            row[f"mean_strain/{c}"] = float((rec.mean_strain or rec.macro_strain)[i])
        row.update({"energy/elastic": float(rec.elastic), "energy/fracture": float(rec.fracture), "energy/external_work": float(rec.external),
                    "energy/dissipation": float(rec.external) - float(rec.elastic), "energy/balance_residual": float(rec.balance_residual),
                    "energy/balance_relative": float(rec.balance_residual_rel)})
        for name in PHASE_NAMES:
            if name in self.phases:
                row[f"energy/fracture_{name}"] = float(rec.fracture_by_phase.get(name, 0.0))
        for key in ("max_damage", "mean_damage", "damaged_volume_fraction", "max_H"):
            row[key] = float(getattr(rec, key))
        if self.closure:
            row["closure_solves"] = float(rec.closure_solves or 0)
            row["min_normal_strain"] = float(rec.min_normal_strain or 0.0)
            row["min_principal_strain"] = float(rec.min_principal_strain or 0.0)
            row["min_det_F"] = 1.0 if rec.min_det_F is None else float(rec.min_det_F)
        if self.pfczm:
            row["newton_iterations_damage"] = float(rec.newton_iters_pf or 0)
        row.update(seconds_wall=float(rec.wall_s), seconds_mechanics=float(rec.mech_s), seconds_damage=float(rec.damage_s), seconds_history=float(rec.history_s),
                   seconds_output=float(output_s))
        return row

    def _fields(self, snap, state) -> dict:
        out = dict(displacement=snap.displacement, displacement_fluctuation=snap.fluctuation, damage=snap.damage, H=state.H.detach().cpu().numpy(),
                   stress=snap.stress, strain=snap.strain, psi_driving=snap.psi_driving, psi_elastic=snap.psi_elastic)
        if self.closure:
            out.update(psi_closure=snap.psi_closure, closure_active=snap.closure_active, det_F_min=snap.det_F_min)
        for name, array in out.items():
            if not np.isfinite(np.asarray(array)).all():
                raise ValueError(f"non-finite field {name} at step {snap.step}")
        return out

    def add(self, record, *, snapshot=None, state=None, output_s: float = 0.0) -> None:
        """Append the history row of an accepted increment and, when ``snapshot`` is given, its fields."""
        row = self._row(record, output_s)
        view = derived = None
        if snapshot is not None:
            fields = self._fields(snapshot, state)
            view = {k: v for k, v in fields.items() if self._rw.fields[k].placement == "view"}
            derived = {k: v for k, v in fields.items() if self._rw.fields[k].placement == "derived"}
            self.stored_steps.append(int(record.step))
        self._rw.append(row, fields=view, derived=derived)
        self.rows += 1

    def close(self, status: str = "complete", reason: str | None = None) -> Path:
        """Close, validate and rename the file; returns the final path."""
        if self._rw.closed:
            return self.path
        if reason:
            self._rw.f.attrs["status_reason"] = str(reason)
        return self._rw.close(status)
