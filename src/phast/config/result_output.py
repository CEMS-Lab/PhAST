"""Layout-2.0 result file for the 2D YAML runner (``config/run_config.py``).

The runner writes ``<output_dir>/result.vtkhdf`` through
:class:`phast.io.result.ResultWriter` unless ``output.result`` is ``false``
(``true``, a mapping, or an omitted key all write it). Settings
(``docs/user_guide/result_file_layout.md``, section 3)::

    output:
      result:
        every: 1                 # stored steps; quasi-static default 1, explicit default output.h5_every
        fields: [displacement, damage, H, stress, strain]
        derived_location: cell   # cell | quadrature_point
        derived_every: 1         # 1 = in the ParaView view; N > 1 = in /derived every N-th increment
        size_limit_gb: 4
        layout: 2.0              # 2.1 adds the items of layout 2.1 that the runner has data for (see below)
        crack_tips: false        # true or {threshold: 0.5, direction: x}: crack-tip table (T3 meshes), implies 2.1

With ``layout: 2.1`` the file also records the initial state with the pre-seeded damage (``initial_conditions/``),
wave speeds in the material table, and, for the velocity-impact ramp, the prescribed velocity of its boundary
condition. The default stays 2.0. The per-solve table ``history/substeps`` of layout 2.1 is not written by the
runner because the damage solver does not report its individual solves.

History rows are written for every converged increment; fields are stored at
increments ``0, every, 2*every, ...``, at the last increment and at a
time-limit stop.
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from typing import Any, Optional

import numpy as np

from ..core.fem_operators import stress_split_of
from ..io.result.crack_tips import locate_tip

RESULT_FILENAME = "result.vtkhdf"
_SETTING_KEYS = ("every", "fields", "derived_location", "derived_every", "size_limit_gb", "layout", "crack_tips")
_LAYOUTS = ("2.0", "2.1")
_DERIVED_NAMES = ("stress", "strain")
_OPTIONAL_STATE = ("velocity", "acceleration")
_RESTART_STATE = ("damage_lower_bound",)          # stored when listed in output.result.fields
# energy splits whose plane-strain out-of-plane stress has a closed form here
_ZZ_SPLITS = ("isotropic", "amor", "star_convex", "spectral", "spectral_stress")


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _relativise(obj: Any, base: str) -> Any:
    """Copy of ``obj`` with absolute filesystem paths rewritten relative to ``base``."""
    if isinstance(obj, dict):
        return {k: _relativise(v, base) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_relativise(v, base) for v in obj]
    if isinstance(obj, str) and os.path.isabs(obj):
        try:
            return os.path.relpath(obj, base)
        except ValueError:          # another drive on Windows
            return os.path.basename(obj)
    return obj


def resolve_result_settings(value: Any, *, explicit: bool, h5_every: int) -> Optional[dict]:
    """Resolve ``output.result`` into a full settings dict, or ``None`` when disabled.

    ``value`` is ``True`` (defaults), ``False`` or ``None`` (disabled) or a
    mapping of the section 3 keys. The returned dict carries ``every_given`` so the
    runner knows whether it may coarsen ``every`` to respect the size limit.
    """
    if value is False or value is None:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"output.result must be true, false or a mapping, got {value!r}")
    unknown = sorted(set(value) - set(_SETTING_KEYS) - {"enabled"})
    if unknown:
        raise ValueError(f"output.result: unknown keys {unknown}; allowed: {list(_SETTING_KEYS)}")
    if value.get("enabled", True) is False:
        return None
    required = ["displacement", "damage", "H"] + (list(_OPTIONAL_STATE) if explicit else [])
    fields = value.get("fields")
    if fields is None:
        fields = required + list(_DERIVED_NAMES)
    fields = [str(f) for f in fields]
    allowed = set(required) | set(_DERIVED_NAMES) | set(_OPTIONAL_STATE) | set(_RESTART_STATE)
    bad = sorted(set(fields) - allowed)
    if bad:
        raise ValueError(f"output.result.fields: unknown fields {bad}; allowed: {sorted(allowed)}")
    layout = str(value.get("layout", "2.0"))
    if layout not in _LAYOUTS:
        raise ValueError(f"output.result.layout must be 2.0 or 2.1, got {value.get('layout')!r}")
    tips = value.get("crack_tips", False)
    if tips is True:
        tips = {}
    if tips in (False, None):
        tips = None
    else:
        if not isinstance(tips, dict) or set(tips) - {"threshold", "direction"}:
            raise ValueError("output.result.crack_tips must be true, false or a mapping with threshold and direction")
        tips = dict(threshold=float(tips.get("threshold", 0.5)), direction=str(tips.get("direction", "x")))
        if not 0.0 < tips["threshold"] <= 1.0:
            raise ValueError("output.result.crack_tips.threshold must lie in (0, 1]")
        if tips["direction"] not in ("x", "-x", "y"):
            raise ValueError("output.result.crack_tips.direction must be x, -x or y")
        layout = "2.1"                                    # the crack-tip table is a layout-2.1 item
    dropped = [f for f in required if f not in fields]
    if dropped:
        raise ValueError(f"output.result.fields drops state or history variables {dropped}; "
                         "the result file always stores them (layout 2.0, section 3)")
    every = value.get("every")
    every_given = every is not None
    every = int(every) if every_given else (max(1, int(h5_every)) if explicit else 1)
    derived_every = int(value.get("derived_every", 1))
    location = str(value.get("derived_location", "cell"))
    if location not in ("cell", "quadrature_point"):
        raise ValueError(f"output.result.derived_location must be cell or quadrature_point, got {location!r}")
    if every < 1 or derived_every < 1:
        raise ValueError("output.result.every and derived_every must be >= 1")
    if location == "quadrature_point" and derived_every > 1:
        raise ValueError("output.result: derived_location quadrature_point requires derived_every 1")
    return dict(every=every, every_given=every_given, fields=fields, derived_location=location,
                derived_every=derived_every, size_limit_gb=float(value.get("size_limit_gb", 4.0)),
                layout=layout, crack_tips=tips)


def plane_strain_stress_zz(fem: Any, d: Any, strain: Any) -> Any:
    """Out-of-plane stress of the degraded plane-strain law, ``(E,)`` for T3 or ``(E, 4)`` for Q4.

    The 2D solver stores no ``sigma_zz``. With ``eps_zz = 0`` it follows from
    the split that governs the in-plane stress (``stress_split_of``: the energy
    split, or ``isotropic`` under ``stress_degradation: full``) with the 3D
    Lame parameters and a third principal strain of zero:

    - ``isotropic``: ``g lam tr`` (equal to ``nu (sxx + syy)``);
    - ``amor``: ``g (kappa <tr>+ - 2 mu tr / 3) + kappa <tr>-``;
    - ``star_convex``: ``g lam tr`` for ``tr >= 0``, ``-2 g mu tr / 3 + kappa tr`` otherwise;
    - ``spectral`` (strain split): ``g lam <tr>+ + lam <tr>-``;
    - ``spectral_stress``: ``g <lam tr>+ + <lam tr>-``;

    with ``tr = eps_xx + eps_yy`` and ``g = g(d)`` at the point. Plane stress
    has ``sigma_zz = 0`` and the plasticity module keeps its own ``sigma_zz``,
    so neither is handled here.
    """
    import torch

    mat, mesh = fem.material, fem.mesh
    if getattr(mat, "plane_stress", False):
        raise ValueError("sigma_zz of a plane-stress run is zero by definition")
    split = stress_split_of(mat)
    if split not in _ZZ_SPLITS:
        raise ValueError(f"no out-of-plane stress for energy_split {split!r}; known: {list(_ZZ_SPLITS)}")
    if getattr(mesh, "element_type", "T3") == "Q4":
        d_e = torch.einsum("qa,ea->eq", mesh.quad_N, d[mesh.elements])
    else:
        d_e = d[mesh.elements].mean(1)
    g = mat.degradation(d_e)
    exx, eyy, _gxy = strain
    tr = exx + eyy
    lam, mu, kappa = (fem._match_ndim(x, tr) for x in fem._resolve_lame())
    if split == "isotropic":
        return g * lam * tr
    if split == "spectral":
        tr_plus = torch.clamp(tr, min=0)
        return g * lam * tr_plus + lam * (tr - tr_plus)
    if split == "spectral_stress":
        s3 = lam * tr
        s3_plus = torch.clamp(s3, min=0)
        return g * s3_plus + (s3 - s3_plus)
    dev_zz = -tr / 3.0
    if split == "amor":
        tr_plus = torch.clamp(tr, min=0)
        return g * (kappa * tr_plus + 2.0 * mu * dev_zz) + kappa * (tr - tr_plus)
    tension = tr >= 0                                                   # star_convex
    return torch.where(tension, g * lam * tr, g * 2.0 * mu * dev_zz + kappa * tr)


class RunResultRecorder:
    """Builds the result file from the runner's state, one converged increment at a time."""

    def __init__(self, *, cfg: Any, settings: dict, output_dir: str, mesh: Any, mat: Any, solver: Any,
                 solver_cfg: Any, n_steps: int, reaction_node_set: Optional[str],
                 config_path: Optional[str] = None) -> None:
        from ..io.result import Field, ResultWriter, SoftStop, UnstructuredMesh, estimate_size

        self.settings = dict(settings)
        self.output_dir = output_dir
        self.mesh0 = mesh
        self.n_nodes0 = int(mesh.n_nodes)
        self.solver = solver
        self.mat = mat
        self.explicit = solver_cfg.solver_type == "explicit"
        self.n_steps = int(n_steps)
        self.traction_loaded = bool(getattr(solver.bcs, "neumann_bcs", None))
        self.q4 = getattr(mesh, "element_type", "T3") == "Q4"
        self.block = "Q4" if self.q4 else "T3"
        self.layout21 = self.settings.get("layout") == "2.1"
        self.tips = self.settings.get("crack_tips")
        self.stopped_reason = None
        self.writer = None
        self.time = 0.0
        self.increment = 0
        damage_on = bool(getattr(solver_cfg, "enable_damage", True))
        self.damage_on = damage_on
        self.subcycled = self.explicit and damage_on and int(getattr(solver_cfg, "damage_every", 1)) > 1
        self.H_damage_input = solver.H_elem.detach().clone() if self.subcycled else None
        self.d_start = solver.d.detach().clone()          # damage at the start of the next increment
        # damage solves counted by the damage solver; the solves made before the first increment (a pre-seeded
        # notch) are not part of an increment
        self.solves_seen = int(getattr(solver.damage_solver, "n_solves", 0))
        # loading work of the applied tractions, accumulated with the left-point rule f_n . (u_(n+1) - u_n)
        self.u_before = solver.u.detach().clone()
        self.external_work = 0.0
        self.initial_total = None

        if self.tips and self.q4:
            raise ValueError("output.result.crack_tips finds the tip on T3 cells; this mesh has Q4 cells")

        # --- mesh ---------------------------------------------------------------
        if self.q4:
            from ..core.quad_elements import quad_gauss_points
            pts, wts = quad_gauss_points(order=2)
            quad = {"Q4": (_np(pts), _np(wts), "gauss 2x2", "eta outer, xi inner, minus then plus")}
            n_qp = int(mesh.quad_N.shape[0])
            if n_qp != 4:
                raise ValueError(f"Q4 mesh with {n_qp} quadrature points; the result writer expects 4")
        else:
            quad = {"T3": ([[1.0 / 3.0, 1.0 / 3.0]], [0.5], "centroid", "single point")}
        node_sets = {name.replace("/", "_"): _np(idx).astype(np.int64)
                     for name, idx in (getattr(mesh, "node_sets", None) or {}).items()}
        wmesh = UnstructuredMesh(_np(mesh.nodes), {self.block: _np(mesh.elements)}, node_sets=node_sets,
                                 quadrature=quad)

        # --- fields ---------------------------------------------------------------
        fields = self.settings["fields"]
        hist_loc, hist_comp = (("quadrature_point", "qp0,qp1,qp2,qp3") if self.q4 else ("cell", "H"))
        decl = [Field("displacement", "node", "x,y", "mm", "state"),
                Field("damage", "node", "d", "1", "state"),
                Field("H", hist_loc, hist_comp, "MPa", "history",
                      attrs={"meaning": "history variable after the increment (crack driving energy density)"})]
        if self.subcycled:
            decl.append(Field("H_damage_input", hist_loc, hist_comp, "MPa", "history",
                              attrs={"meaning": "history variable used by the most recent damage solve "
                                                "(damage is solved every damage_every increments)"}))
        if "damage_lower_bound" in fields:
            decl.append(Field("damage_lower_bound", "node", "d", "1", "state",
                              attrs={"definition": "damage at the start of the increment: the lower bound that the "
                                                   "irreversibility constraint applied in its damage solve (d_prev)"}))
        if "velocity" in fields:
            decl.append(Field("velocity", "node", "x,y", "mm/s", "state"))
        if "acceleration" in fields:
            decl.append(Field("acceleration", "node", "x,y", "mm/s2", "state"))
        method = "qp_mean_volume_weighted" if self.q4 else "exact"
        placement = "view" if self.settings["derived_every"] == 1 else "derived"
        self.derived_names = []
        plane_strain = not bool(getattr(mat, "plane_stress", False))
        self.zz_split = str(stress_split_of(mat))
        self.with_stress_zz = ("stress" in fields and plane_strain and self.zz_split in _ZZ_SPLITS
                               and getattr(mat, "plasticity_model", "none") == "none")
        for name in _DERIVED_NAMES:
            if name not in fields:
                continue
            units = "MPa" if name == "stress" else "1"
            extra = {"derived_method": method}
            if name == "strain":
                extra["shear"] = "engineering"
            decl.append(Field(name, "cell", "xx,yy,xy", units, "derived", placement=placement,
                              dtype=np.float32, attrs=extra))
            self.derived_names.append(name)
            if self.q4 and self.settings["derived_location"] == "quadrature_point":
                decl.append(Field(name, "quadrature_point", "xx,yy,xy", units, "derived",
                                  placement="quadrature:Q4", dtype=np.float32,
                                  attrs={"shear": "engineering"} if name == "strain" else {}))
        if self.with_stress_zz:
            meaning = ("nu*(stress_xx + stress_yy), exact for the isotropic split in plane strain"
                       if self.zz_split == "isotropic" else
                       f"out-of-plane stress of the plane-strain law (eps_zz = 0) for the {self.zz_split} split")
            decl.append(Field("stress_zz", "cell", "zz", "MPa", "derived", placement=placement,
                              dtype=np.float32, attrs={"derived_method": method, "meaning": meaning}))
            self.derived_names.append("stress_zz")
        if not self.q4 and self.settings["derived_location"] == "quadrature_point":
            print("[result] derived_location quadrature_point: T3 cells have one point, "
                  "so the cell values are the point values (stored once, per cell)", flush=True)
        self.per_point = self.q4 and self.settings["derived_location"] == "quadrature_point"

        # --- size rule: coarsen the default cadence when the state alone is too large
        every = self.settings["every"]
        primary = estimate_size(wmesh, [f for f in decl if f.kind in ("state", "history")], 1)["primary"]
        limit = self.settings["size_limit_gb"] * 1e9
        n_stored = math.ceil(self.n_steps / every) + 1
        if not self.settings["every_given"] and primary * n_stored > limit and self.n_steps > 1:
            factor = math.ceil(primary * n_stored / limit)
            every = every * factor
            print(f"[result] stored state estimated at {primary * n_stored / 1e9:.1f} GB; storing every "
                  f"{every} increments (set output.result.every to choose)", flush=True)
        self.settings["every"] = every
        expected = math.ceil(self.n_steps / every) + 1

        # --- materials, loading, history metadata ------------------------------------
        materials = {0: self._material_attrs(cfg, mat, solver_cfg, wave_speeds=self.layout21)}
        bcs, self.damage_bcs, self.load_def = self._loading(cfg, reaction_node_set)
        self.reaction_sets = self._reaction_sets(cfg, reaction_node_set, node_sets)
        self.reaction_idx = {name: _np(mesh.node_sets[orig]).astype(np.int64)
                             for name, orig in self.reaction_sets.items()}
        units, hattrs = self._history_metadata()
        method_name, beta = self._history_update(solver_cfg)

        settings_out = {k: v for k, v in self.settings.items() if k != "every_given"}
        config_dict = _relativise(_config_dict(cfg, settings_out), output_dir)
        meta = {
            "command_line": " ".join([os.path.basename(sys.argv[0])]
                                    + [str(_relativise(a, output_dir)) for a in sys.argv[1:]]),
            "tolerances": {k: getattr(solver_cfg, k, None) for k in
                           ("stagger_tol", "damage_tol", "static_tol", "max_stagger")},
            "solver_type": solver_cfg.solver_type,
            "time_integrator": getattr(solver_cfg, "time_integrator", "") if self.explicit else "",
            "device": str(getattr(solver, "device", "cpu")),
        }
        if config_path:
            meta["config_file"] = os.path.basename(config_path)
        self.stop = SoftStop()
        try:
            self.writer = ResultWriter(
                os.path.join(output_dir, RESULT_FILENAME), wmesh, decl, dim=2,
                analysis="plane_strain" if plane_strain else "plane_stress",
                run_id=os.path.basename(os.path.normpath(output_dir)) or "run",
                history_coupling="fresh", history_update=method_name, history_update_beta=beta,
                kinematics="finite" if getattr(mat, "kinematics", "small") == "finite" else "small",
                materials=materials, bcs=bcs, history_units=units, history_attrs=hattrs,
                config=config_dict, meta=meta, expected_steps=expected,
                size_limit_gb=self.settings["size_limit_gb"], sidecars=False)
            for key, a in self.damage_bcs.items():
                self.writer.write_attrs(f"loading/damage_bc/{key}", a)
            if self.layout21:
                self._write_initial(cfg)
        except BaseException:
            self.stop.restore()
            if self.writer is not None and not self.writer.closed:
                self.writer.close("failed")
            raise
        self.every = every
        if self.writer.moved_out_of_view:
            print(f"[result] estimated size above {self.settings['size_limit_gb']} GB: "
                  f"{', '.join(self.writer.moved_out_of_view)} stored in /derived", flush=True)
            self.derived_cadence = every
        else:
            self.derived_cadence = self.settings["derived_every"]
        self.derived_outside = {f for f in self.writer.derived}
        print(f"[result] {RESULT_FILENAME}: fields every {every} increment(s), history every increment", flush=True)

    # ------------------------------------------------------------------ setup helpers

    @staticmethod
    def _material_attrs(cfg, mat, solver_cfg, wave_speeds: bool = False) -> dict:
        pf = str(getattr(mat, "pf_model", "AT2"))
        law = pf.upper() if pf.upper() in ("AT1", "AT2", "PFCZM") else pf
        if not getattr(solver_cfg, "enable_damage", True):
            law = "none"
        out = dict(name=str(getattr(cfg.material, "preset", None) or "material"), elastic_model="isotropic",
                   E=float(mat.E), nu=float(mat.nu), fracture_law=law, Gc=float(mat.Gc), ell=float(mat.l0),
                   rho=float(mat.rho), energy_split=str(mat.energy_split),
                   degradation_type=str(getattr(mat, "degradation_type", "standard")),
                   stress_degradation=str(getattr(mat, "stress_degradation", "split")),
                   residual_stiffness=float(getattr(mat, "eta_residual", 0.0) or 0.0),
                   driving_force=str(getattr(mat, "driving_force", "strain_energy")))
        units = {"E": "MPa", "Gc": "N/mm", "ell": "mm", "rho": "t/mm^3", "residual_stiffness": "1"}
        if wave_speeds:                                  # layout 2.1: c_p, c_s and the Rayleigh speed c_R of the material
            out.update(c_p=float(mat.c_p), c_s=float(mat.c_s), c_R=float(mat.c_R))
            units.update(c_p="mm/s", c_s="mm/s", c_R="mm/s")
        sigma_ts = float(getattr(mat, "sigma_ts", 0.0) or 0.0)
        if sigma_ts > 0:
            out["strength"] = sigma_ts
            units["strength"] = "MPa"
        out["units"] = units
        return out

    def _loading(self, cfg, reaction_node_set):
        """``loading/bc`` attributes, ``loading/damage_bc`` attributes and the load-parameter definition.

        Prescribed displacements and tractions are ``scale * load_factor``. In
        explicit runs with a time-dependent traction ramp the runner evaluates
        every traction with its own ramp factor (and load factor 1), so each
        traction then points to its own history column ``traction_factor/<key>``.
        """
        comp = {0: "x", 1: "y", "x": "x", "y": "y"}
        bcs, damage_bcs, load_def, k = {}, {}, None, 0
        neumann = list(getattr(self.solver.bcs, "neumann_bcs", None) or [])
        time_ramp = self.explicit and any(getattr(n, "ramp_type", "constant") != "constant" for n in neumann)
        self.traction_columns = {}
        n_traction = 0
        for entry in cfg.boundary_conditions:
            t = entry.type
            key = f"{k:02d}_{entry.nodes}".replace("/", "_")
            k += 1
            a = {"set": entry.nodes.replace("/", "_"), "type": t}
            if t == "fix":
                a.update(kind="displacement", components=comp.get(entry.component, str(entry.component)), value=0.0)
            elif t == "prescribe" and self.layout21 and self.explicit and \
                    getattr(cfg.loading, "ramp_type", "") == "velocity_impact":
                # the solver prescribes the displacement of the velocity ramp; layout 2.1 records the velocity
                a.update(kind="velocity", components=comp.get(entry.component, str(entry.component)),
                         value=float(entry.value) * float(cfg.loading.v0) * 1.0e3, value_units="mm/s",
                         ramp_type="velocity_impact", t_ramp=float(cfg.loading.t_ramp), t_ramp_units="s",
                         displacement="value times u(t) with u = v0 t^2 / (2 t_ramp) for t < t_ramp and "
                                      "v0 (t - t_ramp / 2) afterwards (v0 = value)")
                if load_def is None and float(entry.value) != 0.0:
                    load_def = (entry.nodes, entry.component, float(entry.value))
            elif t == "prescribe":
                a.update(kind="displacement", components=comp.get(entry.component, str(entry.component)),
                         value="load_factor", scale=float(entry.value), scale_units="mm")
                if load_def is None and float(entry.value) != 0.0:
                    load_def = (entry.nodes, entry.component, float(entry.value))
            elif t in ("neumann", "traction"):
                a.update(kind="traction", components=comp.get(entry.component, str(entry.component)),
                         scale=float(entry.value), scale_units="N/mm")
                if t == "traction":
                    a.update(ramp_type=str(entry.ramp_type), t_ramp=float(entry.t_ramp))
                if time_ramp and n_traction < len(neumann):
                    col = f"traction_factor/{key}"
                    self.traction_columns[col] = neumann[n_traction]
                    a.update(value=col)
                else:
                    a.update(value="load_factor")
                n_traction += 1
            elif t == "symmetry":
                a.update(kind="symmetry", components=str(entry.axis), value=0.0)
            elif t == "rigid_connector":
                dofs = entry.dofs if entry.dofs is not None else ["x", "y"]
                a.update(kind="displacement", components=",".join(comp.get(d, str(d)) for d in dofs),
                         constraint="rigid_connector", master=str(entry.master),
                         rotation_free=bool(entry.rotation_free),
                         prescribe=json.dumps({str(c): float(v) for c, v in (entry.prescribe or {}).items()}),
                         value="load_factor")
                if entry.prescribe and load_def is None:
                    c, v = next(iter(entry.prescribe.items()))
                    load_def = (entry.nodes, {"x": 0, "y": 1}.get(c, c), float(v))
            elif t == "pf_dirichlet":
                a.update(components="d", value=float(entry.value), units="1")
                damage_bcs[key] = a
                continue
            else:
                if self.layout21:                        # 2.1 validates the kind of every condition
                    raise ValueError(f"boundary condition type {t!r} has no layout-2.1 kind "
                                     "(displacement, traction, symmetry, periodic, velocity)")
                a.update(kind=t)
            bcs[key] = a
        return bcs, damage_bcs, load_def

    def _write_initial(self, cfg) -> None:
        """``initial_conditions/``: the state before the first increment, with the pre-seeded damage regions."""
        s, w = self.solver, self.writer
        ic = getattr(cfg, "initial_conditions", None)
        spec = None
        if ic is not None and (ic.preseed_notch_nodesets or ic.preseed_damage):
            spec = dict(preseed_notch_nodesets=ic.preseed_notch_nodesets, preseed_damage=ic.preseed_damage)
        w.write_initial("displacement", s.u)
        if "velocity" in w.fields:
            w.write_initial("velocity", s.v)
        if "acceleration" in w.fields:
            w.write_initial("acceleration", s.a)
        w.write_initial("damage", s.d, specification=spec)
        w.write_initial("H", s.H_elem)

    def _reaction_sets(self, cfg, reaction_node_set, node_sets) -> dict:
        names = []
        for entry in cfg.boundary_conditions:
            if entry.type in ("fix", "prescribe", "symmetry", "rigid_connector") and entry.nodes not in names:
                names.append(entry.nodes)
        if reaction_node_set and reaction_node_set not in names:
            names.append(reaction_node_set)
        return {n.replace("/", "_"): n for n in names if n in (getattr(self.mesh0, "node_sets", None) or {})}

    def _history_metadata(self):
        units, attrs = {}, {}
        if self.explicit:
            units.update(time="s", dt="s")
            attrs["time"] = {"definition": "accumulated physical time at the end of the increment"}
        else:
            units.update(time="1", dt="1")
            attrs["time"] = {"definition": "pseudo-time, one unit per load increment (quasi-static)"}
        if self.load_def is not None:
            nodes, c, _scale = self.load_def
            units["load_parameter"] = "mm"
            attrs["load_parameter"] = {"definition": f"prescribed displacement of node set {nodes}, component "
                                                     f"{ {0: 'x', 1: 'y'}.get(c, c) } (load_factor times value)"}
        else:
            units["load_parameter"] = "1"
            attrs["load_parameter"] = {"definition": "load factor (no prescribed displacement in this run)"}
        attrs["load_factor"] = {"definition": "factor multiplying the prescribed boundary values"}
        for col in self.traction_columns:
            units[col] = "1"
            attrs[col] = {"definition": "ramp factor of this traction at the start of the increment"}
        attrs["residual"] = {"definition": ("not applicable (explicit dynamics)" if self.explicit else
                                            "max(u_change, d_change) at stagger acceptance, "
                                            f"criterion {getattr(self.solver.config, 'stagger_criterion', '')}")}
        attrs["stagger_iterations"] = {"definition": "staggered mechanics-damage passes in the increment"}
        attrs["mechanics_iterations"] = {"definition": "iterations of the last mechanics solve"}
        rmethod = ("sum over the set of internal force minus applied external force"
                   + ("; inertia and damping forces not included" if self.explicit else ""))
        for name in self.reaction_sets:
            attrs[f"reaction/{name}"] = {"components": "x,y,z", "method": rmethod,
                                         "sign_convention": "force exerted by the support on the body"}
        q = "2x2 Gauss points times w detJ" if self.q4 else "cell area"
        attrs["energy/elastic"] = {"method": f"sum of (g(d) psi_plus + psi_minus) times {q}"}
        attrs["energy/fracture"] = {"method": f"phase-field crack surface functional ({self.mat.pf_model})"}
        if self.explicit:
            attrs["energy/kinetic"] = {"method": "0.5 rho v.v with lumped nodal mass"}
        attrs["energy/total"] = {"method": "elastic + fracture" + (" + kinetic" if self.explicit else "")}
        if self.traction_loaded:
            attrs["energy/external_work"] = {"method": "applied traction force times displacement increment, "
                                                       "left-point rule, accumulated"}
            attrs["energy/balance_residual"] = {"method": "total - initial total - external_work"}
            attrs["energy/balance_relative"] = {"method": "balance_residual / total"}
            units["energy/balance_relative"] = "1"
        return units, attrs

    @staticmethod
    def _history_update(solver_cfg):
        method = getattr(solver_cfg, "H_update_method", "hard_max") or "hard_max"
        beta = getattr(solver_cfg, "softmax_H_beta", None)
        if method == "hard_max" and beta is not None and beta > 0:
            return "softmax", float(beta)          # legacy softplus form equals log-sum-exp
        if method in ("softmax", "log_smooth") and beta is not None and beta > 0:
            return method, float(beta)
        return method, None

    # ------------------------------------------------------------------ per increment

    def mesh_changed(self) -> bool:
        s = self.solver
        return s.mesh is not self.mesh0 or int(s.mesh.n_nodes) != self.n_nodes0

    def _cell_mean(self, q):
        """(E,) or (E, 4) solver array -> per-cell value (volume-weighted mean for Q4)."""
        a = _np(q)
        if self.q4:
            w = _np(self.solver.mesh.quad_wdetJ)
            return (a * w).sum(axis=1) / w.sum(axis=1)
        return a

    def _derived(self):
        s = self.solver
        strain = s.fem.compute_strain(s.u)
        stress = s.fem.compute_stress(s.u, s.d, strain=strain)
        out, per_point = {}, {}
        for name, comps in (("strain", strain), ("stress", stress)):
            if name not in self.derived_names:
                continue
            out[name] = np.stack([self._cell_mean(c) for c in comps], axis=1)
            if self.per_point:
                per_point[f"Q4/{name}"] = np.stack([_np(c) for c in comps], axis=2)
        if self.with_stress_zz:
            if self.zz_split == "isotropic":
                zz = float(self.mat.nu) * (stress[0] + stress[1])
            else:
                zz = plane_strain_stress_zz(s.fem, s.d, strain)
            out["stress_zz"] = self._cell_mean(zz)
        return out, per_point

    def _energy_balance(self, total: float) -> tuple[float, float, float]:
        """Accumulated loading work, and the residual ``total - initial total - work`` with its relative size.

        The initial total is the stored energy after the first increment, as in ``energy.csv``'s first row; the work
        of an increment is the applied traction force of the increment times the displacement change it caused.
        """
        s = self.solver
        du = s.u.detach() - self.u_before
        self.external_work += float((s.f_ext.detach() * du).sum().item())
        self.u_before = s.u.detach().clone()
        if self.initial_total is None:
            self.initial_total = total
        residual = total - self.initial_total - self.external_work
        scale = max(abs(total), abs(self.initial_total), abs(self.external_work), 1.0e-30)
        return self.external_work, residual, abs(residual) / scale

    def _reactions(self) -> dict:
        if not self.reaction_sets:
            return {}
        s = self.solver
        f = _np(s.fem.internal_force(s.u, s.d)) - _np(s.f_ext)
        out = {}
        for name, idx in self.reaction_idx.items():
            r = f[idx].sum(axis=0)
            out[f"reaction/{name}"] = np.array([r[0], r[1], 0.0])
        return out

    def record(self, *, step: int, dt_inc: float, load_factor: float, energies: dict, last: bool,
               load_time: Optional[float] = None) -> None:
        """Append increment ``step`` (called after the increment has converged).

        ``dt_inc`` is the time step the increment used (quasi-static runs: 1,
        a pseudo-time of one unit per increment). ``load_time`` is the time at
        which the loads of the increment were evaluated (explicit runs: the
        start of the step).
        """
        if self.writer is None or self.writer.closed:
            return
        s = self.solver
        if self.mesh_changed():
            self.stopped_reason = (f"mesh changed at increment {step}; layout 2.0 has no support for "
                                   "meshes that change (mesh_varies), so the file stops before it")
            warnings.warn(f"[result] {self.stopped_reason}. Legacy outputs continue.", RuntimeWarning, stacklevel=2)
            self.writer.f["meta"].attrs["incomplete_reason"] = self.stopped_reason
            self.close("failed")
            return
        solves = int(getattr(s.damage_solver, "n_solves", 0))
        damage_solved = int(self.damage_on and solves > self.solves_seen)
        self.solves_seen = solves
        if self.subcycled and damage_solved:
            self.H_damage_input = s.H_elem.detach().clone()
        self.time += float(dt_inc)
        lf = float(load_factor)
        if self.load_def is not None:
            load_parameter = lf * self.load_def[2]
        else:
            load_parameter = lf
        row = dict(increment=int(step), time=self.time, dt=float(dt_inc), load_parameter=float(load_parameter),
                   load_factor=lf, damage_solved=damage_solved,
                   stagger_iterations=int(getattr(s, "_last_stagger_iter", 0)),
                   mechanics_iterations=int(getattr(s.mechanics, "last_iter", 0) or 0),
                   residual=float(getattr(s, "_last_residual", float("nan"))))
        for col, nbc in self.traction_columns.items():
            row[col] = float(nbc.factor_at(float(load_time or 0.0)))
        row.update(self._reactions())
        row["energy/elastic"] = float(energies["elastic"])
        row["energy/fracture"] = float(energies["fracture"])
        if self.explicit:
            row["energy/kinetic"] = float(energies["kinetic"])
        row["energy/total"] = float(energies["total"])
        if self.traction_loaded:
            external_work, residual, relative = self._energy_balance(float(energies["total"]))
            row["energy/external_work"] = external_work
            row["energy/balance_residual"] = residual
            row["energy/balance_relative"] = relative
        store = (step % self.every == 0) or last or self.stop.requested
        fields = derived = None
        in_view = [n for n in self.derived_names if n not in self.derived_outside]
        need_view = store and bool(in_view or self.per_point)
        need_outside = bool(self.derived_outside) and (step % self.derived_cadence == 0 or last)
        dvals, per_point = self._derived() if (need_view or need_outside) else ({}, {})
        if store:
            fields = dict(displacement=s.u, damage=s.d, H=s.H_elem)
            if self.subcycled:
                fields["H_damage_input"] = self.H_damage_input
            if "damage_lower_bound" in self.writer.fields:
                fields["damage_lower_bound"] = self.d_start
            if "velocity" in self.writer.fields:
                fields["velocity"] = s.v
            if "acceleration" in self.writer.fields:
                fields["acceleration"] = s.a
            fields.update({k: v for k, v in dvals.items() if k not in self.derived_outside})
            fields.update(per_point)
        if need_outside:
            derived = {k: v for k, v in dvals.items() if k in self.derived_outside}
        self.writer.append(row, fields=fields, derived=derived)
        if self.tips:
            tip = locate_tip(s.mesh, s.d, self.tips["threshold"], self.tips["direction"])
            if tip is not None:
                self.writer.append_crack_tip(
                    dict(increment=int(step), tip_id=0, position=tip, threshold=self.tips["threshold"]),
                    method=f"isoline d = {self.tips['threshold']}, farthest along {self.tips['direction']} "
                           "(phast.physics.fracture_mechanics.find_crack_tip), one tip")
        if self.layout21:
            self.writer.flush()
        self.d_start = s.d.detach().clone()
        self.increment = step

    # ------------------------------------------------------------------ closing

    @property
    def stop_requested(self) -> bool:
        return bool(self.writer is not None and self.stop.requested)

    def close(self, status: str = "complete") -> Any:
        if self.writer is None or self.writer.closed:
            return None
        try:
            path = self.writer.close(status)
        finally:
            self.stop.restore()
        print(f"Saved: {os.path.basename(str(path))} (status {status}, {self.writer.n_steps} stored steps, "
              f"{self.writer.n_rows} history rows)", flush=True)
        return path

    def fail(self, exc: BaseException | None = None) -> None:
        if self.writer is None or self.writer.closed:
            return
        if exc is not None:
            try:
                self.writer.f["meta"].attrs["failure"] = f"{type(exc).__name__}: {exc}"[:2000]
            except Exception:
                pass
        self.close("failed")


def _config_dict(cfg, result_settings: dict) -> dict:
    from .config import _dataclass_to_dict
    d = _dataclass_to_dict(cfg)
    d.setdefault("output", {})
    d["output"]["result"] = dict(result_settings)
    return json.loads(json.dumps(d, default=str))


def make_recorder(cfg: Any, **kwargs: Any) -> Optional[RunResultRecorder]:
    """Recorder for this run, or ``None`` when ``output.result`` is false."""
    solver_cfg = kwargs["solver_cfg"]
    settings = resolve_result_settings(getattr(cfg.output, "result", True),
                                       explicit=solver_cfg.solver_type == "explicit",
                                       h5_every=int(getattr(cfg.output, "h5_every", 1) or 1))
    if settings is None:
        return None
    return RunResultRecorder(cfg=cfg, settings=settings, **kwargs)


__all__ = ["RESULT_FILENAME", "RunResultRecorder", "make_recorder", "resolve_result_settings"]
