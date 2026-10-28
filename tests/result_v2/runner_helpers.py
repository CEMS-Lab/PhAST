"""Small decks and a subprocess driver for the tests of the 2D YAML runner's result file.

Each deck is a few increments on a coarse mesh (CPU, float64) with ``output.result`` switched on. ``run`` starts the
runner as users do (``python -m phast run``) through a short driver that, besides
running it, records the final solver state in ``final.npz`` and can inject a
failure, a mesh change or a time-limit signal after a chosen solver step. The
tests compare the stored arrays bit for bit with that final state, so they test
float64 storage, the Q4 point order and the time axis.
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]

_MAT = dict(E=210000.0, nu=0.3, Gc=2.7, l0=0.05, rho=7.8e-9, energy_split="isotropic", pf_model="AT2",
            plane_stress=False)
_BCS = [dict(nodes="bottom", type="fix", component=0), dict(nodes="bottom", type="fix", component=1),
        dict(nodes="top", type="prescribe", component=1, value=1.0)]


def t3_qs(tmp: Path, **output) -> dict:
    """Quasi-static tension of a notched T3 plate, 4 increments."""
    return dict(schema_version=1, problem=dict(name="t3 notched plate"),
                geometry=dict(type="miehe_tension", parameters=dict(output_path=str(tmp / "t3.msh"), L=1.0, a=0.5,
                                                                    l0=0.05, h_crack=0.025, h_coarse=0.2)),
                material=dict(_MAT), boundary_conditions=copy.deepcopy(_BCS),
                loading=dict(protocol="simple", num_steps=4, dt=0.001),
                solver=dict(solver_type="quasi_static", max_stagger=300, stagger_tol=1e-4,
                            fail_on_stagger_nonconvergence=False),
                output=dict(dict(print_every=1, reaction_node_set="bottom", reaction_component=1, result=True), **output),
                device=dict(device="cpu"))


def q4_qs(tmp: Path, **output) -> dict:
    """Quasi-static tension of a structured Q4 plate with a prescribed crack, 3 increments."""
    cfg = t3_qs(tmp, **output)
    cfg["geometry"] = dict(type="rectangular_sent_q4_structured",
                           parameters=dict(output_path=str(tmp / "q4.msh"), W=2.0, H=1.0, a=1.0, l0=0.1,
                                           h_crack=0.125))
    cfg["material"]["l0"] = 0.1
    cfg["loading"] = dict(protocol="simple", num_steps=3, dt=0.0005)
    cfg["boundary_conditions"] = copy.deepcopy(_BCS) + [dict(nodes="crack", type="pf_dirichlet", value=1.0)]
    return cfg


def traction(tmp: Path) -> dict:
    """The dynamic SENT benchmark cut to a few CFL steps, with one ramped traction."""
    cfg = yaml.safe_load((ROOT / "configs/benchmarks/dynamic/B3_dynamic_sent.yaml").read_text(encoding="utf-8"))
    cfg["loading"].update(t_total=3.0e-9, num_steps=0)
    cfg["output"].update(h5=False, result=True)
    top = next(b for b in cfg["boundary_conditions"] if b["nodes"] == "top")
    top.update(ramp_type="linear", t_ramp=1.0e-8)
    return cfg


def dynamic(tmp: Path, **output) -> dict:
    """Explicit dynamics on the T3 plate: 30 steps, damage every third step, adaptive time step."""
    cfg = t3_qs(tmp, h5_every=4, **output)
    cfg["geometry"]["parameters"]["output_path"] = str(tmp / "dyn.msh")
    cfg["boundary_conditions"][2]["value"] = 0.002
    cfg["loading"] = dict(protocol="simple", ramp_type="linear", t_ramp=2.0e-8, num_steps=30, dt=1.0e-9)
    # a tiny threshold makes every step with damage growth cut dt back, so dt varies
    cfg["solver"] = dict(solver_type="explicit", damage_every=3, adaptive_dt=True, adaptive_dt_d_threshold=1e-12)
    cfg["output"]["print_every"] = 100
    return cfg


def kalthoff(tmp: Path) -> dict:
    """The Kalthoff-Winkler benchmark on a coarse mesh, 12 explicit steps."""
    cfg = yaml.safe_load((ROOT / "configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml").read_text(encoding="utf-8"))
    cfg["geometry"]["parameters"].update(h_crack=2.0, h_coarse=12.0, output_path=str(tmp / "kalthoff.msh"))
    cfg["loading"].pop("t_total")
    cfg["loading"].update(num_steps=12, dt=5.0e-8)
    cfg["output"].update(h5=False, h5_every=4, reaction_node_set="left_impact", reaction_component=0, result=True)
    cfg["device"] = dict(device="cpu")
    return cfg


_DRIVER = textwrap.dedent("""
    import atexit, copy, os, signal, sys
    import numpy as np
    from phast.solvers.staggered_solver import StaggeredSolver
    action, at, final = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    last = {}
    original = StaggeredSolver.step_full
    def step_full(self):
        psi = original(self)
        last["solver"] = self
        if action != "none" and self._step_count == at:
            if action == "signal":
                os.kill(os.getpid(), signal.SIGUSR1)
            elif action == "mesh":
                self.mesh = copy.copy(self.mesh)
            elif action == "raise":
                raise RuntimeError("solver failure injected by the test driver")
        return psi
    StaggeredSolver.step_full = step_full
    def save():
        s = last.get("solver")
        if s is None:
            return
        arrays = dict(displacement=s.u.detach().cpu().numpy(), damage=s.d.detach().cpu().numpy(),
                      history_element=s.H_elem.detach().cpu().numpy())
        if s.config.solver_type == "explicit":
            arrays.update(velocity=s.v.detach().cpu().numpy(), acceleration=s.a.detach().cpu().numpy())
        np.savez(final, **arrays)
    atexit.register(save)
    sys.argv = ["run_config"] + sys.argv[4:]
    from phast.config.run_config import main
    main()
""")


def run(tmp: Path, cfg: dict, name: str = "run", inject: tuple | None = None, check: bool = True):
    """Run ``cfg`` and return ``(completed process, output directory)``.

    ``inject`` is ``("signal" | "mesh" | "raise", solver step)``. ``final.npz`` in the output directory holds the
    displacement, damage and element history of the last solver step (explicit runs also velocity, acceleration).
    """
    tmp.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp / f"{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    out = tmp / name
    out.mkdir(parents=True, exist_ok=True)
    driver = tmp / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    action, at = inject or ("none", 0)
    cmd = [sys.executable, str(driver), action, str(at), str(out / "final.npz"), str(cfg_path), "--device", "cpu",
           "--output_dir", str(out)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), SKIP_PRECHECK="1", MPLBACKEND="Agg")
    proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(proc.stdout[-3000:] + proc.stderr[-3000:])
    return proc, out


def validate_file(path: Path) -> None:
    from phast.io.result.validate import validate

    errors = [x for x in validate(path).findings if x.level == "ERROR"]
    assert errors == []


requires_sigusr1 = pytest.mark.skipif(not hasattr(__import__("signal"), "SIGUSR1"),
                                      reason="the time-limit signal is SIGUSR1 or SIGTERM; the test sends SIGUSR1")
