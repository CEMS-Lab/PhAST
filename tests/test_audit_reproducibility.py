"""Smoke tests for the paper audit pipeline.

Each test runs one of the paper demos / audit scripts with tiny inputs
and asserts that (a) it completes, (b) the JSON sidecar is well-formed,
and (c) the headline invariant holds to a loose tolerance. These run
in <30 s on CPU; they are not verification tests for the paper numbers
(those come from the full HPC run -- see
``papers/paper/audit_claims/README.md``) but they catch silent breakage of
the pipeline after code changes.

Run just these via::

    python -m pytest tests/test_audit_reproducibility.py -v
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile

import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_PARENT = os.path.dirname(REPO_DIR)


def _run_module(module: str, args: list, env_extra: dict | None = None) -> str:
    """Run ``python -m <module> <args>`` from the repo parent.

    Returns the combined stdout+stderr. Raises CalledProcessError if the
    subprocess exits non-zero.
    """
    env = os.environ.copy()
    env['PYTHONPATH'] = (PACKAGE_PARENT + os.pathsep
                        + env.get('PYTHONPATH', ''))
    if env_extra:
        env.update(env_extra)
    res = subprocess.run(
        [sys.executable, '-m', module, *args],
        cwd=PACKAGE_PARENT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"{module} failed (exit {res.returncode}):\n"
            f"--- stdout ---\n{res.stdout}\n"
            f"--- stderr ---\n{res.stderr}")
    return res.stdout + res.stderr


@pytest.mark.slow
def test_cost_breakdown_smoke(tmp_path):
    """test_cost_breakdown.py must produce a JSON whose disjoint
    percentages sum to (approximately) 100 %."""
    out_path = tmp_path / 'results.json'
    _run_module(
        'phast.paper.audit_claims.test_cost_breakdown',
        ['--h_crack', '2.0',        # coarse mesh, ~8 k nodes
         '--t_total_us', '3.0',     # ~40 steps
         '--preconditioner', 'jacobi',
         '--out', str(out_path)])
    assert out_path.exists(), "JSON not written"
    with open(out_path) as fh:
        data = json.load(fh)

    # The five disjoint categories the patched instrumentation produces.
    cats = ['mechanics_excl_strain', 'strain', 'psi_plus_H_excl_strain',
            'damage_cg', 'bookkeeping']
    for k in cats:
        assert k in data['percent'], f"missing percent[{k}]"

    total = sum(data['percent'][k] for k in cats)
    assert abs(total - 100.0) < 1.0, (
        f"Disjoint categories must sum to 100 %, got {total:.2f}%")


@pytest.mark.slow
def test_sensitivity_smoke(tmp_path):
    """demo_sensitivity.py with a tiny delta sweep must produce a JSON
    whose autograd/FD agreement is tight at the middle delta value."""
    _run_module(
        'phast.paper.demos.demo_sensitivity',
        ['--n_steps', '5',
         '--delta_min_exp', '-6',
         '--delta_max_exp', '-2',
         '--delta_steps', '5',
         '--output_dir', str(tmp_path)])
    json_path = tmp_path / 'demo_sensitivity.json'
    assert json_path.exists(), "JSON not written"
    with open(json_path) as fh:
        data = json.load(fh)

    assert len(data['deltas']) == 5
    assert len(data['fd_grads']) == 5
    assert len(data['rel_errors']) == 5

    # Loose tolerance -- the 5-step test problem has a constant bias
    # of ~2e-4 between autograd and FD, but it should never blow up.
    min_err = min(data['rel_errors'])
    assert min_err < 1e-2, (
        f"Best autograd-vs-FD agreement was {min_err:.2e}, "
        f"expected < 1e-2 for this tiny smoke run")


@pytest.mark.slow
def test_inversion_adam_smoke(tmp_path):
    """demo_inversion.py Adam path with 2 iters must produce a JSON
    whose loss history has the expected length."""
    _run_module(
        'phast.paper.demos.demo_inversion',
        ['--n_steps', '5',
         '--n_iters', '2',
         '--optimizer', 'adam',
         '--output_dir', str(tmp_path)])
    json_path = tmp_path / 'demo_inversion.json'
    assert json_path.exists(), "JSON not written"
    with open(json_path) as fh:
        data = json.load(fh)
    assert data['optimizer'] == 'adam'
    assert len(data['history']['loss']) == 2
    assert data['total_forward_evals'] == 2 * 3  # 3 evals per Adam iter


@pytest.mark.slow
def test_inversion_lbfgs_smoke(tmp_path):
    """demo_inversion.py L-BFGS path must produce a JSON with a
    forward_evals counter and a well-formed history."""
    _run_module(
        'phast.paper.demos.demo_inversion',
        ['--n_steps', '5',
         '--n_iters', '2',
         '--optimizer', 'lbfgs',
         '--output_dir', str(tmp_path)])
    json_path = tmp_path / 'demo_inversion_lbfgs.json'
    assert json_path.exists(), "JSON not written"
    with open(json_path) as fh:
        data = json.load(fh)
    assert data['optimizer'] == 'lbfgs'
    assert data['total_forward_evals'] >= 3, (
        "L-BFGS should cost at least 3 forward evals even in 1 iteration")
    assert len(data['history']['iter']) >= 1
