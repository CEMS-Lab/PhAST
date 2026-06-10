"""phast test suite.

Pytest-discoverable tests for the matrix-free phase-field solver:

- ``test_autodiff.py``  -- gradient checks through the explicit time
                           loop and the implicit-diff CG damage solve
- ``test_config.py``    -- YAML config loading + run-config emission
- ``test_fem_math.py``  -- gather/scatter, B-matrix, mass lumping
- ``test_physics.py``   -- BC bookkeeping, irreversibility, energy
                           monotonicity, AT1 vs AT2 dispatch
- ``test_audit_reproducibility.py`` -- seed determinism + bit-exact
                           replay for the inverse-problem demos

Long-running gradcheck suites for the inverse-problem demos live under
``inverse_kalthoff_stanic/test_gradcheck*.py`` and are picked up by the
inverse-problem CI step (see ``.github/workflows/ci-testing.yml``).
"""
