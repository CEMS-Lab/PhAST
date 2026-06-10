# Verify install

Use this path when a fresh environment or new machine is being onboarded.

## 1) Runtime smoke test

```bash
python -c "import phast, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m phast --help
python -m phast doctor
python -m phast schema --help
```

`doctor` reports the detected CPU/GPU backend, optional sparse-direct
availability, and the backend selected by `backend='auto'`. For robust
quasi-static, cohesive, and plasticity workflows, PETSc/MUMPS should show as
available on HPC systems where it has been installed.

## 2) Autograd + sparse solve sanity check

```bash
python -c "from phast.sparse_solve import solve; import torch; \
i = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3], [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]], dtype=torch.long); \
v = torch.tensor([2., -1., -1., 2., -1., -1., 2., -1., -1., 2.], dtype=torch.float64); \
K = torch.sparse_coo_tensor(i, v, (4, 4)).coalesce(); b = torch.tensor([1., 0., 0., 1.], dtype=torch.float64); x = solve(K, b); print(x)"
```

Expected output should end with a 4-vector solution close to `[1., 1., 1., 1.]`.

## 3) Validator-level check

```bash
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --validate-only
python -m phast explain-config configs/benchmarks/dynamic/B3_dynamic_sent.yaml
```

## 4) CLI + docs consistency check

```bash
python -m pytest tests/test_config_validation.py tests/test_explain_config_cli.py -q
python scripts/generate_reference_yaml.py --check
python scripts/generate_json_schema.py --check
```

If you keep both checks green, the environment is sufficiently
provisioned for demo runs and CI-like documentation workflows.

## 5) Workflow-default check

For customer-style runs, keep these defaults unless a validation note says
otherwise:

| Workflow | Expected default |
| --- | --- |
| Dynamic explicit | `solver_type: explicit`, `dt_safety: 0.8`, `damage_every: 1` for reference validation. |
| Quasi-static implicit | `solver_type: quasi_static`, `backend: auto`, `preconditioner: jacobi`, `stagger_criterion: linf`. |
| Cohesive contact | Sparse quasi-static mechanics, `backend: auto`, contact penalty only for contact cases. |
| J2 plasticity | Sparse quasi-static guarded material path, `backend: auto`. |

## Next check for production workflows

Run one short smoke benchmark to exercise YAML execution:

```bash
python -m phast run configs/benchmarks/dynamic/B3_dynamic_sent.yaml --device cpu --num_steps 20 --no-plots
```

and confirm it writes `config.yaml`, `run_lockfile.json`, and `results.csv`
into the run folder.
