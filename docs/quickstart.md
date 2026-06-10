# Quick start

## Install

```bash
pip install git+https://github.com/CEMS-Lab/PhAST.git
# Or for development:
git clone https://github.com/CEMS-Lab/PhAST.git
cd phast
pip install -e ".[dev]"
```

For platform-specific PyTorch wheels (CUDA/MPS) and optional preconditioners,
see [Installation](installation.md) or run `bash install.sh` for
auto-detection.

## Your first sparse solve (5 lines)

```python
import torch
from phast.sparse_solve import solve

# Build a 4x4 SPD tridiag K x = b  (autograd-aware adjoint included)
i = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
                  [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]], dtype=torch.long)
v = torch.tensor([2., -1., -1., 2., -1., -1., 2., -1., -1., 2.], dtype=torch.float64)
K = torch.sparse_coo_tensor(i, v, (4, 4)).coalesce()
b = torch.tensor([1., 0., 0., 1.], dtype=torch.float64)
x = solve(K, b)
print(x)  # tensor([1., 1., 1., 1.], dtype=torch.float64)
```

## Try a demo

```bash
python examples/solid_mechanics/linear_plate.py
python examples/solid_mechanics/dynamic_oscillator_genalpha.py
python examples/solid_mechanics/mixed_precision_cg_demo.py
```

## Verify the install

```bash
python -c "from phast.sparse_solve import solve; print('OK')"
pytest tests/test_sparse_solve_autograd.py -x -q
```

## Next steps

- Full reference: `docs/DOCUMENTATION.md` or the
  [docs site](https://cems-lab.github.io/PhAST/)
- Browse `examples/` for benchmarks and tutorial demos
- Issue tracker: https://github.com/CEMS-Lab/PhAST/issues
