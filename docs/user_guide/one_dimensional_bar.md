# One-dimensional elastic bar

The beta 1D Python API generates a line mesh and solves small-strain axial
elasticity with PhAST's sparse autograd solver. It uses CPU float64 tensors,
one displacement per node, uniform positive Young's modulus and area, a
prescribed displacement at the left endpoint and a nodal force at the right.
The API is independent of the 2D `FEMMesh` and `Problem` workflow.

## Geometry and mesh

```python
import phast
import torch

mesh = phast.line_mesh(length=100.0, n_elements=10)
print(mesh.nodes)             # 11 coordinates
print(mesh.elements)          # 10 pairs of adjacent node indices
print(mesh.element_lengths)  # ten lengths of 10 mm
```

Use one consistent unit system. Here length is in mm, force in N, area in mm²
and Young's modulus in N/mm² (MPa). No external mesher is required. For
nonuniform spacing, use increasing CPU float64 coordinates:

```python
mesh = phast.LineMesh(torch.tensor([0., 10., 30., 100.], dtype=torch.float64))
```

`LineMesh` preserves dependence on tensor coordinates on this fixed topology.
The uniform `line_mesh` convenience function accepts numerical length and
origin, with integer `n_elements >= 1`.

## Material, boundary condition and solve

```python
mesh = phast.line_mesh(length=100.0, n_elements=10)
result = phast.solve_bar(
    mesh,
    young_modulus=210000.0,
    area=10.0,
    end_force=4000.0,
    left_displacement=0.0,
)
print(result.displacement[-1])  # 0.190476... mm
print(result.reaction)          # -4000 N
print(result.free_residual.abs().max())
```

Positive force pulls the right endpoint along the positive coordinate axis.
The left endpoint is fixed by default. A nonzero `left_displacement` translates
the complete axial displacement field without changing strain. Poisson's
ratio is unnecessary for this 1D axial model.

Each two-node element has stiffness

```{math}
\mathbf K_e = \frac{EA}{h_e}
\begin{bmatrix}1 & -1\\-1 & 1\end{bmatrix}.
```

PhAST assembles sparse contributions, eliminates the constrained degree of
freedom and calls `SparseSolveAutograd`, backed by SciPy SuperLU. One call is
one static linear solve. Repeated forces produce separate equilibrium states.

For a uniform bar, the analytical solution is

```{math}
u(x)=u_0+\frac{F(x-x_0)}{EA},\qquad \sigma=\frac{F}{A}.
```

The returned `BarResult` contains nodal `displacement`, scalar left `reaction`,
`free_residual`, and elementwise `strain`, `stress` and `axial_force`. These
tensors retain their first-order computational graphs.

## Displacement sensitivity

```python
E = torch.tensor(210000.0, dtype=torch.float64, requires_grad=True)
tip = phast.solve_bar(mesh, E, area=10.0, end_force=4000.0).displacement[-1]
du_dE = torch.autograd.grad(tip, E)[0]
print(du_dE)  # approximately -9.07029e-7 mm / MPa
```

The analytical derivative is $-FL/(AE^2)$. PhAST's sparse backward rule solves
the transposed equilibrium system and returns derivatives to the tensor
inputs. First-order sensitivities are tested for modulus, area, end force,
prescribed displacement and fixed-topology coordinate changes.

## Scope and verification

This beta API covers static, homogeneous axial elasticity and CPU float64
execution. It accepts scalar material and load inputs. Transverse motion,
bending, fracture evolution, body loads, multiple supports, YAML routing,
automatic result directories and higher-order derivatives are outside this
small API's tested scope. The existing 2D fracture pathways are unchanged.

Run the focused verification from the repository root:

```bash
PYTHONPATH=src python -m pytest tests/test_line_bar.py -q
```

The tests cover uniform and nonuniform meshes, one-element and multi-element
solutions, force balance, prescribed translations, zero/compressive/tensile
loads, input validation, analytical derivatives and finite-difference
`gradcheck` of the sparse backward rule.
