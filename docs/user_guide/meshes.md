# Mesh formats

Currently-supported mesh formats and the `meshio`-based path for adding more.

`Gc = infinity` (or simply not solving the damage sub-problem) reduces it to a
linear elastic FEM solver. The `StaticSolver` already does this for the
pre-strain step (d=0 everywhere). However, the solver is purpose-built for
phase-field fracture — for general-purpose linear/nonlinear FEM, dedicated
tools (FEniCS, deal.II, Abaqus) are more appropriate.

## Supported Mesh Formats

Currently the solver reads **Gmsh `.msh` files** (via meshio) with 2D linear
triangular elements (T3). The mesh_generator module creates `.geo` files for
standard benchmarks.

For user-supplied geometry, write or export a `.msh` file and reference that
file from YAML:

```yaml
geometry:
  mesh_path: meshes/my_model.msh
```

Boundary conditions refer to named node sets, so the mesh should include Gmsh
physical groups with names that match the YAML:

```yaml
boundary_conditions:
  - {nodes: left, type: fix, component: 0}
  - {nodes: left, type: fix, component: 1}
  - {nodes: right, type: prescribe, component: 0, value: 0.01}
```

If the starting point is a `.geo` file, generate the `.msh` with Gmsh first.
PhAST's YAML runner loads the `.msh`; built-in YAML primitive geometries are
compiled to cached `.geo`/`.msh` artifacts automatically.

Since the mesh loading is built on [meshio](https://github.com/nschloe/meshio),
adding support for other formats is straightforward — meshio can read 40+ mesh
formats including:

| Format | Extension | Tool |
|--------|-----------|------|
| Gmsh | `.msh` | Gmsh (current) |
| Abaqus | `.inp` | Abaqus/CAE |
| VTK/VTU | `.vtk`, `.vtu` | ParaView, VTK |
| ANSYS | `.ans` | ANSYS Mechanical |
| Salome/MED | `.med` | Salome-Meca |
