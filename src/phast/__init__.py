"""
phast — Modular PyTorch Phase-Field Fracture Solver
=============================================================

A vectorized, GPU-compatible FEM solver for AT1/AT2 phase-field fracture
on unstructured triangular meshes.

Quick start::

    from phast import Problem

    solver = (Problem('SENT')
        .geometry('rectangular_sent', W=100, H=40, a=50, h_crack=0.5)
        .material('glass_borden', l0=0.5, energy_split='spectral')
        .fix('left', dof='x').neumann('top', dof='y', value=1.0)
        .loading(protocol='simple', t_total=80e-6)
        .device('cpu').run())

Or from YAML::

    python -m phast run configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml

Modules
-------
mesh_generator      : Gmsh API mesh generation (Miehe benchmarks, square plate)
mesh                : Gmsh mesh loading and FEM precomputation
material            : Material properties (elastic + fracture)
boundary_conditions : Dirichlet and Neumann BC specification
fem_operators       : Vectorized FEM operators (strain, stress, force, psi+)
damage_solver       : AT1/AT2 damage CG solver
mechanics_solver    : Displacement solvers (explicit, static, quasi-static, L-BFGS)
staggered_solver    : Orchestrator coupling mechanics + damage
device              : Device management and profiling
io_utils            : VTU / Zarr / legacy H5 / CSV output
problem             : Fluent Problem builder API
adaptive            : Adaptive mesh refinement (newest vertex bisection)
"""

# Guard against being imported under the wrong package name. When pytest
# collects modules from a clone whose directory name is not
# ``phast`` (for example ``phast-feature-x``), Python may try to import this file as
# part of the rootdir scan. ``__name__`` is then something other than
# ``phast`` and the relative imports below would fail with
# ``attempted relative import with no known parent package``.
#
# Earlier the guard raised ``ImportError`` to signal "wrong context";
# but pytest reports that error against any test module that triggers
# the parent package import indirectly, which broke fresh-clone CI
# (issue #403 follow-up). The fix is to silently skip the relative
# imports when the package is not loaded under its package name —
# legitimate ``import phast`` uses (after ``pip install -e
# .``) hit the package name and execute the full body.

if __name__ == "phast":
    from .mesh import FEMMesh, build_node_adjacency
    from .material import Material, create_material
    from .boundary_conditions import (
        BoundaryConditions, DirichletBC, NeumannBC,
        symmetric_tension_bcs, shear_bcs,
    )
    from .fem_operators import FEMOperators
    from .damage_solver import PhaseFieldDamageSolver
    from .mechanics_solver import (
        ExplicitDynamics, StaticSolver, QuasiStaticSolver, SecantCGSolver,
        DirectSolver, LBFGSSolver, MonolithicSolver,
    )
    from .staggered_solver import StaggeredSolver, SolverConfig
    from .learned_damage import (
        DamageDecision, DamagePrediction, DamagePredictionRejected,
        DamagePredictor, DamageStepContext, DamageUpdateController,
        load_damage_predictor,
    )
    from .device import DeviceContext, detect_device, get_device_tier, estimate_vram_mb, Profiler
    from .problem import Problem
    from .result import Result, ResultLoadError, load_result
    from .mesh_inspection import inspect_mesh
    from .region_resolution import RegionResolutionError, resolve_regions
    from .workflow import (
        AnalysisStep, BoundaryCondition, FieldOutput, Geometry, HistoryOutput,
        InitialCondition, Mesh, Outputs, Postprocess, Region, SolverSettings,
    )
    from .fracture_mechanics import compute_j_integral, compute_sif, find_crack_tip
    from .mesh_generator import miehe_tension, miehe_shear, square_plate, three_point_bending, bazant_gap_test
    from .io_utils import (
        write_vtu, write_pvd, init_zarr, write_zarr_snapshot,
        init_h5, write_h5_snapshot, load_state_from_h5, load_state_from_zarr,
        compute_edge_index, write_profiler_csv, CSVHistory,
    )
    from .p2_elements import (
        P2_REF_NODES, p2_shape_functions, p2_shape_function_derivs,
        p2_gauss_points, p2_element_stiffness, p2_mesh_from_p1,
        p2_node_indices_from_p1_mesh,
    )
    from .quad_elements import (
        Q4_REF_NODES, Q8_REF_NODES, Q9_REF_NODES, quad_ref_nodes,
        quad_shape_functions, quad_shape_function_derivs, quad_gauss_points,
        quad_element_stiffness, q4_internal_force, q4_laplacian_matvec,
        q4_mass_matvec, q4_quality, q4_quadrature_geometry, q4_signed_areas,
        q4_strain_at_gauss, q4_to_triangles, structured_q4_mesh,
    )
    from .visualization import (
        compute_von_mises_stress, compute_von_mises_strain,
        compute_principal_stress, compute_principal_strain,
        compute_field, FIELD_REGISTRY, GIFRecorder,
        plot_field, plot_damage_stress_strain,
        plot_initial_conditions, plot_final_state,
    )
    from .postprocess_hdf5 import (
        PostProcessor,
        crack_path_error, energy_error, peak_load_error, damage_field_metrics,
    )
    try:
        from .multigrid import NodeAggregation, ScalarMultigrid, AMGPreconditioner, AmgXPreconditioner, VectorMultigrid
    except ImportError:
        pass
    try:
        from .adaptive import (
            compute_refinement_indicator, refine_mesh,
            interpolate_field, interpolate_elem_field,
        )
    except ImportError:
        pass

    __all__ = [
        # Core
        'FEMMesh', 'Material', 'create_material',
        'BoundaryConditions', 'DirichletBC', 'NeumannBC',
        'FEMOperators', 'PhaseFieldDamageSolver',
        'ExplicitDynamics', 'StaticSolver', 'QuasiStaticSolver', 'SecantCGSolver',
        'LBFGSSolver', 'MonolithicSolver',
        'StaggeredSolver', 'SolverConfig',
        'DamageDecision', 'DamagePrediction', 'DamagePredictionRejected',
        'DamagePredictor', 'DamageStepContext', 'DamageUpdateController',
        'load_damage_predictor',
        # Device / performance
        'DeviceContext', 'detect_device', 'get_device_tier', 'estimate_vram_mb', 'Profiler',
        # Convenience
        'Problem', 'Geometry', 'Mesh', 'Region', 'InitialCondition',
        'BoundaryCondition', 'AnalysisStep', 'FieldOutput', 'HistoryOutput',
        'Outputs', 'Postprocess', 'SolverSettings', 'Result',
        'ResultLoadError', 'load_result', 'inspect_mesh',
        'RegionResolutionError', 'resolve_regions',
        'compute_j_integral', 'compute_sif', 'find_crack_tip',
        'DirectSolver', 'build_node_adjacency',
        'symmetric_tension_bcs', 'shear_bcs',
        # Mesh generation
        'miehe_tension', 'miehe_shear', 'square_plate', 'three_point_bending',
        'bazant_gap_test',
        # IO utilities
        'write_vtu', 'write_pvd', 'init_zarr', 'write_zarr_snapshot',
        'init_h5', 'write_h5_snapshot', 'load_state_from_h5',
        'load_state_from_zarr',
        'compute_edge_index', 'write_profiler_csv', 'CSVHistory',
        'P2_REF_NODES', 'p2_shape_functions', 'p2_shape_function_derivs',
        'p2_gauss_points', 'p2_element_stiffness', 'p2_mesh_from_p1',
        'p2_node_indices_from_p1_mesh',
        'Q4_REF_NODES', 'Q8_REF_NODES', 'Q9_REF_NODES', 'quad_ref_nodes',
        'quad_shape_functions', 'quad_shape_function_derivs',
        'quad_gauss_points', 'quad_element_stiffness',
        'q4_internal_force', 'q4_laplacian_matvec', 'q4_mass_matvec',
        'q4_quality', 'q4_quadrature_geometry', 'q4_signed_areas',
        'q4_strain_at_gauss', 'q4_to_triangles', 'structured_q4_mesh',
        # Visualization & post-processing
        'compute_von_mises_stress', 'compute_von_mises_strain',
        'compute_principal_stress', 'compute_principal_strain',
        'compute_field', 'FIELD_REGISTRY', 'GIFRecorder',
        'plot_field', 'plot_damage_stress_strain',
        'plot_initial_conditions', 'plot_final_state',
        'PostProcessor',
        'crack_path_error', 'energy_error', 'peak_load_error', 'damage_field_metrics',
        # Multigrid preconditioner
        'NodeAggregation', 'ScalarMultigrid', 'AMGPreconditioner', 'AmgXPreconditioner',
        'VectorMultigrid',
        # Adaptive mesh refinement
        'compute_refinement_indicator', 'refine_mesh',
        'interpolate_field', 'interpolate_elem_field',
    ]
