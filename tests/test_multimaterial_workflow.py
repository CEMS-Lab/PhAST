from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from phast.workflow import execution_plan_from_spec, problem_spec_from_yaml
from phast.workflow import multimaterial_fracture as backend
from phast.workflow.multimaterial_fracture import (
    MultimaterialWorkflowError, _selector_mask, validate_multimaterial_fracture_spec,
)
from phast.workflow.specs import FieldOutputSpec, HistoryOutputSpec, InitialConditionSpec, PostprocessSpec


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "two_material_dcb_beta" / "config.yaml"
SENT_CONFIG = ROOT / "examples" / "heterogeneous_sent_beta" / "config.yaml"


def test_dcb_config_uses_standard_executable_workflow():
    spec = problem_spec_from_yaml(CONFIG)
    plan = execution_plan_from_spec(spec)
    assert spec.schema_version == 2
    assert len(spec.materials) == 3
    assert plan.route == "run_config"
    assert plan.direct_execution_supported


def test_heterogeneous_sent_uses_standard_executable_workflow():
    spec = problem_spec_from_yaml(SENT_CONFIG)
    plan = execution_plan_from_spec(spec)
    assert spec.schema_version == 2
    assert len(spec.materials) == 2
    assert plan.route == "run_config"
    assert plan.direct_execution_supported


def test_circle_regions_partition_points():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    circle = {"type": "circle", "center": [1.0, 0.0], "radius": 0.25}
    outside = {"type": "outside_circle", "center": [1.0, 0.0], "radius": 0.25}
    inside_mask = _selector_mask(points, circle)
    outside_mask = _selector_mask(points, outside)
    assert np.array_equal(inside_mask, np.array([False, True, False]))
    assert np.array_equal(outside_mask, ~inside_mask)


def test_composable_region_selectors_partition_weak_layer():
    points = np.array([[0.0, 0.0], [0.0, 0.2], [1.0, 0.0]])
    circle = {"type": "circle", "center": [1.0, 0.0], "radius": 0.25}
    weak = {
        "type": "all_of",
        "selectors": [
            {"type": "rectangle", "origin": [0.0, -0.1], "size": [2.0, 0.2]},
            {"type": "not", "selector": circle},
        ],
    }
    assert np.array_equal(
        _selector_mask(points, weak), np.array([True, False, False]))


@pytest.fixture
def dcb_spec():
    return problem_spec_from_yaml(CONFIG)


def _material_change(spec, **parameters):
    materials = list(spec.materials)
    materials[-1] = replace(materials[-1], parameters={**materials[-1].parameters, **parameters})
    return replace(spec, materials=materials)


def _region_change(spec, name, selector):
    return replace(spec, regions=[replace(r, selector=selector) if r.name == name else r for r in spec.regions])


def _geometry_change(spec, **parameters):
    return replace(spec, geometry=replace(spec.geometry, parameters={**spec.geometry.parameters, **parameters}))


@pytest.mark.parametrize("config", [CONFIG, SENT_CONFIG])
def test_valid_preflight_never_constructs_a_kernel_or_creates_output(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preflight must not construct kernels or create output")
    monkeypatch.setattr(backend.FEMMesh, "from_tensors", forbidden)
    monkeypatch.setattr(backend, "FEMOperators", forbidden)
    monkeypatch.setattr(backend, "QuasiStaticSolver", forbidden)
    monkeypatch.setattr(backend, "PhaseFieldDamageSolver", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    assert validate_multimaterial_fracture_spec(problem_spec_from_yaml(config)) is None


@pytest.mark.parametrize("key", ["E", "Gc", "l0", "rho"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_material_is_rejected(dcb_spec, key, value):
    with pytest.raises(MultimaterialWorkflowError, match=key):
        validate_multimaterial_fracture_spec(_material_change(dcb_spec, **{key: value}))


@pytest.mark.parametrize("nu", [-1.0, 0.5, float("nan"), float("inf")])
def test_invalid_poisson_ratio(dcb_spec, nu):
    with pytest.raises(MultimaterialWorkflowError, match="nu"):
        validate_multimaterial_fracture_spec(_material_change(dcb_spec, nu=nu))


@pytest.mark.parametrize("key,value", [("eta_residual", .01), ("rho", 2.), ("l0", .2), ("nu", .25), ("plane_stress", False)])
def test_unsupported_material_differences_are_not_ignored(dcb_spec, key, value):
    with pytest.raises(MultimaterialWorkflowError, match=key):
        validate_multimaterial_fracture_spec(_material_change(dcb_spec, **{key: value}))


def test_unknown_material_parameter(dcb_spec):
    with pytest.raises(MultimaterialWorkflowError, match="unsupported material"):
        validate_multimaterial_fracture_spec(_material_change(dcb_spec, viscosity=1.))


@pytest.mark.parametrize("length,message", [(5., "selects no nodes"), (8., "interior nodes")])
def test_stale_coordinate_clamp_rejected(dcb_spec, length, message):
    spec = _region_change(dcb_spec, "right_clamp", {"type": "boundary", "axis": "x", "value": 6.})
    with pytest.raises(MultimaterialWorkflowError, match=message):
        validate_multimaterial_fracture_spec(_geometry_change(spec, length=length))


def test_empty_seed_rejected_before_solver_and_output(dcb_spec, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached kernel construction or output creation")
    monkeypatch.setattr(backend.FEMMesh, "from_tensors", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    with pytest.raises(MultimaterialWorkflowError, match="initial damage region.*selects no nodes"):
        backend.run_multimaterial_fracture_spec(_geometry_change(dcb_spec, ny=47))


def test_empty_material2_region(dcb_spec):
    spec = _region_change(dcb_spec, "material_2_region", {"type": "circle", "center": [20., 0.], "radius": .2})
    with pytest.raises(MultimaterialWorkflowError, match="material_2_region.*selects no elements"):
        validate_multimaterial_fracture_spec(spec)


@pytest.mark.parametrize("side,axis,end", [("left", 0, 0), ("right", 0, 1), ("bottom", 1, 0), ("top", 1, 1)])
def test_side_selector_uses_full_nodal_bounds(dcb_spec, side, axis, end):
    nodes, elements = backend._build_mesh(dcb_spec)
    bounds = (nodes.min(axis=0), nodes.max(axis=0))
    selector = {"type": "boundary", "side": side}
    mask = _selector_mask(nodes, selector, nodal_bounds=bounds)
    assert np.array_equal(mask, np.isclose(nodes[:, axis], bounds[end][axis]))
    assert not _selector_mask(nodes[elements].mean(axis=1), selector, nodal_bounds=bounds).any()
    with pytest.raises(MultimaterialWorkflowError, match="structured rectangle"):
        _selector_mask(nodes, selector)


def test_right_side_preserves_reference_nodes_and_tracks_length(dcb_spec):
    nodes, elements = backend._build_mesh(dcb_spec)
    old, _ = backend._resolve_regions(dcb_spec, nodes, elements)
    spec = _region_change(dcb_spec, "right_clamp", {"type": "boundary", "side": "right", "tolerance": 1.e-8})
    new, _ = backend._resolve_regions(spec, nodes, elements)
    assert np.array_equal(old["right_clamp"], new["right_clamp"])
    validate_multimaterial_fracture_spec(_geometry_change(spec, length=8.))


@pytest.mark.parametrize("condition", [InitialConditionSpec(field="displacement", region="initial_crack", value=0.), InitialConditionSpec(field="damage", region="initial_crack", value=2.), InitialConditionSpec(field="damage", region="initial_crack", value=float("nan"))])
def test_invalid_initial_condition(dcb_spec, condition):
    with pytest.raises(MultimaterialWorkflowError):
        validate_multimaterial_fracture_spec(replace(dcb_spec, initial_conditions=[condition]))


@pytest.mark.parametrize("field", ["history_field_nodal", "psi_plus", "velocity", "unknown"])
def test_unimplemented_field_requests(dcb_spec, field):
    outputs = replace(dcb_spec.outputs, fields=[FieldOutputSpec(name=field)])
    with pytest.raises(MultimaterialWorkflowError, match="field output"):
        validate_multimaterial_fracture_spec(replace(dcb_spec, outputs=outputs))


def test_energy_history_rejected(dcb_spec):
    outputs = replace(dcb_spec.outputs, history=[HistoryOutputSpec(name="energy")])
    with pytest.raises(MultimaterialWorkflowError, match="history output"):
        validate_multimaterial_fracture_spec(replace(dcb_spec, outputs=outputs))


def test_unimplemented_animation_and_trajectory_format(dcb_spec):
    for outputs in [replace(dcb_spec.outputs, postprocess=[PostprocessSpec(kind="animation", parameters={"field": "stress"})]), replace(dcb_spec.outputs, fields=[FieldOutputSpec(name="trajectory", parameters={"format": "zarr"})])]:
        with pytest.raises(MultimaterialWorkflowError):
            validate_multimaterial_fracture_spec(replace(dcb_spec, outputs=outputs))


def test_reaction_request_honours_region_and_component(dcb_spec):
    nodes, elements = backend._build_mesh(dcb_spec)
    regions, _ = backend._resolve_regions(dcb_spec, nodes, elements)
    bcs = backend._resolve_bcs(dcb_spec, regions)
    reference_indices, reference_component = backend._resolve_reaction(dcb_spec, regions, bcs)
    assert np.array_equal(reference_indices.numpy(), np.flatnonzero(regions["upper_load"]))
    assert reference_component == 1
    output = replace(dcb_spec.outputs, history=[HistoryOutputSpec(name="reaction_force", region="right_clamp", component=0)])
    spec = replace(dcb_spec, outputs=output)
    validate_multimaterial_fracture_spec(spec)
    indices, component = backend._resolve_reaction(spec, regions, bcs)
    assert np.array_equal(indices.numpy(), np.flatnonzero(regions["right_clamp"]))
    assert component == 0
