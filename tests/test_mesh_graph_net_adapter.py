"""Tests for the mesh-graph-network learned-damage adapter.

The adapter is one architecture among several. What is checked here is the
wrapper: its feature contract, its checkpoint handling and its conformance to
the predictor protocol. The finite-element helpers it builds on are covered by
``test_damage_step_context_fem_helpers``.
"""
from __future__ import annotations

import pytest
import torch

from phast.learned_damage import DamageStepContext

from examples.learned_damage.architectures.mesh_graph_net import (
    MeshGraphDamageNet,
    MeshGraphNetPredictor,
    bounding_box_indicator,
    create_predictor,
    export_torchscript,
    load_network,
)

DTYPE = torch.float64

NODES = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], dtype=DTYPE)
ELEMENTS = torch.tensor(
    [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=torch.long)


def make_context(*, history=1.0, Gc=2.7, l0=0.4):
    n_nodes = NODES.shape[0]
    return DamageStepContext(
        step=1,
        time=0.0,
        load_factor=1.0,
        nodes=NODES,
        elements=ELEMENTS,
        displacement=torch.zeros((n_nodes, 2), dtype=DTYPE),
        velocity=torch.zeros((n_nodes, 2), dtype=DTYPE),
        history_element=torch.full((ELEMENTS.shape[0],), history, dtype=DTYPE),
        history_nodal=torch.zeros((n_nodes,), dtype=DTYPE),
        damage_previous=torch.zeros((n_nodes,), dtype=DTYPE),
        material={"Gc": Gc, "l0": l0, "pf_model": "AT2"},
        phase_field_model="AT2",
        energy_split="spectral",
        device=torch.device("cpu"),
        dtype=DTYPE,
    )


def make_net():
    torch.manual_seed(0)
    return MeshGraphDamageNet(
        latent_dim=4, hidden_layers=1, message_passing_steps=2)


def random_graph_inputs(context):
    edge_index = context.graph_edge_index()
    return (
        torch.rand(NODES.shape[0], 2),
        edge_index,
        torch.rand(edge_index.shape[1], 1),
    )


def save_state_dict(path, net):
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "configuration": {
                "latent_dim": 4, "hidden_layers": 1, "message_passing_steps": 2},
            "completed_epochs": 3,
        },
        path,
    )
    return path


# ---------------------------------------------------------------- network

def test_network_returns_one_value_per_node():
    net = make_net()
    with torch.no_grad():
        output = net(*random_graph_inputs(make_context()))
    assert output.shape == (NODES.shape[0],)


def test_network_is_scriptable_and_agrees_with_the_eager_model():
    net = make_net().eval()
    inputs = random_graph_inputs(make_context())
    scripted = torch.jit.script(net)
    with torch.no_grad():
        assert torch.equal(net(*inputs), scripted(*inputs))


# ---------------------------------------------------------------- features

def test_bounding_box_indicator_ignores_the_interior():
    assert bounding_box_indicator(NODES).tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_node_features_use_the_assembled_normalized_history():
    predictor = MeshGraphNetPredictor(make_net())
    context = make_context(history=1.0, Gc=2.7, l0=0.4)

    features = predictor.node_features(context)
    expected = context.assemble_element_to_nodes(
        context.history_element / (2.7 * 0.4))

    assert features.shape == (NODES.shape[0], 2)
    assert torch.allclose(features[:, 0], expected)
    assert torch.allclose(features[:, 1], bounding_box_indicator(NODES))


def test_topological_boundary_indicator_can_be_selected():
    context = make_context()
    predictor = MeshGraphNetPredictor(
        make_net(), boundary_indicator="topological")
    features = predictor.node_features(context)
    assert torch.allclose(
        features[:, 1], context.boundary_node_mask().to(DTYPE))


def test_an_unknown_boundary_indicator_is_rejected():
    with pytest.raises(ValueError, match="bounding_box.*topological"):
        MeshGraphNetPredictor(make_net(), boundary_indicator="guess")


def test_overrides_take_precedence_over_the_running_material():
    context = make_context(Gc=2.7, l0=0.4)
    default = MeshGraphNetPredictor(make_net()).node_features(context)
    overridden = MeshGraphNetPredictor(
        make_net(), Gc=1.0, length_scale=1.0).node_features(context)
    assert not torch.allclose(default[:, 0], overridden[:, 0])


def test_a_material_without_the_normalization_constants_is_rejected():
    stripped = DamageStepContext(
        **{**make_context().__dict__, "material": {"pf_model": "AT2"}})
    with pytest.raises(ValueError, match="exposes no 'Gc'"):
        MeshGraphNetPredictor(make_net()).node_features(stripped)


def test_graph_is_cached_between_calls_on_one_mesh():
    predictor = MeshGraphNetPredictor(make_net())
    context = make_context()
    assert predictor._graph(context)[0] is predictor._graph(context)[0]


def test_graph_cache_invalidates_for_a_different_same_sized_mesh():
    predictor = MeshGraphNetPredictor(make_net())
    context = make_context()
    first_lengths = predictor._graph(context)[1]
    scaled = DamageStepContext(
        **{**context.__dict__, "nodes": context.nodes * 2.0})
    second_lengths = predictor._graph(scaled)[1]
    assert not torch.equal(first_lengths, second_lengths)


# ---------------------------------------------------------------- protocol

def test_predict_returns_one_nodal_value_in_the_context_dtype():
    context = make_context()
    prediction = MeshGraphNetPredictor(make_net()).predict(context)
    assert prediction.damage.shape == context.damage_previous.shape
    assert prediction.damage.dtype == context.dtype
    assert prediction.diagnostics["adapter"] == "mesh-graph-net"
    assert prediction.diagnostics["boundary_indicator"] == "bounding_box"


def test_increment_representation_is_passed_through():
    predictor = MeshGraphNetPredictor(
        make_net(), representation="damage_increment")
    assert predictor.predict(make_context()).representation == "damage_increment"


# ---------------------------------------------------------------- checkpoints

def test_state_dict_checkpoint_round_trips(tmp_path):
    net = make_net().eval()
    loaded, metadata = load_network(save_state_dict(tmp_path / "ck.pt", net))

    assert metadata["format"] == "state_dict"
    assert metadata["completed_epochs"] == 3

    inputs = random_graph_inputs(make_context())
    with torch.no_grad():
        assert torch.equal(net(*inputs), loaded(*inputs))


def test_a_checkpoint_without_architecture_entries_is_rejected(tmp_path):
    path = tmp_path / "incomplete.pt"
    torch.save({"model_state_dict": make_net().state_dict()}, path)
    with pytest.raises(ValueError, match="missing architecture entries"):
        load_network(path)


def test_an_unrecognized_file_is_rejected(tmp_path):
    path = tmp_path / "unrelated.pt"
    torch.save({"something": 1}, path)
    with pytest.raises(ValueError, match="neither a TorchScript archive"):
        load_network(path)


def test_torchscript_export_loads_and_predicts(tmp_path):
    net = make_net().eval()
    archive = export_torchscript(
        save_state_dict(tmp_path / "ck.pt", net),
        tmp_path / "scripted.pt",
        n_nodes=16,
    )

    loaded, metadata = load_network(archive)
    assert metadata["format"] == "torchscript"

    inputs = random_graph_inputs(make_context())
    with torch.no_grad():
        assert torch.equal(net(*inputs), loaded(*inputs))

    predictor = create_predictor(checkpoint=archive, device="cpu", options={})
    assert predictor.predict(make_context()).damage.shape == (NODES.shape[0],)


def test_exporting_an_archive_again_is_rejected(tmp_path):
    archive = export_torchscript(
        save_state_dict(tmp_path / "ck.pt", make_net().eval()),
        tmp_path / "scripted.pt",
        n_nodes=16,
    )
    with pytest.raises(ValueError, match="already a TorchScript archive"):
        export_torchscript(archive, tmp_path / "again.pt")


def test_factory_passes_options_through(tmp_path):
    archive = export_torchscript(
        save_state_dict(tmp_path / "ck.pt", make_net().eval()),
        tmp_path / "scripted.pt",
        n_nodes=16,
    )
    predictor = create_predictor(
        checkpoint=archive,
        device="cpu",
        options={
            "representation": "damage_increment",
            "boundary_indicator": "topological",
        },
    )
    assert predictor.representation == "damage_increment"
    assert predictor.boundary_indicator == "topological"
