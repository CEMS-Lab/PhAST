"""Mesh-graph network adapter for the learned damage subproblem.

One architecture, wrapped so that PhAST can call it. The finite-element
workflow is untouched: geometry, mesh, named regions, boundary conditions,
loading, material and time integration all remain exactly as they are for a
classical run. The only change is the damage subproblem's ``damage_predictor``
entry.

The generic finite-element quantities this adapter needs are supplied by
:class:`phast.learned_damage.DamageStepContext`. What is specific to this
architecture, and therefore lives here, is the feature contract:

``node_features[:, 0]``
    ``sum_e int N_i H_e / (Gc * l0) dOmega``, the assembled and normalized
    history integral. The network applies ``log10(1 + x)`` to this column
    internally, so the raw assembled value is passed.
``node_features[:, 1]``
    A boundary indicator. See :func:`bounding_box_indicator` for the variant
    used when the reference checkpoints were trained.
``edge_features[:, 0]``
    Edge length divided by ``l0``.

A checkpoint evaluated under a different normalization, assembly or edge
feature still produces a plausible crack pattern while no longer representing
the trained operator. Reproducing the training-time construction exactly is the
adapter's responsibility, not the solver's.

Usage
-----
.. code-block:: yaml

    solver:
      damage_update: learned_proposal
      damage_predictor: examples.learned_damage.architectures.mesh_graph_net:create_predictor
      damage_checkpoint: checkpoints/mesh_graph_net.pt
      damage_predictor_options:
        representation: damage
        boundary_indicator: bounding_box
      damage_fallback: true
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from phast.learned_damage import DamagePrediction, DamageStepContext

__all__ = [
    "MeshGraphDamageNet",
    "MeshGraphNetPredictor",
    "bounding_box_indicator",
    "create_predictor",
    "export_torchscript",
    "load_network",
]


# ---------------------------------------------------------------- network

def _mlp(in_features: int, out_features: int, hidden: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    width = in_features
    for _ in range(layers):
        modules.append(nn.Linear(width, hidden))
        modules.append(nn.ReLU())
        width = hidden
    modules.append(nn.Linear(width, out_features))
    return nn.Sequential(*modules)


class SymmetricProcessorBlock(nn.Module):
    """One permutation-symmetric message-passing block."""

    def __init__(self, latent_dim: int, hidden_layers: int) -> None:
        super().__init__()
        triple = 3 * latent_dim
        self.edge_update = _mlp(triple, latent_dim, latent_dim, hidden_layers)
        self.message_norm = nn.LayerNorm(triple)
        self.message = _mlp(triple, latent_dim, latent_dim, hidden_layers)
        self.node_norm = nn.LayerNorm(2 * latent_dim)
        self.node_update = _mlp(2 * latent_dim, latent_dim, latent_dim, hidden_layers)

    def forward(
        self,
        node_latent: torch.Tensor,
        edge_latent: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Indexed rather than unpacked so that the block stays scriptable.
        src, dst = edge_index[0], edge_index[1]
        symmetric = torch.cat(
            [
                node_latent[src] + node_latent[dst],
                torch.abs(node_latent[src] - node_latent[dst]),
                edge_latent,
            ],
            dim=-1,
        )
        edge_latent = edge_latent + self.edge_update(symmetric)
        symmetric = torch.cat(
            [
                node_latent[src] + node_latent[dst],
                torch.abs(node_latent[src] - node_latent[dst]),
                edge_latent,
            ],
            dim=-1,
        )
        messages = self.message(self.message_norm(symmetric))
        aggregate = torch.zeros_like(node_latent)
        aggregate.index_add_(0, dst, messages)
        degree = torch.zeros(
            (node_latent.shape[0], 1),
            dtype=node_latent.dtype,
            device=node_latent.device,
        )
        degree.index_add_(
            0,
            dst,
            torch.ones((dst.numel(), 1), dtype=node_latent.dtype, device=dst.device),
        )
        aggregate = aggregate / degree.clamp_min(1.0)
        node_input = self.node_norm(torch.cat([node_latent, aggregate], dim=-1))
        node_latent = node_latent + self.node_update(node_input)
        return node_latent, edge_latent


class MeshGraphDamageNet(nn.Module):
    """Encoder-processor-decoder mesh-graph surrogate for the damage field.

    The decoder is linear. Box constraints and irreversibility are applied by
    the PhAST runtime coupling rather than by an output activation here.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        hidden_layers: int = 3,
        message_passing_steps: int = 10,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            _mlp(2, latent_dim, latent_dim, hidden_layers),
            nn.LayerNorm(latent_dim),
        )
        self.edge_encoder = nn.Sequential(
            _mlp(1, latent_dim, latent_dim, hidden_layers),
            nn.LayerNorm(latent_dim),
        )
        self.processor = nn.ModuleList(
            SymmetricProcessorBlock(latent_dim, hidden_layers)
            for _ in range(message_passing_steps)
        )
        self.decoder = _mlp(latent_dim, 1, latent_dim, hidden_layers)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        transformed = node_features.clone()
        transformed[:, 0] = torch.log10(1.0 + transformed[:, 0].clamp_min(0.0))
        node_latent = self.node_encoder(transformed)
        edge_latent = self.edge_encoder(edge_features)
        for block in self.processor:
            node_latent, edge_latent = block(node_latent, edge_latent, edge_index)
        return self.decoder(node_latent).squeeze(-1)


# ---------------------------------------------------------------- checkpoints

_ARCHITECTURE_KEYS = ("latent_dim", "hidden_layers", "message_passing_steps")


def load_network(
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a TorchScript archive or a state-dict checkpoint.

    A TorchScript archive carries its own architecture, so evaluating it needs
    no model source. A state-dict checkpoint must supply the three architecture
    entries in a ``configuration`` mapping.
    """
    path = Path(checkpoint)
    try:
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        return model, {"format": "torchscript", "path": str(path.resolve())}
    except (RuntimeError, ValueError):
        pass

    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError(
            f"{path} is neither a TorchScript archive nor a checkpoint holding "
            "'model_state_dict'")
    configuration = dict(payload.get("configuration", {}))
    missing = [key for key in _ARCHITECTURE_KEYS if key not in configuration]
    if missing:
        raise ValueError(
            f"{path} is missing architecture entries {missing} in its "
            "'configuration' mapping")
    model = MeshGraphDamageNet(
        latent_dim=int(configuration["latent_dim"]),
        hidden_layers=int(configuration["hidden_layers"]),
        message_passing_steps=int(configuration["message_passing_steps"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    metadata = {
        "format": "state_dict",
        "path": str(path.resolve()),
        **{key: configuration[key] for key in _ARCHITECTURE_KEYS},
    }
    for key in ("variant", "Gc", "length_scale", "seed"):
        if key in configuration:
            metadata[key] = configuration[key]
    if "completed_epochs" in payload:
        metadata["completed_epochs"] = int(payload["completed_epochs"])
    return model, metadata


def export_torchscript(
    checkpoint: str | Path,
    output: str | Path,
    *,
    n_nodes: int = 64,
    verify: bool = True,
) -> Path:
    """Convert a state-dict checkpoint into a portable TorchScript archive.

    The archive evaluates without this module on the import path, so trained
    weights can be distributed without model source. Optimizer state is
    dropped, which makes the archive substantially smaller than a training
    checkpoint.
    """
    model, metadata = load_network(checkpoint, device="cpu")
    if metadata["format"] == "torchscript":
        raise ValueError(f"{checkpoint} is already a TorchScript archive")

    scripted = torch.jit.script(model)
    if verify:
        generator = torch.Generator().manual_seed(0)
        node_features = torch.rand(n_nodes, 2, generator=generator)
        n_edges = 4 * n_nodes
        edge_index = torch.randint(0, n_nodes, (2, n_edges), generator=generator)
        edge_features = torch.rand(n_edges, 1, generator=generator)
        with torch.no_grad():
            reference = model(node_features, edge_index, edge_features)
            candidate = scripted(node_features, edge_index, edge_features)
        assert torch.equal(reference, candidate), (
            "scripted model disagrees with the eager model")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(output))
    return output


# ---------------------------------------------------------------- features

def bounding_box_indicator(
    nodes: torch.Tensor, *, atol: float = 1.0e-10
) -> torch.Tensor:
    """Flag nodes on the bounding box of the mesh.

    This marks the outer extent only, so interior voids such as holes and slots
    carry no flag. It is retained because the reference checkpoints were trained
    against it. For the topological boundary, which does detect interior voids,
    use :meth:`phast.learned_damage.DamageStepContext.boundary_node_mask`,
    selected here by ``boundary_indicator: topological``.
    """
    x, y = nodes[:, 0], nodes[:, 1]
    on_edge = (
        torch.isclose(x, x.min(), atol=atol)
        | torch.isclose(x, x.max(), atol=atol, rtol=0.0)
        | torch.isclose(y, y.min(), atol=atol, rtol=0.0)
        | torch.isclose(y, y.max(), atol=atol, rtol=0.0)
    )
    return on_edge.to(dtype=nodes.dtype)


# ---------------------------------------------------------------- adapter

class MeshGraphNetPredictor:
    """Wrap :class:`MeshGraphDamageNet` for the PhAST damage-predictor protocol.

    Parameters
    ----------
    model
        A network taking ``(node_features, edge_index, edge_features)``.
    representation
        ``'damage'`` when the network predicts the next field, or
        ``'damage_increment'`` when it predicts an increment relative to the
        previous accepted damage.
    boundary_indicator
        ``'bounding_box'`` to reproduce the training-time construction, or
        ``'topological'`` to mark interior voids as well.
    length_scale, Gc
        Override the values taken from the running material. Supply these only
        when the checkpoint was trained under a different normalization, and
        record the choice in any reported result.
    metadata
        Provenance recovered from the checkpoint, reported in diagnostics.
    """

    name = "mesh-graph-net"

    def __init__(
        self,
        model: nn.Module,
        *,
        representation: str = "damage",
        boundary_indicator: str = "bounding_box",
        length_scale: float | None = None,
        Gc: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if boundary_indicator not in {"bounding_box", "topological"}:
            raise ValueError(
                "boundary_indicator must be 'bounding_box' or 'topological'; "
                f"got {boundary_indicator!r}")
        self.model = model.eval()
        self.representation = str(representation)
        self.boundary_indicator = boundary_indicator
        self.length_scale = length_scale
        self.Gc = Gc
        self.metadata = dict(metadata or {})
        self._graph_cache: tuple[
            tuple[int, int, int, int, float], torch.Tensor, torch.Tensor
        ] | None = None

    @staticmethod
    def _resolve(name: str, override: float | None, context: DamageStepContext) -> float:
        if override is not None:
            return float(override)
        value = context.material.get(name)
        if value is None:
            raise ValueError(
                f"the running material exposes no {name!r}; pass it explicitly "
                "to MeshGraphNetPredictor")
        return float(value)

    def _graph(self, context: DamageStepContext) -> tuple[torch.Tensor, torch.Tensor]:
        """Edges and edge features, rebuilt only when the mesh changes."""
        length_scale = self._resolve("l0", self.length_scale, context)
        key = (
            int(context.nodes.data_ptr()),
            int(context.elements.data_ptr()),
            int(context.nodes.shape[0]),
            int(context.elements.shape[0]),
            float(length_scale),
        )
        if self._graph_cache is not None and self._graph_cache[0] == key:
            return self._graph_cache[1], self._graph_cache[2]
        edge_index = context.graph_edge_index()
        edge_features = (
            context.edge_lengths(edge_index) / length_scale).reshape(-1, 1)
        self._graph_cache = (key, edge_index, edge_features)
        return edge_index, edge_features

    def node_features(self, context: DamageStepContext) -> torch.Tensor:
        """Build the two nodal features this network was trained on."""
        Gc = self._resolve("Gc", self.Gc, context)
        length_scale = self._resolve("l0", self.length_scale, context)
        assembled = context.assemble_element_to_nodes(
            context.history_element.reshape(-1) / (Gc * length_scale))
        if self.boundary_indicator == "bounding_box":
            indicator = bounding_box_indicator(context.nodes)
        else:
            indicator = context.boundary_node_mask().to(dtype=assembled.dtype)
        return torch.stack((assembled, indicator.to(assembled.dtype)), dim=1)

    def predict(self, context: DamageStepContext) -> DamagePrediction:
        edge_index, edge_features = self._graph(context)
        features = self.node_features(context)
        model_dtype = next(
            (parameter.dtype for parameter in self.model.parameters()),
            torch.float32,
        )
        with torch.no_grad():
            raw = self.model(
                features.to(device=context.device, dtype=model_dtype),
                edge_index.to(device=context.device),
                edge_features.to(device=context.device, dtype=model_dtype),
            )
        damage = torch.as_tensor(
            raw, device=context.device, dtype=context.dtype,
        ).reshape_as(context.damage_previous)
        return DamagePrediction(
            damage=damage,
            representation=self.representation,
            diagnostics={
                "adapter": self.name,
                "n_edges": int(edge_index.shape[1]),
                "boundary_indicator": self.boundary_indicator,
                **self.metadata,
            },
        )


def create_predictor(
    *,
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
    options: Mapping[str, Any] | None = None,
) -> MeshGraphNetPredictor:
    """Factory matching the ``module:factory`` entry in a YAML configuration."""
    options = dict(options or {})
    model, metadata = load_network(checkpoint, device=device)
    return MeshGraphNetPredictor(
        model,
        representation=options.get("representation", "damage"),
        boundary_indicator=options.get("boundary_indicator", "bounding_box"),
        length_scale=options.get("length_scale"),
        Gc=options.get("Gc"),
        metadata=metadata,
    )
