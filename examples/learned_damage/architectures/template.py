"""Template for a new learned-damage architecture.

Copy this file, rename it after the architecture, and fill in the two marked
places. Nothing outside this file changes: the geometry, mesh, named regions,
boundary conditions, loading, material, solver and time integration are the same
as for a classical run, and the simulation selects the architecture through one
YAML entry.

.. code-block:: yaml

    solver:
      damage_update: learned_proposal
      damage_predictor: my_package.my_architecture:create_predictor
      damage_checkpoint: checkpoints/my_model.pt
      damage_predictor_options:
        representation: damage
      damage_fallback: true

What PhAST guarantees
---------------------
A predictor is called once per damage subproblem and receives a
:class:`phast.learned_damage.DamageStepContext` describing the current
finite-element state. PhAST owns the admissibility audit, the projected
residual check, the box and irreversibility constraints, route reporting and
the classical fallback. A predictor proposes; it never decides.

What the context offers
-----------------------
Raw state
    ``nodes``, ``elements``, ``displacement``, ``velocity``,
    ``history_element``, ``history_nodal``, ``damage_previous``, ``material``,
    ``step``, ``time``, ``load_factor``, ``device``, ``dtype``.

Finite-element helpers, architecture-neutral
    ``canonical_node_features()`` returns ``[x, y, H, d_prev, u_x, u_y]``.
    ``graph_edge_index()`` returns directed, duplicate-free mesh edges.
    ``element_areas()`` returns the area of every triangle.
    ``assemble_element_to_nodes(values)`` returns
    ``sum_e int N_i value_e dOmega``.
    ``boundary_node_mask()`` flags topological boundary nodes, interior voids
    included.
    ``edge_lengths(edge_index)`` returns the length of every edge.

Anything beyond these, such as a radius graph, a regular-grid projection, a
trunk basis for an operator network, a temporal state buffer, or a
normalization recovered from a training manifest, is built here. That is the
whole purpose of a per-architecture wrapper: the solver stays neutral and each
model brings its own preprocessing.

Obligations
-----------
1. Return one value per mesh node, shaped like ``context.damage_previous``.
2. Return it on ``context.device`` in ``context.dtype``.
3. Declare ``representation``: ``'damage'`` for the next field, or
   ``'damage_increment'`` for an increment on the previous accepted damage.
4. Reproduce the training-time feature construction exactly. A mismatch in
   normalization or feature order still yields a plausible crack pattern while
   no longer representing the trained operator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from phast.learned_damage import DamagePrediction, DamageStepContext

__all__ = ["TemplatePredictor", "create_predictor"]


class TemplatePredictor:
    """Minimal predictor skeleton. Replace the two marked bodies."""

    #: Reported in the route summary and in prediction diagnostics.
    name = "template"

    def __init__(
        self,
        model: Any,
        *,
        representation: str = "damage",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.representation = str(representation)
        self.metadata = dict(metadata or {})

    def build_inputs(self, context: DamageStepContext) -> tuple[Any, ...]:
        """Build whatever this architecture consumes.

        Replace this body. The example below is the simplest useful case: the
        canonical nodal features and the mesh graph. An operator network might
        return coordinates and a history field instead; a radius-graph model
        would build its own neighbourhood here.
        """
        features = context.canonical_node_features()
        edge_index = context.graph_edge_index()
        return features, edge_index

    def predict(self, context: DamageStepContext) -> DamagePrediction:
        """Call the model and return one nodal damage field.

        Replace the model call. Keep the surrounding contract: no gradient, one
        value per node, the context's device and dtype.
        """
        inputs = self.build_inputs(context)
        with torch.no_grad():
            raw = self.model(*inputs)  # replace with this architecture's call

        damage = torch.as_tensor(
            raw, device=context.device, dtype=context.dtype,
        ).reshape_as(context.damage_previous)
        return DamagePrediction(
            damage=damage,
            representation=self.representation,
            diagnostics={"adapter": self.name, **self.metadata},
        )


def create_predictor(
    *,
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
    options: Mapping[str, Any] | None = None,
) -> TemplatePredictor:
    """Factory matching the ``module:factory`` entry in a YAML configuration.

    PhAST calls this with the checkpoint path, the resolved device and the
    ``damage_predictor_options`` mapping. Loading, architecture reconstruction
    and checkpoint validation belong here.
    """
    options = dict(options or {})
    model = torch.jit.load(str(checkpoint), map_location=device)
    model.eval()
    return TemplatePredictor(
        model,
        representation=options.get("representation", "damage"),
        metadata={"path": str(Path(checkpoint).resolve())},
    )
