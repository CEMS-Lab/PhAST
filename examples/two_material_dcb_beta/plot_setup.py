"""Draw the DCB input configuration without running a fracture calculation."""
from __future__ import annotations

import argparse
from pathlib import Path

from phast.workflow import problem_spec_from_yaml
from phast.workflow.multimaterial_fracture import (
    _assign_materials,
    _build_mesh,
    _resolve_bcs,
    _resolve_regions,
)
from phast.workflow.setup_plot import save_material_setup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dcb_setup"))
    args = parser.parse_args()
    spec = problem_spec_from_yaml(args.config)
    # Reuse the solver's mesh and selections, rather than a second model setup.
    nodes, elements = _build_mesh(spec)
    node_regions, element_regions = _resolve_regions(spec, nodes, elements)
    _, _, material_ids, material_names = _assign_materials(spec, element_regions, len(elements))
    path = save_material_setup(
        spec, nodes, elements, material_ids, material_names, node_regions,
        _resolve_bcs(spec, node_regions),
        args.output_dir / "material_regions_and_loading.png",
    )
    print(f"Input diagram saved to {path}; no fracture simulation was run.")


if __name__ == "__main__":
    main()
