"""Process-local loader for the glass source snapshot."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys
from typing import Any


def main() -> None:
    bundle = Path(__file__).resolve().parent
    historical_src = bundle / "historical_src"
    if not sys.flags.isolated:
        raise RuntimeError("Use run.py, which creates an isolated Python process")
    if any(name == "phast" or name.startswith("phast.") for name in sys.modules):
        raise RuntimeError("PhAST was already imported before historical isolation")
    sys.path.insert(0, str(historical_src))
    driver = bundle / "original_driver.py"
    if sys.argv[1:] == ["--import-check"]:
        original = runpy.run_path(str(driver), run_name="historical_glass_import_check")
        invocation = json.loads((bundle / "invocation.json").read_text())
        sys.argv = [str(driver), *invocation["arguments"], "--output_dir", "unused-import-check"]
        args = original["parse_args"]()
        config = original["SolverConfig"](
            solver_type="explicit", differentiable=True, preconditioner="jacobi",
            H_update_method=args.H_update_method, H_update_scale=args.H_update_scale,
            softmax_H_beta=args.softmax_H_beta,
        )
        material = original["create_material"](
            "glass_borden", Gc=original["GC_TRUE"], l0=original["L0_TRUE"],
        )
        imported: dict[str, str] = {}
        for name, module in sorted(sys.modules.items()):
            if name == "phast" or name.startswith("phast."):
                path = Path(module.__file__).resolve()
                if historical_src not in path.parents:
                    raise RuntimeError(f"Non-historical PhAST import: {name}")
                imported[name] = path.relative_to(bundle).as_posix()
        versions: dict[str, Any] = {"python": sys.version.split()[0]}
        for package in ("torch", "numpy", "scipy", "meshio", "gmsh", "matplotlib"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        print(json.dumps({
            "check": "historical-import-only", "runtime_recomputed": False,
            "driver_arguments": vars(args), "solver_config": dataclasses.asdict(config),
            "material": dataclasses.asdict(material),
            "driver_constants": {key: original[key] for key in ("GC_TRUE", "L0_TRUE", "SIGMA_LOAD")},
            "imported_modules": imported, "validation_environment": versions,
        }, indent=2))
        return
    sys.argv[0] = str(driver)
    runpy.run_path(str(driver), run_name="__main__")


if __name__ == "__main__":
    main()
