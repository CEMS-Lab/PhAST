"""
Entry point for `python -m phast`.

Subcommands:
    run         Run a simulation from a YAML config
    precheck    Pre-simulation diagnostics (wave speeds, CFL, mesh quality)
    explain-config
                Explain a YAML config without generating meshes or running
    schema      Export the JSON Schema for YAML configs
    doctor      Report environment and solver backend status
    postprocess Generate plots from a completed run directory
    new         Scaffold a starter YAML config for a new benchmark

Usage:
    python -m phast run configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml
    python -m phast precheck configs/benchmarks/dynamic/B2_kalthoff_winkler.yaml
    python -m phast explain-config configs/benchmarks/dynamic/B3_dynamic_sent.yaml
    python -m phast schema --output configs/phast.schema.json
    python -m phast doctor
    python -m phast postprocess path/to/run_dir --dpi 300
    python -m phast new my_benchmark --type quasi_static --material pmma_bleyer
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__.strip())
        print("\nAvailable subcommands: run, precheck, explain-config, schema, doctor, postprocess, new")
        sys.exit(0)

    cmd = sys.argv[1]
    # Remove the subcommand from argv so argparse in submodules works
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if cmd == 'run':
        from .config.run_config import main as run_main
        run_main()
    elif cmd == 'precheck':
        from .config.precheck import main as pc_main
        pc_main()
    elif cmd in ('explain-config', 'explain'):
        from .config.explain_config import main as explain_main
        raise SystemExit(explain_main())
    elif cmd in ('schema', 'json-schema'):
        from .config.config_schema import main as schema_main
        raise SystemExit(schema_main())
    elif cmd == 'doctor':
        from .utils.doctor import main as doctor_main
        raise SystemExit(doctor_main())
    elif cmd == 'postprocess':
        from .postprocess_paper import main as pp_main
        pp_main()
    elif cmd == 'new':
        from .scaffolder import main as new_main
        new_main()
    else:
        print(f"Unknown subcommand: '{cmd}'")
        print("Available: run, precheck, explain-config, schema, doctor, postprocess, new")
        sys.exit(1)


if __name__ == '__main__':
    main()
