# Command-line reference

The module entry point is `python -m phast`. Run it without a subcommand or with `--help` for the command summary.

## Commands

| Command | Purpose |
|---|---|
| `run CONFIG` | Validate, resolve, and run a supported YAML configuration. |
| `precheck --config CONFIG` | Report pre-simulation diagnostics without running the solve. |
| `explain-config CONFIG` | Explain configuration contents without generating a mesh or running. |
| `schema` | Export the JSON Schema for YAML configurations. |
| `doctor` | Report environment and solver-backend status. |
| `postprocess RUN_DIR` | Generate supported products from a completed run. |
| `new NAME` | Scaffold a starter YAML configuration. |
| `result validate FILE` | Check a `result.vtkhdf` file against its layout and report ERROR, WARN and INFO findings. |
| `result recover FILE` | Write a trimmed, validated copy of an interrupted result file. |
| `result finalize --archive FILE` | Rewrite a closed result file with gzip, shuffle and fletcher32 and check it. |
| `figures RUN` | Draw the standard figures of a result file. |

`explain` is accepted as an alias for `explain-config`; `json-schema` is an alias for `schema`.

## `run` options

`run CONFIG` accepts `--device`, `--output_dir`, `--validate-only`, `--trajectory`, `--trajectory-format {zarr,h5,both}`, `--h5`, `--plots`, `--gif`, `--fast`, `--profile`, `--num_steps`, `--print_every`, `--h5_every`, `--validation-id`, `--time_integrator`, and `--rho_inf`. CLI values override the corresponding YAML values. `--validate-only` performs configuration or workflow validation and exits before execution; it is not runtime validation.

## Other command options

`precheck` accepts a YAML path either as `CONFIG` or as `--config CONFIG`.
Do not supply both forms together. It also accepts material and mesh
parameters such as `--preset`, `--h_min`, and `--t_total`. Run
`python -m phast precheck --help` for all options.

`result validate` accepts `--json` and `--running-ok`, `result recover` accepts `-o` and `--force`, and `figures` accepts `-o`, `--steps`, `--dpi`, `--fields`, `--deform-scale` and `--kinds`. [Result files](../user_guide/result_file.md) describes them.

`schema` accepts `--output`, `--check`, and `--indent`. `postprocess` accepts a run directory and post-processing options such as `--dpi`. `new` accepts a name and starter options including `--type` and `--material`; consult the command help for the complete local option set.

## Examples

```bash
python -m phast precheck examples/dynamic/B2_kalthoff_winkler/config.yaml
python -m phast explain-config examples/dynamic/B2_kalthoff_winkler/config.yaml
python -m phast precheck --config examples/dynamic/B2_kalthoff_winkler/config.yaml
python -m phast run config.yaml --validate-only
python -m phast run config.yaml --device cuda --trajectory --trajectory-format h5
python -m phast schema --output configs/phast.schema.json
python -m phast doctor
python -m phast postprocess runs/example --dpi 300
python -m phast result validate runs/example/result.vtkhdf
python -m phast figures runs/example
```
