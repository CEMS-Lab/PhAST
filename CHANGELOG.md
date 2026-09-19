# Changelog

## Unreleased

### Python workflow execution

- Run the supported linear-plate Python `ProblemSpec` through the existing
  YAML runner and example files.
- Validate the supported declarations before execution and reject inputs
  that the example runner cannot represent.
- Preserve geometry units as metadata in `Problem.geometry`. The linear-plate
  companion declares metres consistently with its SI inputs and results.
- Add Python/YAML parity tests for resolved inputs, response values and field
  images, together with validation-only and unsupported-input regressions.
- Preserve default and relative Python result directories after execution.
- Validate requests against the fixed visual bundle and require `plots=True`.

The geometry keyword records units and leaves numerical values unchanged.
The source-less execution bridge covers the documented single-material,
single-step linear plate with its built-in boundary conditions.
