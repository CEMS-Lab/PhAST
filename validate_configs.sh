#!/bin/bash
for f in configs/REFERENCE.yaml configs/benchmarks/dynamic/*.yaml configs/benchmarks/quasistatic/*.yaml; do
  echo "Validating $f..."
  python -m phast run "$f" --validate-only
  if [ $? -ne 0 ]; then
    echo "FAILED: $f"
  else
    echo "PASSED: $f"
  fi
done
