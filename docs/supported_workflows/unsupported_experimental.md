# Unsupported and Experimental Workflows

This page is the public boundary for features that may exist in the private
development repository, older branches, or research notes but are not public
PhAST workflows.

## Not public workflow claims

- arbitrary weak-form YAML compilation;
- general-purpose coupled elastoplastic cohesive phase-field fracture;
- inverse identification workflows beyond the released paper-specific scope;
- hybrid/deep-learning solver switching workflows;
- Paper-2 or Paper-4 research lanes;
- raw trajectory archives or proprietary COMSOL model binaries.

## How to read experimental material

If a feature is marked scaffold, beta, optional-backend, or unsupported in the
[capability matrix](../user_guide/capability_matrix.md), treat docs and
examples as implementation notes until the feature is promoted through:

1. a public YAML deck or script-contract manifest;
2. required outputs from `docs/user_guide/example_contract.md`;
3. tests that validate the contract;
4. a clear capability-matrix status update.
