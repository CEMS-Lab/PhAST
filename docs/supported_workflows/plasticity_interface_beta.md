# Plasticity and Interface Beta Workflows

Plasticity, cohesive, PF-CZM, and interface examples are beta validation
workflows unless the capability matrix says otherwise. They remain useful and
tested, but they are not presented as a fully coupled production product.

## Public beta path

The beta family is driven through the curated reproducibility manifest:

```bash
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id j2_validation --output_dir examples/plasticity_interface_beta/results/j2_validation
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id structural_dcb_cohesive --output_dir examples/plasticity_interface_beta/results/structural_dcb_cohesive
python -m phast run configs/benchmarks/plasticity_interface/reproducibility_contracts.yaml --validation-id pfczm_uniaxial_strength --output_dir examples/plasticity_interface_beta/results/pfczm_uniaxial_strength
```

Those validation IDs are mirrored in `examples/PUBLIC_EXAMPLES_CONTRACT.yaml`
and documented in the retained result folders.

The curated validation result bundles live under
`examples/plasticity_interface_beta/results/` and are indexed by
`examples/plasticity_interface_beta/results/issue_708_promoted_results.yaml`.
They include YAML-resolved configs, provenance JSON, CSV histories, setup and
final PNGs, GIF animations, and visual manifests. Rerunning a validation case
can regenerate local trajectory stores for field-level inspection with
`phast.load_result(path)`. The matching fluent setup surfaces live in
`examples/plasticity_interface_beta/fluent_setups/` and are for authoring/inspection;
the reproducibility path remains the curated YAML dispatcher above.

## Boundary

Do not generalize a passing beta workflow to arbitrary coupled
plasticity/cohesive fracture. Public docs should state the exact validation ID,
script, output artifacts, and claim boundary for each beta example.
