---
orphan: true
---

# Example curation

This page is for maintainers selecting and updating public examples. It does
not define solver capability or replace the public [example contract](../user_guide/example_contract.md).

## Curation decisions

1. Classify the item as a runnable solve, validate-only configuration, template or manifest, or retained evidence.
2. Keep the example-local README authoritative for inputs, command, output directory, required artifacts, and limitations.
3. Retain only public-safe, lightweight evidence: manifests, CSV summaries, setup and final-state figures, comparisons, and compact animations.
4. Keep raw trajectories, scratch directories, private infrastructure details, and unpublished data outside the public example.
5. Use the labels in the [capability matrix](../user_guide/capability_matrix.md) and avoid implying convergence, benchmark reproduction, or physical validity from configuration preflight alone.

## Review boundary

The linear-plate case is the compact completed solve used for onboarding. The
Miehe case should remain bounded by its documented runtime and retained
comparison evidence. A full reproduction claim requires evidence beyond a
successful `--validate-only` command.

## Public navigation

New examples should be added to the [gallery](../example-gallery.md), while
numerical-method explanations belong in the [user manual](../user_guide/overview.md)
or [reference](../reference/index.md). Do not add maintainer, agent, or release
process material to beginner onboarding pages.
