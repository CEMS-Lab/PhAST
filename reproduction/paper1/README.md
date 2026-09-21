# Paper 1 Reproduction

This directory indexes the PhAST manuscript and supplement. It provides a
bounded retained-data route, not a claim that every paper simulation can
currently be rerun.

The **data record is in preparation**. No data DOI or download URL is assigned
here. Large fields, historical execution sources and environments remain
external to this repository.

## Commands

Run from the PhAST repository root. `list` needs no external data:

```bash
python reproduction/paper1/reproduce.py list
python reproduction/paper1/reproduce.py check --data DIR
python reproduction/paper1/reproduce.py replot q2 --data DIR --output OUTPUT_DIR
python reproduction/paper1/reproduce.py replot inverse --data DIR --output OUTPUT_DIR
python reproduction/paper1/reproduce.py replot forward --data DIR --output OUTPUT_DIR
```

`list` and `check` use only the Python standard library. Before replotting,
create a separate plotting environment from the repository root:

```bash
python3.10 -m venv .venv-paper1-replot
source .venv-paper1-replot/bin/activate
python -m pip install -r reproduction/paper1/requirements-replot.txt
```

The retained-data checks were tested on macOS arm64 with Python 3.10.18 and
the versions in `requirements-replot.txt`. This is a replot-only environment,
not a reconstruction of historical simulation, CUDA or timing environments.
Other platforms have not been verified for this route.

Replace `DIR` with the local extracted `phast-paper-reproduction` data root,
containing `manifest.json`, `document_identity.json`, `forward/`, `inverse/`,
`verification/`, `sources/` and `shared/`. Use a different fresh `OUTPUT_DIR`
outside the data tree for each replot. Data are supplied explicitly; these
commands do not download a record or launch full simulations. Replotting
executes selected archived Python code, so use only a trusted extraction.
Hash matching binds the input identity; it does not sandbox that code.

`check --data DIR` verifies the manifest and document identity against the
catalogue's pinned SHA-256 hashes, then checks the listed payload sizes and
hashes. Another self-consistent dataset is not a substitute for this audited
edition. This checks selected-file integrity, not completeness of catalogue
coverage, missing historical inputs, or full simulation reproduction.
Replots write generated figures and verification records under the requested
output directory without replacing the input data.

## Current Status

| Route | Recorded retained-data checks | Full simulation |
| --- | --- | --- |
| `q2` | Force curve and three damage panels regenerated; 200 saved fields checked | Pending |
| `inverse` | Six figures regenerated from 487 exact inputs, covering scalar, six-start, cross-mesh, joint, memory and finite-difference studies | Pending; original inputs or environments remain incomplete for some studies |
| `forward` | Standing-wave and subcycling figures regenerated; observed orders, selected terminal metrics and timing summaries checked | Pending; this command does not cover all forward studies |
| Other studies | Inputs mapped, with individual restrictions in the catalogue | Pending or not applicable |

These are preparation records, not certification of every metric or a new
solver execution. Original perforated-plate native fields, five-mesh impact
inputs, complete AT1 energy-integration states, historical execution identities
and several study environments remain unresolved. Photograph reuse permission
and reference-curve provenance are separate restrictions.

The new terminal AT2 profile statement reports local/gradient contributions
of **1.53/0.26**, versus **0.5/0.5** for the static profile. Its exact source
SHA-256 is recorded under `prose_at2_terminal_profile`. The calculation script,
mesh/state choice, integration region, quadrature and normalisation are pending;
no independent analysis is claimed.

## Catalogue

[`catalogue.json`](catalogue.json) uses schema version 1, independently of the
existing [public example contract](../../docs/user_guide/example_contract.md):

- `data_identity` binds checks and replots to the tested frozen data extraction's
  manifest and document-identity file hashes.
- `documents` identifies the indexed manuscript and supplement by stable labels
  and PDF SHA-256 hashes, not local paths.
- `items` contains 38 figures/tables in document order: 14 figures and 8 tables
  in the manuscript, followed by 14 figures and 2 tables in the supplement.
  Each entry maps labels to studies and selected archive-relative assets.
- `results` contains 36 grouped text mappings plus the pending AT2 profile
  statement. A mapping is not independent numerical recomputation. Source edits
  after the indexed edition require a final scope refresh before publication.
- `studies` separates `replot_status`, `rerun_status`, `required_inputs` and
  `unresolved_gaps`. `replot_group` is `q2`, `inverse`, `forward` or `null`;
  `null` means no route in this CLI. Archive paths are relative to `--data`.

`retained_data_checked` applies only to the bounded checks described above.
`pending_full_run` means a fresh complete simulation has not been verified;
`pending_inputs` additionally identifies missing or unresolved dependencies.
Rights/provenance holds are not numerical failures. Empty `archive_assets` means
a table is represented by study records or a composite is explicitly excluded;
it does not imply completion. Asset paths preserve exact historical filenames.

`related_example` names an existing public example, **not an identical paper
run**. In particular, paper B1 relates to `B3_dynamic_sent`, paper B3 to
`B7_dynamic_crack_branching_comsol`, and paper B4a-d to the `B6_perforated_*`
family. Public example names, schemas and inputs are unchanged. The paper's
inverse replot route does not promote the documentation-only public inverse
example into a runnable inverse solver.
