# PhAST Public Documentation Style and Conventions

This guide applies to all public-facing prose in the PhAST repository. Check
it whenever creating or reviewing:

- the root `README.md`;
- a README under `examples/`, `configs/`, `scripts/`, or another subdirectory;
- pages under `docs/`;
- tutorial and notebook narrative cells;
- public configuration comments;
- public release notes and changelog entries.

The objective is clear, accurate, academic documentation that an undergraduate
student can follow without prior knowledge of PhAST. Documentation must also
state the implemented scope honestly so that readers can decide whether the
software is suitable for their work.

This file governs public software documentation. Journal manuscripts,
responses to reviewers, private research notes, and generated API text may
have additional requirements.

New YAML examples, example READMEs, and tutorials must also follow
[`CONFIGURATION_STYLE.md`](CONFIGURATION_STYLE.md). It defines the common
schema-2 section order, student-facing explanations, command sequence, and
checks required before promoting an execution route. Preserve numerical
inputs during writing-only edits; do not migrate frozen course assets or
separate 3D research simply to make inputs uniform.

Sections 19 and 20 provide transferable writing directives for scientific
explanations and, when requested, research proposals. Apply the structure
appropriate to the document: a proposal describes work to be tested, whereas
public software documentation describes verified capabilities. Writing
examples below do not establish implemented features or numerical results.

## 1. Required working practice

Before writing:

1. Identify the reader and the task the page supports.
2. Check the current implementation, command help, schema, and runnable
   examples before describing a capability.
3. Decide whether the page is a tutorial, explanation, reference, example
   README, or installation guide.
4. Use the relevant page structure in Section 4.

Before merging:

1. Read the page once from beginning to end as a new user would.
2. Run every new command and runnable configuration when the change requires
   execution verification.
3. Confirm that paths, command-line options, configuration keys, units, and
   output filenames match the repository.
4. Check all capability statements against the public capability matrix.
5. Remove internal project-management language and unsupported claims.
6. Confirm that links, equations, tables, images, and code blocks render.
7. Keep limitations and experimental status visible.

Passing a documentation build confirms that the page renders. It does not
confirm that a scientific claim or numerical result is correct.

## 2. Voice and audience

Write as a researcher explaining a numerical method to another researcher.
Use direct, neutral language. Be precise without assuming that the reader
already knows the repository.

- Put one main idea in each sentence.
- Prefer sentences shorter than 30 words. Split sentences longer than 40.
- Introduce a term before using its abbreviation.
- State what the software does before discussing implementation details.
- Explain why a command, parameter, or numerical choice matters.
- Give a limitation once in the most relevant location.
- Use positive statements without hiding unsupported cases.
- Distinguish mathematical properties, implementation choices, numerical
  results, and physical validation.
- Avoid conversational filler, exaggerated language, and sales language.
- Address the reader as `you` in step-by-step instructions when this improves
  clarity. Use impersonal academic prose for theory and method descriptions.

A technical paragraph should normally follow this order:

1. Setup or question.
2. Method or operation.
3. Result or expected output.
4. Meaning and stated scope.

## 3. Academic language

### 3.1 UK English

Use UK English except in code identifiers, software names, quotations, and
external taxonomy names.

Preferred forms include:

- optimisation, regularisation, discretisation, normalisation;
- behaviour, colour;
- centre, centred;
- modelling, labelled;
- analyse, summarise;
- artefact in ordinary prose.

Do not rename code identifiers to enforce UK spelling.

### 3.2 Punctuation and typography

- Do not use em dashes. Use a comma, parentheses, or a separate sentence.
- Use hyphens consistently in terms such as `phase-field`, `finite-element`,
  `load-displacement`, and `matrix-free`.
- Use commas as thousands separators in Markdown, for example `10,300`.
- Put spaces between numerical values and units in prose.
- Define symbols and units at first use on each self-contained page.
- Use `$G_c$` for fracture toughness and `$\ell_0$` for the regularisation
  length in Markdown equations.
- Do not use raw LaTeX commands outside supported mathematics delimiters.

### 3.3 Remove filler

Delete phrases that delay the technical statement:

- importantly;
- notably;
- crucially;
- it is worth noting that;
- it can be observed that;
- it is clear that;
- needless to say;
- as everyone knows.

Write the fact directly.

## 4. Page structures

### 4.1 Tutorial

A tutorial should contain:

1. Purpose and learning outcome.
2. Prerequisites.
3. Required files and working directory.
4. Geometry and mesh.
5. Material and phase-field parameters.
6. Loads and boundary conditions.
7. Solver selection and convergence settings.
8. Exact execution command.
9. Expected outputs.
10. Visualisation and physical interpretation.
11. Limitations and supported variations.
12. A small next experiment.

Do not hide essential setup in an earlier tutorial unless the dependency is
explicitly linked.

### 4.2 Installation guide

An installation guide should contain:

1. Supported or tested operating systems and Python versions.
2. The recommended installation route first.
3. Alternative routes and their additional requirements.
4. Exact commands from a clean environment.
5. A short verification command.
6. Expected success output.
7. Common errors and recovery steps.
8. Optional backend requirements.

Do not call installation easy, simple, or one-command unless the documented
procedure demonstrates that statement on the stated platforms.

### 4.3 Reference page

A reference page should contain:

1. Purpose.
2. Capability status.
3. Inputs, types, defaults, and units.
4. Supported combinations.
5. Outputs and side effects.
6. Errors and failure conditions.
7. A minimal valid example.
8. Links to relevant tutorials and implementation notes.

Reference pages should be complete enough to use without reading source code.

### 4.4 Explanation or theory page

An explanation page should proceed from the physical model to the numerical
method and then to the PhAST implementation:

1. Physical question and assumptions.
2. Governing equations.
3. Definition of every symbol and unit.
4. Spatial and temporal discretisation.
5. Coupling and solution procedure.
6. Configuration keys controlling the implementation.
7. Numerical limitations and validation scope.

Do not present a mathematical formulation as implemented unless the code and
configuration route support it.

### 4.5 Root README

The root README should remain concise. It should contain:

1. A one-paragraph description of PhAST.
2. Current supported scope and important boundaries.
3. The shortest verified installation and first-run commands.
4. Links to the documentation, examples, capability matrix, citation,
   licence, contribution guide, and issue tracker.
5. One representative result or visual with a precise caption.

Do not duplicate the complete theory, configuration reference, or every
example in the root README.

### 4.6 Example README

Every runnable example README should state:

1. The physical problem and benchmark identifier.
2. Geometry and dimensionality.
3. Material and phase-field formulation.
4. Loading and boundary conditions.
5. Mesh scale relative to `$\ell_0$`.
6. Solver route and important tolerances.
7. Exact command from a stated working directory.
8. Expected runtime only when measured on stated hardware.
9. Expected output files.
10. Which outputs are included in the repository.
11. What comparison or validation supports the example.
12. Known limitations.

Clearly distinguish runnable inputs from retained outputs, manifests,
contracts, schemas, and templates.

## 5. Internal vocabulary and public replacements

Do not expose release engineering, issue tracking, or private research
shorthand in ordinary public prose.

| Avoid in narrative prose | Prefer |
| --- | --- |
| gate, configured gate | verification criterion, validation condition |
| lane | solver path, execution path, workstream |
| tracker | issue, task list, status page |
| receipt | run record, metadata record |
| smoke test | installation check, short verification calculation |
| paper-grade | publication-quality |
| production route | supported solver path |
| production-ready | supported for the documented scope |
| promote, demote | change the documented capability status |
| claim boundary | stated scope or limitation |
| payload | released files, package contents |
| accepted target | converged load increment |
| accepted state | converged state |
| source-locked result | result tied to the stated source revision |
| fail closed | stop with an error when verification cannot be completed |
| safe subcycling | damage subcycling with the stated interval |
| job ID in visible prose | omit it or place it in a reproducibility record |

Use the exact filename or identifier when the reader needs it to run a
command, inspect an output, or reproduce a result.

## 6. Words to avoid

Avoid these words in authored narrative because they are usually vague,
promotional, or associated with formulaic writing:

```text
beacon
bespoke
cornerstone
crucial
delve
foster
game-changing
groundbreaking
harness
holistic
interplay
leverage
leveraging
massive
meticulous
notably
novel
paradigm
paramount
pivotal
revolutionise
seamless
showcase
streamlined
tapestry
testament
underscores
utilise
vital
```

Replace the word with the exact technical statement. For example, replace
`a groundbreaking acceleration strategy` with `the method reduced wall time
from X to Y for the stated benchmark and hardware`.

The list is a writing prompt, not a blind text-replacement rule. Verbatim
quotations, titles, citations, filenames, code, and external project names
are exempt.

## 7. Context-sensitive technical words

The following words are valid when they have a precise software or numerical
meaning. Do not ban them globally.

| Word | Appropriate use |
| --- | --- |
| diagnostic | the documented purpose of `phast doctor`, `precheck`, or a numerical check |
| trajectory | a defined stored time or load sequence |
| provenance | source revision, configuration, and data origin needed for reproducibility |
| artefact or artifact | a generated package, CI artefact, or literal API/file-format term |
| workflow | a GitHub Actions workflow or a clearly defined sequence of operations |
| terminal | instructions that require a command-line terminal |
| proxy | a defined mathematical or computational surrogate measure |
| evidence | an academic argument supported by stated results, not a synonym for files |
| protocol | a formally defined numerical or experimental procedure |

Prefer a more specific term whenever one exists.

## 8. Canonical PhAST terminology

| Concept | Public wording |
| --- | --- |
| Package name | PhAST |
| Framework | modular tensor-based finite-element framework |
| Primary application | phase-field fracture |
| Coupling | staggered solution scheme |
| Damage equation | damage subproblem or phase-field subproblem |
| Mechanics equation | mechanics subproblem |
| Amor model | Amor tension-compression split at first use, then Amor split |
| Miehe model | Miehe spectral split |
| AT2 bound treatment | post-solve projection to `$[d_n,1]$`, then post-clamping |
| AT1 bound treatment | projected conjugate gradients, then projected CG |
| Explicit route | explicit dynamics with the integrator named |
| Quasi-static route | implicit equilibrium solve within the staggered scheme |
| Learned proposal | learned damage proposal followed by the documented checks |
| Learned replacement | audited learned damage replacement with classical fallback |
| Subcycling | damage subcycling with interval `$N$` |
| Fracture toughness | `$G_c$`, with units defined |
| Regularisation length | `$\ell_0$`, with units defined |

Use `subproblem` rather than `sub-problem`.

## 9. Capability and validation statements

Use the public maturity labels consistently:

- **Supported** means the public pathway is documented and covered by the
  stated tests and examples.
- **Beta** means an implemented pathway is available but requires
  case-specific verification.
- **Experimental** means the interface may change and claims must remain
  narrow.
- **Scaffold** means partial infrastructure exists without a complete public
  workflow.
- **Unsupported** means the public solver does not provide the capability.

Do not convert a passing test into a physical-validation claim. Keep these
statements separate:

1. The code executed.
2. The nonlinear or linear solver converged to the stated criterion.
3. The result agrees with a numerical reference.
4. The result agrees with experimental measurements.
5. The method is supported beyond the tested configuration.

Only state the levels demonstrated by the cited result.

When comparing performance, report:

- the complete operation being timed;
- hardware and software environment;
- precision and backend;
- mesh and problem size;
- number of repetitions and summary statistic;
- setup, solve, transfer, inference, and output costs when relevant.

Do not describe one backend as universally fastest.

## 10. Differentiability and learned models

Avoid unqualified statements that PhAST differentiates every fracture
calculation. Prefer:

> Supported tensor operations can participate in PyTorch autograd. History
> updates, bounds, active-set changes, remeshing decisions, and other
> non-smooth operations require case-specific interpretation.

For learned damage updates, state who remains responsible for the physical
solution:

- A learned proposal supplies an initial damage estimate.
- PhAST checks bounds, irreversibility, phase-field boundary conditions, and
  the projected residual where the selected route provides those checks.
- The classical damage solver remains available as the fallback.
- No trained model is implied unless a checkpoint is distributed and linked.

Do not call a learned model faster without measuring its complete inference,
checking, rejection, and fallback cost against the matched classical solve.

## 11. Commands and configurations

- Copy commands from the current CLI help.
- State the working directory when it is not the repository root.
- Use copy-ready fenced code blocks.
- Do not include shell prompts such as `$` inside copy-ready commands.
- Explain placeholders such as `CONFIG` and `OUTPUT_DIR`.
- Use public repository paths rather than private machine paths.
- Show the recommended command before optional variants.
- State whether a command validates, generates files, or executes a solve.

For example:

```bash
python -m phast explain-config CONFIG
python -m phast run CONFIG --validate-only
python -m phast run CONFIG --output_dir RUN
```

Every runnable YAML file should declare its schema version and use only
supported public keys. A file that is not directly runnable must identify
itself as a manifest, contract, schema, or template in both its filename or
heading and its surrounding documentation.

Here `CONFIG` is an input path and `RUN` is a new result directory.
Check this sequence for the selected schema/adapter; legacy
`precheck --config` is not a universal schema-2 command. Separate physical
analysis from integration, and a step limit from final physical time.

## 12. Inputs, outputs, and reproducibility

Document the complete path from input to result:

- geometry source and mesh-generation method;
- named boundaries and physical groups;
- material parameters and units;
- phase-field model and energy split;
- loads and boundary conditions;
- solver type, backend, tolerances, and step count;
- output directory and generated files;
- configuration and source revision recorded with the result.

Do not imply that a retained image proves that its generating calculation can
be reproduced. Distinguish:

- runnable input;
- retained numerical output;
- visualisation generated from retained output;
- manifest describing expected files;
- reference or comparison data;
- template requiring user values.

## 13. Equations, numbers, and tables

- Introduce each equation in prose.
- Define every symbol immediately after the equation or in a nearby table.
- State units and sign conventions.
- Connect equations to the corresponding configuration keys.
- Put more than three related numerical values in a table where practical.
- Use meaningful precision. Do not copy more digits than the result supports.
- Explain a value when it changes the scientific conclusion.
- Do not describe convergence solely through a plot. State the criterion.

## 14. Figures and accessibility

- Use descriptive alternative text.
- Explain what the reader should notice.
- Label axes with quantities and units.
- Use a neutral, high-contrast palette.
- Do not rely on colour alone to distinguish curves or regions.
- Use consistent symbols and colours across related pages.
- Keep captions self-contained and concise.
- Do not write `computed`, `simulated`, or `calculated` before a noun when the
  context already makes this clear.
- Check figures at desktop and mobile widths.

Animations must have a static fallback image and a short explanation of the
displayed time or load interval.

## 15. Citations and links

- Cite the original paper for a mathematical model or benchmark.
- Prefer primary sources and official software documentation.
- Link to the relevant PhAST page rather than duplicating long explanations.
- Check that repository URLs use `CEMS-Lab/PhAST` and documentation URLs use
  `https://cems-lab.github.io/PhAST/`.
- Do not cite a paper as proof of a PhAST implementation unless the public
  code and documentation implement the stated method.
- Keep the citation instructions and software citation metadata consistent.

## 16. Limitations and user support

State unsupported combinations directly. For example:

> Three-dimensional fracture is not supported by the current public solver
> pathway.

Do not hide limitations in vague positive language. Do not repeat the same
qualification after every sentence.

End installation guides and substantial tutorials with a support statement:

> If a documented step fails, open a GitHub issue and include the
> configuration, command, operating system, Python version, PhAST version or
> commit, and complete error output.

Invite contributions without implying that an unsupported feature is already
planned or promised.

## 17. Final checklist

### Accuracy

- [ ] Every described capability exists in the current public code.
- [ ] Maturity labels agree with the capability matrix.
- [ ] Commands and paths are current.
- [ ] Configuration keys agree with the schema.
- [ ] Units and defaults are stated correctly.
- [ ] Numerical and physical validation are not conflated.
- [ ] Optional dependencies are identified.

### Usability

- [ ] The intended reader and task are clear.
- [ ] Prerequisites and working directory are stated.
- [ ] Inputs, commands, and outputs form a complete sequence.
- [ ] A new user can identify the next step.
- [ ] Runnable files are separated from manifests and templates.
- [ ] Failure recovery or support information is present where needed.

### Language

- [ ] The prose uses UK English.
- [ ] Sentences contain one main idea.
- [ ] Internal project vocabulary has been removed.
- [ ] Promotional and formulaic words have been removed.
- [ ] Technical terms are defined and used consistently.
- [ ] Limitations remain visible.
- [ ] No em dashes are present.
- [ ] Physical mechanisms and concrete actions replace abstract promises.
- [ ] Each paragraph adds information rather than restating the aim.

### Presentation

- [ ] Headings describe the content below them.
- [ ] Equations and tables render correctly.
- [ ] Figures have useful captions and alternative text.
- [ ] Links resolve to public locations.
- [ ] Code blocks are copy-ready.
- [ ] The page remains readable on desktop and mobile.

## 18. Exceptions

Code, command output, filenames, configuration keys, API identifiers, quoted
errors, paper titles, and verbatim quotations retain their original spelling
and punctuation. Do not alter a technically meaningful identifier merely to
match prose style.

When this guide conflicts with a file format, external standard, citation,
or required software identifier, preserve the technically correct form and
explain the exception if readers may be confused.

## 19. Scientific writing design directives

The preferred voice is analytical, explanatory, and restrained. Build the
argument through physical causes, named operations, observable consequences,
and appropriately limited conclusions. Precision should make the prose easier
to follow, not merely increase its technical density.

### 19.1 Lead with the physical question

For a research narrative, introduce the material behaviour or unresolved
question before listing computational methods. Explain why the question
matters and what existing descriptions cannot yet resolve. Then introduce
the method as a means of answering it.

Avoid opening with a stack of descriptors such as differentiable, multiscale,
multiphysics, data-efficient, and generalisable. Retain each term where its
specific role is explained. In command references and installation pages,
continue to lead with the user's task rather than a research motivation.

### 19.2 Make causal relationships explicit

Prefer a short physical sequence to a list of interacting phenomena. Name
what changes, what it acts on, and what consequence is expected. Distinguish
a proposed mechanism from an observed one.

For example, a hypothetical account might read:

> Cracks open transport paths. Reaction at the exposed interface changes
> sliding resistance, which can alter load transfer during subsequent loading.

Do not use a model sensitivity alone to establish a physical mechanism.
Explain which comparison, field observation, or measurement supports that
interpretation. Homogenisation transfers effective responses between scales;
differentiation evaluates how those responses depend on specified inputs.

### 19.3 Give each paragraph a distinct job

Use a topic sentence followed by the explanation or evidence needed to
support it. End with an implication or limitation only when it adds meaning.
Vary sentence length to preserve a readable argument; do not turn the
sentence-length guidance into disconnected short statements.

In proposals, distinguish the jobs of adjacent sections:

- The research gap identifies what existing work cannot yet explain.
- The question identifies what the study will investigate.
- The hypothesis states a proposition that a comparison could contradict.
- The aim states the intended contribution.
- The objectives identify the work needed to test the proposition.
- The approach explains how that work will be carried out and assessed.

Do not make each section another version of "we will establish a framework".
Repeat a technical term when needed for accuracy, but remove repeated claims.
Do not substitute loose synonyms merely to make the prose appear varied.

### 19.4 Replace abstract promises with actions and observations

Use verbs such as compute, compare, calibrate, withhold, measure, and test
when they describe the actual work. Specify their objects. Words such as
establish, enable, preserve, and generalise are not prohibited, but require
an explanation of what would demonstrate the claim.

Make the comparison legible: what is varied, what is held fixed, what is
measured, and what result would challenge the interpretation? Include a
numerical target only when it is justified. Never invent a threshold to make
a sentence sound more precise.

Keep physical and methodological hypotheses distinct. A hypothesis about
damage mechanisms differs from a hypothesis that derivative supervision
improves prediction at a given computational cost. A negative result for
one need not invalidate the other.

### 19.5 Match the verb to the evidence

Use present tense for verified capabilities and established descriptions,
past tense for completed work, and future tense for proposed actions. Use
conditional language for outcomes that remain hypotheses. A confident plan
can say "we will compare" without promising "the comparison will prove".

Keep these distinctions visible in the prose:

- Calibration fits parameters; independent measurements test the resulting
  predictions under specified conditions.
- Agreement with a reference solver assesses numerical or approximation
  error; it does not independently validate the physical model.
- An unseen combination within sampled ranges differs from extrapolation
  beyond those ranges or transfer to another material.
- A preliminary calculation demonstrates only the operations and responses
  actually assessed, even if its visualisation resembles a final result.

Place a material limitation next to the claim it qualifies. State it once
clearly, and repeat it only where a figure or section must stand alone.
Preserve unresolved convergence, energy balance, validation, and scope
limitations during editing. Do not replace them with vague assurances.

### 19.6 Use technical terms according to their role

| Term | Writing directive |
| --- | --- |
| Differentiable | Describe the formulation or verified mathematical property, with its domain of validity. |
| Automatic differentiation | Name the computational technique when that is the technique used. |
| Sensitivity | Identify the response and physical input being differentiated. |
| Constitutive tangent | Identify the constitutive update and which prior states are held fixed. |
| Gradient | Name the differentiated quantity; distinguish optimisation gradients from physical-response sensitivities. |
| History dependence | State which earlier changes affect the later response and how this is assessed. |
| Generalisation | Identify the withheld conditions, sampled ranges, and any extrapolation. |

Use the simplest accurate term for the passage. Do not add implementation
details unless they explain the method, its assessment, or a limitation.

### 19.7 Describe feasibility and impact through concrete commitments

For proposed work, connect each difficult step to its required input,
assessment, and useful outcome. If success is uncertain, explain what narrower
result remains achievable. "Revise the model" is a possible activity, not a
complete description of a fallback outcome.

Distinguish a measured runtime for one example from the estimated workload
of a research programme. Identify assumptions when extrapolating costs.
Do not turn a writing edit into a new scientific or resource commitment.

For impact, name the intended user, the decision the output could inform,
and the further testing or adoption needed. Keep long-term sector benefits
separate from project deliverables. Use policy and investment figures
selectively; they establish context but do not substitute for a use case.

### 19.8 Make tables and figures carry specific information

Use a table for repeated mappings, such as measurement source, calibration
use, independent assessment, and conditions. Do not repeat every table entry
in the surrounding paragraph. Explain the relationship or gap that matters.

A scientific figure caption should identify the problem, quantities,
conditions, and interpretation needed to read the result. Distinguish
schematics, hypotheses, preliminary calculations, and validated comparisons.
Define visual alterations such as smoothing, exaggerated geometry, or lines
joining discrete states when they affect interpretation.

Keep essential limitations with the figure. Move routine configuration
detail to a linked method description when appropriate; do not achieve a
shorter caption by hiding unresolved scientific issues.

### 19.9 Remove formulaic writing without flattening the science

Review repeated abstract nouns, identical paragraph openings, method lists,
unsupported superlatives, and stock transitions. Repair the missing meaning
before substituting words. Formal language, repetition, or polished syntax
alone cannot establish whether a person or an AI wrote a passage.

Do not mechanically remove every passive construction or specialist term.
Use active voice to clarify who performs an action. Use passive voice when
the measured quantity or operation is the relevant subject. Preserve logical
connections and qualifications even when a shorter sentence is possible.

## 20. Writing examples and revision checks

The pairs below illustrate editing decisions. They are not quotations from
a manuscript, statements of current PhAST support, or approved research
commitments. Use a replacement only when its scientific content is accurate.

| Less informative wording | More specific wording |
| --- | --- |
| We will establish a generalisable framework for coupled evolution. | We will test whether a model trained on one set of exposure histories predicts the response under withheld histories. |
| The method captures complex interactions across scales. | The tow model supplies stress and transport response to the woven equations, which determine the next local loading and exposure conditions. |
| Extensive validation will demonstrate accuracy. | Separate measurements will test the calibrated model; reference simulations will assess the additional error introduced by learning. |
| Sensitivities reveal the governing mechanism. | Sensitivities identify influential inputs within the model; paired histories and damage fields test the proposed mechanism. |
| Preliminary results demonstrate reliable failure prediction. | The preliminary calculation produces a damage field; its post-peak energy balance has not yet been established. |
| The model will generalise to new materials. | Predictions for a second material will use independently specified properties without refitting the learned weights. |
| The work will transform sustainable aerospace design. | The results could help researchers select interphase properties for subsequent material testing. |

Before completing a scientific prose revision:

1. Read only the opening sentences. They should form a coherent argument
   without repeating the same promise.
2. Underline the main claim in each paragraph and identify its supporting
   mechanism, comparison, or source.
3. Check that every proposed method has a stated purpose and that existing
   results are not written as evidence for an untested extension.
4. Check that a reader can distinguish physical validation, numerical
   verification, learning accuracy, and computational performance.
5. Remove duplicated explanations before deleting substantive caveats.
6. Compare the revision with the original for changes in certainty, scope,
   attribution, and proposed commitments. Flag scientific changes explicitly.
7. Read the result aloud for long noun sequences, abrupt transitions, and
   repeated sentence patterns. Prefer connected explanation over slogans.

## Trajectory And Dataset Storage

Use HDF5 (`training_data.h5`, configuration value `h5`) as the default
trajectory format in documentation, READMEs, examples, and teaching notebooks.
State that trajectory saving must be enabled where applicable; do not imply
every run writes a trajectory.

Show Zarr only as an explicit alternative for a workflow that requires it.
Do not describe HDF5 as legacy-only or Zarr as the preferred default. Keep
commands, expected filenames, readers, and manifests consistent with the
selected format. Preserve the recorded format of historical results, and do
not relabel saved notebook output as evidence of an HDF5 run without rerunning
it. Clear stale outputs when changing a notebook's storage backend.

A single HDF5 file reduces the number of files to synchronise. It does not
make simultaneous writes from different machines safe.
