# Reference data provenance — B5 PMMA dynamic branching (Bleyer 2017)

## Source

`refs/Bleyer, Roux-Langlois, Molinari (2017) - phase-field dynamic
branching velocity-toughening.pdf` — *International Journal of
Fracture* **204**:79–100.

Numerical reference values were read directly from the published
figures (paper pages 88–91; PDF figures 5, 6, 7, 9). The Bleyer paper
reports a **dynamic** (pre-strain + sudden release) experiment; it
does **not** publish a quasi-static force–displacement curve, so the
acceptance gates here are over morphological + energetic quantities,
not F–U.

## Naming clarification

The umbrella issue (#256) describes B5 under "4 quasi-static
benchmarks". In this codebase, B5 is the **dynamic** PMMA branching
benchmark; see `papers/paper/BENCHMARK_SETTINGS.md` §B5 (p. 238). The README
(L 630) and CHANGELOG entries are consistent with this. The acceptance
gates below are aligned with the source's actual reported quantities.

## Figure-level reference values

### Bleyer Fig 5 (p. 88) — branching morphology vs ΔU

Phase-field distribution at the end of crack propagation:

| Panel | ΔU (mm) | Snapshot time | Morphology |
|---|---|---|---|
| (a) | 0.035 | 40 µs | single straight crack |
| (b) | 0.038 | 40 µs | single straight crack (just below threshold) |
| (c) | 0.040 | 40 µs | Y-branch, branching angle ≈ 30° |
| (d) | 0.045 | 20 µs | Y-branch, earlier onset |

→ **Morphology gate**: ΔU ≤ 0.038 mm must give 1 arm in the
right-of-precrack region (no branching); ΔU ≥ 0.040 mm must give
≥ 2 arms (Y-branch). Branching angle: 30° ± 10° for ΔU ≥ 0.040 mm.

### Bleyer Fig 6 (p. 89) — Γ/Gc vs crack-tip horizontal position

Two loading levels overlaid (lower bound at Γ = Gc, upper at Γ = 2Gc):

| Trace | ΔU (mm) | Behaviour | Reading |
|---|---|---|---|
| Red curve  | 0.035 | regular, no branching | rises from Gc to ~1.5 Gc, plateau |
| Blue curve | 0.045 | branching, total | climbs to ~2.0 Gc, then jumps with peaks ~2.5–2.7 Gc |
| Green curve | 0.045 | branching, single tip after split (Γ/2) | falls back near Gc |

Bleyer text (p. 88, right col., last paragraph) explicitly states:
"Γ increased from Gc to 1.5 Gc and no branching has been observed"
for ΔU = 0.035 mm; "branching has been observed slightly after
Γ exceeded 2 Gc" for ΔU = 0.045 mm.

→ **Energy gate** (Γ/Gc envelope at peak):
ΔU = 0.035 mm → 1.0 ≤ Γ/Gc ≤ 1.5 (within ± 20 %);
ΔU = 0.045 mm → Γ/Gc peaks ≥ 2.0 before branch (within ± 20 %).

### Bleyer Fig 7 (p. 89) — damage profile d(y) at five positions

Five positions A–E along the crack path (ΔU = 0.04 mm):
A (x = 5 mm), B (x = 12 mm), C (x = 18 mm), D (x = 20 mm),
E (x = 22 mm). Profiles widen with crack advance; branching
position D shows central depression d < 1 (pre-split signature).

Initial profile A matches the 1-D parabolic d(y) = (1 − |y|/(2 l0))²
("black square line"). This is a model-consistency check rather than
a quantitative gate.

### Bleyer Fig 9 (p. 91) — Γ/Gc vs v / cR master curve

Five loading levels (ΔU = 0.035, 0.038, 0.04, 0.045, 0.05 mm) collapse
onto a single L-shaped master curve. Reading the published figure:

| v / cR | Γ/Gc (envelope) |
|---|---|
| 0.0–0.4 | 1.0–1.2 |
| 0.4–0.6 | 1.2–1.5 |
| 0.6–0.65 | 1.5–2.0 |
| > 0.65  | rises sharply, > 2.0 |

Limiting velocity (Zhou 1996, dotted line): v / cR ≈ 0.75.

→ **Velocity gate**: peak v / cR ∈ [0.40, 0.75] for any ΔU
(plateau ≈ 0.6 cR, ± 20 %). Bleyer p. 91 (Sec 3.5): "limit speed
around 0.7 cR".

## Material constants (from Bleyer p. 86, our `pmma_bleyer` preset)

| Quantity | Value | Unit |
|---|---|---|
| E   | 3 090 | MPa |
| ν   | 0.35 | — |
| ρ   | 1 180 | kg/m³ (= 1.18e-9 t/mm³) |
| Gc  | 0.3 | N/mm (= 300 J/m²) |
| l0  | 0.1 | mm |
| cR (computed from Bleyer p. 86) | 906 | m/s |

## Acceptance summary

See `../ACCEPTANCE.md` for the formal pass/fail rules, including
tolerances. The values above are the source of truth for those rules.
