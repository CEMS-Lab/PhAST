# L-shaped panel reference data — Rudshaug 2024 digitisation

This directory ships the digitised reference solutions for the
quasi-static L-shaped panel benchmark used by `compare.py`.

## Files

| File | Source | Status |
|---|---|---|
| `ambati_2015_lshaped_concrete.csv` | Ambati et al. (2015), Comput. Mech. 55:383-405, Fig 19 (hybrid) | shipped (#133, prior work) |
| `rudshaug_2024_crack_speed_weak.csv`   | Rudshaug et al. (2024), Int. J. Fract. 245:57-73, Fig 12a, p71 | **shipped (this PR, #133/#256)** |
| `rudshaug_2024_crack_speed_strong.csv` | Rudshaug et al. (2024), Fig 12b, p71 | **shipped (this PR)** |
| `rudshaug_2024_crack_path_weak.csv`    | Rudshaug et al. (2024), Fig 12d, p71 | **shipped (this PR)** |
| `rudshaug_2024_crack_path_strong.csv`  | Rudshaug et al. (2024), Fig 12e, p71 | **shipped (this PR)** |
| `rudshaug_2024_fracture_init_force.csv`| Rudshaug et al. (2024), text p64 + Fig 12c colorbar | **shipped (this PR)** |
| `rudshaug_2024_lshaped_glass.csv`      | Rudshaug **2023** (Glass Struct Eng) — F-d curve | **NOT shipped, see below** |

## Paper citation

Rudshaug J, Borvik T, Hopperstad OS (2024). Modeling brittle crack
propagation for varying critical load levels: a dynamic phase-field
approach. *International Journal of Fracture* **245**(1): 57-73.
DOI: [10.1007/s10704-023-00754-3](https://doi.org/10.1007/s10704-023-00754-3).
Open access (CC BY 4.0).

## Honest caveat — the 2024 paper does NOT publish a force-displacement curve

The Rudshaug 2024 paper presents the L-shaped soda-lime glass test
in **Section 4 (pages 64-66 + Fig 9-12)** but reports only:

1. **Crack propagation paths**: Y[mm] vs X[mm] for weak/strong
   specimens (Fig 12d, e).
2. **Crack propagation speed**: speed[m/s] vs time[μs] for weak/strong
   specimens, three tension-compression splits each (Fig 12a, b).
3. **Fracture-initiation force band** as text (p64): "the force level
   at fracture initiation varied from ~75 N to ~115 N" across the 20
   experiments. The Fig 12c colorbar covers 70-120 N.

A **force-displacement curve is not published** in the 2024 paper.
The full F-d data lives in the earlier companion paper:

> Rudshaug J, Hopperstad OS, Borvik T (2023). Effect of load level on
> cracking of L-shaped soda-lime glass specimens. *Glass Struct. Eng.*,
> DOI: [10.1007/s40940-023-00239-8](https://doi.org/10.1007/s40940-023-00239-8).

That paper is **not currently in `refs/`**. To finish the F-d L2
comparison contemplated in #133, fetch the 2023 PDF and digitise its
F-d figure under `rudshaug_2024_lshaped_glass.csv` (the filename is
preserved because `compare.py` already references it).

For now, the available acceptance criterion against the 2024 paper is
the **fracture-initiation force band [75, 115] N** — see
`ACCEPTANCE.md` and `compare.py --reference rudshaug_2024`.

## Material properties (from Rudshaug 2024, p65)

| Property | Value | Notes |
|---|---|---|
| density ρ | 2500 kg/m³ | quasi-static, ρ enters only through inertial spring boundary |
| Young's modulus E | 70 GPa | |
| Poisson ratio ν | 0.23 | |
| critical energy release rate Gc | 8 J/m² | |
| length scale ℓ | 0.4 mm | |
| residual stiffness k | 0 | |
| phase-field viscosity η | 1e-9 Ns/mm² | |
| mesh | 174,375 shell elements at 0.2 mm × 0.2 mm | h = ℓ/2 |
| boundary | upper rigid block displacement-controlled, tie-strap modelled with springs kt=38.0 N/mm, kr=4750 N·mm | Fig 9b |
| critical principal stress σc | tuned per discretisation; chosen to match the experimental fracture-initiation force | weak ≈ 75 N, strong ≈ 115 N |

These match the parameters baked into our `l_shaped_glass` material
preset in `material.py` (do not edit per task constraints).

## Loading rate

Quasi-static, displacement-controlled tensile pull on the upper rigid
block (Fig 9b). The 2024 paper uses an explicit time integration with
mass scaling, so a small artificial inertia is present, but the test
is loaded slowly enough that the dynamic effects are negligible compared
to the post-initiation crack speed (see paper Section 4).

## Digitisation methodology

Hand-digitised from the published PDF using on-screen visual reading
of grid intersections. Sample density:

- crack speed: ~20 points per curve at 5-10 μs spacing
- crack path: ~15 points per curve at 2-5 mm x-spacing
- force-init band: 4 numerical values (text + colorbar bounds)

Estimated uncertainty:

- speed: ±50-100 m/s (weak) / ±80-120 m/s (strong, oscillates more sharply)
- y-coordinate of crack path: ±1-2 mm
- time / x: ±2-3 μs / ±0.5-1 mm
- fracture-init force band bounds: ±5 N (text rounded to nearest 5 N)

This is **engineering-level reference data** suitable for benchmark
anchoring (band + L2 envelope checks), not for sub-percent comparisons
or convergence studies. If finer resolution is needed in the future,
fetch the underlying numerical data from the authors or extract it
from the original LS-DYNA result files.

## CSV schema

All CSVs are comma-separated with header rows. Columns are documented
in each file's prologue comments. The `_speed_*` and `_path_*` files
have one column per tension-compression split (Amor / Miehe / Hybrid)
and a NaN placeholder where a curve runs out of plot range. The
`_fracture_init_force` file has two columns: a textual `quantity` key
and a numeric `value_N` (force in newtons).
