# L-Shaped Panel Reference Data Provenance

**Status: concrete reference shipped; partial 2024 glass references shipped (crack speed/path + initiation-force band); F-d curve still pending the 2023 companion paper. See `README.md` for the 2024 digitisation details.**

## Reference data files

| File | Source | Material | Peak | Status |
|---|---|---|---|---|
| `ambati_2015_lshaped_concrete.csv` | Ambati 2015 Fig 19 (Hybrid curve) | concrete (E=25.85 GPa, ν=0.18, Gc=89 J/m², ℓ=1.1875 mm) | ~16 kN at u=0.30 mm | shipped (#133) |
| `rudshaug_2024_crack_speed_weak.csv`   | Rudshaug 2024 Fig 12a, p71 | glass (E=70 GPa, ν=0.23, Gc=8 J/m², ℓ=0.4 mm) | n/a (speed-vs-time) | shipped (#133/#256) |
| `rudshaug_2024_crack_speed_strong.csv` | Rudshaug 2024 Fig 12b, p71 | glass (strong) | n/a (speed-vs-time) | shipped (#133/#256) |
| `rudshaug_2024_crack_path_weak.csv`    | Rudshaug 2024 Fig 12d, p71 | glass | n/a (Y-vs-X path) | shipped (#133/#256) |
| `rudshaug_2024_crack_path_strong.csv`  | Rudshaug 2024 Fig 12e, p71 | glass (strong) | n/a (Y-vs-X path) | shipped (#133/#256) |
| `rudshaug_2024_fracture_init_force.csv`| Rudshaug 2024 text p64 + Fig 12c colorbar | glass | 75-115 N initiation-force band | shipped (#133/#256) |
| `rudshaug_2024_lshaped_glass.csv`      | Rudshaug 2024 — paper does **not** publish F-d | glass | ~0.27-0.32 kN (legacy band) | **NOT IN 2024 PAPER**, see README |
| `lshape_glass_PLACEHOLDER.csv`         | hand-sampled scaffold (NOT digitised) | glass | ~0.30 kN at u~0.25 mm | scaffold pending Rudshaug 2023 F-d (#133) |
| `lshape_concrete_PLACEHOLDER.csv`      | hand-sampled scaffold (NOT digitised) | concrete (Winkler band) | ~6.5 kN at u~0.30 mm | scaffold pending Winkler 2001 digitisation; Ambati CSV is the active concrete ref |

## Background

The L-shaped panel benchmark is anchored in the literature by:

- **Winkler (2001)** — experimental crack path in concrete. Used as a
  qualitative reference for *crack path*, not load value. Experimental
  peak load is in the **6-8 kN** band.
- **Ambati, Gerasimov, De Lorenzis (2015)**, *Computational Mechanics*
  **55**(5):383-405 — AT2 phase-field reproduction of Winkler's
  geometry with concrete parameters. Fig 19 shows the simulated
  hybrid-formulation peak load at **~16 kN at u=0.30 mm**, then softens
  to ~1 kN by u=1.0 mm.
- **Rudshaug, Hopperstad, Borvik (2024)** — LS-DYNA glass L-panel
  benchmark. Peak load ~0.27-0.32 kN depending on mesh.

### Winkler-vs-Ambati discrepancy (important)

Ambati's simulation **overshoots** Winkler's experimental peak load by
~2× (~16 kN simulation vs ~6-8 kN experiment) at the same nominal
material parameters. Likely causes:

- Concrete is a heterogeneous material; the homogeneous E, ν, Gc fit is
  approximate
- AT2 phase-field is known to over-predict peak loads slightly relative
  to ductile-damage models; an order-2× overshoot is high but plausible
  for unstabilised AT2
- Winkler's experimental geometry / loading rate is not perfectly
  reproduced by the idealised geometry in Ambati's Fig 16a

**Our `l_shaped_concrete` preset matches Ambati's parameters exactly**
(E=25.85 GPa, ν=0.18, Gc=89 J/m², ℓ=1.1875 mm) — so our solver should
match Ambati Fig 19 (~16 kN), **not** Winkler experiment (~6-8 kN).

The 2D solver reports reaction force per unit out-of-plane thickness.
The physical L-panel benchmark is a 100 mm thick specimen, so
`compare.py` multiplies the simulated concrete reaction by 100 before
comparing against the digitised total-force curve.

`compare.py` should use the Ambati 2015 concrete curve / 15-17 kN band for the
homogeneous AT2 concrete preset. The Winkler 6-8 kN experiment remains
crack-path context, not the quantitative load target.

## CSV format

Two columns, whitespace-separated, `u_y [mm]  R [kN]`. Comments allowed
with `#` prefix. The Ambati CSV contains the **monotonic envelope only**
(loading from u=0 to u=0.30 mm peak, then softening to u=1.0 mm). The
cyclic unloading branches in Fig 19 (which return to u=-0.20 mm and
reload) are skipped — they're for testing crack closure and are
secondary to peak/softening matching.
