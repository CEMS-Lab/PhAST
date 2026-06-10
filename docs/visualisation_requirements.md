# Visualisation requirements

This is the repository checklist for paper figures, demo figures, and HPC
visual outputs. It distils the persistent project memory into rules that
figure-generation scripts and result-promotion scripts should follow.

## Publication figures

- Use paper-matching serif typography in every Matplotlib script that writes
  paper or demo figures:

```python
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})
```

- Use column-aware sizing. Single-column figures should be about `3.5 in`
  wide; double-column figures about `7 in`; full-page composites no wider than
  about `8.5 in`.
- Save paper PNGs at about `200 dpi`, with `bbox_inches="tight"` unless the
  layout requires fixed whitespace. If a paper PNG exceeds about `1 MB`, reduce
  dpi, crop whitespace, or switch to a vector format when appropriate.
- Keep text legible at the final paper size. Axis labels, legends, colorbars,
  and panel letters must be readable without zooming.
- Damage-field figures must show physically interpretable cracks. Do not
  promote diffuse smeared fields as successful fracture results unless the
  caption explicitly labels the run as diagnostic or rejected.
- For particle/inclusion demos, show the particle or inclusion geometry on the
  final damage PNG and on the GIF frames used for review.
- Use consistent damage color limits, normally `d in [0, 1]`, and label the
  colorbar as damage `d`.

## Animations

- Prefer MP4 for dense time histories. GIFs are for paper supplements, issue
  review, and quick visual checks.
- Cap GIFs at roughly `60` to `100` frames. Decimate time and crop to the
  region of interest instead of saving every solver step.
- Typical GIF size should be `1` to `5 MB`. If a GIF exceeds about `10 MB`,
  reduce frames, resolution, or crop before promoting it.
- Every promoted crack-propagation GIF should include enough frames to show
  crack initiation, interaction with geometry or particles, and final state.

## Image dimensions for review

- Before opening a PNG/JPG in a coding session, check dimensions first. Do not
  read images larger than `2000 px` on either axis.
- When writing a generator, keep `figsize * dpi < 2000` on the larger axis for
  review-facing PNGs. For example, `figsize=(9, 5.6), dpi=200` gives
  `1800 x 1120`.
- Re-check dimensions after every regeneration, including figures pulled back
  from HPC.
- If a generated image is oversized, fix the generator or downsample a review
  copy before opening it.

## Required visual outputs by run type

Use `docs/STANDARD_OUTPUTS.md` for the full artifact list. At minimum:

- Forward dynamic fracture: `damage_final.png`, damage evolution animation,
  crack-tip/path plot when a crack is present, energy plot, and run metadata.
- Quasistatic validation: `damage_final.png`, load-displacement plot,
  staggered/residual convergence plot, and comparison plot/report when a
  reference exists.
- Inverse demos: truth, init, and recovered final damage PNGs; truth, init, and
  recovered damage evolution animations; loss curve; parameter/error curve; and
  a JSON/CSV trajectory.
- HPC-promoted results: include `run_manifest.json` or `PROMOTION.md` so the
  figure can be traced to job id, git commit, slurm script, and result path.

## Review gates

Before a visual result is used in a paper claim:

1. Verify the run is from the expected git commit and slurm script.
2. Verify the final damage field looks like a crack, not only a diffuse band.
3. Verify crack arrival and interaction happen in the intended time/load window.
4. Verify the promoted figure uses the paper font and publication sizing.
5. Verify images are under the review dimension limit before opening them.
6. Update the related issue with the accepted/rejected status and artifact path.
