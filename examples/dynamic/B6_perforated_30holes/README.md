# B6 Perforated 30-Hole Plate

Selected perforated PMMA plate example. The source archive still uses old
`B4a_perforated_30holes` naming, but this flat public-candidate folder uses the
correct public B6 name.

Run:

```bash
python -m phast run examples/dynamic/B6_perforated_30holes/config.yaml
```

Release note: do not expose the old B4 perforated folder names in public PhAST.
Use this B6 folder as the promotion source.

Expected public visuals are `initial_conditions.png`, `thumbnail.png`,
`damage_multipanel.png`, and the CSV history/energy files.

Related public B6 variants are `B6_perforated_10holes`,
`B6_perforated_1hole_near`, and `B6_perforated_1hole_far`.
