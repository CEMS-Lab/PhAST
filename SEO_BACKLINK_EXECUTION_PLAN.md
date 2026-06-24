# PhAST SEO Backlink & Visibility Execution Plan

Status update from this run:

- Repository metadata update completed on GitHub:
  - Description: `PhAST: A matrix-free, differentiable PyTorch solver for phase-field fracture.`
  - Homepage: `https://cems-lab.github.io/PhAST/`
  - Topics added: `computational-mechanics`, `differentiable-simulation`, `fracture-mechanics`, `phase-field-fracture`, `pytorch`

## Priority A (first 24–48 hours): high-authority backlinks

1. Add repo links to university/lab pages
   - CEMS Lab page: add one sentence linking to both:
     - `https://cems-lab.github.io/PhAST/`
     - `https://github.com/CEMS-Lab/PhAST`
   - PI pages (City, University of London or home institution profiles): add the same two links in research/software/project sections.
   - Co-author pages (Molinari, Ponnusami, Subhash, Ani): add links in “Projects / Software / Publications” blocks.
2. Update arXiv page fields for the paper
   - Add both links in the abstract text or comments:
     - `https://cems-lab.github.io/PhAST/`
     - `https://github.com/CEMS-Lab/PhAST`
   - Keep the link anchor text exactly `PhAST`.
3. Create Zenodo DOI landing
   - Link repository release/tag on Zenodo.
   - In zenodo metadata, include:
     - “Source code: https://github.com/CEMS-Lab/PhAST”
     - “Documentation: https://cems-lab.github.io/PhAST/”
     - “Keywords: phase-field fracture, PyTorch, differentiable mechanics”

## Priority B (days 2–5): authoritative ecosystem backlinks

4. ResearchGate project
   - Add project page in the Lab/software section.
   - Include the canonical site URL and GitHub URL above.
5. PyTorch Ecosystem submission
   - Evaluate and submit PhAST at `https://pytorch.org/ecosystem/`
   - Use this exact phrase in description:
     - `PhAST is a matrix-free, differentiable PyTorch solver for phase-field fracture.`
6. Cross-link in one external conference/demo artifact if any
   - If you have a poster/PDF/supplement site, add a short callout and links.

## Priority C (week 1): community amplification

7. Single launch post (X and/or LinkedIn)
   - Include one cracked-branching GIF and one sentence with target link:
     - `https://cems-lab.github.io/PhAST/`
8. Tutorial/citation hygiene
   - Ensure every downstream tutorial, workshop, and course page that references PhAST links back to:
     - GitHub Pages first, GitHub repo second.

## Copy-ready backlink text (paste this)

- University / lab / PI page blurb:

```text
PhAST is an open-source, matrix-free, differentiable PyTorch solver for phase-field fracture, developed at CEMS Lab.
Source code: https://github.com/CEMS-Lab/PhAST
Documentation: https://cems-lab.github.io/PhAST/
```

- arXiv / publication comments text:

```text
PhAST is available as open software: https://github.com/CEMS-Lab/PhAST.
Documentation is hosted at: https://cems-lab.github.io/PhAST/.
PhAST is an open-source, matrix-free, differentiable PyTorch solver for phase-field fracture.
```

- Zenodo metadata text:

```text
PhAST is an open-source matrix-free and differentiable PyTorch solver for phase-field fracture workflows. See:
GitHub: https://github.com/CEMS-Lab/PhAST
Docs: https://cems-lab.github.io/PhAST/
Keywords: phase-field fracture, PyTorch, differentiable mechanics
```

- ResearchGate project description:

```text
PhAST is a matrix-free, differentiable PyTorch solver for phase-field fracture, providing reproducible examples and public benchmark workflows.
Documentation: https://cems-lab.github.io/PhAST/
GitHub: https://github.com/CEMS-Lab/PhAST
```

- Short social post:

```text
PhAST is now publicly released: a matrix-free, differentiable PyTorch solver for phase-field fracture.
Explore examples, docs, and workflows at https://cems-lab.github.io/PhAST/ .
```

## Internal technical follow-up (already done in-repo)

- README first H1 now includes the exact “PhAST ... phase-field fracture” phrase.
- Docs homepage now uses strong keyword-first framing for solver intent.
- Sphinx metadata now includes:
  - `html_title`: `PhAST | PyTorch Phase-Field Fracture Solver`
  - `html_meta` description + keywords
  - `html_baseurl` and sitemap extension enabled via `sphinx-sitemap`
- `sphinx-sitemap>=2.5` added to docs requirements so GitHub Pages builds a sitemap.

## Commands used from this run

- `gh repo edit CEMS-Lab/PhAST --description "..."`
- `gh repo edit CEMS-Lab/PhAST --homepage https://cems-lab.github.io/PhAST/`
- `gh repo edit CEMS-Lab/PhAST --add-topic ...`
- `gh repo edit CEMS-Lab/PhAST --remove-topic ...`

## Suggested 7-day tracker (copy into issues or project board)

- [ ] PI/lab pages include PhAST links (A1)
- [ ] arXiv comments/abstract includes PhAST links (A2)
- [ ] Zenodo DOI published and links are indexed (A3)
- [ ] ResearchGate project created/updated (B4)
- [ ] PyTorch Ecosystem list submission completed (B5)
- [ ] Social launch post published with tracking (C7)
- [ ] Google Search Console request indexing for `https://cems-lab.github.io/PhAST/` sitemap
