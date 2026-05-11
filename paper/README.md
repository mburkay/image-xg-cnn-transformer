# MLSA 2026 Paper — Build Instructions

This directory contains the LaTeX source for the MLSA 2026 submission:
*"Spatial Resolution and Token Count Matter: A CNN+Transformer Hybrid for Image-Based Expected Goals"*.

## Target venue

- **Workshop:** MLSA 2026 (13th Workshop on Machine Learning and Data Mining for Sports Analytics), co-located with ECML/PKDD 2026, Naples, Italy, 7 September 2026.
- **Submission deadline:** 5 June 2026, 23:59 AoE.
- **Camera-ready deadline:** 10 July 2026.
- **Format:** Springer LNCS (`llncs.cls`), 9 content pages + unlimited references, PDF.
- **Submission system:** CMT.
- **Presentation:** in-person required (no virtual).

## Layout

```
paper/
├── main.tex              # Full first-draft LaTeX source (LNCS, 9 content pages on A4)
├── references.bib        # All citations resolved; Wagenaar 2017 ICPRAM page
│                         # numbers to re-verify against the primary record
│                         # before camera-ready
├── figures/
│   ├── F1_token_sweep.png       # Main-paper figure, Section 3.1 (headline)
│   ├── F2_calibration.png       # Main-paper figure, Section 3.4 (Additional Analyses)
│   └── F3_attention_rollout.png # Supplementary reference only (cited in
│                                 # Section 3.4 but not embedded in the 9-page PDF)
├── README.md             # this file
└── .gitignore            # excludes llncs.cls, splncs04.bst, build artefacts
```

`llncs.cls` and `splncs04.bst` are Springer-distributed files and are
**not** committed to the repository. Download them from Springer before
building (see step 1 below).

## Build steps

1. **Download the Springer LNCS bundle** (one-time setup):
   ```
   # From the Springer LNCS author info page:
   # https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines
   # Download llncs2e.zip and extract llncs.cls + splncs04.bst into paper/
   ```
   Required files in `paper/`:
   - `llncs.cls`
   - `splncs04.bst`

2. **Compile**:
   ```
   cd paper
   pdflatex main
   bibtex main
   pdflatex main
   pdflatex main
   ```
   Or with `latexmk`:
   ```
   latexmk -pdf main
   ```

3. **Output:** `paper/main.pdf` — keep under 9 content pages
   (references unlimited).

## Section ↔ artefact map

`main.tex` follows the structure planned during the claims-freeze
phase. The main paper includes T1, T2, F1, and F2; T3 and F3 are
retained as supplementary artefacts and summarized in text.

| Tag | Asset | Where it appears |
|---|---|---|
| T1 | Controlled six-cell headline ablation | main paper, Section 3.1 |
| T2 | $2{\times}2$ $\sigma{\times}$penalty ablation | main paper, Section 3.3 |
| F1 | `F1_token_sweep.png` | main paper, Section 3.1 |
| F2 | `F2_calibration.png` | main paper, Section 3.4 (Additional Analyses) |
| T3 | Five-point sigma sweep at 128 | supplementary repo only; 2-sentence summary in Section 3.1 paragraph head |
| F3 | `F3_attention_rollout.png` | supplementary repo only; one paragraph in Section 3.4 references it as a qualitative visualization |
| — | Rendering protocol audit paragraph | main paper, Section 3.2 |
| — | Methodology: seed sensitivity paragraph | main paper, Section 3.4 (Additional Analyses) |

Supplementary material (NOT in this 9-page PDF, lives in the
repository root): the full per-cell experiment log
(`EXPERIMENTS.md`), the full hyperparameter document
(`HYPERPARAMETERS.md`), the source-backed positioning notes
(`RELATED_WORK_NOTES.md`), and the per-run eval JSONs once the
checkpoints are deposited on the public archive. These are referenced
from the paper by URL.

## Status

- [x] Every `% TODO` block in `main.tex` is filled (full first-draft prose).
- [x] BibTeX entries resolved with no undefined cites; SoccerTransformer entry removed (not cited); Skor-xG metadata verified against the CVF proceedings.
- [x] Final repository URL inserted in `\subsubsection*{Reproducibility.}` (currently a private GitHub repository; made public upon acceptance).
- [x] ORCID iD set; institute address (Sakarya University, Department of Computer Engineering) set.
- [x] Page fit verified: 10-page PDF on A4 with content $\leq$ 9 pages and references on pages 9–10, matching MLSA 2026's "9 content pages + unlimited references" rule.
- [x] Compile clean except for one minor `Overfull \hbox` (8.18 pt) in the Introduction "Positioning" paragraph and the `amsmath \vec` notice from the LNCS class; both are cosmetic and typical for LNCS submissions.

## Remaining before CMT upload

- [ ] Author read-through pass (abstract through conclusion) for prose and emphasis.
- [ ] (Optional) restructure the Introduction "Positioning" paragraph or apply `\sloppypar` to clear the 8.18 pt overfull.
- [ ] Verify the ICPRAM 2017 page numbers in the Wagenaar entry against the primary record before camera-ready.
- [ ] Run `latexmk -pdf main` one final time on a clean tree just before CMT upload.
- [ ] Flip the repository to public after acceptance (`gh repo edit mburkay/image-xg-cnn-transformer --visibility public`).
