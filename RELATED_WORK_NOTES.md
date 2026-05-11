# Related Work Notes — Source-Backed

> **Status (release-repo readers):** these are *working notes* from the
> claims-freeze phase, not a polished related-work survey. The main
> paper source (`paper/main.tex`) does not contain any TODO marks;
> the unticked checklist items and `[TODO before paper draft v1]`
> placeholders preserved below are historical research-process
> artefacts kept for transparency.

Working notebook for the paper's Related Work section. Citations and direct
quotes from primary sources where possible. Used to constrain the
positioning paragraph of Section 1 in the main paper.

## xG-CNN (Matteotti & Sotudeh 2024)

**Citation:** Matteo Matteotti & Hadi Sotudeh, *"The Power of Pixels:
Exploring the Potential of CNNs for Expected Goals (xG) in Football"*,
ResearchGate technical report, July 2024.
DOI: [10.13140/RG.2.2.14691.98080](https://doi.org/10.13140/RG.2.2.14691.98080)
Code: [github.com/mttmtt31/xg-cnn](https://github.com/mttmtt31/xg-cnn)

**Reported AUC:** 0.801 — and this is **validation AUC, not test AUC**, because
the code uses an 80/20 train/val split only, with no held-out test set.

### xG-CNN technical setup (extracted from code 2026-05-11)

| Aspect | xG-CNN | Source |
|---|---|---|
| Input channels | **2** | `create_heatmaps.py`: `shot_frame[0, ..]` = opponent locations (increment count); `shot_frame[1, ..]` = ball position (single activated cell) |
| Channel 0 semantics | Opponent (defender) player locations | "`shot_frame[0, discrete_x, discrete_y] += 1`" |
| Channel 1 semantics | Ball position (single cell) | "`shot_frame[1, ball_discrete_x, ball_discrete_y] = 1`" |
| Image grid | **30 × 40** | Cropped from initial 60 × 40 via `shot_frame[:, 30:, :]` |
| Pitch coverage | **Offensive half only** | Defensive half cropped out |
| Coordinate discretization | `min(math.floor(x / 2), 59)` then crop | Discrete cell binning, no Gaussian blob at render time |
| Gaussian smoothing at render | **Not applied in final version** (intended `sigma=(0, .75, .75)` is commented out in `create_heatmaps.py`) | Comment in code |
| Gaussian smoothing at training | Optional via `--gaussian-filter`; CLI help text labels the default value `1.25` as "variance", but the code passes it directly as `scipy.ndimage.gaussian_filter(..., sigma=1.25)` — so the effective sigma is 1.25 px on the 30×40 grid. The naming in the help text is misleading. | `main.py` (CLI parsing), `src/dataset.py` (`gaussian_filter(..., sigma=g_filter)` use site) |
| Dataset | StatsBomb open-data, **competitions filter: 2010+ excluding La Liga except 2015/2016** | `create_json.py` |
| Shot filter | **`shot_type == 'Open Play'` only** → penalties implicitly excluded | `create_json.py` |
| Sample size | (TODO confirm from paper; an earlier project note suggested ~56k, but the open-data subset under "2010+, not La Liga except 15/16, Open Play, with freeze frame" should be re-counted before paper submission) | Not in README |
| Train / Val / Test | **80% / 20% / 0%** | `train_val_split(dataset, train_size=0.8)`; **no test set** |
| Eval metric | Validation ROC-AUC, log loss, RMSE | `main.py` |
| Loss | `nn.BCELoss()` | `main.py` |
| Class weighting | **None** | No `pos_weight`, no class_weight |
| Optimizer | SGD default; AdamW/Adam optional | CLI flag `--optim` |
| Default training | LR 0.0005, batch 64, 50 epochs, dropout 0, weight_decay 0 | `main.py` defaults |
| CNN architecture | Loaded by `load_model(dropout=dropout)`; details not in repo README, full paper has the diagram | Inferred |

### Positioning constraints implied for our paper

These force four corrections in CLAIMS.md and PAPER_OUTLINE.md (some already
applied, others new):

1. **xG-CNN is penalty-EXCLUDED**, not penalty-inclusive as I had assumed.
   - Implication: our "legacy 64-cell (penalty-inclusive)" runs are *not* a
     valid xG-CNN-comparability baseline. They cannot be presented as
     "for direct comparability with the xG-CNN rendering convention".
   - Action: **frame the legacy 64-cell as** "Legacy 64-cell rendering
     (pixel-fixed sigma, penalty-inclusive) is retained only as the
     historical baseline that motivated the controlled re-render; it does
     not map onto xG-CNN's protocol".
   - The closer-to-xG-CNN-in-protocol comparison would be our **controlled
     64-cell (sigma_pu=2.0, penalty-excluded)**, despite still differing on
     channels (4 vs 2), grid (64×64 vs 30×40), pitch coverage (full vs
     half), and split (test vs no-test).

2. **xG-CNN's 0.801 is a validation AUC**, not test AUC.
   - Implication: claiming our test ROC-AUC of 0.79 "matches xG-CNN's 0.801"
     is doubly misleading: different metric and different protocol.
   - Action: Section 3 (Related Work) and Section 5.1 of the paper must
     refer to xG-CNN's "reported AUC ≈ 0.80 on a validation split" and not
     present a direct numeric comparison.

3. **xG-CNN uses no class weighting** under a ~10% positive rate.
   - Implication: their decision-threshold behavior and reliability
     calibration are likely very different from ours (we use
     `pos_weight ≈ 8.85`). Their high raw-Brier behavior could be similar
     to ours (~0.20 before Platt scaling).
   - Action: Methods section should note this design choice explicitly when
     contrasting with xG-CNN; do not assume readers will infer it.

4. **xG-CNN has no test set** — the 80/20 split is train/val only.
   - Implication: our reporting of test AUC + PR-AUC + Brier + F1 at
     val-best threshold on a held-out test is methodologically stronger,
     and worth flagging explicitly.
   - Action: Methods section emphasizes "we report test-split metrics with
     a held-out test set; this is stricter than the validation-AUC
     reporting in xG-CNN".

### Open follow-ups before submission

- [ ] Read the full ResearchGate PDF (DOI 10.13140/RG.2.2.14691.98080) to
      confirm: exact sample size, validation metrics other than AUC,
      architecture diagram, any discussion of class imbalance.
- [ ] Note whether xG-CNN reports per-seed variance (paper or code) — they
      probably don't, which is a methodological differentiator for our work.
- [ ] Decide whether to attempt a one-seed xG-CNN replication under our
      protocol — would be supplementary-repo only (no slot in the 9-page
      MLSA main paper), low priority.

---

## Skor-xG (Xu et al., CVPR 2025 W)

[TODO before paper draft v1]
- 3D skeleton + GATv2 attention, Tracab tracking data (closed)
- Cited only as motivation for "attention helps in football", different
  modality from ours

## Wagenaar et al. 2017

[TODO before paper draft v1]
- GoogLeNet + 3-layer CNN, Bundesliga 29 matches
- Cited as the first image-based xG attempt

## Anzer & Bauer 2021 (XGBoost tabular xG)

[TODO before paper draft v1]
- AUC 0.80-0.88 reported, tabular features
- Cited as the tabular state-of-the-art baseline

## SoccerMap / EPV (Fernández, Bornn, Cervone 2021)

[TODO before paper draft v1]
- Fully-CNN regression on tracking data
- Different problem (EPV vs xG), cited as related architecture

## SoccerTransformer (KU Leuven 2024)

[TODO before paper draft v1]
- Transformer on event sequences
- Different input modality, cited as "attention applied to football
  elsewhere"

---

## Bibliographic format checklist (will depend on final venue)

- xG-CNN entry uses ResearchGate DOI (not arXiv); if venue requires an arXiv
  preprint, need to check if the authors have one. (Not found in May 2026
  search; if still missing at submission time, ResearchGate DOI is the
  primary citation.)
