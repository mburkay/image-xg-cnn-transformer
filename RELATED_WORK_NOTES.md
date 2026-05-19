# Related Work Notes — Source-Backed

> **Status (release-repo readers):** these are *working notes* from the
> claims-freeze phase that fed into the polished prose in Section 1 of
> `paper/main.tex`. They are kept for transparency about how the
> positioning paragraph was constructed from primary sources, not as
> a standalone related-work survey.

Working notebook for the paper's Related Work positioning. Citations and direct
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
| Train / Val / Test | **80% / 20% / 0%** | `train_val_split(dataset, train_size=0.8)`; **no test set** |
| Eval metric | Validation ROC-AUC, log loss, RMSE | `main.py` |
| Loss | `nn.BCELoss()` | `main.py` |
| Class weighting | **None** | No `pos_weight`, no class_weight |
| Optimizer | SGD default; AdamW/Adam optional | CLI flag `--optim` |
| Default training | LR 0.0005, batch 64, 50 epochs, dropout 0, weight_decay 0 | `main.py` defaults |
| CNN architecture | Loaded by `load_model(dropout=dropout)`; details not in repo README, full paper has the diagram | Inferred |

### Positioning decisions implied for our paper

These shaped four positioning decisions in the final paper:

1. **xG-CNN is penalty-EXCLUDED**, not penalty-inclusive as initially assumed.
   - Implication: our "legacy 64-cell (penalty-inclusive)" runs are *not* a
     valid xG-CNN-comparability baseline. They cannot be presented as
     "for direct comparability with the xG-CNN rendering convention".
   - Decision: legacy 64-cell rendering (pixel-fixed sigma, penalty-inclusive)
     is retained only as the historical baseline that motivated the controlled
     re-render; it does not map onto xG-CNN's protocol.
   - The closer-to-xG-CNN-in-protocol comparison would be our **controlled
     64-cell (sigma_pu=2.0, penalty-excluded)**, despite still differing on
     channels (4 vs 2), grid (64×64 vs 30×40), pitch coverage (full vs
     half), and split (test vs no-test).

2. **xG-CNN's 0.801 is a validation AUC**, not test AUC.
   - Implication: claiming our test ROC-AUC of 0.79 "matches xG-CNN's 0.801"
     is doubly misleading: different metric and different protocol.
   - Decision: the Introduction's "Positioning" paragraph refers to xG-CNN's
     reported AUC ≈ 0.80 and explicitly declines a direct numeric comparison.

3. **xG-CNN uses no class weighting** under a ~10% positive rate.
   - Implication: their decision-threshold behavior and reliability
     calibration are likely very different from ours (we use
     `pos_weight ≈ 8.85`).
   - Decision: the Method section notes this design choice explicitly when
     contrasting with xG-CNN.

4. **xG-CNN has no test set** — the 80/20 split is train/val only.
   - Implication: our reporting of test AUC + PR-AUC + Brier + F1 at
     val-best threshold on a held-out test is methodologically stronger.
   - Decision: the Method section emphasises "test-split metrics with a
     held-out test set" as a stricter protocol than xG-CNN's validation-AUC
     reporting.

---

## Skor-xG (Xu et al., CVPRW 2025)

- 3D skeleton + GATv2 attention, Tracab tracking data (closed).
- Different modality from ours; cited only as motivation for "attention
  helps in football".

## Wagenaar et al. 2017

- GoogLeNet + 3-layer CNN, Bundesliga 29 matches.
- First CNN applied to **goal-scoring opportunity prediction from top-down
  position images, not StatsBomb-style xG**. The target is a binary
  opportunity label produced algorithmically from positional data, distinct
  from the per-shot scoring probability that xG (and xG-CNN, and our work)
  estimates.

## Anzer & Bauer 2021 (XGBoost tabular xG)

- Table 2 reports model AUC up to approximately 0.83 using synchronized
  positional and event data; cited as the tabular baseline.
- Spatial structure is encoded through engineered features (synchronized
  positional + event data), not from a raw freeze-frame image.

## SoccerMap / EPV (Fernández, Bornn, Cervone 2021)

- Fully-convolutional regression on tracking data; cited as a related
  architecture working on a different problem (EPV, not xG).
