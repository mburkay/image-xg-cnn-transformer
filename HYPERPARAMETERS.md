# Hyperparameter Table (Supplementary Material for MLSA 2026)

> **Status:** locked for MLSA 2026 submission (Springer LNCS, 9 content
> pages + unlimited references, CMT submission, in-person presentation
> on 2026-09-07 in Naples; paper deadline 2026-06-05, camera-ready
> 2026-07-10). The 9-page main paper includes a trimmed inline version
> of Format A (per-config defaults + exceptions list, ~¼ page). This
> full per-run / per-config document lives in the repository as
> supplementary material and is referenced from the paper by URL.
>
> Format A (per-config defaults + exceptions list) is the chosen format;
> Format B (per-run table, ~38 rows) is **not** used in the main paper
> but the underlying per-run JSONs are in `outputs/models/*/eval/`.

## A.1 Rendering

| Field | Value | Notes |
|---|---|---|
| Pitch coordinate frame | StatsBomb 120 × 80 | Not literal meters; 1 pitch unit ≈ 0.875 m on UEFA pitch |
| Channels | 4 | attackers, defenders, goalkeeper, shooter |
| Gaussian amplitude | 1.0 | clipped to `[0, 1]` |
| Gaussian sigma | `sigma_pitch_units × image_width / 120` (pixels) | Controlled runs: σ_pu = 2.0 |
| Attack direction normalization | Yes | Shots toward the left goal flipped horizontally + y-axis mirror |
| Penalty handling | Excluded (controlled) | Legacy 64-cell is penalty-inclusive. **Not** an xG-CNN match: xG-CNN itself filters `shot_type == 'Open Play'` (penalty-excluded). The legacy cell is retained as a historical baseline only. |
| Renderable filter | `freeze_frame` non-empty | Removes 1190/1361 penalties before any filter |
| Output dtype | `float16` memmap (.npy) or .npz | folder mode for ≥128×128 |

## A.2 Splits

| Field | Value |
|---|---|
| Split function | `split_by_match` in `src/xg_project/splitting.py` |
| Split random_state | 42 (fixed across all runs) |
| Train / Val / Test | 70% / 15% / 15% match-level |
| Stratification | Per-match goal count buckets, sklearn `train_test_split` with stratify |
| Goal rate | ~10.15% in all three splits |
| Final counts (controlled, no-penalty) | train 60,893 / val 12,642 / test 13,127 |
| Final counts (legacy, with-penalty) | train 60,906 / val 12,896 / test 13,031 |

## A.3 Architectures

| Model | Conv backbone | Tokens / pooling | Head | Params |
|---|---|---|---|---|
| CNN-XG | 4× ConvBlock (32, 64, 128, 256 ch), each = 2 × Conv3×3 + BN + ReLU + MaxPool2 | AdaptiveAvgPool2d((1,1)) | Linear(256, 128) + ReLU + Dropout(0.3) + Linear(128, 1) | 1.21 M |
| Hybrid CNN+Transformer-XG | Same conv backbone | T = (H/16)·(W/16) tokens of dim 256, learnable position embedding | 4 × TransformerEncoderLayer (8 heads, FFN=512, GELU, pre-norm) → LayerNorm → mean over tokens → Linear(256, 128) + ReLU + Dropout(0.2) + Linear(128, 1) | 3.32 M (64×64) / 3.33 M (128×128, +12k for position embed) |
| ViT-XG | None | 16×16 patch embedding → T tokens, embed_dim=128 | 4 × TransformerEncoderLayer (4 heads, FFN=256) → norm → mean → MLP | ~1.0 M |

## A.4 Loss and optimization

| Field | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` with `pos_weight = N_neg / N_pos` |
| `pos_weight` (controlled train) | ≈ 8.848 (60,893 train / 6,177 positives) |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| LR schedule | `ReduceLROnPlateau` on val_loss, factor=0.5, patience=5, min_lr=1e-7 |
| Early stopping | Patience 10 on val ROC-AUC (best checkpoint = highest val ROC-AUC) |

## A.5 Per-cell exceptions (training)

| Cell | Epochs cap | Batch size | Notes |
|---|---:|---:|---|
| CNN 64 (controlled + legacy) | 15 | 128 | Larger batch fits at 64×64 |
| CNN 96 | 15 | 64 | Batch reduced for memory at higher res |
| CNN 128 | 15 | 64 | Same as 96 |
| Hybrid 64 | 30 | 64 | Patience-10 early stop usually fires at epoch 12-18 |
| Hybrid 96 | 30 | 64 | Same |
| Hybrid 128 | 30 | 64 | Same |
| ViT-XG | (planned) | — | Not yet trained |

## A.6 Augmentation

| Field | Value |
|---|---|
| Top-bottom flip | p = 0.5 (only when `--augment` flag set; current headline runs do **not** use augmentation) |
| Additive Gaussian noise | σ = 0.015, p = 0.5 (same flag) |
| Left-right flip | **No** — attack direction is normalized; flipping would invert the half-pitch semantics |

**Important:** all headline runs in this paper were trained **without
augmentation**. The augmentation code path exists in the data pipeline but
was not used. Adding augmentation is future work.

## A.7 Seeds

| Field | Value |
|---|---|
| Data-split seed (`split_by_match` random_state) | 42 (fixed across all runs) |
| Training seeds (controlled headline cells) | {42, 1, 7} — 3 seeds per cell |
| Training seeds (5-seed extensions, supplementary repo) | {42, 1, 7, 13, 21} on CNN 64 legacy and Hybrid 128 controlled |
| Seed effect | torch + numpy global RNG for model init and mini-batch shuffling; data split is unaffected |

## A.8 Evaluation

| Field | Value |
|---|---|
| Test split selection | Single match-level test split (no cross-validation across folds; seed-level CV only) |
| Best checkpoint | Highest val ROC-AUC during training |
| Ranking metrics | ROC-AUC, PR-AUC (sklearn `roc_auc_score`, `average_precision_score`) |
| Threshold rules | Fixed 0.5, best-F1-on-val, Youden-J-on-val |
| Calibration | Platt (sklearn `LogisticRegression` on logit of val probabilities) and isotonic (sklearn `IsotonicRegression`); both fit on val, applied to test |
| Brier | `brier_score_loss` on raw, Platt-scaled, and isotonic-scaled probabilities |
| Reliability diagram bins | 10 |

## A.9 Compute

| Field | Value |
|---|---|
| Device | Apple MPS (Apple Silicon), no mixed precision |
| Per-run runtime (approximate, MPS) | CNN 64: ~7 min; CNN 96/128: ~25–35 min; Hybrid 64: ~9 min; Hybrid 96: ~25 min; Hybrid 128: ~60–75 min |
| Total experimental compute used in this work | ≈ 19–21 hours of MPS training across 38 checkpoints + ≈ 1 hour of rendering + figure / eval scripts |

## A.10 Software

| Field | Version |
|---|---|
| Python | 3.12 (Anaconda) |
| PyTorch | (record from `pip list` before submission) |
| numpy / pandas / scikit-learn | (record from `pip list` before submission) |
| StatsBomb open-data clone | Snapshot taken on (record date before submission) |

## A.11 Reproducibility checklist

- [ ] `data/raw/shots.pkl` reproducible via `scripts/01_download_shots.py`
- [ ] Rendered datasets reproducible via `scripts/02_build_images.py` with documented flags
- [ ] Each table in the paper's Results section reproducible by `scripts/04_evaluate.py` on the saved best checkpoint
- [ ] `requirements.txt` regenerated with exact versions before MLSA 2026 submission (deadline 5 June 2026)
- [ ] Repo archive on Zenodo (or equivalent) with DOI, by the MLSA camera-ready deadline (10 July 2026)

> Last updated: 2026-05-11. Format A (per-config defaults) is locked for the MLSA 2026 submission.
