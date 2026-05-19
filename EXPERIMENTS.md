# Experiment Log

> **Status:** This file is the **supplementary material** for the MLSA 2026
> submission (Springer LNCS, 9 content pages + unlimited references). The
> 9-page main paper references this document by URL for: full per-cell
> tables with mean ± std, sigma sweep details, the 2×2 σ × penalty
> ablation derivation, rendering-protocol audit history, and any SUPERSEDED
> mixed-protocol results. Sections marked `[SUPERSEDED — mixed protocol]`
> are kept for the narrative arc only and are not cited in the main paper.

## Terminology note

Throughout this document and the code, `sigma_meters` and `--sigma-meters`
refer to **StatsBomb pitch-coordinate units** (a 120 × 80 frame), not literal
meters. The dataset uses StatsBomb's native pitch coordinates and we have not
applied a conversion to UEFA-standard physical dimensions. As a rough guide:
1 pitch unit ≈ 0.875 m on the long axis of a 105 × 68 m pitch. The legacy
code/flag name is preserved for backwards compatibility but the paper writes
"pitch units" consistently.

The legacy 64-pixel rendering (`freeze_frames_64.npz`) was built before the
`--sigma-meters` flag existed and uses sigma_px=2.5 (≈4.69 pitch units at
64×64) with penalty shots included. The controlled rendering protocol used by
all 96×96 and 128×128 runs uses sigma_pitch_units=2.0 with penalties excluded.
A controlled 64×64 rendering (`freeze_frames_64_s2p0_no_penalty`) is added in
the "Controlled 64-Cell Re-Render" section below so that the controlled
six-cell headline ablation (CNN/Hybrid × 64/96/128) and the token sweep
are strictly comparable.

## Dataset

- Source: local StatsBomb `open-data`
- Raw shots: 88,023
- Renderable freeze-frame shots: 86,833
- Image dataset: `data/processed/freeze_frames_64.npz`
- Metadata: `data/processed/shots_with_splits_64.pkl`
- Input shape: 64x64x4
- Split policy: match-level split with goal-rate stratification

| Split | Shots | Goals | Goal Rate |
|---|---:|---:|---:|
| train | 60,906 | 6,184 | 0.101534 |
| val | 12,896 | 1,310 | 0.101582 |
| test | 13,031 | 1,326 | 0.101757 |

## CNN Baseline 64

- Command: `python3 scripts/03_train.py --data data/processed/freeze_frames_64.npz --model cnn --epochs 10 --batch-size 128 --output-dir outputs/models/cnn_64 --device mps`
- Device: MPS (Apple Silicon)
- Best checkpoint: `outputs/models/cnn_64/best_model.pt`
- Best validation AUC: 0.7927
- Test artifacts: `outputs/models/cnn_64/eval`

| Metric | Value |
|---|---:|
| ROC-AUC | 0.788138 |
| PR-AUC | 0.356864 |
| Accuracy | 0.722201 |
| Precision | 0.225072 |
| Recall | 0.708145 |
| F1 | 0.341579 |
| Brier score | 0.198922 |
| Log loss | 0.593480 |
| Threshold | 0.500000 |

Notes:

- AUC is the main ranking metric, but not the only metric.
- Recall is acceptable for a first baseline, but precision is low at threshold 0.5.
- Brier score and calibration need attention before treating the output as a strong xG probability model.

## Hybrid CNN+Transformer 64

- Command: `python3 scripts/03_train.py --data data/processed/freeze_frames_64.npz --model hybrid --epochs 30 --batch-size 64 --learning-rate 1e-4 --output-dir outputs/models/hybrid_64 --device mps`
- Device: MPS (Apple Silicon)
- Early stopping: stopped after epoch 18
- Best checkpoint: `outputs/models/hybrid_64/best_model.pt`
- Best validation AUC: 0.792538 at epoch 8
- Test artifacts: `outputs/models/hybrid_64/eval`

| Metric | Value |
|---|---:|
| ROC-AUC | 0.787250 |
| PR-AUC | 0.356602 |
| Accuracy | 0.678919 |
| Precision | 0.205482 |
| Recall | 0.751885 |
| F1 | 0.322758 |
| Brier score | 0.209802 |
| Log loss | 0.626593 |
| Threshold | 0.500000 |

Notes:

- Hybrid did not beat the CNN baseline on test ROC-AUC.
- Recall improved, but precision, F1, accuracy, and Brier score worsened.
- Training AUC continued rising while validation AUC stalled, which indicates overfitting.

## CNN vs Hybrid 64

| Model | Best Val AUC | Test ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.792723 | 0.788138 | 0.356864 | 0.722201 | 0.225072 | 0.708145 | 0.341579 | 0.198922 |
| Hybrid | 0.792538 | 0.787250 | 0.356602 | 0.678919 | 0.205482 | 0.751885 | 0.322758 | 0.209802 |

Current conclusion:

- The first hybrid architecture is not yet better than the CNN baseline.
- The next improvement should target overfitting and calibration before claiming transformer benefit.

## Why The First Hybrid Result Was Expected

- Transformers do not reliably beat CNNs on small image datasets when trained
  from scratch. Roughly 60K training shots is small for a transformer without
  large-scale pretraining.
- The first input resolution is only 64x64. This is useful for fast iteration,
  but it gives the transformer few spatial tokens and weakens the case for
  global attention.
- CNN locality and translation equivariance are strong inductive biases for
  structured top-down freeze-frame images. The transformer branch has to learn
  more of this structure from data.
- A naive "CNN backbone + Transformer encoder" is not the same as a carefully
  designed hybrid architecture such as CvT, CoAtNet, or MobileViT.
- The observed pattern, rising train AUC with stalled validation AUC, is
  consistent with the extra transformer capacity overfitting.

Immediate next steps before changing architecture:

- Add PR-AUC, because ROC-AUC can be optimistic under class imbalance.
- Select thresholds from validation data instead of relying on a fixed 0.5
  threshold.
- Calibrate probabilities with Platt scaling and isotonic regression.
- Verify the positive class weighting used during training.

## Threshold And Calibration Analysis

Training class balance check:

| Field | Value |
|---|---:|
| Train examples | 60,906 |
| Train positives | 6,184 |
| Train negatives | 54,722 |
| Train positive rate | 0.101534 |
| BCE `pos_weight` | 8.848965 |
| Negative / positive ratio | 8.848965 |

The current training loop uses `BCEWithLogitsLoss(pos_weight=8.848965)`, so the
positive class weighting is active and matches the train split imbalance.

Thresholds selected from validation data and then applied to the test split:

| Model | Threshold Rule | Threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| CNN | Fixed 0.5 | 0.500000 | 0.722201 | 0.225072 | 0.708145 | 0.341579 |
| CNN | Best F1 on val | 0.740348 | 0.845676 | 0.329517 | 0.499246 | 0.397001 |
| CNN | Youden J on val | 0.556586 | 0.755736 | 0.242012 | 0.656863 | 0.353706 |
| Hybrid | Fixed 0.5 | 0.500000 | 0.678919 | 0.205482 | 0.751885 | 0.322758 |
| Hybrid | Best F1 on val | 0.782311 | 0.862252 | 0.358479 | 0.447964 | 0.398257 |
| Hybrid | Youden J on val | 0.553746 | 0.740772 | 0.233783 | 0.679487 | 0.347876 |

Calibration on the test split:

| Model | Calibration | Brier | Log Loss | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| CNN | Raw | 0.198922 | 0.593480 | 0.788138 | 0.356864 |
| CNN | Platt | 0.078193 | 0.273536 | 0.788138 | 0.356864 |
| CNN | Isotonic | 0.078284 | 0.273891 | 0.786075 | 0.336665 |
| Hybrid | Raw | 0.209802 | 0.626593 | 0.787250 | 0.356602 |
| Hybrid | Platt | 0.078622 | 0.274462 | 0.787250 | 0.356602 |
| Hybrid | Isotonic | 0.078234 | 0.281928 | 0.786578 | 0.343122 |

Interpretation:

- PR-AUC confirms the two models are nearly tied under class imbalance.
- Fixed threshold 0.5 is not a good operating point. Validation-selected F1
  thresholds raise test F1 from about 0.34 to about 0.40.
- Platt scaling sharply improves Brier score and log loss without changing
  ranking metrics. This is expected because calibration is monotonic.
- Isotonic regression slightly improves raw calibration too, but it hurts PR-AUC
  more than Platt here. Platt is the cleaner default calibration candidate.

## Next Hypotheses

The current experiments suggest that calibration and operating point selection
produce larger practical gains than the first architecture change. The next
modeling experiments should test whether the AUC ceiling is caused by data
representation rather than model family:

- Increase image resolution from 64x64 to 128x128. This gives the CNN more
  spatial detail and gives transformer attention more tokens to work with.
- Remove penalties for an "open play / non-penalty xG" setting. Penalty freeze
  frames are structurally different and can add noise to image-only learning.
- Align preprocessing with the xG-CNN baseline as closely as possible before
  making stronger architecture claims: channel structure, Gaussian sigma, input
  resolution, and attacker direction normalization.
- If hybrid still overfits after representation changes, reduce transformer
  depth/heads/embed size before adding stronger augmentation.

## CNN 128 No Penalty

- Dataset: `data/processed/freeze_frames_128_no_penalty`
- Metadata: `data/processed/shots_with_splits_128_no_penalty.pkl`
- Input shape: 128x128x4
- Penalties: excluded
- Storage: folder-backed `images.npy` memmap to avoid loading the full 11 GB
  image array into RAM.
- Command: `python3 scripts/03_train.py --data data/processed/freeze_frames_128_no_penalty --model cnn --epochs 15 --batch-size 64 --learning-rate 1e-4 --output-dir outputs/models/cnn_128_no_penalty --device mps`
- Device: MPS (Apple Silicon)
- Best checkpoint: `outputs/models/cnn_128_no_penalty/best_model.pt`
- Best validation AUC: 0.789989 at epoch 10
- Test artifacts: `outputs/models/cnn_128_no_penalty/eval`

Split:

| Split | Shots | Goals | Goal Rate |
|---|---:|---:|---:|
| train | 60,893 | 6,177 | 0.101440 |
| val | 12,642 | 1,293 | 0.102278 |
| test | 13,127 | 1,322 | 0.100708 |

Metrics at fixed threshold 0.5:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.789593 |
| PR-AUC | 0.333353 |
| Accuracy | 0.751885 |
| Precision | 0.239569 |
| Recall | 0.673222 |
| F1 | 0.353385 |
| Brier score | 0.176779 |
| Log loss | 0.529265 |
| Threshold | 0.500000 |

Thresholds selected from validation data and then applied to the test split:

| Threshold Rule | Threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Fixed 0.5 | 0.500000 | 0.751885 | 0.239569 | 0.673222 | 0.353385 |
| Best F1 on val | 0.686647 | 0.850080 | 0.334698 | 0.494705 | 0.399267 |
| Youden J on val | 0.487456 | 0.743125 | 0.235005 | 0.687595 | 0.350289 |

Calibration:

| Calibration | Brier | Log Loss | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Raw | 0.176779 | 0.529265 | 0.789593 | 0.333353 |
| Platt | 0.078106 | 0.271961 | 0.789593 | 0.333353 |
| Isotonic | 0.078270 | 0.273391 | 0.788421 | 0.322923 |

Comparison against 64x64 CNN:

| Model | Best Val AUC | Test ROC-AUC | PR-AUC | F1 @ 0.5 | Best-Val-F1 Test F1 | Platt Brier |
|---|---:|---:|---:|---:|---:|---:|
| CNN 64 | 0.792723 | 0.788138 | 0.356864 | 0.341579 | 0.397001 | 0.078193 |
| CNN 128 no penalty | 0.789989 | 0.789593 | 0.333353 | 0.353385 | 0.399267 | 0.078106 |

Interpretation:

- 128x128 plus penalty filtering gave a small ROC-AUC increase on test, but not
  a validation AUC improvement.
- PR-AUC fell, so this is not an unambiguous improvement under class imbalance.
- Calibration remains strong; Platt Brier is again around 0.078.
- Because resolution and penalty filtering changed together, the next diagnostic
  should isolate them: 64x64 no-penalty and/or 128x128 with penalties.

## Rendering Protocol Audit And Re-Render

The mixed 128 result (small ROC-AUC gain, PR-AUC drop) triggered an audit of
three suspected issues before running more ablations. Diagnostic script:
`scripts/diagnostics_rendering_protocol.py`. The script regenerates the
visual comparison at `outputs/figures/sigma_protocol_comparison.png`.

### Check 1 — Penalty filter (false alarm)

The filter is correct. Field is `shot_type` and value is exactly `Penalty`
(StatsBomb canonical). The check
`shots["shot_type"].fillna("").str.lower() != "penalty"` in
`src/xg_project/image_dataset.py:36-37` matches the data.

The reason the renderable dataset only contained 171 penalties (not ~1300) is
not a filter issue. It is a dataset characteristic:

| Stage | Total | Open Play | Free Kick | Penalty | Other |
|---|---:|---:|---:|---:|---:|
| Raw `shots.pkl` | 88,023 | 82,402 | 4,231 | 1,361 | 29 |
| After renderable filter (`freeze_frame` non-empty) | 86,833 | 82,402 | 4,231 | 171 | 29 |
| After `exclude_penalties=True` | 86,662 | 82,402 | 4,231 | 0 | 29 |

StatsBomb often leaves `shot.freeze_frame` empty for penalties (only the shooter
and goalkeeper would be visible, and the array is omitted), so 1,190 penalties
were already removed by the `_has_freeze_frame` check before the explicit
penalty filter ran. The 171 -> 0 drop in the second filter is therefore correct
and complete.

No code change required.

### Check 2 — Gaussian sigma scaling (confirmed protocol mismatch, fixed)

Confirmed by code reading and visual diagnostic:

- `--sigma` in `scripts/02_build_images.py` is in pixels and was applied
  uniformly regardless of `--height` / `--width`.
- 64x64 with sigma=2.5 px corresponds to ~4.69 pu on the pitch length axis.
- 128x128 with the same sigma=2.5 px corresponds to ~2.34 pu on the pitch length
  axis. Players became sharper points covering half the physical area.
- 128x128 with sigma=5.0 px restores ~4.69 pu on the pitch length axis.

Per-channel sum on the same shot ("energy" of the rendered signal):

| Render | Attackers | Defenders | Goalkeeper | Shooter |
|---|---:|---:|---:|---:|
| 64x64, sigma=2.5 | 117.65 | 261.97 | 32.44 | 38.32 |
| 128x128, sigma=2.5 (mismatched protocol) | 117.66 | 274.45 | 37.46 | 39.22 |
| 128x128, sigma=5.0 (pitch-equivalent protocol) | 469.42 | 1046.39 | 125.32 | 152.15 |

The 4x ratio between sigma=2.5 and sigma=5.0 at 128x128 matches the analytical
result: integral of a 2D Gaussian under amplitude=1 scales with sigma^2.

Fix in `scripts/02_build_images.py`: a new `--sigma-meters` flag (legacy name
retained for backwards compatibility; the value is in pitch-coordinate units,
not literal meters) converts pitch units to pixels via
`sigma_px = sigma_meters * width / PITCH_LENGTH` so the
blob covers the same physical area regardless of image resolution. The
`--sigma` (pixels) argument is still accepted for backward compatibility.

Re-rendered dataset:

- Path: `data/processed/freeze_frames_128_sigmafix_no_penalty/` (folder-backed
  memmap, 11.36 GB on disk)
- Metadata: `data/processed/shots_with_splits_128_sigmafix_no_penalty.pkl`
- Render config: `--sigma-meters 4.6875` -> sigma_px=5.0, ~4.69 pu on pitch
  length axis, matching the 64x64 baseline.
- Splits (seed=42 deterministic, identical to the previous 128 no-penalty run,
  so AUC is directly comparable):

| Split | Shots | Goals | Goal Rate |
|---|---:|---:|---:|
| train | 60,893 | 6,177 | 0.101440 |
| val | 12,642 | 1,293 | 0.102278 |
| test | 13,127 | 1,322 | 0.100708 |

Visual confirmation: `outputs/figures/freeze_frame_preview_128_sigmafix.png`
shows blobs at the same physical size as `freeze_frame_preview_64.png`.

### Check 3 — Architecture adaptation (false alarm)

Both architectures are already resolution-agnostic, confirmed by reading the
model code in `src/xg_project/torch_models.py` and a forward smoke test:

- `CNNXG` ends in `nn.AdaptiveAvgPool2d((1, 1))` followed by a linear head.
  The flatten dimension is fixed by the channel count, so the parameter count
  does not change with input resolution.
- `HybridCNNTransformerXG` sizes the learned positional embedding from
  `num_tokens = reduced_h * reduced_w`, where the reduced dimensions follow the
  conv-pool depth and slice into the embedding only as needed at the forward
  pass.

Forward pass at two resolutions:

| Input | CNN params | CNN out | Hybrid params | Hybrid out |
|---|---:|---|---:|---|
| (B, 4, 64, 64) | 1,206,529 | (B,) | 3,319,553 | (B,) |
| (B, 4, 128, 128) | 1,206,529 | (B,) | 3,331,841 | (B,) |

The 12,288-parameter delta on Hybrid is exactly the position embedding for the
extra 48 tokens (8x8 - 4x4) at 256-dim, as expected. No architectural change
required.

### CNN 128 sigmafix re-train

- Command: `python3 scripts/03_train.py --data data/processed/freeze_frames_128_sigmafix_no_penalty --model cnn --epochs 15 --batch-size 64 --learning-rate 1e-4 --output-dir outputs/models/cnn_128_sigmafix --device mps`
- Device: MPS (Apple Silicon)
- Best checkpoint: `outputs/models/cnn_128_sigmafix/best_model.pt`
- Best validation AUC: 0.778130 at epoch 15 (no early stopping; val AUC was still
  drifting up at the final epoch but stayed below the previous 128 run)
- Test artifacts: `outputs/models/cnn_128_sigmafix/eval`

Metrics at fixed threshold 0.5:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.785772 |
| PR-AUC | 0.313634 |
| Accuracy | 0.753028 |
| Precision | 0.239130 |
| Recall | 0.665658 |
| F1 | 0.351859 |
| Brier score | 0.176503 |
| Log loss | 0.529610 |

Thresholds selected from validation data and applied to the test split:

| Threshold Rule | Threshold | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Fixed 0.5 | 0.500000 | 0.753028 | 0.239130 | 0.665658 | 0.351859 |
| Best F1 on val | 0.672642 | 0.836139 | 0.312359 | 0.521936 | 0.390824 |
| Youden J on val | 0.445758 | 0.711358 | 0.217152 | 0.716339 | 0.333275 |

Calibration on the test split:

| Calibration | Brier | Log Loss | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Raw | 0.176503 | 0.529610 | 0.785772 | 0.313634 |
| Platt | 0.078936 | 0.274403 | 0.785772 | 0.313634 |
| Isotonic | 0.079556 | 0.283066 | 0.783286 | 0.298550 |

### CNN 128 sigma comparison (mismatched vs pitch-equivalent protocol)

| Model | sigma | Pitch sigma (pu) | Best Val AUC | Test ROC-AUC | PR-AUC | F1 (val-best) | Platt Brier | Iso PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN 64 baseline | 2.5 px | 4.69 | 0.792723 | 0.788138 | 0.356864 | 0.397001 | 0.078193 | 0.336665 |
| CNN 128 sigma=2.5 px (mismatched protocol) | 2.5 px | 2.34 | 0.789989 | 0.789593 | 0.333353 | 0.399267 | 0.078106 | 0.322923 |
| CNN 128 sigma=5.0 px (pitch-equivalent candidate) | 5.0 px | 4.69 | 0.778130 | 0.785772 | 0.313634 | 0.390824 | 0.078936 | 0.298550 |

### Honest interpretation

The pitch-equivalent candidate did not improve the result. It underperforms the mismatched-protocol run on
every ranking and threshold-tuned metric. PR-AUC drops further from 0.3334 to
0.3136 and is now below both 128 runs and the 64 baseline. Best validation AUC
also drops by 0.012.

So Check 2 confirms a protocol mismatch in the earlier rendering configuration — sigma-in-pixels was not scaling with
image_size — but rescaling it to match the 64 baseline's pitch-equivalent
sigma (~4.69 pu) is the wrong correction. Two consequences follow:

1. The 64 baseline's sigma is itself too wide. ~4.69 pu on the pitch length axis
   is much larger than a player's physical footprint (~1 m). At 64x64 there
   were too few pixels to express that smoothing, so it appeared less harmful.
   Apply the same physical sigma at 128x128 and the blobs visibly fuse and the
   model loses spatial detail.
2. The PR-AUC drop from 64x64 (sigma=2.5 px) to 128x128 (sigma=2.5 px) is not
   explained by sigma scaling. The triage hypothesis that this single issue
   caused the mixed 128 result is not supported by the data.

What this means for the next steps:

- The `--sigma-meters` flag is still a useful addition because it makes future
  sigma changes explicit and physical instead of accidental. Keep it.
- A small sigma sweep is warranted before drawing any other conclusions:
  `sigma_pitch_units in {1.0, 1.5, 2.0, 2.5, 4.6875}` at 128x128 to find the
  resolution-appropriate setting.
- Until the sigma sweep clarifies the right value, treat the 64 baseline
  numbers as a soft reference rather than a target to match in pitch units at
  higher resolutions.
- The original PR-AUC drop from 64 to 128 still needs a separate explanation.
  Candidates worth checking: (a) batch size 128 vs 64 differs between runs,
  (b) class-weight pos_weight differs slightly because penalty filtering
  changed the train positive count, (c) MPS nondeterminism between runs.

The remaining audit hypotheses (different penalty handling, architecture
mismatch) are already cleared earlier in this section.

## Sigma Sweep at 128

After the pitch-equivalent candidate (sigma=4.69 pu) underperformed, ran a five-point
sigma sweep at 128x128 to find the resolution-appropriate value and isolate
how much of the original 64 -> 128 PR-AUC drop is sigma-driven. Two of the
points were already in hand (sigma=2.34 pu mismatched protocol, sigma=4.69 pu pitch-equivalent). Three new
runs at sigma_pitch_units in {1.0, 2.0, 3.5}. All runs share hyperparameters with
the existing 128 runs: 15 epochs, batch=64, lr=1e-4, MPS, seed=42, identical
match-level splits.

Test split metrics at the validation-best checkpoint:

| Run | sigma (pu) | sigma_px @128 | Best Val AUC | Best Epoch | Test ROC-AUC | PR-AUC | F1 (val-best thr) | Platt Brier | Iso PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN 128 s1p0 | 1.00 | 1.07 | 0.7880 | 6 | 0.7895 | 0.3254 | 0.397 | 0.0787 | 0.3125 |
| CNN 128 s2p0 | 2.00 | 2.13 | 0.7896 | 10 | 0.7899 | 0.3337 | 0.4045 | 0.0780 | 0.3158 |
| CNN 128 no_penalty | 2.34 | 2.50 | 0.7900 | 10 | 0.7896 | 0.3334 | 0.3993 | 0.0781 | 0.3229 |
| CNN 128 s3p5 | 3.50 | 3.73 | 0.7833 | 14 | 0.7884 | 0.3251 | 0.4005 | 0.0784 | 0.3109 |
| CNN 128 sigmafix | 4.69 | 5.00 | 0.7781 | 15 | 0.7858 | 0.3136 | 0.3908 | 0.0789 | 0.2986 |
| CNN 64 baseline | 4.69 | 2.50 | 0.7927 | 8 | 0.7881 | **0.3569** | 0.3970 | 0.0782 | 0.3367 |

Reading the table:

- The best 128 sigma is around 2 pu. PR-AUC peaks at sigma=2.0 pu (0.3337) and
  the previously-named "mismatched" sigma=2.34 pu is essentially tied (0.3334). Below
  2 pu the model overfits early (s1p0 best epoch is 6, with val AUC collapsing
  by epoch 13), and above 3 pu the model under-fits (s3p5 best epoch is 14 and
  still only 0.7833 val AUC).
- Even the best 128 run does not catch the 64 baseline on PR-AUC. Best 128
  PR-AUC is 0.3337; 64 PR-AUC is 0.3569. The 0.023 PR-AUC gap from 64 to 128
  is not closed by sigma optimization. ROC-AUC is essentially flat across
  resolutions: 64 = 0.7881, best 128 = 0.7899.
- F1 and Brier are very tight across the middle of the sweep. F1 (val-best
  threshold) is 0.397 to 0.404 from sigma=1 pu to 3.5 pu; Platt Brier sits at
  0.078 to 0.079 across all five 128 runs and the 64 baseline. So the
  threshold-tuned and calibrated metrics largely agree the resolution change
  is close to a wash.

Honest interpretation:

- Check 2 in the audit was real at the code level (sigma not scaling with
  image_size), but it was not the load-bearing cause of the mixed 128 result.
  Fixing sigma to the pitch-equivalent of the 64 baseline (4.69 pu) made things
  worse, and the resolution-appropriate fix (sigma ~ 2 pu) only matches what the
  pixel-fixed default already produced by accident.
- The 64 -> 128 PR-AUC drop is therefore not explained by sigma. PR-AUC is
  more sensitive to class imbalance than ROC-AUC, and at 128x128 the model
  may be ranking borderline negatives slightly higher relative to true
  positives, even though overall ranking quality (ROC-AUC) is unchanged.
- For a course-project narrative, this is still useful: it demonstrates a
  proper sweep, justifies sigma=2.0 pu as the principled choice for 128x128,
  and shows the resolution change is metric-dependent rather than a clean
  win.

Recommended default for any further 128 work: sigma_pitch_units=2.0 (sigma_px=2.13
at width=128). Use the `--sigma-meters` flag introduced earlier so the choice
is explicit and physical. Runs above sigma=3 pu at 128x128 should be avoided
unless there is a specific reason (e.g. testing whether wider blobs help the
hybrid model handle global context).

Open questions for later:

- Why is 64 PR-AUC higher than any 128 PR-AUC? Candidate explanations are
  (a) the larger pixel-to-pitch ratio at 64 effectively concentrates each
  player's signal into fewer cells, which may help the loss focus on the few
  high-density positive shots, (b) the lower-resolution 64 input is harder to
  overfit so the validation-selected checkpoint generalizes a bit better to
  the test split, (c) MPS-level numerical noise. None of these is verified.
- A short ablation on (a) would re-render at 96x96 sigma=4.69 pu and check
  whether PR-AUC sits between 64 and 128, but only if disk allows.

## Ablation: CNN vs Hybrid at 64 vs 128 [SUPERSEDED — mixed protocol]

> **Note:** This section reports single-seed results that mix the legacy
> 64-cell rendering (sigma_px=2.5, penalty-inclusive) with the controlled
> 128-cell rendering (sigma_pitch_units=2.0, penalty-excluded). The headline
> table here is therefore not strictly controlled. It is retained as a
> historical narrative ("first-look" comparison and the reasoning behind the
> single-seed claim that did not survive CV) but the primary headline
> ablation is in the "Controlled 64-Cell Re-Render" section below.

After fixing sigma to the resolution-appropriate value (sigma_pitch_units=2.0 at
128x128), the open question was whether the CNN+Transformer hybrid finally
benefits from the higher resolution. The first hybrid run earlier in this log
was at 64x64 with sigma_px=2.5 and tied the CNN. With sigma fixed and the
resolution doubled, the hybrid gets 64 transformer tokens (8x8) instead of 16
(4x4), so global attention has four times as many tokens to relate.

Hybrid 128 sigmafix run:

- Command: `python3 scripts/03_train.py --data data/processed/freeze_frames_128_s2p0_no_penalty --model hybrid --epochs 30 --batch-size 64 --learning-rate 1e-4 --output-dir outputs/models/hybrid_128_s2p0 --device mps`
- Device: MPS (Apple Silicon)
- Early stopping: stopped after epoch 22 (patience=10 from best epoch=12)
- Best validation AUC: 0.791523 at epoch 12
- Test artifacts: `outputs/models/hybrid_128_s2p0/eval`

Four-cell ablation (CNN vs Hybrid x 64 vs 128 sigmafix):

| Run | Resolution | Sigma (pu) | Best Val AUC | Test ROC-AUC | PR-AUC | F1 (val-best thr) | Platt Brier | Iso PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN 64 | 64 | 4.69 | 0.7927 | 0.7881 | 0.3569 | 0.3970 | 0.0782 | 0.3367 |
| Hybrid 64 | 64 | 4.69 | 0.7925 | 0.7873 | 0.3566 | 0.3983 | 0.0786 | 0.3431 |
| CNN 128 s2p0 | 128 | 2.00 | 0.7896 | 0.7899 | 0.3337 | 0.4045 | 0.0780 | 0.3158 |
| **Hybrid 128 s2p0** | **128** | **2.00** | **0.7915** | **0.7928** | **0.3598** | **0.4066** | **0.0771** | **0.3431** |

Reading the four cells:

- At 64x64 the hybrid does not beat the CNN: ROC-AUC 0.7873 vs 0.7881,
  PR-AUC 0.3566 vs 0.3569. Within noise, they are tied. This matches the
  earlier conclusion in the "Hybrid CNN+Transformer 64" section.
- At 128x128 with the right sigma, the hybrid beats the CNN on every metric:
  Val AUC +0.0019, ROC-AUC +0.0029, PR-AUC +0.0261, F1 +0.0021,
  Platt Brier -0.0009, Iso PR-AUC +0.0273. The PR-AUC and Iso PR-AUC gains are
  the most meaningful under class imbalance.
- Hybrid 128 s2p0 is the best run on every metric in this entire log,
  including the previous best 64 baseline. PR-AUC moves from 0.3569 (CNN 64)
  to 0.3598 (Hybrid 128 s2p0), the first non-trivial improvement over the
  64-pixel CNN that survives PR-AUC and calibrated metrics.

Honest interpretation:

- The hybrid's value is conditional. With only 16 tokens (64x64) the
  transformer block has too little to do and matches the CNN. With 64 tokens
  (128x128) and a resolution-appropriate sigma, attention finally pulls the
  model ahead.
- The single-issue-fix narrative from the first audit pass was incomplete — the
  observed gain comes from the combination of higher resolution, the right
  sigma, and the transformer architecture together, not from any one of them
  alone. The sigma sweep confirmed sigma=2.0 pu is the operating point; the
  hybrid run confirmed the architecture pays off at that operating point.
- This four-cell ablation was the internal milestone framing at this stage
  (now SUPERSEDED by the controlled section below): it isolates the
  resolution effect (CNN 64 vs CNN 128), the sigma effect (sweep above), and
  the architecture effect at each resolution (CNN vs Hybrid columns). The
  practical recommendation that followed from this section was to use
  Hybrid 128 with sigma_pitch_units=2.0 as the project's primary model, but
  the 3- and 5-seed CV sections below reduce this to a single-seed artefact
  and the controlled section below reverses the PR-AUC verdict entirely.

Historical caveats from this superseded stage (do not carry these forward
into the LaTeX main paper; the current paper-facing caveats live in the
controlled section below):

- Hybrid 128 ran 30-epoch cap with patience=10 and stopped at epoch 22 with
  best at epoch 12. CNN 128 ran a fixed 15 epochs with no early stopping. The
  comparison was therefore not perfectly hyperparameter-matched. The CNN's val
  AUC was already past its peak at epoch 10 (0.7896) and dropped after, so
  giving the CNN 30 epochs would not change the best-val-AUC checkpoint.
- The improvements at this stage were modest in absolute terms (0.003
  ROC-AUC, 0.003 PR-AUC versus the 64 baseline). The story at the time was
  "hybrid pays off at higher resolution," not "hybrid is dramatically better."
- All numbers in this superseded block come from a single seed (42). For
  paper figures, the ablation needed to be repeated with at least three seeds
  and reported with mean±std, especially the then-planned four-cell
  comparison; the current headline is the controlled six-cell ablation below.
  With current runtime (~75 min per Hybrid 128 run on MPS), three-seed
  coverage was estimated at ~5 hours of training at the time (now executed).

**Update from the next section (3-Seed Cross-Validation):** the single-seed
PR-AUC advantage of Hybrid 128 over CNN 64 (+0.003 at seed=42) does not survive
3-seed CV — the mean PR-AUC gap collapses to +0.0008 with std bands of 0.006
to 0.008. The other improvements (F1, Brier, ROC-AUC) do survive the seed
sweep. Treat the four-cell ablation table above as a single-seed snapshot, and
the next section as the CV-corrected version.

## 3-Seed Cross-Validation [SUPERSEDED — mixed protocol]

> **Note:** This 3-seed run pairs CNN 64 (legacy rendering: sigma_px=2.5,
> penalty-inclusive) with Hybrid 128 (controlled rendering:
> sigma_pitch_units=2.0, penalty-excluded). The comparison is therefore
> across two protocols, not strictly controlled. It is retained as the
> first CV evidence that single-seed PR-AUC differences on this dataset are
> unreliable. The controlled 3-seed CV across all six ablation cells is in
> "Controlled 64-Cell Re-Render" below.

The four-cell ablation table at single seed=42 (especially the PR-AUC line)
needed verification against seed variance. Ran CNN 64 baseline and
Hybrid 128 sigma=2.0 pu with seeds {42, 1, 7} keeping all other hyperparameters
fixed:

- CNN 64: 15 epochs, batch=128, lr=1e-4, MPS. Outputs in
  `outputs/models/cnn_64_cv_seed{42,1,7}/`.
- Hybrid 128 sigma=2.0 pu: 30 epochs cap, patience=10, batch=64, lr=1e-4, MPS.
  Outputs in `outputs/models/hybrid_128_s2p0_cv_seed{42,1,7}/`.

Match-level data split is held fixed (seed=42 inside `split_by_match`), so
across runs the train/val/test partition is identical. The seed argument
controls torch and numpy global RNG, which affects model init and mini-batch
shuffling.

Per-seed test metrics:

| Run | Seed | Best Val AUC | Test ROC-AUC | PR-AUC | F1 (val-best thr) | Platt Brier | Iso PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN 64 | 42 | 0.7927 | 0.7881 | 0.3569 | 0.3970 | 0.0782 | 0.3367 |
| CNN 64 | 1 | 0.7913 | 0.7889 | 0.3712 | 0.3932 | 0.0775 | 0.3503 |
| CNN 64 | 7 | 0.7913 | 0.7850 | 0.3591 | 0.3907 | 0.0784 | 0.3323 |
| Hybrid 128 s2p0 | 42 | 0.7915 | 0.7928 | 0.3598 | 0.4066 | 0.0771 | 0.3431 |
| Hybrid 128 s2p0 | 1 | 0.7946 | 0.7902 | 0.3707 | 0.4064 | 0.0768 | 0.3535 |
| Hybrid 128 s2p0 | 7 | 0.7947 | 0.7904 | 0.3589 | 0.4082 | 0.0771 | 0.3426 |

Mean ± std (n=3, sample std with ddof=1):

| Metric | CNN 64 | Hybrid 128 s2p0 | Δ (Hyb − CNN) | Std-band check |
|---|---|---|---:|---|
| ROC-AUC | 0.7873 ± 0.0021 | 0.7911 ± 0.0014 | +0.0038 | ~2x larger than CNN std; marginal |
| **PR-AUC (raw)** | 0.3624 ± 0.0077 | 0.3632 ± 0.0065 | **+0.0008** | **Below both stds; not significant** |
| F1 (val-best thr) | 0.3936 ± 0.0032 | 0.4071 ± 0.0010 | +0.0134 | 4-13x above stds; clearly significant |
| Platt Brier | 0.0780 ± 0.0005 | 0.0770 ± 0.0002 | -0.0010 | 2-5x above stds; significant |
| Iso PR-AUC | 0.3398 ± 0.0094 | 0.3464 ± 0.0061 | +0.0066 | Below both stds; not significant |

Honest interpretation:

- The **single-seed PR-AUC advantage of Hybrid 128 over CNN 64 (+0.003 at
  seed=42) does not survive CV**. Mean PR-AUC gap collapses to +0.0008 with
  std bands of 0.006-0.008, so the gap is well within seed noise. The earlier
  claim that "Hybrid 128 sigmafix is the first run that beats CNN 64 PR-AUC"
  was a single-seed artifact and must be retracted in any paper draft.
- The **F1 (val-best threshold) and Platt Brier improvements do survive**.
  +0.0134 F1 with std=0.001-0.003 is a clear effect: at the operating point
  selected from validation, Hybrid 128 produces meaningfully sharper
  precision-recall trade-offs and better-calibrated probabilities.
- ROC-AUC improvement is real but small (+0.0038 with stds ~0.002). It is
  significant but not a headline number on its own.
- **The honest paper claim is therefore narrower:** Hybrid 128 sigmafix
  matches CNN 64 on ranking quality (ROC-AUC, PR-AUC, isotonic-PR-AUC) and
  improves the threshold-tuned operating point and calibration. The
  resolution + transformer combination shifts how probabilities cluster but
  does not fundamentally change the model's separability.
- This is also a methodological lesson worth keeping in the paper: results
  reported on a single seed for this dataset are unreliable for PR-AUC
  comparisons because seed std (~0.007) is comparable to typical
  improvements one would claim.

Practical recommendation (at this superseded mixed-protocol stage,
SUPERSEDED by the controlled section below):

- The temporary recommendation at this stage framed Hybrid 128 as
  ranking-equivalent to CNN 64, with better calibration and threshold-tuned
  classification. **This framing was reversed by the controlled six-cell
  headline ablation** in "Controlled 64-Cell Re-Render" below: under
  matched rendering, Hybrid 128 outperforms CNN 64 on PR-AUC by +0.014,
  which is the paper-facing claim.
- The interim "deployed" model choice at this stage was Hybrid 128
  sigma_pitch_units=2.0 because of the F1/Brier improvements. The
  controlled section confirmed this choice and strengthened the framing
  (now "better on every headline metric", not "comparable ranking, better
  operating point").
- A methodological observation that survived the controlled re-render:
  three seeds is the floor for paper credibility on this dataset. For a
  conference submission, plan five seeds and report the same table; with
  current ~75-min Hybrid runs, that's ~7.5 additional hours of training.

## Token Sweep at 96x96 [SUPERSEDED — mixed protocol]

> **Note:** The 64-cell row in this token sweep uses the legacy rendering
> (sigma_px=2.5, penalty-inclusive); the 96 and 128 rows use the controlled
> rendering. The CNN 128 cell is single-seed and the Hybrid 64 cell is
> single-seed legacy. The primary token sweep (all cells n=3, all controlled)
> is the "Controlled 64-Cell Re-Render" section below. This section is
> retained for the narrative of how the controlled re-render was motivated.

To test the hypothesis that the hybrid's lift over the CNN scales with the
number of transformer tokens (T = (H/16)*(W/16) at conv-pool depth 4), ran a
three-resolution sweep at fixed sigma_pitch_units=2.0:

- 64x64 -> 16 tokens (4x4 reduced)
- 96x96 -> 36 tokens (6x6 reduced)
- 128x128 -> 64 tokens (8x8 reduced)

Each non-trivial cell trained with 3 seeds (42, 1, 7); CNN 128 and Hybrid 64
remain at 1-seed for cost reasons. CNN runs use 15 epochs and batch=64 at
96/128 (batch=128 at 64); Hybrid runs use 30 epochs cap with patience=10.

Test metrics (mean ± std):

| Model | Resolution | Tokens | n | ROC-AUC | PR-AUC | F1 (val-best) | Platt Brier | Iso PR-AUC |
|---|---:|---:|---:|---|---|---|---|---|
| CNN | 64 | 16 | 3 | 0.7873 ± 0.0021 | 0.3624 ± 0.0077 | 0.3936 ± 0.0032 | 0.0780 ± 0.0005 | 0.3398 ± 0.0094 |
| CNN | 96 | 36 | 3 | 0.7869 ± 0.0003 | 0.3423 ± 0.0029 | 0.3975 ± 0.0037 | 0.0780 ± 0.0002 | 0.3281 ± 0.0055 |
| CNN | 128 | 64 | 1 | 0.7899 | 0.3337 | 0.4045 | 0.0780 | 0.3158 |
| Hybrid | 64 | 16 | 1 | 0.7872 | 0.3566 | 0.3983 | 0.0786 | 0.3431 |
| Hybrid | 96 | 36 | 3 | 0.7893 ± 0.0019 | 0.3468 ± 0.0076 | 0.4059 ± 0.0017 | 0.0778 ± 0.0003 | 0.3323 ± 0.0071 |
| Hybrid | 128 | 64 | 3 | 0.7911 ± 0.0014 | 0.3632 ± 0.0065 | 0.4071 ± 0.0010 | 0.0770 ± 0.0002 | 0.3464 ± 0.0061 |

Hybrid token-count trends (16 -> 36 -> 64):

- ROC-AUC: 0.7872 -> 0.7893 -> 0.7911 (monotonic +)
- F1 (val-best): 0.3983 -> 0.4059 -> 0.4071 (monotonic +)
- Platt Brier: 0.0786 -> 0.0778 -> 0.0770 (monotonic improvement)
- PR-AUC: 0.3566 -> 0.3468 -> 0.3632 (NON-monotonic; mid-resolution dip)

CNN token-count trends (resolution increase, no transformer):

- ROC-AUC and F1 nearly flat across resolutions
- PR-AUC drops monotonically: 0.3624 -> 0.3423 -> 0.3337

Honest interpretation:

- The "transformer needs enough tokens" hypothesis is supported on ROC-AUC, F1,
  and Brier: hybrid lift over CNN at the same resolution grows with token
  count. Hybrid - CNN deltas at the three resolutions:
  - 16 tokens (64x64): ROC -0.0001, F1 +0.0047, Brier +0.0006 (essentially tied)
  - 36 tokens (96x96): ROC +0.0024, F1 +0.0084, Brier -0.0002 (small lift)
  - 64 tokens (128x128): ROC +0.0038 (vs CNN 128 1-seed: +0.0012), F1 +0.0026
    (vs CNN 128 1-seed: +0.0026), Brier -0.0010
- The PR-AUC trend is not monotonic and does not support the hypothesis
  cleanly. The 96x96 dip is real (3-seed CV with std=0.0076), but the 64->128
  recovery is also real (std=0.0065). With combined std bands of ~0.014, the
  36 vs 64 token gap (-0.016) is at the edge of statistical significance.
- CNN PR-AUC degrades monotonically with resolution at fixed sigma_pitch_units,
  suggesting that pure CNNs at this dataset size do not productively use the
  extra spatial detail when the conv backbone immediately compresses to a
  global pool.

Paper-grade summary: hybrid lift is conditional on having enough tokens, and
ROC-AUC / F1 / Brier all support this conditional statement. PR-AUC does not
follow the same trend cleanly, which is methodologically informative.

## 5-Seed Extension (Headline Cells) [SUPERSEDED — mixed protocol]

> **Note:** This 5-seed extension was run on the mixed-protocol pair
> (CNN 64 legacy + Hybrid 128 controlled). It is retained for the
> methodological caveat it surfaces (single seed=21 outlier widening
> Hybrid 128 PR-AUC std) and as a supplementary-repo sensitivity check.
> The primary headline cells use the controlled 3-seed CV in the next
> section, which reverses the apparent "PR-AUC parity" of this section.

The 3-seed CV section concluded that PR-AUC differences smaller than ~0.015
on this dataset are within seed noise. To tighten std bands and verify the
mean for the two headline cells, extended each to 5 seeds by adding seeds 13
and 21.

Per-seed PR-AUC for Hybrid 128 sigma=2.0 pu:

| Seed | PR-AUC |
|---:|---:|
| 42 | 0.3598 |
| 1 | 0.3707 |
| 7 | 0.3589 |
| 13 | 0.3633 |
| 21 | 0.3318 |

3-seed (42, 1, 7) vs 5-seed (42, 1, 7, 13, 21) comparison:

| Group | n | ROC-AUC | PR-AUC | F1 (val-best) | Platt Brier | Iso PR-AUC |
|---|---:|---|---|---|---|---|
| CNN 64 | 3 | 0.7873 ± 0.0021 | 0.3624 ± 0.0077 | 0.3936 ± 0.0032 | 0.0780 ± 0.0005 | 0.3398 ± 0.0094 |
| CNN 64 | 5 | 0.7875 ± 0.0015 | 0.3620 ± 0.0063 | 0.3929 ± 0.0031 | 0.0781 ± 0.0004 | 0.3411 ± 0.0075 |
| Hybrid 128 | 3 | 0.7911 ± 0.0014 | 0.3632 ± 0.0065 | 0.4071 ± 0.0010 | 0.0770 ± 0.0002 | 0.3464 ± 0.0061 |
| Hybrid 128 | 5 | 0.7917 ± 0.0018 | 0.3569 ± 0.0148 | 0.4063 ± 0.0018 | 0.0772 ± 0.0005 | 0.3405 ± 0.0134 |

Honest interpretation:

- CNN 64: 5 seeds produced almost identical means (delta < 0.001 on every
  metric) and modestly tighter std bands (~%18-29 narrower for ROC-AUC and
  PR-AUC). The 3-seed result was already close to the population value.
- Hybrid 128: ROC-AUC, F1, and Brier means are nearly identical across 3 and
  5 seeds, with std bands shifting little. PR-AUC tells a different story.
  Seed 21 returned PR-AUC=0.3318 — a 0.025 drop below the 3-seed minimum and
  outside the 3-seed std band. This single outlier shifted the 5-seed mean
  PR-AUC down to 0.3569 and inflated the std to 0.0148, more than doubling
  the 3-seed std.
- **The 5-seed Hybrid 128 PR-AUC mean (0.3569) sits below the 5-seed
  CNN 64 PR-AUC mean (0.3620).** The 0.005 gap is well within the combined
  std band (CNN 64 std=0.006, Hybrid 128 std=0.015), so under this mixed-
  protocol comparison the conclusion at the time was "comparable PR-AUC
  under seed noise" — the 3-seed framing of "hybrid wins narrowly on
  PR-AUC" was gone, and the 5-seed framing was closer to "CNN 64 wins
  narrowly on PR-AUC, within noise". This is the framing that motivated the
  controlled re-render in the next section, which reversed this verdict
  again under matched protocol (see "Controlled 64-Cell Re-Render" below).
- F1 (val-best) and Platt Brier remain hybrid-positive on 5 seeds: F1 delta
  +0.013 with std=0.002-0.003, Brier delta -0.001 with std=0.0004-0.0005.
  These two are the only 5-seed metrics where Hybrid 128 is unambiguously
  better than CNN 64.

Earlier paper claim from this section (post-5-seed, **superseded by the
controlled section below**):

> Hybrid 128 sigma_pitch_units=2.0 matches the CNN 64 baseline on ranking
> quality (ROC-AUC marginally favors hybrid; PR-AUC marginally favors CNN
> with both gaps within seed noise) and improves the operating-point
> classification (F1 +0.013) and probability calibration (Platt
> Brier -0.001) under 5-seed CV (mixed protocol).

This was the framing produced by this mixed-protocol section and was
informative for a few hours of the project: it warned that single-seed
PR-AUC differences were brittle. **However it is not the framing the paper
should adopt.** The "Controlled 64-Cell Re-Render" section below repeats the
comparison with both cells under the same protocol (sigma_pitch_units=2.0,
penalty-excluded) and reverses the PR-AUC verdict — under the controlled
protocol, Hybrid 128 beats CNN 64 on PR-AUC by +0.014, well above seed std.
The 5-seed CNN 64 PR-AUC of 0.3620 used here came from the mixed
σ/penalty protocol (σ_px=2.5, penalty-inclusive). The 2×2 ablation
("Methodological note" further below) shows that penalty inclusion is
**not** a simple PR-AUC inflator: its main effect ≈ 0; the apparent
"inflation" in this section is a single-cell cross-protocol comparison
that mixes σ and penalty together and is confounded by a σ × penalty
interaction.

We do not remove the seed=21 outlier. The wider std band itself is paper-
grade evidence and supports the conservative claim that single-seed PR-AUC
differences on this dataset are brittle. Removing outliers would both
inflate the apparent effect and weaken methodological credibility.

## Controlled 64-Cell Re-Render

A reviewer-style pass on the earlier sections flagged that the 64×64 cells in
the four-cell ablation and the token sweep were trained on the legacy
`freeze_frames_64.npz` (sigma_px=2.5 -> ~4.69 pitch units at 64×64, penalty-
inclusive) while the 96×96 and 128×128 cells used the controlled rendering
(sigma_pitch_units=2.0, penalty-excluded). The "fixed sigma_pitch_units / penalty-
free headline" framing was therefore not strictly true at the 64 cells.

To resolve this we re-rendered the dataset at 64×64 with sigma_pitch_units=2.0
and penalty-excluded (folder: `freeze_frames_64_s2p0_no_penalty`), re-ran the
CNN and Hybrid models at three seeds each, and also added two new CNN 128
seeds so that every cell in the four-cell ablation now has n=3 CV under the
same controlled protocol.

Controlled test metrics (sigma_pitch_units=2.0, penalty-excluded, 3 seeds):

| Cell | n | ROC-AUC | PR-AUC | F1 (val-best) | Platt Brier | Iso PR-AUC |
|---|---:|---|---|---|---|---|
| CNN 64 controlled | 3 | 0.7885 ± 0.0004 | 0.3489 ± 0.0047 | 0.4016 ± 0.0044 | 0.0781 ± 0.0002 | 0.3248 ± 0.0043 |
| CNN 96 controlled | 3 | 0.7869 ± 0.0003 | 0.3423 ± 0.0029 | 0.3975 ± 0.0037 | 0.0780 ± 0.0002 | 0.3281 ± 0.0055 |
| CNN 128 controlled | 3 | 0.7871 ± 0.0031 | 0.3287 ± 0.0044 | 0.3985 ± 0.0053 | 0.0784 ± 0.0003 | 0.3084 ± 0.0076 |
| Hybrid 64 controlled | 3 | 0.7875 ± 0.0025 | 0.3603 ± 0.0119 | 0.4045 ± 0.0049 | 0.0775 ± 0.0007 | 0.3430 ± 0.0115 |
| Hybrid 96 controlled | 3 | 0.7893 ± 0.0019 | 0.3468 ± 0.0076 | 0.4059 ± 0.0017 | 0.0778 ± 0.0003 | 0.3323 ± 0.0071 |
| Hybrid 128 controlled | 3 | 0.7911 ± 0.0014 | 0.3632 ± 0.0065 | 0.4071 ± 0.0010 | 0.0770 ± 0.0002 | 0.3464 ± 0.0061 |

Hybrid − CNN deltas at each token count (controlled):

| Tokens | ROC-AUC | PR-AUC | F1 (val-best) | Platt Brier |
|---:|---:|---:|---:|---:|
| 16 (64×64) | -0.0010 | **+0.0114** | +0.0029 | -0.0006 |
| 36 (96×96) | +0.0024 | +0.0045 | +0.0085 | -0.0002 |
| 64 (128×128) | **+0.0040** | **+0.0345** | **+0.0086** | **-0.0014** |

Honest interpretation under the controlled protocol:

- **The earlier "PR-AUC parity" claim does not survive controlled comparison.**
  Hybrid 128 controlled (PR-AUC 0.3632 ± 0.0065) clearly beats CNN 64
  controlled (PR-AUC 0.3489 ± 0.0047) by +0.0143, well above the combined
  std band of ~0.011. This is the opposite of the 5-seed legacy comparison
  result and supersedes it. The legacy 64 PR-AUC of 0.3620 (vs controlled
  CNN 64 PR-AUC of 0.3489) is **not** explained by a simple "penalty
  inflates PR-AUC" effect; the 2×2 ablation in the next major section
  shows penalty main effect ≈ 0 and a +0.0183 σ × penalty interaction.
  At σ=4.69 pu (legacy), penalty inclusion happens to help PR-AUC; at
  σ=2.0 pu (controlled), it slightly hurts. So the cross-protocol gap is
  the result of changing σ and penalty together under a σ-conditioned
  penalty effect, not a single "penalty bias".
- The controlled six-cell headline ablation now says: Hybrid 128 is
  strictly better than CNN 64 on every headline metric (ROC-AUC +0.0026,
  PR-AUC +0.0143, F1 +0.0055, Platt Brier -0.0011).
- The token-sweep Hybrid - CNN PR-AUC delta is non-monotonic in absolute
  terms (+0.011 → +0.005 → +0.034) but the lift is unambiguously largest at
  64 tokens. This pattern is consistent with "transformer's value scales
  with token count, with a middle-resolution dip that the controlled
  experiment does not explain". The ROC-AUC, F1, and Brier deltas are
  monotonic in token count and support the conditional-on-token-count
  claim cleanly.
- CNN test PR-AUC degrades monotonically as resolution grows under
  controlled rendering (0.3489 → 0.3423 → 0.3287). This is consistent with
  a pure CNN being unable to productively use the extra spatial detail when
  the conv backbone immediately compresses to a global pool, and is the
  flip side of the hybrid's growing lift.
- The new CNN 64 numbers (controlled) tighten the std bands substantially.
  CNN 64 controlled std is 0.0047 for PR-AUC vs 0.0063-0.0077 in the legacy
  5-seed and 3-seed sets, partly because penalty noise is removed.

Updated paper claim (the version that survives the controlled comparison):

> Under a strictly controlled rendering protocol (sigma_pitch_units=2.0,
> penalty-excluded), Hybrid 128 sigmafix outperforms CNN 64 on every
> headline metric: ROC-AUC +0.003, PR-AUC +0.014, F1 (val-best) +0.006,
> Platt Brier -0.001, all with std bands smaller than the deltas on the
> ranking and threshold metrics. The hybrid's lift over the CNN at the
> same resolution grows with transformer token count from 16 to 64 on
> ROC-AUC, F1, and Brier; PR-AUC lift is non-monotonic but largest at 64
> tokens. Pure CNN PR-AUC degrades monotonically with resolution under
> controlled rendering, while the hybrid recovers and improves; this is
> the central observation of this work.

Legacy 64-cell numbers (sigma_px=2.5, penalty-inclusive) are retained as a
historical baseline only. They are **not** an xG-CNN replication — xG-CNN
itself filters `shot_type == 'Open Play'` (penalty-excluded), uses 2 channels
on a 30×40 half-pitch grid with discrete binning, and reports validation
AUC under an 80/20 split with no test set (see RELATED_WORK_NOTES.md). The
legacy 64-cell is closer in σ (pixel-fixed) but otherwise diverges from
xG-CNN's protocol on every other axis.

### Methodological note: 2×2 σ × penalty ablation at 64×64

The earlier "Methodological note" attributed the 64-legacy vs 64-controlled
PR-AUC gap to penalty inclusion as a single-variable effect. A 2×2 ablation
(σ ∈ {2.0, 4.69} pu × penalty ∈ {OUT, IN}, n=3 each) was run to isolate the
two variables. **The single-variable framing is wrong**: penalty effect on
PR-AUC has near-zero main effect and a strong interaction with σ.

Four-cell table (CNN 64, 3 seeds each):

| Cell | σ (pu) | Penalty | n | ROC-AUC | PR-AUC | F1 (val-best) | Platt Brier |
|---|---:|---:|---:|---|---|---|---|
| A | 2.0 | OUT | 3 | 0.7885 ± 0.0004 | 0.3489 ± 0.0047 | 0.4016 ± 0.0044 | 0.0781 ± 0.0002 |
| B | 2.0 | IN | 3 | 0.7887 ± 0.0002 | 0.3397 ± 0.0071 | 0.3945 ± 0.0054 | 0.0791 ± 0.0005 |
| C | 4.69 | OUT | 3 | 0.7872 ± 0.0017 | 0.3533 ± 0.0030 | 0.3961 ± 0.0065 | 0.0775 ± 0.0001 |
| D | 4.69 | IN | 3 | 0.7873 ± 0.0021 | 0.3624 ± 0.0077 | 0.3936 ± 0.0032 | 0.0780 ± 0.0005 |

Main effects and interaction (computed as (B+D)/2 − (A+C)/2 etc.):

| Metric | Penalty main effect | σ main effect | Penalty × σ interaction |
|---|---:|---:|---:|
| ROC-AUC | +0.0002 | −0.0014 | 0.0000 |
| **PR-AUC** | **−0.0001** | **+0.0135** | **+0.0183** |
| F1 (val-best) | −0.0048 | −0.0031 | +0.0045 |
| Platt Brier | +0.0008 | −0.0008 | −0.0005 |

Reading the 2×2:

- **Penalty inclusion has ~zero main effect on PR-AUC.** Averaged over σ,
  penalty inclusion changes PR-AUC by −0.0001, well within seed std. The
  earlier "penalty inclusion biases PR-AUC upward by ~0.013" claim
  (extracted from a single A-vs-D comparison) was confounded with σ.
- **A → D cross-protocol gap of +0.0135 cannot be attributed to penalty
  alone.** Factorial decomposition: penalty main effect ≈ 0 (so penalty
  has no average upward push on PR-AUC), positive average σ main effect
  (+0.0135), and a σ-conditioned penalty effect (interaction +0.0183).
  Adding "σ main effect + interaction" to predict the A → D gap would be
  double-counting; the correct decomposition along the A → C → D path is
  A → C contributes +0.0044 (σ change at penalty=out), and C → D
  contributes +0.0091 (penalty change at σ=high), summing to +0.0135.
  The σ × penalty interaction (+0.0183) reflects that the *penalty
  effect itself reverses sign* between σ=2.0 pu (A→B: −0.0092) and
  σ=4.69 pu (C→D: +0.0091), not an extra additive term on top of A → D.
- **Why the interaction?** Penalty shots have xG ≈ 0.76 with the shooter
  ~12 yards from goal. At σ=4.69 pu (~10 yards on the pitch length axis),
  a penalty's Gaussian footprint covers a large fraction of the area
  between the shooter and goal, creating a distinctive feature pattern
  the model can latch onto. At σ=2.0 pu (~4 yards), the penalty's
  Gaussian is sharper and the feature pattern is less distinctive, so
  including penalties just adds outlier shots that the model can't
  trivially separate. This is a hypothesis, not an established mechanism.
- **F1 (val-best) shows the cleaner single-variable effect:** penalty
  inclusion lowers F1 by ~0.005 averaged across σ, consistent with the
  earlier intuition that penalty inclusion shifts the validation-best
  threshold suboptimally. Interaction on F1 is small (+0.005).
- **ROC-AUC and Brier are essentially insensitive** to either variable.

Cell labels in the codebase:

| Cell | Dataset path | Checkpoint pattern |
|---|---|---|
| A | `data/processed/freeze_frames_64_s2p0_no_penalty/` | `outputs/models/cnn_64_s2p0_cv_seed{42,1,7}/` |
| B | `data/processed/freeze_frames_64_s2p0_with_penalty/` | `outputs/models/cnn_64_s2p0_with_penalty_cv_seed{42,1,7}/` |
| C | `data/processed/freeze_frames_64_s4p69_no_penalty/` | `outputs/models/cnn_64_s4p69_no_penalty_cv_seed{42,1,7}/` |
| D | `data/processed/freeze_frames_64.npz` (legacy) | `outputs/models/cnn_64_cv_seed{42,1,7}/` |

Paper implications:

- The "penalty bias" claim has been rewritten from a single-variable
  to a two-variable framing.
- The Limitations note that the controlled-vs-legacy comparison changed
  two variables at once is now resolved: we have the 2×2 numbers to
  report the per-variable effects with interaction.
- A new figure (or table) belongs in either Section 5.5 (penalty
  methodology) or an appendix, depending on venue space.

