# Image-based xG with CNN+Transformer Hybrids

Reference implementation and reproducibility artefacts for the paper

> **Spatial Resolution and Token Count Matter: A CNN+Transformer Hybrid
> for Image-Based Expected Goals**
> Mustafa Burkay Özdemir and Nejat Yumuşak. MLSA 2026 (under review).

The paper studies image-based Expected Goals (xG) on StatsBomb shot
freeze frames. The main contribution is a controlled six-cell ablation
(CNN/Hybrid × 64/96/128) under a strictly controlled rendering protocol
(sigma in pitch-coordinate units, penalty-excluded, 4-channel
full-pitch), along with a 5-point sigma sweep at 128×128 and a 2×2
σ×penalty factorial ablation at 64×64.

## Data

This repository does **not** ship the StatsBomb open-data dump. Clone
it separately into the project root:

```bash
git clone https://github.com/statsbomb/open-data.git
```

The StatsBomb open-data licence applies to that dataset; this
repository does not re-license it.

After cloning, `scripts/01_download_shots.py` reads from the local
`open-data/` clone (despite the historical script name) and produces
`data/raw/shots.pkl` (~88k shot rows; ~86.8k renderable freeze frames;
~86.7k after penalty exclusion).

## Setup

```bash
# Python 3.10+ recommended; tested on 3.12 with Apple Silicon MPS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or, as an editable install:
pip install -e .
```

The PyTorch device defaults to `auto` (CUDA → MPS → CPU). All
experiments in the paper used Apple Silicon MPS with no mixed
precision. CPU and CUDA paths are exercised by the same `--device`
flag (see `python scripts/03_train.py --help`).

## Reproducing the paper

The four numbered scripts run the full pipeline from raw StatsBomb
JSON to per-cell test metrics:

```bash
# (1) StatsBomb JSON → shots.pkl
python scripts/01_download_shots.py

# (2) shots.pkl → folder-backed image dataset
#     (controlled headline rendering: sigma_pu=2.0, penalty-excluded)
python scripts/02_build_images.py \
    --shots data/raw/shots.pkl \
    --output data/processed/freeze_frames_128_s2p0_no_penalty \
    --storage folder \
    --metadata-output data/processed/shots_with_splits_128_s2p0_no_penalty.pkl \
    --height 128 --width 128 \
    --sigma-meters 2.0 \
    --exclude-penalties

# (3) train one cell (CNN or hybrid, any seed)
python scripts/03_train.py \
    --data data/processed/freeze_frames_128_s2p0_no_penalty \
    --model hybrid \
    --epochs 30 --batch-size 64 --learning-rate 1e-4 \
    --output-dir outputs/models/hybrid_128_s2p0_cv_seed42 \
    --seed 42 --device auto

# (4) evaluate the val-best checkpoint on the test split
python scripts/04_evaluate.py \
    --data data/processed/freeze_frames_128_s2p0_no_penalty \
    --model-path outputs/models/hybrid_128_s2p0_cv_seed42/best_model.pt \
    --output-dir outputs/models/hybrid_128_s2p0_cv_seed42/eval \
    --device auto
```

The three orchestrator shell scripts under `scripts/orchestrate_*.sh`
chain together the headline six-cell ablation, the 2×2 σ×penalty
ablation, and the controlled 64-cell re-render. They write per-run
output to `outputs/models/` and a log to `/tmp/*.log`.

`scripts/figure_*.py` regenerate the four paper-grade figures
(token sweep, calibration reliability, attention rollout, training
trajectories) from the per-run checkpoints and eval JSONs.

> A note on the rendering flag. The argparse flag is named
> `--sigma-meters` for backwards compatibility; the value is in
> **StatsBomb pitch-coordinate units** (a 120 × 80 frame), not literal
> meters. See `src/xg_project/constants.py` and Section 2.2 of the
> paper.

## Expected outputs

After running step (4) on the controlled `hybrid_128_s2p0_cv_seed42`
cell, `eval/metrics.json` should report numbers within seed std of:

| Metric | Value |
|---|---|
| ROC-AUC | 0.793 |
| PR-AUC | 0.360 |
| F1 (val-best threshold) | 0.407 |
| Platt-scaled Brier | 0.077 |

The full controlled six-cell headline table and the 2×2 σ×penalty
factorial table are in `EXPERIMENTS.md` under the "Controlled
64-Cell Re-Render" and "2×2 σ × penalty ablation" subsections.

## Model checkpoints

The paper reports results from 38 model checkpoints (six-cell
headline × 3 seeds plus the sigma sweep at 128, the 2×2 ablation, and
the 5-seed sensitivity on the legacy-rendering pair). The checkpoints
themselves are **not** committed here (~600 MB); they will be deposited
on a public archive (Zenodo or equivalent) upon paper acceptance, and
the DOI will be added back to this README. In the meantime, every
checkpoint is reproducible end-to-end from this codebase plus the
StatsBomb open-data clone.

## Hardware notes

- Tested on Apple Silicon MPS (macOS 14, PyTorch 2.x). The training
  loop does not use mixed precision.
- CUDA and CPU paths are exercised by the `--device` flag but were
  not the primary development target; per-run wall-clock numbers in
  the paper are MPS-specific.
- We report seed standard deviations for every controlled cell; we
  did not separately isolate hardware/backend nondeterminism, so both
  effects are mixed into the reported std bands.

## Repository layout

```
.
├── LICENSE                     # MIT for code; StatsBomb data licensed separately
├── README.md                   # this file
├── pyproject.toml              # editable install (src/ layout)
├── requirements.txt
├── EXPERIMENTS.md              # full per-cell experiment log, supplementary
├── HYPERPARAMETERS.md          # per-config defaults + exceptions, supplementary
├── RELATED_WORK_NOTES.md       # source-backed notes on xG-CNN and friends
├── scripts/
│   ├── 01_download_shots.py
│   ├── 02_build_images.py
│   ├── 03_train.py
│   ├── 04_evaluate.py
│   ├── diagnostics_rendering_protocol.py
│   ├── figure_*.py             # four paper-figure regeneration scripts
│   └── orchestrate_*.sh        # three multi-run orchestrators
├── src/xg_project/             # importable Python package
│   ├── constants.py            # pitch coordinates, default image size
│   ├── data_collection.py      # StatsBomb open-data → shots.pkl
│   ├── dataset.py              # npz / folder loaders, split helpers
│   ├── image_dataset.py        # full image-dataset build pipeline
│   ├── rendering.py            # 4-channel Gaussian rendering
│   ├── splitting.py            # match-level stratified split
│   ├── torch_models.py         # CNN-XG, HybridCNNTransformer-XG, ViT-XG
│   ├── torch_training.py       # training loop, MPS device selection
│   ├── torch_evaluation.py     # test metrics, Platt + isotonic calibration
│   └── visualization.py        # plots used by the figure scripts
└── paper/                      # LaTeX source for the MLSA 2026 submission
    ├── main.tex
    ├── references.bib
    ├── figures/F1_*, F2_*, F3_*.png
    └── README.md               # build instructions (llncs.cls download)
```

## Licence

The code in this repository is released under the MIT License (see
`LICENSE`). The StatsBomb open-data is **not** distributed by this
repository; download it directly from
<https://github.com/statsbomb/open-data> and respect its licence.
The Springer LNCS class file `llncs.cls` referenced by `paper/main.tex`
is not redistributed here; it is downloaded separately from Springer
(see `paper/README.md`).

## Citation

```bibtex
@inproceedings{ozdemir2026imagexg,
  author    = {{\"O}zdemir, Mustafa Burkay and Yumu{\c{s}}ak, Nejat},
  title     = {Spatial Resolution and Token Count Matter:
               A {CNN+Transformer} Hybrid for Image-Based Expected Goals},
  booktitle = {Proceedings of the 13th Workshop on Machine Learning and
               Data Mining for Sports Analytics (MLSA), co-located with
               ECML/PKDD 2026},
  year      = {2026},
  note      = {Under review}
}
```
