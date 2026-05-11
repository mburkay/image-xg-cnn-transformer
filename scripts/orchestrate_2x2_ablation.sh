#!/bin/bash
# 2x2 sigma x penalty ablation at 64x64.
# Cells:
#   A: sigma_pu=2.0, penalty-OUT  → already in outputs/models/cnn_64_s2p0_cv_seed{42,1,7}
#   B: sigma_pu=2.0, penalty-IN   → NEW (this script)
#   C: sigma_pu=4.69, penalty-OUT → NEW (this script)
#   D: sigma_pu=4.69, penalty-IN  → already in outputs/models/cnn_64_cv_seed{42,1,7} (legacy)
# 6 new CNN runs total (B and C, 3 seeds each). 64x64 batch=128 → ~7 min each.

set -e
cd "$(dirname "$0")/.."

LOG=/tmp/ablation_2x2.log
echo "=== 2x2 ablation START $(date) ===" | tee "$LOG"

# Renders
echo "BEGIN_RENDER B sigma=2.0 penalty-IN $(date)" | tee -a "$LOG"
python3 scripts/02_build_images.py \
  --shots data/raw/shots.pkl \
  --output data/processed/freeze_frames_64_s2p0_with_penalty \
  --storage folder \
  --metadata-output data/processed/shots_with_splits_64_s2p0_with_penalty.pkl \
  --height 64 --width 64 \
  --sigma-meters 2.0 \
  --preview-output outputs/figures/freeze_frame_preview_64_s2p0_with_penalty.png 2>&1 | tee -a "$LOG" | tail -3
echo "END_RENDER B $(date)" | tee -a "$LOG"

echo "BEGIN_RENDER C sigma=4.69 penalty-OUT $(date)" | tee -a "$LOG"
python3 scripts/02_build_images.py \
  --shots data/raw/shots.pkl \
  --output data/processed/freeze_frames_64_s4p69_no_penalty \
  --storage folder \
  --metadata-output data/processed/shots_with_splits_64_s4p69_no_penalty.pkl \
  --height 64 --width 64 \
  --sigma-meters 4.6875 \
  --exclude-penalties \
  --preview-output outputs/figures/freeze_frame_preview_64_s4p69_no_penalty.png 2>&1 | tee -a "$LOG" | tail -3
echo "END_RENDER C $(date)" | tee -a "$LOG"

run_train() {
  local data=$1 seed=$2 outdir=$3
  echo "BEGIN_TRAIN $outdir seed=$seed $(date)" | tee -a "$LOG"
  python3 scripts/03_train.py \
    --data "$data" --model cnn --epochs 15 --batch-size 128 \
    --learning-rate 1e-4 --output-dir "$outdir" --seed "$seed" --device mps 2>&1 \
    | tee -a "$LOG" | grep -E "^(epoch=|Best )" | tail -5 || true
  echo "END_TRAIN $outdir $(date)" | tee -a "$LOG"
}

run_eval() {
  local data=$1 outdir=$2
  echo "BEGIN_EVAL $outdir $(date)" | tee -a "$LOG"
  python3 scripts/04_evaluate.py \
    --data "$data" --model-path "$outdir/best_model.pt" \
    --output-dir "$outdir/eval" --device mps 2>&1 | tee -a "$LOG" | tail -10 || true
  echo "END_EVAL $outdir $(date)" | tee -a "$LOG"
}

# Cell B: sigma=2.0, penalty-IN
for seed in 42 1 7; do
  run_train data/processed/freeze_frames_64_s2p0_with_penalty "$seed" "outputs/models/cnn_64_s2p0_with_penalty_cv_seed${seed}"
  run_eval data/processed/freeze_frames_64_s2p0_with_penalty "outputs/models/cnn_64_s2p0_with_penalty_cv_seed${seed}"
done

# Cell C: sigma=4.69, penalty-OUT
for seed in 42 1 7; do
  run_train data/processed/freeze_frames_64_s4p69_no_penalty "$seed" "outputs/models/cnn_64_s4p69_no_penalty_cv_seed${seed}"
  run_eval data/processed/freeze_frames_64_s4p69_no_penalty "outputs/models/cnn_64_s4p69_no_penalty_cv_seed${seed}"
done

echo "=== 2x2 ablation DONE $(date) ===" | tee -a "$LOG"
