#!/bin/bash
# Orchestrate the controlled-rendering 64-cell re-runs + CNN 128 seed extension.
# - 64x64 sigma_pitch_units=2.0 + no-penalty re-render
# - 128x128 sigma_pitch_units=2.0 + no-penalty re-render (s2p0 dataset was deleted earlier)
# - CNN 64 controlled x 3 seeds, Hybrid 64 controlled x 3 seeds
# - CNN 128 (s2p0) x 2 new seeds (1, 7) to complete its 3-seed CV

set -e
cd "$(dirname "$0")/.."

LOG=/tmp/controlled_runs.log
echo "=== orchestrate_controlled_64_runs START $(date) ===" | tee "$LOG"

# --- Renders ---
echo "BEGIN_RENDER 64_s2p0_no_penalty $(date)" | tee -a "$LOG"
python3 scripts/02_build_images.py \
  --shots data/raw/shots.pkl \
  --output data/processed/freeze_frames_64_s2p0_no_penalty \
  --storage folder \
  --metadata-output data/processed/shots_with_splits_64_s2p0_no_penalty.pkl \
  --height 64 --width 64 \
  --sigma-meters 2.0 \
  --exclude-penalties \
  --preview-output outputs/figures/freeze_frame_preview_64_s2p0.png 2>&1 | tee -a "$LOG" | tail -5 || true
echo "END_RENDER 64_s2p0_no_penalty $(date)" | tee -a "$LOG"

echo "BEGIN_RENDER 128_s2p0_no_penalty (re-render) $(date)" | tee -a "$LOG"
python3 scripts/02_build_images.py \
  --shots data/raw/shots.pkl \
  --output data/processed/freeze_frames_128_s2p0_no_penalty \
  --storage folder \
  --metadata-output data/processed/shots_with_splits_128_s2p0_no_penalty.pkl \
  --height 128 --width 128 \
  --sigma-meters 2.0 \
  --exclude-penalties \
  --preview-output outputs/figures/freeze_frame_preview_128_s2p0.png 2>&1 | tee -a "$LOG" | tail -5 || true
echo "END_RENDER 128_s2p0_no_penalty $(date)" | tee -a "$LOG"

run_train() {
  local model=$1 data=$2 epochs=$3 batch=$4 seed=$5 outdir=$6
  echo "BEGIN_TRAIN model=$model seed=$seed outdir=$outdir $(date)" | tee -a "$LOG"
  python3 scripts/03_train.py \
    --data "$data" --model "$model" --epochs "$epochs" --batch-size "$batch" \
    --learning-rate 1e-4 --output-dir "$outdir" --seed "$seed" --device mps 2>&1 \
    | tee -a "$LOG" | grep -E "^(epoch=|Best )" | tail -5 || true
  echo "END_TRAIN outdir=$outdir $(date)" | tee -a "$LOG"
}

run_eval() {
  local data=$1 outdir=$2
  echo "BEGIN_EVAL outdir=$outdir $(date)" | tee -a "$LOG"
  python3 scripts/04_evaluate.py \
    --data "$data" --model-path "$outdir/best_model.pt" \
    --output-dir "$outdir/eval" --device mps 2>&1 | tee -a "$LOG" | tail -10 || true
  echo "END_EVAL outdir=$outdir $(date)" | tee -a "$LOG"
}

# CNN 64 controlled x 3 seeds (fastest)
for seed in 42 1 7; do
  run_train cnn data/processed/freeze_frames_64_s2p0_no_penalty 15 128 "$seed" "outputs/models/cnn_64_s2p0_cv_seed${seed}"
  run_eval data/processed/freeze_frames_64_s2p0_no_penalty "outputs/models/cnn_64_s2p0_cv_seed${seed}"
done

# CNN 128 controlled (s2p0) - 2 new seeds (existing cnn_128_s2p0 already covers seed=42)
for seed in 1 7; do
  run_train cnn data/processed/freeze_frames_128_s2p0_no_penalty 15 64 "$seed" "outputs/models/cnn_128_s2p0_cv_seed${seed}"
  run_eval data/processed/freeze_frames_128_s2p0_no_penalty "outputs/models/cnn_128_s2p0_cv_seed${seed}"
done

# Hybrid 64 controlled x 3 seeds (slowest at 64; ~25-30 min each)
for seed in 42 1 7; do
  run_train hybrid data/processed/freeze_frames_64_s2p0_no_penalty 30 64 "$seed" "outputs/models/hybrid_64_s2p0_cv_seed${seed}"
  run_eval data/processed/freeze_frames_64_s2p0_no_penalty "outputs/models/hybrid_64_s2p0_cv_seed${seed}"
done

echo "=== orchestrate_controlled_64_runs DONE $(date) ===" | tee -a "$LOG"
