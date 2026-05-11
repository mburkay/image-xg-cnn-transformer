#!/bin/bash
# Orchestrate the 10 remaining training runs sequentially.
# - 96x96 sigma=2.0m token-sweep: CNN x 3 seeds + Hybrid x 3 seeds
# - 5-seed extension: CNN 64 + 2 new seeds, Hybrid 128 s2p0 + 2 new seeds
# Logs to /tmp/extended_runs.log; emits BEGIN/END markers for the Monitor watcher.

set -e
cd "$(dirname "$0")/.."

LOG=/tmp/extended_runs.log
echo "=== orchestrate_extended_runs START $(date) ===" | tee "$LOG"

run_train() {
  local model=$1
  local data=$2
  local epochs=$3
  local batch=$4
  local seed=$5
  local outdir=$6
  echo "BEGIN_TRAIN model=$model seed=$seed outdir=$outdir $(date)" | tee -a "$LOG"
  python3 scripts/03_train.py \
    --data "$data" \
    --model "$model" \
    --epochs "$epochs" \
    --batch-size "$batch" \
    --learning-rate 1e-4 \
    --output-dir "$outdir" \
    --seed "$seed" \
    --device mps 2>&1 | tee -a "$LOG" | grep -E "^(epoch=|Best )" | tail -5 || true
  echo "END_TRAIN outdir=$outdir $(date)" | tee -a "$LOG"
}

run_eval() {
  local data=$1
  local outdir=$2
  echo "BEGIN_EVAL outdir=$outdir $(date)" | tee -a "$LOG"
  python3 scripts/04_evaluate.py \
    --data "$data" \
    --model-path "$outdir/best_model.pt" \
    --output-dir "$outdir/eval" \
    --device mps 2>&1 | tee -a "$LOG" | tail -10 || true
  echo "END_EVAL outdir=$outdir $(date)" | tee -a "$LOG"
}

# 5-seed extension first (CNN 64, fastest) so partial results land sooner.
for seed in 13 21; do
  run_train cnn data/processed/freeze_frames_64.npz 15 128 "$seed" "outputs/models/cnn_64_cv_seed${seed}"
  run_eval data/processed/freeze_frames_64.npz "outputs/models/cnn_64_cv_seed${seed}"
done

# 96x96 CNN sweep (3 seeds)
for seed in 42 1 7; do
  run_train cnn data/processed/freeze_frames_96_s2p0_no_penalty 15 64 "$seed" "outputs/models/cnn_96_s2p0_cv_seed${seed}"
  run_eval data/processed/freeze_frames_96_s2p0_no_penalty "outputs/models/cnn_96_s2p0_cv_seed${seed}"
done

# 96x96 Hybrid sweep (3 seeds)
for seed in 42 1 7; do
  run_train hybrid data/processed/freeze_frames_96_s2p0_no_penalty 30 64 "$seed" "outputs/models/hybrid_96_s2p0_cv_seed${seed}"
  run_eval data/processed/freeze_frames_96_s2p0_no_penalty "outputs/models/hybrid_96_s2p0_cv_seed${seed}"
done

# 5-seed extension Hybrid 128 (slowest, last)
for seed in 13 21; do
  run_train hybrid data/processed/freeze_frames_128_s2p0_no_penalty 30 64 "$seed" "outputs/models/hybrid_128_s2p0_cv_seed${seed}"
  run_eval data/processed/freeze_frames_128_s2p0_no_penalty "outputs/models/hybrid_128_s2p0_cv_seed${seed}"
done

echo "=== orchestrate_extended_runs DONE $(date) ===" | tee -a "$LOG"
