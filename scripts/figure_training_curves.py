#!/usr/bin/env python3
"""Training curves: train + val AUC trajectories for the 4 headline models.

Reads `history.json` from each model's output directory and plots AUC over
epochs in a 2x2 grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    ("CNN 64 (controlled)", "outputs/models/cnn_64_s2p0_cv_seed42"),
    ("CNN 128 (controlled)", "outputs/models/cnn_128_s2p0"),
    ("Hybrid 64 (controlled)", "outputs/models/hybrid_64_s2p0_cv_seed42"),
    ("Hybrid 128 (controlled)", "outputs/models/hybrid_128_s2p0_cv_seed42"),
]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    out_path = ROOT / "outputs/figures/figure_training_curves.png"

    for ax, (label, model_dir) in zip(axes.flatten(), MODELS):
        history = json.load(open(ROOT / model_dir / "history.json"))
        epochs = np.arange(1, len(history["auc"]) + 1)
        ax.plot(epochs, history["auc"], color="C3", marker="o", lw=1.5,
                ms=4, label="Train AUC")
        ax.plot(epochs, history["val_auc"], color="C0", marker="s", lw=1.5,
                ms=4, label="Val AUC")
        best_epoch = int(np.argmax(history["val_auc"])) + 1
        best_val = float(history["val_auc"][best_epoch - 1])
        ax.axvline(best_epoch, color="gray", ls=":", lw=1)
        ax.scatter([best_epoch], [best_val], color="C0", s=120, zorder=5,
                   marker="*", label=f"Best val={best_val:.4f} @ epoch {best_epoch}")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.74, 0.93)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Training trajectories: 4 headline models (controlled rendering, sigma_pitch_units=2.0, penalty-excluded, seed=42)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
