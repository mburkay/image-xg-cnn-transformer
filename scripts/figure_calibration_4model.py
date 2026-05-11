#!/usr/bin/env python3
"""Side-by-side reliability diagram + calibration metrics for the 4 headline models.

Uses the existing eval/y_prob.npy and eval/y_true.npy plus Platt calibrators
from the eval pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS = [
    ("CNN 64 (controlled)", "outputs/models/cnn_64_s2p0_cv_seed42"),
    ("CNN 128 (controlled)", "outputs/models/cnn_128_s2p0"),
    ("Hybrid 64 (controlled)", "outputs/models/hybrid_64_s2p0_cv_seed42"),
    ("Hybrid 128 (controlled)", "outputs/models/hybrid_128_s2p0_cv_seed42"),
]


def load_probs(model_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eval_dir = model_dir / "eval"
    y_true = np.load(eval_dir / "y_true.npy")
    y_prob_raw = np.load(eval_dir / "y_prob.npy")
    y_prob_platt = np.load(eval_dir / "y_prob_platt.npy")
    return y_true, y_prob_raw, y_prob_platt


def reliability_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    return calibration_curve(y_true, y_prob, n_bins=n_bins)


def main() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    out_path = ROOT / "outputs/figures/figure_calibration_4model.png"

    for ax, (label, model_dir) in zip(axes, MODELS):
        y_true, y_prob_raw, y_prob_platt = load_probs(ROOT / model_dir)

        prob_true_raw, prob_pred_raw = reliability_points(y_true, y_prob_raw)
        prob_true_platt, prob_pred_platt = reliability_points(y_true, y_prob_platt)

        cal = json.load(open(ROOT / model_dir / "eval/calibration_metrics.json"))
        brier_raw = cal["raw"]["brier_score"]
        brier_platt = cal["platt"]["brier_score"]

        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")
        ax.plot(prob_pred_raw, prob_true_raw, marker="o", color="C3",
                label=f"Raw (Brier={brier_raw:.3f})")
        ax.plot(prob_pred_platt, prob_true_platt, marker="s", color="C0",
                label=f"Platt (Brier={brier_platt:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Observed goal rate")
    fig.suptitle("Reliability diagrams: 4-model comparison (controlled rendering, sigma_pitch_units=2.0, penalty-excluded, seed=42)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
