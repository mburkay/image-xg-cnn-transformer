#!/usr/bin/env python3
"""Token-count sweep figure: 4 metrics x token counts (16, 36, 64) x model (CNN, Hybrid).

Reads CV results pickled by the orchestrator post-processing step.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "outputs/models"


def load_run(name):
    eval_dir = MODELS / name / "eval"
    metrics = json.load(open(eval_dir / "metrics.json"))
    cal = json.load(open(eval_dir / "calibration_metrics.json"))
    thr = json.load(open(eval_dir / "threshold_analysis.json"))
    return {
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "platt_brier": cal["platt"]["brier_score"],
        "f1_val_best": thr["test"]["best_f1_from_val"]["f1"],
    }


def cell_stats(dirs, key):
    rows = [load_run(d) for d in dirs]
    vals = np.array([r[key] for r in rows])
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


cells = {
    "CNN": {
        16: ["cnn_64_s2p0_cv_seed42", "cnn_64_s2p0_cv_seed1", "cnn_64_s2p0_cv_seed7"],
        36: ["cnn_96_s2p0_cv_seed42", "cnn_96_s2p0_cv_seed1", "cnn_96_s2p0_cv_seed7"],
        64: ["cnn_128_s2p0", "cnn_128_s2p0_cv_seed1", "cnn_128_s2p0_cv_seed7"],
    },
    "Hybrid": {
        16: ["hybrid_64_s2p0_cv_seed42", "hybrid_64_s2p0_cv_seed1", "hybrid_64_s2p0_cv_seed7"],
        36: ["hybrid_96_s2p0_cv_seed42", "hybrid_96_s2p0_cv_seed1", "hybrid_96_s2p0_cv_seed7"],
        64: ["hybrid_128_s2p0_cv_seed42", "hybrid_128_s2p0_cv_seed1", "hybrid_128_s2p0_cv_seed7"],
    },
}

metrics_to_plot = [
    ("roc_auc", "ROC-AUC", "higher is better"),
    ("pr_auc", "PR-AUC", "higher is better"),
    ("f1_val_best", "F1 at val-best threshold", "higher is better"),
    ("platt_brier", "Platt-scaled Brier score", "lower is better"),
]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    out = ROOT / "outputs/figures/figure_token_sweep.png"

    token_counts = [16, 36, 64]
    res_labels = {16: "64x64", 36: "96x96", 64: "128x128"}

    for ax, (key, title, dirn) in zip(axes.flatten(), metrics_to_plot):
        for model, color in [("CNN", "C3"), ("Hybrid", "C0")]:
            means, stds, ns = [], [], []
            for t in token_counts:
                m, s = cell_stats(cells[model][t], key)
                means.append(m); stds.append(s); ns.append(len(cells[model][t]))
            ax.errorbar(
                token_counts, means, yerr=stds,
                marker="o", lw=2, ms=8, capsize=5, color=color, label=model,
            )
            for t, m, s, n in zip(token_counts, means, stds, ns):
                if n == 1:
                    ax.scatter([t], [m], marker="x", s=140, color=color, zorder=5)
        ax.set_xticks(token_counts)
        ax.set_xticklabels([f"{t}\n({res_labels[t]})" for t in token_counts])
        ax.set_xlabel("Token count (resolution)")
        ax.set_ylabel(title)
        ax.set_title(f"{title}  —  {dirn}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")

    fig.suptitle("Token sweep at sigma_pitch_units = 2.0  (controlled: penalty-excluded, n=3 per cell)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
