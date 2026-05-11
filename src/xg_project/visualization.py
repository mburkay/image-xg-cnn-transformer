"""Plotting utilities for training and evaluation artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_training_history(history: dict[str, list[float]], output_path: str | Path) -> None:
    """Save loss/accuracy curves."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.get("loss", []), label="Train Loss")
    axes[0].plot(history.get("val_loss", []), label="Val Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="Train Accuracy")
    axes[1].plot(history.get("val_accuracy", []), label="Val Accuracy")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    """Save a confusion matrix heatmap."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(cm, display_labels=["No Goal", "Goal"])
    fig, ax = plt.subplots(figsize=(5, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path,
    n_bins: int = 10,
) -> None:
    """Save a reliability diagram."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_channel_preview(image: np.ndarray, output_path: str | Path) -> None:
    """Save a four-channel freeze-frame preview for visual QA."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    titles = ["Attackers", "Defenders", "Goalkeeper", "Shooter"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for idx, ax in enumerate(axes):
        sns.heatmap(
            image[:, :, idx],
            ax=ax,
            cmap="mako",
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(titles[idx])
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
