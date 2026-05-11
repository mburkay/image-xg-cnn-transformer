"""PyTorch evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from xg_project.dataset import compute_class_weight, load_dataset, split_indices
from xg_project.torch_models import build_torch_model
from xg_project.torch_training import FreezeFrameDataset, choose_device
from xg_project.visualization import plot_calibration, plot_confusion


def load_torch_checkpoint(path: str | Path, device: torch.device):
    """Load a PyTorch checkpoint with compatibility across torch versions."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def predict_torch(
    data_path: str | Path,
    model_path: str | Path,
    split: str = "test",
    batch_size: int = 64,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(y_true, y_prob)` for a saved PyTorch model."""

    torch_device = choose_device(device)
    checkpoint = load_torch_checkpoint(model_path, torch_device)
    data = load_dataset(data_path)
    indices = split_indices(data, split)
    y_true = np.asarray(data["labels"][indices], dtype=int)

    model = build_torch_model(
        str(checkpoint["model_name"]),
        input_shape=tuple(checkpoint["input_shape"]),
    ).to(torch_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    loader = DataLoader(
        FreezeFrameDataset(data["images"], data["labels"], indices=indices, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    probs: list[float] = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(torch_device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    return y_true, np.asarray(probs, dtype=np.float32)


def _prob_to_logit(y_prob: np.ndarray) -> np.ndarray:
    eps = 1e-6
    clipped = np.clip(y_prob, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    metrics = {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12,
    )
    return float(thresholds[int(np.nanargmax(f1))])


def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5
    scores = tpr[finite] - fpr[finite]
    return float(thresholds[finite][int(np.nanargmax(scores))])


def _class_weight_report(data_path: str | Path) -> dict[str, float]:
    data = load_dataset(data_path)
    train_idx = split_indices(data, "train")
    y_train = np.asarray(data["labels"][train_idx], dtype=np.float32)
    labels = y_train.astype(int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    class_weight = compute_class_weight(labels)
    return {
        "train_examples": int(len(labels)),
        "train_positives": positives,
        "train_negatives": negatives,
        "train_positive_rate": float(positives / len(labels)),
        "class_weight_0": float(class_weight[0]),
        "class_weight_1": float(class_weight[1]),
        "bce_pos_weight": float(class_weight[1] / class_weight[0]),
        "negative_to_positive_ratio": float(negatives / positives),
    }


def _fit_calibrators(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
) -> dict[str, object]:
    calibrators: dict[str, object] = {}
    if len(np.unique(y_val)) < 2:
        return calibrators

    platt = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="liblinear"),
    )
    platt.fit(_prob_to_logit(y_prob_val).reshape(-1, 1), y_val)
    calibrators["platt"] = platt

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(y_prob_val, y_val)
    calibrators["isotonic"] = isotonic
    return calibrators


def _apply_calibrator(name: str, calibrator: object, y_prob: np.ndarray) -> np.ndarray:
    if name == "platt":
        return calibrator.predict_proba(_prob_to_logit(y_prob).reshape(-1, 1))[:, 1]
    if name == "isotonic":
        return calibrator.predict(y_prob)
    raise ValueError(f"Unknown calibrator {name!r}")


def _save_multi_calibration_plot(
    y_true: np.ndarray,
    curves: dict[str, np.ndarray],
    output_path: str | Path,
    n_bins: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_prob in curves.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate_torch_model(
    data_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    batch_size: int = 64,
    threshold: float = 0.5,
    device: str = "auto",
) -> dict[str, float]:
    """Evaluate a saved PyTorch model and write report artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_prob = predict_torch(
        data_path=data_path,
        model_path=model_path,
        split=split,
        batch_size=batch_size,
        device=device,
    )
    metrics = {
        **_metrics_at_threshold(y_true, y_prob, threshold),
        **_ranking_metrics(y_true, y_prob),
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "class_weight_report.json").write_text(
        json.dumps(_class_weight_report(data_path), indent=2),
        encoding="utf-8",
    )

    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Goal", "Goal"],
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    np.save(output_dir / "y_prob.npy", y_prob)
    np.save(output_dir / "y_true.npy", y_true)
    plot_confusion(y_true, y_pred, output_dir / "confusion_matrix.png")
    if len(np.unique(y_true)) == 2:
        plot_calibration(y_true, y_prob, output_dir / "calibration_curve.png")

    y_val, y_prob_val = predict_torch(
        data_path=data_path,
        model_path=model_path,
        split="val",
        batch_size=batch_size,
        device=device,
    )
    best_f1_threshold = _best_f1_threshold(y_val, y_prob_val)
    youden_threshold = _youden_threshold(y_val, y_prob_val)
    threshold_report = {
        "validation": {
            "best_f1_threshold": best_f1_threshold,
            "best_f1_at_threshold": float(
                f1_score(y_val, (y_prob_val >= best_f1_threshold).astype(int))
            ),
            "youden_j_threshold": youden_threshold,
        },
        "test": {
            "default_0_5": _metrics_at_threshold(y_true, y_prob, threshold),
            "best_f1_from_val": _metrics_at_threshold(y_true, y_prob, best_f1_threshold),
            "youden_j_from_val": _metrics_at_threshold(y_true, y_prob, youden_threshold),
        },
    }
    (output_dir / "threshold_analysis.json").write_text(
        json.dumps(threshold_report, indent=2),
        encoding="utf-8",
    )

    calibrators = _fit_calibrators(y_val, y_prob_val)
    calibration_metrics: dict[str, dict[str, float]] = {
        "raw": _ranking_metrics(y_true, y_prob)
    }
    calibration_curves = {"raw": y_prob}
    for name, calibrator in calibrators.items():
        calibrated = _apply_calibrator(name, calibrator, y_prob)
        np.save(output_dir / f"y_prob_{name}.npy", calibrated)
        calibration_metrics[name] = _ranking_metrics(y_true, calibrated)
        calibration_curves[name] = calibrated

    (output_dir / "calibration_metrics.json").write_text(
        json.dumps(calibration_metrics, indent=2),
        encoding="utf-8",
    )
    if len(np.unique(y_true)) == 2:
        _save_multi_calibration_plot(
            y_true,
            calibration_curves,
            output_dir / "calibration_curve_with_scaling.png",
        )

    return metrics
