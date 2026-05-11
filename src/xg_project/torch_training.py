"""PyTorch training loop for the local verified backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from xg_project.dataset import compute_class_weight, load_dataset, split_indices
from xg_project.torch_models import build_torch_model
from xg_project.visualization import plot_training_history


class FreezeFrameDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.arange(len(labels)) if indices is None else indices.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_idx = int(self.indices[idx])
        image = np.asarray(self.images[source_idx], dtype=np.float32)
        if self.augment:
            if np.random.random() > 0.5:
                image = np.flip(image, axis=0).copy()
            if np.random.random() > 0.5:
                noise = np.random.normal(0.0, 0.015, size=image.shape).astype("float32")
                image = np.clip(image + noise, 0.0, 1.0)

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).copy())
        label_tensor = torch.tensor(float(self.labels[source_idx]), dtype=torch.float32)
        return image_tensor, label_tensor


def choose_device(device: str = "auto") -> torch.device:
    if device != "auto":
        if device == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS was requested but is not available in this process. "
                "Run the training command in an environment where "
                "torch.backends.mps.is_available() returns True."
            )
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _epoch_metrics(y_true: list[float], y_prob: list[float], losses: list[float]) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob_arr >= 0.5).astype(np.int64)
    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
    }
    if len(np.unique(y_true_arr)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    else:
        metrics["auc"] = float("nan")
    return metrics


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    y_true: list[float] = []
    y_prob: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.detach().cpu().numpy().tolist())

    return _epoch_metrics(y_true, y_prob, losses)


def _checkpoint_payload(
    model: nn.Module,
    model_name: str,
    input_shape: tuple[int, int, int],
    history: dict[str, list[float]],
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "input_shape": tuple(int(v) for v in input_shape),
        "state_dict": model.state_dict(),
        "history": history,
    }


def train_torch_from_npz(
    data_path: str | Path,
    output_dir: str | Path,
    model_name: str = "hybrid",
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    augment: bool = False,
    seed: int = 42,
    patience: int = 10,
    device: str = "auto",
) -> Path:
    """Train a PyTorch model from a rendered `.npz` dataset."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = choose_device(device)
    print(f"Using device: {torch_device}", flush=True)

    data = load_dataset(data_path)
    train_idx = split_indices(data, "train")
    val_idx = split_indices(data, "val")
    y_train = np.asarray(data["labels"][train_idx], dtype=np.float32)

    train_loader = DataLoader(
        FreezeFrameDataset(data["images"], data["labels"], indices=train_idx, augment=augment),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        FreezeFrameDataset(data["images"], data["labels"], indices=val_idx, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_torch_model(model_name, input_shape=tuple(data["images"].shape[1:])).to(torch_device)
    class_weight = compute_class_weight(y_train)
    pos_weight = torch.tensor(class_weight[1] / class_weight[0], device=torch_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    )

    history: dict[str, list[float]] = {
        "loss": [],
        "accuracy": [],
        "auc": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
    }
    best_auc = -np.inf
    best_epoch = -1
    best_path = output_dir / "best_model.pt"

    for epoch in range(epochs):
        train_metrics = _run_epoch(model, train_loader, criterion, torch_device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, torch_device)
        scheduler.step(val_metrics["loss"])

        for key in ("loss", "accuracy", "auc"):
            history[key].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])

        val_auc = val_metrics["auc"]
        improved = np.isfinite(val_auc) and val_auc > best_auc
        if improved:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(
                _checkpoint_payload(model, model_name, tuple(data["images"].shape[1:]), history),
                best_path,
            )

        print(
            f"epoch={epoch + 1:03d} "
            f"loss={train_metrics['loss']:.4f} "
            f"auc={train_metrics['auc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}",
            flush=True,
        )

        if epoch - best_epoch >= patience:
            break

    if not best_path.exists():
        torch.save(_checkpoint_payload(model, model_name, tuple(data["images"].shape[1:]), history), best_path)

    final_path = output_dir / "final_model.pt"
    torch.save(_checkpoint_payload(model, model_name, tuple(data["images"].shape[1:]), history), final_path)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    plot_training_history(history, output_dir / "training_curves.png")
    return best_path
