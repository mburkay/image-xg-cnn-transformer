"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load a rendered dataset from `.npz` or mmap-friendly folder format."""

    path = Path(path)
    if path.is_dir():
        return {
            "images": np.load(path / "images.npy", mmap_mode="r"),
            "labels": np.load(path / "labels.npy", mmap_mode="r"),
            "shot_ids": np.load(path / "shot_ids.npy", allow_pickle=True),
            "match_ids": np.load(path / "match_ids.npy", allow_pickle=True),
            "splits": np.load(path / "splits.npy", allow_pickle=True),
        }

    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_npz_dataset(path: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for `load_dataset`."""

    return load_dataset(path)


def split_indices(data: dict[str, Any], split: str) -> np.ndarray:
    """Return integer row indices for one split."""

    splits = np.asarray(data["splits"]).astype(str)
    indices = np.flatnonzero(splits == split)
    if len(indices) == 0:
        raise ValueError(f"No rows found for split {split!r}")
    return indices


def select_split(data: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return `(images, labels)` for one split."""

    indices = split_indices(data, split)
    images = data["images"][indices].astype("float32", copy=False)
    labels = data["labels"][indices].astype("float32", copy=False)
    return images, labels


def compute_class_weight(labels: np.ndarray) -> dict[int, float]:
    """Compute balanced binary class weights."""

    labels = labels.astype(int)
    total = len(labels)
    positives = int(labels.sum())
    negatives = total - positives
    if positives == 0 or negatives == 0:
        return {0: 1.0, 1: 1.0}

    return {
        0: total / (2.0 * negatives),
        1: total / (2.0 * positives),
    }
