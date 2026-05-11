"""Match-level train/validation/test splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _validate_ratios(train_size: float, val_size: float, test_size: float) -> None:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    for name, value in {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def _goal_buckets(goals: pd.Series) -> pd.Series:
    return goals.clip(upper=4).astype(int).astype(str)


def _safe_stratify(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def split_by_match(
    df: pd.DataFrame,
    match_col: str = "match_id",
    label_col: str = "is_goal",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.Series:
    """Assign each row to train/val/test while keeping whole matches together.

    A greedy group splitter is used to keep both shot counts and goal counts
    close to the requested ratios without leaking a match across splits.
    """

    _validate_ratios(train_size, val_size, test_size)
    missing = {match_col, label_col}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        return pd.Series(dtype="object", index=df.index)

    row_labels = df[label_col].astype(int)
    groups = (
        df.assign(_label=row_labels)
        .groupby(match_col, dropna=False)
        .agg(n=("_label", "size"), goals=("_label", "sum"))
        .reset_index()
    )
    groups["_bucket"] = _goal_buckets(groups["goals"])

    if len(groups) < 3:
        split_names = np.array(["train", "val", "test"], dtype=object)
        assignments = {
            row[match_col]: str(split_names[idx % len(split_names)])
            for idx, row in groups.iterrows()
        }
        return df[match_col].map(assignments).astype("category")

    temp_size = val_size + test_size
    train_groups, temp_groups = train_test_split(
        groups,
        test_size=temp_size,
        random_state=random_state,
        shuffle=True,
        stratify=_safe_stratify(groups["_bucket"]),
    )
    relative_test_size = test_size / temp_size
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=relative_test_size,
        random_state=random_state + 1,
        shuffle=True,
        stratify=_safe_stratify(temp_groups["_bucket"]),
    )

    assignments: dict[object, str] = {}
    assignments.update({match_id: "train" for match_id in train_groups[match_col]})
    assignments.update({match_id: "val" for match_id in val_groups[match_col]})
    assignments.update({match_id: "test" for match_id in test_groups[match_col]})

    return df[match_col].map(assignments).astype("category")


def split_summary(df: pd.DataFrame, split_col: str = "split", label_col: str = "is_goal") -> pd.DataFrame:
    """Return shot count and goal rate by split."""

    missing = {split_col, label_col}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    summary = (
        df.groupby(split_col, observed=True)
        .agg(shots=(label_col, "size"), goals=(label_col, "sum"), goal_rate=(label_col, "mean"))
        .reset_index()
    )
    return summary
