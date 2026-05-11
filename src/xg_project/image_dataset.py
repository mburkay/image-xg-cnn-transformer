"""Build rendered image datasets from collected shot rows."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from tqdm import tqdm

from xg_project.constants import DEFAULT_IMAGE_SIZE
from xg_project.rendering import create_freeze_frame_image, parse_jsonish, parse_location
from xg_project.splitting import split_by_match


def _has_freeze_frame(value: object) -> bool:
    parsed = parse_jsonish(value)
    return isinstance(parsed, list) and len(parsed) > 0


def filter_renderable_shots(
    shots: pd.DataFrame,
    exclude_penalties: bool = False,
) -> pd.DataFrame:
    """Keep shots that have location, freeze-frame data, and labels."""

    required = {"location", "freeze_frame", "is_goal", "match_id"}
    missing = required.difference(shots.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    mask = shots["location"].map(parse_location).notna()
    mask &= shots["freeze_frame"].map(_has_freeze_frame)
    mask &= shots["is_goal"].notna()
    if exclude_penalties and "shot_type" in shots.columns:
        mask &= shots["shot_type"].fillna("").str.lower() != "penalty"

    return shots.loc[mask].copy().reset_index(drop=True)


def build_image_dataset(
    shots: pd.DataFrame,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    sigma: float = 2.5,
    exclude_penalties: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Render all valid shots and return metadata plus array payloads."""

    df = filter_renderable_shots(shots, exclude_penalties=exclude_penalties)
    if df.empty:
        raise ValueError("No renderable shots found")

    df["split"] = split_by_match(df, random_state=random_state).astype(str)
    images = np.empty((len(df), *image_size, 4), dtype=np.float16)

    for idx, row in enumerate(
        tqdm(df.itertuples(index=False), total=len(df), desc="Rendering freeze frames")
    ):
        images[idx] = create_freeze_frame_image(
            freeze_frame=getattr(row, "freeze_frame"),
            shooter_location=getattr(row, "location"),
            image_size=image_size,
            sigma=sigma,
        )

    labels = df["is_goal"].astype("int8").to_numpy()
    shot_ids = df.get("shot_id", pd.Series(np.arange(len(df)))).astype(str).to_numpy()
    match_ids = df["match_id"].astype(str).to_numpy()
    splits = df["split"].astype(str).to_numpy()

    payload = {
        "images": images,
        "labels": labels,
        "shot_ids": shot_ids,
        "match_ids": match_ids,
        "splits": splits,
    }
    return df, payload


def build_image_dataset_to_npz(
    shots: pd.DataFrame,
    output_path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    sigma: float = 2.5,
    exclude_penalties: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Render shots to a compressed `.npz` without keeping all images in RAM."""

    df = filter_renderable_shots(shots, exclude_penalties=exclude_penalties)
    if df.empty:
        raise ValueError("No renderable shots found")

    df["split"] = split_by_match(df, random_state=random_state).astype(str)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=output_path.parent) as tmp_dir:
        image_path = Path(tmp_dir) / "images.dat"
        images = np.memmap(
            image_path,
            dtype=np.float16,
            mode="w+",
            shape=(len(df), *image_size, 4),
        )

        for idx, row in enumerate(
            tqdm(df.itertuples(index=False), total=len(df), desc="Rendering freeze frames")
        ):
            images[idx] = create_freeze_frame_image(
                freeze_frame=getattr(row, "freeze_frame"),
                shooter_location=getattr(row, "location"),
                image_size=image_size,
                sigma=sigma,
            )

        images.flush()
        labels = df["is_goal"].astype("int8").to_numpy()
        shot_ids = df.get("shot_id", pd.Series(np.arange(len(df)))).astype(str).to_numpy()
        match_ids = df["match_id"].astype(str).to_numpy()
        splits = df["split"].astype(str).to_numpy()

        np.savez_compressed(
            output_path,
            images=images,
            labels=labels,
            shot_ids=shot_ids,
            match_ids=match_ids,
            splits=splits,
        )

    payload_preview = {
        "images": np.asarray([create_freeze_frame_image(
            freeze_frame=df.iloc[0]["freeze_frame"],
            shooter_location=df.iloc[0]["location"],
            image_size=image_size,
            sigma=sigma,
        )], dtype=np.float16),
        "labels": labels[:1],
        "shot_ids": shot_ids[:1],
        "match_ids": match_ids[:1],
        "splits": splits[:1],
    }
    return df, payload_preview


def build_image_dataset_to_folder(
    shots: pd.DataFrame,
    output_dir: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    sigma: float = 2.5,
    exclude_penalties: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Render shots to a folder-backed dataset with mmap-friendly arrays."""

    df = filter_renderable_shots(shots, exclude_penalties=exclude_penalties)
    if df.empty:
        raise ValueError("No renderable shots found")

    df["split"] = split_by_match(df, random_state=random_state).astype(str)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = np.lib.format.open_memmap(
        output_dir / "images.npy",
        dtype=np.float16,
        mode="w+",
        shape=(len(df), *image_size, 4),
    )
    for idx, row in enumerate(
        tqdm(df.itertuples(index=False), total=len(df), desc="Rendering freeze frames")
    ):
        images[idx] = create_freeze_frame_image(
            freeze_frame=getattr(row, "freeze_frame"),
            shooter_location=getattr(row, "location"),
            image_size=image_size,
            sigma=sigma,
        )
    images.flush()

    labels = df["is_goal"].astype("int8").to_numpy()
    shot_ids = df.get("shot_id", pd.Series(np.arange(len(df)))).astype(str).to_numpy()
    match_ids = df["match_id"].astype(str).to_numpy()
    splits = df["split"].astype(str).to_numpy()
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "shot_ids.npy", shot_ids)
    np.save(output_dir / "match_ids.npy", match_ids)
    np.save(output_dir / "splits.npy", splits)

    payload_preview = {
        "images": np.asarray(images[:1]),
        "labels": labels[:1],
        "shot_ids": shot_ids[:1],
        "match_ids": match_ids[:1],
        "splits": splits[:1],
    }
    return df, payload_preview


def save_npz_dataset(payload: dict[str, np.ndarray], output_path: str | Path) -> None:
    """Save compressed image arrays."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
