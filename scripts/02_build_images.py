#!/usr/bin/env python3
"""Render shot freeze frames as model-ready image arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg_project.constants import PITCH_LENGTH
from xg_project.image_dataset import (
    build_image_dataset,
    build_image_dataset_to_folder,
    build_image_dataset_to_npz,
    save_npz_dataset,
)
from xg_project.splitting import split_summary
from xg_project.visualization import plot_channel_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", default="data/raw/shots.pkl", help="Input shots pickle.")
    parser.add_argument("--output", default="data/processed/freeze_frames.npz", help="Output npz path or folder.")
    parser.add_argument(
        "--storage",
        choices=["npz", "folder"],
        default="npz",
        help="Use folder for mmap-friendly large datasets.",
    )
    parser.add_argument(
        "--metadata-output",
        default="data/processed/shots_with_splits.pkl",
        help="Output metadata pickle.",
    )
    parser.add_argument("--height", type=int, default=224, help="Rendered image height.")
    parser.add_argument("--width", type=int, default=224, help="Rendered image width.")
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.5,
        help="Gaussian blob sigma in pixels. Ignored when --sigma-meters is set.",
    )
    parser.add_argument(
        "--sigma-meters",
        type=float,
        default=None,
        help=(
            "Gaussian blob sigma in StatsBomb pitch-coordinate units (a 120 x 80 "
            "frame, NOT literal meters; 1 pitch unit ~ 0.875 m on a UEFA-standard "
            "pitch). Flag name kept for backwards compatibility. When set, sigma "
            "is rescaled to pixels using sigma_px = sigma_pu * width / "
            "PITCH_LENGTH so the blob covers the same physical area regardless "
            "of image resolution."
        ),
    )
    parser.add_argument(
        "--exclude-penalties",
        action="store_true",
        help="Exclude penalty shots from the dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split assignment.")
    parser.add_argument(
        "--preview-output",
        default="outputs/figures/freeze_frame_preview.png",
        help="Preview image path for visual QA.",
    )
    parser.add_argument(
        "--stream-to-disk",
        action="store_true",
        help="Render through a memmap so large datasets do not stay fully in RAM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shots = pd.read_pickle(args.shots)

    if args.sigma_meters is not None:
        sigma_px = args.sigma_meters * args.width / PITCH_LENGTH
        print(
            f"sigma={args.sigma_meters:.4f} pitch units -> sigma_px={sigma_px:.4f} "
            f"(width={args.width}, PITCH_LENGTH={PITCH_LENGTH}; pitch units are "
            f"StatsBomb's 120x80 frame, not literal meters)"
        )
    else:
        sigma_px = args.sigma
        print(
            f"sigma_px={sigma_px:.4f} (~{sigma_px / args.width * PITCH_LENGTH:.3f} "
            f"pitch units on pitch length axis at width={args.width})"
        )

    if args.storage == "folder":
        metadata, payload = build_image_dataset_to_folder(
            shots,
            output_dir=args.output,
            image_size=(args.height, args.width),
            sigma=sigma_px,
            exclude_penalties=args.exclude_penalties,
            random_state=args.seed,
        )
    elif args.stream_to_disk:
        metadata, payload = build_image_dataset_to_npz(
            shots,
            output_path=args.output,
            image_size=(args.height, args.width),
            sigma=sigma_px,
            exclude_penalties=args.exclude_penalties,
            random_state=args.seed,
        )
    else:
        metadata, payload = build_image_dataset(
            shots,
            image_size=(args.height, args.width),
            sigma=sigma_px,
            exclude_penalties=args.exclude_penalties,
            random_state=args.seed,
        )
        save_npz_dataset(payload, args.output)

    metadata_output = Path(args.metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_pickle(metadata_output)

    plot_channel_preview(payload["images"][0].astype("float32"), args.preview_output)
    print(f"Saved {len(metadata):,} rendered shots to {args.output}")
    print(f"Saved metadata to {metadata_output}")
    print(split_summary(metadata).to_string(index=False))


if __name__ == "__main__":
    main()
