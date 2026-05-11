#!/usr/bin/env python3
"""Evaluate a trained xG model on a held-out split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg_project.torch_evaluation import evaluate_torch_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/freeze_frames.npz", help="Input npz dataset.")
    parser.add_argument("--model-path", required=True, help="Saved model path.")
    parser.add_argument("--output-dir", default="outputs/models/hybrid/eval")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="torch device: auto, cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_torch_model(
        data_path=args.data,
        model_path=args.model_path,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=args.device,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
