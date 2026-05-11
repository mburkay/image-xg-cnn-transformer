#!/usr/bin/env python3
"""Train an xG model from rendered freeze-frame images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg_project.torch_training import train_torch_from_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/freeze_frames.npz", help="Input npz dataset.")
    parser.add_argument("--model", choices=["cnn", "hybrid", "vit"], default="hybrid")
    parser.add_argument("--output-dir", default="outputs/models/hybrid")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto", help="torch device: auto, cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_path = train_torch_from_npz(
        data_path=args.data,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        augment=args.augment,
        seed=args.seed,
        patience=args.patience,
        device=args.device,
    )
    print(f"Best model saved to {best_path}")


if __name__ == "__main__":
    main()
