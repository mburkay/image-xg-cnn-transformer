#!/usr/bin/env python3
"""Diagnostics for the rendering-protocol audit (see paper Section 3.2).

Check 1: Penalty filter — count `shot_type == "Penalty"` rows.
Check 2: Gaussian sigma scaling — render same shot at 64/128 fixed sigma vs scaled sigma.
Check 3: Architecture — smoke-test CNN and Hybrid forward at 64 and 128.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg_project.image_dataset import filter_renderable_shots
from xg_project.rendering import create_freeze_frame_image
from xg_project.torch_models import CNNXG, HybridCNNTransformerXG, count_parameters


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# -----------------------------------------------------------------------------
# Check 1 — Penalty filter
# -----------------------------------------------------------------------------
section("Check 1 — Penalty filter")

shots = pd.read_pickle(ROOT / "data/raw/shots.pkl")
print(f"Total raw shots: {len(shots):,}")
print(f"Columns: {list(shots.columns)}")

if "shot_type" in shots.columns:
    print("\nshot_type top-10 value_counts (raw):")
    print(shots["shot_type"].fillna("<NaN>").value_counts().head(10).to_string())
    raw_penalty = (shots["shot_type"].fillna("").str.lower() == "penalty").sum()
    print(f"\nRaw penalty count via shot_type lowercase match: {raw_penalty:,}")

renderable = filter_renderable_shots(shots, exclude_penalties=False)
print(f"\nRenderable shots (no penalty filter): {len(renderable):,}")
if "shot_type" in renderable.columns:
    print("Renderable shot_type top-10:")
    print(renderable["shot_type"].fillna("<NaN>").value_counts().head(10).to_string())
    rend_penalty = (renderable["shot_type"].fillna("").str.lower() == "penalty").sum()
    print(f"Renderable penalties: {rend_penalty:,}")

renderable_no_pen = filter_renderable_shots(shots, exclude_penalties=True)
print(f"Renderable shots (with penalty filter): {len(renderable_no_pen):,}")
print(f"Difference (penalties removed): {len(renderable) - len(renderable_no_pen):,}")

# Cross-check existing processed metadata files.
for tag in ["shots_with_splits_64.pkl", "shots_with_splits_128_no_penalty.pkl"]:
    path = ROOT / "data/processed" / tag
    if path.exists():
        meta = pd.read_pickle(path)
        st_col = meta.get("shot_type")
        if st_col is not None:
            penalties_in_meta = (st_col.fillna("").str.lower() == "penalty").sum()
        else:
            penalties_in_meta = "shot_type column missing"
        print(f"{tag}: rows={len(meta):,}, penalties={penalties_in_meta}")

# -----------------------------------------------------------------------------
# Check 2 — Gaussian sigma scaling
# -----------------------------------------------------------------------------
section("Check 2 — Gaussian sigma scaling")

sample = renderable.iloc[0]
freeze_frame = sample["freeze_frame"]
shooter_loc = sample["location"]

# Three renders of the SAME shot:
img_64_s25 = create_freeze_frame_image(freeze_frame, shooter_loc, image_size=(64, 64), sigma=2.5)
img_128_s25 = create_freeze_frame_image(freeze_frame, shooter_loc, image_size=(128, 128), sigma=2.5)
# Pitch-aligned sigma: at 64 px width, 2.5 px = 2.5/64 of pitch length.
# To preserve that on 128 px, sigma should be 2.5 * (128 / 64) = 5.0.
img_128_s5 = create_freeze_frame_image(freeze_frame, shooter_loc, image_size=(128, 128), sigma=5.0)

print(f"Same shot rendered at three settings.")
print(f"  64x64,  sigma=2.5 px  (~{2.5/64*120:.2f} pu on pitch length axis)")
print(f"  128x128, sigma=2.5 px (~{2.5/128*120:.2f} pu on pitch length axis)  <- mismatched protocol")
print(f"  128x128, sigma=5.0 px (~{5.0/128*120:.2f} pu on pitch length axis)  <- pitch-equivalent candidate")

# Sum over each channel as a coarse "energy" measure of the rendered signal.
def chan_energy(img: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(img[:, :, c].sum()) for c in range(4))


print("\nPer-channel sum (energy) — should be roughly comparable across sizes:")
print(f"  64x64,  s=2.5: {chan_energy(img_64_s25)}")
print(f"  128x128,s=2.5: {chan_energy(img_128_s25)}  <- much smaller than 64x64? then blobs shrank")
print(f"  128x128,s=5.0: {chan_energy(img_128_s5)}")

# Save side-by-side comparison.
fig_path = ROOT / "outputs/figures/sigma_protocol_comparison.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
titles = ["Attackers", "Defenders", "Goalkeeper", "Shooter"]
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
rows = [
    ("64x64, sigma=2.5", img_64_s25),
    ("128x128, sigma=2.5 (MISMATCHED)", img_128_s25),
    ("128x128, sigma=5.0 (PITCH-EQUIVALENT)", img_128_s5),
]
for r, (label, img) in enumerate(rows):
    for c in range(4):
        ax = axes[r, c]
        ax.imshow(img[:, :, c], cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{label} | {titles[c]}", fontsize=9)
        ax.axis("off")
fig.tight_layout()
fig.savefig(fig_path, dpi=140)
plt.close(fig)
print(f"\nSaved diagnostic figure: {fig_path}")

# -----------------------------------------------------------------------------
# Check 3 — Architecture adapts to input size
# -----------------------------------------------------------------------------
section("Check 3 — Architecture forward smoke test")

torch.manual_seed(0)
for h, w in [(64, 64), (128, 128)]:
    cnn = CNNXG(input_shape=(h, w, 4))
    hyb = HybridCNNTransformerXG(input_shape=(h, w, 4))
    x = torch.randn(2, 4, h, w)
    with torch.no_grad():
        y_cnn = cnn(x)
        y_hyb = hyb(x)
    print(
        f"input {h}x{w}: "
        f"CNN params={count_parameters(cnn):,} out={tuple(y_cnn.shape)} | "
        f"Hybrid params={count_parameters(hyb):,} out={tuple(y_hyb.shape)}"
    )

print("\nDone.")
