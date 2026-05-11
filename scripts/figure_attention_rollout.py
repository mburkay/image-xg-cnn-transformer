#!/usr/bin/env python3
"""Attention rollout (Abnar & Zuidema 2020) for Hybrid 128 sigma=2.0m on test shots.

For a few high-xG positive shots (true goals), runs a manual forward pass of
the Hybrid CNN+Transformer model, captures per-layer self-attention weights
from each `TransformerEncoderLayer`, computes the attention rollout as the
product of (A + I)/2 normalized matrices, and overlays the resulting
importance map on the freeze-frame input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg_project.dataset import load_dataset, split_indices  # noqa: E402
from xg_project.torch_models import HybridCNNTransformerXG  # noqa: E402

CKPT = ROOT / "outputs/models/hybrid_128_s2p0_cv_seed42/best_model.pt"
DATA = ROOT / "data/processed/freeze_frames_128_s2p0_no_penalty"
OUT = ROOT / "outputs/figures/figure_attention_rollout.png"


def manual_forward_with_attention(model: HybridCNNTransformerXG, x: torch.Tensor):
    """Replicate forward but capture attention weights from each encoder layer."""

    feat = model.features(x)  # (B, C, H', W')
    tokens = feat.flatten(2).transpose(1, 2)  # (B, T, C)
    tokens = tokens + model.position[:, : tokens.shape[1], :]

    attn_per_layer = []
    h = tokens
    for layer in model.encoder.layers:
        h_norm = layer.norm1(h)
        attn_out, attn_w = layer.self_attn(
            h_norm, h_norm, h_norm,
            need_weights=True,
            average_attn_weights=True,
        )
        h = h + attn_out
        h_norm2 = layer.norm2(h)
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h_norm2))))
        h = h + ff
        attn_per_layer.append(attn_w[0].detach())  # (T, T) for batch index 0

    pooled = model.norm(h).mean(dim=1)
    logit = model.head(pooled).squeeze(1)
    return logit, attn_per_layer


def attention_rollout(attentions: list[torch.Tensor]) -> torch.Tensor:
    n = attentions[0].shape[0]
    eye = torch.eye(n, dtype=attentions[0].dtype)
    rollout = eye.clone()
    for a in attentions:
        a_aug = (a + eye) / 2.0
        a_aug = a_aug / a_aug.sum(dim=-1, keepdim=True)
        rollout = a_aug @ rollout
    return rollout.mean(dim=0)  # (T,)


def main() -> None:
    device = torch.device("cpu")
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = HybridCNNTransformerXG(input_shape=tuple(ckpt["input_shape"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = load_dataset(DATA)
    test_idx = split_indices(data, "test")
    y_true = np.asarray(data["labels"][test_idx], dtype=int)
    positives = test_idx[y_true == 1]

    rng = np.random.default_rng(0)
    chosen = rng.choice(positives, size=4, replace=False)

    fig, axes = plt.subplots(4, 5, figsize=(18, 13))
    titles = ["Attackers", "Defenders", "Goalkeeper", "Shooter", "Attention rollout"]

    for row, idx in enumerate(chosen):
        image = np.asarray(data["images"][idx], dtype=np.float32)
        x = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0)

        with torch.no_grad():
            logit, attentions = manual_forward_with_attention(model, x)
            prob = torch.sigmoid(logit).item()

        rollout = attention_rollout(attentions)  # (T,)
        side = int(np.sqrt(rollout.shape[0]))
        attn_grid = rollout.reshape(side, side).numpy()
        attn_up = np.kron(attn_grid, np.ones((image.shape[0] // side, image.shape[1] // side)))

        composite = image[:, :, 0] + image[:, :, 1] + image[:, :, 2] + image[:, :, 3]
        composite = np.clip(composite, 0, 1)

        for c in range(4):
            ax = axes[row, c]
            ax.imshow(image[:, :, c], cmap="magma", vmin=0, vmax=1)
            if row == 0:
                ax.set_title(titles[c])
            ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row, 4]
        ax.imshow(composite, cmap="gray", alpha=0.85)
        ax.imshow(attn_up, cmap="viridis", alpha=0.55)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{titles[4]} — pred xG = {prob:.3f}", fontsize=10)
        axes[row, 0].set_ylabel(f"Test shot #{int(idx)} (goal=1)", fontsize=9)

    fig.suptitle("Hybrid 128 σ=2.0m attention rollout on 4 random test goals", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
