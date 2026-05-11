"""PyTorch model builders for xG freeze-frame images."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TorchModelConfig:
    name: str
    input_shape: tuple[int, int, int]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNXG(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 4),
        filters: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = input_shape[2]
        blocks: list[nn.Module] = []
        for idx, out_channels in enumerate(filters):
            blocks.append(ConvBlock(channels, out_channels, dropout=0.05 if idx >= 2 else 0.0))
            channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


class HybridCNNTransformerXG(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 4),
        filters: tuple[int, ...] = (32, 64, 128, 256),
        transformer_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        height, width, in_channels = input_shape
        channels = in_channels
        blocks: list[nn.Module] = []
        for idx, out_channels in enumerate(filters):
            blocks.append(ConvBlock(channels, out_channels, dropout=0.05 if idx >= 2 else 0.0))
            channels = out_channels

        reduced_h = height // (2 ** len(filters))
        reduced_w = width // (2 ** len(filters))
        if reduced_h < 1 or reduced_w < 1:
            raise ValueError("input image is too small for the CNN pooling depth")

        self.features = nn.Sequential(*blocks)
        num_tokens = reduced_h * reduced_w
        self.position = nn.Parameter(torch.zeros(1, num_tokens, channels))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Sequential(
            nn.Linear(channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position[:, : x.shape[1], :]
        x = self.encoder(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x).squeeze(1)


class ViTXG(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 4),
        patch_size: int = 16,
        embed_dim: int = 128,
        transformer_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        height, width, channels = input_shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("image height and width must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(
            channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_tokens = (height // patch_size) * (width // patch_size)
        self.position = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position[:, : x.shape[1], :]
        x = self.encoder(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x).squeeze(1)


def build_torch_model(name: str, input_shape: tuple[int, int, int]) -> nn.Module:
    normalized = name.lower().strip()
    if normalized == "cnn":
        return CNNXG(input_shape=input_shape)
    if normalized == "hybrid":
        return HybridCNNTransformerXG(input_shape=input_shape)
    if normalized == "vit":
        return ViTXG(input_shape=input_shape)
    raise ValueError(f"Unknown model {name!r}; expected cnn, hybrid, or vit")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
