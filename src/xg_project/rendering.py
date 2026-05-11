"""Render StatsBomb freeze frames as multi-channel top-down images."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from typing import Any

import numpy as np

from xg_project.constants import DEFAULT_IMAGE_SIZE, PITCH_LENGTH, PITCH_WIDTH


def parse_jsonish(value: Any) -> Any:
    """Return Python data from a nested object or a stringified object."""

    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return None
    return value


def parse_location(value: Any) -> tuple[float, float] | None:
    """Parse a StatsBomb location into an `(x, y)` tuple."""

    value = parse_jsonish(value)
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return None

    values = list(value)
    if len(values) < 2:
        return None

    try:
        x = float(values[0])
        y = float(values[1])
    except (TypeError, ValueError):
        return None

    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def should_flip_attack(shooter_location: tuple[float, float]) -> bool:
    """Return true when a shot appears oriented toward the left-side goal."""

    return shooter_location[0] < (PITCH_LENGTH / 2.0)


def normalize_location(
    location: tuple[float, float],
    flip_attack: bool,
    mirror_y_when_flipping: bool = True,
) -> tuple[float, float]:
    """Normalize coordinates so the attack is toward the right-side goal."""

    x, y = location
    if not flip_attack:
        return x, y

    y_norm = PITCH_WIDTH - y if mirror_y_when_flipping else y
    return PITCH_LENGTH - x, y_norm


def location_to_pixel(
    location: tuple[float, float],
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[float, float]:
    """Map StatsBomb 120x80 coordinates to image `(col, row)` pixels."""

    height, width = image_size
    x, y = location
    x = np.clip(x, 0.0, PITCH_LENGTH)
    y = np.clip(y, 0.0, PITCH_WIDTH)
    col = (x / PITCH_LENGTH) * (width - 1)
    row = (y / PITCH_WIDTH) * (height - 1)
    return col, row


def _nested_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name", ""))
    if value is None:
        return ""
    return str(value)


def _is_goalkeeper(player: dict[str, Any]) -> bool:
    if bool(player.get("keeper")):
        return True
    position = _nested_name(player.get("position")).lower()
    return "goalkeeper" in position or position == "keeper"


def _is_actor(player: dict[str, Any]) -> bool:
    return bool(player.get("actor")) or bool(player.get("shooter"))


def add_gaussian_blob(
    channel: np.ndarray,
    col: float,
    row: float,
    sigma: float,
    amplitude: float = 1.0,
) -> None:
    """Add a clipped Gaussian blob to one image channel in place."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    height, width = channel.shape
    radius = max(1, int(np.ceil(3 * sigma)))
    col_i = int(round(col))
    row_i = int(round(row))

    x0 = max(0, col_i - radius)
    x1 = min(width - 1, col_i + radius)
    y0 = max(0, row_i - radius)
    y1 = min(height - 1, row_i + radius)
    if x0 > x1 or y0 > y1:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    blob = amplitude * np.exp(-((grid_x - col) ** 2 + (grid_y - row) ** 2) / (2 * sigma**2))
    channel[y0 : y1 + 1, x0 : x1 + 1] += blob.astype(channel.dtype, copy=False)


def create_freeze_frame_image(
    freeze_frame: Any,
    shooter_location: Any,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    sigma: float = 2.5,
    normalize_attack_direction: bool = True,
    mirror_y_when_flipping: bool = True,
    clip: bool = True,
) -> np.ndarray:
    """Create a 4-channel image: attackers, defenders, goalkeeper, shooter."""

    parsed_shooter = parse_location(shooter_location)
    if parsed_shooter is None:
        raise ValueError("shooter_location must contain at least x and y")

    frame = parse_jsonish(freeze_frame)
    if frame is None:
        frame = []
    if not isinstance(frame, list):
        raise ValueError("freeze_frame must be a list of player dictionaries")

    flip_attack = normalize_attack_direction and should_flip_attack(parsed_shooter)
    image = np.zeros((*image_size, 4), dtype=np.float32)

    for player in frame:
        if not isinstance(player, dict):
            continue
        location = parse_location(player.get("location"))
        if location is None:
            continue

        location = normalize_location(location, flip_attack, mirror_y_when_flipping)
        col, row = location_to_pixel(location, image_size)

        if _is_actor(player):
            channel_idx = 3
        elif _is_goalkeeper(player):
            channel_idx = 2
        elif bool(player.get("teammate")):
            channel_idx = 0
        else:
            channel_idx = 1

        add_gaussian_blob(image[:, :, channel_idx], col, row, sigma=sigma)

    shooter_norm = normalize_location(parsed_shooter, flip_attack, mirror_y_when_flipping)
    shooter_col, shooter_row = location_to_pixel(shooter_norm, image_size)
    add_gaussian_blob(image[:, :, 3], shooter_col, shooter_row, sigma=sigma)

    if clip:
        np.clip(image, 0.0, 1.0, out=image)
    return image

