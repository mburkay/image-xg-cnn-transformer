"""Project-wide constants.

PITCH_LENGTH and PITCH_WIDTH are in StatsBomb pitch-coordinate units (120 × 80),
not literal meters. On a UEFA-standard 105 × 68 m pitch, 1 pitch unit ≈ 0.875 m
on the long axis and ≈ 0.85 m on the short axis. The codebase and the
`--sigma-meters` flag retain the legacy "meters" name for backwards
compatibility; paper and documentation refer to these as "pitch units".
"""

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_SIZE = (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
CHANNELS = ("attackers", "defenders", "goalkeeper", "shooter")

