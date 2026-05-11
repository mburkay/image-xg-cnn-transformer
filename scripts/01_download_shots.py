#!/usr/bin/env python3
"""Collect StatsBomb shot rows from the local open-data clone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg_project.data_collection import collect_local_open_data_shots, parse_competition_filters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-data-root", default="open-data", help="Local StatsBomb open-data root.")
    parser.add_argument("--output", default="data/raw/shots.pkl", help="Output pickle path.")
    parser.add_argument(
        "--competition",
        action="append",
        help="Optional competition filter in competition_id:season_id form. Repeatable.",
    )
    parser.add_argument("--max-matches", type=int, default=None, help="Limit matches for a smoke run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filters = parse_competition_filters(args.competition)
    shots = collect_local_open_data_shots(
        open_data_root=args.open_data_root,
        competition_filters=filters,
        max_matches=args.max_matches,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shots.to_pickle(output)
    print(f"Saved {len(shots):,} shot rows to {output}")
    if len(shots):
        print(shots[["match_id", "outcome", "is_goal", "shot_type"]].head())


if __name__ == "__main__":
    main()
