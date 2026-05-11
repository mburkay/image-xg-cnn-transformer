"""StatsBomb open-data collection helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class CompetitionFilter:
    """A StatsBomb competition/season pair."""

    competition_id: int
    season_id: int


def _name(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("name")
    if isinstance(value, list):
        return None
    return None if pd.isna(value) else str(value)


def _extract_shot_field(shot: object, field: str) -> object:
    if not isinstance(shot, dict):
        return None
    return shot.get(field)


def _extract_nested_name(shot: object, field: str) -> str | None:
    return _name(_extract_shot_field(shot, field))


def _event_type_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("name", ""))
    return str(value)


def _match_metadata(match: dict[str, Any]) -> dict[str, Any]:
    competition = match.get("competition") or {}
    season = match.get("season") or {}
    home_team = match.get("home_team") or {}
    away_team = match.get("away_team") or {}
    return {
        "match_id": match.get("match_id"),
        "competition_id": competition.get("competition_id"),
        "season_id": season.get("season_id"),
        "competition_name": competition.get("competition_name"),
        "season_name": season.get("season_name"),
        "match_date": match.get("match_date"),
        "home_team": home_team.get("home_team_name"),
        "away_team": away_team.get("away_team_name"),
        "home_score": match.get("home_score"),
        "away_score": match.get("away_score"),
        "match_week": match.get("match_week"),
    }


def load_local_match_metadata(open_data_root: str | Path) -> dict[int, dict[str, Any]]:
    """Load match metadata from a local StatsBomb open-data clone."""

    root = Path(open_data_root)
    matches_dir = root / "data" / "matches"
    if not matches_dir.exists():
        raise FileNotFoundError(f"StatsBomb matches directory not found: {matches_dir}")

    metadata: dict[int, dict[str, Any]] = {}
    for path in sorted(matches_dir.glob("*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            matches = json.load(handle)
        for match in matches:
            match_id = match.get("match_id")
            if match_id is None:
                continue
            metadata[int(match_id)] = _match_metadata(match)
    return metadata


def _competition_allowed(
    metadata: dict[str, Any],
    requested: set[tuple[int, int]] | None,
) -> bool:
    if requested is None:
        return True
    competition_id = metadata.get("competition_id")
    season_id = metadata.get("season_id")
    if competition_id is None or season_id is None:
        return False
    return (int(competition_id), int(season_id)) in requested


def _shot_row_from_event(
    event: dict[str, Any],
    match_id: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    shot = event.get("shot") or {}
    outcome = _extract_nested_name(shot, "outcome")
    return {
        "shot_id": event.get("id"),
        "match_id": match_id,
        "competition_id": metadata.get("competition_id"),
        "season_id": metadata.get("season_id"),
        "competition_name": metadata.get("competition_name"),
        "season_name": metadata.get("season_name"),
        "match_date": metadata.get("match_date"),
        "home_team": metadata.get("home_team"),
        "away_team": metadata.get("away_team"),
        "home_score": metadata.get("home_score"),
        "away_score": metadata.get("away_score"),
        "period": event.get("period"),
        "minute": event.get("minute"),
        "second": event.get("second"),
        "team": _name(event.get("team")),
        "player": _name(event.get("player")),
        "location": event.get("location"),
        "freeze_frame": _extract_shot_field(shot, "freeze_frame"),
        "statsbomb_xg": _extract_shot_field(shot, "statsbomb_xg"),
        "outcome": outcome,
        "is_goal": outcome == "Goal",
        "shot_type": _extract_nested_name(shot, "type"),
        "body_part": _extract_nested_name(shot, "body_part"),
        "technique": _extract_nested_name(shot, "technique"),
        "play_pattern": _name(event.get("play_pattern")),
    }


def collect_local_open_data_shots(
    open_data_root: str | Path = "open-data",
    competition_filters: Iterable[CompetitionFilter] | None = None,
    max_matches: int | None = None,
) -> pd.DataFrame:
    """Collect shot rows from a local StatsBomb open-data clone."""

    root = Path(open_data_root)
    events_dir = root / "data" / "events"
    if not events_dir.exists():
        raise FileNotFoundError(f"StatsBomb events directory not found: {events_dir}")

    requested = (
        {(item.competition_id, item.season_id) for item in competition_filters}
        if competition_filters
        else None
    )
    match_metadata = load_local_match_metadata(root)
    rows: list[dict[str, Any]] = []
    seen_matches = 0

    for path in sorted(events_dir.glob("*.json")):
        match_id = int(path.stem)
        metadata = match_metadata.get(match_id, {"match_id": match_id})
        if not _competition_allowed(metadata, requested):
            continue
        if max_matches is not None and seen_matches >= max_matches:
            break

        with path.open("r", encoding="utf-8") as handle:
            events = json.load(handle)
        seen_matches += 1

        for event in events:
            if _event_type_name(event.get("type")) != "Shot":
                continue
            rows.append(_shot_row_from_event(event, match_id, metadata))

    return pd.DataFrame(rows)


def parse_competition_filters(values: list[str] | None) -> list[CompetitionFilter] | None:
    """Parse CLI values in `competition_id:season_id` form."""

    if not values:
        return None

    filters: list[CompetitionFilter] = []
    for value in values:
        try:
            competition_id, season_id = value.split(":", maxsplit=1)
            filters.append(CompetitionFilter(int(competition_id), int(season_id)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid competition filter {value!r}; use competition_id:season_id"
            ) from exc
    return filters
