"""Predict a race that hasn't happened yet.

To score an upcoming race we need one gold-style feature row per driver *before*
the race runs. Those features come from two places:

- **grid / qualifying** — from the qualifying session (fetched from the API).
  Qualifying happens the day before the race, so this is available on race eve.
- **form / season points** — from each driver's history (the silver tables).

The important trick: instead of re-deriving the form features with new code
(which risks *train/serve skew* — features computed differently at prediction
time than at training time), we append the upcoming race's rows to the historical
results and run the **same** ``engineer_features`` the gold layer uses. The
shift-based rolling then computes the upcoming race's form from prior races
exactly as it did during training. The upcoming rows have no result yet, which is
fine — a row's own result never feeds its own features.

Note: the starting grid is estimated from the qualifying order. The real grid can
differ slightly due to penalties, but qualifying order is the best pre-race signal.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ingestion.jolpica_client import JolpicaClient
from src.transform.silver import parse_lap_time_ms
from src.transform.gold import engineer_features
from src.modeling.predict import predict_race, load_model

logger = logging.getLogger(__name__)

_UPCOMING_RESULT_COLS = [
    "season", "round", "race_name", "driver_id", "driver_code",
    "constructor_id", "constructor_name", "grid", "position_order", "points",
]


def _read_silver(silver_path, name: str, fmt: str) -> pd.DataFrame:
    path = Path(silver_path) / f"{name}.{fmt}"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path)


def _parse_qualifying(payload: dict, season: int, rnd: int):
    """Turn a raw qualifying API response into (race_name, list-of-rows)."""
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return None, []
    race = races[0]
    rows = []
    for q in race.get("QualifyingResults", []):
        driver = q.get("Driver", {})
        constructor = q.get("Constructor", {})
        pos = int(q["position"]) if str(q.get("position", "")).isdigit() else None
        best = [t for t in (parse_lap_time_ms(q.get("Q1")),
                            parse_lap_time_ms(q.get("Q2")),
                            parse_lap_time_ms(q.get("Q3"))) if t is not None]
        rows.append({
            "season": season, "round": rnd, "race_name": race.get("raceName"),
            "driver_id": driver.get("driverId"), "driver_code": driver.get("code"),
            "constructor_id": constructor.get("constructorId"),
            "constructor_name": constructor.get("name"),
            "grid": pos,                 # starting grid estimated from qualifying order
            "quali_position": pos,
            "best_quali_ms": min(best) if best else None,
            "position_order": pd.NA,     # race hasn't run — unknown
            "points": 0.0,
        })
    return race.get("raceName"), rows


def assemble_upcoming_features(season: int, rnd: int, silver_path="data/silver",
                               fmt: str = "parquet", client: JolpicaClient | None = None) -> pd.DataFrame:
    """Build the gold-style feature rows for an upcoming race."""
    client = client or JolpicaClient()
    _, up_rows = _parse_qualifying(client.get_qualifying(season, rnd), season, rnd)
    if not up_rows:
        raise ValueError(
            f"Qualifying for {season} round {rnd} isn't available yet — "
            "predictions need qualifying to have happened."
        )

    up_results = pd.DataFrame([{c: r[c] for c in _UPCOMING_RESULT_COLS} for r in up_rows])
    up_quali = pd.DataFrame([{
        "season": r["season"], "round": r["round"], "driver_id": r["driver_id"],
        "quali_position": r["quali_position"], "best_quali_ms": r["best_quali_ms"],
    } for r in up_rows])

    hist_results = _read_silver(silver_path, "results", fmt)
    hist_quali = _read_silver(silver_path, "qualifying", fmt)

    all_results = pd.concat([hist_results, up_results], ignore_index=True)
    all_quali = pd.concat([hist_quali, up_quali], ignore_index=True) if not hist_quali.empty else up_quali

    gold = engineer_features(all_results, all_quali)
    upcoming = gold[(gold["season"] == season) & (gold["round"] == rnd)].reset_index(drop=True)
    if upcoming.empty:
        raise ValueError(f"Could not assemble features for {season} round {rnd}.")
    return upcoming


def predict_upcoming(season: int, rnd: int, model_path="data/models/win_model.joblib",
                     silver_path="data/silver", fmt: str = "parquet",
                     client: JolpicaClient | None = None, model=None) -> pd.DataFrame:
    """Predict the winner of an upcoming race. Returns drivers ranked by win prob."""
    features = assemble_upcoming_features(season, rnd, silver_path, fmt, client)
    model = model if model is not None else load_model(model_path)
    ranked = predict_race(model, features)
    cols = ["driver_code", "driver_id", "constructor_name", "grid",
            "quali_position", "win_probability", "predicted_winner"]
    return ranked[[c for c in cols if c in ranked.columns]]


def find_next_race(client: JolpicaClient | None = None, season="current", today=None):
    """Return (season, round, race_name) of the next race on/after `today`."""
    client = client or JolpicaClient()
    today = today or datetime.now(timezone.utc).date()
    races = client.fetch(f"{season}").get("MRData", {}).get("RaceTable", {}).get("Races", [])
    for race in races:
        try:
            race_date = datetime.strptime(race.get("date", ""), "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        if race_date >= today:
            return int(race.get("season")), int(race.get("round")), race.get("raceName")
    return None


def predict_next_race(model_path="data/models/win_model.joblib", silver_path="data/silver",
                      fmt: str = "parquet", client: JolpicaClient | None = None, model=None) -> pd.DataFrame:
    """Find the next scheduled race and predict it (needs its qualifying to be out)."""
    client = client or JolpicaClient()
    nxt = find_next_race(client)
    if nxt is None:
        raise ValueError("No upcoming race found on the schedule.")
    season, rnd, name = nxt
    logger.info("Next race: %s (%s round %s)", name, season, rnd)
    return predict_upcoming(season, rnd, model_path, silver_path, fmt, client, model)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Predict an upcoming race winner.")
    parser.add_argument("--season", type=int)
    parser.add_argument("--round", type=int)
    parser.add_argument("--next", action="store_true", help="predict the next scheduled race")
    parser.add_argument("--model-path", default="data/models/win_model.joblib")
    parser.add_argument("--silver-path", default="data/silver")
    parser.add_argument("--fmt", default="parquet")
    args, _ = parser.parse_known_args()

    if args.next or args.season is None:
        ranked = predict_next_race(args.model_path, args.silver_path, args.fmt)
    else:
        ranked = predict_upcoming(args.season, args.round, args.model_path, args.silver_path, args.fmt)
    print("\nPredicted order:\n")
    print(ranked.to_string(index=False))


if __name__ == "__main__":
    _main()
