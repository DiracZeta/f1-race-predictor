"""Predict the winner of a race.

Loads the trained model, scores every driver entered in a race, and ranks them
by win probability. The top of the ranking is the predicted winner.

To predict a *future* race you need that race's feature row assembled first:
grid and qualifying come from the just-completed qualifying session, and the
form/points features come from the drivers' history (the gold layer). Once those
rows exist, ``predict_race`` ranks them. ``predict_from_gold`` is a convenience
for scoring a race that's already in the gold table (great for backtesting: score
a past race and compare the prediction to what actually happened).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from joblib import load

from src.modeling.train import FEATURES, win_probability, _read_gold

logger = logging.getLogger(__name__)


def load_model(model_path="data/models/win_model.joblib"):
    return load(model_path)


def predict_race(model, features_df: pd.DataFrame) -> pd.DataFrame:
    """Rank the drivers in a single race by win probability."""
    out = features_df.copy()
    out["win_probability"] = win_probability(model, out)
    out = out.sort_values("win_probability", ascending=False).reset_index(drop=True)
    out["predicted_winner"] = False
    if len(out):
        out.loc[0, "predicted_winner"] = True
    return out


def predict_from_gold(season, rnd, model_path="data/models/win_model.joblib",
                      gold_path="data/gold", fmt: str = "parquet") -> pd.DataFrame:
    """Score a race that already exists in the gold table."""
    df = _read_gold(gold_path, fmt)
    race = df[(df["season"] == season) & (df["round"] == rnd)]
    if race.empty:
        raise ValueError(f"No race {season}/round {rnd} found in the gold table.")
    ranked = predict_race(load_model(model_path), race)
    cols = ["driver_code", "driver_id", "grid", "win_probability", "predicted_winner"]
    return ranked[[c for c in cols if c in ranked.columns]]


def _main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Predict a race winner from the gold table.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--model-path", default="data/models/win_model.joblib")
    parser.add_argument("--gold-path", default="data/gold")
    parser.add_argument("--fmt", default="parquet")
    args, _ = parser.parse_known_args()

    ranked = predict_from_gold(args.season, args.round, args.model_path, args.gold_path, args.fmt)
    print(f"\nPredicted order — {args.season} round {args.round}:\n")
    print(ranked.to_string(index=False))


if __name__ == "__main__":
    _main()
