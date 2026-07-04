"""Gold layer: model-ready race feature tables.

Reads the silver results + qualifying tables and engineers one row per driver
per race with the features a model needs to predict the outcome, plus the
prediction targets (won / podium).

The core work is in ``engineer_features`` — a pure pandas function that's unit
tested directly. ``build_gold`` is the thin read/write wrapper.

Leakage safety
--------------
Every "form" feature is computed from races **before** the current one (via a
shift), so a row never sees its own result. Get this wrong and a model looks
brilliant in testing and useless in reality — so it's the most important detail
in the file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.transform.silver import _write_table  # reuse the format-aware writer

logger = logging.getLogger(__name__)

GOLD_COLUMNS = [
    "season", "round", "race_name", "driver_id", "driver_code",
    "constructor_id", "constructor_name",
    "grid", "quali_position", "best_quali_ms",
    "driver_form_last3", "driver_season_points_before", "constructor_form_last3",
    "won", "podium",
]


def engineer_features(results_df: pd.DataFrame, qualifying_df: pd.DataFrame) -> pd.DataFrame:
    """Build the gold feature table from silver results + qualifying."""
    if results_df.empty:
        return pd.DataFrame(columns=GOLD_COLUMNS)

    df = results_df.copy()

    # Sort by (season, round, driver_id): this puts every driver's rows AND every
    # constructor's rows in chronological order, so the grouped shifts below are correct.
    df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)

    # --- driver recent form: mean finishing position over the previous 3 races ---
    df["driver_form_last3"] = (
        df.groupby("driver_id")["position_order"]
          .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    # --- driver points accumulated earlier this season (before the current race) ---
    df["driver_season_points_before"] = (
        df.groupby(["driver_id", "season"])["points"]
          .transform(lambda s: s.shift(1).cumsum())
          .fillna(0.0)
    )

    # --- constructor recent form: mean finishing position of the team's cars, last 3 ---
    df["constructor_form_last3"] = (
        df.groupby("constructor_id")["position_order"]
          .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    # --- bring in qualifying (left join; some races have no qualifying data) ---
    if qualifying_df is not None and not qualifying_df.empty:
        q = qualifying_df[["season", "round", "driver_id", "quali_position", "best_quali_ms"]]
        df = df.merge(q, on=["season", "round", "driver_id"], how="left")
    else:
        df["quali_position"] = pd.NA
        df["best_quali_ms"] = pd.NA

    # --- targets ---
    df["won"] = (df["position_order"] == 1).astype(int)
    df["podium"] = (df["position_order"] <= 3).astype(int)

    return df[GOLD_COLUMNS].reset_index(drop=True)


def _read_table(path: Path, fmt: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported fmt: {fmt!r}")


def build_gold(silver_path="data/silver", gold_path="data/gold", fmt: str = "parquet") -> pd.DataFrame:
    """Build the gold feature table from silver. Returns the DataFrame it wrote."""
    silver_path = Path(silver_path)
    gold_path = Path(gold_path)

    results_df = _read_table(silver_path / f"results.{fmt}", fmt)
    qualifying_df = _read_table(silver_path / f"qualifying.{fmt}", fmt)

    gold_df = engineer_features(results_df, qualifying_df)
    _write_table(gold_df, gold_path / f"race_features.{fmt}", fmt)

    logger.info("Gold: %d feature rows -> %s", len(gold_df), gold_path)
    return gold_df
