"""Train a race-winner classifier on the gold feature table.

The model predicts, for each driver in a race, the probability that they win.
To predict a race we score every driver and take the highest probability.

Two ideas matter here and both are worth being able to explain:

1. **Time-based split.** We train on earlier seasons and test on a later one —
   never a random split. A random split would let the model peek at the future,
   which you'd never have in reality. This mirrors how the model is actually used.
2. **Top-1 accuracy.** Raw classification accuracy is misleading here (predicting
   "nobody wins" scores ~95% because only one of ~20 drivers wins). The honest
   metric is: in what fraction of races is our highest-probability driver the one
   who actually won?
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

FEATURES = [
    "grid",
    "quali_position",
    "best_quali_ms",
    "driver_form_last3",
    "driver_season_points_before",
    "constructor_form_last3",
]
TARGET = "won"


def build_model() -> Pipeline:
    """Impute missing values (early-season form / missing qualifying are NaN),
    then a random forest weighted to handle the win/no-win class imbalance."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def win_probability(model, features: pd.DataFrame):
    """P(win) for each row, robust to a model that only saw one class."""
    proba = model.predict_proba(features[FEATURES])
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, 0] * 0.0  # positive class never seen -> all zeros


def evaluate_top1(model, test_df: pd.DataFrame) -> float:
    """Fraction of races where the highest-probability driver actually won."""
    scored = test_df[["season", "round", TARGET]].copy()
    scored["p"] = win_probability(model, test_df)
    hits = races = 0
    for _, group in scored.groupby(["season", "round"]):
        races += 1
        if group.loc[group["p"].idxmax(), TARGET] == 1:
            hits += 1
    return hits / races if races else float("nan")


def _time_split(df: pd.DataFrame, test_season):
    seasons = sorted(df["season"].dropna().unique())
    if test_season is None:
        if len(seasons) >= 2:
            test_season = seasons[-1]
        else:  # single season: hold out the later rounds
            rounds = sorted(df["round"].dropna().unique())
            if len(rounds) < 2:
                return df, df.iloc[0:0]
            cut = rounds[int(len(rounds) * 0.7)]
            return df[df["round"] < cut], df[df["round"] >= cut]
    return df[df["season"] != test_season], df[df["season"] == test_season]


def _read_gold(gold_path, fmt: str) -> pd.DataFrame:
    path = Path(gold_path) / f"race_features.{fmt}"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path)


def train(gold_path="data/gold", model_path="data/models/win_model.joblib",
          test_season=None, fmt: str = "parquet") -> dict:
    """Train, evaluate on a held-out season, then save a model fit on all data."""
    df = _read_gold(gold_path, fmt)
    if df.empty:
        raise ValueError("Gold table is empty — run the bronze/silver/gold pipeline first.")

    train_df, test_df = _time_split(df, test_season)
    metrics = {"n_train": int(len(train_df)), "n_test": int(len(test_df))}

    model = build_model().fit(train_df[FEATURES], train_df[TARGET])
    if not test_df.empty:
        metrics["top1_accuracy"] = evaluate_top1(model, test_df)
        try:
            metrics["roc_auc"] = float(roc_auc_score(test_df[TARGET], win_probability(model, test_df)))
        except ValueError:
            pass  # test set had only one class

    # Final model: refit on ALL available history so predictions use everything.
    final_model = clone(model).fit(df[FEATURES], df[TARGET])
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(final_model, model_path)

    logger.info("Trained on %d rows; test top-1 accuracy: %s. Saved -> %s",
                metrics["n_train"], metrics.get("top1_accuracy", "n/a"), model_path)
    return metrics
