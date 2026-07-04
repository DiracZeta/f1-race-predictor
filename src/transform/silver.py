"""Silver layer: clean, typed, one-row-per-driver-per-race tables.

Reads the raw JSON the bronze layer landed, flattens the deeply nested
Ergast/Jolpica structure, coerces types, standardizes IDs, and writes tidy
tables that the gold layer (and any ad-hoc analysis) can consume.

Design
------
The transformation logic lives in **pure functions** (``flatten_results``,
``flatten_qualifying``, ``parse_lap_time_ms``) that take a bronze record and
return plain lists of dicts — no file I/O, no Spark — so they are fast and
trivial to unit-test. ``build_silver`` is a thin wrapper that reads bronze,
applies those functions, and writes the output.

Output format defaults to Parquet (the lakehouse norm; needs pyarrow, which is
available on Databricks). Pass ``fmt="csv"`` for environments without pyarrow.

F1 is small data, so pandas is the right tool. The same pattern scales to
PySpark/Delta by swapping only the read/write helpers — the pure functions above
stay identical.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O — unit-tested directly)
# --------------------------------------------------------------------------- #
def parse_lap_time_ms(text) -> int | None:
    """Convert a lap-time string to milliseconds. '1:29.123' -> 89123.

    Returns None for missing/invalid values (None, '', '\\N')."""
    if not text or text in ("\\N", "-"):
        return None
    try:
        text = str(text).strip()
        if ":" in text:
            minutes, seconds = text.split(":")
            total_seconds = int(minutes) * 60 + float(seconds)
        else:
            total_seconds = float(text)
        return int(round(total_seconds * 1000))
    except (ValueError, TypeError):
        return None


def _to_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _races(bronze_record: dict) -> list[dict]:
    return (
        bronze_record.get("payload", {})
        .get("MRData", {})
        .get("RaceTable", {})
        .get("Races", [])
    )


def flatten_results(bronze_record: dict) -> list[dict]:
    """One row per driver per race from a bronze 'results' record."""
    rows = []
    for race in _races(bronze_record):
        season = _to_int(race.get("season"))
        rnd = _to_int(race.get("round"))
        circuit = race.get("Circuit", {})
        location = circuit.get("Location", {})
        for res in race.get("Results", []):
            driver = res.get("Driver", {})
            constructor = res.get("Constructor", {})
            position_text = res.get("positionText", "")
            fastest = res.get("FastestLap", {})
            rows.append({
                "season": season,
                "round": rnd,
                "race_name": race.get("raceName"),
                "race_date": race.get("date"),
                "circuit_id": circuit.get("circuitId"),
                "circuit_name": circuit.get("circuitName"),
                "country": location.get("country"),
                "driver_id": driver.get("driverId"),
                "driver_code": driver.get("code"),
                "driver_number": _to_int(res.get("number")),
                "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                "driver_nationality": driver.get("nationality"),
                "constructor_id": constructor.get("constructorId"),
                "constructor_name": constructor.get("name"),
                "grid": _to_int(res.get("grid")),
                # Ergast 'position' is always a number (classification order, even for DNFs);
                # 'positionText' is 'R'/'D'/etc. for cars that didn't classify.
                "position_order": _to_int(res.get("position")),
                "finish_position": _to_int(position_text) if str(position_text).isdigit() else None,
                "position_text": position_text,
                "classified": str(position_text).isdigit(),
                "points": float(res.get("points", 0) or 0),
                "laps": _to_int(res.get("laps")),
                "status": res.get("status"),
                "fastest_lap_rank": _to_int(fastest.get("rank")) if fastest else None,
            })
    return rows


def flatten_qualifying(bronze_record: dict) -> list[dict]:
    """One row per driver per race from a bronze 'qualifying' record."""
    rows = []
    for race in _races(bronze_record):
        season = _to_int(race.get("season"))
        rnd = _to_int(race.get("round"))
        for q in race.get("QualifyingResults", []):
            driver = q.get("Driver", {})
            constructor = q.get("Constructor", {})
            q1 = parse_lap_time_ms(q.get("Q1"))
            q2 = parse_lap_time_ms(q.get("Q2"))
            q3 = parse_lap_time_ms(q.get("Q3"))
            best = [t for t in (q1, q2, q3) if t is not None]
            rows.append({
                "season": season,
                "round": rnd,
                "driver_id": driver.get("driverId"),
                "driver_code": driver.get("code"),
                "constructor_id": constructor.get("constructorId"),
                "quali_position": _to_int(q.get("position")),
                "q1_ms": q1,
                "q2_ms": q2,
                "q3_ms": q3,
                "best_quali_ms": min(best) if best else None,
            })
    return rows


RESULTS_COLUMNS = [
    "season", "round", "race_name", "race_date", "circuit_id", "circuit_name", "country",
    "driver_id", "driver_code", "driver_number", "driver_name", "driver_nationality",
    "constructor_id", "constructor_name", "grid", "position_order", "finish_position",
    "position_text", "classified", "points", "laps", "status", "fastest_lap_rank",
]
QUALIFYING_COLUMNS = [
    "season", "round", "driver_id", "driver_code", "constructor_id",
    "quali_position", "q1_ms", "q2_ms", "q3_ms", "best_quali_ms",
]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def read_bronze(bronze_path, record_type: str) -> list[dict]:
    """Read every landed bronze record of a given type (filesystem only)."""
    base = Path(bronze_path) / record_type
    if not base.exists():
        return []
    records = []
    for fp in sorted(base.glob("season=*/round=*.json")):
        records.append(json.loads(fp.read_text()))
    return records


def _write_table(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported fmt: {fmt!r}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_silver(bronze_path="data/bronze", silver_path="data/silver", fmt: str = "parquet") -> dict:
    """Build the silver tables from bronze. Returns the DataFrames it wrote."""
    silver_path = Path(silver_path)

    results_rows: list[dict] = []
    for rec in read_bronze(bronze_path, "results"):
        results_rows.extend(flatten_results(rec))
    results_df = pd.DataFrame(results_rows, columns=RESULTS_COLUMNS)

    qual_rows: list[dict] = []
    for rec in read_bronze(bronze_path, "qualifying"):
        qual_rows.extend(flatten_qualifying(rec))
    qualifying_df = pd.DataFrame(qual_rows, columns=QUALIFYING_COLUMNS)

    _write_table(results_df, silver_path / f"results.{fmt}", fmt)
    _write_table(qualifying_df, silver_path / f"qualifying.{fmt}", fmt)

    logger.info("Silver: %d result rows, %d qualifying rows -> %s",
                len(results_df), len(qualifying_df), silver_path)
    return {"results": results_df, "qualifying": qualifying_df}
