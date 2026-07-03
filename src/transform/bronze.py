"""Bronze layer: land raw Jolpica-F1 API responses to disk, immutable.

The bronze layer is the reproducibility anchor for the whole pipeline. It stores
each API response *exactly as received*, wrapped with a little metadata
(ingestion timestamp, source, resolved season/round). No cleaning, no typing —
that happens downstream in silver. Because bronze holds the raw data, silver and
gold can be rebuilt at any time without re-hitting the rate-limited API.

Layout (Hive-style partitions, so Spark/Databricks can read them natively later)::

    data/bronze/
      results/season=2026/round=08.json
      qualifying/season=2026/round=08.json
      driverStandings/season=2026/round=08.json

Design decisions
----------------
- **One file per (record_type, season, round).** Makes writes idempotent
  (last-write-wins on a single file) and keeps the raw data easy to inspect.
- **Skip-if-already-landed.** ``land`` won't re-fetch a race it already has on
  disk (unless ``overwrite=True``), so backfills are incremental and cheap.
- **Resolve ``current``/``last`` to real numbers.** A response fetched via
  ``current``/``last`` is stored under its actual season/round (e.g. 2026/08),
  never under a "last" filename that would go stale.
- **Never land empty races.** A race that hasn't happened yet returns no data;
  bronze skips it, which is also how ``backfill`` knows to stop.

Usage (command line)::

    python -m src.transform.bronze --season current --round last
    python -m src.transform.bronze --record-type qualifying --season 2023 --round 1

Usage (notebook / Databricks) — import, don't run as a script::

    from src.transform.bronze import BronzeWriter

    bronze = BronzeWriter(bronze_path="data/bronze")   # or a /Volumes/... path on Databricks
    bronze.land("results", "current", "last")          # land the latest race
    bronze.backfill(["2021", "2022", "2023", "2024"])  # build a training set
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.jolpica_client import JolpicaClient

logger = logging.getLogger(__name__)

SOURCE = "jolpica-f1"

# record_type -> (client method, MRData table key, inner list key)
RECORD_TYPES = {
    "results":         ("get_race_results",    "RaceTable",      "Races"),
    "qualifying":      ("get_qualifying",       "RaceTable",      "Races"),
    "driverStandings": ("get_driver_standings", "StandingsTable", "StandingsLists"),
}


def _is_numeric(value) -> bool:
    return str(value).isdigit()


def _pad(rnd) -> str:
    return f"{int(rnd):02d}"


class BronzeWriter:
    """Fetches from the Jolpica client and lands raw responses to the bronze layer."""

    def __init__(self, bronze_path: str | Path = "data/bronze", client: JolpicaClient | None = None) -> None:
        self.bronze_path = Path(bronze_path)
        self.client = client or JolpicaClient()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def land(self, record_type: str, season, rnd, *, overwrite: bool = False) -> Path | None:
        """Fetch one endpoint and land the raw response. Returns the path, or
        ``None`` if there was no data (e.g. the race hasn't run yet)."""
        self._check_record_type(record_type)

        # For concrete season+round we can skip before spending an API call.
        if not overwrite and _is_numeric(season) and _is_numeric(rnd):
            existing = self._path(record_type, str(season), rnd)
            if existing.exists():
                logger.info("Skip (already landed): %s", existing)
                return existing

        method = getattr(self.client, RECORD_TYPES[record_type][0])
        payload = method(season, rnd)

        if not self._has_data(payload, record_type):
            logger.info("No %s data for %s/%s (race not run yet?) — skipping", record_type, season, rnd)
            return None

        r_season, r_round = self._resolve_season_round(payload, season, rnd)
        path = self._path(record_type, r_season, r_round)
        if path.exists() and not overwrite:
            logger.info("Skip (already landed): %s", path)
            return path

        record = {
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "source": SOURCE,
            "record_type": record_type,
            "season": r_season,
            "round": r_round,
            "payload": payload,
        }
        self._atomic_write(path, record)
        logger.info("Landed %s -> %s", record_type, path)
        return path

    def backfill(self, seasons, record_types=("results",), *, overwrite: bool = False) -> list[Path]:
        """Land every completed round for each season. Stops a season as soon as
        it hits a round with no data (the first race that hasn't happened),
        since rounds are chronological. The first record type acts as the gate."""
        if not record_types:
            raise ValueError("record_types must be non-empty")
        for rt in record_types:
            self._check_record_type(rt)
        gate = record_types[0]

        landed: list[Path] = []
        for season in seasons:
            season = str(season)
            for rnd in self._rounds_in_season(season):
                rnd = str(rnd)
                gate_path = self.land(gate, season, rnd, overwrite=overwrite)
                if gate_path is None:
                    logger.info("No data at %s round %s — stopping backfill for this season.", season, rnd)
                    break
                landed.append(gate_path)
                for rt in record_types[1:]:
                    p = self.land(rt, season, rnd, overwrite=overwrite)
                    if p:
                        landed.append(p)
        return landed

    def read(self, record_type: str, season=None, rnd=None) -> list[dict]:
        """Read landed bronze records back (for silver to consume)."""
        self._check_record_type(record_type)
        base = self.bronze_path / record_type
        if not base.exists():
            return []
        season_dirs = [base / f"season={season}"] if season is not None else sorted(
            d for d in base.iterdir() if d.is_dir()
        )
        records: list[dict] = []
        for sdir in season_dirs:
            if not sdir.exists():
                continue
            files = [sdir / f"round={_pad(rnd)}.json"] if rnd is not None else sorted(sdir.glob("round=*.json"))
            for fp in files:
                if fp.exists():
                    records.append(json.loads(fp.read_text()))
        return records

    def list_landed(self, record_type: str | None = None) -> list[Path]:
        """List the bronze files on disk (handy for inspection and QA)."""
        types = [record_type] if record_type else list(RECORD_TYPES)
        out: list[Path] = []
        for rt in types:
            base = self.bronze_path / rt
            if base.exists():
                out.extend(sorted(base.glob("season=*/round=*.json")))
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_record_type(record_type: str) -> None:
        if record_type not in RECORD_TYPES:
            raise ValueError(f"Unknown record_type {record_type!r}; expected one of {list(RECORD_TYPES)}")

    def _path(self, record_type: str, season: str, rnd) -> Path:
        return self.bronze_path / record_type / f"season={season}" / f"round={_pad(rnd)}.json"

    def _rounds_in_season(self, season: str) -> list[int]:
        """Round numbers scheduled in a season, from the season schedule endpoint."""
        payload = self.client.fetch(f"{season}")
        races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        return [int(r["round"]) for r in races if r.get("round")]

    @staticmethod
    def _has_data(payload: dict, record_type: str) -> bool:
        _, table_key, list_key = RECORD_TYPES[record_type]
        table = payload.get("MRData", {}).get(table_key, {})
        return bool(table.get(list_key))

    @staticmethod
    def _resolve_season_round(payload: dict, season, rnd):
        """Pull the concrete season/round from the payload so 'current'/'last'
        land under their real numbers."""
        mr = payload.get("MRData", {})
        table = mr.get("RaceTable") or mr.get("StandingsTable") or {}
        r_season = table.get("season") or (season if _is_numeric(season) else None)
        r_round = table.get("round")
        if not r_round:
            nested = table.get("Races") or table.get("StandingsLists") or []
            if nested:
                r_round = nested[0].get("round")
        if not r_round and _is_numeric(rnd):
            r_round = str(rnd)
        return str(r_season), str(r_round)

    @staticmethod
    def _atomic_write(path: Path, obj: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(obj, f, indent=2)
            os.replace(tmp, path)  # atomic on the same filesystem
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Land raw F1 data into the bronze layer.")
    parser.add_argument("--record-type", default="results", choices=list(RECORD_TYPES))
    parser.add_argument("--season", default="current")
    parser.add_argument("--round", default="last")
    parser.add_argument("--bronze-path", default="data/bronze")
    # parse_known_args so notebook kernel args (e.g. -f connection.json) don't crash it.
    args, _ = parser.parse_known_args()

    writer = BronzeWriter(bronze_path=args.bronze_path)
    path = writer.land(args.record_type, args.season, args.round)
    print(f"Landed: {path}" if path else "Nothing landed (no data for that race yet).")


if __name__ == "__main__":
    _main()
