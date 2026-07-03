"""Tests for the bronze layer. A fake client serves canned payloads — no network.

Run with:  pytest -q
"""

import json
import tempfile

import pytest

from src.transform.bronze import BronzeWriter


def race_payload(season, rnd, drivers=("VER", "HAM", "LEC")):
    """A results/qualifying-shaped MRData payload. Empty drivers => no data."""
    races = []
    if drivers:
        races = [{
            "season": str(season), "round": str(rnd), "raceName": f"Round {rnd}",
            "Results": [{"position": str(i + 1), "Driver": {"driverId": d}} for i, d in enumerate(drivers)],
        }]
    return {"MRData": {"RaceTable": {"season": str(season), "round": str(rnd) if races else "", "Races": races}}}


def schedule_payload(season, n_rounds):
    races = [{"season": str(season), "round": str(r), "raceName": f"Round {r}"} for r in range(1, n_rounds + 1)]
    return {"MRData": {"RaceTable": {"season": str(season), "Races": races}}}


class FakeClient:
    """Serves a small canned universe:
      - season 2023: 3 rounds, all with results
      - season 2026 (current): schedule has 5 rounds, but only rounds 1-2 have run
      - current/last resolves to 2026 round 2
    """
    def __init__(self):
        self.calls = []

    def get_race_results(self, season, rnd):
        self.calls.append(("results", str(season), str(rnd)))
        if season == "current" and rnd == "last":
            return race_payload("2026", "2")
        if str(season) == "2026" and int(str(rnd)) >= 3:
            return race_payload("2026", rnd, drivers=())      # not run yet -> empty
        return race_payload(season, rnd)

    def get_qualifying(self, season, rnd):
        self.calls.append(("qualifying", str(season), str(rnd)))
        return race_payload(season, rnd, drivers=("VER", "HAM"))

    def get_driver_standings(self, season, rnd):
        self.calls.append(("driverStandings", str(season), str(rnd)))
        return {"MRData": {"StandingsTable": {"season": str(season), "round": str(rnd),
                                              "StandingsLists": [{"round": str(rnd)}]}}}

    def fetch(self, endpoint):
        self.calls.append(("fetch", str(endpoint)))
        return schedule_payload(endpoint, 3 if str(endpoint) == "2023" else 5)


@pytest.fixture
def tmp_bronze():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def _writer(tmp):
    return BronzeWriter(bronze_path=tmp, client=FakeClient())


def test_land_writes_envelope_and_payload(tmp_bronze):
    w = _writer(tmp_bronze)
    path = w.land("results", "2023", "1")
    assert path.exists()
    record = json.loads(path.read_text())
    for key in ("ingested_at", "source", "record_type", "season", "round", "payload"):
        assert key in record
    assert record["source"] == "jolpica-f1"
    assert record["season"] == "2023" and record["round"] == "1"
    assert record["payload"]["MRData"]["RaceTable"]["Races"]        # raw payload preserved
    assert path.name == "round=01.json"                             # zero-padded


def test_current_last_resolves_to_real_season_round(tmp_bronze):
    w = _writer(tmp_bronze)
    path = w.land("results", "current", "last")
    # should land under the resolved 2026/round 02, not a "last" file
    assert path.name == "round=02.json"
    assert "season=2026" in str(path)
    assert json.loads(path.read_text())["round"] == "2"


def test_land_is_idempotent_and_overwrite(tmp_bronze):
    w = _writer(tmp_bronze)
    w.land("results", "2023", "1")
    n_after_first = len(w.client.calls)
    w.land("results", "2023", "1")                 # already on disk -> no API call
    assert len(w.client.calls) == n_after_first
    w.land("results", "2023", "1", overwrite=True) # overwrite -> fetches again
    assert len(w.client.calls) == n_after_first + 1


def test_land_skips_when_no_data(tmp_bronze):
    w = _writer(tmp_bronze)
    assert w.land("results", "2026", "4") is None  # round 4 hasn't run
    assert w.list_landed("results") == []


def test_backfill_stops_at_first_empty_round(tmp_bronze):
    w = _writer(tmp_bronze)
    landed = w.backfill(["2026"])                  # schedule=5 rounds, only 1-2 have run
    assert len(landed) == 2
    rounds = sorted(p.name for p in w.list_landed("results"))
    assert rounds == ["round=01.json", "round=02.json"]


def test_backfill_multiple_record_types(tmp_bronze):
    w = _writer(tmp_bronze)
    landed = w.backfill(["2023"], record_types=("results", "qualifying"))
    assert len(w.list_landed("results")) == 3
    assert len(w.list_landed("qualifying")) == 3
    assert len(landed) == 6


def test_read_returns_landed_records(tmp_bronze):
    w = _writer(tmp_bronze)
    w.backfill(["2023"])
    recs = w.read("results", "2023")
    assert len(recs) == 3
    assert [r["round"] for r in recs] == ["1", "2", "3"]
