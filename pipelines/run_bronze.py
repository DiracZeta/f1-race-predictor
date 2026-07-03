"""Bronze stage entry point — run by the 'bronze_ingest' task in the Databricks job.

Lands the most recent completed race, then (idempotently) backfills any historical
seasons passed in. Safe to re-run: bronze skips races already on disk.

Runs the same whether launched by Databricks (spark_python_task) or locally:
    python pipelines/run_bronze.py --bronze-path data/bronze --seasons 2023,2024
"""
import argparse
import logging
import os
import sys

# Make `import src...` work regardless of where this runs from. On serverless
# Databricks `__file__` may be undefined, so fall back to searching upward from
# the current working directory for the repo root (the folder containing `src`).
def _add_repo_root_to_path():
    candidates = []
    try:
        candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except NameError:
        pass  # __file__ not defined (e.g. serverless / notebook execution)
    here = os.path.abspath(os.getcwd())
    while True:
        candidates.append(here)
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    for path in candidates:
        if os.path.isdir(os.path.join(path, "src")):
            if path not in sys.path:
                sys.path.insert(0, path)
            return
    # Fallback: at least add the first candidate so imports have a chance.
    if candidates and candidates[0] not in sys.path:
        sys.path.insert(0, candidates[0])

_add_repo_root_to_path()

from src.transform.bronze import BronzeWriter  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Bronze stage: land raw F1 data.")
    parser.add_argument("--bronze-path", default="data/bronze")
    parser.add_argument("--seasons", default="", help="comma-separated seasons to backfill, e.g. 2023,2024")
    parser.add_argument("--record-types", default="results,qualifying")
    args, _ = parser.parse_known_args()

    record_types = tuple(r.strip() for r in args.record_types.split(",") if r.strip())
    writer = BronzeWriter(bronze_path=args.bronze_path)

    # 1) always pick up the latest completed race
    writer.land(record_types[0], "current", "last")

    # 2) backfill historical seasons (cheap after the first run — already-landed races are skipped)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    if seasons:
        writer.backfill(seasons, record_types=record_types)

    logging.getLogger("run_bronze").info("Bronze stage complete.")


if __name__ == "__main__":
    main()
