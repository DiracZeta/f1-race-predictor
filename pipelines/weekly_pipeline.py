"""Local end-to-end runner: bronze -> silver -> gold, in order.

On Databricks these stages run as three separate, dependent job tasks (see
databricks.yml). This script is a convenience for running the whole flow on your
own machine. Run from the repo root:  python pipelines/weekly_pipeline.py
"""
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.transform.bronze import BronzeWriter          # noqa: E402
from src.transform import silver, gold                 # noqa: E402


def run(bronze_path="data/bronze", seasons=("2024",), record_types=("results", "qualifying")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("pipeline")

    writer = BronzeWriter(bronze_path=bronze_path)
    writer.land(record_types[0], "current", "last")
    if seasons:
        writer.backfill(list(seasons), record_types=record_types)

    for name, fn in (("silver", silver.build_silver), ("gold", gold.build_gold)):
        try:
            fn()
        except NotImplementedError:
            log.warning("%s stage not implemented yet — skipping.", name)


if __name__ == "__main__":
    run()
