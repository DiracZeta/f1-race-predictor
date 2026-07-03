"""Silver stage entry point — run by the 'silver_transform' task.

Calls build_silver(). Until you implement src/transform/silver.py, it logs a
warning and exits successfully so the job's DAG runs green. Once you build the
stage, remove the NotImplementedError guard below.
"""
import argparse
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.transform import silver  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Silver stage: clean + type the bronze data.")
    parser.add_argument("--bronze-path", default="data/bronze")
    parser.add_argument("--silver-path", default="data/silver")
    args, _ = parser.parse_known_args()

    try:
        silver.build_silver()  # TODO: pass args.bronze_path / args.silver_path once implemented
    except NotImplementedError:
        logging.getLogger("run_silver").warning(
            "silver stage not implemented yet — skipping. Build src/transform/silver.py, then remove this guard."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
