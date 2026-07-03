"""Gold stage entry point — run by the 'gold_features' task.

Calls build_gold(). Guarded the same way as silver until you implement
src/transform/gold.py.
"""
import argparse
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.transform import gold  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Gold stage: build race feature tables.")
    parser.add_argument("--silver-path", default="data/silver")
    parser.add_argument("--gold-path", default="data/gold")
    args, _ = parser.parse_known_args()

    try:
        gold.build_gold()  # TODO: pass args.silver_path / args.gold_path once implemented
    except NotImplementedError:
        logging.getLogger("run_gold").warning(
            "gold stage not implemented yet — skipping. Build src/transform/gold.py, then remove this guard."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
