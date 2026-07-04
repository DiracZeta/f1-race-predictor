"""Gold stage entry point — run by the 'gold_features' task.

Calls build_gold(). Guarded the same way as silver until you implement
src/transform/gold.py.
"""
import argparse
import logging
import os
import sys

# Serverless-safe repo-root discovery (`__file__` may be undefined on serverless).
def _add_repo_root_to_path():
    candidates = []
    try:
        candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except NameError:
        pass
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
    if candidates and candidates[0] not in sys.path:
        sys.path.insert(0, candidates[0])

_add_repo_root_to_path()

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
        return


if __name__ == "__main__":
    main()
