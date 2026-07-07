"""Training stage entry point — trains the win model on the gold table.

Optional 4th pipeline stage. Wire it into the Databricks job as a task that
depends on gold_features if you want the model retrained on each run.
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

from src.modeling.train import train  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train the race-winner model.")
    parser.add_argument("--gold-path", default="data/gold")
    parser.add_argument("--model-path", default="data/models/win_model.joblib")
    parser.add_argument("--fmt", default="parquet")
    args, _ = parser.parse_known_args()

    metrics = train(gold_path=args.gold_path, model_path=args.model_path, fmt=args.fmt)
    logging.getLogger("run_train").info("Training complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
