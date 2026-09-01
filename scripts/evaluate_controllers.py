from __future__ import annotations

import argparse
from resilient_traffic.config import load_config
from resilient_traffic.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    args = parser.parse_args()
    _, episodes = evaluate(load_config(args.profile), include_learned=True)
    print(f"Saved fair evaluation for {len(episodes)} controller/scenario/seed episodes.")


if __name__ == "__main__":
    main()

