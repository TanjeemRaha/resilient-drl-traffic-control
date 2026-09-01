from __future__ import annotations

import argparse
from resilient_traffic.config import load_config
from resilient_traffic.evaluation import evaluate
from resilient_traffic.plotting import plot_baseline_preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    args = parser.parse_args()
    config = load_config(args.profile)
    raw, _ = evaluate(config, include_learned=False, prefix="baseline")
    plot_baseline_preview(raw)
    print("Baseline records, episode summaries, and preliminary traffic-state plot saved.")


if __name__ == "__main__":
    main()
