from __future__ import annotations

import argparse
from resilient_traffic.config import load_config
from resilient_traffic.plotting import generate_figures
from resilient_traffic.reporting import write_research_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    args = parser.parse_args()
    config = load_config(args.profile)
    figures = generate_figures(config)
    write_research_summary(config)
    print(f"Generated {len(figures)} PNG figures plus PDF counterparts and research summary.")


if __name__ == "__main__":
    main()
