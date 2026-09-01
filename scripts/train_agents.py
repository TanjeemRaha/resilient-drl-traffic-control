from __future__ import annotations

import argparse
from resilient_traffic.config import load_config
from resilient_traffic.training import train_agents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    paths = train_agents(load_config(args.profile), force=args.force_train)
    print(f"Verified {len(paths)} final model target(s).")


if __name__ == "__main__":
    main()

