from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "scripts/validate_environment.py"],
        [sys.executable, "scripts/run_baselines.py", "--profile", args.profile],
        [sys.executable, "scripts/train_agents.py", "--profile", args.profile],
        [sys.executable, "scripts/evaluate_controllers.py", "--profile", args.profile],
        [sys.executable, "scripts/generate_figures.py", "--profile", args.profile],
        [sys.executable, "scripts/audit_results.py", "--profile", args.profile],
    ]
    if args.force_train:
        commands[3].append("--force-train")
    for command in commands:
        print(f"\n>>> {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=root, check=True)
    print(f"\nComplete {args.profile} pipeline succeeded.")


if __name__ == "__main__":
    main()
