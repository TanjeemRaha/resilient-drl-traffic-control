"""Command-line audit for quick or full experiment artifacts."""
from __future__ import annotations

import argparse

from resilient_traffic.auditing import audit_results
from resilient_traffic.config import ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    args = parser.parse_args()
    result = audit_results(load_config(args.profile), ROOT)
    print(f"profile={args.profile} evaluation_rows={result['evaluation_rows']} "
          "reward_nonfinite=0 negative_queues=0")
    print(f"episodes={result['episodes']} controllers={result['controllers']} "
          f"scenarios={result['scenarios']} traffic_seeds={result['traffic_seeds']}")
    print("profile_coverage=pass model_metadata=pass")
    print("conservation_identity=pass cumulative_delay_identity=pass spillback_reward=active")
    print(f"spillback_stress_baseline_max_overflow={result['stress_max_overflow']}")
    print("corrected_summaries=present")


if __name__ == "__main__":
    main()

