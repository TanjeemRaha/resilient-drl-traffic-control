"""Profile-aware validation of generated experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .training import AGENTS

BASELINES = ("FixedTime", "Actuated")
ALL_CONTROLLERS = (*BASELINES, *AGENTS.keys())
REQUIRED_SUMMARIES = (
    "per_model_seed_summary.csv",
    "controller_summary.csv",
    "paired_controller_comparisons.csv",
    "model_behavior_audit.csv",
)


def validate_profile_coverage(episodes: pd.DataFrame, config: dict[str, Any]) -> None:
    """Validate profile labels and the complete configured evaluation design."""
    required = {"profile", "output_label", "traffic_seed", "scenario", "controller",
                "trained_model_seed", "episode"}
    missing = sorted(required.difference(episodes.columns))
    if missing:
        raise ValueError(f"Episode metrics are missing profile columns: {missing}")
    if episodes["profile"].isna().any() or episodes["output_label"].isna().any():
        raise ValueError("Every episode row must contain a profile and output label")
    profiles = set(episodes["profile"].astype(str))
    if profiles != {str(config["profile"])}:
        raise ValueError(f"Episode profile mismatch: expected {config['profile']!r}, found {sorted(profiles)}")
    labels = set(episodes["output_label"].astype(str))
    if labels != {str(config["label"])}:
        raise ValueError(f"Episode output-label mismatch: expected {config['label']!r}, found {sorted(labels)}")

    expected_traffic = {int(seed) for seed in config["evaluation"]["traffic_seeds"]}
    actual_traffic = {int(seed) for seed in episodes["traffic_seed"].unique()}
    if actual_traffic != expected_traffic:
        raise ValueError(f"Traffic-seed mismatch: expected {sorted(expected_traffic)}, found {sorted(actual_traffic)}")
    expected_scenarios = set(config["scenarios"])
    actual_scenarios = set(episodes["scenario"].astype(str).unique())
    if actual_scenarios != expected_scenarios:
        raise ValueError(f"Scenario mismatch: expected {sorted(expected_scenarios)}, found {sorted(actual_scenarios)}")
    actual_controllers = set(episodes["controller"].astype(str).unique())
    if actual_controllers != set(ALL_CONTROLLERS):
        raise ValueError(f"Controller mismatch: expected {sorted(ALL_CONTROLLERS)}, found {sorted(actual_controllers)}")
    design_key = ["controller", "trained_model_seed", "scenario", "traffic_seed", "episode"]
    if episodes.duplicated(design_key).any():
        raise ValueError("Episode metrics contain duplicate evaluation-design rows")

    expected_model_seeds = {int(seed) for seed in config["training"]["seeds"]}
    expected_episodes = int(config["evaluation"]["episodes_per_seed"])
    for controller in ALL_CONTROLLERS:
        controller_rows = episodes[episodes["controller"] == controller]
        expected_seeds = {-1} if controller in BASELINES else expected_model_seeds
        actual_seeds = {int(seed) for seed in controller_rows["trained_model_seed"].unique()}
        if actual_seeds != expected_seeds:
            raise ValueError(f"{controller} model-seed mismatch: expected {sorted(expected_seeds)}, "
                             f"found {sorted(actual_seeds)}")
        for model_seed in expected_seeds:
            seed_rows = controller_rows[controller_rows["trained_model_seed"] == model_seed]
            for scenario in expected_scenarios:
                scenario_rows = seed_rows[seed_rows["scenario"] == scenario]
                scenario_traffic = {int(seed) for seed in scenario_rows["traffic_seed"].unique()}
                if scenario_traffic != expected_traffic:
                    raise ValueError(f"Incomplete traffic seeds for {controller}, model seed {model_seed}, "
                                     f"scenario {scenario}")
                counts = scenario_rows.groupby("traffic_seed")["episode"].nunique()
                if len(counts) != len(expected_traffic) or not (counts == expected_episodes).all():
                    raise ValueError(f"Incomplete episodes for {controller}, model seed {model_seed}, "
                                     f"scenario {scenario}")


def validate_model_metadata(config: dict[str, Any], root: Path) -> None:
    """Require every configured learned-model artifact and exact metadata contract."""
    for controller, specification in AGENTS.items():
        for seed in config["training"]["seeds"]:
            stem = root / "models" / f"{controller}_seed{seed}"
            zip_path = stem.with_suffix(".zip")
            json_path = stem.with_suffix(".json")
            if not zip_path.exists():
                raise FileNotFoundError(f"Missing model ZIP for {controller}, seed {seed}: {zip_path}")
            if not json_path.exists():
                raise FileNotFoundError(f"Missing model metadata for {controller}, seed {seed}: {json_path}")
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            expected = {
                "profile": config["profile"],
                "timesteps": int(config["training"]["timesteps"][controller]),
                "reward_configuration": config["reward"],
                "algorithm": specification["algorithm"],
                "training_mode": specification["mode"],
                "reward": specification["reward"],
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(f"Metadata mismatch for {controller}, seed {seed}, {key}: "
                                     f"expected {value!r}, found {metadata.get(key)!r}")


def audit_results(config: dict[str, Any], root: Path) -> dict[str, int]:
    """Run profile, artifact, numerical, conservation, and reward audits."""
    raw_path = root / "results" / "raw" / "evaluation_records.csv"
    episode_path = root / "results" / "summaries" / "episode_metrics.csv"
    if not raw_path.exists() or not episode_path.exists():
        raise FileNotFoundError("Run the evaluation pipeline before auditing results.")
    raw = pd.read_csv(raw_path)
    episodes = pd.read_csv(episode_path)
    validate_profile_coverage(episodes, config)
    validate_model_metadata(config, root)
    missing_summaries = [name for name in REQUIRED_SUMMARIES
                         if not (root / "results" / "summaries" / name).exists()]
    if missing_summaries:
        raise FileNotFoundError(f"Missing corrected summary files: {missing_summaries}")
    queues = raw[[f"queue_{approach}" for approach in "NSEW"]]
    rewards = pd.to_numeric(raw["reward"], errors="coerce").to_numpy()
    if not np.isfinite(rewards).all():
        raise RuntimeError("Evaluation records contain NaN or infinite rewards.")
    if (queues < 0).any().any():
        raise RuntimeError("Evaluation records contain negative queues.")
    conservation = (episodes["total_arrivals_vehicles"] - episodes["throughput_vehicles"]
                    - episodes["ending_queue_vehicles"] - episodes["overflow_vehicles"])
    if not (conservation == 0).all():
        raise RuntimeError("Vehicle conservation identity failed.")
    wait_identity = (episodes["cumulative_delay_vehicle_seconds"]
                     - episodes["departed_wait_vehicle_seconds"]
                     - episodes["ending_wait_vehicle_seconds"])
    if not np.allclose(wait_identity, 0.0, atol=1e-9):
        raise RuntimeError("Cumulative-delay accounting identity failed.")
    if ("spillback_risk_normalized" not in raw
            or not raw["spillback_risk_normalized"].between(0, 1).all()):
        raise RuntimeError("Spillback risk is missing or outside [0, 1].")
    resilient = raw[raw.controller == "PPO_resilient"]
    weights = config["reward"]["weights"]
    reconstructed = -(weights["total_queue"] * resilient["total_queue_normalized"]
        + weights["mean_wait"] * resilient["mean_wait_normalized"]
        + weights["max_wait"] * resilient["max_wait_normalized"]
        + weights["directional_imbalance"] * resilient["directional_imbalance_normalized"]
        + weights["spillback_risk"] * resilient["spillback_risk_normalized"]
        + resilient["switch_penalty"])
    if resilient.empty or not np.allclose(resilient["reward"], reconstructed, atol=1e-12):
        raise RuntimeError("PPO_resilient records do not use the configured spillback-risk reward.")
    stress = episodes[(episodes.scenario == "spillback_stress")
                      & episodes.controller.isin(BASELINES)]
    if stress.empty or stress["overflow_vehicles"].max() <= 0:
        raise RuntimeError("Baseline spillback_stress diagnostics did not activate storage overflow.")
    return {
        "evaluation_rows": len(raw), "episodes": len(episodes),
        "controllers": episodes.controller.nunique(), "scenarios": episodes.scenario.nunique(),
        "traffic_seeds": episodes.traffic_seed.nunique(),
        "stress_max_overflow": int(stress["overflow_vehicles"].max()),
    }
