"""Fair common-random-number evaluation for baseline and learned controllers."""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .config import ROOT
from .behavior_audit import model_behavior_audit
from .controllers import ActuatedController, FixedTimeController
from .environment import TrafficSignalEnv
from .metrics import episode_metrics, performance_deterioration
from .scenarios import disruption_window
from .statistics import build_statistical_summaries
from .training import AGENTS, load_model
from .utils import ensure_output_dirs


def _controller_entries(config: dict[str, Any], include_learned: bool) -> list[tuple[str, int, Any]]:
    entries: list[tuple[str, int, Any]] = [
        ("FixedTime", -1, FixedTimeController(config["actuated"]["fixed_green_seconds"])),
        ("Actuated", -1, ActuatedController(config["actuated"])),
    ]
    if include_learned:
        for name in AGENTS:
            for seed in config["training"]["seeds"]:
                entries.append((name, seed, load_model(name, seed)))
    return entries


def evaluate(config: dict[str, Any], include_learned: bool = True,
             scenarios: Iterable[str] | None = None, prefix: str = "evaluation") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate with identical scenario/traffic seeds; save raw and episode CSV files."""
    ensure_output_dirs()
    scenario_names = list(scenarios or config["scenarios"].keys())
    raw_frames: list[pd.DataFrame] = []
    episode_rows: list[dict[str, Any]] = []
    for controller_name, model_seed, controller in _controller_entries(config, include_learned):
        for scenario_name in scenario_names:
            reward_name = AGENTS.get(controller_name, {}).get("reward", "queue_reward")
            for traffic_seed in config["evaluation"]["traffic_seeds"]:
                for episode in range(int(config["evaluation"]["episodes_per_seed"])):
                    env = TrafficSignalEnv(config, scenario_name=scenario_name, reward_name=reward_name)
                    observation, _ = env.reset(seed=int(traffic_seed))
                    done, decision = False, 0
                    while not done:
                        if controller_name in {"FixedTime", "Actuated"}:
                            action = controller.act(env)
                        else:
                            action, _ = controller.predict(observation, deterministic=True)
                            action = int(action)
                        start_row = len(env.simulator.records)
                        observation, reward, terminated, truncated, info = env.step(action)
                        for record in env.simulator.records[start_row:]:
                            record.update({"decision": decision, "requested_action": action,
                                "applied_action": info["applied_action"], "reward": reward,
                                "total_queue_normalized": info["total_queue_normalized"],
                                "mean_wait_normalized": info["mean_wait_normalized"],
                                "max_wait_normalized": info["max_wait_normalized"],
                                "directional_imbalance_normalized": info["directional_imbalance_normalized"],
                                "spillback_risk_normalized": info["spillback_risk_normalized"],
                                "overflow_normalized": info["overflow_normalized"],
                                "switch_penalty": info["switch_penalty"]})
                        decision += 1
                        done = terminated or truncated
                    records = pd.DataFrame(env.simulator.records)
                    identifiers = {"controller": controller_name, "trained_model_seed": model_seed,
                                   "scenario": scenario_name, "traffic_seed": traffic_seed,
                                   "episode": episode, "profile": config["profile"],
                                   "output_label": config["label"]}
                    for key, value in identifiers.items():
                        records[key] = value
                    raw_frames.append(records)
                    values = episode_metrics(records, env.simulator.departed_wait_sum,
                                             env.simulator.departed_wait_max, env.simulator.switches,
                                             disruption_window(env.current_scenario),
                                             cumulative_wait_vehicle_seconds=(
                                                 env.simulator.cumulative_wait_vehicle_seconds),
                                             ending_wait_sum=sum(sum(queue) for queue in env.simulator.queues.values()),
                                             ending_wait_max=max(env.simulator.max_waits.values()),
                                             total_arrivals=sum(env.simulator.total_arrivals.values()),
                                             total_overflow=sum(env.simulator.total_overflow.values()))
                    episode_rows.append({**identifiers, **values})
                    env.close()
            print(f"Evaluated {controller_name} on {scenario_name}")
    raw = pd.concat(raw_frames, ignore_index=True)
    episodes = pd.DataFrame(episode_rows)
    raw.to_csv(ROOT / "results" / "raw" / f"{prefix}_records.csv", index=False)
    episodes.to_csv(ROOT / "results" / "summaries" / f"{prefix}_episode_metrics.csv", index=False)
    if prefix == "evaluation":
        episodes.to_csv(ROOT / "results" / "summaries" / "episode_metrics.csv", index=False)
        per_model, controller, paired = build_statistical_summaries(
            episodes, int(config["bootstrap_samples"]), seed=2026)
        per_model.to_csv(ROOT / "results" / "summaries" / "per_model_seed_summary.csv", index=False)
        controller.to_csv(ROOT / "results" / "summaries" / "controller_summary.csv", index=False)
        paired.to_csv(ROOT / "results" / "summaries" / "paired_controller_comparisons.csv", index=False)
        model_behavior_audit(raw, learned_controllers=AGENTS.keys()).to_csv(
            ROOT / "results" / "summaries" / "model_behavior_audit.csv", index=False)
        performance_deterioration(episodes).to_csv(
            ROOT / "results" / "summaries" / "performance_deterioration.csv", index=False)
        recovery = episodes[episodes.scenario != "normal"][["controller", "trained_model_seed", "scenario",
            "traffic_seed", "recovery_time_seconds", "recovered"]]
        recovery.to_csv(ROOT / "results" / "summaries" / "recovery_summary.csv", index=False)
    return raw, episodes
