"""Transparent normalized reward functions."""
from __future__ import annotations

from typing import Any

import numpy as np

from .simulator import TrafficSimulator


def reward_components(sim: TrafficSimulator, overflow_delta: int, switch_initiated: bool,
                      config: dict[str, Any]) -> dict[str, float]:
    """Calculate clipped components; each lies in [0, 1] except the penalty."""
    queues = sim.queue_lengths
    means = sim.mean_waits
    maxima = sim.max_waits
    max_queue = float(sim.config["max_queue"])
    wait_ref = float(sim.config["wait_reference_seconds"])
    ns_queue = queues["N"] + queues["S"]
    ew_queue = queues["E"] + queues["W"]
    occupancies = np.asarray([queues[approach] / max_queue for approach in ("N", "S", "E", "W")])
    return {
        "total_queue_normalized": float(np.clip(sum(queues.values()) / (4 * max_queue), 0, 1)),
        "mean_wait_normalized": float(np.clip(np.mean(list(means.values())) / wait_ref, 0, 1)),
        "max_wait_normalized": float(np.clip(max(maxima.values()) / wait_ref, 0, 1)),
        "directional_imbalance_normalized": float(np.clip(abs(ns_queue - ew_queue) / (2 * max_queue), 0, 1)),
        "spillback_risk_normalized": float(np.clip(np.mean(np.square(occupancies)), 0, 1)),
        "overflow_normalized": float(np.clip(overflow_delta / (4 * max_queue), 0, 1)),
        "switch_penalty": float(config["switch_penalty"] if switch_initiated else 0.0),
    }


def calculate_reward(name: str, components: dict[str, float], config: dict[str, Any]) -> float:
    """Return queue-only or resilience-focused negative cost."""
    if name == "queue_reward":
        return -components["total_queue_normalized"]
    if name == "resilience_reward":
        weights = config["weights"]
        cost = (weights["total_queue"] * components["total_queue_normalized"]
                + weights["mean_wait"] * components["mean_wait_normalized"]
                + weights["max_wait"] * components["max_wait_normalized"]
                + weights["directional_imbalance"] * components["directional_imbalance_normalized"]
                + weights["spillback_risk"] * components["spillback_risk_normalized"]
                + components["switch_penalty"])
        return -float(cost)
    raise ValueError(f"Unknown reward: {name}")
