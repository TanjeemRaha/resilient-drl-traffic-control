"""Scenario schedules and randomized training distributions."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .config import APPROACHES


def traffic_state(scenario: dict[str, Any], second: int) -> tuple[dict[str, float], dict[str, float]]:
    """Return arrival rates (veh/h) and capacity factors for a simulation second."""
    rates = {a: float(scenario["base_arrival_vph"][a]) for a in APPROACHES}
    capacities = {a: 1.0 for a in APPROACHES}
    for event in scenario.get("events", []):
        if event["start"] <= second < event["end"]:
            if event["type"] == "demand_multiplier":
                for approach in event["approaches"]:
                    rates[approach] *= float(event["value"])
            elif event["type"] == "capacity_factor":
                for approach in event["approaches"]:
                    capacities[approach] *= float(event["value"])
            elif event["type"] == "capacity_ramp":
                fraction = (second - event["start"]) / (event["end"] - event["start"])
                value = event["start_value"] + fraction * (event["end_value"] - event["start_value"])
                for approach in event["approaches"]:
                    capacities[approach] *= float(value)
            else:
                raise ValueError(f"Unsupported event type: {event['type']}")
    return rates, capacities


def sample_training_scenario(config: dict[str, Any], mode: str, rng: np.random.Generator) -> dict[str, Any]:
    """Sample normal or disruption-randomized training conditions."""
    if mode not in config["training"]:
        raise ValueError(f"Unknown training mode: {mode}")
    spec = config["training"][mode]
    scenario: dict[str, Any] = {"base_arrival_vph": {}, "events": []}
    for approach in APPROACHES:
        low, high = spec["arrival_ranges_vph"][approach]
        scenario["base_arrival_vph"][approach] = float(rng.uniform(low, high))
    if rng.random() < spec["disruption_probability"]:
        start = int(rng.integers(*spec["disruption_start_range"]))
        duration = int(rng.integers(*spec["disruption_duration_range"]))
        factor = float(rng.uniform(*spec["capacity_factor_range"]))
        low_count, high_count = spec["approaches_per_event"]
        count = int(rng.integers(low_count, high_count + 1))
        selected = rng.choice(APPROACHES, size=count, replace=False).tolist()
        scenario["events"].append({"type": "capacity_factor", "approaches": selected,
                                   "start": start, "end": start + duration, "value": factor})
    return deepcopy(scenario)


def disruption_window(scenario: dict[str, Any]) -> tuple[int, int] | None:
    """Return the encompassing disruption window, excluding pure demand events."""
    events = [e for e in scenario.get("events", []) if e["type"].startswith("capacity")]
    if not events:
        events = scenario.get("events", [])
    if not events:
        return None
    return min(e["start"] for e in events), max(e["end"] for e in events)

