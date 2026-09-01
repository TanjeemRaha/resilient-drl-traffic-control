"""Configuration loading and validation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
APPROACHES = ("N", "S", "E", "W")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return value


def _deep_update(target: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge profile settings without discarding training modes."""
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_config(profile: str = "quick") -> dict[str, Any]:
    """Load base, scenario, and selected experiment profile configuration."""
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'")
    config = _load_yaml(ROOT / "configs" / "base.yaml")
    _deep_update(config, _load_yaml(ROOT / "configs" / "scenarios.yaml"))
    _deep_update(config, _load_yaml(ROOT / "configs" / f"{profile}.yaml"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Reject invalid simulator, arrival, capacity, and duration values."""
    sim = config["simulation"]
    for key in ("time_step_seconds", "decision_interval_seconds", "episode_duration_seconds",
                "min_green_seconds", "max_green_seconds", "yellow_seconds", "max_queue",
                "saturation_flow_vph", "wait_reference_seconds"):
        if float(sim[key]) <= 0:
            raise ValueError(f"simulation.{key} must be positive")
    if sim["min_green_seconds"] > sim["max_green_seconds"]:
        raise ValueError("minimum green cannot exceed maximum green")
    for name, scenario in config["scenarios"].items():
        rates = scenario["base_arrival_vph"]
        if set(rates) != set(APPROACHES) or any(float(v) < 0 for v in rates.values()):
            raise ValueError(f"Scenario {name} has invalid arrival rates")
        for event in scenario.get("events", []):
            if event["start"] < 0 or event["end"] <= event["start"]:
                raise ValueError(f"Scenario {name} has invalid event timing")
            if event["type"] == "capacity_factor" and not 0 <= event["value"] <= 1:
                raise ValueError(f"Scenario {name} has invalid capacity factor")
            if event["type"] == "capacity_ramp" and not (
                0 <= event["start_value"] <= 1 and 0 <= event["end_value"] <= 1
            ):
                raise ValueError(f"Scenario {name} has invalid capacity ramp")
            if event["type"] == "demand_multiplier" and event["value"] < 0:
                raise ValueError(f"Scenario {name} has invalid demand multiplier")
    for mode in ("normal_training", "resilient_training"):
        spec = config["training"][mode]
        for approach, bounds in spec["arrival_ranges_vph"].items():
            if approach not in APPROACHES or len(bounds) != 2 or bounds[0] < 0 or bounds[1] < bounds[0]:
                raise ValueError(f"{mode} has invalid arrival range for {approach}")
        probability = float(spec["disruption_probability"])
        if not 0 <= probability <= 1:
            raise ValueError(f"{mode} disruption probability must lie in [0, 1]")
    resilient = config["training"]["resilient_training"]
    low_factor, high_factor = resilient["capacity_factor_range"]
    if not 0 <= low_factor <= high_factor <= 1:
        raise ValueError("resilient training capacity factors must lie in [0, 1]")


def scenario_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return an independent scenario definition."""
    try:
        return deepcopy(config["scenarios"][name])
    except KeyError as exc:
        raise ValueError(f"Unknown scenario: {name}") from exc
