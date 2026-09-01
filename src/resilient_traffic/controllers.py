"""Non-learning reference traffic-signal controllers."""
from __future__ import annotations

from typing import Any

from .config import APPROACHES
from .environment import TrafficSignalEnv


class FixedTimeController:
    """Request alternating directions after a fixed green duration (default 30 s)."""

    name = "FixedTime"

    def __init__(self, green_seconds: int = 30):
        self.green_seconds = green_seconds

    def act(self, env: TrafficSignalEnv) -> int:
        sim = env.simulator
        if sim is None:
            raise RuntimeError("environment has not been reset")
        current = 0 if (sim.phase == "NS_GREEN" or sim.previous_green == "NS_GREEN" and sim.phase == "YELLOW") else 1
        return 1 - current if sim.phase != "YELLOW" and sim.green_elapsed >= self.green_seconds else current


class ActuatedController:
    """Select the direction with greater queue + mean-wait + max-wait demand score.

    Scores are sums over the two approaches. Queue weight is 1.0; waiting values
    in seconds use smaller configurable weights. The environment independently
    enforces minimum green, maximum green, and yellow clearance.
    """

    name = "Actuated"

    def __init__(self, settings: dict[str, Any]):
        self.queue_weight = float(settings["queue_weight"])
        self.mean_wait_weight = float(settings["mean_wait_weight"])
        self.max_wait_weight = float(settings["max_wait_weight"])

    def act(self, env: TrafficSignalEnv) -> int:
        sim = env.simulator
        if sim is None:
            raise RuntimeError("environment has not been reset")
        q, mean, maximum = sim.queue_lengths, sim.mean_waits, sim.max_waits
        def score(group: tuple[str, str]) -> float:
            return sum(self.queue_weight * q[a] + self.mean_wait_weight * mean[a]
                       + self.max_wait_weight * maximum[a] for a in group)
        ns, ew = score(("N", "S")), score(("E", "W"))
        if ns == ew:
            return 0 if sim.phase == "NS_GREEN" else 1
        return 0 if ns > ew else 1

