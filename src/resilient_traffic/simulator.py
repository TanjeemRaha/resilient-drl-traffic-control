"""Seeded, one-second, FIFO queue traffic simulator."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

import numpy as np

from .config import APPROACHES
from .scenarios import traffic_state

NS_GREEN = "NS_GREEN"
EW_GREEN = "EW_GREEN"
YELLOW = "YELLOW"
PHASE_FOR_ACTION = {0: NS_GREEN, 1: EW_GREEN}
ACTION_FOR_PHASE = {NS_GREEN: 0, EW_GREEN: 1}


@dataclass
class TickResult:
    """Measurements produced by one internal second."""

    time: int
    arrivals: dict[str, int]
    departed: dict[str, int]
    overflow: dict[str, int]
    queue_lengths: dict[str, int]
    mean_waits: dict[str, float]
    max_waits: dict[str, float]
    capacity_factors: dict[str, float]
    phase: str


class TrafficSimulator:
    """A deliberately simple intersection model with per-vehicle FIFO waiting times."""

    def __init__(self, simulation: dict[str, Any], scenario: dict[str, Any], seed: int = 0):
        self.config = simulation
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """Restore an empty intersection and optionally reseed arrivals."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time = 0
        self.queues: dict[str, Deque[float]] = {a: deque() for a in APPROACHES}
        self.service_credit = {a: 0.0 for a in APPROACHES}
        self.phase = str(self.config.get("initial_phase", NS_GREEN))
        self.previous_green = self.phase
        self.target_phase: str | None = None
        self.green_elapsed = 0
        self.yellow_remaining = 0
        self.total_arrivals = {a: 0 for a in APPROACHES}
        self.total_departed = {a: 0 for a in APPROACHES}
        self.total_overflow = {a: 0 for a in APPROACHES}
        self.departed_wait_sum = 0.0
        self.departed_wait_max = 0.0
        self.cumulative_wait_vehicle_seconds = 0.0
        self.switches = 0
        self.blocked_switches = 0
        self.records: list[dict[str, Any]] = []

    @property
    def queue_lengths(self) -> dict[str, int]:
        return {a: len(self.queues[a]) for a in APPROACHES}

    @property
    def mean_waits(self) -> dict[str, float]:
        return {a: float(np.mean(self.queues[a])) if self.queues[a] else 0.0 for a in APPROACHES}

    @property
    def max_waits(self) -> dict[str, float]:
        return {a: float(max(self.queues[a])) if self.queues[a] else 0.0 for a in APPROACHES}

    def request_phase(self, action: int) -> dict[str, Any]:
        """Apply signal constraints and return auditable action information."""
        if action not in PHASE_FOR_ACTION:
            raise ValueError("action must be 0 (NS) or 1 (EW)")
        requested = PHASE_FOR_ACTION[action]
        result = {"requested_action": action, "applied_action": ACTION_FOR_PHASE.get(self.phase, -1),
                  "blocked_switch": False, "successful_switch": False, "switch_initiated": False}
        if self.phase == YELLOW:
            result["blocked_switch"] = True
            self.blocked_switches += 1
            return result
        current_action = ACTION_FOR_PHASE[self.phase]
        must_switch = self.green_elapsed >= int(self.config["max_green_seconds"])
        wants_switch = requested != self.phase
        if wants_switch and self.green_elapsed < int(self.config["min_green_seconds"]) and not must_switch:
            result["blocked_switch"] = True
            result["applied_action"] = current_action
            self.blocked_switches += 1
            return result
        if wants_switch or must_switch:
            target = requested if wants_switch else (EW_GREEN if self.phase == NS_GREEN else NS_GREEN)
            self.previous_green = self.phase
            self.target_phase = target
            self.phase = YELLOW
            self.yellow_remaining = int(self.config["yellow_seconds"])
            self.green_elapsed = 0
            self.switches += 1
            result.update({"applied_action": -1, "successful_switch": True, "switch_initiated": True})
        return result

    def tick(self) -> TickResult:
        """Advance arrivals, waiting, service, and the signal by one second."""
        if self.phase != YELLOW and self.green_elapsed >= int(self.config["max_green_seconds"]):
            self.request_phase(ACTION_FOR_PHASE[self.phase])
        rates, capacities = traffic_state(self.scenario, self.time)
        arrivals: dict[str, int] = {}
        overflow: dict[str, int] = {}
        departed: dict[str, int] = {a: 0 for a in APPROACHES}
        max_queue = int(self.config["max_queue"])

        for approach in APPROACHES:
            count = int(self.rng.poisson(rates[approach] / 3600.0))
            accepted = min(count, max_queue - len(self.queues[approach]))
            lost = count - accepted
            self.queues[approach].extend([0.0] * accepted)
            arrivals[approach] = count
            overflow[approach] = lost
            self.total_arrivals[approach] += count
            self.total_overflow[approach] += lost

        for approach in APPROACHES:
            self.queues[approach] = deque(wait + 1.0 for wait in self.queues[approach])
        self.cumulative_wait_vehicle_seconds += float(sum(len(queue) for queue in self.queues.values()))

        green_approaches: tuple[str, ...] = ()
        if self.phase == NS_GREEN:
            green_approaches = ("N", "S")
        elif self.phase == EW_GREEN:
            green_approaches = ("E", "W")
        saturation_per_second = float(self.config["saturation_flow_vph"]) / 3600.0
        for approach in APPROACHES:
            if approach not in green_approaches:
                self.service_credit[approach] = 0.0
                continue
            self.service_credit[approach] += saturation_per_second * capacities[approach]
            discharge = min(len(self.queues[approach]), int(self.service_credit[approach]))
            for _ in range(discharge):
                waited = self.queues[approach].popleft()
                self.departed_wait_sum += waited
                self.departed_wait_max = max(self.departed_wait_max, waited)
            departed[approach] = discharge
            self.total_departed[approach] += discharge
            self.service_credit[approach] -= discharge
            if not self.queues[approach]:
                self.service_credit[approach] = min(self.service_credit[approach], 0.999999)

        state_during_tick = self.phase
        if self.phase == YELLOW:
            self.yellow_remaining -= 1
            if self.yellow_remaining <= 0:
                if self.target_phase is None:
                    raise RuntimeError("yellow phase has no target")
                self.phase = self.target_phase
                self.target_phase = None
                self.green_elapsed = 0
        else:
            self.green_elapsed += 1

        self.time += 1
        result = TickResult(self.time, arrivals, departed, overflow, self.queue_lengths,
                            self.mean_waits, self.max_waits, capacities, state_during_tick)
        record: dict[str, Any] = {"time": self.time, "phase": state_during_tick,
                                  "total_queue": sum(result.queue_lengths.values()),
                                  "throughput": sum(departed.values()),
                                  "overflow": sum(overflow.values())}
        for approach in APPROACHES:
            record.update({f"queue_{approach}": result.queue_lengths[approach],
                           f"mean_wait_{approach}": result.mean_waits[approach],
                           f"max_wait_{approach}": result.max_waits[approach],
                           f"arrival_{approach}": arrivals[approach],
                           f"departed_{approach}": departed[approach],
                           f"capacity_{approach}": capacities[approach]})
        self.records.append(record)
        return result

    def run_seconds(self, seconds: int) -> list[TickResult]:
        """Advance an integer number of internal seconds."""
        return [self.tick() for _ in range(seconds)]
