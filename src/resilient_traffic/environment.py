"""Gymnasium environment wrapping the queue simulator."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import APPROACHES, scenario_config
from .rewards import calculate_reward, reward_components
from .scenarios import sample_training_scenario
from .simulator import ACTION_FOR_PHASE, TrafficSimulator


class TrafficSignalEnv(gym.Env[np.ndarray, int]):
    """Two-action signal control with a fixed 12-value continuous observation."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any], scenario_name: str = "normal",
                 reward_name: str = "queue_reward", training_mode: str | None = None):
        super().__init__()
        self.config = deepcopy(config)
        self.scenario_name = scenario_name
        self.reward_name = reward_name
        self.training_mode = training_mode
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(12,), dtype=np.float32)
        self.simulator: TrafficSimulator | None = None
        self.current_scenario: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        simulation_seed = int(self.np_random.integers(0, 2**31 - 1))
        if self.training_mode:
            self.current_scenario = sample_training_scenario(self.config, self.training_mode, self.np_random)
        else:
            self.current_scenario = scenario_config(self.config, self.scenario_name)
        self.simulator = TrafficSimulator(self.config["simulation"], self.current_scenario, simulation_seed)
        obs = self._observation()
        return obs, {"seed": seed, "scenario": self.scenario_name if not self.training_mode else self.training_mode}

    def _observation(self) -> np.ndarray:
        if self.simulator is None:
            raise RuntimeError("reset must be called before observation")
        sim = self.simulator
        max_queue = float(sim.config["max_queue"])
        wait_ref = float(sim.config["wait_reference_seconds"])
        queues = [sim.queue_lengths[a] / max_queue for a in APPROACHES]
        waits = [min(sim.mean_waits[a] / wait_ref, 1.0) for a in APPROACHES]
        phase = [float(sim.phase == "NS_GREEN"), float(sim.phase == "EW_GREEN")]
        elapsed = min(sim.green_elapsed / float(sim.config["max_green_seconds"]), 1.0)
        progress = min(sim.time / float(sim.config["episode_duration_seconds"]), 1.0)
        return np.asarray(queues + waits + phase + [elapsed, progress], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.simulator is None:
            raise RuntimeError("reset must be called before step")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        sim = self.simulator
        switches_before = sim.switches
        action_info = sim.request_phase(int(action))
        overflow_before = sum(sim.total_overflow.values())
        duration = int(sim.config["decision_interval_seconds"])
        remaining = int(sim.config["episode_duration_seconds"]) - sim.time
        sim.run_seconds(min(duration, remaining))
        overflow_delta = sum(sim.total_overflow.values()) - overflow_before
        components = reward_components(sim, overflow_delta, sim.switches > switches_before, self.config["reward"])
        reward = calculate_reward(self.reward_name, components, self.config["reward"])
        truncated = sim.time >= int(sim.config["episode_duration_seconds"])
        info: dict[str, Any] = {**action_info, **components, "time": sim.time,
                                "phase": sim.phase, "queue_lengths": sim.queue_lengths,
                                "mean_waits": sim.mean_waits, "max_waits": sim.max_waits,
                                "throughput": sum(sim.total_departed.values()),
                                "overflow": sum(sim.total_overflow.values()),
                                "switches": sim.switches,
                                "applied_action": ACTION_FOR_PHASE.get(sim.phase, -1)}
        return self._observation(), float(reward), False, truncated, info
