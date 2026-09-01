from __future__ import annotations

import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "logs" / ".matplotlib"))

import numpy as np
from stable_baselines3.common.env_checker import check_env

from resilient_traffic.config import load_config
from resilient_traffic.environment import TrafficSignalEnv


def main() -> None:
    config = load_config("quick")
    env = TrafficSignalEnv(config)
    check_env(env, warn=True)
    for seed in range(3):
        observation, _ = env.reset(seed=seed)
        done, steps = False, 0
        while not done:
            action = int(env.np_random.integers(0, 2))
            observation, reward, terminated, truncated, _ = env.step(action)
            if not env.observation_space.contains(observation) or not np.isfinite(reward):
                raise RuntimeError("Non-finite or out-of-space environment output")
            done, steps = terminated or truncated, steps + 1
        print(f"Validated complete episode seed={seed}, steps={steps}")
    print("Gymnasium check_env and three complete episodes passed.")


if __name__ == "__main__":
    main()
