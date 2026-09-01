import numpy as np
from stable_baselines3.common.env_checker import check_env

from resilient_traffic.environment import TrafficSignalEnv


def test_environment_api_and_observation(config):
    env = TrafficSignalEnv(config)
    check_env(env, warn=True)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (12,) and env.observation_space.contains(obs)
    for _ in range(360):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert not terminated
        if truncated:
            break
    assert truncated


def test_three_complete_seeded_episodes(config):
    env = TrafficSignalEnv(config)
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(int(env.np_random.integers(0, 2)))
            assert env.observation_space.contains(obs)
            done = terminated or truncated


def test_yellow_one_hot_is_zero(config):
    env = TrafficSignalEnv(config)
    env.reset(seed=1)
    env.simulator.green_elapsed = config["simulation"]["min_green_seconds"]
    obs, *_ = env.step(1)
    env.simulator.phase = "YELLOW"
    obs = env._observation()
    assert np.array_equal(obs[8:10], [0.0, 0.0])
