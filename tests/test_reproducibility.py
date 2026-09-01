from resilient_traffic.environment import TrafficSignalEnv


def rollout(config, seed):
    env = TrafficSignalEnv(config)
    obs, _ = env.reset(seed=seed)
    values = []
    for action in [0, 0, 0, 1, 1, 1] * 10:
        obs, reward, _, done, info = env.step(action)
        values.append((obs.tolist(), reward, info["throughput"]))
        if done:
            break
    return values


def test_same_seed_same_actions(config):
    assert rollout(config, 17) == rollout(config, 17)

