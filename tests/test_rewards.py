from resilient_traffic.environment import TrafficSignalEnv
from resilient_traffic.rewards import calculate_reward, reward_components


def test_queue_reward(config):
    components = {"total_queue_normalized": 0.25}
    assert calculate_reward("queue_reward", components, config["reward"]) == -0.25


def test_resilience_components_are_bounded(config):
    env = TrafficSignalEnv(config, reward_name="resilience_reward")
    env.reset(seed=1)
    _, reward, _, _, info = env.step(0)
    names = ["total_queue_normalized", "mean_wait_normalized", "max_wait_normalized",
             "directional_imbalance_normalized", "spillback_risk_normalized", "overflow_normalized"]
    assert all(0 <= info[name] <= 1 for name in names)
    assert reward <= 0


def test_spillback_risk_is_zero_bounded_and_increases(config):
    env = TrafficSignalEnv(config, reward_name="resilience_reward")
    env.reset(seed=1)
    empty = reward_components(env.simulator, 0, False, config["reward"])["spillback_risk_normalized"]
    assert empty == 0.0
    for approach in "NSEW":
        env.simulator.queues[approach].extend([0.0] * 50)
    half = reward_components(env.simulator, 0, False, config["reward"])["spillback_risk_normalized"]
    for approach in "NSEW":
        env.simulator.queues[approach].extend([0.0] * 50)
    full = reward_components(env.simulator, 0, False, config["reward"])["spillback_risk_normalized"]
    assert 0.0 <= half < full <= 1.0
