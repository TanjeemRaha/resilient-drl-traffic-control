from resilient_traffic.controllers import ActuatedController, FixedTimeController
from resilient_traffic.environment import TrafficSignalEnv


def test_fixed_time_switches(config):
    env = TrafficSignalEnv(config)
    env.reset(seed=1)
    env.simulator.green_elapsed = 30
    assert FixedTimeController().act(env) == 1


def test_actuated_selects_larger_demand(config):
    env = TrafficSignalEnv(config)
    env.reset(seed=1)
    env.simulator.queues["E"].extend([0.0] * 10)
    assert ActuatedController(config["actuated"]).act(env) == 1

