from resilient_traffic.scenarios import sample_training_scenario, traffic_state
import numpy as np


def test_capacity_and_demand_events(config):
    peak = config["scenarios"]["peak_ew"]
    before, _ = traffic_state(peak, 100)
    during, _ = traffic_state(peak, 700)
    assert during["E"] > before["E"]
    _, capacity = traffic_state(config["scenarios"]["flood_ew"], 700)
    assert capacity["E"] == capacity["W"] == 0.35


def test_resilient_sampling_is_seeded(config):
    a = sample_training_scenario(config, "resilient_training", np.random.default_rng(3))
    b = sample_training_scenario(config, "resilient_training", np.random.default_rng(3))
    assert a == b


def test_spillback_stress_is_held_out_and_reduces_capacity(config):
    stress = config["scenarios"]["spillback_stress"]
    assert stress["evaluation_only"] is True
    _, capacities = traffic_state(stress, 700)
    assert capacities["E"] == capacities["W"] == 0.25
    sampled = sample_training_scenario(config, "resilient_training", np.random.default_rng(8))
    assert sampled != stress
