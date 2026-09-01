from __future__ import annotations

from collections import deque

from resilient_traffic.config import scenario_config
from resilient_traffic.simulator import TrafficSimulator, YELLOW


def simulator(config, scenario="normal", seed=1):
    return TrafficSimulator(config["simulation"], scenario_config(config, scenario), seed)


def test_queues_never_negative(config):
    sim = simulator(config)
    sim.run_seconds(200)
    assert all(value >= 0 for row in sim.records for key, value in row.items() if key.startswith("queue_"))


def test_red_does_not_discharge_and_green_does(config):
    sim = simulator(config)
    sim.scenario["base_arrival_vph"] = {a: 0 for a in "NSEW"}
    sim.queues["N"] = deque([1.0] * 5)
    sim.queues["E"] = deque([1.0] * 5)
    results = sim.run_seconds(2)
    assert sum(result.departed["E"] for result in results) == 0
    assert sum(result.departed["N"] for result in results) > 0


def test_lower_capacity_lowers_discharge(config):
    normal = simulator(config, "normal")
    reduced = simulator(config, "lane_closure_e")
    for sim in (normal, reduced):
        sim.time = 600
        sim.phase = "EW_GREEN"
        sim.queues["E"] = deque([1.0] * 50)
        sim.run_seconds(20)
    assert normal.total_departed["E"] > reduced.total_departed["E"]


def test_seed_reproducibility(config):
    first, second = simulator(config, seed=4), simulator(config, seed=4)
    first.run_seconds(100)
    second.run_seconds(100)
    assert first.records == second.records


def test_different_seeds_change_arrivals(config):
    first, second = simulator(config, seed=4), simulator(config, seed=5)
    first.run_seconds(100)
    second.run_seconds(100)
    assert [r["arrival_N"] for r in first.records] != [r["arrival_N"] for r in second.records]


def test_yellow_has_no_discharge(config):
    sim = simulator(config)
    sim.phase, sim.target_phase, sim.yellow_remaining = YELLOW, "EW_GREEN", 3
    sim.queues["N"] = deque([1.0] * 5)
    sim.queues["E"] = deque([1.0] * 5)
    assert sum(sim.tick().departed.values()) == 0


def test_minimum_green_and_maximum_green(config):
    sim = simulator(config)
    blocked = sim.request_phase(1)
    assert blocked["blocked_switch"] and sim.phase == "NS_GREEN"
    sim.green_elapsed = config["simulation"]["min_green_seconds"]
    switched = sim.request_phase(1)
    assert switched["successful_switch"] and sim.phase == YELLOW
    sim = simulator(config)
    sim.green_elapsed = config["simulation"]["max_green_seconds"]
    forced = sim.request_phase(0)
    assert forced["successful_switch"] and sim.target_phase == "EW_GREEN"


def test_maximum_green_is_enforced_inside_decision_interval(config):
    sim = simulator(config)
    sim.green_elapsed = config["simulation"]["max_green_seconds"] - 1
    sim.run_seconds(2)
    assert sim.phase == YELLOW and sim.target_phase == "EW_GREEN"


def test_vehicle_conservation_and_exact_cumulative_wait(config):
    sim = simulator(config, "spillback_stress", seed=91)
    sim.run_seconds(config["simulation"]["episode_duration_seconds"])
    total_arrivals = sum(sim.total_arrivals.values())
    departed = sum(sim.total_departed.values())
    ending_queue = sum(sim.queue_lengths.values())
    overflow = sum(sim.total_overflow.values())
    assert total_arrivals == departed + ending_queue + overflow
    ending_wait = sum(sum(queue) for queue in sim.queues.values())
    assert sim.cumulative_wait_vehicle_seconds == sim.departed_wait_sum + ending_wait
