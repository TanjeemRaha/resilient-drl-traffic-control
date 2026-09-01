import numpy as np
import pandas as pd

from resilient_traffic.metrics import bootstrap_summary, episode_metrics, recovery_metric


def test_recovery_detected():
    records = pd.DataFrame({"time": range(1, 401), "total_queue": [5] * 100 + [20] * 100 + [5] * 200})
    value, recovered = recovery_metric(records, (101, 200))
    assert recovered and value >= 119


def test_missing_recovery_is_nan():
    records = pd.DataFrame({"time": range(1, 301), "total_queue": [5] * 100 + [20] * 200})
    value, recovered = recovery_metric(records, (101, 200))
    assert not recovered and np.isnan(value)


def test_bootstrap_is_seeded():
    frame = pd.DataFrame({"controller": ["x", "x"], "trained_model_seed": [0, 0],
        "scenario": ["normal", "normal"], **{m: [1.0, 2.0] for m in [
        "mean_wait_departed_seconds", "time_average_total_queue_vehicles", "max_total_queue_vehicles",
        "max_individual_wait_seconds", "throughput_vehicles", "overflow_vehicles",
        "cumulative_delay_vehicle_seconds", "signal_switches", "directional_queue_imbalance_vehicles",
        "recovery_time_seconds", "ending_queue_vehicles", "accepted_arrivals_vehicles",
        "mean_accumulated_wait_all_accepted_vehicles_seconds"]}})
    assert bootstrap_summary(frame, 20, 2).equals(bootstrap_summary(frame, 20, 2))


def test_episode_metrics_use_exact_wait_and_include_censored_queue():
    records = pd.DataFrame({"time": [1, 2], "total_queue": [2, 2], "throughput": [1, 1],
        "overflow": [0, 1], "queue_N": [2, 2], "queue_S": [0, 0],
        "queue_E": [0, 0], "queue_W": [0, 0]})
    values = episode_metrics(records, departed_wait_sum=8.0, departed_wait_max=5.0,
        switches=1, disruption=None, cumulative_wait_vehicle_seconds=22.0,
        ending_wait_sum=14.0, ending_wait_max=9.0, total_arrivals=5, total_overflow=1)
    assert values["cumulative_delay_vehicle_seconds"] == 22.0
    assert values["max_individual_wait_seconds"] == 9.0
    assert values["ending_queue_vehicles"] == 2
    assert values["accepted_arrivals_vehicles"] == 4
    assert values["mean_accumulated_wait_all_accepted_vehicles_seconds"] == 5.5
