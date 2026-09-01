"""Episode metrics, recovery, deterioration, and uncertainty."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

METRICS = ["mean_wait_departed_seconds", "time_average_total_queue_vehicles",
           "max_total_queue_vehicles", "max_individual_wait_seconds", "throughput_vehicles",
           "overflow_vehicles", "cumulative_delay_vehicle_seconds", "ending_queue_vehicles",
           "accepted_arrivals_vehicles", "mean_accumulated_wait_all_accepted_vehicles_seconds",
           "signal_switches", "directional_queue_imbalance_vehicles", "recovery_time_seconds"]


def recovery_metric(records: pd.DataFrame, window: tuple[int, int] | None) -> tuple[float, bool]:
    """Apply the preregistered 120-s baseline and sustained 60-s recovery definition."""
    if window is None:
        return np.nan, False
    start, end = window
    pre = records.loc[(records.time > max(0, start - 120)) & (records.time <= start), "total_queue"]
    if pre.empty:
        return np.nan, False
    baseline = float(pre.mean())
    threshold = max(1.2 * baseline, baseline + 2.0)
    after = records.loc[records.time > end, ["time", "total_queue"]].copy()
    if len(after) < 60:
        return np.nan, False
    after["rolling"] = after["total_queue"].rolling(60, min_periods=60).mean()
    below = after["rolling"] <= threshold
    sustained = below.rolling(60, min_periods=60).sum() >= 60
    if not sustained.any():
        return np.nan, False
    recovery_time = float(after.loc[sustained, "time"].iloc[0] - end)
    return recovery_time, True


def episode_metrics(records: pd.DataFrame, departed_wait_sum: float, departed_wait_max: float,
                    switches: int, disruption: tuple[int, int] | None, *,
                    cumulative_wait_vehicle_seconds: float, ending_wait_sum: float,
                    ending_wait_max: float, total_arrivals: int, total_overflow: int) -> dict[str, Any]:
    """Calculate episode outcomes using exact simulator accounting counters.

    Departed-vehicle mean wait excludes vehicles censored in the ending queue. The
    all-accepted mean uses the exact accumulated wait counter and includes them.
    """
    throughput = int(records["throughput"].sum())
    ending_queue = int(records["total_queue"].iloc[-1])
    accepted_arrivals = int(total_arrivals - total_overflow)
    recovery, recovered = recovery_metric(records, disruption)
    imbalance = ((records["queue_N"] + records["queue_S"])
                 - (records["queue_E"] + records["queue_W"])).abs()
    return {
        "mean_wait_departed_seconds": departed_wait_sum / throughput if throughput else np.nan,
        "time_average_total_queue_vehicles": float(records["total_queue"].mean()),
        "max_total_queue_vehicles": int(records["total_queue"].max()),
        "max_individual_wait_seconds": float(max(departed_wait_max, ending_wait_max)),
        "throughput_vehicles": throughput,
        "overflow_vehicles": int(total_overflow),
        "cumulative_delay_vehicle_seconds": float(cumulative_wait_vehicle_seconds),
        "ending_queue_vehicles": ending_queue,
        "accepted_arrivals_vehicles": accepted_arrivals,
        "mean_accumulated_wait_all_accepted_vehicles_seconds": (
            float(cumulative_wait_vehicle_seconds) / accepted_arrivals if accepted_arrivals else np.nan),
        "total_arrivals_vehicles": int(total_arrivals),
        "departed_wait_vehicle_seconds": float(departed_wait_sum),
        "ending_wait_vehicle_seconds": float(ending_wait_sum),
        "signal_switches": int(switches),
        "directional_queue_imbalance_vehicles": float(imbalance.mean()),
        "recovery_time_seconds": recovery,
        "recovered": recovered,
    }


def bootstrap_summary(episodes: pd.DataFrame, samples: int, seed: int = 2026) -> pd.DataFrame:
    """Summarize traffic-seed variability with seeded percentile bootstrap intervals."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    groups = episodes.groupby(["controller", "trained_model_seed", "scenario"], dropna=False)
    for keys, group in groups:
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(float)
            if not len(values):
                mean = std = low = high = np.nan
            else:
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                boot = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
                low, high = np.percentile(boot, [2.5, 97.5]).tolist()
            rows.append({"controller": keys[0], "trained_model_seed": keys[1], "scenario": keys[2],
                         "metric": metric, "mean": mean, "std": std, "ci95_low": low,
                         "ci95_high": high, "n_traffic_runs": len(values)})
    return pd.DataFrame(rows)


def performance_deterioration(episodes: pd.DataFrame) -> pd.DataFrame:
    """Calculate paired percent change from normal for each traffic/model seed."""
    id_cols = ["controller", "trained_model_seed", "traffic_seed", "episode"]
    base = episodes[episodes.scenario == "normal"].set_index(id_cols)
    rows: list[dict[str, Any]] = []
    for _, row in episodes[episodes.scenario != "normal"].iterrows():
        key = tuple(row[c] for c in id_cols)
        if key not in base.index:
            continue
        normal = base.loc[key]
        for metric in ("mean_wait_departed_seconds", "time_average_total_queue_vehicles",
                       "throughput_vehicles", "overflow_vehicles"):
            denominator = float(normal[metric])
            change = np.nan if denominator == 0 else 100.0 * (float(row[metric]) - denominator) / denominator
            rows.append({**{c: row[c] for c in id_cols}, "scenario": row.scenario,
                         "metric": metric, "percent_change_from_normal": change})
    return pd.DataFrame(rows)
