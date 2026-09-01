"""Uncertainty estimates that respect trained-model and traffic-seed structure.

Evaluation rows for learned controllers form a crossed design: every trained
model seed is evaluated on the same traffic seeds.  Consequently, model
replicates are not additional independent traffic observations.  This module
first reduces repeated episodes to model-seed/traffic-seed cells and then uses
separate resampling levels for the two sources of variation.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import METRICS

BASELINE_CONTROLLERS = frozenset({"FixedTime", "Actuated"})
IDENTIFIER_COLUMNS = ("controller", "trained_model_seed", "scenario", "traffic_seed")


def _metric_names(episodes: pd.DataFrame, metrics: Iterable[str] | None) -> list[str]:
    names = list(METRICS if metrics is None else metrics)
    missing = [name for name in names if name not in episodes.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")
    return names


def _validate(episodes: pd.DataFrame, samples: int) -> None:
    missing = [name for name in IDENTIFIER_COLUMNS if name not in episodes.columns]
    if missing:
        raise ValueError(f"Missing identifier columns: {missing}")
    if samples < 1:
        raise ValueError("samples must be at least one")


def _episode_cells(episodes: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Average repeated episodes without pretending they are new seed draws."""
    return (
        episodes.groupby(list(IDENTIFIER_COLUMNS), as_index=False, dropna=False)[list(metrics)]
        .mean()
    )


def _interval(draws: np.ndarray) -> tuple[float, float, float]:
    if draws.size == 1:
        value = float(draws[0])
        return 0.0, value, value
    std = float(draws.std(ddof=1))
    low, high = np.percentile(draws, [2.5, 97.5])
    return std, float(low), float(high)


def _traffic_bootstrap(values: np.ndarray, samples: int, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return values[indices].mean(axis=1)


def per_model_seed_summary(
    episodes: pd.DataFrame,
    samples: int,
    seed: int = 2026,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return traffic-seed bootstrap summaries for each individual model seed.

    Fixed-time and actuated controllers are retained with their sentinel model
    seed so this table can also serve as a complete seed-level audit table.
    """
    _validate(episodes, samples)
    metric_names = _metric_names(episodes, metrics)
    cells = _episode_cells(episodes, metric_names)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    group_columns = ["controller", "trained_model_seed", "scenario"]
    for keys, group in cells.groupby(group_columns, sort=True, dropna=False):
        controller, model_seed, scenario = keys
        for metric in metric_names:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size:
                draws = _traffic_bootstrap(values, samples, rng)
                bootstrap_std, low, high = _interval(draws)
                mean = float(values.mean())
                sample_std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            else:
                mean = sample_std = bootstrap_std = low = high = np.nan
            rows.append(
                {
                    "controller": controller,
                    "trained_model_seed": model_seed,
                    "scenario": scenario,
                    "metric": metric,
                    "mean": mean,
                    "std": sample_std,
                    "bootstrap_std": bootstrap_std,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_traffic_seeds": int(group.loc[group[metric].notna(), "traffic_seed"].nunique()),
                    "n_episode_rows": int(
                        episodes.loc[
                            (episodes["controller"] == controller)
                            & (episodes["trained_model_seed"] == model_seed)
                            & (episodes["scenario"] == scenario)
                            & episodes[metric].notna()
                        ].shape[0]
                    ),
                    "bootstrap_method": "traffic_seed",
                }
            )
    return pd.DataFrame(rows)


def _hierarchical_bootstrap(
    group: pd.DataFrame,
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, int, int]:
    pivot = group.pivot_table(
        index="trained_model_seed", columns="traffic_seed", values=metric, aggfunc="mean"
    )
    pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if pivot.empty:
        return np.nan, np.array([], dtype=float), 0, 0

    values = pivot.to_numpy(dtype=float)
    model_means = np.nanmean(values, axis=1)
    point_estimate = float(model_means.mean())
    n_models, n_traffic = values.shape
    model_draws = rng.integers(0, n_models, size=(samples, n_models))
    traffic_draws = rng.integers(0, n_traffic, size=(samples, n_traffic))
    draws = np.empty(samples, dtype=float)
    for draw_index, (sampled_models, sampled_traffic) in enumerate(
        zip(model_draws, traffic_draws)
    ):
        sampled_means = []
        for model_index in sampled_models:
            traffic_values = values[int(model_index), sampled_traffic]
            traffic_values = traffic_values[np.isfinite(traffic_values)]
            if traffic_values.size:
                sampled_means.append(float(traffic_values.mean()))
        draws[draw_index] = float(np.mean(sampled_means)) if sampled_means else np.nan
    return point_estimate, draws[np.isfinite(draws)], n_models, n_traffic


def controller_summary(
    episodes: pd.DataFrame,
    samples: int,
    seed: int = 2026,
    metrics: Iterable[str] | None = None,
    baseline_controllers: Iterable[str] = BASELINE_CONTROLLERS,
) -> pd.DataFrame:
    """Summarize controllers using the uncertainty structure appropriate to each.

    Learned controllers use a two-factor hierarchical bootstrap: trained-model
    labels and shared traffic-seed labels are sampled separately.  The same
    sampled traffic labels are used for all sampled models because traffic seeds
    are crossed with models in this common-random-number design.  Baselines have
    no trained-model uncertainty and use a traffic-seed bootstrap only.
    """
    _validate(episodes, samples)
    metric_names = _metric_names(episodes, metrics)
    baselines = frozenset(baseline_controllers)
    cells = _episode_cells(episodes, metric_names)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for keys, group in cells.groupby(["controller", "scenario"], sort=True, dropna=False):
        controller, scenario = keys
        learned = controller not in baselines
        for metric in metric_names:
            if learned:
                mean, draws, n_models, n_traffic = _hierarchical_bootstrap(
                    group, metric, samples, rng
                )
                method = "hierarchical_model_seed_then_traffic_seed"
            else:
                traffic_values = (
                    group.groupby("traffic_seed", dropna=False)[metric].mean().dropna().to_numpy(float)
                )
                mean = float(traffic_values.mean()) if traffic_values.size else np.nan
                draws = (
                    _traffic_bootstrap(traffic_values, samples, rng)
                    if traffic_values.size
                    else np.array([], dtype=float)
                )
                n_models = 0
                n_traffic = int(traffic_values.size)
                method = "traffic_seed"
            if draws.size:
                bootstrap_std, low, high = _interval(draws)
            else:
                bootstrap_std = low = high = np.nan
            rows.append(
                {
                    "controller": controller,
                    "scenario": scenario,
                    "metric": metric,
                    "mean": mean,
                    "std": bootstrap_std,
                    "bootstrap_std": bootstrap_std,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_model_seeds": n_models,
                    "n_traffic_seeds": n_traffic,
                    "n_episode_rows": int(
                        episodes.loc[
                            (episodes["controller"] == controller)
                            & (episodes["scenario"] == scenario)
                            & episodes[metric].notna()
                        ].shape[0]
                    ),
                    "bootstrap_method": method,
                }
            )
    return pd.DataFrame(rows)


def paired_controller_comparisons(
    episodes: pd.DataFrame,
    samples: int,
    seed: int = 2026,
    metrics: Iterable[str] | None = None,
    references: Iterable[str] = ("FixedTime", "Actuated"),
    baseline_controllers: Iterable[str] = BASELINE_CONTROLLERS,
) -> pd.DataFrame:
    """Return traffic-seed-paired differences (controller minus reference).

    Before pairing, learned-model replicates are averaged within each
    scenario/traffic-seed cell.  Thus three trained models evaluated under one
    traffic seed contribute one paired observation, not three.
    """
    _validate(episodes, samples)
    metric_names = _metric_names(episodes, metrics)
    baselines = frozenset(baseline_controllers)
    references = tuple(references)
    cells = _episode_cells(episodes, metric_names)

    collapsed_frames: list[pd.DataFrame] = []
    for controller, group in cells.groupby("controller", sort=True, dropna=False):
        collapsed = (
            group.groupby(["scenario", "traffic_seed"], as_index=False, dropna=False)[metric_names]
            .mean()
        )
        collapsed["controller"] = controller
        collapsed["model_replicates_averaged"] = (
            int(group["trained_model_seed"].nunique()) if controller not in baselines else 0
        )
        collapsed_frames.append(collapsed)
    collapsed_cells = pd.concat(collapsed_frames, ignore_index=True)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    controllers = sorted(collapsed_cells["controller"].dropna().unique())
    scenarios = sorted(collapsed_cells["scenario"].dropna().unique())
    for scenario in scenarios:
        scenario_cells = collapsed_cells[collapsed_cells["scenario"] == scenario]
        for controller in controllers:
            target = scenario_cells[scenario_cells["controller"] == controller]
            if target.empty:
                continue
            for reference in references:
                if controller == reference:
                    continue
                reference_rows = scenario_cells[scenario_cells["controller"] == reference]
                if reference_rows.empty:
                    continue
                for metric in metric_names:
                    paired = target[["traffic_seed", metric]].merge(
                        reference_rows[["traffic_seed", metric]],
                        on="traffic_seed",
                        how="inner",
                        suffixes=("_controller", "_reference"),
                    ).dropna()
                    target_values = paired[f"{metric}_controller"].to_numpy(float)
                    reference_values = paired[f"{metric}_reference"].to_numpy(float)
                    differences = target_values - reference_values
                    if differences.size:
                        draws = _traffic_bootstrap(differences, samples, rng)
                        bootstrap_std, low, high = _interval(draws)
                        target_mean = float(target_values.mean())
                        reference_mean = float(reference_values.mean())
                        mean_difference = float(differences.mean())
                    else:
                        target_mean = reference_mean = mean_difference = np.nan
                        bootstrap_std = low = high = np.nan
                    rows.append(
                        {
                            "controller": controller,
                            "reference_controller": reference,
                            "scenario": scenario,
                            "metric": metric,
                            "controller_mean": target_mean,
                            "reference_mean": reference_mean,
                            "mean_difference": mean_difference,
                            "std": bootstrap_std,
                            "bootstrap_std": bootstrap_std,
                            "ci95_low": low,
                            "ci95_high": high,
                            "n_paired_traffic_seeds": int(differences.size),
                            "n_model_replicates_averaged": int(
                                target["model_replicates_averaged"].iloc[0]
                            ),
                            "difference_definition": "controller_minus_reference",
                            "bootstrap_method": "paired_traffic_seed",
                        }
                    )
    return pd.DataFrame(rows)


def build_statistical_summaries(
    episodes: pd.DataFrame,
    samples: int,
    seed: int = 2026,
    metrics: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the three CSV-ready tables required by the evaluation pipeline."""
    return (
        per_model_seed_summary(episodes, samples, seed, metrics),
        controller_summary(episodes, samples, seed, metrics),
        paired_controller_comparisons(episodes, samples, seed, metrics),
    )
