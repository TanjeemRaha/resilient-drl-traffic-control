"""Publication-style figures generated only from saved experiment CSV data."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ROOT
from .training_curves import plot_training_curves

COLORS = {"FixedTime": "#4C78A8", "Actuated": "#F58518", "DQN_standard": "#54A24B",
          "PPO_standard": "#E45756", "PPO_resilient": "#B279A2"}


def _save(fig: plt.Figure, name: str) -> None:
    directory = ROOT / "results" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _bar(summary: pd.DataFrame, metric: str, ylabel: str, title: str, name: str) -> None:
    """Plot corrected precomputed means and confidence intervals."""
    data = summary[summary["metric"] == metric].copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    scenarios = list(dict.fromkeys(data["scenario"].tolist()))
    controllers = [name for name in COLORS if name in set(data["controller"])]
    x = np.arange(len(scenarios), dtype=float)
    width = 0.8 / max(len(controllers), 1)
    for index, controller in enumerate(controllers):
        rows = data[data["controller"] == controller].set_index("scenario").reindex(scenarios)
        means = rows["mean"].to_numpy(float)
        lows = rows["ci95_low"].to_numpy(float)
        highs = rows["ci95_high"].to_numpy(float)
        errors = np.vstack([np.maximum(means - lows, 0), np.maximum(highs - means, 0)])
        positions = x - 0.4 + width / 2 + index * width
        ax.bar(positions, means, width, label=controller, color=COLORS[controller],
               yerr=errors, capsize=2, linewidth=0.5, edgecolor="white")
    ax.set(title=title, xlabel="Scenario", ylabel=ylabel)
    ax.set_xticks(x, scenarios, rotation=20)
    ax.legend(title="Controller", fontsize=8)
    _save(fig, name)


def _training_curves() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    curves, _ = plot_training_curves(ax, ROOT / "logs", COLORS, rolling_window=10)
    if not curves.empty:
        ax.legend(title="Controller")
    else:
        ax.text(0.5, 0.5, "No training monitor records", ha="center", va="center")
    ax.set(title="Training Reward by Independent Model Seed (10-Episode Rolling Mean)",
           xlabel="Cumulative training timesteps", ylabel="Episode reward")
    _save(fig, "08_training_reward_curves")


def plot_baseline_preview(raw: pd.DataFrame) -> None:
    """Save a preliminary baseline traffic-state plot immediately after phase 4."""
    sample = raw[(raw["scenario"] == "combined_unseen") &
                 (raw["traffic_seed"] == raw["traffic_seed"].min())]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=sample, x="time", y="total_queue", hue="controller",
                 palette=COLORS, estimator=None, units="controller", ax=ax)
    ax.axvspan(600, 1200, color="grey", alpha=0.15)
    ax.set(title="Preliminary Baseline Queue State (No Conclusions)", xlabel="Time (s)",
           ylabel="Total queue (vehicles)")
    _save(fig, "00_preliminary_baseline_queue")


def generate_figures(config: dict[str, Any]) -> list[Path]:
    """Read saved evaluation results and emit ten figures as 300-dpi PNG and PDF."""
    episode_path = ROOT / "results" / "summaries" / "episode_metrics.csv"
    summary_path = ROOT / "results" / "summaries" / "controller_summary.csv"
    raw_path = ROOT / "results" / "raw" / "evaluation_records.csv"
    deterioration_path = ROOT / "results" / "summaries" / "performance_deterioration.csv"
    if not episode_path.exists() or not raw_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Evaluation results are unavailable. Run evaluate_controllers.py first.")
    episodes, raw, summary = pd.read_csv(episode_path), pd.read_csv(raw_path), pd.read_csv(summary_path)
    sns.set_theme(style="whitegrid", context="paper")
    _bar(summary, "mean_wait_departed_seconds", "Mean wait (s/departed vehicle)",
         "Average Waiting Time by Scenario", "01_average_waiting_time")
    _bar(summary, "time_average_total_queue_vehicles", "Time-average queue (vehicles)",
         "Average Total Queue by Scenario", "02_average_queue")
    _bar(summary, "throughput_vehicles", "Throughput (vehicles/episode)",
         "Intersection Throughput by Scenario", "03_throughput")
    _bar(summary, "overflow_vehicles", "Unserved arrivals (vehicles/episode)",
         "Queue Overflow by Scenario", "04_overflow")
    _bar(summary[summary.scenario != "normal"], "recovery_time_seconds", "Recovery time (s)",
         "Post-Disruption Recovery (Missing Means No Recovery)", "05_recovery_time")

    example = raw[(raw.scenario == config["plotting"]["example_scenario"])
                  & (raw.traffic_seed == config["plotting"]["example_traffic_seed"])]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.lineplot(data=example, x="time", y="total_queue", hue="controller", palette=COLORS,
                 estimator="mean", errorbar=None, ax=ax)
    ax.axvspan(600, 1200, color="grey", alpha=0.15, label="Configured event window")
    ax.set(title="Queue Response Before, During, and After Disruption", xlabel="Time (s)",
           ylabel="Total queue (vehicles)")
    _save(fig, "06_queue_time_series")

    if not deterioration_path.exists():
        raise FileNotFoundError(f"Missing deterioration results: {deterioration_path}")
    deterioration = pd.read_csv(deterioration_path)
    subset = deterioration[deterioration.metric == "mean_wait_departed_seconds"]
    table = subset.pivot_table(index="controller", columns="scenario",
                               values="percent_change_from_normal", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(table, annot=True, fmt=".1f", cmap="RdYlBu_r", center=0,
                cbar_kws={"label": "Change from normal (%)"}, ax=ax)
    ax.set(title="Waiting-Time Deterioration Relative to Normal", xlabel="Scenario", ylabel="Controller")
    _save(fig, "07_performance_deterioration_heatmap")
    _training_curves()

    signal = example[example.controller == "PPO_resilient"]
    if signal.empty:
        signal = example[example.controller == "Actuated"]
    signal = signal[signal.trained_model_seed == signal.trained_model_seed.min()]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for approach in "NSEW":
        ax1.plot(signal.time, signal[f"queue_{approach}"], label=approach, linewidth=0.9)
    phase_values = signal.phase.map({"NS_GREEN": 1, "YELLOW": 0, "EW_GREEN": -1})
    ax2.step(signal.time, phase_values, where="post", color="#333333")
    ax1.set(title="Example Signal and Queue Response", ylabel="Queue (vehicles)")
    ax1.legend(ncol=4, title="Approach")
    ax2.set(xlabel="Time (s)", ylabel="Signal state", yticks=[-1, 0, 1],
            yticklabels=["EW", "Yellow", "NS"])
    _save(fig, "09_signal_queue_response")

    components = ["total_queue_normalized", "mean_wait_normalized", "max_wait_normalized",
                  "directional_imbalance_normalized", "spillback_risk_normalized",
                  "overflow_normalized", "switch_penalty"]
    reward_data = signal.groupby("decision", as_index=False)[components].first().head(120)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for component in components:
        ax.plot(reward_data.decision, reward_data[component], label=component.replace("_normalized", ""))
    ax.set(title="Normalized Reward Components: Example Episode", xlabel="Decision step",
           ylabel="Normalized component / penalty")
    ax.legend(fontsize=7, ncol=3)
    _save(fig, "10_reward_components")
    return sorted((ROOT / "results" / "figures").glob("*.png"))


def write_research_summary(config: dict[str, Any]) -> Path:
    """Backward-compatible wrapper for profile-aware reporting."""
    from .reporting import write_research_summary as write_profile_summary
    return write_profile_summary(config, ROOT)
