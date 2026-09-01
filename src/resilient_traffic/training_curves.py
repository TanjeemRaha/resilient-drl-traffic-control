"""Seed-preserving Stable-Baselines3 Monitor curve processing."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import matplotlib.axes
import numpy as np
import pandas as pd

_RUN_PATTERN = re.compile(r"^(?P<controller>.+)_seed(?P<seed>-?\d+)$")
CURVE_COLUMNS = ["controller", "model_seed", "episode", "training_timesteps", "reward", "rolling_reward"]


def load_monitor_curves(log_root: Path, rolling_window: int = 10) -> pd.DataFrame:
    """Load one independent cumulative-timestep history per Monitor file."""
    if rolling_window < 1:
        raise ValueError("rolling_window must be at least one")
    frames: list[pd.DataFrame] = []
    for path in sorted(log_root.glob("*/monitor.csv")):
        match = _RUN_PATTERN.fullmatch(path.parent.name)
        if match is None:
            continue
        try:
            monitor = pd.read_csv(path, comment="#")
        except pd.errors.EmptyDataError:
            continue
        if monitor.empty:
            continue
        missing = {"r", "l"}.difference(monitor.columns)
        if missing:
            raise ValueError(f"Monitor file {path} is missing columns: {sorted(missing)}")
        lengths = pd.to_numeric(monitor["l"], errors="raise")
        rewards = pd.to_numeric(monitor["r"], errors="raise")
        if (lengths <= 0).any():
            raise ValueError(f"Monitor file {path} contains non-positive episode lengths")
        frame = pd.DataFrame({
            "controller": match.group("controller"),
            "model_seed": int(match.group("seed")),
            "episode": np.arange(1, len(monitor) + 1, dtype=int),
            "training_timesteps": lengths.cumsum().to_numpy(float),
            "reward": rewards.to_numpy(float),
            "rolling_reward": rewards.rolling(rolling_window, min_periods=1).mean().to_numpy(float),
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CURVE_COLUMNS)


def aligned_controller_means(curves: pd.DataFrame, grid_points: int = 200) -> pd.DataFrame:
    """Interpolate model-seed curves to a shared within-controller timestep grid."""
    if curves.empty:
        return pd.DataFrame(columns=["controller", "training_timesteps", "mean_reward", "n_model_seeds"])
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    rows: list[pd.DataFrame] = []
    for controller, controller_data in curves.groupby("controller", sort=True):
        seeds = list(controller_data.groupby("model_seed", sort=True))
        if len(seeds) == 1:
            seed_data = seeds[0][1].sort_values("training_timesteps")
            grid = seed_data["training_timesteps"].to_numpy(float)
            values = seed_data["rolling_reward"].to_numpy(float)
        else:
            first_common = max(group["training_timesteps"].min() for _, group in seeds)
            last_common = min(group["training_timesteps"].max() for _, group in seeds)
            if last_common < first_common:
                continue
            grid = np.linspace(first_common, last_common, grid_points)
            interpolated = []
            for _, group in seeds:
                ordered = group.sort_values("training_timesteps")
                interpolated.append(np.interp(grid, ordered["training_timesteps"], ordered["rolling_reward"]))
            values = np.mean(np.vstack(interpolated), axis=0)
        rows.append(pd.DataFrame({"controller": controller, "training_timesteps": grid,
                                  "mean_reward": values, "n_model_seeds": len(seeds)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["controller", "training_timesteps", "mean_reward", "n_model_seeds"])


def plot_training_curves(ax: matplotlib.axes.Axes, log_root: Path,
                         colors: Mapping[str, str], rolling_window: int = 10
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot independent seed histories and their aligned controller means."""
    curves = load_monitor_curves(log_root, rolling_window)
    means = aligned_controller_means(curves)
    for controller, controller_data in curves.groupby("controller", sort=True):
        seeds = list(controller_data.groupby("model_seed", sort=True))
        color = colors.get(controller)
        if len(seeds) == 1:
            seed, data = seeds[0]
            ordered = data.sort_values("training_timesteps")
            ax.plot(ordered["training_timesteps"], ordered["rolling_reward"], color=color,
                    linewidth=2.2, alpha=1.0, label=f"{controller} seed {seed}")
            continue
        for seed, data in seeds:
            ordered = data.sort_values("training_timesteps")
            ax.plot(ordered["training_timesteps"], ordered["rolling_reward"], color=color,
                    linewidth=0.9, alpha=0.35, label=f"{controller} seed {seed}")
        mean = means[means["controller"] == controller]
        ax.plot(mean["training_timesteps"], mean["mean_reward"], color=color,
                linewidth=2.5, alpha=1.0, label=f"{controller} mean")
    return curves, means

