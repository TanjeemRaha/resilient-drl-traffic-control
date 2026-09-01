from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from resilient_traffic.training_curves import aligned_controller_means, load_monitor_curves, plot_training_curves


def _monitor(root, run, rewards, lengths):
    directory = root / run
    directory.mkdir(parents=True)
    path = directory / "monitor.csv"
    path.write_text('#{"t_start": 0, "env_id": "test"}\nr,l,t\n', encoding="utf-8")
    pd.DataFrame({"r": rewards, "l": lengths, "t": range(len(rewards))}).to_csv(
        path, mode="a", header=False, index=False)


def test_monitor_seed_histories_are_never_concatenated(tmp_path):
    _monitor(tmp_path, "PPO_standard_seed0", [1.0, 2.0], [10, 10])
    _monitor(tmp_path, "PPO_standard_seed1", [10.0, 20.0], [100, 100])
    curves = load_monitor_curves(tmp_path, rolling_window=1)
    seed0 = curves[curves.model_seed == 0]
    seed1 = curves[curves.model_seed == 1]
    assert seed0.training_timesteps.tolist() == [10.0, 20.0]
    assert seed1.training_timesteps.tolist() == [100.0, 200.0]
    assert seed0.episode.tolist() == [1, 2]
    assert seed1.episode.tolist() == [1, 2]

    fig, ax = plt.subplots()
    plot_training_curves(ax, tmp_path, {"PPO_standard": "blue"}, rolling_window=1)
    seed_lines = [line for line in ax.lines if "seed" in line.get_label()]
    assert len(seed_lines) == 2
    assert all(len(line.get_xdata()) == 2 for line in seed_lines)
    assert not any(line.get_xdata().tolist() == [10.0, 20.0, 100.0, 200.0] for line in ax.lines)
    plt.close(fig)


def test_quick_single_seed_has_no_invented_seed_band_or_mean_line(tmp_path):
    _monitor(tmp_path, "DQN_standard_seed0", [1.0, 3.0], [5, 5])
    fig, ax = plt.subplots()
    curves, means = plot_training_curves(ax, tmp_path, {"DQN_standard": "green"}, rolling_window=1)
    assert curves.model_seed.unique().tolist() == [0]
    assert means.n_model_seeds.unique().tolist() == [1]
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == "DQN_standard seed 0"
    assert not ax.collections
    plt.close(fig)


def test_multi_seed_mean_uses_common_timestep_grid(tmp_path):
    _monitor(tmp_path, "DQN_standard_seed0", [0.0, 2.0], [10, 10])
    _monitor(tmp_path, "DQN_standard_seed1", [2.0, 4.0], [10, 10])
    means = aligned_controller_means(load_monitor_curves(tmp_path, rolling_window=1), grid_points=3)
    assert means.training_timesteps.tolist() == [10.0, 15.0, 20.0]
    assert means.mean_reward.tolist() == [1.0, 2.0, 3.0]
