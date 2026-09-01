import numpy as np
import pandas as pd

from resilient_traffic.statistics import (
    build_statistical_summaries,
    controller_summary,
    paired_controller_comparisons,
    per_model_seed_summary,
)


METRIC = "score"


def _row(controller, model_seed, traffic_seed, score, scenario="normal", episode=0):
    return {
        "controller": controller,
        "trained_model_seed": model_seed,
        "scenario": scenario,
        "traffic_seed": traffic_seed,
        "episode": episode,
        METRIC: score,
    }


def test_controller_summary_balances_model_seeds_instead_of_pooling_rows():
    episodes = pd.DataFrame(
        [_row("Learned", 0, traffic, 0.0) for traffic in (1, 2, 3)]
        + [_row("Learned", 1, 1, 100.0)]
    )

    result = controller_summary(episodes, samples=500, seed=7, metrics=[METRIC]).iloc[0]

    assert result["mean"] == 50.0
    assert result["n_model_seeds"] == 2
    assert result["n_traffic_seeds"] == 3
    assert result["bootstrap_method"] == "hierarchical_model_seed_then_traffic_seed"


def test_learned_models_are_one_paired_observation_per_traffic_seed():
    rows = []
    for traffic_seed, baseline in ((10, 10.0), (20, 20.0)):
        rows.append(_row("FixedTime", -1, traffic_seed, baseline))
        rows.append(_row("Learned", 0, traffic_seed, baseline + 1.0))
        rows.append(_row("Learned", 1, traffic_seed, baseline + 3.0))
        rows.append(_row("Learned", 2, traffic_seed, baseline + 2.0))
    episodes = pd.DataFrame(rows)

    result = paired_controller_comparisons(
        episodes,
        samples=200,
        seed=11,
        metrics=[METRIC],
        references=["FixedTime"],
    ).iloc[0]

    assert result["mean_difference"] == 2.0
    assert result["n_paired_traffic_seeds"] == 2
    assert result["n_model_replicates_averaged"] == 3
    assert result["ci95_low"] == 2.0
    assert result["ci95_high"] == 2.0


def test_extra_identical_model_replicates_do_not_create_extra_traffic_draws():
    episodes = pd.DataFrame(
        [
            _row("Learned", model_seed, traffic_seed, score)
            for model_seed in (0, 1, 2)
            for traffic_seed, score in ((10, 0.0), (20, 100.0))
        ]
    )

    result = controller_summary(episodes, samples=2_000, seed=23, metrics=[METRIC]).iloc[0]

    assert result["n_model_seeds"] == 3
    assert result["n_traffic_seeds"] == 2
    assert result["ci95_low"] == 0.0
    assert result["ci95_high"] == 100.0


def test_paired_comparison_uses_only_shared_traffic_seeds():
    episodes = pd.DataFrame(
        [
            _row("Actuated", -1, 1, 10.0),
            _row("Actuated", -1, 2, 20.0),
            _row("Learned", 0, 2, 25.0),
            _row("Learned", 0, 3, 999.0),
        ]
    )

    result = paired_controller_comparisons(
        episodes, samples=20, seed=3, metrics=[METRIC], references=["Actuated"]
    ).iloc[0]

    assert result["n_paired_traffic_seeds"] == 1
    assert result["mean_difference"] == 5.0


def test_repeated_episodes_are_collapsed_inside_traffic_seed():
    episodes = pd.DataFrame(
        [
            _row("FixedTime", -1, 1, 0.0, episode=0),
            _row("FixedTime", -1, 1, 10.0, episode=1),
            _row("FixedTime", -1, 2, 15.0, episode=0),
        ]
    )

    result = per_model_seed_summary(episodes, samples=50, seed=5, metrics=[METRIC]).iloc[0]

    assert result["mean"] == 10.0
    assert result["n_traffic_seeds"] == 2
    assert result["n_episode_rows"] == 3


def test_statistics_are_seeded_and_bundle_returns_all_tables():
    episodes = pd.DataFrame(
        [_row("FixedTime", -1, seed, value) for seed, value in enumerate((1.0, 2.0, 8.0))]
    )

    first = build_statistical_summaries(episodes, samples=100, seed=19, metrics=[METRIC])
    second = build_statistical_summaries(episodes, samples=100, seed=19, metrics=[METRIC])

    assert len(first) == 3
    for left, right in zip(first, second):
        pd.testing.assert_frame_equal(left, right)
    assert np.isfinite(first[1].loc[0, "ci95_low"])
