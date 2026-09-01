from __future__ import annotations

from pathlib import Path

import pandas as pd

from resilient_traffic.reporting import (
    METRIC_LABELS,
    REPORTED_METRICS,
    write_research_summary,
)


def _write_artificial_summaries(root: Path) -> None:
    directory = root / "results" / "summaries"
    directory.mkdir(parents=True)
    controllers = ("FixedTime", "Actuated", "DQN_standard")
    scenarios = ("normal", "incident")
    summary_rows = []
    for controller_index, controller in enumerate(controllers):
        for scenario_index, scenario in enumerate(scenarios):
            for metric_index, metric in enumerate(REPORTED_METRICS):
                summary_rows.append({
                    "controller": controller,
                    "scenario": scenario,
                    "metric": metric,
                    "mean": controller_index * 100 + scenario_index * 10 + metric_index,
                })
    pd.DataFrame(summary_rows).to_csv(directory / "controller_summary.csv", index=False)

    paired_rows = []
    for controller in controllers:
        for reference in ("FixedTime", "Actuated"):
            if controller == reference:
                continue
            for scenario in scenarios:
                for metric_index, metric in enumerate(REPORTED_METRICS):
                    paired_rows.append({
                        "controller": controller,
                        "reference_controller": reference,
                        "scenario": scenario,
                        "metric": metric,
                        "mean_difference": -1.0 + metric_index,
                        "ci95_low": -2.0 if metric_index == 0 else 0.25,
                        "ci95_high": 2.0 if metric_index == 0 else 3.0,
                    })
    pd.DataFrame(paired_rows).to_csv(
        directory / "paired_controller_comparisons.csv", index=False
    )

    pd.DataFrame([{
        "controller_a": "DQN_standard",
        "controller_b": "PPO_standard",
        "scenario": "normal",
        "evaluated_decisions": 20,
        "action_agreement_percentage": 75.0,
        "action_disagreements": 5,
    }]).to_csv(directory / "model_behavior_audit.csv", index=False)


def _config(profile: str) -> dict:
    return {
        "profile": profile,
        "label": "smoke-test" if profile == "quick" else "research",
        "scenarios": {"normal": {}, "incident": {}},
    }


def test_quick_summary_uses_smoke_test_wording_without_superiority_claim(tmp_path):
    _write_artificial_summaries(tmp_path)

    output = write_research_summary(_config("quick"), root=tmp_path)
    text = output.read_text(encoding="utf-8").lower()

    assert output == tmp_path / "results" / "summaries" / "research_summary.md"
    assert "smoke-test" in text
    assert "not a final research finding" in text
    assert "superiority claims" in text
    assert "is superior" not in text
    assert "outperformed" not in text


def test_full_summary_reports_all_metrics_pairs_uncertainty_and_behavior(tmp_path):
    _write_artificial_summaries(tmp_path)

    text = write_research_summary(
        _config("full"), root=tmp_path
    ).read_text(encoding="utf-8").lower()

    assert "smoke-test" not in text
    assert "full experiment" in text
    assert "normal" in text and "incident" in text
    for metric in REPORTED_METRICS:
        assert METRIC_LABELS[metric].lower() in text
    assert "vs fixedtime" in text
    assert "vs actuated" in text
    assert "95% ci [-2.000, 2.000]" in text
    assert "uncertain because its 95% confidence interval includes zero" in text
    assert "75.000% across 20 decisions" in text
    assert "synthetic proof-of-concept" in text
    assert "outperformed" not in text


def test_full_summary_rejects_incomplete_scenario_metric_coverage(tmp_path):
    _write_artificial_summaries(tmp_path)
    path = tmp_path / "results" / "summaries" / "controller_summary.csv"
    summary = pd.read_csv(path)
    summary = summary[~(
        (summary["controller"] == "DQN_standard")
        & (summary["scenario"] == "incident")
        & (summary["metric"] == REPORTED_METRICS[-1])
    )]
    summary.to_csv(path, index=False)

    try:
        write_research_summary(_config("full"), root=tmp_path)
    except ValueError as exc:
        assert "lacks configured controller/scenario/metric rows" in str(exc)
    else:
        raise AssertionError("Incomplete full-profile summaries must be rejected")
