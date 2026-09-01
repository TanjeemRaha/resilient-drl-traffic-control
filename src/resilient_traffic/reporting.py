"""Profile-aware, evidence-limited research reporting from saved summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ROOT


REPORTED_METRICS: tuple[str, ...] = (
    "mean_accumulated_wait_all_accepted_vehicles_seconds",
    "time_average_total_queue_vehicles",
    "throughput_vehicles",
    "ending_queue_vehicles",
    "overflow_vehicles",
    "recovery_time_seconds",
)

METRIC_LABELS = {
    "mean_accumulated_wait_all_accepted_vehicles_seconds":
        "mean accumulated wait across accepted vehicles (s/vehicle)",
    "time_average_total_queue_vehicles": "time-average queue (vehicles)",
    "throughput_vehicles": "throughput (vehicles)",
    "ending_queue_vehicles": "ending queue (vehicles)",
    "overflow_vehicles": "overflow (vehicles)",
    "recovery_time_seconds": "recovery time (s)",
}

_SUMMARY_COLUMNS = {"controller", "scenario", "metric", "mean"}
_PAIRED_COLUMNS = {
    "controller", "reference_controller", "scenario", "metric",
    "mean_difference", "ci95_low", "ci95_high",
}
_BEHAVIOR_COLUMNS = {
    "controller_a", "controller_b", "scenario", "evaluated_decisions",
    "action_agreement_percentage", "action_disagreements",
}


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required summary file is unavailable: {path}")
    frame = pd.read_csv(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return frame


def _number(value: Any) -> str:
    """Format a measured value without representing missing recovery as zero."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not observed"
    if not np.isfinite(numeric):
        return "not observed"
    return f"{numeric:.3f}"


def _scenario_order(config: dict[str, Any], summary: pd.DataFrame) -> list[str]:
    configured = list(config.get("scenarios", {}))
    return configured or sorted(summary["scenario"].astype(str).unique())


def _append_controller_means(
    lines: list[str], config: dict[str, Any], summary: pd.DataFrame, *, full: bool,
) -> None:
    scenarios = _scenario_order(config, summary)
    controllers = sorted(summary["controller"].astype(str).unique())
    selected = summary[summary["metric"].isin(REPORTED_METRICS)].copy()

    if full:
        keys = set(zip(selected["controller"], selected["scenario"], selected["metric"]))
        missing = [
            (controller, scenario, metric)
            for controller in controllers
            for scenario in scenarios
            for metric in REPORTED_METRICS
            if (controller, scenario, metric) not in keys
        ]
        if missing:
            preview = ", ".join("/".join(item) for item in missing[:5])
            raise ValueError(
                "Full-profile controller summary lacks configured controller/scenario/metric "
                f"rows: {preview}"
            )

    heading = "## Measured controller means across every scenario"
    if not full:
        heading = "## Smoke-test controller means across configured scenarios"
    lines.extend([heading, ""])
    for scenario in scenarios:
        lines.extend([f"### {scenario}", ""])
        scenario_rows = selected[selected["scenario"] == scenario]
        for controller in controllers:
            values = scenario_rows[scenario_rows["controller"] == controller]
            lookup = values.drop_duplicates("metric").set_index("metric")["mean"].to_dict()
            rendered = "; ".join(
                f"{METRIC_LABELS[metric]}: {_number(lookup.get(metric))}"
                for metric in REPORTED_METRICS
            )
            lines.append(f"- {controller}: {rendered}.")
        lines.append("")


def _append_paired_results(lines: list[str], paired: pd.DataFrame, *, full: bool) -> None:
    selected = paired[
        paired["metric"].isin(REPORTED_METRICS)
        & paired["reference_controller"].isin(("FixedTime", "Actuated"))
    ].copy()
    if full:
        missing_references = {"FixedTime", "Actuated"}.difference(
            selected["reference_controller"].unique()
        )
        if missing_references:
            raise ValueError(
                "Full-profile paired comparisons lack reference controllers: "
                f"{sorted(missing_references)}"
            )

    heading = "## Paired controller differences"
    if not full:
        heading = "## Smoke-test paired controller differences"
    lines.extend([
        heading,
        "",
        "Each difference is controller minus reference on matched traffic seeds. "
        "Its sign alone is not evidence of superiority.",
        "",
    ])
    sort_columns = ["scenario", "controller", "reference_controller", "metric"]
    for row in selected.sort_values(sort_columns).itertuples(index=False):
        difference = _number(row.mean_difference)
        low, high = _number(row.ci95_low), _number(row.ci95_high)
        try:
            finite_ci = np.isfinite(float(row.ci95_low)) and np.isfinite(float(row.ci95_high))
            includes_zero = finite_ci and float(row.ci95_low) <= 0 <= float(row.ci95_high)
        except (TypeError, ValueError):
            finite_ci = includes_zero = False
        if includes_zero:
            interpretation = "uncertain because its 95% confidence interval includes zero"
        elif finite_ci:
            interpretation = "its 95% confidence interval excludes zero"
        else:
            interpretation = "uncertain because a finite 95% confidence interval is unavailable"
        lines.append(
            f"- {row.scenario}, {row.controller} vs {row.reference_controller}, "
            f"{METRIC_LABELS[row.metric]}: difference {difference}, 95% CI [{low}, {high}]; "
            f"{interpretation}."
        )
    if selected.empty:
        lines.append("- No paired comparisons were available.")
    lines.append("")


def _append_behavior_audit(lines: list[str], audit: pd.DataFrame, *, full: bool) -> None:
    heading = "## Full experiment model-behavior agreement"
    intro = "In the full experiment, requested-action agreement was:"
    if not full:
        heading = "## Smoke-test model-behavior agreement"
        intro = "In these smoke-test measurements, requested-action agreement was:"
    lines.extend([heading, "", intro, ""])
    for row in audit.sort_values(
        ["scenario", "controller_a", "controller_b"]
    ).itertuples(index=False):
        lines.append(
            f"- {row.scenario}, {row.controller_a} vs {row.controller_b}: "
            f"{_number(row.action_agreement_percentage)}% across "
            f"{int(row.evaluated_decisions)} decisions "
            f"({int(row.action_disagreements)} disagreements)."
        )
    if audit.empty:
        lines.append("- No learned-controller behavior comparisons were available.")
    lines.append("")


def write_research_summary(config: dict[str, Any], root: Path = ROOT) -> Path:
    """Write a profile-aware report based only on corrected saved summary CSVs."""
    profile = config.get("profile")
    if profile not in {"quick", "full"}:
        raise ValueError("config.profile must be 'quick' or 'full'")
    summaries = Path(root) / "results" / "summaries"
    controller_summary = _read_csv(
        summaries / "controller_summary.csv", _SUMMARY_COLUMNS
    )
    paired = _read_csv(
        summaries / "paired_controller_comparisons.csv", _PAIRED_COLUMNS
    )
    behavior = _read_csv(
        summaries / "model_behavior_audit.csv", _BEHAVIOR_COLUMNS
    )
    full = profile == "full"
    if full:
        controllers = set(controller_summary["controller"].astype(str))
        scenarios = set(_scenario_order(config, controller_summary))
        available_pairs = set(zip(paired["controller"].astype(str),
                                  paired["reference_controller"].astype(str),
                                  paired["scenario"].astype(str), paired["metric"].astype(str)))
        missing_pairs = [(controller, reference, scenario, metric)
                         for controller in controllers
                         for reference in ("FixedTime", "Actuated")
                         if controller != reference
                         for scenario in scenarios
                         for metric in REPORTED_METRICS
                         if (controller, reference, scenario, metric) not in available_pairs]
        if missing_pairs:
            preview = ", ".join("/".join(item) for item in missing_pairs[:5])
            raise ValueError(f"Full-profile paired comparisons are incomplete: {preview}")

    if full:
        lines = [
            "# Full-profile measured research summary",
            "",
            "Profile: **full** (research).",
            "",
            "These measured results summarize the full experiment. They do not establish "
            "controller superiority solely from the sign of a sample mean.",
            "",
        ]
    else:
        lines = [
            "# Smoke-test measured research summary",
            "",
            f"Profile: **quick** ({config.get('label', 'smoke-test')}).",
            "",
            "Everything below is a smoke-test measurement, not a final research finding. "
            "The small training and traffic-seed sample is insufficient for controller-"
            "superiority claims.",
            "",
        ]

    _append_controller_means(lines, config, controller_summary, full=full)
    _append_paired_results(lines, paired, full=full)
    _append_behavior_audit(lines, behavior, full=full)
    lines.extend([
        "Recovery time is reported as not observed whenever the preregistered sustained-"
        "recovery rule was not met.",
        "",
        "This remains a synthetic proof-of-concept and is not a field-calibrated "
        "intersection study.",
        "",
    ])

    output = summaries / "research_summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
