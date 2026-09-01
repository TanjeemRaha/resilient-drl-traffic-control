"""Decision-level comparisons of learned-controller behavior.

Evaluation records contain one row per simulated second, so the action selected at
one decision epoch is normally repeated across several rows.  This module first
collapses those rows to one action per model-run decision.  Controller families
are then compared only where ``trained_model_seed``, ``traffic_seed``, ``episode``,
and ``decision`` all match.  In particular, model seeds are *not* cross-joined:
the reported count is a count of matched model-run decisions, with each trained
replicate retained as a separate behavior observation.
"""
from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import numpy as np
import pandas as pd


BASELINE_CONTROLLERS = frozenset({"FixedTime", "Actuated"})
AUDIT_COLUMNS = [
    "controller_a",
    "controller_b",
    "scenario",
    "evaluated_decisions",
    "action_agreement_percentage",
    "action_disagreements",
]
_DECISION_KEY = ["trained_model_seed", "traffic_seed", "episode", "decision"]


def model_behavior_audit(
    records: pd.DataFrame,
    learned_controllers: Iterable[str] | None = None,
    *,
    action_column: str = "requested_action",
) -> pd.DataFrame:
    """Return action agreement for every learned-controller pair and scenario.

    Parameters
    ----------
    records:
        Either per-second evaluation records or records already reduced to one
        row per decision.  Required identifier columns are ``controller``,
        ``trained_model_seed``, ``scenario``, ``traffic_seed``, ``episode``, and
        ``decision``.
    learned_controllers:
        Controller names to compare.  By default, every controller except the
        two rule-based baselines (``FixedTime`` and ``Actuated``) is used.
        Names not present in ``records`` are ignored.
    action_column:
        The action to compare.  ``requested_action`` is the default because it
        reflects model behavior even when signal-transition constraints cause
        the applied action to differ temporarily.

    Notes
    -----
    Trained-model seeds are matched by their seed value and are never combined
    in a Cartesian product.  Traffic decisions lacking a matching model seed or
    decision for either controller are excluded.  Consequently,
    ``evaluated_decisions`` is the number of comparable *model-run decisions*,
    rather than the number of unique traffic decisions after pooling model
    replicates.
    """
    required = {
        "controller",
        "trained_model_seed",
        "scenario",
        "traffic_seed",
        "episode",
        "decision",
        action_column,
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Evaluation records are missing required columns: {missing}")

    present = set(records["controller"].dropna().astype(str))
    if learned_controllers is None:
        controllers = sorted(present.difference(BASELINE_CONTROLLERS))
    else:
        controllers = list(dict.fromkeys(str(name) for name in learned_controllers))
        controllers = [name for name in controllers if name in present]

    if len(controllers) < 2:
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    learned = records.loc[records["controller"].astype(str).isin(controllers)].copy()
    learned["controller"] = learned["controller"].astype(str)
    if learned[action_column].isna().any():
        raise ValueError(f"{action_column!r} contains missing values")

    unique_key = ["controller", "scenario", *_DECISION_KEY]
    action_counts = learned.groupby(unique_key, dropna=False)[action_column].nunique(dropna=False)
    inconsistent = action_counts[action_counts > 1]
    if not inconsistent.empty:
        example = inconsistent.index[0]
        raise ValueError(
            "A model-run decision has multiple actions; cannot collapse per-second rows "
            f"unambiguously (example key: {example!r})"
        )

    decisions = learned.drop_duplicates(unique_key, keep="first")
    scenarios = list(pd.unique(decisions["scenario"]))
    try:
        scenarios.sort()
    except TypeError:
        scenarios.sort(key=str)

    rows: list[dict[str, object]] = []
    for controller_a, controller_b in combinations(controllers, 2):
        for scenario in scenarios:
            in_scenario = decisions["scenario"].eq(scenario)
            left = decisions.loc[
                in_scenario & decisions["controller"].eq(controller_a),
                [*_DECISION_KEY, action_column],
            ].rename(columns={action_column: "action_a"})
            right = decisions.loc[
                in_scenario & decisions["controller"].eq(controller_b),
                [*_DECISION_KEY, action_column],
            ].rename(columns={action_column: "action_b"})
            matched = left.merge(right, on=_DECISION_KEY, how="inner", validate="one_to_one")
            evaluated = int(len(matched))
            disagreements = int(matched["action_a"].ne(matched["action_b"]).sum())
            agreement = (
                100.0 * (evaluated - disagreements) / evaluated
                if evaluated
                else np.nan
            )
            rows.append(
                {
                    "controller_a": controller_a,
                    "controller_b": controller_b,
                    "scenario": scenario,
                    "evaluated_decisions": evaluated,
                    "action_agreement_percentage": agreement,
                    "action_disagreements": disagreements,
                }
            )

    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)
