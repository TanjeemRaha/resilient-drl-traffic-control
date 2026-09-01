import pandas as pd
import pytest

from resilient_traffic.behavior_audit import model_behavior_audit


def _row(controller, model_seed, scenario, traffic_seed, decision, action, second=0):
    return {
        "controller": controller,
        "trained_model_seed": model_seed,
        "scenario": scenario,
        "traffic_seed": traffic_seed,
        "episode": 0,
        "decision": decision,
        "requested_action": action,
        "time": second,
    }


def test_audit_collapses_second_rows_and_reports_each_pair_and_scenario():
    rows = []
    actions = {
        "DQN_standard": {"normal": [0, 1], "incident": [1, 1]},
        "PPO_standard": {"normal": [0, 0], "incident": [1, 1]},
        "PPO_resilient": {"normal": [1, 0], "incident": [0, 1]},
    }
    for controller, scenarios in actions.items():
        for scenario, selected in scenarios.items():
            for decision, action in enumerate(selected):
                rows.extend(
                    _row(controller, 11, scenario, 100, decision, action, second)
                    for second in range(5)
                )
    rows.append(_row("FixedTime", -1, "normal", 100, 0, 0))

    result = model_behavior_audit(pd.DataFrame(rows))

    assert len(result) == 6
    dqn_ppo_normal = result[
        (result.controller_a == "DQN_standard")
        & (result.controller_b == "PPO_standard")
        & (result.scenario == "normal")
    ].iloc[0]
    assert dqn_ppo_normal.evaluated_decisions == 2
    assert dqn_ppo_normal.action_disagreements == 1
    assert dqn_ppo_normal.action_agreement_percentage == pytest.approx(50.0)


def test_model_seeds_are_matched_without_a_cartesian_product():
    rows = [
        _row("A", 1, "normal", 100, 0, 0),
        _row("A", 2, "normal", 100, 0, 1),
        _row("A", 3, "normal", 100, 0, 0),
        _row("B", 1, "normal", 100, 0, 0),
        _row("B", 2, "normal", 100, 0, 0),
        _row("B", 4, "normal", 100, 0, 1),
    ]

    audit = model_behavior_audit(pd.DataFrame(rows)).iloc[0]

    assert audit.evaluated_decisions == 2
    assert audit.action_disagreements == 1
    assert audit.action_agreement_percentage == pytest.approx(50.0)


def test_identical_actions_are_reported_as_complete_agreement():
    rows = [
        _row(controller, 7, "smoke", 100, decision, action)
        for controller in ("A", "B")
        for decision, action in enumerate((0, 1, 1))
    ]

    audit = model_behavior_audit(pd.DataFrame(rows)).iloc[0]

    assert audit.evaluated_decisions == 3
    assert audit.action_disagreements == 0
    assert audit.action_agreement_percentage == pytest.approx(100.0)


def test_conflicting_actions_within_one_decision_are_rejected():
    rows = [
        _row("A", 1, "normal", 100, 0, 0, second=1),
        _row("A", 1, "normal", 100, 0, 1, second=2),
        _row("B", 1, "normal", 100, 0, 0, second=1),
    ]

    with pytest.raises(ValueError, match="multiple actions"):
        model_behavior_audit(pd.DataFrame(rows))
