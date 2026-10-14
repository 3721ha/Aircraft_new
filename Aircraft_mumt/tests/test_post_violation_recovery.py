from copy import deepcopy

from aircraft_sim import SimConfig
from run_post_violation_recovery_experiments import (
    make_trigger, recovery_metrics, rollout_from_trigger, snapshot_hash,
)
from run_short_recovery_experiments import PAPER_PROTOCOL_OVERRIDES, scenario_at_distance


def states(flags):
    return [{"restored_safe_state": safe, "rule_clear": safe,
             "target_violations": [] if safe else ["S02:0"]} for safe in flags]


def test_recovery_is_entry_after_violation_not_no_violation_in_window():
    result = recovery_metrics(states([False, False, True, True]))
    assert result["first_recovery_step"] == 3
    assert result["recovered_within_3_steps"] == 1
    assert result["confirmed_within_3_steps"] == 0
    assert result["confirmed_recovery_step"] == 4


def test_fourth_action_cannot_count_as_three_step_recovery():
    result = recovery_metrics(states([False, False, False, True, True]))
    assert result["first_recovery_step"] == 4
    assert result["recovered_within_3_steps"] == 0


def test_recurrence_is_not_sustained_recovery():
    result = recovery_metrics(states([True, True, False, True]))
    assert result["confirmed_within_3_steps"] == 1
    assert result["recovered_by_3_and_held_to_30"] == 0
    assert result["unsafe_state_steps_after_first_recovery"] == [3]


def test_s03_grace_period_alone_is_not_recovery():
    data = states([False, False, False])
    for state in data:
        state.update(rule_clear=True, target_violations=[])
    result = recovery_metrics(data)
    assert result["rule_clear_within_3_steps"] == 1
    assert result["recovered_within_3_steps"] == 0


def test_trigger_is_actual_violation_and_copies_do_not_erase_history():
    config = SimConfig(seed=11, horizon=42, **PAPER_PROTOCOL_OVERRIDES)
    snapshot, info = make_trigger(config, scenario_at_distance(2000))
    assert info["initial"]["target_violations"] == []
    assert info["trigger"]["minimum_separation_m"] < config.min_separation
    assert any(rule.startswith("S02") for rule in info["trigger"]["target_violations"])
    expected = snapshot_hash(snapshot)
    for method in ("NoShield", "PartialShield", "JointHeuristic", "DG-QP"):
        metrics, trace = rollout_from_trigger(snapshot, method, 3, 2)
        assert metrics["takeover_snapshot_sha256"] == expected
        assert trace[0]["absolute_state_step"] == info["trigger"]["absolute_state_step"] + 1
        assert snapshot_hash(snapshot) == expected
    cloned = deepcopy(snapshot)
    assert cloned[1]._safe_trend_deadlines == snapshot[1]._safe_trend_deadlines
    assert cloned[1]._last_truth_step == snapshot[0].step_count


def test_missing_trigger_cannot_be_counted_as_success():
    config = SimConfig(seed=11, horizon=42, **PAPER_PROTOCOL_OVERRIDES)
    snapshot, info = make_trigger(config, scenario_at_distance(3200), max_steps=1)
    assert snapshot is None
    assert info["status"] == "trigger_not_reached"
