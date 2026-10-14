from copy import deepcopy

import numpy as np

from aircraft_sim import Action, AircraftEnv, Mode, RuleMonitor, SimConfig
from run_intervention_timing_oracle_experiments import (
    common_prefix, evaluate_reference, one_step_pair_distance_upper_bound,
    predict_controls, timing_metrics,
)
from run_short_recovery_experiments import PAPER_PROTOCOL_OVERRIDES, scenario_at_distance


def config():
    return SimConfig(seed=11, horizon=45, **PAPER_PROTOCOL_OVERRIDES)


def test_projection_matches_actual_environment_with_clipping():
    env = AircraftEnv(3, config())
    env.reset(scenario_at_distance(2000))
    env.aircraft[0].climb_angle = .44
    env.aircraft[1].speed = 315
    controls = np.random.default_rng(7).uniform(-1, 1, (3, 3, 3))
    predicted = predict_controls(env, controls)
    for k, row in enumerate(controls):
        env.step({uid: Action(Mode.RECOVER, turn=u[0], climb=u[1], acceleration=u[2])
                  for uid, u in enumerate(row)})
        np.testing.assert_allclose(predicted["positions"][k], [a.position for a in env.aircraft], atol=1e-10)
        np.testing.assert_allclose(predicted["speeds"][k], [a.speed for a in env.aircraft], atol=1e-10)
        np.testing.assert_allclose(predicted["energies"][k], [a.energy for a in env.aircraft], atol=1e-10)


def test_early_warning_is_not_expired_s03_or_s02_violation():
    snapshots, case = common_prefix(config(), scenario_at_distance(2000))
    assert case["warning_step"] == 0
    assert case["s02_step"] == 3
    assert case["takeovers"]["warning_now"]["state"]["target_violations"] == []
    assert snapshots["first_S02_violation"][1]._safe_trend_deadlines["S03:0-1"] == 4
    assert snapshots["warning_plus_2"][0].step_count == 2


def test_three_step_witness_does_not_require_one_step_feasibility():
    snapshots, _ = common_prefix(config(), scenario_at_distance(2000))
    snapshot = snapshots["first_S02_violation"]
    control = np.zeros((3, 3, 3))
    control[:, :2, 0] = 1
    control[:, 0, 1] = 1
    control[:, 1, 1] = -1
    metrics, states = evaluate_reference(snapshot, control)
    assert metrics["recovered_within_3_steps"] == 1
    assert metrics["first_recovery_step"] == 3
    assert metrics["physical_envelope_energy_valid"]
    assert "recovered_by_30" not in metrics
    assert "recovered_by_3_and_held_to_30" not in metrics
    assert metrics["reference_executed_steps"] == 3
    assert states[0]["minimum_separation_m"] < 800
    assert one_step_pair_distance_upper_bound(snapshot[0]) < 800


def test_outer_bound_contains_sampled_one_step_distances():
    snapshots, _ = common_prefix(config(), scenario_at_distance(2000))
    env = snapshots["first_S02_violation"][0]
    upper = one_step_pair_distance_upper_bound(env)
    for _ in range(30):
        controls = np.random.default_rng(_).uniform(-1,1,(1,3,3))
        assert predict_controls(env, controls)["distances"][0,0] <= upper + 1e-8


def test_waiting_prefix_violations_are_not_erased_from_timing_endpoint():
    snapshots, case = common_prefix(config(), scenario_at_distance(2000))
    warning = case["prefix"][0]
    states = deepcopy(case["prefix"][1:4])
    # Late controller clears every later state; prefix S02 breach still fails
    # safety retention on the common warning clock.
    safe = deepcopy(states[0])
    safe.update(target_violations=[], truth_violations=[], restored_safe_state=True, rule_clear=True)
    states += [deepcopy(safe) for _ in range(27)]
    metrics = timing_metrics(states, warning)
    assert metrics["target_safety_retention_3step"] == 0
    assert metrics["target_safety_retention_30step"] == 0
    assert metrics["first_S02_violation_step_from_warning"] == 3
