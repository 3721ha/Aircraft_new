import unittest

from aircraft_sim import AircraftEnv, ConflictAwareQPSafetyShield, RuleMonitor, SafetyShield, SimConfig, evaluate_policy, pareto_front
from aircraft_sim.policies import rule_agnostic_policy
from aircraft_sim.scenarios import scenario_suite


class EvaluationTests(unittest.TestCase):
    def test_frozen_evaluation_returns_metrics(self):
        env = AircraftEnv(3, SimConfig(horizon=2, seed=9))
        result = evaluate_policy(env, RuleMonitor(env.cfg), rule_agnostic_policy, scenario_suite(3), SafetyShield(RuleMonitor(env.cfg)), horizon=2, name="test")
        self.assertEqual(result.episodes, 2)
        self.assertGreaterEqual(result.joint_satisfaction_rate, 0.0)
        self.assertLessEqual(result.joint_satisfaction_rate, 1.0)
        self.assertGreaterEqual(result.initial_infeasible_rate, 0.0)
        self.assertGreaterEqual(result.mean_actor_time_ms, 0.0)
        self.assertTrue(pareto_front([result]))

    def test_belief_calibration_uses_executed_post_filter_risk(self):
        env = AircraftEnv(3, SimConfig(horizon=3, seed=12))
        result = evaluate_policy(env, RuleMonitor(env.cfg), rule_agnostic_policy, scenario_suite(3), ConflictAwareQPSafetyShield(env.cfg), horizon=3, name="belief")
        pairs = [pair for episode in result.episode_records for pair in episode.get("calibration_pairs", [])]
        self.assertTrue(pairs)
        self.assertTrue(all(pair["prediction_stage"] == "post_filter_executed_action" for pair in pairs))
        self.assertLessEqual(result.mean_pre_filter_belief_risk, 1.0)
        self.assertLessEqual(result.mean_post_filter_belief_risk, 1.0)
