import unittest

from aircraft_sim import AircraftEnv, MAPPOConfig, MAPPOPolicy, RuleMonitor, SafetyShield, SimConfig
from aircraft_sim.policies import rule_agnostic_policy


class MAPPOTests(unittest.TestCase):
    def test_collect_and_update(self):
        env = AircraftEnv(3, SimConfig(horizon=3, seed=4))
        policy = MAPPOPolicy(env, SafetyShield(RuleMonitor(env.cfg)), seed=4)
        rollout = policy.collect(horizon=3)
        self.assertEqual(len(rollout), 3)
        self.assertTrue(all("old_log_prob" in item and "value" in item for item in rollout))
        stats = policy.update(rollout)
        self.assertIn("intervention_loss", stats)
        self.assertIn("intervention_fraction", stats)
        self.assertTrue(stats["loss"] == stats["loss"])
        policy.set_anchor()
        policy.config.anchor_kl_coef = 0.05
        anchored = policy.update(rollout)
        self.assertIn("anchor_kl", anchored)
        policy.enable_residual_mode(0.2)
        residual_action = policy.act(env.reset()[0], 0, deterministic=True)
        self.assertTrue(-1.0 <= residual_action.turn <= 1.0)
        policy.enable_discrete_safety_gate()
        gated_observation = env.reset()[0]
        gated_observation["self"]["resource"] = 0.1
        gated_action = policy.act(gated_observation, 0, deterministic=True)
        self.assertEqual(gated_action.mode.value, "recover")
        self.assertGreater(policy.discrete_gate_count, 0)

    def test_behavior_clone_accepts_local_demonstrations(self):
        env = AircraftEnv(3, SimConfig(horizon=2, seed=5))
        policy = MAPPOPolicy(env, SafetyShield(RuleMonitor(env.cfg)), seed=5)
        observations = env.reset()
        demos = [(observation, uid, rule_agnostic_policy(observation, uid)) for uid, observation in observations.items()]
        stats = policy.behavior_clone(demos, epochs=2)
        self.assertEqual(stats["samples"], 3)
        self.assertTrue(stats["behavior_clone_loss"] == stats["behavior_clone_loss"])

    def test_dwell_penalty_is_reported_and_applied(self):
        env = AircraftEnv(3, SimConfig(horizon=3, seed=6))
        policy = MAPPOPolicy(
            env,
            SafetyShield(RuleMonitor(env.cfg)),
            config=MAPPOConfig(dwell_penalty_coef=0.2),
            seed=6,
        )
        rollout = policy.collect(horizon=3)
        stats = policy.update(rollout)
        self.assertIn("dwell_penalty", stats)
        self.assertIn("role_change_rate", stats)
