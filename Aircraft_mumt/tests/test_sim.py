import unittest

from aircraft_sim import Action, AircraftEnv, JointSafetyShield, Mode, RuleMonitor, SafetyShield, SimConfig
from aircraft_sim.scenarios import boundary_scenario


class SimulatorTests(unittest.TestCase):
    def test_shield_blocks_low_confidence_critical_action(self):
        env = AircraftEnv(3)
        obs = env.reset(boundary_scenario())
        shield = SafetyShield(RuleMonitor(env.cfg))
        safe, events = shield.filter(obs, {0: Action(Mode.CRITICAL), 1: Action(Mode.HOLD), 2: Action(Mode.HOLD)})
        self.assertNotEqual(safe[0].mode, Mode.CRITICAL)
        self.assertTrue(events)

    def test_truth_monitor_detects_close_formation(self):
        env = AircraftEnv(3)
        env.reset(boundary_scenario())
        report = RuleMonitor(env.cfg).evaluate_truth(env.truth_state())
        self.assertTrue(any(v.startswith("S02") for v in report.violations))

    def test_joint_shield_changes_close_pair_actions(self):
        env = AircraftEnv(3, SimConfig(noise_position=0.0, sensor_drop=0.0))
        observations = env.reset(boundary_scenario())
        safe, events = JointSafetyShield(RuleMonitor(env.cfg)).filter(observations, {i: Action(Mode.HOLD) for i in observations})
        self.assertNotEqual(safe[0].mode, Mode.HOLD)
        self.assertTrue(any("S02_joint_separation" in event.reasons for event in events))
