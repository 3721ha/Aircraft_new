import math
import unittest

from aircraft_sim import Action, HigherFidelityAircraftEnv, Mode, SimConfig, jsbsim_available, make_validation_env


class HigherFidelityTests(unittest.TestCase):
    def test_proxy_preserves_api_and_applies_lag(self):
        env = HigherFidelityAircraftEnv(config=SimConfig(seed=3, horizon=3))
        env.reset()
        result = env.step({0: Action(Mode.HOLD, turn=1.0), 1: Action(), 2: Action()})
        self.assertEqual(result.info["dynamics"]["backend"], "lagged_3dof_validation_proxy")
        self.assertLess(abs(result.info["applied_actions"][0].turn), 1.0)
        self.assertIn("truth", result.info)

    def test_proxy_dynamics_randomization_is_bounded(self):
        env = HigherFidelityAircraftEnv(
            config=SimConfig(seed=3, horizon=2),
            randomize_dynamics=True,
        )
        env.reset()
        metadata = env.dynamics_metadata
        self.assertTrue(0.30 <= metadata["episode_actuator_tau"] <= 0.70)
        self.assertTrue(0.80 <= metadata["episode_rate_scale"] <= 1.20)

    def test_unknown_jsbsim_is_explicit(self):
        if not jsbsim_available():
            with self.assertRaises(RuntimeError):
                make_validation_env("jsbsim")

    @unittest.skipUnless(jsbsim_available(), "optional JSBSim dependency is not installed")
    def test_jsbsim_backend_runs_six_dof_step(self):
        env = make_validation_env("jsbsim", 1, SimConfig(seed=8, horizon=2))
        before = env.reset({"positions": [(0.0, 0.0, 5000.0)], "resources": [0.9], "energies": [0.8]})
        self.assertLess(max(abs(value - target) for value, target in zip(before[0]["self"]["position"], (0.0, 0.0, 5000.0))), 150.0)
        result = env.step({0: Action(Mode.HOLD, turn=0.2, climb=0.1)})
        self.assertEqual(result.info["dynamics"]["backend"], "jsbsim_f16_6dof")
        self.assertNotEqual(before[0]["self"]["position"], result.observations[0]["self"]["position"])
        self.assertTrue(math.isfinite(result.info["truth"]["aircraft"][0].roll))
