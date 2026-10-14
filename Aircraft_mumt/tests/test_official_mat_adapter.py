import math
import tempfile
import unittest
from pathlib import Path

from aircraft_sim import AircraftEnv, RuleMonitor, SimConfig, evaluate_policy
from aircraft_sim.scenarios import sample_scenario
from external_adapters import OfficialMATAdapter, OfficialMATConfig
from external_adapters.official_mat import PINNED_COMMIT


class OfficialMATAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = OfficialMATConfig().source
        if not (source / ".git").exists():
            raise unittest.SkipTest("pinned official MAT repository is not available")

    def test_official_joint_policy_trains_evaluates_and_round_trips(self):
        config = SimConfig(seed=31, horizon=3)
        adapter = OfficialMATAdapter(
            AircraftEnv(3, config), OfficialMATConfig(ppo_epoch=1), seed=31
        )
        self.assertEqual(adapter.commit, PINNED_COMMIT)
        observations = adapter.env.reset()
        before = adapter.act_joint(observations)
        self.assertEqual(set(before), {0, 1, 2})
        stats = adapter.train_episode(sample_scenario(3, seed=3101, difficulty=0.4))
        self.assertEqual(stats["environment_steps"], 3)
        self.assertTrue(math.isfinite(stats["value_loss"]))
        result = evaluate_policy(
            AircraftEnv(3, config), RuleMonitor(config), adapter,
            [sample_scenario(3, seed=3102)], horizon=3, name="mat_test",
        )
        self.assertEqual(result.episodes, 1)

        observations = adapter.env.reset()
        expected = adapter.act_joint(observations)
        with tempfile.TemporaryDirectory() as directory:
            adapter.save(Path(directory))
            restored = OfficialMATAdapter(
                AircraftEnv(3, config), OfficialMATConfig(ppo_epoch=1), seed=99
            )
            restored.load(Path(directory))
            actual = restored.act_joint(observations)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()

