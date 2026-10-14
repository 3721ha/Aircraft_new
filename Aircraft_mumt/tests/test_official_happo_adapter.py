import math
import tempfile
import unittest
from pathlib import Path

from aircraft_sim import AircraftEnv, SimConfig
from aircraft_sim.models import Mode
from aircraft_sim.scenarios import sample_scenario
from external_adapters import OfficialHAPPOAdapter, OfficialHAPPOConfig
from external_adapters.official_happo import PINNED_COMMIT


class OfficialHAPPOAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = OfficialHAPPOConfig().source
        if not (source / ".git").exists():
            raise unittest.SkipTest("pinned official HAPPO repository is not available")

    def test_official_core_trains_separate_policies_and_round_trips(self):
        config = SimConfig(seed=29, horizon=3)
        adapter = OfficialHAPPOAdapter(
            AircraftEnv(3, config), OfficialHAPPOConfig(ppo_epoch=1), seed=29
        )
        self.assertEqual(adapter.commit, PINNED_COMMIT)
        self.assertEqual(len({id(policy.actor) for policy in adapter.policies}), 3)
        stats = adapter.train_episode(sample_scenario(3, seed=2901, difficulty=0.4))
        self.assertEqual(stats["updates"], 1)
        self.assertEqual(stats["environment_steps"], 3)
        self.assertEqual(len(stats["agents"]), 3)
        self.assertTrue(all(math.isfinite(item["value_loss"]) for item in stats["agents"]))
        observation = adapter.env.reset()[0]
        before = adapter.act(observation, 0, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            adapter.save(Path(directory))
            restored = OfficialHAPPOAdapter(
                AircraftEnv(3, config), OfficialHAPPOConfig(ppo_epoch=1), seed=99
            )
            restored.load(Path(directory))
            after = restored.act(observation, 0, deterministic=True)
        self.assertEqual(before, after)

    def test_common_continuous_action_semantics(self):
        adapter = OfficialHAPPOAdapter(
            AircraftEnv(3, SimConfig(horizon=2)), OfficialHAPPOConfig(ppo_epoch=1)
        )
        low = adapter.decode_action([-2, -2, -2, -2, -2])
        high = adapter.decode_action([2, 2, 2, 2, 2])
        self.assertEqual(low.mode, Mode.HOLD)
        self.assertEqual(high.mode, Mode.CRITICAL)
        self.assertIsNone(low.target)
        self.assertEqual(high.target, 2)


if __name__ == "__main__":
    unittest.main()

