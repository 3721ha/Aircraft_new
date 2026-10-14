import math
import tempfile
import unittest
from pathlib import Path

from aircraft_sim import AircraftEnv, SimConfig
from aircraft_sim.models import Mode
from aircraft_sim.scenarios import sample_scenario
from external_adapters import OfficialMACPOAdapter, OfficialMACPOConfig
from external_adapters.official_macpo import PINNED_COMMIT


class OfficialMACPOAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = OfficialMACPOConfig().source
        if not (source / ".git").exists():
            raise unittest.SkipTest("pinned official MACPO repository is not available")

    def test_official_core_trains_and_checkpoint_round_trips(self):
        config = SimConfig(seed=19, horizon=3)
        adapter = OfficialMACPOAdapter(
            AircraftEnv(3, config),
            OfficialMACPOConfig(ppo_epoch=1, line_search_steps=2),
            seed=19,
        )
        self.assertEqual(adapter.commit, PINNED_COMMIT)
        stats = adapter.train_episode(sample_scenario(3, seed=1901, difficulty=0.4))
        self.assertEqual(stats["updates"], 1)
        self.assertEqual(stats["environment_steps"], 3)
        self.assertEqual(stats["agent_decisions"], 9)
        self.assertIn(stats["episode_hard_violation_cost"], (0.0, 1.0))
        self.assertEqual(stats["constraint_cost"], stats["episode_hard_violation_cost"])
        self.assertEqual(stats["cost_signal"], "episode_binary")
        self.assertEqual(len(stats["agents"]), 3)
        self.assertTrue(all(math.isfinite(agent["value_loss"]) for agent in stats["agents"]))
        observation = adapter.env.reset()[0]
        before = adapter.act(observation, 0, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            adapter.save(Path(directory))
            restored = OfficialMACPOAdapter(
                AircraftEnv(3, config),
                OfficialMACPOConfig(ppo_epoch=1, line_search_steps=2),
                seed=99,
            )
            restored.load(Path(directory))
            after = restored.act(observation, 0, deterministic=True)
        self.assertEqual(before, after)

    def test_continuous_adapter_reaches_semantic_bounds(self):
        adapter = OfficialMACPOAdapter(
            AircraftEnv(3, SimConfig(horizon=2)),
            OfficialMACPOConfig(ppo_epoch=1, line_search_steps=2),
        )
        low = adapter.decode_action([-2, -2, -2, -2, -2])
        high = adapter.decode_action([2, 2, 2, 2, 2])
        middle = adapter.decode_action([0, 0, 0, 0, -1])
        self.assertEqual(low.mode, Mode.HOLD)
        self.assertEqual((low.turn, low.climb, low.acceleration, low.target), (-1.0, -1.0, -1.0, None))
        self.assertEqual(high.mode, Mode.CRITICAL)
        self.assertEqual((high.turn, high.climb, high.acceleration, high.target), (1.0, 1.0, 1.0, 2))
        self.assertIsNone(middle.target)


if __name__ == "__main__":
    unittest.main()
