import tempfile
import unittest
from pathlib import Path

from aircraft_sim import AircraftEnv, SimConfig
from aircraft_sim.scenarios import sample_scenario
from external_adapters import OfficialMAPPOAdapter, OfficialMAPPOConfig
from external_adapters.official_mappo import PINNED_COMMIT


class OfficialMAPPOAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = OfficialMAPPOConfig().source
        if not (source / ".git").exists():
            raise unittest.SkipTest("pinned official MAPPO repository is not available")

    def test_official_core_trains_and_checkpoint_round_trips(self):
        config = SimConfig(seed=17, horizon=3)
        adapter = OfficialMAPPOAdapter(
            AircraftEnv(3, config),
            OfficialMAPPOConfig(ppo_epoch=1, control_bins=5),
            seed=17,
        )
        self.assertEqual(adapter.commit, PINNED_COMMIT)
        stats = adapter.train_episode(sample_scenario(3, seed=1701, difficulty=0.4))
        self.assertEqual(stats["updates"], 1)
        self.assertEqual(stats["environment_steps"], 3)
        self.assertEqual(stats["agent_decisions"], 9)
        self.assertTrue(stats["value_loss"] == stats["value_loss"])
        observation = adapter.env.reset()[0]
        before = adapter.act(observation, 0, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            adapter.save(Path(directory))
            restored = OfficialMAPPOAdapter(
                AircraftEnv(3, config),
                OfficialMAPPOConfig(ppo_epoch=1, control_bins=5),
                seed=99,
            )
            restored.load(Path(directory))
            after = restored.act(observation, 0, deterministic=True)
        self.assertEqual(before, after)

    def test_action_grid_reaches_bounds_and_no_target(self):
        adapter = OfficialMAPPOAdapter(
            AircraftEnv(3, SimConfig(horizon=2)),
            OfficialMAPPOConfig(action_adapter="multidiscrete", ppo_epoch=1, control_bins=5),
        )
        low = adapter.decode_action([0, 0, 0, 0, 0])
        high = adapter.decode_action([5, 4, 4, 4, 3])
        self.assertEqual((low.turn, low.climb, low.acceleration, low.target), (-1.0, -1.0, -1.0, None))
        self.assertEqual((high.turn, high.climb, high.acceleration, high.target), (1.0, 1.0, 1.0, 2))

    def test_default_continuous_adapter_reaches_bounds(self):
        adapter = OfficialMAPPOAdapter(
            AircraftEnv(3, SimConfig(horizon=2)),
            OfficialMAPPOConfig(ppo_epoch=1),
        )
        low = adapter.decode_action([-2, -2, -2, -2, -2])
        high = adapter.decode_action([2, 2, 2, 2, 2])
        self.assertEqual((low.turn, low.climb, low.acceleration, low.target), (-1.0, -1.0, -1.0, None))
        self.assertEqual((high.turn, high.climb, high.acceleration, high.target), (1.0, 1.0, 1.0, 2))


if __name__ == "__main__":
    unittest.main()
