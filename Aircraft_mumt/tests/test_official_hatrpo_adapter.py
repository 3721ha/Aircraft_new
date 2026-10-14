import math
import tempfile
import unittest
from pathlib import Path

from aircraft_sim import AircraftEnv, SimConfig
from aircraft_sim.scenarios import sample_scenario
from external_adapters import OfficialHATRPOAdapter, OfficialHATRPOConfig
from external_adapters.official_happo import PINNED_COMMIT


class OfficialHATRPOAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = OfficialHATRPOConfig().source
        if not (source / ".git").exists():
            raise unittest.SkipTest("pinned official HATRPO repository is not available")

    def test_official_trust_region_update_and_checkpoint(self):
        config = SimConfig(seed=37, horizon=3)
        adapter = OfficialHATRPOAdapter(
            AircraftEnv(3, config),
            OfficialHATRPOConfig(line_search_steps=2),
            seed=37,
        )
        self.assertEqual(adapter.commit, PINNED_COMMIT)
        stats = adapter.train_episode(sample_scenario(3, seed=3701, difficulty=0.4))
        self.assertEqual(stats["environment_steps"], 3)
        self.assertEqual(len(stats["agents"]), 3)
        self.assertTrue(all(math.isfinite(item["kl"]) for item in stats["agents"]))
        observation = adapter.env.reset()[0]
        before = adapter.act(observation, 0, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            adapter.save(Path(directory))
            restored = OfficialHATRPOAdapter(
                AircraftEnv(3, config),
                OfficialHATRPOConfig(line_search_steps=2),
                seed=99,
            )
            restored.load(Path(directory))
            after = restored.act(observation, 0, deterministic=True)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

