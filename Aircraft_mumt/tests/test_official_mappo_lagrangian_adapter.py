import math
import tempfile
import unittest
from pathlib import Path

from aircraft_sim import AircraftEnv, SimConfig
from aircraft_sim.models import Mode
from aircraft_sim.scenarios import boundary_scenario
from external_adapters import OfficialMAPPOLagrangianAdapter, OfficialMAPPOLagrangianConfig
from external_adapters.official_mappo_lagrangian import PINNED_COMMIT


class OfficialMAPPOLagrangianAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = OfficialMAPPOLagrangianConfig().source
        if not (source / ".git").exists():
            raise unittest.SkipTest("pinned official MAPPO-Lagrangian repository is not available")

    def test_official_core_updates_multiplier_and_round_trips(self):
        config = SimConfig(seed=23, horizon=3, support_response_deadline=0)
        adapter = OfficialMAPPOLagrangianAdapter(
            AircraftEnv(3, config),
            OfficialMAPPOLagrangianConfig(ppo_epoch=1),
            seed=23,
        )
        self.assertEqual(adapter.commit, PINNED_COMMIT)
        initial = [adapter._scalar(item.lamda_lagr) for item in adapter.trainers]
        stats = adapter.train_episode(boundary_scenario())
        self.assertEqual(stats["updates"], 1)
        self.assertEqual(stats["environment_steps"], 3)
        self.assertEqual(stats["episode_hard_violation_cost"], 1.0)
        self.assertTrue(all(math.isfinite(item["cost_loss"]) for item in stats["agents"]))
        updated = [adapter._scalar(item.lamda_lagr) for item in adapter.trainers]
        self.assertTrue(all(after > before for before, after in zip(initial, updated)))

        observation = adapter.env.reset()[0]
        before_action = adapter.act(observation, 0, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            adapter.save(Path(directory))
            restored = OfficialMAPPOLagrangianAdapter(
                AircraftEnv(3, config),
                OfficialMAPPOLagrangianConfig(ppo_epoch=1),
                seed=99,
            )
            restored.load(Path(directory))
            after_action = restored.act(observation, 0, deterministic=True)
            restored_multipliers = [restored._scalar(item.lamda_lagr) for item in restored.trainers]
        self.assertEqual(before_action, after_action)
        self.assertEqual(updated, restored_multipliers)

    def test_common_continuous_action_semantics(self):
        adapter = OfficialMAPPOLagrangianAdapter(
            AircraftEnv(3, SimConfig(horizon=2)),
            OfficialMAPPOLagrangianConfig(ppo_epoch=1),
        )
        low = adapter.decode_action([-2, -2, -2, -2, -2])
        high = adapter.decode_action([2, 2, 2, 2, 2])
        self.assertEqual(low.mode, Mode.HOLD)
        self.assertEqual(high.mode, Mode.CRITICAL)
        self.assertEqual((low.turn, low.climb, low.acceleration, low.target), (-1.0, -1.0, -1.0, None))
        self.assertEqual((high.turn, high.climb, high.acceleration, high.target), (1.0, 1.0, 1.0, 2))


if __name__ == "__main__":
    unittest.main()

