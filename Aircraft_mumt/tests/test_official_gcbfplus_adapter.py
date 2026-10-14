import math
import unittest

from external_adapters.official_gcbfplus import (
    DEFAULT_SOURCE,
    PINNED_COMMIT,
    OfficialGCBFPlusConfig,
    OfficialGCBFPlusShield,
    _commit,
    dependencies_available,
)
from run_official_gcbfplus import aggregate, physical_metrics


class OfficialGCBFPlusAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (DEFAULT_SOURCE / ".git").exists():
            raise unittest.SkipTest("pinned official GCBF+ repository is not available")

    def test_commit_and_dependency_probe(self):
        self.assertEqual(_commit(DEFAULT_SOURCE), PINNED_COMMIT)
        self.assertIsInstance(dependencies_available(), bool)

    def test_adapter_matches_pretrained_environment_metadata(self):
        config = OfficialGCBFPlusConfig()
        self.assertEqual(config.pretrained_agents, 8)
        self.assertEqual(config.checkpoint_obstacles, 8)
        self.assertEqual(config.deployed_obstacles, 0)
        self.assertEqual(config.lidar_rays, 32)

    def test_cartesian_action_maps_to_aircraft_axes(self):
        turn, climb, acceleration = OfficialGCBFPlusShield.map_physical_control([1.0, 0.0, 0.5], 0.0)
        self.assertEqual((turn, climb, acceleration), (0.0, 0.5, 1.0))
        turn, _, acceleration = OfficialGCBFPlusShield.map_physical_control([1.0, 0.0, 0.0], math.pi / 2)
        self.assertAlmostEqual(turn, -1.0)
        self.assertAlmostEqual(acceleration, 0.0, places=7)

    def test_physical_metrics_exclude_nonphysical_rules(self):
        metrics = physical_metrics([
            {"violations": {"I01": 0}},
            {"violations": {"S02": 1, "C02": 0}},
        ])
        self.assertEqual(metrics["physical_joint_satisfaction_rate"], 0.5)
        self.assertEqual(metrics["physical_violation_rate"]["S02"], 0.5)
        self.assertEqual(metrics["physical_violation_rate"]["S01"], 0.0)

    def test_specialist_aggregate_keeps_comparison_rows_separate(self):
        records = []
        for policy, satisfaction in (("nominal", 0.25), ("gcbfplus", 0.75)):
            records.append({
                "policy": policy,
                "physical_joint_satisfaction_rate": satisfaction,
                "mean_reward": 1.0,
                "mean_intervention_rate": 0.0,
                "mean_intervention_distance": 0.0,
            })
        summaries = aggregate(records)
        self.assertEqual([row["policy"] for row in summaries], ["nominal", "gcbfplus"])
        self.assertEqual(summaries[1]["physical_joint_satisfaction_rate_mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
