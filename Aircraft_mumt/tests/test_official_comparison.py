import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from run_official_comparison import build_command, completed_run, merge_outputs


class OfficialComparisonTests(unittest.TestCase):
    def arguments(self):
        return Namespace(
            seeds=[11, 22],
            updates=3,
            episodes=4,
            horizon=5,
            aircraft=3,
            ppo_epoch=2,
            checkpoint_interval=1,
            line_search_steps=2,
            safety_bound=0.1,
        )

    def test_commands_share_protocol_and_constrained_cost(self):
        args = self.arguments()
        macpo = build_command("macpo", args, Path("out/macpo"))
        mat = build_command("mat", args, Path("out/mat"))
        for command in (macpo, mat):
            self.assertIn("--seeds", command)
            self.assertIn("--updates", command)
            self.assertIn("--checkpoint-interval", command)
        self.assertIn("episode_binary", macpo)
        self.assertNotIn("--cost-signal", mat)

    def test_resume_requires_matching_arguments_and_merge_keeps_all_rows(self):
        args = self.arguments()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method in ("mappo", "mat"):
                directory = root / method
                directory.mkdir()
                manifest = {"method": method, "arguments": {
                    "seeds": args.seeds,
                    "updates": args.updates,
                    "episodes": args.episodes,
                    "horizon": args.horizon,
                    "aircraft": args.aircraft,
                    "ppo_epoch": args.ppo_epoch,
                    "checkpoint_interval": args.checkpoint_interval,
                    "action_adapter": "continuous",
                }}
                (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                (directory / "summary.json").write_text(
                    json.dumps({"methods": [{"policy": method}]}), encoding="utf-8"
                )
                (directory / "per_seed.json").write_text(
                    json.dumps([{"policy": method}]), encoding="utf-8"
                )
                (directory / "per_episode.json").write_text(
                    json.dumps([{"policy": method}]), encoding="utf-8"
                )
            self.assertTrue(completed_run(root / "mappo", args, "mappo"))
            payload = merge_outputs(root, ["mappo", "mat"], vars(args))
            self.assertEqual([row["policy"] for row in payload["methods"]], ["mappo", "mat"])
            self.assertEqual(len(json.loads((root / "per_seed.json").read_text())), 2)

    def test_merge_accepts_only_matching_new_protocol_proposed_results(self):
        args = self.arguments()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            method_dir = root / "mappo"
            method_dir.mkdir()
            (method_dir / "manifest.json").write_text(json.dumps({"method": "MAPPO"}), encoding="utf-8")
            (method_dir / "summary.json").write_text(json.dumps({"methods": [{"policy": "official_mappo"}]}), encoding="utf-8")
            (method_dir / "per_seed.json").write_text("[]", encoding="utf-8")
            (method_dir / "per_episode.json").write_text("[]", encoding="utf-8")
            proposed = root / "proposed"
            proposed.mkdir()
            proposed_manifest = {
                "arguments": {
                    "seeds": args.seeds,
                    "updates": args.updates,
                    "episodes": args.episodes,
                    "horizon": args.horizon,
                    "aircraft": args.aircraft,
                },
                "seed_families": {"training": "seed*100000 + 300000 + update"},
                "checkpoint_selection": "maximize conditional joint STL, then minimize truth hard violation, then maximize reward",
            }
            (proposed / "manifest.json").write_text(json.dumps(proposed_manifest), encoding="utf-8")
            (proposed / "summary.json").write_text(json.dumps({"methods": [{"policy": "proposed_belief_stl_conflict_qp"}]}), encoding="utf-8")
            (proposed / "per_seed.json").write_text(json.dumps([{"policy": "proposed_belief_stl_conflict_qp"}]), encoding="utf-8")
            (proposed / "per_episode.json").write_text(json.dumps([{"policy": "proposed_belief_stl_conflict_qp"}]), encoding="utf-8")
            payload = merge_outputs(root, ["mappo"], vars(args), proposed)
            self.assertEqual(payload["methods"][0]["policy"], "proposed_belief_stl_conflict_qp")
            proposed_manifest["seed_families"]["training"] = "old protocol"
            (proposed / "manifest.json").write_text(json.dumps(proposed_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "predate"):
                merge_outputs(root, ["mappo"], vars(args), proposed)


if __name__ == "__main__":
    unittest.main()
