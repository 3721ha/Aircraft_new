"""Run and merge the six full-rule official Aircraft baselines.

GCBF+ is deliberately excluded because it is evaluated only on the physical
S01/S02/S03 subtask; use run_official_gcbfplus.py for that experiment.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


METHODS = {
    "mappo": "run_official_mappo.py",
    "happo": "run_official_happo.py",
    "hatrpo": "run_official_hatrpo.py",
    "macpo": "run_official_macpo.py",
    "mappo_lagrangian": "run_official_mappo_lagrangian.py",
    "mat": "run_official_mat.py",
}


def build_command(method: str, args, output: Path) -> list[str]:
    command = [
        sys.executable,
        METHODS[method],
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--updates",
        str(args.updates),
        "--episodes",
        str(args.episodes),
        "--horizon",
        str(args.horizon),
        "--aircraft",
        str(args.aircraft),
        "--ppo-epoch",
        str(args.ppo_epoch),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--output",
        str(output),
    ]
    if method == "happo":
        command.extend(["--algorithm", "happo"])
    if method in {"hatrpo", "macpo"}:
        command.extend(["--line-search-steps", str(args.line_search_steps)])
    if method in {"macpo", "mappo_lagrangian"}:
        command.extend([
            "--safety-bound",
            str(args.safety_bound),
            "--cost-signal",
            "episode_binary",
        ])
    return command


def completed_run(directory: Path, args, method: str) -> bool:
    summary_path = directory / "summary.json"
    manifest_path = directory / "manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = manifest["arguments"]
    except (KeyError, json.JSONDecodeError, OSError):
        return False
    expected = {
        "seeds": args.seeds,
        "updates": args.updates,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "aircraft": args.aircraft,
        "ppo_epoch": args.ppo_epoch,
        "checkpoint_interval": args.checkpoint_interval,
    }
    if method in {"hatrpo", "macpo"}:
        expected["line_search_steps"] = args.line_search_steps
    if method in {"macpo", "mappo_lagrangian"}:
        expected["safety_bound"] = args.safety_bound
        expected["cost_signal"] = "episode_binary"
    if method == "mappo":
        expected["action_adapter"] = "continuous"
    return all(previous.get(key) == value for key, value in expected.items())


def _load_proposed(path: Path, expected: dict) -> tuple[dict, list[dict], list[dict], dict]:
    summary_payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        row for row in summary_payload["methods"]
        if row["policy"] == "proposed_belief_stl_conflict_qp"
    ]
    if len(rows) != 1:
        raise ValueError("proposed results must contain exactly one proposed_belief_stl_conflict_qp summary")
    previous = manifest.get("arguments", {})
    for key in ("seeds", "updates", "episodes", "horizon", "aircraft"):
        if previous.get(key) != expected[key]:
            raise ValueError(f"proposed results use a different {key}: {previous.get(key)!r} != {expected[key]!r}")
    if manifest.get("seed_families", {}).get("training") != "seed*100000 + 300000 + update":
        raise ValueError("proposed results predate the shared official training-seed protocol; rerun them")
    if manifest.get("checkpoint_selection") != (
        "maximize conditional joint STL, then minimize truth hard violation, then maximize reward"
    ):
        raise ValueError("proposed results use a different checkpoint-selection rule; rerun them")
    seed_records = [
        row for row in json.loads((path / "per_seed.json").read_text(encoding="utf-8"))
        if row["policy"] == "proposed_belief_stl_conflict_qp"
    ]
    episode_records = [
        row for row in json.loads((path / "per_episode.json").read_text(encoding="utf-8"))
        if row["policy"] == "proposed_belief_stl_conflict_qp"
    ]
    return rows[0], seed_records, episode_records, manifest


def merge_outputs(
    root: Path,
    methods: list[str],
    arguments: dict,
    proposed_results: Path | None = None,
) -> dict:
    summaries = []
    seed_records = []
    episode_records = []
    method_manifests = {}
    for method in methods:
        directory = root / method
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        summaries.extend(summary["methods"])
        seed_records.extend(json.loads((directory / "per_seed.json").read_text(encoding="utf-8")))
        episode_records.extend(json.loads((directory / "per_episode.json").read_text(encoding="utf-8")))
        method_manifests[method] = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )

    if proposed_results is not None:
        proposed, proposed_seeds, proposed_episodes, proposed_manifest = _load_proposed(
            proposed_results, arguments
        )
        summaries.insert(0, proposed)
        seed_records.extend(proposed_seeds)
        episode_records.extend(proposed_episodes)
        method_manifests["proposed_belief_stl_conflict_qp"] = proposed_manifest

    manifest = {
        "scope": "proposed method and six official methods evaluated on the complete Aircraft rule task",
        "methods": (["proposed_belief_stl_conflict_qp"] if proposed_results is not None else []) + methods,
        "excluded_specialist": {
            "method": "GCBF+",
            "reason": "physical S01/S02/S03 shield only; reported in a separate specialist table",
            "entry": "run_official_gcbfplus.py",
        },
        "comparison_rule": "only summaries generated by this invocation belong in the same main table",
        "proposed_results": str(proposed_results) if proposed_results is not None else None,
        "arguments": arguments,
        "method_manifests": method_manifests,
    }
    payload = {"metadata": manifest, "methods": summaries}
    (root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "per_seed.json").write_text(
        json.dumps(seed_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "per_episode.json").write_text(
        json.dumps(episode_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Run the six official full-rule baselines")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--ppo-epoch", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--line-search-steps", type=int, default=10)
    parser.add_argument("--safety-bound", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--proposed-results",
        type=Path,
        default=None,
        help="matching run_trainable_baselines output containing the proposed method",
    )
    parser.add_argument("--output", type=Path, default=Path("results/official_comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        directory = args.output / method
        if args.resume and completed_run(directory, args, method):
            print(f"[{method}] reusing completed matching run", flush=True)
            continue
        print(f"[{method}] starting", flush=True)
        subprocess.run(build_command(method, args, directory), check=True)
        print(f"[{method}] complete", flush=True)

    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = merge_outputs(args.output, args.methods, arguments, args.proposed_results)
    print(json.dumps({
        "output": str(args.output),
        "methods": [item["policy"] for item in payload["methods"]],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
