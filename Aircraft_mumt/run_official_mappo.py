"""Smoke or benchmark the pinned marlbenchmark/on-policy MAPPO implementation."""

import argparse
import copy
import json
import math
from pathlib import Path
from statistics import mean, stdev

from aircraft_sim import AircraftEnv, RuleMonitor, SimConfig, evaluate_policy
from aircraft_sim.scenarios import sample_scenario
from external_adapters import OfficialMAPPOAdapter, OfficialMAPPOConfig
from run_paper_experiments import SCALAR_FIELDS, make_scenarios, make_validation_scenarios


def aggregate(records):
    summary = {"policy": "official_mappo", "seeds": len(records)}
    for field in SCALAR_FIELDS:
        values = [float(row[field]) for row in records]
        sd = stdev(values) if len(values) > 1 else 0.0
        summary[f"{field}_mean"] = round(mean(values), 6)
        summary[f"{field}_std"] = round(sd, 6)
        summary[f"{field}_ci95"] = round(1.96 * sd / math.sqrt(len(values)), 6)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the pinned official MAPPO core on Aircraft")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--action-adapter", choices=["continuous", "multidiscrete"], default="continuous")
    parser.add_argument("--control-bins", type=int, default=11)
    parser.add_argument("--ppo-epoch", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/official_mappo_smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    episodes = []
    training = []
    fingerprints = set()
    for seed in args.seeds:
        config = SimConfig(seed=seed, horizon=args.horizon)
        policy = OfficialMAPPOAdapter(
            AircraftEnv(args.aircraft, config),
            OfficialMAPPOConfig(
                action_adapter=args.action_adapter,
                control_bins=args.control_bins,
                ppo_epoch=args.ppo_epoch,
            ),
            seed=seed,
        )
        fingerprints.add(policy.source_fingerprint)
        validation = make_validation_scenarios(args.aircraft, seed)
        checkpoints = set(range(0, args.updates + 1, args.checkpoint_interval)) | {args.updates}
        best_actor = copy.deepcopy(policy.policy.actor.state_dict())
        best_critic = copy.deepcopy(policy.policy.critic.state_dict())
        best_score = None
        selected_update = 0
        validation_trace = []
        for update in range(args.updates + 1):
            if update in checkpoints:
                validation_result = evaluate_policy(
                    AircraftEnv(args.aircraft, config),
                    RuleMonitor(config),
                    lambda observation, uid: policy.act(observation, uid, deterministic=True),
                    validation,
                    shield=None,
                    horizon=args.horizon,
                    name="official_mappo_validation",
                )
                score = (
                    validation_result.conditional_joint_satisfaction_rate,
                    -validation_result.post_truth_hard_violation_rate,
                    validation_result.mean_reward,
                )
                validation_trace.append({
                    "update": update,
                    "score": list(score),
                    "conditional_joint_satisfaction_rate": validation_result.conditional_joint_satisfaction_rate,
                    "post_truth_hard_violation_rate": validation_result.post_truth_hard_violation_rate,
                    "mean_reward": validation_result.mean_reward,
                })
                if best_score is None or score > best_score:
                    best_score = score
                    selected_update = update
                    best_actor = copy.deepcopy(policy.policy.actor.state_dict())
                    best_critic = copy.deepcopy(policy.policy.critic.state_dict())
            if update < args.updates:
                scenario = sample_scenario(
                    args.aircraft,
                    seed=seed * 100000 + 300000 + update,
                    difficulty=0.2 + 0.6 * ((update % 5) / 4.0),
                )
                training.append({"seed": seed, "update": update + 1, **policy.train_episode(scenario)})

        policy.policy.actor.load_state_dict(best_actor)
        policy.policy.critic.load_state_dict(best_critic)

        model_dir = args.output / "checkpoints" / f"seed_{seed}"
        policy.save(model_dir)
        # Reload into a fresh adapter before evaluation to test the complete
        # checkpoint handoff rather than evaluating only in-memory weights.
        restored = OfficialMAPPOAdapter(
            AircraftEnv(args.aircraft, config),
            OfficialMAPPOConfig(
                action_adapter=args.action_adapter,
                control_bins=args.control_bins,
                ppo_epoch=args.ppo_epoch,
            ),
            seed=seed + 1000000,
        )
        restored.load(model_dir)
        scenarios = make_scenarios(args.aircraft, seed, args.episodes)
        result = evaluate_policy(
            AircraftEnv(args.aircraft, config),
            RuleMonitor(config),
            lambda observation, uid: restored.act(observation, uid, deterministic=True),
            scenarios,
            shield=None,
            horizon=args.horizon,
            name="official_mappo",
        )
        row = result.as_dict()
        row.update({
            "seed": seed,
            "repository": "https://github.com/marlbenchmark/on-policy",
            "commit": policy.commit,
            "source_fingerprint": policy.source_fingerprint,
            "action_adapter": args.action_adapter,
            "control_bins": args.control_bins,
            "selected_update": selected_update,
            "validation_trace": validation_trace,
            "environment_steps": policy.environment_steps,
            "agent_decisions": policy.agent_decisions,
        })
        records.append(row)
        episodes.extend({
            "policy": "official_mappo",
            "seed": seed,
            "scenario_type": scenarios[item["episode"]].get("scenario_type", "unknown"),
            **item,
        } for item in result.episode_records)

    manifest = {
        "method": "MAPPO",
        "implementation": "marlbenchmark/on-policy official R_MAPPOPolicy + R_MAPPO + SharedReplayBuffer",
        "repository": "https://github.com/marlbenchmark/on-policy",
        "commit": records[0]["commit"] if records else None,
        "source_fingerprints": sorted(fingerprints),
        "license": "MIT",
        "third_party_source_modified": False,
        "adapter": "external_adapters/official_mappo.py",
        "observation": "Aircraft local ObservationEncoder for actor; concatenated team observations for centralized critic",
        "action_adapter": (
            "continuous Gaussian latent Box(5), clipped then decoded to mode, controls and target"
            if args.action_adapter == "continuous"
            else f"MultiDiscrete mode + three {args.control_bins}-bin controls + target"
        ),
        "adaptation_limit": "Mode and target are rounded at the simulator boundary; compare the legacy control grid as an adapter sensitivity check.",
        "seed_policy": {
            "training": "seed*100000 + 300000 + update (shared by all external baselines)",
            "validation": "550000 + seed*1000 + index",
            "test": "same frozen make_scenarios protocol as the paper benchmark",
        },
        "checkpoint_selection": "maximize conditional joint STL, then minimize truth hard violation, then maximize reward on validation scenarios",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output / "per_seed.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "per_episode.json").write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "training_trace.json").write_text(json.dumps(training, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({"metadata": manifest, "methods": [aggregate(records)]}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": len(records), "updates": len(training)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
