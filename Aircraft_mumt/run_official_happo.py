"""Smoke or benchmark the pinned marlbenchmark/on-policy HAPPO implementation."""

import argparse
import copy
import json
from pathlib import Path

from aircraft_sim import AircraftEnv, RuleMonitor, SimConfig, evaluate_policy
from aircraft_sim.scenarios import sample_scenario
from external_adapters import (
    OfficialHAPPOAdapter,
    OfficialHAPPOConfig,
    OfficialHATRPOAdapter,
    OfficialHATRPOConfig,
)
from run_official_mappo import aggregate as aggregate_mappo
from run_paper_experiments import make_scenarios, make_validation_scenarios


def policy_state(policy):
    return [
        {
            "actor": copy.deepcopy(item.actor.state_dict()),
            "critic": copy.deepcopy(item.critic.state_dict()),
        }
        for item in policy.policies
    ]


def restore_policy_state(policy, state):
    for item, saved in zip(policy.policies, state):
        item.actor.load_state_dict(saved["actor"])
        item.critic.load_state_dict(saved["critic"])


def main(default_algorithm="happo"):
    parser = argparse.ArgumentParser(description="Run the pinned official HAPPO/HATRPO core on Aircraft")
    parser.add_argument("--algorithm", choices=["happo", "hatrpo"], default=default_algorithm)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--ppo-epoch", type=int, default=5)
    parser.add_argument("--kl-threshold", type=float, default=0.01)
    parser.add_argument("--line-search-steps", type=int, default=10)
    parser.add_argument("--accept-ratio", type=float, default=0.5)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"results/official_{args.algorithm}_smoke")
    args.output.mkdir(parents=True, exist_ok=True)
    method_name = args.algorithm.upper()
    policy_type = OfficialHAPPOAdapter if args.algorithm == "happo" else OfficialHATRPOAdapter

    records = []
    episodes = []
    training = []
    fingerprints = set()
    for seed in args.seeds:
        sim_config = SimConfig(seed=seed, horizon=args.horizon)
        method_config = (
            OfficialHAPPOConfig(ppo_epoch=args.ppo_epoch)
            if args.algorithm == "happo"
            else OfficialHATRPOConfig(
                ppo_epoch=args.ppo_epoch,
                kl_threshold=args.kl_threshold,
                line_search_steps=args.line_search_steps,
                accept_ratio=args.accept_ratio,
            )
        )
        policy = policy_type(AircraftEnv(args.aircraft, sim_config), method_config, seed=seed)
        fingerprints.add(policy.source_fingerprint)
        validation = make_validation_scenarios(args.aircraft, seed)
        checkpoints = set(range(0, args.updates + 1, args.checkpoint_interval)) | {args.updates}
        best_state = policy_state(policy)
        best_score = None
        selected_update = 0
        validation_trace = []
        for update in range(args.updates + 1):
            if update in checkpoints:
                validation_result = evaluate_policy(
                    AircraftEnv(args.aircraft, sim_config),
                    RuleMonitor(sim_config),
                    lambda observation, uid: policy.act(observation, uid, deterministic=True),
                    validation,
                    shield=None,
                    horizon=args.horizon,
                    name=f"official_{args.algorithm}_validation",
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
                    best_state = policy_state(policy)
            if update < args.updates:
                scenario = sample_scenario(
                    args.aircraft,
                    seed=seed * 100000 + 300000 + update,
                    difficulty=0.2 + 0.6 * ((update % 5) / 4.0),
                )
                training.append({"seed": seed, "update": update + 1, **policy.train_episode(scenario)})

        restore_policy_state(policy, best_state)
        model_dir = args.output / "checkpoints" / f"seed_{seed}"
        policy.save(model_dir)
        restored = policy_type(
            AircraftEnv(args.aircraft, sim_config), method_config, seed=seed + 1000000
        )
        restored.load(model_dir)
        scenarios = make_scenarios(args.aircraft, seed, args.episodes)
        result = evaluate_policy(
            AircraftEnv(args.aircraft, sim_config),
            RuleMonitor(sim_config),
            lambda observation, uid: restored.act(observation, uid, deterministic=True),
            scenarios,
            shield=None,
            horizon=args.horizon,
            name=f"official_{args.algorithm}",
        )
        row = result.as_dict()
        row.update({
            "seed": seed,
            "repository": "https://github.com/marlbenchmark/on-policy",
            "component": method_name,
            "commit": policy.commit,
            "source_fingerprint": policy.source_fingerprint,
            "selected_update": selected_update,
            "validation_trace": validation_trace,
            "environment_steps": policy.environment_steps,
            "agent_decisions": policy.agent_decisions,
        })
        records.append(row)
        episodes.extend({
            "policy": f"official_{args.algorithm}",
            "seed": seed,
            "scenario_type": scenarios[item["episode"]].get("scenario_type", "unknown"),
            **item,
        } for item in result.episode_records)

    summary = aggregate_mappo(records)
    summary["policy"] = f"official_{args.algorithm}"
    manifest = {
        "method": method_name,
        "implementation": f"official {method_name}_Policy + {method_name} + SeparatedReplayBuffer",
        "repository": "https://github.com/marlbenchmark/on-policy",
        "commit": records[0]["commit"] if records else None,
        "source_fingerprints": sorted(fingerprints),
        "license": "MIT",
        "third_party_source_modified": False,
        "adapter": f"external_adapters/official_{args.algorithm}.py",
        "observation": "separate local actor observations; concatenated team observations for centralized critics",
        "action_adapter": "common continuous Gaussian latent Box(5), clipped then decoded to mode, controls and target",
        "safety_constraint": f"none; {method_name} is an unconstrained heterogeneous MARL baseline",
        "compatibility_note": (
            "none"
            if args.algorithm == "happo"
            else "adapter ignores an upstream always-None availability argument passed to the continuous Gaussian layer"
        ),
        "adaptation_limit": "mode and target are rounded at the simulator boundary",
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
    (args.output / "summary.json").write_text(
        json.dumps({"metadata": manifest, "methods": [summary]}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": len(records), "updates": len(training)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
