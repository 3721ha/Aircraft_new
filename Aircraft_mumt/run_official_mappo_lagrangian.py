"""Smoke or benchmark the pinned official MAPPO-Lagrangian implementation."""

import argparse
import copy
import json
import math
from pathlib import Path
from statistics import mean, stdev

from aircraft_sim import AircraftEnv, RuleMonitor, SimConfig, evaluate_policy
from aircraft_sim.scenarios import sample_scenario
from external_adapters import OfficialMAPPOLagrangianAdapter, OfficialMAPPOLagrangianConfig
from run_paper_experiments import SCALAR_FIELDS, make_scenarios, make_validation_scenarios


def aggregate(records):
    summary = {"policy": "official_mappo_lagrangian", "seeds": len(records)}
    for field in SCALAR_FIELDS:
        values = [float(row[field]) for row in records]
        sd = stdev(values) if len(values) > 1 else 0.0
        summary[f"{field}_mean"] = round(mean(values), 6)
        summary[f"{field}_std"] = round(sd, 6)
        summary[f"{field}_ci95"] = round(1.96 * sd / math.sqrt(len(values)), 6)
    return summary


def policy_state(policy):
    return [
        {
            "actor": copy.deepcopy(item.actor.state_dict()),
            "critic": copy.deepcopy(item.critic.state_dict()),
            "cost_critic": copy.deepcopy(item.cost_critic.state_dict()),
            "lagrangian_multiplier": policy._scalar(trainer.lamda_lagr),
        }
        for item, trainer in zip(policy.policies, policy.trainers)
    ]


def restore_policy_state(policy, state):
    for item, trainer, saved in zip(policy.policies, policy.trainers, state):
        item.actor.load_state_dict(saved["actor"])
        item.critic.load_state_dict(saved["critic"])
        item.cost_critic.load_state_dict(saved["cost_critic"])
        trainer.lamda_lagr = saved["lagrangian_multiplier"]


def main():
    parser = argparse.ArgumentParser(description="Run official MAPPO-Lagrangian on Aircraft")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--ppo-epoch", type=int, default=5)
    parser.add_argument("--safety-bound", type=float, default=0.10)
    parser.add_argument("--initial-multiplier", type=float, default=0.78)
    parser.add_argument("--multiplier-lr", type=float, default=5e-4)
    parser.add_argument("--cost-signal", choices=["episode_binary", "step_indicator"], default="episode_binary")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/official_mappo_lagrangian_smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    episodes = []
    training = []
    fingerprints = set()
    for seed in args.seeds:
        sim_config = SimConfig(seed=seed, horizon=args.horizon)
        method_config = OfficialMAPPOLagrangianConfig(
            ppo_epoch=args.ppo_epoch,
            safety_bound=args.safety_bound,
            initial_lagrangian_multiplier=args.initial_multiplier,
            lagrangian_learning_rate=args.multiplier_lr,
            cost_signal=args.cost_signal,
        )
        policy = OfficialMAPPOLagrangianAdapter(
            AircraftEnv(args.aircraft, sim_config), method_config, seed=seed
        )
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
                    name="official_mappo_lagrangian_validation",
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
                    "lagrangian_multipliers": [
                        policy._scalar(trainer.lamda_lagr) for trainer in policy.trainers
                    ],
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
        restored = OfficialMAPPOLagrangianAdapter(
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
            name="official_mappo_lagrangian",
        )
        row = result.as_dict()
        row.update({
            "seed": seed,
            "repository": "https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation",
            "component": "MAPPO-Lagrangian",
            "commit": policy.commit,
            "source_fingerprint": policy.source_fingerprint,
            "safety_bound": args.safety_bound,
            "cost_signal": args.cost_signal,
            "selected_update": selected_update,
            "selected_lagrangian_multipliers": [
                restored._scalar(trainer.lamda_lagr) for trainer in restored.trainers
            ],
            "validation_trace": validation_trace,
            "environment_steps": policy.environment_steps,
            "agent_decisions": policy.agent_decisions,
        })
        records.append(row)
        episodes.extend({
            "policy": "official_mappo_lagrangian",
            "seed": seed,
            "scenario_type": scenarios[item["episode"]].get("scenario_type", "unknown"),
            **item,
        } for item in result.episode_records)

    manifest = {
        "method": "MAPPO-Lagrangian",
        "implementation": "official separated MACPPOPolicy + R_MAPPO_Lagr + SeparatedReplayBuffer",
        "repository": "https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation",
        "component": "MAPPO-Lagrangian",
        "commit": records[0]["commit"] if records else None,
        "source_fingerprints": sorted(fingerprints),
        "license": "MIT text present; upstream LICENSE contains unresolved merge markers",
        "third_party_source_modified": False,
        "adapter": "external_adapters/official_mappo_lagrangian.py",
        "observation": "per-agent local observation; concatenated team observations for reward and cost critics",
        "action_adapter": "common continuous Gaussian latent Box(5), clipped then decoded to mode, controls and target",
        "training_cost": (
            "terminal binary indicator of any trajectory-level hard-rule violation"
            if args.cost_signal == "episode_binary"
            else "shared per-step hard-rule violation indicator; constrained as episodic sum"
        ),
        "truth_boundary": "ground truth forms the training cost only and is absent from actor observations",
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
        json.dumps({"metadata": manifest, "methods": [aggregate(records)]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": len(records), "updates": len(training)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
