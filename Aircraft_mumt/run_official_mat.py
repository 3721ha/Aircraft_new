"""Smoke or benchmark PKU-MARL's pinned Multi-Agent Transformer."""

import argparse
import copy
import json
from pathlib import Path

from aircraft_sim import AircraftEnv, RuleMonitor, SimConfig, evaluate_policy
from aircraft_sim.scenarios import sample_scenario
from external_adapters import OfficialMATAdapter, OfficialMATConfig
from run_official_mappo import aggregate as aggregate_mappo
from run_paper_experiments import make_scenarios, make_validation_scenarios


def main():
    parser = argparse.ArgumentParser(description="Run the pinned official MAT core on Aircraft")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--ppo-epoch", type=int, default=5)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--transformer-blocks", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/official_mat_smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    episodes = []
    training = []
    fingerprints = set()
    for seed in args.seeds:
        sim_config = SimConfig(seed=seed, horizon=args.horizon)
        method_config = OfficialMATConfig(
            ppo_epoch=args.ppo_epoch,
            embedding_size=args.embedding_size,
            transformer_blocks=args.transformer_blocks,
            attention_heads=args.attention_heads,
        )
        policy = OfficialMATAdapter(AircraftEnv(args.aircraft, sim_config), method_config, seed=seed)
        fingerprints.add(policy.source_fingerprint)
        validation = make_validation_scenarios(args.aircraft, seed)
        checkpoints = set(range(0, args.updates + 1, args.checkpoint_interval)) | {args.updates}
        best_state = copy.deepcopy(policy.policy.transformer.state_dict())
        best_score = None
        selected_update = 0
        validation_trace = []
        for update in range(args.updates + 1):
            if update in checkpoints:
                validation_result = evaluate_policy(
                    AircraftEnv(args.aircraft, sim_config),
                    RuleMonitor(sim_config),
                    policy,
                    validation,
                    shield=None,
                    horizon=args.horizon,
                    name="official_mat_validation",
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
                    best_state = copy.deepcopy(policy.policy.transformer.state_dict())
            if update < args.updates:
                scenario = sample_scenario(
                    args.aircraft,
                    seed=seed * 100000 + 300000 + update,
                    difficulty=0.2 + 0.6 * ((update % 5) / 4.0),
                )
                training.append({"seed": seed, "update": update + 1, **policy.train_episode(scenario)})

        policy.policy.transformer.load_state_dict(best_state)
        model_dir = args.output / "checkpoints" / f"seed_{seed}"
        policy.save(model_dir)
        restored = OfficialMATAdapter(
            AircraftEnv(args.aircraft, sim_config), method_config, seed=seed + 1000000
        )
        restored.load(model_dir)
        scenarios = make_scenarios(args.aircraft, seed, args.episodes)
        result = evaluate_policy(
            AircraftEnv(args.aircraft, sim_config),
            RuleMonitor(sim_config),
            restored,
            scenarios,
            shield=None,
            horizon=args.horizon,
            name="official_mat",
        )
        row = result.as_dict()
        row.update({
            "seed": seed,
            "repository": "https://github.com/PKU-MARL/Multi-Agent-Transformer",
            "commit": policy.commit,
            "source_fingerprint": policy.source_fingerprint,
            "execution_information": policy.execution_information,
            "selected_update": selected_update,
            "validation_trace": validation_trace,
            "environment_steps": policy.environment_steps,
            "agent_decisions": policy.agent_decisions,
        })
        records.append(row)
        episodes.extend({
            "policy": "official_mat",
            "seed": seed,
            "scenario_type": scenarios[item["episode"]].get("scenario_type", "unknown"),
            **item,
        } for item in result.episode_records)

    summary = aggregate_mappo(records)
    summary["policy"] = "official_mat"
    manifest = {
        "method": "MAT",
        "implementation": "official TransformerPolicy + MATTrainer + SharedReplayBuffer",
        "repository": "https://github.com/PKU-MARL/Multi-Agent-Transformer",
        "commit": records[0]["commit"] if records else None,
        "source_fingerprints": sorted(fingerprints),
        "license": "no LICENSE file in pinned upstream repository; redistribution permission is not established",
        "third_party_source_modified": False,
        "adapter": "external_adapters/official_mat.py",
        "observation": "joint sequence of all agents' local observations",
        "execution_information": "centralized communication; stronger than decentralized local execution",
        "action_adapter": "common continuous latent Box(5), decoded to mode, controls and target",
        "safety_constraint": "none; MAT is a strong sequence-modeling MARL baseline",
        "adaptation_limit": "mode and target are rounded at the simulator boundary; communication advantage must be disclosed",
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

