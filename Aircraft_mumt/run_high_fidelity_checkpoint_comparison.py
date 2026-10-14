"""Evaluate rebuilt decision-level checkpoints on JSBSim without fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

import torch

from aircraft_sim import (
    AircraftEnv,
    ConflictAwareQPSafetyShield,
    MAPPOConfig,
    MAPPOPolicy,
    PassThroughShield,
    RuleMonitor,
    SimConfig,
    evaluate_policy,
    make_validation_env,
)
from external_adapters import (
    OfficialMACPOAdapter,
    OfficialMACPOConfig,
    OfficialMAPPOAdapter,
    OfficialMAPPOConfig,
    OfficialMATAdapter,
    OfficialMATConfig,
)
from run_paper_experiments import make_scenarios


METHODS = ("proposed", "mappo", "macpo", "mat")


def checkpoint_path(root: Path, method: str, seed: int, proposed_root: Path, official_root: Path) -> Path:
    if method == "proposed":
        return proposed_root / "checkpoints" / "proposed_belief_stl_conflict_qp" / f"seed_{seed}.pt"
    return official_root / method / "checkpoints" / f"seed_{seed}"


def build_policy(method: str, seed: int, horizon: int, root: Path, proposed_root: Path, official_root: Path):
    config = SimConfig(seed=seed, horizon=horizon)
    decision_env = AircraftEnv(3, config)
    checkpoint = checkpoint_path(root, method, seed, proposed_root, official_root)
    if method == "proposed":
        policy = MAPPOPolicy(decision_env, PassThroughShield(), config=MAPPOConfig(), seed=seed)
        policy.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    elif method == "mappo":
        policy = OfficialMAPPOAdapter(
            decision_env,
            OfficialMAPPOConfig(action_adapter="continuous", ppo_epoch=5),
            seed=seed,
        )
        policy.load(checkpoint)
    elif method == "macpo":
        policy = OfficialMACPOAdapter(
            decision_env,
            OfficialMACPOConfig(ppo_epoch=5, line_search_steps=10, safety_bound=0.1, cost_signal="episode_binary"),
            seed=seed,
        )
        policy.load(checkpoint)
    elif method == "mat":
        policy = OfficialMATAdapter(decision_env, OfficialMATConfig(ppo_epoch=5), seed=seed)
        policy.load(checkpoint)
    else:
        raise ValueError(method)
    return policy, checkpoint


def aggregate(rows: list[dict], policy: str) -> dict:
    selected = [row for row in rows if row["policy"] == policy]

    def stat(key: str) -> dict:
        values = [float(row.get(key, 0.0)) for row in selected]
        sd = stdev(values) if len(values) > 1 else 0.0
        return {
            "mean": round(mean(values), 6) if values else 0.0,
            "std": round(sd, 6),
            "ci95": round(1.96 * sd / math.sqrt(len(values)), 6) if values else 0.0,
        }

    return {
        "policy": policy,
        "seeds": len(selected),
        "mean_reward": stat("mean_reward"),
        "joint_satisfaction_rate": stat("joint_satisfaction_rate"),
        "conditional_joint_satisfaction_rate": stat("conditional_joint_satisfaction_rate"),
        "post_truth_hard_violation_rate": stat("post_truth_hard_violation_rate"),
        "mean_intervention_rate": stat("mean_intervention_rate"),
        "mean_online_time_ms": stat("mean_online_time_ms"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rebuilt checkpoints on JSBSim")
    parser.add_argument("--backend", choices=["surrogate", "jsbsim"], default="jsbsim")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--decision-horizon", type=int, default=30)
    parser.add_argument("--proposed-root", type=Path, default=Path("results/proposed_rebuild_official_10seed"))
    parser.add_argument("--official-root", type=Path, default=Path("results/official_comparison_rebuild_10seed"))
    parser.add_argument("--output", type=Path, default=Path("results/high_fidelity_checkpoint_comparison"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, episodes = [], []
    root = Path.cwd()
    for seed in args.seeds:
        config = SimConfig(seed=seed, horizon=args.horizon)
        scenarios = make_scenarios(3, seed, args.episodes)
        for method in args.methods:
            policy, checkpoint = build_policy(
                method,
                seed,
                min(args.horizon, args.decision_horizon),
                root,
                args.proposed_root,
                args.official_root,
            )
            for shielded in (True, False):
                name = f"{method}_{'shared_qp' if shielded else 'unshielded'}"
                env = make_validation_env(args.backend, 3, config)
                shield = ConflictAwareQPSafetyShield(config) if shielded else None
                if method == "mat":
                    evaluation_policy = policy
                else:
                    evaluation_policy = lambda observation, uid, policy=policy: policy.act(
                        observation, uid, deterministic=True
                    )
                result = evaluate_policy(
                    env,
                    RuleMonitor(config),
                    evaluation_policy,
                    scenarios,
                    shield=shield,
                    horizon=args.horizon,
                    name=name,
                )
                row = result.as_dict()
                row.update({
                    "policy": name,
                    "base_method": method,
                    "seed": seed,
                    "backend": getattr(env, "backend_name", args.backend),
                    "shielded": shielded,
                    "checkpoint": str(checkpoint),
                })
                rows.append(row)
                episodes.extend({
                    "policy": name,
                    "base_method": method,
                    "seed": seed,
                    "scenario_type": scenarios[item["episode"]].get("scenario_type", "unknown"),
                    **item,
                } for item in result.episode_records)
    summaries = [aggregate(rows, f"{method}_{shield}") for method in args.methods for shield in ("shared_qp", "unshielded")]
    metadata = {
        "backend": args.backend,
        "methods": args.methods,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "decision_horizon": args.decision_horizon,
        "zero_shot": True,
        "checkpoint_protocol": "rebuilt decision-level checkpoints; no JSBSim fine-tuning",
    }
    (args.output / "summary.json").write_text(json.dumps({"metadata": metadata, "methods": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "per_seed.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "per_episode.json").write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(rows), "backend": args.backend}, ensure_ascii=False))


if __name__ == "__main__":
    main()
