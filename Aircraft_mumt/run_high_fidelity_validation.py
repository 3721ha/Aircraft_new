"""Run the legacy cross-dynamics validation on the surrogate or JSBSim backend."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from aircraft_sim import (
    AircraftEnv,
    ConflictAwareQPSafetyShield,
    MAPPOPolicy,
    PassThroughShield,
    RuleMonitor,
    SimConfig,
    evaluate_policy,
    make_validation_env,
)
from run_paper_experiments import make_scenarios, make_training_scenario


def train_decision_policy(method: str, seed: int, aircraft: int, horizon: int, updates: int):
    """Train the decision-level Actor before transferring it to validation dynamics."""

    config = SimConfig(seed=seed, horizon=horizon)
    train_env = AircraftEnv(aircraft, config)
    shield = ConflictAwareQPSafetyShield(config) if method == "proposed" else PassThroughShield()
    policy = MAPPOPolicy(train_env, shield, seed=seed)
    for update in range(max(0, updates)):
        rollout = policy.collect(
            scenario=make_training_scenario(aircraft, seed, update),
            horizon=horizon,
            deterministic=False,
        )
        policy.update(rollout)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Aircraft policies on a higher-fidelity backend")
    parser.add_argument("--backend", choices=["surrogate", "jsbsim"], default="surrogate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--include-mappo", action="store_true")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--decision-horizon", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results/high_fidelity_validation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records, episodes = [], []
    for seed in args.seeds:
        config = SimConfig(seed=seed, horizon=args.horizon)
        scenarios = make_scenarios(args.aircraft, seed, args.episodes)
        decision_horizon = min(args.horizon, args.decision_horizon)
        proposed_policy = train_decision_policy("proposed", seed, args.aircraft, decision_horizon, args.updates)
        comparisons = [
            ("proposed_shared_qp", proposed_policy, True),
            ("proposed_unshielded", proposed_policy, False),
        ]
        if args.include_mappo:
            mappo_policy = train_decision_policy("mappo", seed, args.aircraft, decision_horizon, args.updates)
            comparisons.extend([("mappo_shared_qp", mappo_policy, True), ("mappo_unshielded", mappo_policy, False)])
        for name, policy, shielded in comparisons:
            env = make_validation_env(args.backend, args.aircraft, config)
            shield = ConflictAwareQPSafetyShield(config) if shielded else None
            result = evaluate_policy(
                env,
                RuleMonitor(config),
                lambda observation, uid, policy=policy: policy.act(observation, uid, deterministic=True),
                scenarios,
                shield=shield,
                horizon=args.horizon,
                name=name,
            )
            row = result.as_dict()
            row.update({"policy": name, "seed": seed, "backend": getattr(env, "backend_name", args.backend), "episodes": args.episodes, "horizon": args.horizon})
            records.append(row)
            episodes.extend({"policy": name, "seed": seed, "scenario_type": scenarios[item["episode"]].get("scenario_type", "unknown"), **item} for item in result.episode_records)

    summaries = []
    for policy in dict.fromkeys(row["policy"] for row in records):
        selected = [row for row in records if row["policy"] == policy]
        def agg(key):
            vals = [float(row.get(key, 0.0)) for row in selected]
            sd = stdev(vals) if len(vals) > 1 else 0.0
            return {"mean": round(mean(vals), 6), "std": round(sd, 6), "ci95": round(1.96 * sd / math.sqrt(len(vals)), 6) if vals else 0.0}
        summaries.append({"policy": policy, "seeds": len(selected), "mean_reward": agg("mean_reward"), "joint_satisfaction_rate": agg("joint_satisfaction_rate"), "conditional_joint_satisfaction_rate": agg("conditional_joint_satisfaction_rate"), "post_truth_hard_violation_rate": agg("post_truth_hard_violation_rate"), "mean_intervention_rate": agg("mean_intervention_rate"), "mean_online_time_ms": agg("mean_online_time_ms")})
    manifest = {
        "backend": args.backend,
        "records": len(records),
        "seeds": args.seeds,
        "episodes": args.episodes,
        "horizon": args.horizon,
        "decision_horizon": args.decision_horizon,
        "zero_shot": True,
        "include_mappo": args.include_mappo,
        "training_before_transfer": True,
        "policy_scope": "decision-level MAPPO actor trained before validation; validation backend receives no policy updates",
    }
    (args.output / "summary.json").write_text(json.dumps({"metadata": manifest, "methods": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "per_seed.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "per_episode.json").write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records), "backend": args.backend}, ensure_ascii=False))


if __name__ == "__main__":
    main()
