"""Train the legacy shared-backbone Aircraft baseline family.

The runner keeps the 9/15 protocol: identical local observations, actor/critic
capacity and update budget; only the objective and safety layer differ.
"""

from __future__ import annotations

import argparse
import copy
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
    PhysicalCBFQPShield,
    PassThroughShield,
    RuleMonitor,
    SimConfig,
    evaluate_policy,
)
from aircraft_sim.policies import rule_agnostic_policy
from run_paper_experiments import SCALAR_FIELDS, make_scenarios, make_training_scenario, make_validation_scenarios


METHODS = {
    "mappo_unconstrained": {"objective": "task", "shield": "none"},
    "mappo_fixed_penalty": {"objective": "fixed_penalty", "shield": "none"},
    "stl_reward_mappo": {"objective": "stl_reward", "shield": "none"},
    "mappo_lagrangian": {"objective": "lagrangian", "shield": "none"},
    "macpo_internal_reference": {"objective": "macpo", "shield": "none"},
    "mappo_physical_cbf_qp": {"objective": "task", "shield": "cbf_qp"},
    "proposed_belief_stl_conflict_qp": {"objective": "task", "shield": "proposed"},
}


def _aggregate(records: list[dict], policy: str) -> dict:
    selected = [row for row in records if row["policy"] == policy]
    result = {"policy": policy, "seeds": len(selected)}
    for field in SCALAR_FIELDS:
        values = [float(row.get(field, 0.0)) for row in selected]
        sd = stdev(values) if len(values) > 1 else 0.0
        result[f"{field}_mean"] = round(mean(values), 6) if values else 0.0
        result[f"{field}_std"] = round(sd, 6)
        result[f"{field}_ci95"] = round(1.96 * sd / math.sqrt(len(values)), 6) if values else 0.0
    return result


def _make_policy(method: str, env: AircraftEnv, seed: int, args) -> MAPPOPolicy:
    spec = METHODS[method]
    config = MAPPOConfig(
        epochs=args.ppo_epoch,
        safety_objective=spec["objective"],
        fixed_penalty_coef=args.fixed_penalty_coef,
        stl_reward_coef=args.stl_reward_coef,
        cost_limit=args.safety_bound,
    )
    if spec["shield"] == "proposed":
        shield = ConflictAwareQPSafetyShield(env.cfg)
    elif spec["shield"] == "cbf_qp":
        shield = PhysicalCBFQPShield(env.cfg)
    else:
        shield = PassThroughShield()
    return MAPPOPolicy(env, shield, config=config, seed=seed)


def _warm_start(policy: MAPPOPolicy, env: AircraftEnv, seed: int, epochs: int) -> dict:
    if epochs <= 0:
        return {"samples": 0, "behavior_clone_loss": 0.0}
    observations = env.reset(make_training_scenario(env.n, seed, 0))
    demonstrations = [
        (observation, uid, rule_agnostic_policy(observation, uid))
        for uid, observation in observations.items()
    ]
    return policy.behavior_clone(demonstrations, epochs=epochs)


def train_one(method: str, seed: int, args) -> tuple[dict, list[dict], dict]:
    config = SimConfig(seed=seed, horizon=args.horizon)
    env = AircraftEnv(args.aircraft, config)
    policy = _make_policy(method, env, seed, args)
    warm = _warm_start(policy, env, seed, args.warmstart_epochs)
    trace = []
    validation_scenarios = make_validation_scenarios(args.aircraft, seed)
    checkpoints = set(range(0, args.updates + 1, args.checkpoint_interval)) | {args.updates}
    best_model = copy.deepcopy(policy.model.state_dict())
    best_score = None
    selected_update = 0
    validation_trace = []
    for update in range(args.updates):
        if update in checkpoints:
            validation_result = evaluate_policy(
                AircraftEnv(args.aircraft, config),
                RuleMonitor(config),
                lambda observation, uid: policy.act(observation, uid, deterministic=True),
                validation_scenarios,
                shield=None,
                horizon=args.horizon,
                name=f"{method}_validation_{update}",
            )
            score = (
                validation_result.conditional_joint_satisfaction_rate,
                -validation_result.post_truth_hard_violation_rate,
                validation_result.mean_reward,
            )
            validation_trace.append({
                "update": update,
                "score": list(score),
                "conditional_joint_stl": validation_result.conditional_joint_satisfaction_rate,
                "post_truth_hard_violation_rate": validation_result.post_truth_hard_violation_rate,
                "mean_reward": validation_result.mean_reward,
            })
            if best_score is None or score > best_score:
                best_score = score
                selected_update = update
                best_model = copy.deepcopy(policy.model.state_dict())
        rollout = policy.collect(
            scenario=make_training_scenario(args.aircraft, seed, update),
            horizon=args.horizon,
            deterministic=False,
        )
        stats = policy.update(rollout)
        trace.append({"seed": seed, "update": update + 1, **{key: float(value) for key, value in stats.items() if isinstance(value, (int, float))}})

    if args.updates in checkpoints:
        validation_result = evaluate_policy(
            AircraftEnv(args.aircraft, config),
            RuleMonitor(config),
            lambda observation, uid: policy.act(observation, uid, deterministic=True),
            validation_scenarios,
            shield=None,
            horizon=args.horizon,
            name=f"{method}_validation_{args.updates}",
        )
        score = (
            validation_result.conditional_joint_satisfaction_rate,
            -validation_result.post_truth_hard_violation_rate,
            validation_result.mean_reward,
        )
        validation_trace.append({
            "update": args.updates,
            "score": list(score),
            "conditional_joint_stl": validation_result.conditional_joint_satisfaction_rate,
            "post_truth_hard_violation_rate": validation_result.post_truth_hard_violation_rate,
            "mean_reward": validation_result.mean_reward,
        })
        if best_score is None or score > best_score:
            best_score = score
            selected_update = args.updates
            best_model = copy.deepcopy(policy.model.state_dict())

    policy.model.load_state_dict(best_model)

    # Validation is independent from the frozen test scenarios and selects the
    # checkpoint lexicographically before the final test evaluation.
    validation = evaluate_policy(
        AircraftEnv(args.aircraft, config),
        RuleMonitor(config),
        lambda observation, uid: policy.act(observation, uid, deterministic=True),
        validation_scenarios,
        shield=None,
        horizon=args.horizon,
        name=f"{method}_validation",
    )
    test_env = AircraftEnv(args.aircraft, config)
    test_shield = _make_policy(method, test_env, seed + 1, args).shield
    # Evaluation must use the trained actor, but its safety layer is a fresh
    # monitor instance so belief history is not carried over from validation.
    result = evaluate_policy(
        test_env,
        RuleMonitor(config),
        lambda observation, uid: policy.act(observation, uid, deterministic=True),
        make_scenarios(args.aircraft, seed, args.episodes),
        shield=test_shield,
        horizon=args.horizon,
        name=method,
    )
    row = result.as_dict()
    row.update({
        "seed": seed,
        "warm_start": warm,
        "selected_update": selected_update,
        "validation": validation.as_dict(),
        "validation_trace": validation_trace,
    })
    return row, result.episode_records, {"trace": trace, "model": policy.model.state_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainable legacy Aircraft baselines")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--warmstart-epochs", type=int, default=10)
    parser.add_argument("--ppo-epoch", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--fixed-penalty-coef", type=float, default=0.35)
    parser.add_argument("--stl-reward-coef", type=float, default=0.20)
    parser.add_argument("--safety-bound", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("results/trainable_baselines_smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows, all_episodes, all_training = [], [], []
    manifests = {}
    for method in args.methods:
        for seed in args.seeds:
            row, episodes, extra = train_one(method, seed, args)
            all_rows.append({"policy": method, **row})
            all_episodes.extend({"policy": method, "seed": seed, **item} for item in episodes)
            all_training.extend([{**item, "policy": method} for item in extra["trace"]])
            checkpoint = args.output / "checkpoints" / method / f"seed_{seed}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(extra["model"], checkpoint)
        manifests[method] = METHODS[method]

    manifest = {
        "protocol": "shared actor/critic architecture, initialization, train/validation/test seed families, update budget and checkpoint rule",
        "third_party_reproduction_status": "internal integration references; do not label as official SOTA reproductions",
        "seed_families": {
            "warm_start": "seed*100000 + 80000 + episode",
            "training": "seed*100000 + 300000 + update",
            "validation": "550000 + seed*1000 + index",
            "test": "seed*10000 + episode plus fixed templates",
        },
        "checkpoint_selection": "maximize conditional joint STL, then minimize truth hard violation, then maximize reward",
        "selected_methods": args.methods,
        "methods": manifests,
        "arguments": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
    }
    payload = {"metadata": manifest, "methods": [_aggregate(all_rows, method) for method in args.methods]}
    (args.output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "per_seed.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "per_episode.json").write_text(json.dumps(all_episodes, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "training_trace.json").write_text(json.dumps(all_training, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "methods": len(args.methods), "records": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
