"""Record step-by-step conflict traces from the rebuilt policy checkpoints.

This runner is intentionally separate from the aggregate conflict experiment.
The aggregate experiment uses a deterministic rule-agnostic policy to isolate
the safety layer; this script loads the trained actor checkpoints and records
the actual nominal action, shield decision, belief report, graph events and
truth outcome at every decision step.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from aircraft_sim import (
    AircraftEnv,
    ConflictAwareQPSafetyShield,
    MAPPOConfig,
    MAPPOPolicy,
    RuleMonitor,
    SimConfig,
    PassThroughShield,
)
from aircraft_sim.models import Action
from external_adapters import (
    OfficialHAPPOAdapter,
    OfficialHAPPOConfig,
    OfficialHATRPOAdapter,
    OfficialHATRPOConfig,
    OfficialMAPPOAdapter,
    OfficialMAPPOConfig,
    OfficialMACPOAdapter,
    OfficialMACPOConfig,
    OfficialMAPPOLagrangianAdapter,
    OfficialMAPPOLagrangianConfig,
    OfficialMATAdapter,
    OfficialMATConfig,
)
from run_conflict_arbitration_experiments import SCENARIOS


METHODS = (
    "proposed",
    "mappo",
    "happo",
    "hatrpo",
    "macpo",
    "mappo_lagrangian",
    "mat",
)


def json_safe(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def action_dict(action: Action) -> dict:
    return {
        "mode": action.mode.value,
        "turn": float(action.turn),
        "climb": float(action.climb),
        "acceleration": float(action.acceleration),
        "target": action.target,
        "target_kind": getattr(action.target_kind, "value", action.target_kind),
    }


def report_dict(report) -> dict | None:
    if report is None:
        return None
    return json_safe({
        "violations": report.violations,
        "hard_safe": report.hard_safe,
        "violation_probability": getattr(report, "violation_probability", {}),
        "robust_lower": getattr(report, "robust_lower", {}),
    })


def _checkpoint_root(root: Path, method: str, seed: int, proposed_root: Path, official_root: Path) -> Path:
    if method == "proposed":
        return proposed_root / "checkpoints" / "proposed_belief_stl_conflict_qp" / f"seed_{seed}.pt"
    return official_root / method / "checkpoints" / f"seed_{seed}"


def build_actor(method: str, seed: int, horizon: int, root: Path, proposed_root: Path, official_root: Path):
    config = SimConfig(seed=seed, horizon=horizon)
    env = AircraftEnv(3, config)
    checkpoint = _checkpoint_root(root, method, seed, proposed_root, official_root)
    if method == "proposed":
        policy = MAPPOPolicy(env, PassThroughShield(), config=MAPPOConfig(), seed=seed)
        policy.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        return env, lambda observations: {
            uid: policy.act(observations[uid], uid, deterministic=True)
            for uid in observations
        }, checkpoint
    if method == "mappo":
        policy = OfficialMAPPOAdapter(env, OfficialMAPPOConfig(action_adapter="continuous", ppo_epoch=5), seed=seed)
        policy.load(checkpoint)
        return env, lambda observations: {
            uid: policy.act(observations[uid], uid, deterministic=True)
            for uid in observations
        }, checkpoint
    if method == "happo":
        policy = OfficialHAPPOAdapter(env, OfficialHAPPOConfig(ppo_epoch=5), seed=seed)
        policy.load(checkpoint)
        return env, lambda observations: {
            uid: policy.act(observations[uid], uid, deterministic=True)
            for uid in observations
        }, checkpoint
    if method == "hatrpo":
        policy = OfficialHATRPOAdapter(
            env,
            OfficialHATRPOConfig(ppo_epoch=5, line_search_steps=10, kl_threshold=0.01, accept_ratio=0.5),
            seed=seed,
        )
        policy.load(checkpoint)
        return env, lambda observations: {
            uid: policy.act(observations[uid], uid, deterministic=True)
            for uid in observations
        }, checkpoint
    if method == "macpo":
        policy = OfficialMACPOAdapter(
            env,
            OfficialMACPOConfig(ppo_epoch=5, line_search_steps=10, safety_bound=0.1, cost_signal="episode_binary"),
            seed=seed,
        )
        policy.load(checkpoint)
        return env, lambda observations: {
            uid: policy.act(observations[uid], uid, deterministic=True)
            for uid in observations
        }, checkpoint
    if method == "mappo_lagrangian":
        policy = OfficialMAPPOLagrangianAdapter(
            env,
            OfficialMAPPOLagrangianConfig(ppo_epoch=5, safety_bound=0.1, cost_signal="episode_binary"),
            seed=seed,
        )
        policy.load(checkpoint)
        return env, lambda observations: {
            uid: policy.act(observations[uid], uid, deterministic=True)
            for uid in observations
        }, checkpoint
    if method == "mat":
        policy = OfficialMATAdapter(env, OfficialMATConfig(ppo_epoch=5), seed=seed)
        policy.load(checkpoint)
        return env, lambda observations: policy.act_joint(observations, deterministic=True), checkpoint
    raise ValueError(f"unknown method: {method}")


def trace_episode(method: str, seed: int, case: str, horizon: int, root: Path, shared_qp: bool, proposed_root: Path, official_root: Path) -> list[dict]:
    env, actor, checkpoint = build_actor(method, seed, horizon, root, proposed_root, official_root)
    scenario = SCENARIOS[case](env.n)
    observations = env.reset(scenario)
    monitor = RuleMonitor(env.cfg)
    shield = ConflictAwareQPSafetyShield(env.cfg) if shared_qp else None
    rows = []
    for step_index in range(horizon):
        nominal = actor(observations)
        before_truth = env.truth_state()
        if shield is None:
            safe = nominal
            interventions = []
            solution = None
        else:
            if hasattr(shield.monitor, "reset") and step_index == 0:
                shield.monitor.reset()
            safe, interventions = shield.filter(observations, nominal)
            solution = shield.last_solution
        result = env.step(safe)
        truth_report = monitor.evaluate_truth(result.info["truth"], safe)
        payload = {
            "case": case,
            "seed": seed,
            "method": method,
            "shield": "shared_qp" if shared_qp else "unshielded",
            "checkpoint": str(checkpoint),
            "t": step_index,
            "observations": observations,
            "nominal_actions": {str(uid): action_dict(action) for uid, action in nominal.items()},
            "safe_actions": {str(uid): action_dict(action) for uid, action in safe.items()},
            "interventions": json_safe(interventions),
            "truth_before": before_truth,
            "executed_truth": result.info["truth"],
            "truth_violations": sorted(truth_report.violations),
            "truth_hard_safe": truth_report.hard_safe,
            "truth_margins": truth_report.margins,
        }
        if solution is not None:
            payload["belief_report"] = report_dict(solution.belief_report)
            payload["post_belief_report"] = report_dict(solution.post_belief_report)
            payload["dependency_graph"] = solution.dependency_graph.as_dict() if solution.dependency_graph else None
            payload["graph_conflict_events"] = solution.conflict_events
            payload["qp"] = {
                "status": solution.status,
                "objective": solution.objective,
                "infeasibility_reason": solution.infeasibility_reason,
                "hard_constraints": solution.hard_constraints,
                "relaxed_constraints": solution.relaxed_constraints,
                "conflict_rules": solution.conflict_rules,
                "solver_agents": solution.solver_agents,
                "solver_components": solution.solver_components,
                "cross_component_violations": solution.cross_component_violations,
                "cross_component_repairs": solution.cross_component_repairs,
            }
        else:
            payload["belief_report"] = None
            payload["post_belief_report"] = None
            payload["dependency_graph"] = None
            payload["graph_conflict_events"] = []
            payload["qp"] = None
        rows.append(json_safe(payload))
        observations = result.observations
        if result.done:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Record trained-checkpoint rule-conflict traces")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/conflict_trace_rebuild"))
    parser.add_argument("--policy-only", action="store_true", help="record only unshielded policy traces")
    parser.add_argument("--proposed-root", type=Path, default=Path("results/proposed_rebuild_official_10seed"))
    parser.add_argument("--official-root", type=Path, default=Path("results/official_comparison_rebuild_10seed"))
    args = parser.parse_args()
    if args.aircraft != 3:
        raise ValueError("the rebuilt checkpoints were trained for three aircraft")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        for case in SCENARIOS:
            for method in args.methods:
                for shared_qp in ([False] if args.policy_only else [False, True]):
                    rows.extend(trace_episode(method, seed, case, args.horizon, Path.cwd(), shared_qp, args.proposed_root, args.official_root))
    metadata = {
        "seeds": args.seeds,
        "methods": args.methods,
        "cases": list(SCENARIOS),
        "horizon": args.horizon,
        "aircraft": args.aircraft,
        "shared_qp": not args.policy_only,
        "source": "rebuilt trained checkpoints; per-step observations, actions, belief reports, graph events and truth outcomes",
    }
    (args.output / "step_trace.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    conflict_rows = [row for row in rows if row["graph_conflict_events"] or row["truth_violations"]]
    (args.output / "conflict_steps.json").write_text(json.dumps(conflict_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(rows), "conflict_records": len(conflict_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
