"""Run the short-horizon support/separation safety-retention experiment.

The tested scenarios are initially hard-safe.  Therefore the reported
three-/thirty-step endpoints measure retention of S02/S03 safety from an
initially safe close-range state; they are not post-violation recovery rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aircraft_sim import (
    AircraftEnv,
    Action,
    ConflictAwareQPSafetyShield,
    JointSafetyShield,
    Mode,
    RuleMonitor,
    SafetyShield,
    SimConfig,
)
from aircraft_sim.policies import rule_agnostic_policy
from aircraft_sim.rules import rule_family
from aircraft_sim.scenarios import support_separation_conflict_scenario


METHODS = {
    "NoShield": None,
    "PartialShield": "partial",
    "JointHeuristic": "joint",
    "DG-QP": "dg_qp",
}
DISTANCES = [2000.0, 2200.0, 2400.0, 2800.0, 3200.0]
PAPER_PROTOCOL_OVERRIDES = {
    "support_response_deadline": 0,
    "noise_position": 0.0,
    "noise_speed": 0.0,
    "belief_ttc_threshold": 6.0,
    "safe_trend_window": 4,
}


def scenario_at_distance(distance: float) -> dict:
    scenario = support_separation_conflict_scenario(3)
    scenario["positions"] = [(0.0, 0.0, 5000.0), (distance, 0.0, 5000.0), (0.0, 3000.0, 5100.0)]
    return scenario


def lateral_probe_feasible(config: SimConfig, scenario: dict, steps: int = 8) -> bool:
    env = AircraftEnv(3, config)
    monitor = RuleMonitor(config)
    observations = env.reset(scenario)
    for _ in range(steps):
        # Fixed symmetric lateral turn used only as a feasibility probe; no
        # policy or safety shield is involved in this pre-check.
        actions = {
            0: Action(Mode.RECOVER, turn=0.8, acceleration=0.0),
            1: Action(Mode.RECOVER, turn=0.8, acceleration=0.0),
            2: Action(Mode.HOLD),
        }
        result = env.step(actions)
        report = monitor.evaluate_truth(result.info["truth"], actions)
        if any(rule_family(rule) in {"S02", "S03"} for rule in report.violations):
            return False
        observations = result.observations
    return True


def shield_for(method: str, config: SimConfig):
    if METHODS[method] == "partial":
        return SafetyShield(RuleMonitor(config))
    if METHODS[method] == "joint":
        return JointSafetyShield(RuleMonitor(config))
    if METHODS[method] == "dg_qp":
        return ConflictAwareQPSafetyShield(config)
    return None


def rollout(config: SimConfig, scenario: dict, method: str, horizon: int) -> dict:
    env = AircraftEnv(3, config)
    monitor = RuleMonitor(config)
    shield = shield_for(method, config)
    observations = env.reset(scenario)
    target_violations = []
    interventions = 0
    fallback = 0
    for step in range(horizon):
        nominal = {uid: rule_agnostic_policy(observations[uid], uid) for uid in observations}
        if shield is None:
            actions, events = nominal, []
        else:
            actions, events = shield.filter(observations, nominal)
            fallback += int(getattr(getattr(shield, "last_solution", None), "status", "") == "infeasible_fallback")
        interventions += len(events)
        result = env.step(actions)
        report = monitor.evaluate_truth(result.info["truth"], actions)
        if any(rule_family(rule) in {"S02", "S03"} for rule in report.violations):
            target_violations.append(step)
        observations = result.observations
    return {
        "target_hard_safe_3step": float(not any(step < 3 for step in target_violations)),
        "target_hard_safe_30step": float(not target_violations),
        "target_violation_steps": target_violations,
        "intervention_rate": interventions / max(1, horizon),
        "qp_fallback_rate": fallback / max(1, horizon),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run initially-safe high-conflict safety retention (not post-violation recovery)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505, 606, 707, 808, 909, 1001])
    parser.add_argument("--distances", nargs="+", type=float, default=DISTANCES)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--probe-steps", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        # Section 6.5 uses a frozen high-conflict protocol.  These overrides
        # are part of the protocol and must not silently fall back to the
        # general-purpose simulator defaults.
        config = SimConfig(seed=seed, horizon=args.horizon, **PAPER_PROTOCOL_OVERRIDES)
        for distance in args.distances:
            scenario = scenario_at_distance(distance)
            env = AircraftEnv(3, config)
            env.reset(scenario)
            initial_safe = RuleMonitor(config).evaluate_truth(env.truth_state()).hard_safe
            probe_safe = initial_safe and lateral_probe_feasible(config, scenario, args.probe_steps)
            for method in METHODS:
                metrics = rollout(config, scenario, method, args.horizon) if probe_safe else {
                    "target_hard_safe_3step": None,
                    "target_hard_safe_30step": None,
                    "target_violation_steps": [],
                    "intervention_rate": 0.0,
                    "qp_fallback_rate": 0.0,
                }
                rows.append({"seed": seed, "distance": distance, "method": method, "initial_safe": initial_safe, "probe_safe": probe_safe, **metrics})

    summary = []
    for distance in args.distances:
        for method in METHODS:
            selected = [row for row in rows if row["distance"] == distance and row["method"] == method and row["probe_safe"]]
            summary.append({
                "distance": distance,
                "method": method,
                "n": len(selected),
                "three_step_safety_retention_rate": sum(row["target_hard_safe_3step"] for row in selected) / max(1, len(selected)),
                "thirty_step_safety_retention_rate": sum(row["target_hard_safe_30step"] for row in selected) / max(1, len(selected)),
                "mean_intervention_rate": sum(row["intervention_rate"] for row in selected) / max(1, len(selected)),
                "mean_qp_fallback_rate": sum(row["qp_fallback_rate"] for row in selected) / max(1, len(selected)),
            })
    payload = {
        "metadata": {
            "seeds": args.seeds,
            "distances": args.distances,
            "horizon": args.horizon,
            "probe_steps": args.probe_steps,
            "target_rules": ["S02", "S03"],
            "protocol_overrides": PAPER_PROTOCOL_OVERRIDES,
            "initial_state": "hard_safe",
            "endpoint_definition": "No S02/S03 violation within the first 3 or 30 executed steps; this is safety retention, not post-violation recovery.",
        },
        "summary": summary,
    }
    (args.output / "per_seed.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| Initial distance (m) | Method | N | 3-step safety retention | 30-step safety retention | Intervention | QP fallback |", "|---:|---|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {row['distance']:.0f} | {row['method']} | {row['n']} | {row['three_step_safety_retention_rate']:.3f} | {row['thirty_step_safety_retention_rate']:.3f} | {row['mean_intervention_rate']:.3f} | {row['mean_qp_fallback_rate']:.3f} |" for row in summary)
    (args.output / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
