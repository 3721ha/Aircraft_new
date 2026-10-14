"""Controlled activation experiment for the dynamic rule-priority mechanism.

This is deliberately separate from the official protocol.  It keeps the
existing nominal policy and safety layer unchanged, but initializes one
aircraft so that an information rule (I02) and a resource rule (F04) are
simultaneously active and propose different discrete modes (RECOVER vs EXIT).
The paired dynamic/fixed runs use the same seed and scenario.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from aircraft_sim import (
    AircraftEnv,
    ConflictAwareQPSafetyShield,
    RuleMonitor,
    SimConfig,
    evaluate_policy,
)
from aircraft_sim.models import Action
from aircraft_sim.policies import rule_agnostic_policy


SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]
SCENARIO_NAME = "information_resource_priority_conflict"

# The positions are intentionally far apart: this experiment isolates the
# discrete arbitration choice rather than adding a second separation conflict.
# A0 is nominally CRITICAL.  id_confidence=.64 activates I02 while resource
#=.12 activates F04.  Their proposals are RECOVER and EXIT respectively.
SCENARIO = {
    "positions": [(0.0, 0.0, 5000.0), (10000.0, 0.0, 5000.0), (0.0, 10000.0, 5100.0)],
    "resources": [0.12, 0.80, 0.80],
    "id_confidence": {0: 0.64},
    "task_progress": {0: 0.10, 1: 0.10, 2: 0.10},
}

VARIANTS = {
    "Dynamic-Priority": True,
    "Fixed-Priority": False,
}


def _action_dict(action: Action) -> dict:
    return {
        "mode": action.mode.value,
        "turn": round(float(action.turn), 6),
        "climb": round(float(action.climb), 6),
        "acceleration": round(float(action.acceleration), 6),
        "target": action.target,
        "target_kind": action.target_kind,
    }


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def _initial_arbitration(seed: int, horizon: int, dynamic: bool) -> dict:
    config = SimConfig(seed=seed, horizon=horizon)
    env = AircraftEnv(3, config)
    observations = env.reset(SCENARIO)
    nominal = {i: rule_agnostic_policy(observations[i], i) for i in range(3)}
    shield = ConflictAwareQPSafetyShield(config, use_dynamic_priority=dynamic)
    safe, _ = shield.filter(observations, nominal)
    solution = shield.last_solution
    report = solution.belief_report
    graph = solution.dependency_graph
    active = sorted(report.violations) if report is not None else []
    priorities = {}
    if graph is not None:
        priorities = {
            key: round(float(node.dynamic_priority), 6)
            for key, node in graph.rules.items()
            if key in {"I02:0", "F04:0"}
        }
    return {
        "active_rules": active,
        "target_rules_active": [key for key in ("I02:0", "F04:0") if key in active],
        "target_rule_probabilities": {
            key: round(float(report.violation_probability[key]), 6)
            for key in ("I02:0", "F04:0")
            if report is not None and key in report.violation_probability
        },
        "dynamic_priorities": priorities,
        "nominal_actions": {str(i): _action_dict(action) for i, action in nominal.items()},
        "selected_actions": {str(i): _action_dict(action) for i, action in safe.items()},
        "agent0_selected_mode": safe[0].mode.value,
        "status": solution.status,
        "infeasibility_reason": solution.infeasibility_reason,
    }


def _rollout(seed: int, horizon: int, dynamic: bool) -> dict:
    config = SimConfig(seed=seed, horizon=horizon)
    result = evaluate_policy(
        AircraftEnv(3, config),
        RuleMonitor(config),
        rule_agnostic_policy,
        [SCENARIO],
        shield=ConflictAwareQPSafetyShield(config, use_dynamic_priority=dynamic),
        horizon=horizon,
        name="Dynamic-Priority" if dynamic else "Fixed-Priority",
    )
    return result.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final_protocol_20261007/dynamic_priority_activation_10seed"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        initial = {
            variant: _initial_arbitration(seed, args.horizon, dynamic)
            for variant, dynamic in VARIANTS.items()
        }
        rollout = {
            variant: _rollout(seed, args.horizon, dynamic)
            for variant, dynamic in VARIANTS.items()
        }
        dynamic_mode = initial["Dynamic-Priority"]["agent0_selected_mode"]
        fixed_mode = initial["Fixed-Priority"]["agent0_selected_mode"]
        mode_diff = float(dynamic_mode != fixed_mode)
        both_active = float(
            set(("I02:0", "F04:0"))
            <= set(initial["Dynamic-Priority"]["active_rules"])
        )
        for variant in VARIANTS:
            row = {
                "seed": seed,
                "scenario": SCENARIO_NAME,
                "variant": variant,
                "dynamic_priority_enabled": VARIANTS[variant],
                "initial_target_pair_active": bool(both_active),
                "dynamic_fixed_mode_difference": mode_diff,
                "initial_arbitration": initial[variant],
                "metrics": rollout[variant],
            }
            rows.append(row)

    (args.output / "per_seed.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "protocol": "dynamic_priority_activation_v1",
        "seeds": args.seeds,
        "horizon": args.horizon,
        "scenario": SCENARIO_NAME,
        "scenario_definition": SCENARIO,
        "variants": VARIANTS,
        "paired_same_seed": True,
        "official_protocol_unchanged": True,
        "primary_endpoint": "different discrete arbitration mode under simultaneous I02/F04 activation",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "records": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
