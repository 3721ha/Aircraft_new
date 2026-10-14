"""Run controlled component ablations for the complete belief-STL/DG-QP layer.

All variants use the same full rule reports, frozen conflict templates, seeds,
nominal policy, and evaluation horizon.  Only one arbitration component is
disabled at a time, so the output can support paired net-benefit analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aircraft_sim import AircraftEnv, ConflictAwareQPSafetyShield, RuleMonitor, SimConfig, evaluate_policy
from aircraft_sim.policies import rule_agnostic_policy
from run_conflict_arbitration_experiments import SCENARIOS, TARGET_RULES


VARIANTS = {
    "DG-QP": {},
    "No-Graph": {"use_dependency_graph": False},
    "Fixed-Priority": {"use_dynamic_priority": False},
    "Gate-Only": {"use_continuous_qp": False},
}


def _target_success(episode: dict, scenario: str) -> float:
    observed = set(episode.get("violations", {}))
    hard_targets, _ = TARGET_RULES[scenario]
    families = {key.split(":", 1)[0] for key in observed}
    return float(not families.intersection(hard_targets))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505, 606, 707, 808, 909, 1001])
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--initial-feasible-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/component_ablation_10seed"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    retained_cases = []
    for seed in args.seeds:
        config = SimConfig(seed=seed, horizon=args.horizon)
        for scenario_name, factory in SCENARIOS.items():
            scenario = factory(args.aircraft)
            probe = AircraftEnv(args.aircraft, config)
            probe.reset(scenario)
            initial_report = RuleMonitor(config).evaluate_truth(probe.truth_state())
            if args.initial_feasible_only and not initial_report.hard_safe:
                continue
            retained_cases.append({"seed": seed, "scenario": scenario_name})
            for variant, switches in VARIANTS.items():
                env = AircraftEnv(args.aircraft, config)
                shield = ConflictAwareQPSafetyShield(config, **switches)
                result = evaluate_policy(
                    env,
                    RuleMonitor(config),
                    rule_agnostic_policy,
                    [scenario],
                    shield=shield,
                    horizon=args.horizon,
                    name=variant,
                )
                episode = result.episode_records[0] if result.episode_records else {}
                rows.append({
                    "seed": seed,
                    "scenario": scenario_name,
                    "variant": variant,
                    "initial_hard_safe": initial_report.hard_safe,
                    "mean_reward": result.mean_reward,
                    "joint_satisfaction_rate": result.joint_satisfaction_rate,
                    "conditional_joint_satisfaction_rate": result.conditional_joint_satisfaction_rate,
                    "post_truth_hard_violation_rate": result.post_truth_hard_violation_rate,
                    "target_hard_success_rate": _target_success(episode, scenario_name),
                    "mean_intervention_rate": result.mean_intervention_rate,
                    "mean_online_time_ms": result.mean_online_time_ms,
                    "qp_infeasible_rate": result.qp_infeasible_rate if variant != "Gate-Only" else None,
                    "rule_rule_conflict_step_rate": result.rule_rule_conflict_step_rate,
                    "rule_rule_resolution_rate": result.rule_rule_resolution_rate,
                    "steps": episode.get("steps", args.horizon),
                    "violations": episode.get("violations", {}),
                })

    manifest = {
        "protocol": "controlled_component_ablation_v1",
        "seeds": args.seeds,
        "horizon": args.horizon,
        "aircraft": args.aircraft,
        "initial_feasible_only": args.initial_feasible_only,
        "scenarios": list(SCENARIOS),
        "variants": VARIANTS,
        "common_rule_coverage": True,
        "nominal_policy": "rule_agnostic_policy",
        "variant_definitions": {
            "DG-QP": "dependency graph + dynamic priority + continuous residual QP",
            "No-Graph": "same full rule reports and priorities, no explicit dependency edges/conflict events",
            "Fixed-Priority": "dependency graph + fixed P0-P5 priority, no probability/urgency reordering",
            "Gate-Only": "dependency graph + dynamic priority + discrete gating, no continuous residual solve",
        },
    }
    (args.output / "per_seed.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(rows), "retained_cases": len(retained_cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
