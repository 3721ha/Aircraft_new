"""Focused eight-template conflict arbitration experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aircraft_sim import ConflictAwareQPSafetyShield, JointSafetyShield, RuleMonitor, SafetyShield, SimConfig, evaluate_policy
from aircraft_sim.policies import rule_agnostic_policy
from aircraft_sim.rules import rule_family
from aircraft_sim.scenarios import (
    capability_dwell_conflict_scenario,
    communication_intent_conflict_scenario,
    coverage_resource_conflict_scenario,
    deadline_information_conflict_scenario,
    dual_support_conflict_scenario,
    mission_recovery_conflict_scenario,
    risk_continuity_conflict_scenario,
    support_separation_conflict_scenario,
)


SCENARIOS = {
    "support_separation": support_separation_conflict_scenario,
    # These templates provide an explicitly feasible initial state for the
    # partitioned conflict protocol.  Calling the factories without the
    # keyword uses their unit-test defaults, which are intentionally already
    # infeasible and would be filtered out by --initial-feasible-only.
    "coverage_resource": lambda n: coverage_resource_conflict_scenario(n, initially_feasible=True),
    "deadline_information": deadline_information_conflict_scenario,
    "dual_support": dual_support_conflict_scenario,
    "mission_recovery": lambda n: mission_recovery_conflict_scenario(n, initially_feasible=True),
    "communication_intent": communication_intent_conflict_scenario,
    "capability_dwell": lambda n: capability_dwell_conflict_scenario(n, initially_feasible=True),
    "risk_continuity": risk_continuity_conflict_scenario,
}


# Target rules are separated from global truth-STL violations.  This is the
# paper's distinction between the conflict under study and unrelated residual
# rules (for example, a coverage soft residual in a separation case).
TARGET_RULES = {
    "support_separation": ({"S02", "S03"}, {"C03"}),
    "coverage_resource": ({"F04"}, {"C02"}),
    "deadline_information": ({"I02", "I03"}, {"M02"}),
    "dual_support": (set(), {"C03"}),
    "mission_recovery": ({"F03", "S04"}, {"M04", "M06"}),
    "communication_intent": ({"F01"}, {"C05"}),
    "capability_dwell": ({"F03"}, {"E02"}),
    "risk_continuity": ({"F06"}, {"M05"}),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused rule-conflict arbitration experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--initial-feasible-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/conflict_arbitration_focus"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        config = SimConfig(seed=seed, horizon=args.horizon)
        for case_name, factory in SCENARIOS.items():
            scenario = factory(args.aircraft)
            env = __import__("aircraft_sim", fromlist=["AircraftEnv"]).AircraftEnv(args.aircraft, config)
            env.reset(scenario)
            initial = RuleMonitor(config).evaluate_truth(env.truth_state()).hard_safe
            if args.initial_feasible_only and not initial:
                continue
            for method, shield in (
                ("NoShield", None),
                ("PartialShield", SafetyShield(RuleMonitor(config))),
                ("JointHeuristic", JointSafetyShield(RuleMonitor(config))),
                ("DG-QP", ConflictAwareQPSafetyShield(config)),
            ):
                result = evaluate_policy(
                    __import__("aircraft_sim", fromlist=["AircraftEnv"]).AircraftEnv(args.aircraft, config),
                    RuleMonitor(config), rule_agnostic_policy, [scenario], shield=shield, horizon=args.horizon, name=method,
                )
                episode = result.episode_records[0] if result.episode_records else {}
                hard_targets, soft_targets = TARGET_RULES[case_name]
                observed = {rule_family(key) for key in episode.get("violations", {})}
                hard_target_violations = observed.intersection(hard_targets)
                soft_target_violations = observed.intersection(soft_targets)
                target_steps = max(1, int(episode.get("steps", args.horizon)))
                rows.append({
                    "seed": seed, "scenario": case_name, "method": method,
                    "initial_hard_safe": initial,
                    "joint_satisfaction_rate": result.joint_satisfaction_rate,
                    "truth_hard_violation_rate": result.post_truth_hard_violation_rate,
                    "target_hard_success_rate": float(not hard_target_violations),
                    "target_hard_rule_violation_count_per_step": len(hard_target_violations) / target_steps,
                    "hard_target_rule_violation_count_per_step": len(hard_target_violations) / target_steps,
                    "soft_target_rule_violation_count_per_step": len(soft_target_violations) / target_steps,
                    "target_hard_rules_observed": sorted(hard_target_violations),
                    "target_soft_rules_observed": sorted(soft_target_violations),
                    "intervention_rate": result.mean_intervention_rate,
                    "qp_infeasible_rate": result.qp_infeasible_rate,
                    "episode": episode,
                })
    metadata = {"seeds": args.seeds, "horizon": args.horizon, "aircraft": args.aircraft, "initial_feasible_only": args.initial_feasible_only, "scenarios": list(SCENARIOS)}
    (args.output / "per_seed.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
