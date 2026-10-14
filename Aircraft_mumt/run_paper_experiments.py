"""Shared scenario and metric helpers for the legacy Aircraft experiments.

This module is intentionally small and deterministic.  The official baseline
entry points import it for the common train/validation/test scenario protocol;
keeping the protocol in one place prevents a baseline from silently seeing a
different frozen test set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aircraft_sim import (
    AircraftEnv,
    ConflictAwareQPSafetyShield,
    JointSafetyShield,
    RuleMonitor,
    SimConfig,
    evaluate_policy,
)
from aircraft_sim.policies import (
    conservative_policy,
    fixed_penalty_policy,
    lagrangian_policy,
    rule_agnostic_policy,
    stl_robustness_policy,
)
from aircraft_sim.scenarios import (
    boundary_scenario,
    conflict_scenario,
    coverage_resource_conflict_scenario,
    deadline_information_conflict_scenario,
    protocol_conflict_scenario,
    safe_scenario,
    sample_scenario,
)


# Numeric fields consumed by all official adapters.  Dictionary-valued rule
# diagnostics remain in per-seed records and are not averaged here.
SCALAR_FIELDS = [
    "mean_reward",
    "joint_satisfaction_rate",
    "conditional_joint_satisfaction_rate",
    "initial_infeasible_rate",
    "mean_intervention_rate",
    "mean_intervention_distance",
    "online_hard_violation_rate",
    "post_shield_hard_violation_rate",
    "post_truth_hard_violation_rate",
    "qp_infeasible_rate",
    "conflict_rate",
    "replan_trigger_rate",
    "mean_belief_risk",
    "mean_pre_filter_belief_risk",
    "mean_post_filter_belief_risk",
    "mean_actor_time_ms",
    "mean_online_time_ms",
    "tail_violation_rate",
    "risk_calibration_brier",
    "rule_conflict_event_rate",
    "rule_rule_conflict_step_rate",
    "rule_rule_resolution_rate",
    "mean_conflict_residual",
    "mean_solver_agents",
    "mean_solver_components",
    "max_solver_component_size",
    "cross_component_violation_rate",
    "cross_component_repair_rate",
]


def _fixed_scenarios(n_aircraft: int) -> list[dict]:
    return [
        {**safe_scenario(n_aircraft), "scenario_type": "safe_internal"},
        {**boundary_scenario(n_aircraft), "scenario_type": "boundary"},
        {**conflict_scenario(n_aircraft), "scenario_type": "conflict"},
        {**protocol_conflict_scenario(n_aircraft), "scenario_type": "protocol_conflict"},
    ]


def make_scenarios(n_aircraft: int, seed: int, episodes: int) -> list[dict]:
    """Return the frozen test suite used by the 9/15-era paper runs.

    The first four rows are fixed diagnostic templates.  Remaining rows are
    reproducible sampled cases, giving the 10-seed protocol 4 fixed + 36
    sampled episodes when ``episodes=40``.
    """

    fixed = _fixed_scenarios(n_aircraft)
    sampled = [
        {
            **sample_scenario(
                n_aircraft,
                seed=seed * 10_000 + index,
                difficulty=0.15 + 0.75 * ((index % 10) / 9.0),
            ),
            "scenario_type": "sampled",
        }
        for index in range(max(0, episodes - len(fixed)))
    ]
    return (fixed + sampled)[:episodes]


def make_validation_scenarios(n_aircraft: int, seed: int, count: int = 12) -> list[dict]:
    """Construct a seed-separated validation suite."""

    fixed = _fixed_scenarios(n_aircraft)
    sampled = [
        {
            **sample_scenario(
                n_aircraft,
                seed=550_000 + seed * 1_000 + index,
                difficulty=0.20 + 0.65 * ((index % 8) / 7.0),
            ),
            "scenario_type": "validation_sampled",
        }
        for index in range(max(0, count - len(fixed)))
    ]
    return (fixed + sampled)[:count]


def make_training_scenario(n_aircraft: int, seed: int, update: int) -> dict:
    return {
        **sample_scenario(
            n_aircraft,
            seed=seed * 100_000 + 300_000 + update,
            difficulty=0.20 + 0.60 * ((update % 5) / 4.0),
        ),
        "scenario_type": "training_sampled",
    }


def _policy(name: str):
    return {
        "nominal": rule_agnostic_policy,
        "conservative": conservative_policy,
        "fixed_penalty": fixed_penalty_policy,
        "stl_robustness": stl_robustness_policy,
        "lagrangian": lagrangian_policy,
    }[name]


def run_legacy_benchmark(
    seeds: list[int],
    episodes: int,
    horizon: int,
    aircraft: int,
    output: Path,
) -> dict:
    """Run the small pre-official benchmark used for interface validation."""

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        config = SimConfig(seed=seed, horizon=horizon)
        scenarios = make_scenarios(aircraft, seed, episodes)
        methods = [
            ("nominal_unshielded", "nominal", None),
            ("fixed_penalty", "fixed_penalty", None),
            ("stl_robustness_greedy", "stl_robustness", None),
            ("lagrangian_heuristic", "lagrangian", None),
            ("nominal_shielded", "nominal", JointSafetyShield(RuleMonitor(config))),
            ("belief_qp_shielded", "nominal", ConflictAwareQPSafetyShield(config)),
        ]
        for name, policy_name, shield in methods:
            result = evaluate_policy(
                AircraftEnv(aircraft, config),
                RuleMonitor(config),
                _policy(policy_name),
                scenarios,
                shield=shield,
                horizon=horizon,
                name=name,
            )
            row = result.as_dict()
            row["seed"] = seed
            rows.append(row)

    (output / "benchmark.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    scalar_rows = []
    for row in rows:
        scalar_rows.append({"seed": row["seed"], "policy": row["policy"], **{key: row.get(key, 0.0) for key in SCALAR_FIELDS}})
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]) if scalar_rows else ["seed", "policy"])
        writer.writeheader()
        writer.writerows(scalar_rows)
    return {"rows": len(rows), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy decision-level Aircraft benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
    args = parser.parse_args()
    print(json.dumps(run_legacy_benchmark(vars(args)["seeds"], args.episodes, args.horizon, args.aircraft, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
