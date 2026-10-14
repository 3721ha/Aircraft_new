"""Evaluate official pretrained GCBF+ on Aircraft's physical-rule subtask."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from aircraft_sim import AircraftEnv, PhysicalCBFQPShield, RuleMonitor, SimConfig, evaluate_policy
from aircraft_sim.policies import rule_agnostic_policy
from external_adapters import (
    GCBFPlusDependencyError,
    OfficialGCBFPlusConfig,
    OfficialGCBFPlusShield,
    gcbfplus_dependencies_available,
)
from aircraft_sim.scenarios import (
    boundary_scenario,
    conflict_scenario,
    protocol_conflict_scenario,
    safe_scenario,
    sample_scenario,
)


PHYSICAL_RULES = ("S01", "S02", "S03")


def make_scenarios(n_aircraft: int, seed: int, episodes: int) -> list[dict]:
    """Build a deterministic physical-safety suite without legacy paper code."""
    fixed = [
        {**safe_scenario(n_aircraft), "scenario_type": "safe_internal"},
        {**boundary_scenario(n_aircraft), "scenario_type": "boundary"},
        {**conflict_scenario(n_aircraft), "scenario_type": "conflict"},
        {**protocol_conflict_scenario(n_aircraft), "scenario_type": "protocol_conflict"},
    ]
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


def physical_metrics(episode_records: list[dict]) -> dict:
    count = max(1, len(episode_records))
    violated = {rule: 0 for rule in PHYSICAL_RULES}
    satisfied = 0
    for episode in episode_records:
        rules = set(episode.get("violations", {}))
        episode_physical = {rule for rule in PHYSICAL_RULES if rule in rules}
        satisfied += int(not episode_physical)
        for rule in episode_physical:
            violated[rule] += 1
    return {
        "physical_joint_satisfaction_rate": satisfied / count,
        "physical_violation_rate": {rule: violated[rule] / count for rule in PHYSICAL_RULES},
    }


def aggregate(records):
    summaries = []
    for policy in dict.fromkeys(row["policy"] for row in records):
        selected = [row for row in records if row["policy"] == policy]
        values = [row["physical_joint_satisfaction_rate"] for row in selected]
        sd = stdev(values) if len(values) > 1 else 0.0
        summaries.append({
            "policy": policy,
            "seeds": len(selected),
            "physical_joint_satisfaction_rate_mean": round(mean(values), 6),
            "physical_joint_satisfaction_rate_std": round(sd, 6),
            "physical_joint_satisfaction_rate_ci95": round(1.96 * sd / math.sqrt(len(values)), 6),
            "mean_reward_mean": round(mean(row["mean_reward"] for row in selected), 6),
            "mean_intervention_rate_mean": round(mean(row["mean_intervention_rate"] for row in selected), 6),
            "mean_intervention_distance_mean": round(mean(row["mean_intervention_distance"] for row in selected), 6),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Run official GCBF+ as a physical Aircraft shield")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--aircraft", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/official_gcbfplus_physical_smoke"))
    args = parser.parse_args()

    available = gcbfplus_dependencies_available()
    if args.check_only:
        print(json.dumps({"gcbfplus_dependencies_available": available}))
        return
    if not available:
        raise SystemExit(
            "GCBF+ dependencies are unavailable. Use the isolated .venv-gcbf environment described in "
            "external_methods/inbox/gcbfplus/REPO_INFO.md."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    episodes = []
    fingerprints = set()
    for seed in args.seeds:
        sim_config = SimConfig(seed=seed, horizon=args.horizon)
        env = AircraftEnv(args.aircraft, sim_config)
        try:
            shield = OfficialGCBFPlusShield(env, OfficialGCBFPlusConfig())
        except GCBFPlusDependencyError as error:
            raise SystemExit(str(error)) from error
        fingerprints.add(shield.source_fingerprint)
        scenarios = make_scenarios(args.aircraft, seed, args.episodes)
        comparisons = (
            ("nominal_unshielded_physical", None),
            ("internal_physical_cbf_qp", PhysicalCBFQPShield(sim_config)),
            ("official_gcbfplus_physical", shield),
        )
        for policy_name, active_shield in comparisons:
            result = evaluate_policy(
                AircraftEnv(args.aircraft, sim_config),
                RuleMonitor(sim_config),
                rule_agnostic_policy,
                scenarios,
                shield=active_shield,
                horizon=args.horizon,
                name=policy_name,
            )
            row = result.as_dict()
            row.update(physical_metrics(result.episode_records))
            row.update({
                "policy": policy_name,
                "seed": seed,
                "repository": "https://github.com/MIT-REALM/gcbfplus" if active_shield is shield else "Aircraft internal reference",
                "commit": shield.commit if active_shield is shield else None,
                "source_fingerprint": shield.source_fingerprint if active_shield is shield else None,
                "pretrained_source_agents": shield.config.pretrained_agents if active_shield is shield else None,
                "checkpoint_obstacles": shield.config.checkpoint_obstacles if active_shield is shield else None,
                "deployed_obstacles": shield.config.deployed_obstacles if active_shield is shield else None,
                "lidar_rays": shield.config.lidar_rays if active_shield is shield else None,
                "deployed_aircraft": args.aircraft,
            })
            records.append(row)
            episodes.extend({
                "policy": policy_name,
                "seed": seed,
                "scenario_type": scenarios[item["episode"]].get("scenario_type", "unknown"),
                **item,
            } for item in result.episode_records)

    official_record = next(
        (row for row in records if row["policy"] == "official_gcbfplus_physical"), None
    )
    manifest = {
        "method": "GCBF+",
        "scope": "physical rules S01/S02/S03 only; not a full rule-knowledge baseline",
        "implementation": "official pretrained LinearDrone GCBF+ actor used as a continuous residual shield",
        "repository": "https://github.com/MIT-REALM/gcbfplus",
        "commit": official_record["commit"] if official_record else None,
        "source_fingerprints": sorted(fingerprints),
        "license": "MIT",
        "third_party_source_modified": False,
        "adapter": "external_adapters/official_gcbfplus.py",
        "nominal_semantic_policy": "rule_agnostic_policy",
        "physical_mapping": "Aircraft relative position/velocity to a LinearDrone-compatible graph; GCBF Cartesian acceleration to turn/climb/acceleration",
        "obstacle_mapping": "checkpoint training used 8 random obstacles; Aircraft has no obstacle state, so deployment uses zero obstacle instances without adding phantom hazards",
        "transfer_limit": "official 8-agent LinearDrone checkpoint is deployed on 3 Aircraft agents without Aircraft-domain retraining",
        "reporting_limit": "full-STL joint satisfaction is diagnostic only; primary comparison uses physical_joint_satisfaction_rate",
        "comparison_rows": [
            "nominal_unshielded_physical",
            "internal_physical_cbf_qp",
            "official_gcbfplus_physical",
        ],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output / "per_seed.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "per_episode.json").write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps({"metadata": manifest, "methods": aggregate(records)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
