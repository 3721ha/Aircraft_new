"""Paired S02/S03 recovery after a shared, actually observed violation.

This does not change the environment, policies, shields, or rule thresholds.
The old initially-safe retention experiment remains a separate experiment.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from aircraft_sim import AircraftEnv, RuleMonitor, SimConfig
from aircraft_sim.policies import rule_agnostic_policy
from aircraft_sim.rules import rule_family
from run_short_recovery_experiments import (
    DISTANCES, METHODS, PAPER_PROTOCOL_OVERRIDES, scenario_at_distance, shield_for,
)


def target_violations(report):
    return [rule for rule in report.violations if rule_family(rule) in {"S02", "S03"}]


def pair_diagnostics(truth, config):
    """Use exactly the velocity/TTC convention of the existing truth monitor.

    Pending unsafe trends are recorded separately from expired S03 deadlines;
    a still-unsafe trend inside its grace period is not a restored safe trend.
    """
    pairs = []
    aircraft = truth["aircraft"]
    for i, first in enumerate(aircraft):
        for second in aircraft[i + 1:]:
            rel = [second.position[k] - first.position[k] for k in range(3)]
            velocity = [
                second.speed * math.cos(second.heading) - first.speed * math.cos(first.heading),
                second.speed * math.sin(second.heading) - first.speed * math.sin(first.heading),
                0.0,
            ]
            dot = sum(p * v for p, v in zip(rel, velocity))
            norm_sq = sum(v * v for v in velocity)
            ttc = -dot / norm_sq if dot < 0 and norm_sq > 1e-9 else math.inf
            closest = math.sqrt(sum((p + v * ttc) ** 2 for p, v in zip(rel, velocity))) if math.isfinite(ttc) else math.inf
            pairs.append({
                "agents": [first.uid, second.uid],
                "distance_m": math.sqrt(sum(p * p for p in rel)),
                "ttc_s": ttc if math.isfinite(ttc) else None,
                "predicted_closest_distance_m": closest if math.isfinite(closest) else None,
                "unsafe_trend": ttc < config.belief_ttc_threshold and closest < config.min_separation,
            })
    return pairs


def state_record(env, monitor, report, actions=None):
    truth = env.truth_state()
    pairs = pair_diagnostics(truth, env.cfg)
    violations = target_violations(report)
    return {
        "absolute_state_step": env.step_count,
        "truth": {**truth, "aircraft": [asdict(a) for a in truth["aircraft"]]},
        "truth_violations": list(report.violations),
        "target_violations": violations,
        "s03_deadlines": dict(monitor._safe_trend_deadlines),
        "pairs": pairs,
        "minimum_separation_m": min(p["distance_m"] for p in pairs),
        "rule_clear": not violations,
        "restored_safe_state": not violations and not any(p["unsafe_trend"] for p in pairs),
        "actions": {str(uid): asdict(a) for uid, a in (actions or {}).items()},
    }


def recovery_metrics(states, stable_steps=2, short_window=3):
    """states[k-1] is the state AFTER recovery action k (one-based).

    Report first entry, confirmed entry, and subsequent recurrence separately.
    The trigger state k=0 is never counted as an executed recovery step.
    """
    flags = [s["restored_safe_state"] for s in states]
    first = next((k for k, safe in enumerate(flags, 1) if safe), None)
    confirmed = next((k for k in range(stable_steps, len(flags) + 1)
                      if all(flags[k - stable_steps:k])), None)
    later_failures = [k for k in range((first or len(flags)) + 1, len(flags) + 1) if not flags[k - 1]]
    violation_steps = [k for k, state in enumerate(states, 1) if state["target_violations"]]
    return {
        "first_recovery_step": first,
        "confirmed_recovery_step": confirmed,
        "recovered_within_3_steps": float(first is not None and first <= short_window),
        "confirmed_within_3_steps": float(confirmed is not None and confirmed <= short_window),
        "recovered_by_3_and_held_to_30": float(first is not None and first <= short_window and not later_failures),
        "recovered_by_30": float(first is not None),
        "unsafe_state_steps_after_first_recovery": later_failures,
        "target_violation_steps_after_takeover": violation_steps,
        "target_violation_state_rate": len(violation_steps) / len(states),
        "rule_clear_within_3_steps": float(any(s["rule_clear"] for s in states[:short_window])),
    }


def make_trigger(config, scenario, max_steps=12, trigger_rule="S02"):
    env = AircraftEnv(3, config)
    monitor = RuleMonitor(config)
    observations = env.reset(scenario)
    initial_report = monitor.evaluate_truth(env.truth_state())
    initial = state_record(env, monitor, initial_report)
    if not initial_report.hard_safe:
        return None, {"status": "initial_hard_unsafe", "initial": initial, "prefix": []}
    prefix = []
    history = []
    for _ in range(max_steps):
        history.append(deepcopy(observations))
        nominal = {uid: rule_agnostic_policy(obs, uid) for uid, obs in observations.items()}
        result = env.step(nominal)
        report = monitor.evaluate_truth(result.info["truth"], nominal)
        record = state_record(env, monitor, report, nominal)
        prefix.append(record)
        observations = result.observations
        observed = {rule_family(r) for r in record["target_violations"]}
        triggered = bool(observed) if trigger_rule == "either" else trigger_rule in observed
        if triggered:
            # Clone both simulator RNG and temporal-monitor history; never
            # reset at takeover and thereby erase a pending S03 obligation.
            return (env, monitor, observations, history), {
                "status": "triggered", "initial": initial, "prefix": prefix,
                "trigger": record,
            }
    return None, {"status": "trigger_not_reached", "initial": initial, "prefix": prefix}


def snapshot_hash(snapshot):
    env, monitor, observations, _ = snapshot
    payload = {
        "truth": {**env.truth_state(), "aircraft": [asdict(a) for a in env.aircraft]},
        "rng_state": env.rng.getstate(), "observations": observations,
        "s03_deadlines": monitor._safe_trend_deadlines,
        "last_truth_step": monitor._last_truth_step,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def rollout_from_trigger(snapshot, method, horizon, stable_steps, trace_sink=None, identity=None):
    env, monitor, observations, history = deepcopy(snapshot)
    shield = shield_for(method, env.cfg)
    # DG-QP gets only the common pre-trigger observation stream, never truth.
    if method == "DG-QP":
        for observation in history:
            shield.monitor.update(observation)
    states = []
    interventions = fallback = 0
    takeover_hash = snapshot_hash((env, monitor, observations, history))
    for recovery_step in range(1, horizon + 1):
        nominal = {uid: rule_agnostic_policy(obs, uid) for uid, obs in observations.items()}
        actions, events = (nominal, []) if shield is None else shield.filter(observations, nominal)
        solution = getattr(shield, "last_solution", None)
        status = getattr(solution, "status", None)
        fallback += int(status == "infeasible_fallback")
        interventions += len(events)
        before = observations
        result = env.step(actions)
        report = monitor.evaluate_truth(result.info["truth"], actions)
        state = state_record(env, monitor, report, actions)
        state.update({
            "recovery_step": recovery_step,
            "nominal_actions": {str(uid): asdict(a) for uid, a in nominal.items()},
            "observations_before_action": before,
            "interventions": [asdict(event) for event in events],
            "qp_status": status,
            "qp_infeasibility_reason": getattr(solution, "infeasibility_reason", None),
            "relaxed_constraints": getattr(solution, "relaxed_constraints", {}),
            "conflict_events": getattr(solution, "conflict_events", []),
            "reward": result.reward,
        })
        states.append(state)
        if trace_sink is not None:
            trace_sink.write(json.dumps({**(identity or {}), "method": method, **state}, ensure_ascii=False, allow_nan=False) + "\n")
        observations = result.observations
    metrics = recovery_metrics(states, stable_steps)
    metrics.update({
        "takeover_snapshot_sha256": takeover_hash,
        "minimum_separation_after_takeover_m": min(s["minimum_separation_m"] for s in states),
        "minimum_separation_in_first_3_steps_m": min(s["minimum_separation_m"] for s in states[:3]),
        "intervention_rate": interventions / horizon,
        "qp_fallback_rate": fallback / horizon,
        "cumulative_reward_after_takeover": sum(s["reward"] for s in states),
    })
    return metrics, states


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    parser.add_argument("--distances", nargs="+", type=float, default=DISTANCES)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--position-jitter", type=float, default=0.0, help="Uniform horizontal perturbation in metres, fixed before evaluating methods")
    parser.add_argument("--trigger-rule", choices=["S02", "S03", "either"], default="S02")
    parser.add_argument("--max-trigger-steps", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=30, choices=[30])
    parser.add_argument("--stable-steps", type=int, default=2, choices=[2, 3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_trigger_steps < 1 or args.position_jitter < 0 or any(d <= 0 for d in args.distances):
        parser.error("episodes/max-trigger-steps/distances must be positive; position-jitter must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.distances)) != len(args.distances):
        parser.error("duplicate seeds/distances would duplicate statistical samples")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is nonempty; use a new directory to preserve previous results")
    args.output.mkdir(parents=True, exist_ok=True)
    rows, triggers, summary = [], [], []
    with (args.output / "step_trace.jsonl").open("w", encoding="utf-8") as trace:
        for seed in args.seeds:
            for distance in args.distances:
                for episode in range(args.episodes):
                    identity = {"seed": seed, "distance": distance, "episode": episode}
                    scenario = scenario_at_distance(distance)
                    rng = random.Random(f"recovery:{seed}:{distance}:{episode}")
                    scenario["positions"] = [(x + rng.uniform(-args.position_jitter, args.position_jitter),
                                              y + rng.uniform(-args.position_jitter, args.position_jitter), z)
                                             for x, y, z in scenario["positions"]]
                    config = SimConfig(seed=rng.randrange(2**31) if args.episodes > 1 or args.position_jitter else seed,
                                       horizon=args.horizon + args.max_trigger_steps, **PAPER_PROTOCOL_OVERRIDES)
                    snapshot, info = make_trigger(config, scenario, args.max_trigger_steps, args.trigger_rule)
                    info.update({**identity, "scenario": scenario, "environment_seed": config.seed})
                    if snapshot is None:
                        triggers.append(info)
                        continue
                    expected_hash = snapshot_hash(snapshot)
                    info["takeover_snapshot_sha256"] = expected_hash
                    triggers.append(info)
                    for method in METHODS:
                        metrics, _ = rollout_from_trigger(snapshot, method, args.horizon, args.stable_steps, trace, identity)
                        assert metrics["takeover_snapshot_sha256"] == expected_hash, "paired takeover mismatch"
                        rows.append({**identity, "method": method,
                                     "initial_hard_safe": True, "triggered_violation": True,
                                     "trigger_state_step": snapshot[0].step_count,
                                     "trigger_target_violations": info["trigger"]["target_violations"],
                                     "trigger_minimum_separation_m": info["trigger"]["minimum_separation_m"], **metrics})
            print(json.dumps({"seed_complete": seed, "records_so_far": len(rows)}), flush=True)
    keys = ["recovered_within_3_steps", "confirmed_within_3_steps", "recovered_by_3_and_held_to_30",
            "recovered_by_30", "rule_clear_within_3_steps", "target_violation_state_rate",
            "minimum_separation_after_takeover_m", "intervention_rate", "qp_fallback_rate"]
    for distance in args.distances:
        for method in METHODS:
            selected = [r for r in rows if r["distance"] == distance and r["method"] == method]
            entry = {"distance": distance, "method": method, "n_triggered_episodes": len(selected)}
            for key in keys:
                values = [r[key] for r in selected]
                seed_means = [statistics.mean([r[key] for r in selected if r["seed"] == seed])
                              for seed in args.seeds if any(r["seed"] == seed for r in selected)]
                entry[key] = {"mean": statistics.mean(values) if values else None,
                              "seed_mean_sd": statistics.stdev(seed_means) if len(seed_means) > 1 else None,
                              "n_seeds": len(seed_means)}
            times = [r["first_recovery_step"] for r in selected if r["first_recovery_step"] is not None]
            entry["mean_first_recovery_step_among_recovered"] = statistics.mean(times) if times else None
            entry["n_recovered_by_30"] = len(times)
            summary.append(entry)
    code_files = [Path(__file__), Path("run_short_recovery_experiments.py"), *sorted(Path("aircraft_sim").glob("*.py"))]
    metadata = {
        "protocol": "shared_actual_violation_then_paired_takeover_v1",
        "seeds": args.seeds, "distances": args.distances, "episodes_per_seed_distance": args.episodes,
        "position_jitter_m": args.position_jitter, "trigger_rule": args.trigger_rule,
        "max_trigger_steps": args.max_trigger_steps, "horizon_after_takeover": args.horizon,
        "stable_steps": args.stable_steps, "target_rules": ["S02", "S03"],
        "protocol_overrides": PAPER_PROTOCOL_OVERRIDES,
        "endpoint_definition": "After a common actual target-rule violation (k=0), return to a state with no S02/S03 violation AND no unsafe S03 trend within k=1,2,3. Confirmation and recurrence are reported separately.",
        "thirty_step_definition": "Enter the restored safe set within 3 actions and remain in it at every subsequent executed state through action 30.",
        "trigger_policy": "unshielded rule_agnostic_policy; identical to the old nominal policy",
        "inclusion": "initially hard-safe and shared trigger reached; no feasibility/success filtering after trigger",
        "pairing": "deepcopy common environment, RNG, observations, and truth-monitor deadlines; DG belief warmed with common observations only",
        "interpretation": "Shield ablation with a fixed nominal policy, not retrained-policy comparison. Post-violation recovery does not erase the triggering safety breach; minimum separation must also be reported. Exact zero-jitter geometry is not 10 independently sampled geometries.",
        "requested_cases": len(triggers), "triggered_cases": sum(t["status"] == "triggered" for t in triggers),
        "excluded_initial_unsafe": sum(t["status"] == "initial_hard_unsafe" for t in triggers),
        "trigger_not_reached": sum(t["status"] == "trigger_not_reached" for t in triggers),
        "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
    }
    save_json(args.output / "triggers.json", triggers)
    save_json(args.output / "per_seed.json", rows)
    save_json(args.output / "summary.json", {"metadata": metadata, "summary": summary})
    lines = ["# Actual post-violation recovery (paired takeover)", "",
             "k=0 is an observed violation. k=1..3 are three executed actions after takeover. All methods clone the same trigger state.", "",
             "| Initial distance (m) | Method | Triggered episodes | Recovered by 3 | Confirmed by 3 | Recovered by 3, held to 30 | Min separation after takeover (m) | QP fallback |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        def fmt(key):
            value = row[key]["mean"]
            return f"{value:.3f}" if value is not None else "NA"
        lines.append(f"| {row['distance']:.0f} | {row['method']} | {row['n_triggered_episodes']} | {fmt('recovered_within_3_steps')} | {fmt('confirmed_within_3_steps')} | {fmt('recovered_by_3_and_held_to_30')} | {fmt('minimum_separation_after_takeover_m')} | {fmt('qp_fallback_rate')} |")
    lines.extend(["", "A failed trigger is excluded from all methods together and remains in triggers.json. No post-trigger feasibility filtering is applied.",
                  "Recovery can result from aircraft passing each other after a deep separation breach. Consult the minimum separation and full trace; this endpoint alone does not demonstrate collision avoidance.",
                  "Rows are shield configurations over the same fixed nominal policy, not seven learned algorithms."])
    (args.output / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(rows), "triggered_cases": metadata["triggered_cases"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
