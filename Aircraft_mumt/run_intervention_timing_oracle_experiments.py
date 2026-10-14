"""Frozen paired intervention-timing and truth-informed recovery diagnostics.

No simulator, learned policy, rule threshold, or shield implementation changes.
The warning clock is an offline common truth-risk clock, not an operational
claim that all deployed policies can observe truth or share the same belief.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import statistics

import numpy as np
from scipy.optimize import minimize

from aircraft_sim import Action, AircraftEnv, Mode, RuleMonitor, SimConfig
from aircraft_sim.policies import rule_agnostic_policy
from aircraft_sim.rules import rule_family
from run_post_violation_recovery_experiments import (
    pair_diagnostics, recovery_metrics, rollout_from_trigger, save_json,
    snapshot_hash, state_record,
)
from run_short_recovery_experiments import DISTANCES, METHODS, PAPER_PROTOCOL_OVERRIDES, scenario_at_distance


TIMINGS = ("warning_now", "warning_plus_1", "warning_plus_2", "first_S02_violation")


def common_prefix(config, scenario, max_steps=12):
    env, monitor = AircraftEnv(3, config), RuleMonitor(config)
    observations = env.reset(scenario)
    report = monitor.evaluate_truth(env.truth_state())
    initial = state_record(env, monitor, report)
    if not report.hard_safe:
        return None, {"status": "initial_hard_unsafe", "initial": initial}
    history, records, snapshots = [], [initial], {}
    warning_step = s02_step = None
    for step in range(max_steps + 1):
        record = records[-1]
        if warning_step is None and any(p["unsafe_trend"] for p in record["pairs"]):
            warning_step = step
        if s02_step is None and any(rule_family(v) == "S02" for v in record["target_violations"]):
            s02_step = step
        snapshots[step] = deepcopy((env, monitor, observations, history))
        if warning_step is not None and s02_step is not None and step >= warning_step + 2:
            selections = {
                "warning_now": warning_step,
                "warning_plus_1": warning_step + 1,
                "warning_plus_2": warning_step + 2,
                "first_S02_violation": s02_step,
            }
            return {key: snapshots[t] for key, t in selections.items()}, {
                "status": "ready", "warning_step": warning_step, "s02_step": s02_step,
                "initial": initial, "prefix": records,
                "takeovers": {key: {"state": records[t], "snapshot_sha256": snapshot_hash(snapshots[t])}
                              for key, t in selections.items()},
            }
        if step == max_steps:
            break
        history.append(deepcopy(observations))
        nominal = {uid: rule_agnostic_policy(obs, uid) for uid, obs in observations.items()}
        result = env.step(nominal)
        observations = result.observations
        report = monitor.evaluate_truth(result.info["truth"], nominal)
        record = state_record(env, monitor, report, nominal)
        record["reward"] = result.reward
        records.append(record)
    return None, {"status": "warning_or_S02_not_reached", "initial": initial, "prefix": records}


def timing_metrics(states, warning_record):
    n = len(states)
    target_steps = [k for k, s in enumerate(states, 1) if s["target_violations"]]
    s02_steps = [k for k, s in enumerate(states, 1) if any(rule_family(r) == "S02" for r in s["target_violations"])]
    restored = recovery_metrics(states)
    # This is risk resolution, not post-violation recovery, when takeover is
    # still S02 safe. Preserve both names explicitly in stored results.
    return {
        "target_safety_retention_3step": float(not any(k <= 3 for k in target_steps)),
        "target_safety_retention_30step": float(not target_steps),
        "first_target_violation_step_from_warning": target_steps[0] if target_steps else None,
        "first_S02_violation_step_from_warning": s02_steps[0] if s02_steps else None,
        "target_violation_steps_from_warning": target_steps,
        "target_violation_state_rate": len(target_steps) / n,
        "S02_violation_state_rate": len(s02_steps) / n,
        "minimum_separation_from_warning_m": min([warning_record["minimum_separation_m"]] + [s["minimum_separation_m"] for s in states]),
        "risk_resolved_within_3_steps_from_warning": restored["recovered_within_3_steps"],
        "first_risk_resolution_step_from_warning": restored["first_recovery_step"],
        "risk_resolved_by_3_and_held_to_30": restored["recovered_by_3_and_held_to_30"],
        "C03_violation_state_rate": sum(any(rule_family(v) == "C03" for v in s["truth_violations"]) for s in states) / n,
        "C02_violation_state_rate": sum("C02" in s["truth_violations"] for s in states) / n,
        "support_service_state_rate": sum(any(a["mode"] == "support" and a["target"] == 0 for a in s["actions"].values()) for s in states) / n,
        "intervention_rate_from_warning": sum(len(s.get("interventions", [])) for s in states) / n,
        "qp_fallback_rate_from_warning": sum(s.get("qp_status") == "infeasible_fallback" for s in states) / n,
        "cumulative_reward_from_warning": sum(s["reward"] for s in states),
        "mean_task_progress_gain": states[-1]["truth"]["task_progress"] - warning_record["truth"]["task_progress"],
    }


def predict_controls(env, controls):
    """Deterministic physical projection; verified against AircraftEnv.step.

    The recovery reference assigns RECOVER to all aircraft. Position, speed,
    heading, climb angle and energy are the unchanged simulator equations.
    """
    config = env.cfg
    position = np.asarray([a.position for a in env.aircraft], dtype=float)
    speed = np.asarray([a.speed for a in env.aircraft], dtype=float)
    heading = np.asarray([a.heading for a in env.aircraft], dtype=float)
    climb = np.asarray([a.climb_angle for a in env.aircraft], dtype=float)
    energy = np.asarray([a.energy for a in env.aircraft], dtype=float)
    positions, speeds, energies, headings = [], [], [], []
    for action in np.clip(np.asarray(controls), -1, 1):
        heading = heading + action[:, 0] * 0.22 * config.dt
        climb = np.clip(climb + action[:, 1] * 0.08 * config.dt, -0.45, 0.45)
        speed = np.clip(speed + action[:, 2] * 12 * config.dt, config.min_speed * 0.7, config.max_speed)
        position = position + config.dt * speed[:, None] * np.column_stack((np.cos(climb) * np.cos(heading),
                         np.cos(climb) * np.sin(heading), np.sin(climb)))
        position[:, 2] = np.maximum(0, position[:, 2])
        energy = np.clip(energy - 0.012 * (np.abs(action[:, 0]) + np.abs(action[:, 1]))
                         - 0.002 * np.abs(action[:, 2]) + 0.003, 0, 1)
        positions.append(position.copy()); speeds.append(speed.copy())
        energies.append(energy.copy()); headings.append(heading.copy())
    positions, speeds, energies, headings = map(np.asarray, (positions, speeds, energies, headings))
    pair_indices = list(itertools.combinations(range(env.n), 2))
    distances = np.column_stack([np.linalg.norm(positions[:, j] - positions[:, i], axis=1) for i, j in pair_indices])
    trend_margins = []
    for i, j in pair_indices:
        rel = positions[-1, j] - positions[-1, i]
        vel = np.array([speeds[-1, j] * math.cos(headings[-1, j]) - speeds[-1, i] * math.cos(headings[-1, i]),
                        speeds[-1, j] * math.sin(headings[-1, j]) - speeds[-1, i] * math.sin(headings[-1, i]), 0.0])
        dot = np.dot(rel, vel); norm_sq = np.dot(vel, vel)
        if dot >= 0 or norm_sq <= 1e-9:
            trend_margins.append(1.0)
        else:
            ttc = -dot / norm_sq
            closest = np.linalg.norm(rel + vel * ttc)
            trend_margins.append(max((ttc - config.belief_ttc_threshold) / config.belief_ttc_threshold,
                                     (closest - config.min_separation) / config.min_separation))
    physical = np.concatenate(((speeds - config.min_speed).ravel() / 100,
        (config.max_speed - speeds).ravel() / 100,
        (positions[:, :, 2] - config.min_altitude).ravel() / 1000,
        (config.max_altitude - positions[:, :, 2]).ravel() / 1000,
        (energies - config.min_energy).ravel()))
    endpoint = np.concatenate((distances[-1] / config.min_separation - 1, trend_margins))
    return {"positions": positions, "speeds": speeds, "headings": headings,
            "energies": energies, "distances": distances, "physical_margins": physical,
            "endpoint_margins": endpoint}


def evaluate_reference(snapshot, controls):
    env, monitor, _, _ = deepcopy(snapshot)
    states = []
    for k, row in enumerate(controls, 1):
        actions = {uid: Action(Mode.RECOVER, turn=float(u[0]), climb=float(u[1]), acceleration=float(u[2]))
                   for uid, u in enumerate(row)}
        result = env.step(actions)
        report = monitor.evaluate_truth(result.info["truth"], actions)
        state = state_record(env, monitor, report, actions)
        state.update(recovery_step=k, reward=result.reward)
        states.append(state)
    physical_valid = all(not any(rule_family(v) in {"S01", "S04"} for v in s["truth_violations"]) for s in states)
    metrics = recovery_metrics(states)
    # Only three actions were executed: do not expose fields whose names
    # suggest a thirty-step observation that this reference did not perform.
    metrics.pop("recovered_by_3_and_held_to_30")
    metrics.pop("recovered_by_30")
    metrics.update({"physical_envelope_energy_valid": physical_valid,
                    "reference_executed_steps": len(states),
                    "minimum_separation_first_3_steps_m": min(s["minimum_separation_m"] for s in states)})
    return metrics, states


def constant_candidates(n=3):
    # All combinations are fixed before seeing outcomes. Opposite vertical
    # commands and both possible turn conventions are included explicitly.
    for turn0, turn1, climb, acceleration in itertools.product((-1., 0., 1.), repeat=4):
        row = np.zeros((n, 3))
        row[0] = (turn0, climb, acceleration)
        row[1] = (turn1, -climb, acceleration)
        yield np.repeat(row[None, :, :], 3, axis=0)
    nominal = np.zeros((3, n, 3)); nominal[:, :, 0] = 0.1; nominal[:, :, 2] = 0.2
    yield nominal


def trig_range(lo, hi, func):
    critical = [lo, hi]
    offset = math.pi / 2 if func == "sin" else 0.0
    for k in range(math.ceil((lo - offset) / math.pi), math.floor((hi - offset) / math.pi) + 1):
        critical.append(offset + k * math.pi)
    values = [(math.sin(x) if func == "sin" else math.cos(x)) for x in critical]
    return min(values), max(values)


def interval_product(*ranges):
    products = [math.prod(x) for x in itertools.product(*[(lo, hi) for lo, hi in ranges])]
    return min(products), max(products)


def one_step_pair_distance_upper_bound(env):
    """Outer-box bound ignoring component correlations; never a 3-step bound."""
    boxes = []
    for a in env.aircraft[:2]:
        dt = env.cfg.dt
        speed = (max(env.cfg.min_speed * .7, min(env.cfg.max_speed, a.speed - 12 * dt)),
                 max(env.cfg.min_speed * .7, min(env.cfg.max_speed, a.speed + 12 * dt)))
        yaw = (a.heading - .22 * dt, a.heading + .22 * dt)
        climb = (max(-.45, min(.45, a.climb_angle - .08 * dt)), max(-.45, min(.45, a.climb_angle + .08 * dt)))
        increments = [interval_product(speed, trig_range(*climb, "cos"), trig_range(*yaw, "cos")),
                      interval_product(speed, trig_range(*climb, "cos"), trig_range(*yaw, "sin")),
                      interval_product(speed, trig_range(*climb, "sin"))]
        box = [(a.position[k] + dt * lo, a.position[k] + dt * hi) for k, (lo, hi) in enumerate(increments)]
        box[2] = (max(0, box[2][0]), max(0, box[2][1]))
        boxes.append(box)
    upper = math.sqrt(sum(max(abs(boxes[1][k][0] - boxes[0][k][1]),
                             abs(boxes[1][k][1] - boxes[0][k][0])) ** 2 for k in range(3)))
    return upper


def truth_recovery_reference(snapshot, starts=3, maxiter=80):
    env = snapshot[0]
    candidates, feasible = [], []
    for controls in constant_candidates(env.n):
        pred = predict_controls(env, controls)
        is_feasible = min(pred["physical_margins"]) >= -1e-9 and min(pred["endpoint_margins"]) >= -1e-9
        candidate = {"controls": controls, "minimum": float(np.min(pred["distances"])), "feasible": is_feasible}
        candidates.append(candidate)
        if is_feasible:
            feasible.append(candidate)
    pool = sorted(feasible if feasible else candidates, key=lambda c: c["minimum"], reverse=True)
    grid_best = pool[0]
    grid_metrics, grid_states = evaluate_reference(snapshot, grid_best["controls"])
    best = grid_best
    attempts = []
    # Smooth epigraph: maximize the smallest sampled separation over all 3
    # steps, subject to a restored safe terminal state and physical envelope.
    for candidate in pool[:starts]:
        dimension = 3 * env.n * 3
        x0 = np.r_[candidate["controls"].ravel(), candidate["minimum"] / env.cfg.min_separation]
        def projection(x):
            return predict_controls(env, x[:dimension].reshape(3, env.n, 3))
        def constraints(x):
            p = projection(x)
            return np.r_[p["physical_margins"], p["endpoint_margins"],
                         p["distances"].ravel() / env.cfg.min_separation - x[-1]]
        result = minimize(lambda x: -x[-1], x0, method="SLSQP",
                          bounds=[(-1, 1)] * dimension + [(0, None)],
                          constraints=[{"type": "ineq", "fun": constraints}],
                          options={"maxiter": maxiter, "ftol": 1e-8})
        controls = np.clip(result.x[:dimension].reshape(3, env.n, 3), -1, 1)
        metrics, _ = evaluate_reference(snapshot, controls)
        valid = metrics["physical_envelope_energy_valid"] and metrics["recovered_within_3_steps"] == 1
        attempts.append({"optimizer_success": bool(result.success), "message": str(result.message),
                         "iterations": int(result.nit), "validated_recovery_witness": valid,
                         "minimum_separation_m": metrics["minimum_separation_first_3_steps_m"]})
        if valid and (not best["feasible"] or metrics["minimum_separation_first_3_steps_m"] > best["minimum"]):
            best = {"controls": controls, "minimum": metrics["minimum_separation_first_3_steps_m"], "feasible": True}
    metrics, states = evaluate_reference(snapshot, best["controls"])
    return {"grid_candidate_count": len(candidates), "grid_terminal_feasible_count": len(feasible),
            "grid_best_controls": grid_best["controls"].tolist(), "grid_best_metrics": grid_metrics,
            "grid_best_states": grid_states, "optimized_controls": best["controls"].tolist(),
            "reference_metrics": metrics, "reference_states": states, "optimization_attempts": attempts,
            "one_step_pair_distance_outer_bound_m": one_step_pair_distance_upper_bound(env),
            "interpretation": "Validated truth-informed feasible reference, not a certified global optimum. Failure to find a witness does not establish physical impossibility."}


def aggregate(rows, group_keys, metric_keys):
    groups = sorted({tuple(row[k] for k in group_keys) for row in rows})
    summary = []
    for group in groups:
        selected = [row for row in rows if tuple(row[k] for k in group_keys) == group]
        entry = dict(zip(group_keys, group)); entry["n_episodes"] = len(selected)
        for metric in metric_keys:
            seed_means = [statistics.mean([r[metric] for r in selected if r["seed"] == seed])
                          for seed in sorted({r["seed"] for r in selected})]
            entry[metric] = {"mean": statistics.mean(seed_means), "seed_sd": statistics.stdev(seed_means) if len(seed_means) > 1 else None,
                             "n_seeds": len(seed_means)}
        summary.append(entry)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11,22,33,44,55,66,77,88,99,111])
    parser.add_argument("--distances", nargs="+", type=float, default=DISTANCES)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--position-jitter", type=float, default=0)
    parser.add_argument("--oracle-starts", type=int, default=3)
    parser.add_argument("--oracle-maxiter", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.position_jitter < 0 or args.oracle_starts < 1 or args.oracle_maxiter < 1 or any(d <= 0 for d in args.distances):
        parser.error("invalid nonpositive experiment setting")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.distances)) != len(args.distances):
        parser.error("duplicate seeds/distances")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is nonempty; preserve existing results")
    args.output.mkdir(parents=True, exist_ok=True)
    timing_rows, oracle_rows, cases, oracle_cache = [], [], [], {}
    with gzip.open(args.output / "timing_step_trace.jsonl.gz", "wt", encoding="utf-8") as trace:
        for seed in args.seeds:
            for distance in args.distances:
                for episode in range(args.episodes):
                    identity = {"seed": seed, "distance": distance, "episode": episode}
                    rng = random.Random(f"timing:{seed}:{distance}:{episode}")
                    scenario = scenario_at_distance(distance)
                    scenario["positions"] = [(x+rng.uniform(-args.position_jitter,args.position_jitter),
                                              y+rng.uniform(-args.position_jitter,args.position_jitter),z)
                                             for x,y,z in scenario["positions"]]
                    env_seed = rng.randrange(2**31) if args.position_jitter or args.episodes > 1 else seed
                    config = SimConfig(seed=env_seed, horizon=45, **PAPER_PROTOCOL_OVERRIDES)
                    snapshots, case = common_prefix(config, scenario)
                    case.update({**identity, "scenario": scenario, "environment_seed": env_seed})
                    cases.append(case)
                    if snapshots is None:
                        continue
                    for timing in TIMINGS:
                        snapshot = snapshots[timing]
                        delay = snapshot[0].step_count - case["warning_step"]
                        assert 0 <= delay < 30
                        for method in METHODS:
                            recovery, branch = rollout_from_trigger(snapshot, method, 30-delay, 2)
                            assert recovery["takeover_snapshot_sha256"] == case["takeovers"][timing]["snapshot_sha256"]
                            prefix = deepcopy(case["prefix"][case["warning_step"]+1:snapshot[0].step_count+1])
                            combined = prefix + branch
                            assert len(combined) == 30
                            metrics = timing_metrics(combined, case["prefix"][case["warning_step"]])
                            timing_rows.append({**identity, "timing": timing, "method": method,
                                "takeover_step": snapshot[0].step_count, "delay_from_warning_steps": delay,
                                "warning_step": case["warning_step"],
                                "takeover_snapshot_sha256": recovery["takeover_snapshot_sha256"], **metrics})
                            for k,state in enumerate(combined,1):
                                trace.write(json.dumps({**identity,"timing":timing,"method":method,
                                    "step_from_warning":k,"phase":"common_unshielded_prefix" if k<=delay else "after_takeover", **state},
                                    ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
                    snapshot = snapshots["first_S02_violation"]
                    # Cache only identical physical states, not policy results;
                    # each cached control sequence is revalidated in that seed.
                    physical = [[a.position,a.speed,a.heading,a.climb_angle,a.energy] for a in snapshot[0].aircraft]
                    cache_key = hashlib.sha256(json.dumps(physical).encode()).hexdigest()
                    if cache_key not in oracle_cache:
                        oracle_cache[cache_key] = truth_recovery_reference(snapshot,args.oracle_starts,args.oracle_maxiter)
                    reference = deepcopy(oracle_cache[cache_key])
                    metrics,states = evaluate_reference(snapshot,reference["optimized_controls"])
                    reference["reference_metrics"],reference["reference_states"] = metrics,states
                    grid_metrics,grid_states = evaluate_reference(snapshot,reference["grid_best_controls"])
                    reference["grid_best_metrics"],reference["grid_best_states"] = grid_metrics,grid_states
                    oracle_rows.append({**identity,"takeover_snapshot_sha256":snapshot_hash(snapshot),
                        "trigger_step":snapshot[0].step_count,"trigger_minimum_separation_m":case["takeovers"]["first_S02_violation"]["state"]["minimum_separation_m"],
                        "physical_cache_key":cache_key,"reference_recovered_by_3":metrics["recovered_within_3_steps"] if metrics["physical_envelope_energy_valid"] else 0.,
                        "grid_recovered_by_3":grid_metrics["recovered_within_3_steps"] if grid_metrics["physical_envelope_energy_valid"] else 0.,
                        "reference_minimum_separation_m":metrics["minimum_separation_first_3_steps_m"],
                        "one_step_pair_distance_upper_bound_m":reference["one_step_pair_distance_outer_bound_m"],
                        **reference})
            print(json.dumps({"seed_complete":seed,"timing_records":len(timing_rows),"oracle_records":len(oracle_rows),"unique_oracle_states":len(oracle_cache)}),flush=True)
    metric_keys = ["target_safety_retention_3step","target_safety_retention_30step","minimum_separation_from_warning_m",
        "target_violation_state_rate","risk_resolved_within_3_steps_from_warning","risk_resolved_by_3_and_held_to_30",
        "C03_violation_state_rate","C02_violation_state_rate","support_service_state_rate",
        "cumulative_reward_from_warning","mean_task_progress_gain","qp_fallback_rate_from_warning"]
    timing_summary = aggregate(timing_rows,["distance","timing","method"],metric_keys)
    oracle_summary = aggregate(oracle_rows,["distance"],["reference_recovered_by_3","grid_recovered_by_3",
        "reference_minimum_separation_m","one_step_pair_distance_upper_bound_m"])
    source_files = [Path(__file__),Path("run_post_violation_recovery_experiments.py"),Path("run_short_recovery_experiments.py"),
                    *sorted(Path("aircraft_sim").glob("*.py"))]
    for source in source_files:
        destination = args.output / "source_snapshot" / source.name if source.parent.name != "aircraft_sim" else args.output / "source_snapshot" / "aircraft_sim" / source.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    metadata = {"protocol":"warning_clock_timing_plus_truth_recovery_reference_v1",
        "seeds":args.seeds,"distances":args.distances,"episodes_per_seed_distance":args.episodes,"position_jitter_m":args.position_jitter,
        "horizon_from_warning":30,"timings":TIMINGS,"target_rules":["S02","S03"],"protocol_overrides":PAPER_PROTOCOL_OVERRIDES,
        "warning_definition":"First unsafe TTC/closest-distance trend on a shared unshielded prefix, assessed offline in truth. Not expired S03 and not an operational online trigger.",
        "comparison_clock":"Every branch ends 30 executed steps after the same warning; pre-takeover violations/rewards are included. No reset of S03 history.",
        "nominal_policy":"fixed rule_agnostic_policy; shield ablation, not learned-policy evaluation",
        "reference_definition":"82 fixed 3-step sequences plus multi-start full-truth SLSQP max-min sampled separation with physically valid state and restored safety at step 3; validated in actual environment, not certified global upper bound.",
        "oracle_starts":args.oracle_starts,"oracle_maxiter":args.oracle_maxiter,"unique_oracle_physical_states":len(oracle_cache),
        "requested_cases":len(cases),"ready_cases":sum(c["status"]=="ready" for c in cases),
        "excluded_cases":sum(c["status"]!="ready" for c in cases),
        "code_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    save_json(args.output/"cases.json",cases)
    save_json(args.output/"timing_per_seed.json",timing_rows)
    save_json(args.output/"oracle_per_seed.json",oracle_rows)
    save_json(args.output/"summary.json",{"metadata":metadata,"timing_summary":timing_summary,"oracle_summary":oracle_summary})
    lines=["# Intervention timing (fixed nominal policy)","","All endpoints use 30 steps from the same offline warning, including the unshielded delay.","",
        "| Distance | Timing | Method | N | Safe 3 | Safe 30 | Min separation m | C03 violation | Support service | Reward | QP fallback |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in timing_summary:
        values=[r[k]["mean"] for k in ["target_safety_retention_3step","target_safety_retention_30step","minimum_separation_from_warning_m",
            "C03_violation_state_rate","support_service_state_rate","cumulative_reward_from_warning","qp_fallback_rate_from_warning"]]
        lines.append(f"| {r['distance']:.0f} | {r['timing']} | {r['method']} | {r['n_episodes']} | "+" | ".join(f"{v:.3f}" for v in values)+" |")
    lines += ["","# Truth-informed recovery reference","",
        "| Distance | N | Grid recovered by 3 | Reference recovered by 3 | Reference min separation m | One-step outer distance bound m |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in oracle_summary:
        lines.append(f"| {r['distance']:.0f} | {r['n_episodes']} | "+" | ".join(f"{r[k]['mean']:.3f}" for k in
            ["grid_recovered_by_3","reference_recovered_by_3","reference_minimum_separation_m","one_step_pair_distance_upper_bound_m"])+" |")
    lines += ["","The reference is a validated feasible witness, not a globally optimal or deployable competitor. Recovery does not erase intermediate separation breaches.",
        "Means aggregate episodes within seed first. Zero-jitter seeds are not independent geometries; varied cases are a separate protocol."]
    (args.output/"paper_table.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"timing_records":len(timing_rows),"oracle_records":len(oracle_rows)},ensure_ascii=False))


if __name__=="__main__":
    main()
