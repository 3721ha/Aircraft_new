"""Audit frozen timing/reference logs and summarize independent seed units.

Descriptive fixed-geometry results and perturbed-geometry inference are kept
separate. This script does not modify experiment data or the simulator.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from scipy.stats import binomtest, t

from aircraft_sim import SimConfig
from aircraft_sim.rules import rule_family
from run_intervention_timing_oracle_experiments import (
    common_prefix, evaluate_reference, timing_metrics,
)
from run_post_violation_recovery_experiments import recovery_metrics, save_json


def ident(row):
    return tuple(row[k] for k in ("seed", "distance", "episode"))


def branch_key(row):
    return ident(row) + (row["timing"], row["method"])


def restore_scenario_agent_keys(scenario):
    # JSON converts integer agent keys to strings; env.reset indexes these
    # maps by integer uid. Restore only the scenario's uid-indexed maps.
    return {key: {int(uid): value for uid, value in field.items()} if isinstance(field, dict)
            else field for key, field in scenario.items()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_numbers(a, b, context):
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        require(abs(a - b) <= 1e-7, f"{context}: {a} != {b}")
    else:
        require(a == b, f"{context}: mismatch")


def means(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for group, selected in sorted(groups.items()):
        entry = dict(zip(keys, group))
        entry["n_episodes"] = len(selected)
        for metric in metrics:
            units = defaultdict(list)
            for row in selected:
                units[row["seed"]].append(row[metric])
            values = [statistics.mean(v) for v in units.values()]
            entry[metric] = {
                "mean": statistics.mean(values), "n_seeds": len(values),
                "seed_sd": statistics.stdev(values) if len(values) > 1 else None,
            }
        result.append(entry)
    return result


def audit(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    meta = summary["metadata"]
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    rows = json.loads((directory / "timing_per_seed.json").read_text(encoding="utf-8"))
    references = json.loads((directory / "oracle_per_seed.json").read_text(encoding="utf-8"))
    case_map = {ident(c): c for c in cases}
    require(len(case_map) == len(cases), "duplicate case")
    expected = len(meta["seeds"]) * len(meta["distances"]) * meta["episodes_per_seed_distance"]
    require(expected == len(cases) == meta["requested_cases"], "case count")
    ready = [c for c in cases if c["status"] == "ready"]
    require(len(rows) == len(ready) * 16, "timing row count")
    require(len(references) == len(ready), "reference row count")
    row_map = {branch_key(r): r for r in rows}
    require(len(row_map) == len(rows), "duplicate branch")
    branches = defaultdict(list)
    with gzip.open(directory / "timing_step_trace.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            state = json.loads(line)
            key = branch_key(state)
            require(key in row_map, "orphan trace")
            branches[key].append(state)
    require(len(branches) == len(rows), "missing trace")
    for key, states in branches.items():
        row = row_map[key]
        case = case_map[key[:3]]
        delay = row["delay_from_warning_steps"]
        require(len(states) == 30, f"branch length {key}")
        require(row["takeover_snapshot_sha256"] == case["takeovers"][row["timing"]]["snapshot_sha256"], "takeover mismatch")
        for k, state in enumerate(states, 1):
            require(state["step_from_warning"] == k, "trace ordering")
            require(state["absolute_state_step"] == case["warning_step"] + k, "absolute clock mismatch")
            require(state["phase"] == ("common_unshielded_prefix" if k <= delay else "after_takeover"), "phase mismatch")
            pairs = state["pairs"]
            require(state["restored_safe_state"] == (not state["target_violations"] and not any(p["unsafe_trend"] for p in pairs)), "restoration definition")
            equal_numbers(state["minimum_separation_m"], min(p["distance_m"] for p in pairs), "pair minimum")
        computed = timing_metrics(states, case["prefix"][case["warning_step"]])
        for metric, value in computed.items():
            equal_numbers(value, row[metric], f"{key}/{metric}")
        row["S02_safety_retention_30step"] = float(not any(any(rule_family(v) == "S02" for v in s["target_violations"]) for s in states))
        if row["timing"] == "first_S02_violation":
            after = [s for s in states if s["phase"] == "after_takeover"]
            metrics = recovery_metrics(after)
            row["post_S02_recovered_within_3_steps"] = metrics["recovered_within_3_steps"]
            row["post_S02_confirmed_within_3_steps"] = metrics["confirmed_within_3_steps"]
            row["post_S02_recovered_by_3_and_held_to_common_end"] = metrics["recovered_by_3_and_held_to_30"]
            row["post_S02_first_recovery_step"] = metrics["first_recovery_step"]
            row["post_S02_observed_steps"] = len(after)
    # An unshielded branch must be identical regardless of nominal takeover.
    for case in ready:
        identity = ident(case)
        baseline = branches[identity + ("warning_now", "NoShield")]
        for timing in meta["timings"]:
            other = branches[identity + (timing, "NoShield")]
            for a, b in zip(baseline, other):
                for field in ("truth", "truth_violations", "actions", "reward", "pairs"):
                    require(a[field] == b[field], f"NoShield clock/trajectory differs: {identity}/{timing}/{field}")
    # Rebuild every common prefix and replay grid and optimized controls in the
    # actual environment. Check saved witnesses, not optimizer success flags.
    for reference in references:
        case = case_map[ident(reference)]
        config = SimConfig(seed=case["environment_seed"], horizon=45, **meta["protocol_overrides"])
        snapshots, rebuilt = common_prefix(config, restore_scenario_agent_keys(case["scenario"]))
        require(snapshots is not None, "reference prefix not reproducible")
        for timing in meta["timings"]:
            require(rebuilt["takeovers"][timing]["snapshot_sha256"] == case["takeovers"][timing]["snapshot_sha256"], "common prefix hash changed")
        require(reference["takeover_snapshot_sha256"] == case["takeovers"]["first_S02_violation"]["snapshot_sha256"], "reference takeover mismatch")
        for name, controls_name, metrics_name, states_name in (
            ("optimized", "optimized_controls", "reference_metrics", "reference_states"),
            ("grid", "grid_best_controls", "grid_best_metrics", "grid_best_states"),
        ):
            computed, states = evaluate_reference(snapshots["first_S02_violation"], reference[controls_name])
            require(len(states) == computed["reference_executed_steps"] == 3, "reference length")
            require("recovered_by_30" not in reference[metrics_name], "invalid reference 30-step endpoint")
            for metric, value in computed.items():
                equal_numbers(value, reference[metrics_name][metric], f"reference {name}/{metric}")
            require(json.loads(json.dumps(states)) == reference[states_name], "reference state replay mismatch")
    source_checks = 0
    for name, digest in meta["code_sha256"].items():
        source = Path(name)
        saved = directory / "source_snapshot" / (Path("aircraft_sim") / source.name if source.parent.name == "aircraft_sim" else Path(source.name))
        require(hashlib.sha256(saved.read_bytes()).hexdigest() == digest, "saved source hash mismatch")
        require(hashlib.sha256(source.read_bytes()).hexdigest() == digest, "running source changed")
        source_checks += 1
    checked = {
        "directory": str(directory), "requested_cases": expected,
        "ready_cases": len(ready), "excluded_cases": meta["excluded_cases"],
        "timing_branches": len(rows), "trace_steps": sum(map(len, branches.values())),
        "reference_states_replayed": len(references) * 6,
        "verified_source_files": source_checks, "all_checks_passed": True,
        "unique_reference_physical_states": meta["unique_oracle_physical_states"],
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()},
    }
    return meta, rows, references, checked, branches


def paired_statistics(rows):
    # Exploratory family: six contrasts x two endpoints. Units are seed means
    # across both episodes and all five distance templates, never raw steps.
    contrasts = [("warning_now", m) for m in ("NoShield", "PartialShield", "JointHeuristic")]
    contrasts += [(timing, "DG-QP") for timing in ("warning_plus_1", "warning_plus_2", "first_S02_violation")]
    result = []
    for timing, method in contrasts:
        for metric in ("target_safety_retention_30step", "minimum_separation_from_warning_m"):
            left = {ident(r): r[metric] for r in rows if r["timing"] == "warning_now" and r["method"] == "DG-QP"}
            right = {ident(r): r[metric] for r in rows if r["timing"] == timing and r["method"] == method}
            require(left.keys() == right.keys(), "unpaired statistics")
            by_seed = defaultdict(list)
            for identity in left:
                by_seed[identity[0]].append(left[identity] - right[identity])
            deltas = [statistics.mean(v) for _, v in sorted(by_seed.items())]
            nonzero = [v for v in deltas if abs(v) > 1e-9]
            p = float(binomtest(sum(v > 0 for v in nonzero), len(nonzero), .5).pvalue) if nonzero else 1.
            ci = float(t.ppf(.975, len(deltas)-1)) * statistics.stdev(deltas) / len(deltas)**.5
            result.append({"left": "DG-QP warning_now", "right": f"{method} {timing}", "metric": metric,
                           "n_seeds": len(deltas), "seed_deltas": deltas, "mean_delta": statistics.mean(deltas),
                           "ci95_halfwidth_t": ci, "sign_p_two_sided": p, "non_tie_seeds": len(nonzero)})
    running = 0.
    for rank, row in enumerate(sorted(result, key=lambda r: r["sign_p_two_sided"])):
        running = max(running, min(1., row["sign_p_two_sided"] * (len(result)-rank)))
        row["sign_p_holm_family12"] = running
    return result


METRICS = ["target_safety_retention_3step", "target_safety_retention_30step", "S02_safety_retention_30step",
           "minimum_separation_from_warning_m", "target_violation_state_rate", "S02_violation_state_rate",
           "C03_violation_state_rate", "C02_violation_state_rate", "support_service_state_rate",
           "cumulative_reward_from_warning", "mean_task_progress_gain", "qp_fallback_rate_from_warning"]
RECOVERY = ["post_S02_recovered_within_3_steps", "post_S02_confirmed_within_3_steps",
            "post_S02_recovered_by_3_and_held_to_common_end"]


def tables(rows, references):
    immediate = [r for r in rows if r["timing"] == "warning_now"]
    broad = means(immediate, ["method"], METRICS)
    dg = means([r for r in rows if r["method"] == "DG-QP"], ["distance", "timing"], METRICS)
    recovery = means([r for r in rows if r["timing"] == "first_S02_violation"], ["distance", "method"], RECOVERY)
    oracle = means(references, ["distance"], ["reference_recovered_by_3", "grid_recovered_by_3", "reference_minimum_separation_m", "one_step_pair_distance_upper_bound_m"])
    lines = ["## 立即介入：跨五类距离模板的种子均值", "",
             "| 配置 | 三步 S02/S03 保持率 | 30步 S02/S03 保持率 | 30步仅S02保持率 | 最小间隔/m | C03违规状态比例 | C02违规状态比例 | 有效支援动作比例 | 30步累计奖励 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in broad:
        fields = [row[k]["mean"] for k in METRICS[:4]] + [row[k]["mean"] for k in ("C03_violation_state_rate", "C02_violation_state_rate", "support_service_state_rate", "cumulative_reward_from_warning")]
        lines.append(f"| {row['method']} | " + " | ".join(f"{v:.4f}" for v in fields) + " |")
    lines += ["", "## DG-QP：介入时机与最小间隔", "", "| 距离模板/m | 立即 | 延迟1步 | 延迟2步 | 首次S02违规后 |", "|---:|---:|---:|---:|---:|"]
    for distance in sorted({r["distance"] for r in rows}):
        selected = {r["timing"]: r for r in dg if r["distance"] == distance}
        values = [selected[k]["minimum_separation_from_warning_m"]["mean"] for k in ("warning_now", "warning_plus_1", "warning_plus_2", "first_S02_violation")]
        lines.append(f"| {distance:.0f} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", "## 首次S02违规后的三步恢复：相同接管状态", "", "| 距离模板/m | NoShield | PartialShield | JointHeuristic | DG-QP | 真值可行参考 |", "|---:|---:|---:|---:|---:|---:|"]
    for distance in sorted({r["distance"] for r in rows}):
        selected = {r["method"]: r for r in recovery if r["distance"] == distance}
        values = [selected[k][RECOVERY[0]]["mean"] for k in ("NoShield", "PartialShield", "JointHeuristic", "DG-QP")]
        values.append(next(r for r in oracle if r["distance"] == distance)["reference_recovered_by_3"]["mean"])
        lines.append(f"| {distance:.0f} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    return {"immediate": broad, "DG_timing": dg, "post_S02_recovery": recovery, "reference": oracle}, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    analyses, report = [], ["# 高冲突介入时机与恢复能力：实验审计与汇总", "",
        "本实验固定规则无关的名义动作，比较安全层配置，不是重新训练的 MAPPO 性能比较。",
        "所有介入分支共用真值离线预警时钟，观察到预警后第30步；延迟期间的违规和奖励保留。",
        "三步恢复是首次进入无S02/S03违规且无待解决危险趋势的状态，不代表前三步全程安全或连续两步确认恢复。",
        "真值参考只执行三步，具有全状态和联合控制权限，是可行性证据，不是公平竞争基线或全局最优上界。", ""]
    for directory in args.inputs:
        meta, rows, references, checked, branches = audit(directory)
        descriptions, lines = tables(rows, references)
        varied = meta["position_jitter_m"] > 0
        statistics_rows = paired_statistics(rows) if varied else []
        analysis = {"metadata": meta, "audit": checked, **descriptions,
                    "exploratory_paired_statistics": statistics_rows,
                    "one_step_upper_bound_below_800_count": sum(r["one_step_pair_distance_upper_bound_m"] < 800 for r in references),
                    "reference_valid_three_step_witness_count": sum(r["reference_recovered_by_3"] == 1 for r in references),
                    "grid_valid_three_step_witness_count": sum(r["grid_recovered_by_3"] == 1 for r in references)}
        save_json(args.output / ("varied_analysis.json" if varied else "exact_analysis.json"), analysis)
        save_json(args.output / ("varied_post_S02_rows.json" if varied else "exact_post_S02_rows.json"), [r for r in rows if r["timing"] == "first_S02_violation"])
        analyses.append(analysis)
        report += [f"# {'位置扰动验证' if varied else '固定几何复核'}", "", f"来源：`{directory}`", "",
                   f"完成 {checked['ready_cases']} 个场景、{checked['timing_branches']} 条轨迹、{checked['trace_steps']} 个执行步；排除 {checked['excluded_cases']} 个场景。",
                   f"源文件校验、共用快照校验、时钟校验、逐步指标重算、NoShield跨接管时机一致性检查和 {checked['reference_states_replayed']} 个参考状态重放均通过。", ""] + lines
        if not varied:
            report += ["", "固定几何中10种子并不代表10个独立几何；物理参考仅5个唯一状态，不对该组做显著性推广。"]
        else:
            report += ["", "## 探索性配对统计", "", "每种子先平均两个episode及五类距离；10个种子作为统计单位。六个对比×两个终点共12项作Holm校正，未预注册。", "",
                       "| 对比：立即DG-QP 减去 | 终点 | 均值差 | 95% t区间半宽 | 双侧符号检验p | Holm p |", "|---|---|---:|---:|---:|---:|"]
            for row in statistics_rows:
                report.append(f"| {row['right']} | {row['metric']} | {row['mean_delta']:.4f} | {row['ci95_halfwidth_t']:.4f} | {row['sign_p_two_sided']:.4f} | {row['sign_p_holm_family12']:.4f} |")
        report += ["", f"三步真值可行参考成功 {analysis['reference_valid_three_step_witness_count']}/{len(references)}；固定候选成功 {analysis['grid_valid_three_step_witness_count']}/{len(references)}；一步间隔外包上界低于800m共 {analysis['one_step_upper_bound_below_800_count']}/{len(references)}。", ""]
        # Exact seed11/2000 example, with explicit action-step indexing.
        if not varied:
            identity = (11, 2000., 0)
            row = next(r for r in rows if ident(r) == identity and r["timing"] == "first_S02_violation" and r["method"] == "DG-QP")
            reference = next(r for r in references if ident(r) == identity)
            after = [s for s in branches[identity + ("first_S02_violation", "DG-QP")] if s["phase"] == "after_takeover"]
            report += ["## 可核验案例：seed11，2000m模板", "", f"首次S02违规状态t={reference['trigger_step']}，接管间隔{reference['trigger_minimum_separation_m']:.3f}m。", "", "| 接管后执行步k | DG-QP间隔/m | DG-QP已恢复 | 可行参考间隔/m | 参考已恢复 |", "|---:|---:|---|---:|---|"]
            for k in range(3):
                s, q = after[k], reference["reference_states"][k]
                report.append(f"| {k+1} | {s['minimum_separation_m']:.3f} | {s['restored_safe_state']} | {q['minimum_separation_m']:.3f} | {q['restored_safe_state']} |")
            report += ["", f"DG-QP首次恢复在接管后k={row['post_S02_first_recovery_step']}；三步参考只说明可恢复，途中仍有严重间隔违规。"]
    save_json(args.output / "audit_summary.json", [r["audit"] for r in analyses])
    (args.output / "RESULTS_AND_SECTION_6_5.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "audits": [r["audit"] for r in analyses]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
