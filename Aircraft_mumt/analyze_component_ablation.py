"""Aggregate controlled component ablations and paired net benefits."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import t


VARIANTS = ("DG-QP", "No-Graph", "Fixed-Priority", "Gate-Only")
HIGHER_BETTER = {
    "mean_reward",
    "joint_satisfaction_rate",
    "conditional_joint_satisfaction_rate",
    "target_hard_success_rate",
    "rule_rule_resolution_rate",
}
LOWER_BETTER = {
    "post_truth_hard_violation_rate",
    "qp_infeasible_rate",
}
REPORT_METRICS = (
    "mean_reward",
    "joint_satisfaction_rate",
    "conditional_joint_satisfaction_rate",
    "post_truth_hard_violation_rate",
    "target_hard_success_rate",
    "mean_intervention_rate",
    "mean_online_time_ms",
    "qp_infeasible_rate",
    "rule_rule_conflict_step_rate",
    "rule_rule_resolution_rate",
)


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(t.ppf(0.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values)))


def sign_pvalue(values: list[float]) -> float:
    nonzero = [value for value in values if abs(value) > 1e-12]
    n = len(nonzero)
    if not n:
        return 1.0
    positive = sum(value > 0 for value in nonzero)
    lower = sum(math.comb(n, k) for k in range(positive + 1))
    upper = sum(math.comb(n, k) for k in range(n - positive + 1))
    return min(1.0, 2.0 * min(lower, upper) / (2 ** n))


def mean_by_seed(rows: list[dict]) -> dict[str, dict[int, dict[str, float]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], int(row["seed"]))].append(row)
    result = defaultdict(dict)
    for (variant, seed), values in grouped.items():
        result[variant][seed] = {
            metric: sum(float(row[metric]) for row in values if row[metric] is not None) / max(1, sum(row[metric] is not None for row in values))
            if any(row[metric] is not None for row in values) else None
            for metric in REPORT_METRICS
        }
    return result


def summarize(seed_values: dict[str, dict[int, dict[str, float]]]) -> list[dict]:
    rows = []
    for variant in VARIANTS:
        row = {"variant": variant, "n_seeds": len(seed_values.get(variant, {}))}
        for metric in REPORT_METRICS:
            values = [v[metric] for v in seed_values.get(variant, {}).values() if v[metric] is not None]
            row[f"{metric}_mean"] = sum(values) / max(1, len(values))
            row[f"{metric}_ci95"] = ci95(values)
        rows.append(row)
    return rows


def paired(seed_values: dict[str, dict[int, dict[str, float]]]) -> list[dict]:
    result = []
    full = seed_values["DG-QP"]
    for ablation in VARIANTS[1:]:
        common = sorted(set(full).intersection(seed_values[ablation]))
        for metric in REPORT_METRICS:
            values = []
            raw = []
            for seed in common:
                left = full[seed][metric]
                right = seed_values[ablation][seed][metric]
                if left is None or right is None:
                    continue
                raw_delta = float(left - right)
                raw.append(raw_delta)
                if metric in LOWER_BETTER:
                    values.append(float(right - left))
                elif metric in HIGHER_BETTER:
                    values.append(raw_delta)
                else:
                    # Reward/intervention/time are reported as raw deltas;
                    # their trade-offs should not be collapsed into a score.
                    values.append(raw_delta)
            result.append({
                "component": {"No-Graph": "dependency_graph", "Fixed-Priority": "dynamic_priority", "Gate-Only": "continuous_qp"}[ablation],
                "ablation": ablation,
                "metric": metric,
                "n": len(values),
                "mean_delta_better": sum(values) / max(1, len(values)),
                "ci95_halfwidth": ci95(values),
                "sign_p_two_sided": sign_pvalue(values),
                "mean_delta_raw_full_minus_ablation": sum(raw) / max(1, len(raw)),
            })
    return result


def markdown(summary: list[dict], deltas: list[dict]) -> str:
    lines = [
        "# Controlled component ablation",
        "",
        "All variants use the same full rule reports, templates, seeds and nominal policy.",
        "Positive `delta_better` means the complete DG-QP is better than the ablation for that metric.",
        "",
        "## Seed-level summary",
        "",
        "| Variant | Reward | Joint STL | Truth hard violation | Target success | Intervention/step | Online ms | QP fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def cell(metric: str) -> str:
            return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_ci95']:.4f}"
        qp = "N/A" if row["variant"] == "Gate-Only" else cell("qp_infeasible_rate")
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["variant"], cell("mean_reward"), cell("joint_satisfaction_rate"),
            cell("post_truth_hard_violation_rate"), cell("target_hard_success_rate"),
            cell("mean_intervention_rate"), cell("mean_online_time_ms"), qp,
        ))
    lines += [
        "", "## Paired component net benefits", "",
        "| Component removed | Metric | N seeds | Delta (better direction) | 95% CI half-width | Sign-test p |", 
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in deltas:
        lines.append(f"| {row['component']} | {row['metric']} | {row['n']} | {row['mean_delta_better']:.4f} | {row['ci95_halfwidth']:.4f} | {row['sign_p_two_sided']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    seed_values = mean_by_seed(rows)
    summary = summarize(seed_values)
    deltas = paired(seed_values)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "paired_deltas.json").write_text(json.dumps(deltas, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "paper_table.md").write_text(markdown(summary, deltas), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "variants": len(summary), "paired_rows": len(deltas)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
