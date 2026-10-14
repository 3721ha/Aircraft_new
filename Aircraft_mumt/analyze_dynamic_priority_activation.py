"""Summarize the isolated dynamic-priority activation experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def metric(row: dict, key: str) -> float:
    return float(row["metrics"].get(key, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    variants = sorted({row["variant"] for row in rows})
    by_variant = {variant: [row for row in rows if row["variant"] == variant] for variant in variants}
    summary = {
        "protocol": "dynamic_priority_activation_v1",
        "variants": {},
        "paired": {},
    }
    for variant, subset in by_variant.items():
        modes = [row["initial_arbitration"]["agent0_selected_mode"] for row in subset]
        summary["variants"][variant] = {
            "N": len(subset),
            "mode_counts": {mode: modes.count(mode) for mode in sorted(set(modes))},
            "target_pair_active_rate": mean(float(row["initial_target_pair_active"]) for row in subset),
            "mean_reward": mean(metric(row, "mean_reward") for row in subset),
            "ci95_reward": ci95([metric(row, "mean_reward") for row in subset]),
            "mean_joint_stl": mean(metric(row, "joint_satisfaction_rate") for row in subset),
            "ci95_joint_stl": ci95([metric(row, "joint_satisfaction_rate") for row in subset]),
            "mean_truth_hard_violation": mean(metric(row, "post_truth_hard_violation_rate") for row in subset),
            "ci95_truth_hard_violation": ci95([metric(row, "post_truth_hard_violation_rate") for row in subset]),
            "mean_online_ms": mean(metric(row, "mean_online_time_ms") for row in subset),
        }

    dynamic = {row["seed"]: row for row in by_variant.get("Dynamic-Priority", [])}
    fixed = {row["seed"]: row for row in by_variant.get("Fixed-Priority", [])}
    paired = []
    for seed in sorted(set(dynamic) & set(fixed)):
        d = dynamic[seed]
        f = fixed[seed]
        paired.append(
            {
                "seed": seed,
                "dynamic_mode": d["initial_arbitration"]["agent0_selected_mode"],
                "fixed_mode": f["initial_arbitration"]["agent0_selected_mode"],
                "mode_different": d["initial_arbitration"]["agent0_selected_mode"]
                != f["initial_arbitration"]["agent0_selected_mode"],
                "reward_delta_dynamic_minus_fixed": metric(d, "mean_reward") - metric(f, "mean_reward"),
                "joint_stl_delta_dynamic_minus_fixed": metric(d, "joint_satisfaction_rate") - metric(f, "joint_satisfaction_rate"),
                "truth_violation_delta_dynamic_minus_fixed": metric(d, "post_truth_hard_violation_rate") - metric(f, "post_truth_hard_violation_rate"),
            }
        )
    mode_diff = [float(item["mode_different"]) for item in paired]
    summary["paired"] = {
        "N": len(paired),
        "mode_difference_rate": mean(mode_diff) if mode_diff else 0.0,
        "reward_delta_mean": mean(item["reward_delta_dynamic_minus_fixed"] for item in paired) if paired else 0.0,
        "joint_stl_delta_mean": mean(item["joint_stl_delta_dynamic_minus_fixed"] for item in paired) if paired else 0.0,
        "truth_violation_delta_mean": mean(item["truth_violation_delta_dynamic_minus_fixed"] for item in paired) if paired else 0.0,
        "per_seed": paired,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "paired_deltas.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Dynamic-priority activation experiment",
        "",
        "This isolated experiment does not alter the official protocol. The primary endpoint is whether simultaneous I02/F04 activation changes the discrete arbitration mode.",
        "",
        "| Variant | Pair active | A0 mode counts | Reward | Joint STL | Truth hard violation | Online ms |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for variant in variants:
        item = summary["variants"][variant]
        counts = ", ".join(f"{mode}={count}" for mode, count in item["mode_counts"].items())
        lines.append(
            f"| {variant} | {item['target_pair_active_rate']:.3f} | {counts} | "
            f"{item['mean_reward']:.4f} +/- {item['ci95_reward']:.4f} | "
            f"{item['mean_joint_stl']:.4f} +/- {item['ci95_joint_stl']:.4f} | "
            f"{item['mean_truth_hard_violation']:.4f} +/- {item['ci95_truth_hard_violation']:.4f} | "
            f"{item['mean_online_ms']:.3f} |"
        )
    lines.extend([
        "",
        f"Paired mode-difference rate: {summary['paired']['mode_difference_rate']:.3f}",
        f"Reward delta (dynamic - fixed): {summary['paired']['reward_delta_mean']:.6f}",
        f"Joint STL delta (dynamic - fixed): {summary['paired']['joint_stl_delta_mean']:.6f}",
        f"Truth violation delta (dynamic - fixed): {summary['paired']['truth_violation_delta_mean']:.6f}",
    ])
    (args.output / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "paired": len(paired)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
