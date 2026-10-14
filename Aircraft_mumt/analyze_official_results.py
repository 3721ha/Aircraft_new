"""Create paper-ready tables from a completed official comparison run."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


MAIN_FIELDS = (
    "mean_reward_mean",
    "joint_satisfaction_rate_mean",
    "conditional_joint_satisfaction_rate_mean",
    "post_truth_hard_violation_rate_mean",
    "mean_intervention_rate_mean",
    "mean_actor_time_ms_mean",
    "mean_online_time_ms_mean",
)


def exact_sign_pvalue(deltas: list[float]) -> float | None:
    nonzero = [delta for delta in deltas if abs(delta) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positive = sum(delta > 0 for delta in nonzero)
    # Two-sided exact sign test; n is small (normally five seeds).
    tail = sum(__import__("math").comb(n, k) for k in range(positive + 1))
    other = sum(__import__("math").comb(n, k) for k in range(n - positive + 1))
    return min(1.0, 2.0 * min(tail, other) / (2**n))


def load(path: Path) -> tuple[dict, list[dict], list[dict]]:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    seeds = json.loads((path / "per_seed.json").read_text(encoding="utf-8"))
    episodes = json.loads((path / "per_episode.json").read_text(encoding="utf-8"))
    return payload, seeds, episodes


def write_main_table(root: Path, summaries: list[dict]) -> None:
    fields = ["policy", "seeds", *MAIN_FIELDS]
    with (root / "main_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in summaries)
    lines = [
        "| Method | Seeds | Reward | Joint STL | Conditional STL | Truth hard violation (step) | Intervention | Actor inference ms | Shield online ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['policy']} | {row['seeds']} | {row['mean_reward_mean']:.3f} | "
            f"{row['joint_satisfaction_rate_mean']:.3f} | {row['conditional_joint_satisfaction_rate_mean']:.3f} | "
            f"{row['post_truth_hard_violation_rate_mean']:.3f} | {row['mean_intervention_rate_mean']:.3f} | "
            f"{row.get('mean_actor_time_ms_mean', 0.0):.3f} | {row['mean_online_time_ms_mean']:.3f} |"
        )
    (root / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def paired_comparison(root: Path, rows: list[dict]) -> list[dict]:
    proposed = "proposed_belief_stl_conflict_qp"
    grouped = {}
    for row in rows:
        grouped.setdefault(row["policy"], {})[int(row["seed"])] = row
    output = []
    for policy, by_seed in grouped.items():
        if policy == proposed:
            continue
        common = sorted(set(grouped[proposed]) & set(by_seed))
        deltas = [
            float(grouped[proposed][seed]["joint_satisfaction_rate"])
            - float(by_seed[seed]["joint_satisfaction_rate"])
            for seed in common
        ]
        output.append({
            "baseline": policy,
            "seeds": len(common),
            "proposed_minus_baseline_joint_stl_mean": round(mean(deltas), 6),
            "proposed_minus_baseline_joint_stl_std": round(stdev(deltas), 6) if len(deltas) > 1 else 0.0,
            "proposed_wins": sum(delta > 0 for delta in deltas),
            "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
            "exact_sign_pvalue": round(exact_sign_pvalue(deltas), 6),
        })
    (root / "paired_comparison.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def scenario_table(root: Path, episodes: list[dict]) -> None:
    grouped = {}
    for row in episodes:
        key = (row["policy"], row.get("scenario_type", "unknown"))
        bucket = grouped.setdefault(key, {"total": 0, "satisfied": 0})
        bucket["total"] += 1
        bucket["satisfied"] += int(bool(row.get("joint_hard_satisfied", False)))
    output = [
        {
            "policy": policy,
            "scenario_type": scenario,
            "episodes": bucket["total"],
            "joint_stl_rate": round(bucket["satisfied"] / max(1, bucket["total"]), 6),
        }
        for (policy, scenario), bucket in sorted(grouped.items())
    ]
    (root / "scenario_breakdown.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    fields = ["policy", "scenario_type", "episodes", "joint_stl_rate"]
    with (root / "scenario_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/official_comparison"))
    args = parser.parse_args()
    payload, seeds, episodes = load(args.results)
    write_main_table(args.results, payload["methods"])
    paired = paired_comparison(args.results, seeds)
    scenario_table(args.results, episodes)
    print(json.dumps({
        "output": str(args.results),
        "methods": len(payload["methods"]),
        "paired_comparisons": len(paired),
        "scenario_rows": len(json.loads((args.results / "scenario_breakdown.json").read_text(encoding="utf-8"))),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
