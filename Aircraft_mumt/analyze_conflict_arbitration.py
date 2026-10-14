"""Aggregate focused conflict-arbitration records into paper tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["method"])].append(row)
    summary = []
    for (scenario, method), values in sorted(grouped.items()):
        summary.append({
            "scenario": scenario,
            "method": method,
            "n": len(values),
            "truth_hard_violation_rate": sum(v["truth_hard_violation_rate"] for v in values) / max(1, len(values)),
            "target_hard_success_rate": sum(v["target_hard_success_rate"] for v in values) / max(1, len(values)),
            "intervention_rate": sum(v["intervention_rate"] for v in values) / max(1, len(values)),
            "qp_infeasible_rate": sum(v["qp_infeasible_rate"] for v in values) / max(1, len(values)),
        })
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(summary[0]) if summary else ["scenario", "method", "n"]
    with (args.output / "paper_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summary)
    lines = ["| Scenario | Method | N | Truth violation | Target success | Intervention | QP fallback |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {r['scenario']} | {r['method']} | {r['n']} | {r['truth_hard_violation_rate']:.3f} | {r['target_hard_success_rate']:.3f} | {r['intervention_rate']:.3f} | {r['qp_infeasible_rate']:.3f} |" for r in summary)
    (args.output / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
