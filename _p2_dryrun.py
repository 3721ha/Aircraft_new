# -*- coding: utf-8 -*-
"""P2 dry-run (Gate G2): HEUR / TWS-Script / Oracle across K in {2,4,6,8}.

Outputs: stdout markdown tables + _p2_dryrun.json
  T1  success-rate bracket (floor HEUR / ceiling Oracle) per K
  T2  K* shape dry-run (judged ratio vs K, HEUR & TWS)
  T3  conflict-pair trigger spectrum (steps where BOTH families fire)
Run from workspace root with the project venv python.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from aircraft_sim.detect_env import DetectEnv                       # noqa: E402
from aircraft_sim.detect_baselines import BASELINES                 # noqa: E402
from aircraft_sim.detect_rules import DETECT_CONFLICT_PAIRS, rule_family  # noqa: E402

SEEDS = list(range(8))
KS = [2, 4, 6, 8, 12]


def run_episode(base_name: str, k: int, seed: int) -> dict:
    env = DetectEnv()
    env.reset({"seed": 1000 + seed, "k_red": k})
    pol = BASELINES[base_name]()
    ret = 0.0
    pair_steps = Counter()
    while env.terminal is None:
        res = env.step(pol(env))
        ret += res.reward
        fired = {rule_family(k2) for k2 in
                 list(res.info["report"].violations) + list(res.info["report"].active)}
        for pair in DETECT_CONFLICT_PAIRS:
            a, b = sorted(pair)
            if a in fired and b in fired:
                pair_steps[f"{a}-{b}"] += 1
    m = env.metrics
    return {
        "baseline": base_name,
        "terminal": env.terminal, "t_end": env.t, "k": k, "seed": seed,
        "judged": m["judged"], "return": round(ret, 2),
        "handoffs": m["handoffs"],
        "mor": m["apparent_mor"], "mar": m["apparent_mar"],
        "hidden_mor": m["hidden_mor"], "hidden_mar": m["hidden_mar"],
        "pair_steps": dict(pair_steps),
    }


results = []
for k in KS:
    for name in BASELINES:
        for seed in SEEDS:
            results.append(run_episode(name, k, seed))
        print(f"done K={k} {name}", file=sys.stderr)

(ROOT / "_p2_dryrun.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

# ---------------------------------------------------------------- T1
print("\n## T1 成功率夹逼（8 种子）\n")
print("| K | 基线 | 成功率 | 平均判定 | 平均回合回报 | 平均时长 | MAR 对·步 | MOR 对·步 |")
print("|---|---|---|---|---|---|---|---|")
for k in KS:
    for name in BASELINES:
        rs = [r for r in results if r["k"] == k and r["baseline"] == name]
        succ = sum(1 for r in rs if r["terminal"] == "success") / len(rs)
        judged = sum(r["judged"] for r in rs) / len(rs)
        ret = sum(r["return"] for r in rs) / len(rs)
        tend = sum(r["t_end"] for r in rs) / len(rs)
        mar = sum(r["mar"] for r in rs) / len(rs)
        mor = sum(r["mor"] for r in rs) / len(rs)
        print(f"| {k} | {name} | {succ:.0%} | {judged:.1f}/{k} | {ret:+.0f} | "
              f"{tend:.0f}s | {mar:.0f} | {mor:.0f} |")

# ---------------------------------------------------------------- T2
print("\n## T2 K* 形态 dry-run（判定完成率 = judged/K，8 种子均值）\n")
print("| K | HEUR | TWS-Script | Oracle |")
print("|---|---|---|---|")
for k in KS:
    row = [f"| {k} "]
    for name in BASELINES:
        rs = [r for r in results if r["k"] == k and r["baseline"] == name]
        frac = sum(r["judged"] for r in rs) / (k * len(rs))
        row.append(f"| {frac:.0%} ")
    row.append("|")
    print("".join(row))

# ---------------------------------------------------------------- T3
print("\n## T3 冲突对触发频谱（12 对；全 K 合并，每对触发步数 / 占比）\n")
total_pairs = Counter()
total_steps = 0
for r in results:
    total_pairs.update(r["pair_steps"])
    total_steps += r["t_end"]
print("| 冲突对 | 触发步数 | 占总步数 |")
print("|---|---|---|")
for pair, cnt in sorted(total_pairs.items(), key=lambda kv: -kv[1]):
    print(f"| {pair} | {cnt} | {cnt / total_steps:.1%} |")
silent = [f"{a}-{b}" for a, b in (sorted(p) for p in DETECT_CONFLICT_PAIRS)
          if f"{a}-{b}" not in total_pairs]
print(f"\n静默对（0 触发）: {', '.join(silent) if silent else '无'}")
