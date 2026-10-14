# -*- coding: utf-8 -*-
"""Reward decomposition: split episode return into w1..w9 components.
Faithfulness check: component sum must equal env-returned episode return.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from aircraft_sim.detect_env import DetectEnv, ZONE_TRACKING, ZONE_DETECTION  # noqa: E402
from aircraft_sim.detect_baselines import BASELINES  # noqa: E402
from aircraft_sim.detect_rules import RULE_LAYER, rule_family  # noqa: E402


def run_decomposed(base_name, k, seed):
    env = DetectEnv()
    env.reset({"seed": 1000 + seed, "k_red": k})
    cfg = env.cfg
    pol = BASELINES[base_name]()
    comp = defaultdict(float)
    ret = 0.0
    while env.terminal is None:
        judged_before = sum(env.judged.values())
        res = env.step(pol(env))
        ret += res.reward
        report = res.info["report"]
        red_by_uid = {a.uid: a for a in env.reds.agents}
        # w1/w2
        for i, w in enumerate(env.wingmen):
            y0, y1 = cfg.subzone_bounds(w.subzone)
            for tr in w.tracks.slots:
                if tr is None:
                    continue
                a = red_by_uid.get(tr.target_id)
                if a is None:
                    continue
                zone = cfg.zone_of(a.x)
                if zone == ZONE_TRACKING and env.responsibility.get(tr.target_id) == i \
                        and tr.age(env.t) < cfg.AGE_STD_TRACK:
                    comp["w1 跟踪达标(+0.02/s/目标)"] += cfg.W1_TRACK_TICK
                elif zone == ZONE_DETECTION and y0 <= a.y < y1 \
                        and tr.age(env.t) < cfg.AGE_STD_DETECT:
                    comp["w2 探测达标(+0.005/s/目标)"] += cfg.W2_DETECT_TICK
        # w3
        nj = sum(env.judged.values()) - judged_before
        comp["w3 稳定窗判定(+10/目标)"] += cfg.W3_STABLE * nj
        # w5/w6
        blue_xy = [(w.x, w.y) for w in env.wingmen]
        for a in env.reds.agents:
            for bx, by in blue_xy:
                d = math.hypot(a.x - bx, a.y - by)
                if d < cfg.MAR:
                    comp["w6 MAR(-0.1/对·步)"] -= cfg.W6_MAR
                elif d < cfg.MOR:
                    comp["w5 MOR梯度"] -= cfg.W5_MOR_GRAD * (cfg.MOR - d) / cfg.MOR_GRADIENT_SPAN
        # w7
        comp["w7 交接(-0.5/次)"] -= cfg.W7_HANDOFF * len(res.info["handoffs"])
        # w8 by family
        wl = cfg.W8_LAYER
        for key in report.violations:
            fam = rule_family(key)
            if fam in ("S04", "S05"):
                comp["w8豁免 S04/S05(不计罚)"] -= 0.0
                comp[f"   [豁免计数 {fam}]"] -= 0.0  # placeholder, counted below
            else:
                pen = wl[RULE_LAYER[fam]]
                comp[f"w8 违规 {fam}(P{RULE_LAYER[fam]},-{pen}/步)"] -= pen
        comp["步代价(-0.005/s)"] -= cfg.STEP_COST
        if res.info["terminal"] == "success":
            comp["w4 任务成功(+50)"] += cfg.W4_SUCCESS
        elif res.info["terminal"] == "failure":
            comp["w9 任务失败(-50)"] -= cfg.W9_FAILURE
    # exempt counts (informational)
    exempt = defaultdict(int)
    return comp, ret, env.t, env.terminal, exempt


def show(name, k, seed):
    comp, ret, t, terminal, _ = run_decomposed(name, k, seed)
    print(f"\n=== {name} K={k} seed={seed} | terminal={terminal} t={t:.0f}s | 实际回报 {ret:+.1f}")
    tot = 0.0
    for key, v in sorted(comp.items(), key=lambda kv: kv[1]):
        if abs(v) < 1e-9:
            continue
        print(f"  {key:34s} {v:+10.1f}")
        tot += v
    print(f"  {'分量合计':34s} {tot:+10.1f}   校验差 {tot - ret:+.2f}")


for nm in ["heur", "tws", "oracle"]:
    show(nm, 4, 0)
show("tws", 8, 0)
show("heur", 12, 0)
