# -*- coding: utf-8 -*-
"""P3 pilot: BC warm start (HEUR) + short MAPPO run + Gate-4 metrics.

Phases:
  1. BC   -- imitate HeurPolicy demos (plan §9.4, 50-100 epochs), then eval
  2. MAPPO pilot -- iterations of (collect N episodes -> PPO update)
  3. Gate 4 readout -- intervention-type distribution (continuous QP share
     must be >= 30% of interventions) + reward/violation trend

Outputs: runs/detect_pilot/pilot_log.json, pilot_bc.pt, pilot_final.pt
Run from workspace root:  Aircraft_mumt/.venv/Scripts/python.exe _p3_pilot.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "Aircraft_mumt"))

from aircraft_sim.detect_baselines import HeurPolicy              # noqa: E402
from aircraft_sim.detect_env import DetectEnv                     # noqa: E402
from aircraft_sim.detect_mappo import DetectMAPPOPolicy           # noqa: E402
from aircraft_sim.detect_shield import DetectQPShield             # noqa: E402

OUT = ROOT / "Aircraft_mumt" / "runs" / "detect_pilot"
OUT.mkdir(parents=True, exist_ok=True)

# CLI: _p3_pilot.py [phase] ; phase = bc | pilot <iter0> <n_iters> | eval
#      | curr <k> <iter0> <n_iters>  (curriculum: train at K=k, eval each 10)
PHASE = sys.argv[1] if len(sys.argv) > 1 else "all"
ITERS0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
CURR_K = int(sys.argv[2]) if PHASE == "curr" and len(sys.argv) > 2 else 2
if PHASE == "curr":
    ITERS0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    N_ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 50

BC_EPISODES = 8          # K=4, seeds 100+
BC_EPOCHS = 60
CURRICULUM_START = 100          # gate-6: penalty anneal begins at iter 100
CURRICULUM_SPAN = 30            # w8 scale 0.2 -> 1.0 linear over 30 iters
EVAL_SEEDS = [200, 201, 202, 203, 204]
EPISODES_PER_ITER = 4


def evaluate(pol, shield, seeds, k=4):
    rows = []
    for s in seeds:
        env = DetectEnv()
        env.reset({"seed": s, "k_red": k})
        pol.env = env
        ret, steps, vio = 0.0, 0, 0
        while env.terminal is None:
            obs = env.observe()
            nom = {i: pol.act_with_log_prob(obs[i], deterministic=True)[0] for i in obs}
            safe, _ = shield.filter(env, nom)
            res = env.step(safe)
            ret += res.reward
            vio += len(res.info["report"].violations)
            steps += 1
        rows.append({"seed": s, "terminal": env.terminal, "return": round(ret, 1),
                     "steps": steps, "violations": vio})
    return rows


def _load_log():
    f = OUT / "pilot_log.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"iters": []}


def main():
    t_start = time.time()
    env = DetectEnv()
    shield = DetectQPShield(env.cfg)
    pol = DetectMAPPOPolicy(env, shield, seed=0)

    # ---------------- Phase 1: BC ----------------
    if PHASE in ("all", "bc"):
        heur = HeurPolicy()
        demos = []
        for ep in range(BC_EPISODES):
            env.reset({"seed": 100 + ep, "k_red": 4})
            obs = env.observe()
            while env.terminal is None:
                acts = heur(env)
                for i in acts:
                    demos.append((pol.encoder.encode(obs[i]), acts[i]))
                env.step(acts)
                obs = env.observe()
        bc_stats = pol.behavior_clone(demos, epochs=BC_EPOCHS)
        pol.save(str(OUT / "pilot_bc.pt"))
        bc_eval = evaluate(pol, shield, EVAL_SEEDS)
        print(f"[BC] {bc_stats} | eval: "
              f"{sum(r['terminal'] == 'success' for r in bc_eval)}/{len(bc_eval)} success, "
              f"mean return {sum(r['return'] for r in bc_eval) / len(bc_eval):.0f}",
              flush=True)
        log = _load_log()
        log["bc"] = {"stats": bc_stats, "eval": bc_eval}
        (OUT / "pilot_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
        if PHASE == "bc":
            return

    # ---------------- Phase 2: MAPPO pilot ----------------
    if PHASE in ("all", "pilot"):
        pol.load(str(OUT / "pilot_bc.pt"))
        env = DetectEnv()
        pol.env = env
        log = _load_log()
        for it in range(ITERS0, ITERS0 + N_ITERS):
            rollout, ep_rets, ep_term, ep_steps, ep_vio = [], [], [], [], []
            n_qp = n_fb = 0
            for ep in range(EPISODES_PER_ITER):
                data = pol.collect(k_red=4, seed=1000 + it * 10 + ep)
                rollout.extend(data)
                ep_rets.append(sum(d["reward"] for d in data) / pol.config.reward_scale)
                ep_term.append(data[-1]["terminal"])
                ep_steps.append(len(data))
                ep_vio.append(sum(d["violations"] for d in data))
                n_qp += sum(d["n_qp"] for d in data)
                n_fb += sum(d["n_fallback"] for d in data)
            stats = pol.update(rollout)
            entry = {
                "iter": it,
                "mean_return": round(sum(ep_rets) / len(ep_rets), 1),
                "success": sum(t == "success" for t in ep_term) / len(ep_term),
                "mean_steps": round(sum(ep_steps) / len(ep_steps)),
                "vio_per_step": round(sum(ep_vio) / max(1, sum(ep_steps)), 3),
                "qp_share": round(n_qp / max(1, n_qp + n_fb), 3),
                "interventions_per_ep": round((n_qp + n_fb) / EPISODES_PER_ITER, 1),
                "policy_loss": round(stats["policy_loss"], 4),
                "value_loss": round(stats["value_loss"], 2),
                "approx_kl": round(stats["approx_kl"], 4),
            }
            log["iters"].append(entry)
            print(f"[{it:02d}] ret={entry['mean_return']:>8} succ={entry['success']:.0%} "
                  f"vio/step={entry['vio_per_step']:.2f} qp_share={entry['qp_share']:.2f} "
                  f"kl={entry['approx_kl']:.3f}", flush=True)
            pol.save(str(OUT / "pilot_iter.pt"))
            (OUT / "pilot_log.json").write_text(
                json.dumps(log, ensure_ascii=False, indent=1))
        (OUT / "pilot_final.pt").rename(OUT / "pilot_final_prev.pt") \
            if (OUT / "pilot_final.pt").exists() else None
        (OUT / "pilot_iter.pt").rename(OUT / "pilot_final.pt")

    # ---------------- Phase 2b: curriculum long run ----------------
    if PHASE == "curr":
        # warm-start priority: existing curriculum ckpt > dagger > pilot_final > pilot_bc
        ckpt = OUT / f"curr_k{CURR_K}.pt"
        if ITERS0 == 0 and not ckpt.exists():
            for cand in ("dagger_latest.pt", "pilot_final.pt", "pilot_bc.pt"):
                if (OUT / cand).exists():
                    pol.load(str(OUT / cand))
                    break
        else:
            pol.load(str(ckpt))
        env = DetectEnv()
        pol.env = env
        log = _load_log()
        ck = f"curr_k{CURR_K}"
        log.setdefault(ck, [])
        for it in range(ITERS0, ITERS0 + N_ITERS):
            # gate-6 penalty curriculum: cheap rules early, full bite by it 130
            # (per-episode cfg override; eval path never passes it -> always 1.0)
            w8_scale = 0.2 + 0.8 * min(1.0, max(0.0,
                                                (it - CURRICULUM_START)
                                                / CURRICULUM_SPAN))
            rollout, ep_rets, ep_term, ep_steps, ep_vio = [], [], [], [], []
            n_qp = n_fb = 0
            for ep in range(EPISODES_PER_ITER):
                data = pol.collect(k_red=CURR_K, seed=2000 + it * 10 + ep,
                                   w8_scale=w8_scale)
                rollout.extend(data)
                ep_rets.append(sum(d["reward"] for d in data) / pol.config.reward_scale)
                ep_term.append(data[-1]["terminal"])
                ep_steps.append(len(data))
                ep_vio.append(sum(d["violations"] for d in data))
                n_qp += sum(d["n_qp"] for d in data)
                n_fb += sum(d["n_fallback"] for d in data)
            stats = pol.update(rollout)
            entry = {
                "iter": it,
                "mean_return": round(sum(ep_rets) / len(ep_rets), 1),
                "success": sum(t == "success" for t in ep_term) / len(ep_term),
                "vio_per_step": round(sum(ep_vio) / max(1, sum(ep_steps)), 3),
                "qp_share": round(n_qp / max(1, n_qp + n_fb), 3),
                "approx_kl": round(stats["approx_kl"], 4),
                "w8_scale": round(w8_scale, 2),
            }
            log[ck].append(entry)
            print(f"[{ck} {it:02d}] ret={entry['mean_return']:>8} succ={entry['success']:.0%} "
                  f"vio/step={entry['vio_per_step']:.2f} qp={entry['qp_share']:.2f} "
                  f"kl={entry['approx_kl']:.3f}", flush=True)
            pol.save(str(OUT / f"curr_k{CURR_K}.pt"))
            if (it + 1) % 10 == 0:
                ev = evaluate(pol, shield, [300, 301, 302], k=CURR_K)
                esucc = sum(r["terminal"] == "success" for r in ev) / len(ev)
                eret = sum(r["return"] for r in ev) / len(ev)
                log[ck][-1]["eval"] = {"success": esucc, "mean_return": round(eret, 1)}
                print(f"  [eval K={CURR_K}] {esucc:.0%} success, ret {eret:.0f}", flush=True)
            (OUT / "pilot_log.json").write_text(
                json.dumps(log, ensure_ascii=False, indent=1))
        return

    # ---------------- Phase 3: final eval ----------------
    if PHASE in ("all", "eval"):
        pol.load(str(OUT / "pilot_final.pt"))
        env = DetectEnv()
        pol.env = env
        final_eval = evaluate(pol, shield, EVAL_SEEDS)
        log = _load_log()
        log["final_eval"] = final_eval
        succ = sum(r["terminal"] == "success" for r in final_eval) / len(final_eval)
        mean_ret = sum(r["return"] for r in final_eval) / len(final_eval)
        print(f"[FINAL eval] {succ:.0%} success, mean return {mean_ret:.0f}", flush=True)
        (OUT / "pilot_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))

    print(f"wall {round((time.time() - t_start) / 60, 1)} min -> {OUT}")


if __name__ == "__main__":
    main()
