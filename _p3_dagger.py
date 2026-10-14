# -*- coding: utf-8 -*-
"""P3 DAgger: on-distribution warm start with shield-in-the-loop (Gate-5 plan A).

Problem (see P3课程段中段报告-决策门5.md): one-shot BC learned HEUR's action on
HEUR's state distribution; at execution the QP shield rewrites nearly every
action (qp_share=1.0), pushing the policy off-distribution -> copy bleeds
-15000 where the teacher scores -500. PPO cannot fix this because the policy
never reaches the positive-reward region (w1/w3 = 0).

DAgger loop (per round):
  1. roll out CURRENT policy (stochastic, shield in loop) at mixed K in {2,4}
  2. label every visited state with HEUR's nominal action
  3. aggregate dataset, weighted BC for BC_EPOCHS epochs
  4. deterministic eval on the gate-5 protocol (K=2 seeds 300-302, K=4 200-204)

CLI: _p3_dagger.py [round0] [n_rounds]     # one invocation = n_rounds rounds
Outputs: runs/detect_pilot/dagger_r{N}.pt, dagger_latest.pt, pilot_log.json["dagger"]
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

ROUND0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
PHASE = sys.argv[1] if len(sys.argv) > 1 else "round"   # expert | round

EPISODES_PER_ROUND = 4          # K alternates 2,4,2,4
BC_EPOCHS_PER_ROUND = 40        # 20 was under-trained (loss 0.55 vs 0.31 @60ep)
REJECT_TOL = 200.0              # accept candidate unless eval degrades by >tol
EVAL_PROTOCOL = [(2, [300, 301, 302]), (4, [200, 201, 202, 203, 204])]


def evaluate(pol, shield):
    rows = []
    for k, seeds in EVAL_PROTOCOL:
        for s in seeds:
            env = DetectEnv()
            env.reset({"seed": s, "k_red": k})
            pol.env = env
            ret = 0.0
            while env.terminal is None:
                obs = env.observe()
                nom = {i: pol.act_with_log_prob(obs[i], deterministic=True)[0]
                       for i in obs}
                safe, _ = shield.filter(env, nom)
                ret += env.step(safe).reward
            rows.append({"k": k, "seed": s, "terminal": env.terminal,
                         "return": round(ret, 1)})
    return rows


def collect_demos(pol, shield, heur, k, seed):
    """Roll out current policy (shielded); label visited states with HEUR."""
    env = DetectEnv()
    obs = env.reset({"seed": seed, "k_red": k})
    demos = []
    while env.terminal is None:
        teacher = heur(env)                       # nominal expert action
        for i in teacher:
            demos.append((pol.encoder.encode(obs[i]), teacher[i]))
        nom, _ = pol.act_batch(obs, deterministic=False)
        safe, _ = shield.filter(env, nom)
        obs = env.step(safe).observations
    return demos


def collect_expert_demos(pol, shield, heur, k, seed):
    """Roll out HEUR itself WITH the shield in loop; label = HEUR nominal.

    Fixes the original BC mismatch (plan §9.4 BC was collected shield-free but
    executed shield-in-loop -> qp_share 1.0 shifted states off-distribution).
    Expert rollouts also visit the POSITIVE-reward region (successful eps)."""
    env = DetectEnv()
    obs = env.reset({"seed": seed, "k_red": k})
    demos = []
    while env.terminal is None:
        teacher = heur(env)
        for i in teacher:
            demos.append((pol.encoder.encode(obs[i]), teacher[i]))
        safe, _ = shield.filter(env, teacher)
        obs = env.step(safe).observations
    return demos


def _load_log():
    f = OUT / "pilot_log.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def main():
    t0 = time.time()
    env = DetectEnv()
    shield = DetectQPShield(env.cfg)
    pol = DetectMAPPOPolicy(env, shield, seed=0)

    latest = OUT / "dagger_latest.pt"

    # ---------------- Phase "expert": shield-in-loop HEUR BC ----------------
    if PHASE == "expert":
        start = OUT / "pilot_final.pt"
        pol.load(str(start if start.exists() else OUT / "pilot_bc.pt"))
        inc_ev = evaluate(pol, shield)
        incumbent_ret = sum(e["return"] for e in inc_ev) / len(inc_ev)
        print(f"[expert] incumbent (pilot_final) eval ret {incumbent_ret:+.0f}",
              flush=True)
        incumbent_state = {k: v.clone()
                           for k, v in pol.model.state_dict().items()}
        heur = HeurPolicy()
        demos = []
        for ep in range(8):
            k = 2 if ep < 4 else 4
            demos += collect_expert_demos(pol, shield, heur, k, seed=100 + ep)
            print(f"[expert] ep{ep} K={k} demos total {len(demos)}", flush=True)
        stats = pol.behavior_clone(demos, epochs=60)
        ev = evaluate(pol, shield)
        cand_ret = sum(e["return"] for e in ev) / len(ev)
        succ = sum(1 for e in ev if e["terminal"] == "success")
        accepted = cand_ret >= incumbent_ret - REJECT_TOL
        if accepted:
            pol.save(str(OUT / "expert_shielded.pt"))
            pol.save(str(latest))
        else:
            pol.model.load_state_dict(incumbent_state)
        log = _load_log()
        log["expert_bc"] = {"demos": len(demos), "bc": stats, "eval": ev,
                            "accepted": accepted,
                            "incumbent_return": round(incumbent_ret, 1),
                            "eval_success": f"{succ}/{len(ev)}",
                            "eval_mean_return": round(cand_ret, 1)}
        (OUT / "pilot_log.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=1))
        print(f"[expert] bc_loss {stats['loss']:.3f} | eval succ {succ}/8 "
              f"ret {cand_ret:+.0f} -> {'ACCEPT' if accepted else 'REJECT'}",
              flush=True)
        for e in ev:
            print(f"    K={e['k']} seed={e['seed']} {e['terminal']:8s} "
                  f"ret {e['return']:+.0f}", flush=True)
        print(f"wall {round((time.time() - t0) / 60, 1)} min")
        return

    # ---------------- Phase "round": DAgger refinement rounds ----------------
    if ROUND0 == 0:
        src = latest if latest.exists() else (
            OUT / "pilot_final.pt" if (OUT / "pilot_final.pt").exists()
            else OUT / "pilot_bc.pt")
        pol.load(str(src))
    else:
        pol.load(str(latest))

    log = _load_log()
    dag = log.setdefault("dagger", {"rounds": [], "demos_total": 0})
    agg = []          # aggregated dataset lives only within one invocation's rounds

    for r in range(ROUND0, ROUND0 + N_ROUNDS):
        # ---- incumbent baseline (accept/reject guard) ----
        if r == 0 or "incumbent_ret" not in locals():
            inc_ev = evaluate(pol, shield)
            incumbent_ret = sum(e["return"] for e in inc_ev) / len(inc_ev)
            print(f"[dagger r{r}] incumbent eval ret {incumbent_ret:+.0f}",
                  flush=True)
        incumbent_state = {k: v.clone() for k, v in pol.model.state_dict().items()}

        heur = HeurPolicy()
        demos = []
        for ep in range(EPISODES_PER_ROUND):
            k = 2 if ep % 2 == 0 else 4
            if ep < 2:
                # anchor: keep successful expert trajectories in the aggregate
                demos += collect_expert_demos(pol, shield, heur, k,
                                              seed=5000 + r * 10 + ep)
            else:
                # correction: label the learner's own visited states
                demos += collect_demos(pol, shield, heur, k,
                                       seed=4000 + r * 10 + ep)
        agg += demos
        # gentle refinement: full-lr BC (1e-3 x 40ep) destroys the incumbent
        # (r0: -5457 -> -9745 REJECTED); fine-tune at 1e-4 instead
        pol.config.bc_learning_rate = 1e-4
        stats = pol.behavior_clone(agg, epochs=BC_EPOCHS_PER_ROUND)
        ev = evaluate(pol, shield)
        cand_ret = sum(e["return"] for e in ev) / len(ev)
        accepted = cand_ret >= incumbent_ret - REJECT_TOL
        if accepted:
            incumbent_ret = max(incumbent_ret, cand_ret)
            pol.save(str(OUT / f"dagger_r{r}.pt"))
            pol.save(str(latest))
        else:
            pol.model.load_state_dict(incumbent_state)   # reject, revert
        succ = sum(1 for e in ev if e["terminal"] == "success")
        entry = {"round": r, "new_demos": len(demos), "agg_demos": len(agg),
                 "bc": stats, "eval": ev, "accepted": accepted,
                 "incumbent_return": round(incumbent_ret, 1),
                 "eval_success": f"{succ}/{len(ev)}",
                 "eval_mean_return": round(cand_ret, 1)}
        dag["rounds"].append(entry)
        dag["demos_total"] += len(demos)
        (OUT / "pilot_log.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=1))
        print(f"[dagger r{r}] demos +{len(demos)} (agg {len(agg)}) "
              f"bc_loss {stats['loss']:.3f} | eval succ {succ}/{len(ev)} "
              f"ret {cand_ret:+.0f} -> {'ACCEPT' if accepted else 'REJECT'}",
              flush=True)
        for e in ev:
            print(f"    K={e['k']} seed={e['seed']} {e['terminal']:8s} "
                  f"ret {e['return']:+.0f}", flush=True)

    print(f"wall {round((time.time() - t0) / 60, 1)} min -> {OUT}")


if __name__ == "__main__":
    main()
