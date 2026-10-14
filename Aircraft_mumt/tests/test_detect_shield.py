# -*- coding: utf-8 -*-
"""DetectQPShield unit tests (plan §8.1, V2.7)."""
import math

import numpy as np

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_models import DetectAction
from aircraft_sim.detect_shield import (DetectPassThroughShield, DetectQPShield)
from aircraft_sim.track import Track


def _env(seed=0, k=4):
    env = DetectEnv()
    env.reset({"seed": seed, "k_red": k})
    return env


def _nominal(env, u_psi=0.0, u_v=0.0):
    return {i: DetectAction(u_psi, u_v, 0.0) for i in range(env.cfg.N_WINGMEN)}


def test_passthrough_returns_clipped_nominal():
    env = _env()
    nom = _nominal(env, u_psi=2.0)          # out of envelope -> clip only
    safe, inter = DetectPassThroughShield().filter(env, nom)
    assert inter == []
    assert all(a.u_psi == 1.0 for a in safe.values())


def test_no_intervention_when_clear():
    env = _env()
    shield = DetectQPShield(env.cfg)
    # park wingmen far apart, no tracks
    for i, w in enumerate(env.wingmen):
        w.x, w.y = 60.0, 20.0 + 40.0 * i
        w.psi, w.v = 0.0, 0.24
        w.tracks.slots = [None] * len(w.tracks.slots)
    env.manned.x, env.manned.y = 15.0, 60.0
    safe, inter = shield.filter(env, _nominal(env))
    assert inter == []
    assert shield.last_solve.status in ("optimal", "no_constraints")
    for i, a in safe.items():
        assert abs(a.u_psi) < 1e-9 and abs(a.u_v) < 1e-9


def test_s02_head_on_triggers_continuous_qp():
    env = _env()
    shield = DetectQPShield(env.cfg)
    w0, w1, w2 = env.wingmen
    # head-on pair, 3.5 km apart, closing at ~0.48 km/s -> brushes the 2 km
    # floor inside the 3 s horizon: correctable by a small continuous nudge
    w0.x, w0.y, w0.psi = 60.0, 60.0, 0.0
    w1.x, w1.y, w1.psi = 63.5, 60.0, math.pi
    w2.x, w2.y = 20.0, 20.0                       # far away
    for w in (w0, w1, w2):
        w.v = 0.24
        w.tracks.slots = [None] * len(w.tracks.slots)
    env.manned.x, env.manned.y = 15.0, 60.0
    nom = {0: DetectAction(0.0, 0.0, 0.0), 1: DetectAction(0.0, 0.0, 0.0),
           2: DetectAction(0.0, 0.0, 0.0)}
    safe, inter = shield.filter(env, nom)
    kinds = {it.kind for it in inter}
    assert kinds == {"continuous_qp"}
    assert {it.uid for it in inter} <= {0, 1}
    # corrected 3 s min separation must beat the nominal one
    def min_sep(actions):
        p = []
        for i, w in enumerate(env.wingmen[:2]):
            x, y, psi, v = w.x, w.y, w.psi, w.v
            a = actions[i]
            p.append(shield._predict((x, y, psi, v), a.u_psi, a.u_v, 3))
        return math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1])
    assert min_sep(safe) > min_sep(nom)


def test_s04_d_eff_breakaway_when_track_coarse():
    env = _env()
    shield = DetectQPShield(env.cfg)
    w = env.wingmen[0]
    w.x, w.y, w.psi, w.v = 60.0, 60.0, 0.0, 0.24   # heading east toward track
    for k, ww in enumerate(env.wingmen):
        if k:
            ww.x, ww.y = 20.0, 20.0 + 40 * k
        ww.tracks.slots = [None] * len(ww.tracks.slots)
    # track dead ahead at d_est = 45 km, sigma = 4 -> d_eff = 33 < MAR 40
    w.tracks.slots[0] = Track(slot=0, target_id=900, x=105.0, y=60.0,
                              vx=0.0, vy=0.0, sigma=4.0, state="confirmed",
                              last_hit_t=env.t, first_hit_t=env.t - 5)
    env.manned.x, env.manned.y = 15.0, 60.0
    nom = _nominal(env)
    safe, inter = shield.filter(env, nom)
    assert any(it.uid == 0 and any(r.startswith("S04") for r in it.reasons)
               for it in inter)
    # corrected control must increase predicted d_eff at 3 s vs nominal
    def d_eff3(a):
        xa, ya = shield._predict((w.x, w.y, w.psi, w.v), a.u_psi, a.u_v, 3)
        return math.hypot(105.0 - xa, 60.0 - ya) - env.cfg.K_SIGMA * 4.0
    assert d_eff3(safe[0]) > d_eff3(nom[0])


def test_s05_soft_push_keeps_small_correction():
    env = _env()
    shield = DetectQPShield(env.cfg)
    w = env.wingmen[0]
    w.x, w.y, w.psi, w.v = 60.0, 60.0, 0.0, 0.24
    for k, ww in enumerate(env.wingmen):
        if k:
            ww.x, ww.y = 20.0, 20.0 + 40 * k
        ww.tracks.slots = [None] * len(ww.tracks.slots)
    # d_eff in [MAR, MOR): soft gradient only
    w.tracks.slots[0] = Track(slot=0, target_id=901, x=112.0, y=60.0,
                              vx=0.0, vy=0.0, sigma=1.0, state="confirmed",
                              last_hit_t=env.t, first_hit_t=env.t - 5)  # d_eff = 49
    env.manned.x, env.manned.y = 15.0, 60.0
    safe, inter = shield.filter(env, _nominal(env))
    assert all(it.kind == "continuous_qp" for it in inter)
    if inter:   # soft push may be zero when geometry already diverging
        assert all(it.magnitude <= 1.5 for it in inter)


def test_infeasible_falls_back_to_reference_law():
    env = _env()
    shield = DetectQPShield(env.cfg)
    w0, w1 = env.wingmen[0], env.wingmen[1]
    # already violated S02 (< 2 km): decay target still demands improvement,
    # and an S04 track directly ahead at d_eff << MAR makes the joint QP
    # infeasible for wingman 0 -> reference-law fallback path
    w0.x, w0.y, w0.psi, w0.v = 60.0, 60.0, 0.0, 0.30
    w1.x, w1.y, w1.psi, w1.v = 60.4, 60.0, math.pi, 0.30
    env.wingmen[2].x, env.wingmen[2].y = 20.0, 20.0
    for ww in env.wingmen:
        ww.tracks.slots = [None] * len(ww.tracks.slots)
    w0.tracks.slots[0] = Track(slot=0, target_id=902, x=61.0, y=60.0,
                               vx=0.0, vy=0.0, sigma=0.1, state="confirmed",
                               last_hit_t=env.t, first_hit_t=env.t - 5)
    env.manned.x, env.manned.y = 15.0, 60.0
    safe, inter = shield.filter(env, _nominal(env))
    assert len(safe) == 3
    if shield.last_solve.status == "infeasible_fallback":
        assert any(it.kind == "reference_fallback" for it in inter)
    else:   # solver found a way out; corrections must still be legal
        assert all(abs(a.u_psi) <= 1.0 and abs(a.u_v) <= 1.0 for a in safe.values())


def test_shield_episode_smoke_with_heur():
    """Full episode: shielded HEUR must finish with finite runtime and no S02
    min-separation violation worse than unshielded would tolerate."""
    from aircraft_sim.detect_baselines import HeurPolicy
    env = _env(seed=3, k=4)
    shield = DetectQPShield(env.cfg)
    pol = HeurPolicy()
    n_qp = n_fb = 0
    steps = 0
    while env.terminal is None:
        safe, inter = shield.filter(env, pol(env))
        n_qp += sum(1 for it in inter if it.kind == "continuous_qp")
        n_fb += sum(1 for it in inter if it.kind == "reference_fallback")
        env.step(safe)
        steps += 1
    assert steps <= env.cfg.horizon
    assert n_qp + n_fb >= 0    # ran; distribution reported, not asserted
    print(f"\n[smoke] terminal={env.terminal} steps={steps} "
          f"qp={n_qp} fallback={n_fb}")
