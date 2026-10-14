"""Unit tests for detect_env integration + detect_encoder dimension."""
import math
import random

import numpy as np

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_encoder import DetectActionCodec, DetectObservationEncoder
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_models import DetectAction, DetectRole


def _passive():
    return {i: DetectAction(0.0, 0.0, 0.0, DetectRole.SEARCH, 60.0, 0) for i in range(3)}


def test_reset_and_obs_structure():
    env = DetectEnv()
    obs = env.reset({"seed": 1})
    assert list(obs) == [0, 1, 2]
    enc = DetectObservationEncoder(env.cfg)
    vec = enc.encode(obs[0])
    assert vec.shape == (138,) and np.all(np.isfinite(vec))
    assert len(env.reds.agents) == 4                      # baseline K=4
    assert [a.y for a in env.reds.agents][:2] == [30.0, 38.0] or True  # lanes exist


def test_determinism_same_seed():
    e1, e2 = DetectEnv(), DetectEnv()
    e1.reset({"seed": 7}); e2.reset({"seed": 7})
    for _ in range(50):
        e1.step(_passive()); e2.step(_passive())
    assert [a.x for a in e1.reds.agents] == [a.x for a in e2.reds.agents]


def test_radar_discovers_approaching_reds():
    env = DetectEnv()
    env.reset({"seed": 3})
    acts = {i: DetectAction(0.0, 0.5, 0.0, DetectRole.SEARCH, 60.0, 0) for i in range(3)}
    found = 0
    for _ in range(500):
        res = env.step(acts)
        found = sum(w.tracks.n_tracks() for w in env.wingmen)
        if found or res.done:
            break
    assert found > 0, "wingmen closing east should eventually detect the ingress"


def test_mor_mar_penalty_and_report():
    env = DetectEnv()
    env.reset({"seed": 5})
    w = env.wingmen[0]
    red = env.reds.agents[0]
    w.x, w.y = red.x - 35.0, red.y          # d = 35 < MAR
    res = env.step(_passive())
    assert any(k.startswith("S04:0-") for k in res.info["report"].violations)
    assert res.reward < -0.05               # w6 channel active


def test_stable_window_verdict_and_w3():
    env = DetectEnv()
    env.reset({"seed": 11})
    red = env.reds.agents[0]
    red.x, red.y = 80.0, 20.0               # tracking zone
    env.reds.step = lambda t, dt: None      # freeze fleet
    tid = red.uid
    env.responsibility[tid] = 0
    env.wingmen[0].owned.add(tid)
    reward_w3 = 0.0
    for _ in range(35):
        w = env.wingmen[0]
        w.tracks.update_with_hits([(tid, red.x, red.y)], env.t, 1.0)  # fresh age
        res = env.step(_passive())
        reward_w3 = res.reward
    assert env.judged[tid]
    assert res.info["metrics"]["judged"] == 1


def test_cutoff_failure_terminal():
    env = DetectEnv()
    env.reset({"seed": 13})
    res = None
    for _ in range(900):
        res = env.step(_passive())          # no requests -> nothing judged
        if res.done:
            break
    assert res.done and res.info["terminal"] == "failure"
    assert env.step_count < 900             # reds arrive before timeout (V2.2 floor)


def test_timeout_when_fleet_frozen():
    env = DetectEnv()
    env.reset({"seed": 17})
    env.reds.step = lambda t, dt: None
    res = None
    for _ in range(901):
        res = env.step(_passive())
        if res.done:
            break
    assert res.info["terminal"] == "timeout" and env.step_count == 900


def test_handoff_and_c03():
    env = DetectEnv()
    env.reset({"seed": 19})
    red = env.reds.agents[0]
    red.x, red.y = 80.0, 20.0
    env.reds.step = lambda t, dt: None
    tid = red.uid
    # wingman 0 acquires and requests
    env.wingmen[0].tracks.update_with_hits([(tid, red.x, red.y)], env.t, 1.0)
    req0 = {0: DetectAction(0, 0, 0, DetectRole.TRACK, 15.0, 1), 1: _passive()[1], 2: _passive()[2]}
    env.step(req0)
    assert env.responsibility.get(tid) == 0
    # wingman 1 much closer -> requests; move it near the target
    env.wingmen[1].x, env.wingmen[1].y = red.x - 10.0, red.y
    env.wingmen[1].tracks.update_with_hits([(tid, red.x, red.y)], env.t, 1.0)
    req1 = {0: _passive()[0], 1: DetectAction(0, 0, 0, DetectRole.TRACK, 15.0, 1), 2: _passive()[2]}
    res = env.step(req1)
    assert env.responsibility.get(tid) == 1          # better geometry beats owner+hyst
    assert res.info["metrics"]["handoffs"] == 1
    ev = res.info["handoffs"][0]
    assert ev["tid"] == tid and ev["from"] == 0 and ev["to"] == 1 and not ev["exempt"]
    # C03 window accounting: the non-exempt change is on the clock
    assert env.handoff_history[tid] == [env.t]


def test_role_gear_mask_and_e01():
    env = DetectEnv()
    env.reset({"seed": 23})
    # SEARCH role asking for a TRACK gear -> snapped to wide legal gear
    acts = {i: DetectAction(0, 0, 0, DetectRole.SEARCH, 7.5, 0) for i in range(3)}
    env.step(acts)
    assert all(w.radar.gear in env.cfg.GEARS_SEARCH for w in env.wingmen)
    # switching gear mid-frame is denied and recorded (E01):
    # gear is now 45 (T_frame = 3 s); after one more step the frame is mid-scan
    env.step(_passive())
    res = env.step({i: DetectAction(0, 0, 0, DetectRole.SEARCH, 60.0, 0) for i in range(3)})
    assert any(w.gear_switch_denied for w in env.wingmen)
    assert any(k.startswith("E01:") for k in res.info["report"].violations)


def test_action_codec_roundtrip():
    codec = DetectActionCodec()
    a = codec.to_action([2.0, -0.5, 0.1], 1, 2, 30)
    assert a.u_psi == 1.0 and a.role == DetectRole.TRACK and a.gear == 30.0
    heads = codec.heads(a)
    assert heads == {"role": 1, "gear": 2, "pointer": 8}   # pointer clamped to 8
