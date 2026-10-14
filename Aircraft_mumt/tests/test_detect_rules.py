"""Unit tests for detect_rules: 17 predicates, layered penalty, pair table."""
import math

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.detect_rules import (
    DETECT_CONFLICT_PAIRS, HARD_RULES, RULE_LAYER, DetectRuleMonitor, is_hard_rule,
)


def _blue(x=50.0, y=60.0, psi=0.0, v=0.25, **kw):
    base = dict(
        x=x, y=y, psi=psi, v=v,
        role="SEARCH", role_age=10.0, role_age_prev=10.0, role_changed=False,
        gear=60.0, axis_deg=0.0, subzone=1,
        link_degraded=False, uplink_loss=0.0,
        coverage_max_age=5.0, cover_area_rate=8.0,
        tracks={}, pointer_target=None, requests=[],
        handoff_attempted=False, gear_switch_denied=False, owned=[],
    )
    base.update(kw)
    return base


def _truth(cfg, blue=None, red=None, **kw):
    t = dict(
        t=100.0, step=100,
        blue=blue if blue is not None else [_blue(x=50.0, y=y) for y in (20, 60, 100)] + [_blue(x=10.0, y=60.0)],
        red=red if red is not None else [
            {"uid": 0, "x": 150.0, "y": 30.0, "judged": False},
            {"uid": 1, "x": 150.0, "y": 90.0, "judged": False},
        ],
        responsibility={}, handoff_history={}, c04_breach={}, m03_stall={},
        raw_actions={},
    )
    t.update(kw)
    return t


def test_clean_truth_has_no_violations():
    cfg = DetectSimConfig()
    rep = DetectRuleMonitor(cfg).evaluate_truth(_truth(cfg))
    assert rep.violations == []
    assert rep.hard_safe


def test_s02_predicted_separation():
    cfg = DetectSimConfig()
    # wingman 0 at (50,60) heading east; wingman 1 parked 1 km ahead -> d_min 1 < 2
    blue = [_blue(x=50.0, y=60.0, psi=0.0), _blue(x=51.0, y=60.0, psi=0.0),
            _blue(x=50.0, y=110.0), _blue(x=10.0, y=60.0)]
    rep = DetectRuleMonitor(cfg).evaluate_truth(_truth(cfg, blue=blue))
    assert "S02:0-1" in rep.violations
    assert math.isclose(rep.margins["S02:0-1"], 1.0 - cfg.D_SAFE, abs_tol=1e-6)
    assert not rep.hard_safe and is_hard_rule("S02:0-1")


def test_s04_mar_and_s05_mor_bands():
    cfg = DetectSimConfig()
    red = [{"uid": 0, "x": 85.0, "y": 60.0, "judged": False},   # d=35 -> MAR
           {"uid": 1, "x": 105.0, "y": 60.0, "judged": False}]  # d=55 -> MOR only
    blue = [_blue(x=50.0, y=60.0), _blue(x=20.0, y=20.0), _blue(x=20.0, y=100.0), _blue(x=5.0, y=60.0)]
    rep = DetectRuleMonitor(cfg).evaluate_truth(_truth(cfg, blue=blue, red=red))
    assert "S04:0-0" in rep.violations
    assert "S05:0-0" not in rep.violations      # MAR pair not double-counted as MOR
    assert "S05:0-1" in rep.violations
    assert math.isclose(rep.margins["S05:0-1"], 55.0 - cfg.MOR, abs_tol=1e-6)


def test_c03_rate_limit_counts_window():
    cfg = DetectSimConfig()
    truth = _truth(cfg, handoff_history={3: [80.0, 95.0]})   # two within 30 s of t=100
    rep = DetectRuleMonitor(cfg).evaluate_truth(truth)
    assert "C03:3" in rep.violations
    truth2 = _truth(cfg, handoff_history={3: [60.0, 95.0]})  # 40 s apart -> clean
    assert "C03:3" not in DetectRuleMonitor(cfg).evaluate_truth(truth2).violations


def test_m02_owned_stale_track():
    cfg = DetectSimConfig()
    blue = [_blue(tracks={7: {"age": 3.0, "state": "confirmed", "sigma": 1.0}}, owned=[7]),
            _blue(y=110.0), _blue(y=10.0), _blue(x=10.0)]
    red = [{"uid": 7, "x": 90.0, "y": 60.0, "judged": False}]  # tracking zone
    rep = DetectRuleMonitor(cfg).evaluate_truth(
        _truth(cfg, blue=blue, red=red, responsibility={7: 0}))
    assert "M02:0-7" in rep.violations
    assert math.isclose(rep.margins["M02:0-7"], -1.0, abs_tol=1e-6)


def test_m01_only_counts_seen_targets():
    cfg = DetectSimConfig()
    red = [{"uid": 7, "x": 130.0, "y": 60.0, "judged": False}]  # detection zone, subzone 1
    unseen = DetectRuleMonitor(cfg).evaluate_truth(_truth(cfg, red=red))
    assert not any(k.startswith("M01") for k in unseen.violations)
    blue = [_blue(x=50.0, y=20.0), _blue(x=100.0, y=60.0,
            tracks={7: {"age": 7.0, "state": "coast", "sigma": 2.0}}),
            _blue(x=50.0, y=100.0), _blue(x=10.0, y=60.0)]
    rep = DetectRuleMonitor(cfg).evaluate_truth(_truth(cfg, blue=blue, red=red))
    assert "M01:1-7" in rep.violations


def test_i02_request_on_unconfirmed():
    cfg = DetectSimConfig()
    # V2.7: tentative (>=2 hits) is a legitimate request basis -- the request
    # leg is what reaches "confirmed"; dishonest = no track / single blip
    blue = [_blue(tracks={7: {"age": 1.0, "state": "tentative", "sigma": 1.0}},
                  requests=[7]),
            _blue(y=110.0), _blue(y=10.0), _blue(x=10.0)]
    rep = DetectRuleMonitor(cfg).evaluate_truth(_truth(cfg, blue=blue))
    assert "I02:0-7" not in rep.violations
    blue[0] = _blue(tracks={7: {"age": 1.0, "state": "detected", "sigma": 4.0}},
                    requests=[7])
    rep = DetectRuleMonitor(cfg).evaluate_truth(_truth(cfg, blue=blue))
    assert "I02:0-7" in rep.violations and not rep.hard_safe


def test_f01_active_and_c05_violation():
    cfg = DetectSimConfig()
    blue = [_blue(link_degraded=True, uplink_loss=0.5, handoff_attempted=True),
            _blue(y=110.0), _blue(y=10.0), _blue(x=10.0)]
    rep = DetectRuleMonitor(cfg).evaluate_truth(_truth(cfg, blue=blue))
    assert "F01:0" in rep.active
    assert "C05:0" in rep.violations


def test_m04_cutoff_unjudged():
    cfg = DetectSimConfig()
    red = [{"uid": 0, "x": 39.0, "y": 30.0, "judged": False}]
    rep = DetectRuleMonitor(cfg).evaluate_truth(_truth(cfg, red=red))
    assert "M04:0" in rep.violations
    red2 = [{"uid": 0, "x": 39.0, "y": 30.0, "judged": True}]   # judged -> clean
    assert "M04:0" not in DetectRuleMonitor(cfg).evaluate_truth(_truth(cfg, red=red2)).violations


def test_penalty_layer_sums():
    cfg = DetectSimConfig()
    red = [{"uid": 7, "x": 90.0, "y": 60.0, "judged": False}]
    blue = [_blue(x=50.0, y=60.0, psi=0.0,
                  tracks={7: {"age": 3.0, "state": "confirmed", "sigma": 1.0}}, owned=[7]),
            _blue(x=51.0, y=60.0), _blue(x=50.0, y=110.0), _blue(x=10.0, y=60.0)]
    rep = DetectRuleMonitor(cfg).evaluate_truth(
        _truth(cfg, blue=blue, red=red, responsibility={7: 0}))
    mon = DetectRuleMonitor(cfg)
    # S02:0-1 (P0=10) + M02:0-7 (P3=5)
    # V2.7: S05:0-7 / S04:1-7 still REPORT as violations but are exempt from
    # w8 (proximity flows through the dedicated w5/w6 channels instead)
    assert "S05:0-7" in rep.violations and "S04:1-7" in rep.violations
    expected = cfg.W8_LAYER[0] + cfg.W8_LAYER[3]
    assert math.isclose(mon.penalty(rep), expected, abs_tol=1e-9)


def test_conflict_pairs_table():
    assert len(DETECT_CONFLICT_PAIRS) == 13
    fams = {f for pair in DETECT_CONFLICT_PAIRS for f in pair}
    assert fams <= set(RULE_LAYER)
    assert frozenset(("S04", "M02")) in DETECT_CONFLICT_PAIRS
    assert frozenset(("M05", "E03")) in DETECT_CONFLICT_PAIRS  # gate-6 #13
    assert sum(1 for p in DETECT_CONFLICT_PAIRS if "M02" in p) == 9  # hub node


def test_penalty_w8_scale_training_only():
    cfg = DetectSimConfig()
    red = [{"uid": 7, "x": 90.0, "y": 60.0, "judged": False}]
    blue = [_blue(x=50.0, y=60.0, psi=0.0,
                  tracks={7: {"age": 3.0, "state": "confirmed", "sigma": 1.0}}, owned=[7]),
            _blue(x=51.0, y=60.0), _blue(x=50.0, y=110.0), _blue(x=10.0, y=60.0)]
    rep = DetectRuleMonitor(cfg).evaluate_truth(
        _truth(cfg, blue=blue, red=red, responsibility={7: 0}))
    full = DetectRuleMonitor(cfg).penalty(rep)
    from dataclasses import replace as _replace
    cfg = _replace(cfg, W8_SCALE=0.2)   # frozen dataclass: per-episode override
    scaled = DetectRuleMonitor(cfg).penalty(rep)
    import math as _m
    assert _m.isclose(scaled, 0.2 * full, abs_tol=1e-9)
    assert "S02:0-1" in rep.violations   # violation REPORTING untouched


def _m05_truth(cfg, **over):
    """Confirmed track 7 in the tracking zone (x=90), blue0 nearest at (50,60)."""
    red = [{"uid": 7, "x": 90.0, "y": 60.0, "judged": False}]
    blue = [_blue(x=50.0, y=60.0,
                  tracks={7: {"age": 1.0, "state": "tentative", "sigma": 1.0}}),
            _blue(x=50.0, y=20.0), _blue(x=50.0, y=100.0), _blue(x=10.0, y=60.0)]
    base = dict(blue=blue, red=red,
                tz_track_since={7: 90.0})      # t=100 -> elapsed 10 > grace 5
    base.update(over)
    return _truth(cfg, **base)


def test_m05_unowned_confirmed_track_penalised():
    cfg = DetectSimConfig()
    rep = DetectRuleMonitor(cfg).evaluate_truth(_m05_truth(cfg))
    assert "M05:0-7" in rep.violations          # nearest wingman 0 carries it
    assert math.isclose(rep.margins["M05:0-7"], cfg.M05_GRACE - 10.0)
    assert "M05" in RULE_LAYER and RULE_LAYER["M05"] == 4


def test_m05_exemptions():
    cfg = DetectSimConfig()
    mon = DetectRuleMonitor(cfg)
    # owned -> clean
    rep = mon.evaluate_truth(_m05_truth(cfg, responsibility={7: 0}))
    assert not any(k.startswith("M05") for k in rep.violations)
    # within grace (elapsed 2 s) -> clean
    rep = mon.evaluate_truth(_m05_truth(cfg, tz_track_since={7: 98.0}))
    assert not any(k.startswith("M05") for k in rep.violations)
    # detection zone (x=150) -> clean
    rep = mon.evaluate_truth(_m05_truth(cfg, red=[{"uid": 7, "x": 150.0,
                                                   "y": 60.0, "judged": False}]))
    assert not any(k.startswith("M05") for k in rep.violations)
    # judged -> clean
    rep = mon.evaluate_truth(_m05_truth(cfg, red=[{"uid": 7, "x": 90.0,
                                                   "y": 60.0, "judged": True}]))
    assert not any(k.startswith("M05") for k in rep.violations)
    # never tracked in zone (no tz_track_since entry) -> clean
    rep = mon.evaluate_truth(_m05_truth(cfg, tz_track_since={}))
    assert not any(k.startswith("M05") for k in rep.violations)
