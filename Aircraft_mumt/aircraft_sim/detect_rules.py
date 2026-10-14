"""Rule library for the MUMT cooperative-detection scenario (plan V2.4 §8).

18 predicates + 13 conflict pairs.  Every rule is evaluated on a *truth*
dictionary produced by ``detect_env`` (the referee ledger, plan §1.3: rewards
and termination use truth; the shield only ever sees track-file views, which
is a P2 module).  Rule keys follow the legacy ``"S02:0-1"`` convention so
``rule_graph.RuleAgentDependencyGraph`` consumes them unchanged.

Layer map (plan §8 preamble, P0..P5) and penalty weights (plan §7.3 w8,
``DetectSimConfig.W8_LAYER``):

    P0 hard safety      S01 S02 S03                       penalty 10.0
    P1 consistency      S04 M04 I02                       penalty  7.5
    P2 fault/physical   S05 E01 F01                       penalty 10.0
    P3 coordination     C01 C03 C05 M02                   penalty  5.0
    P4 mission cont.    C02 C04 M01 M03 I03 E02 M05       penalty  2.5
    P5 weak efficiency  E03                               penalty  1.25

Truth-dict contract (all lists index wingmen 0..2, manned node = index 3)::

    truth = {
      "t": float, "step": int,
      "blue": [ {x, y, psi, v,                       # kinematics (km, rad, km/s)
                 role, role_age,                     # DetectRole, s since change
                 gear, axis_deg,                     # radar sector state
                 link_degraded, uplink_loss,         # F01 window loss rate
                 coverage_max_age,                   # s, own subzone worst cell
                 cover_area_rate,                    # E03: km^2/km over 60 s window
                 tracks: {tid: {"age","state","sigma"}},   # own track file view
                 pointer_target,                     # tid designated by pointer or None
                 requests: [tid, ...],               # responsibility requests this step
                 handoff_attempted,                  # tried a responsibility change
                 gear_switch_denied,                 # tried switching mid-frame (E01)
                 role_changed, role_age_prev,        # switched this step; dwell before switch
                 subzone,                            # responsibility subzone index 0..2
                 owned: [tid, ...],                  # current responsibility set
                } x3, plus manned dict at index 3 (kinematics only) ],
      "red":  [ {uid, x, y, judged} ... ],
      "responsibility": {tid: owner_index},
      "handoff_history": {tid: [t, ...]},            # non-exempt changes only (C03)
      "c04_breach": {(i, tid): since_t},             # capability-gate breach timers
      "m03_stall": {(i, tid): stalled_seconds},      # zero-progress timers (M03)
      "raw_actions": {i: DetectAction},              # pre-clip requests (S01)
    }
"""
from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple

from .detect_config import DetectSimConfig, ZONE_DETECTION, ZONE_TRACKING
from .detect_models import DetectRole


# --- static tables ---------------------------------------------------------

RULE_LAYER: Dict[str, int] = {
    "S01": 0, "S02": 0, "S03": 0,
    "S04": 1, "M04": 1, "I02": 1,
    "S05": 2, "E01": 2, "F01": 2,
    "C01": 3, "C03": 3, "C05": 3, "M02": 3,
    "C02": 4, "C04": 4, "M01": 4, "M03": 4, "I03": 4, "E02": 4, "M05": 4,
    "E03": 5,
}

#: Rules whose violation is a hard (non-relaxable) event.  S04/M04 stay out
#: of the QP-hard set by design: S04 is "P1 soft, heavy penalty channel" and
#: M04 is terminal, not a QP constraint (plan §8.1/§8.3 attributes).
HARD_RULES = frozenset({"S01", "S02", "S03", "C01", "I02", "E01", "F01"})

#: The 13 conflict pairs of plan §8.5 (family level, symmetric; #13 M05-E03
#: added at gate-6: forced ownership -> narrow-gear precision tracking ->
#: lower cover-area rate, the "steady vs moving" core trade-off).
DETECT_CONFLICT_PAIRS = frozenset({
    frozenset(("S04", "M02")),   # 1  signature: quality vs MAR line
    frozenset(("S05", "M02")),   # 2  MOR band geometry management
    frozenset(("C01", "M02")),   # 3  ownership arbitration vs update pressure
    frozenset(("C03", "M02")),   # 4  handoff rate limit vs better geometry
    frozenset(("C02", "M01")),   # 5  coverage vs update time budget
    frozenset(("M01", "M02")),   # 6  one radar, two zones
    frozenset(("S02", "M02")),   # 7  separation vs pointing convergence
    frozenset(("S03", "M02")),   # 8  boundary vs forward leaning
    frozenset(("I02", "M02")),   # 9  honest report vs credit
    frozenset(("I03", "M02")),   # 10 stale track vs this-frame update
    frozenset(("M04", "S04")),   # 11 endgame: press MAR for the verdict
    frozenset(("C01", "C03")),   # 12 mis-assignment cannot be fixed at once
    frozenset(("M05", "E03")),   # 13 forced ownership vs anti-loiter (gate-6)
})


def rule_family(rule_key: str) -> str:
    return rule_key.split(":", 1)[0]


def is_hard_rule(rule_key: str) -> bool:
    return rule_family(rule_key) in HARD_RULES


@dataclass
class DetectRuleReport:
    violations: List[str] = field(default_factory=list)
    margins: Dict[str, float] = field(default_factory=dict)
    active: List[str] = field(default_factory=list)      # condition-true, not violated (F01)
    hard_safe: bool = True

    def by_family(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for key in self.violations:
            out.setdefault(rule_family(key), []).append(key)
        return out


class DetectRuleMonitor:
    """Predicate evaluation over the referee truth dict (plan §8)."""

    def __init__(self, cfg: DetectSimConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    def penalty(self, report: DetectRuleReport) -> float:
        """w8 layered penalty sum (positive number; env subtracts it).

        V2.7: S04/S05 are EXEMPT from w8 -- proximity already flows through
        the dedicated w5/w6 gradient channels (plan §7.3); counting them in
        w8 as well double-charged the same event at ~4 orders of magnitude
        above the intended gradient (G2 dry-run finding).  The predicates
        still evaluate and report normally (spectrum/hard-safe intact)."""
        w = self.cfg.W8_LAYER
        return self.cfg.W8_SCALE * sum(
            w[RULE_LAYER[rule_family(k)]] for k in report.violations
            if rule_family(k) not in ("S04", "S05"))

    # ------------------------------------------------------------------
    def evaluate_truth(self, truth: dict, raw_actions=None) -> DetectRuleReport:
        cfg = self.cfg
        rep = DetectRuleReport()
        vio, mar, act = rep.violations, rep.margins, rep.active
        blue, red = truth["blue"], truth["red"]
        t = truth["t"]
        resp = truth["responsibility"]
        wingmen = range(3)

        def add(key: str, margin: float):
            vio.append(key)
            mar[key] = float(margin)

        # ---- S01 envelope: pre-clip request exceeded |u| <= 1 -------------
        raw = truth.get("raw_actions") or {}
        for i in wingmen:
            a = raw.get(i)
            if a is None:
                continue
            excess = max(abs(a.u_psi), abs(a.u_v), abs(a.u_xi)) - 1.0
            mar[f"S01:{i}"] = -excess
            if excess > 1e-9:
                add(f"S01:{i}", -excess)

        # ---- S02 predicted 3 s min separation (all blue pairs, manned = 3)
        n_blue = len(blue)
        for i in range(n_blue):
            for k in range(i + 1, n_blue):
                bi, bk = blue[i], blue[k]
                d_min = math.inf
                for tau in (0.0, 1.0, 2.0, cfg.D_SAFE_LOOKAHEAD):
                    dx = (bi["x"] + bi["v"] * math.cos(bi["psi"]) * tau) \
                       - (bk["x"] + bk["v"] * math.cos(bk["psi"]) * tau)
                    dy = (bi["y"] + bi["v"] * math.sin(bi["psi"]) * tau) \
                       - (bk["y"] + bk["v"] * math.sin(bk["psi"]) * tau)
                    d_min = min(d_min, math.hypot(dx, dy))
                key = f"S02:{i}-{k}"
                mar[key] = d_min - cfg.D_SAFE
                if d_min < cfg.D_SAFE:
                    add(key, d_min - cfg.D_SAFE)

        # ---- S03 airspace boundary (wingmen; manned is scripted inside) ---
        for i in wingmen:
            b = blue[i]
            clear = min(b["x"], cfg.L_X - b["x"], b["y"], cfg.L_Y - b["y"])
            px = b["x"] + b["v"] * math.cos(b["psi"]) * cfg.BOUNDARY_PREDICT
            py = b["y"] + b["v"] * math.sin(b["psi"]) * cfg.BOUNDARY_PREDICT
            out_now = clear < 0.0
            out_soon = not (0.0 <= px <= cfg.L_X and 0.0 <= py <= cfg.L_Y)
            key = f"S03:{i}"
            mar[key] = clear
            if out_now or out_soon:
                add(key, min(clear, 0.0) if out_now else -0.0)

        # ---- S04 MAR / S05 MOR: truth-distance referee (plan §1.3) --------
        # V2.6: wingmen only -- the scripted manned node is excluded from the
        # approach-cost ledger (deployment-guaranteed safety, non-learning).
        for i in wingmen:
            b = blue[i]
            for r in red:
                d = math.hypot(b["x"] - r["x"], b["y"] - r["y"])
                k4, k5 = f"S04:{i}-{r['uid']}", f"S05:{i}-{r['uid']}"
                mar[k4] = d - cfg.MAR
                mar[k5] = d - cfg.MOR
                if d < cfg.MAR:
                    add(k4, d - cfg.MAR)
                elif d < cfg.MOR:
                    add(k5, d - cfg.MOR)

        # ---- C01 unique ownership ----------------------------------------
        owners: Dict[int, List[int]] = {}
        for tid, owner in resp.items():
            owners.setdefault(tid, []).append(owner)
        for tid, olist in owners.items():
            if len(olist) > 1:
                add(f"C01:{tid}", 1 - len(olist))

        # ---- C02 subzone coverage period ----------------------------------
        for i in wingmen:
            age = blue[i]["coverage_max_age"]
            key = f"C02:{i}"
            mar[key] = cfg.T_SCAN_REQ - age
            if age > cfg.T_SCAN_REQ:
                add(key, cfg.T_SCAN_REQ - age)

        # ---- C03 handoff rate limit (non-exempt changes, 30 s window) -----
        for tid, hist in truth["handoff_history"].items():
            recent = [ht for ht in hist if t - ht <= cfg.HANDOFF_MIN_INTERVAL]
            key = f"C03:{tid}"
            mar[key] = 1 - len(recent)
            if len(recent) > 1:
                add(key, 1 - len(recent))

        # ---- C04 capability gate (response window expired) ----------------
        for (i, tid), since in truth["c04_breach"].items():
            key = f"C04:{i}-{tid}"
            mar[key] = -(t - since)
            if t - since > cfg.C04_RESPONSE:
                add(key, -(t - since))

        # ---- C05 degraded link: keep assignment, no changes ---------------
        for i in wingmen:
            if blue[i]["link_degraded"]:
                if blue[i]["handoff_attempted"]:
                    add(f"C05:{i}", -1.0)

        # ---- M01 detection-zone 5 s standard (own subzone, seen targets) --
        # Engineering reading (registered): "at least one look every 5 s"
        # applies to targets the wingman has seen at least once (a track
        # exists, incl. coast); never-seen targets are C02's job, so the
        # penalty stays bounded before first contact.
        for i in wingmen:
            y0, y1 = cfg.subzone_bounds(blue[i]["subzone"])
            for r in red:
                if cfg.zone_of(r["x"]) != ZONE_DETECTION or not (y0 <= r["y"] < y1):
                    continue
                tr = blue[i]["tracks"].get(r["uid"])
                if tr is None:
                    continue
                key = f"M01:{i}-{r['uid']}"
                mar[key] = cfg.AGE_STD_DETECT - tr["age"]
                if tr["age"] >= cfg.AGE_STD_DETECT:
                    add(key, cfg.AGE_STD_DETECT - tr["age"])

        # ---- M02 tracking-zone 2 s standard on owned targets --------------
        for tid, owner in resp.items():
            r = next((rr for rr in red if rr["uid"] == tid), None)
            if r is None or cfg.zone_of(r["x"]) != ZONE_TRACKING:
                continue
            tr = blue[owner]["tracks"].get(tid)
            age = tr["age"] if tr else math.inf
            key = f"M02:{owner}-{tid}"
            mar[key] = cfg.AGE_STD_TRACK - min(age, cfg.T_SCAN_REQ)
            if age >= cfg.AGE_STD_TRACK:
                add(key, cfg.AGE_STD_TRACK - min(age, cfg.T_SCAN_REQ))

        # ---- M03 stable-window progress must advance ----------------------
        for (i, tid), stalled in truth["m03_stall"].items():
            key = f"M03:{i}-{tid}"
            mar[key] = cfg.STABLE_WINDOW - stalled
            if stalled > cfg.STABLE_WINDOW:
                add(key, cfg.STABLE_WINDOW - stalled)

        # ---- M05 tracking-zone tracked target must be OWNED (gate-6) ------
        # Closes the free-rider loophole found at gate-6: M02/M03 bind only
        # OWNED targets, so "never request ownership" was locally rational
        # (loiter + wide-scan, pay only the E03 tax).  Real C2 tasks every
        # tracked contact; this predicate makes that duty explicit.
        # Clock = tz_track_since (any track state counts -- the v1 "confirmed
        # only" variant was itself gameable by staying in tentative forever;
        # uid-keyed so drop/re-detect cycling does not reset it).
        # Grace M05_GRACE (5 s, aligned with the C04 window); attributed to
        # the positionally nearest wingman; judged targets exempt.
        conf_since = truth.get("tz_track_since", {})
        for r in red:
            tid = r["uid"]
            if r["judged"] or cfg.zone_of(r["x"]) != ZONE_TRACKING \
                    or tid in resp:
                continue
            t0 = conf_since.get(tid)
            if t0 is None:
                continue
            elapsed = t - t0
            owner = min(wingmen, key=lambda ii: (blue[ii]["x"] - r["x"]) ** 2
                        + (blue[ii]["y"] - r["y"]) ** 2)
            key = f"M05:{owner}-{tid}"
            mar[key] = cfg.M05_GRACE - elapsed
            if elapsed > cfg.M05_GRACE:
                add(key, cfg.M05_GRACE - elapsed)

        # ---- M04 cutoff endgame (terminal, hard) --------------------------
        for r in red:
            key = f"M04:{r['uid']}"
            mar[key] = r["x"] - cfg.X_CUTOFF
            if r["x"] < cfg.X_CUTOFF and not r["judged"]:
                add(key, r["x"] - cfg.X_CUTOFF)

        # ---- I02 honest reporting: request implies a correlated basis -----
        # V2.7: requesting with a TENTATIVE track (>=2 hits) is legitimate --
        # the request leg itself is what narrows the gear and reaches
        # "confirmed" (confirmation needs <=1.6 s update interval, i.e. a
        # narrow gear the requester only gets in TRACK role).  Dishonest =
        # requesting with no track or a single-hit "detected" blip.
        for i in wingmen:
            for tid in blue[i]["requests"]:
                tr = blue[i]["tracks"].get(tid)
                state = tr["state"] if tr else "none"
                if state in ("none", "detected"):
                    add(f"I02:{i}-{tid}", -1.0)

        # ---- I03 no stale basis: pointing/request on age > 8 s tracks -----
        for i in wingmen:
            tid = blue[i]["pointer_target"]
            if tid is None:
                continue
            tr = blue[i]["tracks"].get(tid)
            if tr is not None and tr["age"] > cfg.AGE_DEGRADE:
                add(f"I03:{i}-{tid}", cfg.AGE_DEGRADE - tr["age"])

        # ---- E01 gear dwell: switch denied mid-frame ----------------------
        for i in wingmen:
            if blue[i]["gear_switch_denied"]:
                add(f"E01:{i}", -1.0)

        # ---- E02 role dwell -----------------------------------------------
        for i in wingmen:
            if blue[i]["role_changed"] and 0.0 < blue[i]["role_age_prev"] < cfg.T_DWELL:
                add(f"E02:{i}", blue[i]["role_age_prev"] - cfg.T_DWELL)

        # ---- E03 anti-loiter ----------------------------------------------
        for i in wingmen:
            rate = blue[i]["cover_area_rate"]
            key = f"E03:{i}"
            mar[key] = rate - cfg.E03_AREA_PER_KM
            if rate < cfg.E03_AREA_PER_KM:
                add(key, rate - cfg.E03_AREA_PER_KM)

        # ---- F01 link degradation (condition rule -> active, C05 trigger) -
        for i in wingmen:
            if blue[i]["link_degraded"]:
                act.append(f"F01:{i}")
            mar[f"F01:{i}"] = cfg.LINK_DROP_RATE - blue[i]["uplink_loss"]

        rep.hard_safe = not any(is_hard_rule(k) for k in rep.violations)
        return rep
