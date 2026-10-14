"""DetectEnv: MUMT cooperative-detection scenario environment (plan V2.4).

Integration of the P1 modules: 3 learning wingmen (2D point-mass, first-order
heading/speed response), 1 scripted manned CAP node (responsibility arbiter),
K scripted red aircraft.  Per 1 s decision step:

    mask/clip actions -> dynamics -> manned CAP -> red fleet
    -> radar dwell scans (+ coverage marking) -> track files
    -> responsibility arbitration (machine-speed channel, delay 0)
    -> stable-window verdicts -> link model -> rule monitor (truth)
    -> reward (plan §7.3) -> termination (plan §1.4) -> dual bookkeeping

Local-observation discipline (plan §1.3/§5.3): wingman observations contain
only own-sensor products, teammate kinematics and the link-delivered
assignment table / pointers; truth is used solely by the referee (reward,
metrics, termination) and by the arbitration fiction (registered below).

Registered engineering decisions (see _改动登记.md):
  D1 omega_max flat at cruise 3.0 deg/s; maneuver 6.0 deg/s held in reserve
     (plan §2 does not define the switch condition).
  D2 arbitration scores on truth positions (command-node fiction, as V1.23);
     the shield (P2) remains track-file based.
  D3 stable-window progress resets when the target leaves the tracking zone
     (strict reading of "a continuous 30 s window inside the zone").
  D4 never-seen targets do not trigger M01 (bounded early-episode penalty);
     coverage pressure on unseen areas is C02's job.
  D5 dual bookkeeping counts per red-second at min-distance pair (reward
     w5/w6 stay per-pair exact).
  D6 static subzone assignment (dynamic re-partitioning deferred, §6.2 note).
  D7 TRACK with pointer=0: free-track mode (points at best own confirmed
     track, no responsibility request); masked only when link-degraded
     requests are attempted.
"""
from collections import deque
from dataclasses import dataclass, field, replace
import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .detect_config import (A_MACH_KMS, DetectSimConfig, ZONE_DETECTION,
                            ZONE_TRACKING, ma_to_kms)
from .detect_models import DetectAction, DetectRole
from .detect_rules import DetectRuleMonitor
from .models import StepResult
from .radar import RadarState, wrap_pi
from .red_agent import RedFleet
from .track import TrackFile

CELL_KM = 4.0          # coverage raster cell edge (C02/E03), registered


@dataclass
class BlueWingman:
    idx: int
    x: float
    y: float
    psi: float                       # rad, east = 0
    v: float                         # km/s
    role: DetectRole = DetectRole.SEARCH
    role_age: float = 99.0
    role_age_prev: float = 99.0
    role_changed: bool = False
    radar: Optional[RadarState] = None
    tracks: Optional[TrackFile] = None
    subzone: int = 0
    owned: Set[int] = field(default_factory=set)
    # link
    uplink_window: deque = field(default_factory=lambda: deque(maxlen=10))
    link_degraded: bool = False
    # coverage raster (own subzone)
    last_scan: Optional[np.ndarray] = None
    # E03 sliding window (60 s): per-dwell new covered area km^2 and path km
    e03_window: deque = field(default_factory=lambda: deque(maxlen=120))
    # per-step flags for the rule monitor
    gear_switch_denied: bool = False
    handoff_attempted: bool = False
    requests: List[int] = field(default_factory=list)
    pointer_target: Optional[int] = None

    def pos(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass
class MannedNode:
    """Scripted command node on a fixed-anchor racetrack CAP (V2.6).

    Anchor (CAP_X, CAP_Y) in the entry band; legs run north-south.  No
    centroid following -- the command node holds a fixed station behind the
    threat axis, and it is excluded from MOR/MAR penalties and the dual
    bookkeeping (it is scripted, non-learning; its safety is guaranteed by
    deployment, not managed by the learning system).  S02 collision
    avoidance still includes it as an endpoint.
    """

    x: float = 15.0
    y: float = 50.0
    psi: float = math.pi / 2
    v: float = 0.7 * A_MACH_KMS
    wp_index: int = 0

    def pos(self) -> Tuple[float, float]:
        return self.x, self.y


class DetectEnv:
    """Gym-ish environment; duck-type compatible surface for P2/P3:

    ``reset(scenario) -> Dict[int, dict]``, ``step(actions) -> StepResult``,
    ``observe()``, attributes ``n`` (learning agents) and ``cfg``.
    """

    def __init__(self, cfg: Optional[DetectSimConfig] = None):
        self.cfg = cfg or DetectSimConfig()
        self.n = self.cfg.N_WINGMEN
        self.monitor = DetectRuleMonitor(self.cfg)
        self.rng = random.Random(self.cfg.seed)
        # coverage raster per subzone: x in [40,160) step 4, y per bounds
        self._cell_centres: List[np.ndarray] = []
        for s in range(self.cfg.N_SUBZONES):
            y0, y1 = self.cfg.subzone_bounds(s)
            xs = np.arange(self.cfg.X_CUTOFF + CELL_KM / 2, self.cfg.L_X, CELL_KM)
            ys = np.arange(y0 + CELL_KM / 2, y1, CELL_KM)
            gx, gy = np.meshgrid(xs, ys)
            self._cell_centres.append(np.stack([gx.ravel(), gy.ravel()], axis=1))
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, scenario: Optional[dict] = None) -> Dict[int, dict]:
        scenario = scenario or {}
        cfg = self.cfg
        if any(k in scenario for k in ("k_red", "mar", "mor", "handoff_min_interval", "uplink_loss_p")):
            overrides = {}
            if "k_red" in scenario:
                overrides["K_RED"] = int(scenario["k_red"])
            if "mar" in scenario:
                overrides["MAR"] = float(scenario["mar"])
            if "mor" in scenario:
                overrides["MOR"] = float(scenario["mor"])
            if "handoff_min_interval" in scenario:
                overrides["HANDOFF_MIN_INTERVAL"] = float(scenario["handoff_min_interval"])
            if "uplink_loss_p" in scenario:
                overrides["UPLINK_LOSS_P"] = float(scenario["uplink_loss_p"])
            if "w8_scale" in scenario:
                # gate-6 penalty curriculum (TRAINING ONLY; eval never passes it)
                overrides["W8_SCALE"] = float(scenario["w8_scale"])
            cfg = replace(cfg, **overrides)
            self.cfg, self.monitor = cfg, DetectRuleMonitor(cfg)
        self.rng = random.Random(scenario.get("seed", cfg.seed))

        self.t = 0.0
        self.step_count = 0
        self.wingmen: List[BlueWingman] = []
        centres = [20.0, 60.0, 100.0]
        for i in range(self.n):
            w = BlueWingman(idx=i, x=20.0, y=centres[i], psi=0.0,
                            v=0.7 * A_MACH_KMS, subzone=i)
            w.radar = RadarState(cfg=cfg)
            w.tracks = TrackFile(cfg)
            w.last_scan = np.full(len(self._cell_centres[i]), -1e9)
            w.cell_reach_since = np.full(len(self._cell_centres[i]), np.nan)
            self.wingmen.append(w)
        self.manned = MannedNode()
        self.reds = RedFleet(cfg, k=cfg.K_RED, seed=self.rng.randrange(1 << 30))

        self.responsibility: Dict[int, int] = {}
        self.tz_track_since: Dict[int, float] = {}   # M05: first time a track on
        # tid exists (ANY state) while tid is in the tracking zone; keyed by
        # uid so drop/re-detect cycling does NOT reset the duty clock
        self.handoff_history: Dict[int, List[float]] = {}
        self.last_handoff_t: Dict[int, float] = {}
        self.c04_breach: Dict[Tuple[int, int], float] = {}
        self.stable_progress: Dict[int, int] = {}
        self.judged: Dict[int, bool] = {a.uid: False for a in self.reds.agents}
        self.m03_stall: Dict[Tuple[int, int], float] = {}
        self.metrics = {"apparent_mor": 0, "hidden_mor": 0,
                        "apparent_mar": 0, "hidden_mar": 0,
                        "handoffs": 0, "judged": 0}
        self.terminal: Optional[str] = None
        return self.observe()

    # ------------------------------------------------------------------
    # dynamics & scripted nodes
    # ------------------------------------------------------------------
    def _apply_controls(self, actions: Dict[int, DetectAction]) -> Dict[int, DetectAction]:
        """Mask illegal discrete combos + clip; return raw requests (S01)."""
        cfg = self.cfg
        raw = {}
        for i, w in enumerate(self.wingmen):
            a = actions.get(i, DetectAction())
            raw[i] = a
            a = a.clipped()
            w.role_changed = a.role != w.role
            if w.role_changed:
                w.role_age_prev = w.role_age
                w.role, w.role_age = a.role, 0.0
            else:
                w.role_age += cfg.dt
            # role-gear mask: snap to nearest legal gear for the role
            legal = cfg.GEARS_SEARCH if w.role == DetectRole.SEARCH else cfg.GEARS_TRACK
            gear = a.gear if a.gear in legal else min(legal, key=lambda g: abs(g - a.gear))
            w.gear_switch_denied = not w.radar.switch_gear(gear, self.step_count) \
                and gear != w.radar.gear
            w.radar.trim_axis(a.u_xi)
            # pointer: TRACK responsibility request carrier
            w.requests, w.pointer_target = [], None
            if a.pointer > 0 and w.role == DetectRole.TRACK:
                slot = a.pointer - 1
                if 0 <= slot < len(w.tracks.slots) and w.tracks.slots[slot] is not None:
                    tid = w.tracks.slots[slot].target_id
                    w.pointer_target = tid
                    if not w.link_degraded:          # C05 mask at interface
                        w.requests.append(tid)
            # kinematics (D1: flat cruise turn rate)
            omega = math.radians(cfg.OMEGA_CRUISE_DEG)
            w.psi = wrap_pi(w.psi + cfg.K_PSI * a.u_psi * omega * cfg.dt)
            v_min, v_max = ma_to_kms(cfg.BLUE_V_MIN_MA), ma_to_kms(cfg.BLUE_V_MAX_MA)
            w.v = max(v_min, min(v_max, w.v + (cfg.K_V_MS / 1000.0) * a.u_v * cfg.dt))
            w.x += w.v * math.cos(w.psi) * cfg.dt
            w.y += w.v * math.sin(w.psi) * cfg.dt
        return raw

    def _step_manned(self) -> None:
        """Fixed-anchor racetrack: legs along y around (CAP_X, CAP_Y)."""
        cfg = self.cfg
        m = self.manned
        wps = [(cfg.CAP_X - cfg.CAP_LEG_X, cfg.CAP_Y - cfg.CAP_LEG_Y),
               (cfg.CAP_X + cfg.CAP_LEG_X, cfg.CAP_Y - cfg.CAP_LEG_Y),
               (cfg.CAP_X + cfg.CAP_LEG_X, cfg.CAP_Y + cfg.CAP_LEG_Y),
               (cfg.CAP_X - cfg.CAP_LEG_X, cfg.CAP_Y + cfg.CAP_LEG_Y)]
        tx, ty = wps[m.wp_index]
        if math.hypot(tx - m.x, ty - m.y) < 3.0:
            m.wp_index = (m.wp_index + 1) % 4
            tx, ty = wps[m.wp_index]
        want = math.atan2(ty - m.y, tx - m.x)
        dpsi = wrap_pi(want - m.psi)
        max_d = math.radians(cfg.MANNED_OMEGA_DEG) * cfg.dt
        m.psi = wrap_pi(m.psi + max(-max_d, min(max_d, dpsi)))
        m.x += m.v * math.cos(m.psi) * cfg.dt
        m.y += m.v * math.sin(m.psi) * cfg.dt

    # ------------------------------------------------------------------
    # sensing & coverage
    # ------------------------------------------------------------------
    def _sense(self) -> None:
        cfg = self.cfg
        truth_targets = {a.uid: (a.x, a.y) for a in self.reds.agents}
        half = math.radians(cfg.FOV_HALF_DEG)
        n_sub = int(round(cfg.dt / cfg.BEAM_DWELL))       # dwell-resolution sensing
        for sub in range(n_sub):
            ts = self.t - cfg.dt + cfg.BEAM_DWELL * (sub + 1)
            for w in self.wingmen:
                hits, bore = w.radar.scan_dwell(w.psi, w.pos(), truth_targets, self.rng)
                w.tracks.update_with_hits(
                    [(tid, *truth_targets[tid]) for tid in hits], ts,
                    w.radar.frame_period())
                # coverage raster marking (C02/E03) at dwell resolution
                cells = self._cell_centres[w.subzone]
                dx = cells[:, 0] - w.x
                dy = cells[:, 1] - w.y
                dist = np.hypot(dx, dy)
                ang = (np.arctan2(dy, dx) - bore + math.pi) % (2 * math.pi) - math.pi
                covered = (dist <= cfg.R_MAX) & (np.abs(ang) <= half)
                # V2.7: C02 counts only REACHABLE cells (within R_MAX) in the
                # DETECTION zone (x > X_TRACK; tracking-zone cells are the
                # target-level rules' job); clock starts at first reach
                reach_now = (dist <= cfg.R_MAX) & (cells[:, 0] > cfg.X_TRACK)
                first = np.isnan(w.cell_reach_since) & reach_now
                w.cell_reach_since[first] = ts
                newly = covered & (ts - w.last_scan > cfg.T_SCAN_REQ)
                new_area = float(newly.sum() * CELL_KM * CELL_KM)
                w.last_scan[covered] = ts
                w.e03_window.append((new_area, w.v * cfg.BEAM_DWELL))
        for w in self.wingmen:
            w.tracks.tick(self.t, cfg.dt)
        # M05 bookkeeping: first time ANY wingman holds a track (any state)
        # on tid while tid sits in the tracking zone
        red_by_uid = {a.uid: a for a in self.reds.agents}
        for w in self.wingmen:
            for tr in w.tracks.slots:
                if tr is None or tr.target_id in self.tz_track_since:
                    continue
                a = red_by_uid.get(tr.target_id)
                if a is not None and cfg.zone_of(a.x) == ZONE_TRACKING:
                    self.tz_track_since[tr.target_id] = self.t

    # ------------------------------------------------------------------
    # responsibility arbitration (machine-speed channel, delay = 0; §5.2)
    # ------------------------------------------------------------------
    def _gate_ok(self, i: int, tx: float, ty: float) -> bool:
        """C04 capability gate: in range and inside usable sector half-width."""
        w = self.wingmen[i]
        d = math.hypot(tx - w.x, ty - w.y)
        if d > self.cfg.C04_RANGE:
            return False
        bearing = math.atan2(ty - w.y, tx - w.x)
        off = math.degrees(abs(wrap_pi(bearing - (w.psi + math.radians(w.radar.axis_deg)))))
        usable = min(w.radar.gear, self.cfg.SECTOR_AXIS_MAX_DEG - abs(w.radar.axis_deg))
        return off <= usable

    def _score(self, i: int, tid: int, tx: float, ty: float) -> float:
        """S_i(j) = wq*Q + wg*G - wl*L - wm*M + h*1[owner]  (plan §5.2)."""
        wq, wg, wl, wm, h = self.cfg.ARB_WEIGHTS
        w = self.wingmen[i]
        tr = w.tracks.track_of(tid)
        if tr is not None:
            q = 0.5 * (1 - min(1.0, tr.age(self.t) / self.cfg.AGE_DROP)) \
              + 0.5 * (1 - min(1.0, tr.sigma / self.cfg.SIGMA_REF))
        else:
            q = 0.0
        d = math.hypot(tx - w.x, ty - w.y)
        g = max(0.0, 1.0 - d / self.cfg.R_MAX)
        load = len(w.owned) / self.cfg.TRACK_SLOTS
        bearing = math.atan2(ty - w.y, tx - w.x)
        m_cost = abs(wrap_pi(bearing - w.psi)) / math.pi
        s = wq * q + wg * g - wl * load - wm * m_cost
        if self.responsibility.get(tid) == i:
            s += h
        return s

    def _arbitrate(self) -> List[dict]:
        cfg = self.cfg
        events: List[dict] = []
        red_pos = {a.uid: (a.x, a.y) for a in self.reds.agents}

        # release judged targets
        for tid in list(self.responsibility):
            if self.judged.get(tid):
                self.wingmen[self.responsibility[tid]].owned.discard(tid)
                del self.responsibility[tid]

        # candidate set: requested targets + owned targets whose owner breaches C04
        requested: Dict[int, List[int]] = {}
        for i, w in enumerate(self.wingmen):
            for tid in w.requests:
                requested.setdefault(tid, []).append(i)
        candidates_tids = set(requested)
        for tid, owner in self.responsibility.items():
            if (owner, tid) in self.c04_breach:
                candidates_tids.add(tid)

        for tid in candidates_tids:
            if tid not in red_pos:
                continue
            tx, ty = red_pos[tid]
            owner = self.responsibility.get(tid)
            pool = set(requested.get(tid, []))
            if owner is not None:
                pool.add(owner)
            pool = {i for i in pool if self._gate_ok(i, tx, ty)}
            if not pool:
                continue
            best = max(pool, key=lambda i: self._score(i, tid, tx, ty))
            if owner is None:
                if best in requested.get(tid, []):
                    self.responsibility[tid] = best
                    self.wingmen[best].owned.add(tid)
                    self.last_handoff_t[tid] = self.t
            elif best != owner and \
                    self._score(best, tid, tx, ty) > self._score(owner, tid, tx, ty) + cfg.ARB_HYSTERESIS:
                exempt = (owner, tid) in self.c04_breach     # V2.4 C04 exemption
                self.wingmen[owner].owned.discard(tid)
                self.wingmen[best].owned.add(tid)
                self.responsibility[tid] = best
                if not exempt:
                    self.handoff_history.setdefault(tid, []).append(self.t)
                self.last_handoff_t[tid] = self.t
                self.metrics["handoffs"] += 1
                events.append({"tid": tid, "from": owner, "to": best, "exempt": exempt})
        # C04 breach timers (all owned pairs)
        for tid, owner in self.responsibility.items():
            if tid not in red_pos:
                continue
            ok = self._gate_ok(owner, *red_pos[tid])
            if not ok:
                self.c04_breach.setdefault((owner, tid), self.t)
                # handoff attempts while breaching: wingman requesting elsewhere
            else:
                self.c04_breach.pop((owner, tid), None)
        # C05 flag: any responsibility change attempted while degraded
        for i, w in enumerate(self.wingmen):
            w.handoff_attempted = bool(w.requests) and w.link_degraded
        return events

    # ------------------------------------------------------------------
    # verdicts & link
    # ------------------------------------------------------------------
    def _update_verdicts(self) -> List[int]:
        cfg = self.cfg
        newly_judged: List[int] = []
        red_by_uid = {a.uid: a for a in self.reds.agents}
        for tid, owner in self.responsibility.items():
            a = red_by_uid.get(tid)
            if a is None or self.judged[tid]:
                continue
            tr = self.wingmen[owner].tracks.track_of(tid)
            in_zone = cfg.zone_of(a.x) == ZONE_TRACKING
            ok = in_zone and tr is not None and tr.age(self.t) < cfg.AGE_STD_TRACK
            prev = self.stable_progress.get(tid, 0)
            self.stable_progress[tid] = prev + 1 if ok else 0   # D3: zone exit resets
            key = (owner, tid)
            if self.stable_progress[tid] > prev:
                self.m03_stall.pop(key, None)
            else:
                self.m03_stall[key] = self.m03_stall.get(key, 0.0) + cfg.dt
            if self.stable_progress[tid] >= cfg.STABLE_WINDOW:
                self.judged[tid] = True
                self.metrics["judged"] += 1
                newly_judged.append(tid)
        # stall timers for owned unjudged targets outside zone too
        for tid, owner in self.responsibility.items():
            if not self.judged[tid] and (owner, tid) not in self.m03_stall:
                self.m03_stall[(owner, tid)] = 0.0
        return newly_judged

    def _step_link(self) -> None:
        cfg = self.cfg
        for w in self.wingmen:
            lost = 1.0 if self.rng.random() < cfg.UPLINK_LOSS_P else 0.0
            w.uplink_window.append(lost)
            w.link_degraded = (sum(w.uplink_window) / max(1, len(w.uplink_window))) > cfg.LINK_DROP_RATE

    # ------------------------------------------------------------------
    # truth / reward / termination
    # ------------------------------------------------------------------
    def _truth(self, raw: Dict[int, DetectAction]) -> dict:
        cfg = self.cfg
        blue = []
        for w in self.wingmen:
            # C02 (V2.7): worst coverage age over REACHABLE cells only;
            # age counts from max(last_scan, first-reachable time)
            if len(w.last_scan):
                base = np.maximum(w.last_scan,
                                  np.nan_to_num(w.cell_reach_since, nan=-1e9))
                reach = ~np.isnan(w.cell_reach_since)
                cov_age = float(self.t - base[reach].min()) if reach.any() else 0.0
            else:
                cov_age = 0.0
            area = sum(a for a, _ in w.e03_window)
            dist = sum(d for _, d in w.e03_window)
            blue.append(dict(
                x=w.x, y=w.y, psi=w.psi, v=w.v,
                role=w.role.value, role_age=w.role_age, role_age_prev=w.role_age_prev,
                role_changed=w.role_changed,
                gear=w.radar.gear, axis_deg=w.radar.axis_deg, subzone=w.subzone,
                link_degraded=w.link_degraded,
                uplink_loss=sum(w.uplink_window) / max(1, len(w.uplink_window)),
                coverage_max_age=cov_age,
                cover_area_rate=area / dist if dist > 1e-6 else math.inf,
                tracks={tr.target_id: {"age": tr.age(self.t), "state": tr.state,
                                       "sigma": tr.sigma}
                        for tr in w.tracks.slots if tr is not None},
                pointer_target=w.pointer_target, requests=list(w.requests),
                handoff_attempted=w.handoff_attempted,
                gear_switch_denied=w.gear_switch_denied, owned=sorted(w.owned),
            ))
        m = self.manned
        blue.append(dict(x=m.x, y=m.y, psi=m.psi, v=m.v))
        red = [dict(uid=a.uid, x=a.x, y=a.y, judged=self.judged[a.uid])
               for a in self.reds.agents]
        return dict(t=self.t, step=self.step_count, blue=blue, red=red,
                    responsibility=dict(self.responsibility),
                    tz_track_since=dict(self.tz_track_since),
                    handoff_history=self.handoff_history,
                    c04_breach=self.c04_breach, m03_stall=self.m03_stall,
                    raw_actions=raw)

    def _reward(self, report, newly_judged: List[int], handoffs: List[dict]) -> float:
        cfg = self.cfg
        r = 0.0
        red_by_uid = {a.uid: a for a in self.reds.agents}
        # w1/w2 update-standard ticks (own responsibility only, §7.3 anti-farming)
        for i, w in enumerate(self.wingmen):
            y0, y1 = cfg.subzone_bounds(w.subzone)
            for tr in w.tracks.slots:
                if tr is None:
                    continue
                a = red_by_uid.get(tr.target_id)
                if a is None:
                    continue
                zone = cfg.zone_of(a.x)
                if zone == ZONE_TRACKING and self.responsibility.get(tr.target_id) == i \
                        and tr.age(self.t) < cfg.AGE_STD_TRACK:
                    r += cfg.W1_TRACK_TICK
                elif zone == ZONE_DETECTION and y0 <= a.y < y1 \
                        and tr.age(self.t) < cfg.AGE_STD_DETECT:
                    r += cfg.W2_DETECT_TICK
        r += cfg.W3_STABLE * len(newly_judged)
        # w5/w6 approach cost, truth per pair, wingmen only (V2.6: the scripted
        # manned node is out of the approach-cost ledger -- uncontrollable by
        # the policy, its penalty would be pure reward noise)
        blue_xy = [(w.x, w.y) for w in self.wingmen]
        for a in self.reds.agents:
            for bx, by in blue_xy:
                d = math.hypot(a.x - bx, a.y - by)
                if d < cfg.MAR:
                    r -= cfg.W6_MAR
                elif d < cfg.MOR:
                    r -= cfg.W5_MOR_GRAD * (cfg.MOR - d) / cfg.MOR_GRADIENT_SPAN
        r -= cfg.W7_HANDOFF * len(handoffs)
        r -= self.monitor.penalty(report)
        r -= cfg.STEP_COST
        if self.terminal == "success":
            r += cfg.W4_SUCCESS
        elif self.terminal == "failure":
            r -= cfg.W9_FAILURE
        return r

    def _bookkeep(self) -> None:
        """Dual bookkeeping (V2.1): apparent vs hidden approach violations.
        V2.6: wingmen only (manned node out of the approach-cost ledger)."""
        cfg = self.cfg
        blue_xy = [(w.x, w.y) for w in self.wingmen]
        any_tracks = [set(w.tracks._by_target) for w in self.wingmen]
        for a in self.reds.agents:
            d = min(math.hypot(a.x - bx, a.y - by) for bx, by in blue_xy)
            seen = any(a.uid in s for s in any_tracks)
            if d < cfg.MAR:
                self.metrics["apparent_mar" if seen else "hidden_mar"] += 1
            elif d < cfg.MOR:
                self.metrics["apparent_mor" if seen else "hidden_mor"] += 1

    def _check_terminal(self) -> None:
        cfg = self.cfg
        for a in self.reds.agents:
            if a.x < cfg.X_CUTOFF and not self.judged[a.uid]:
                self.terminal = "failure"
                return
        if all(self.judged.values()):
            self.terminal = "success"
        elif self.step_count >= cfg.horizon:
            self.terminal = "timeout"

    # ------------------------------------------------------------------
    def step(self, actions: Dict[int, DetectAction]) -> StepResult:
        self.t += self.cfg.dt
        self.step_count += 1
        raw = self._apply_controls(actions)
        self._step_manned()
        self.reds.step(self.t, self.cfg.dt)
        self._sense()
        handoffs = self._arbitrate()
        newly_judged = self._update_verdicts()
        self._step_link()
        self._check_terminal()
        truth = self._truth(raw)
        report = self.monitor.evaluate_truth(truth)
        reward = self._reward(report, newly_judged, handoffs)
        self._bookkeep()
        info = {"truth": truth, "actions": raw, "report": report,
                "terminal": self.terminal, "handoffs": handoffs,
                "metrics": dict(self.metrics)}
        return StepResult(self.observe(), reward, self.terminal is not None, info)

    # ------------------------------------------------------------------
    # observation (structured; 138-dim vector lives in detect_encoder)
    # ------------------------------------------------------------------
    def _observation_for(self, i: int) -> dict:
        cfg = self.cfg
        w = self.wingmen[i]
        teammates = {}
        for k, o in enumerate(self.wingmen):
            if k == i:
                continue
            teammates[k] = {"dx": o.x - w.x, "dy": o.y - w.y, "psi": o.psi, "v": o.v,
                            "has_pointer": 1.0 if o.pointer_target is not None else 0.0}
        m = self.manned
        own_tracks = [tr for tr in w.tracks.slots if tr is not None]
        n_det = n_trk = 0
        red_by_uid = {a.uid: a for a in self.reds.agents}
        for tr in own_tracks:
            ex, _ey = tr.extrapolate(self.t)
            zone = cfg.zone_of(ex)
            n_det += zone == ZONE_DETECTION
            n_trk += zone == ZONE_TRACKING
        mission = {"step": self.step_count, "t": self.t,
                   "n_detect_est": n_det, "n_track_est": n_trk,
                   "n_judged": sum(self.judged.values()),
                   "responsibility": dict(self.responsibility),
                   "table_age": cfg.LINK_DELAY_S,
                   "pointers": {}}
        # link-delivered pointer-level info (no fusion): who tracks whom, where
        for k, o in enumerate(self.wingmen):
            if k == i:
                continue
            for tid in o.owned:
                tr = o.tracks.track_of(tid)
                if tr is not None:
                    mission["pointers"][tid] = tr.extrapolate(self.t)
        return {"self": w, "teammates": teammates,
                "manned": {"dx": m.x - w.x, "dy": m.y - w.y, "v": m.v,
                           "link": 0.0 if w.link_degraded else 1.0},
                "coverage_max_age": self.t - float(w.last_scan.min()),
                "mission": mission,
                "tracks": w.tracks.features(w.pos(), self.t)}

    def observe(self) -> Dict[int, dict]:
        return {i: self._observation_for(i) for i in range(self.n)}
