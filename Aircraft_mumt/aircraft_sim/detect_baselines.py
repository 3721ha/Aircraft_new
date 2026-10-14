# -*- coding: utf-8 -*-
"""P2 scripted baselines for the MUMT detection scenario (plan §9.4 gate 2/3).

Three deterministic controllers, all duck-typing ``policy(env) ->
Dict[int, DetectAction]``:

- ``HeurPolicy``   -- the §6.1 reference control laws as a behaviour本体:
  SEARCH steers to the stalest coverage-cell centroid of the own subzone
  (C02-driven), TRACK closes on the owned target with the d* = 62 km
  distance loop; responsibility is requested via the pointer carrier.
  Info discipline: own-ship products only (tracks / coverage raster /
  responsibility table), no truth.
- ``TwsPolicy``    -- fixed TWS-style sensor programme (the paper's
  structural-failure contrast): while owning, gear cycle 2x(30 deg, 2 s
  frame) + 1x(15 deg, 1 s frame) — the "4 s search + 1 s fine scan"
  pattern mapped onto legal TRACK gears (role-gear mask); same reference
  laws for motion.  E02-clean (role dwell respected).
- ``OraclePolicy`` -- truth scheduler (ceiling): exact pointing on the
  true position, tightest gear, same distance loop.  Uses red truth
  (positions AND velocities) — never a training input, ceiling only.

All three respect the interface masks (role-gear, C8 frame dwell is left
to the radar machine: requests persist until the switch lands).
"""
import math
from typing import Dict, List, Optional, Tuple

from .detect_config import (A_MACH_KMS, DetectSimConfig, ZONE_TRACKING,
                            ma_to_kms)
from .detect_models import DetectAction, DetectRole
from .radar import wrap_pi

V_CRUISE = 0.8 * A_MACH_KMS          # SEARCH reference speed (plan §6.1)


# ----------------------------------------------------------------------
# helpers: reference-law -> raw control mapping (plan §6.1 unified map)
# ----------------------------------------------------------------------
def _u_psi(w, psi_ref: float, cfg: DetectSimConfig) -> float:
    omega = math.radians(cfg.OMEGA_CRUISE_DEG)
    d = wrap_pi(psi_ref - w.psi)
    return max(-1.0, min(1.0, d / (cfg.K_PSI * omega * cfg.dt)))


def _u_v(w, v_ref_kms: float, cfg: DetectSimConfig) -> float:
    dv = v_ref_kms - w.v
    return max(-1.0, min(1.0, dv / (cfg.K_V_MS / 1000.0 * cfg.dt)))


def _u_xi(w, xi_ref_deg: float) -> float:
    return max(-1.0, min(1.0, (xi_ref_deg - w.radar.axis_deg) / 15.0))


def _slot_of(w, tid: int) -> int:
    """pointer carrier: 1-based slot index of tid in own track file (0 none)."""
    for s, tr in enumerate(w.tracks.slots):
        if tr is not None and tr.target_id == tid:
            return s + 1
    return 0


def _gear_request(w, desired: float) -> float:
    """C8-clean gear request: ask for the new gear only when the switch
    would land (frame boundary + full frame dwell); otherwise keep asking
    the CURRENT gear (a no-op) so E01 denial spam never fires."""
    if desired != w.radar.gear and not w.radar.switch_ready():
        return w.radar.gear
    return desired


def _stale_cell_centroid(env, i: int) -> Tuple[float, float]:
    """Centroid of the top-decile stale coverage cells of own subzone."""
    w = env.wingmen[i]
    cells = env._cell_centres[w.subzone]
    stale = env.t - w.last_scan
    k = max(1, len(cells) // 10)
    idx = stale.argsort()[-k:]
    return float(cells[idx, 0].mean()), float(cells[idx, 1].mean())


def _distance_loop(w, tx: float, ty: float, tvx: float, tvy: float,
                   cfg: DetectSimConfig) -> Tuple[float, float, float]:
    """TRACK reference law: bearing, axis offset, distance-loop V_ref."""
    dx, dy = tx - w.x, ty - w.y
    d = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    # radial component of target velocity along the line of sight
    v_rad = (dx * tvx + dy * tvy) / max(d, 1e-6)
    v_ref = w.v + v_rad + cfg.TRACK_K_D * (d - cfg.TRACK_SWEET_DIST)
    v_ref = max(ma_to_kms(cfg.BLUE_V_MIN_MA), min(ma_to_kms(cfg.BLUE_V_MAX_MA), v_ref))
    xi_ref = math.degrees(wrap_pi(bearing - w.psi))
    return bearing, xi_ref, v_ref


# ----------------------------------------------------------------------
class HeurPolicy:
    """§6.1 reference laws + active responsibility requests (own-ship info)."""

    def __call__(self, env) -> Dict[int, DetectAction]:
        acts: Dict[int, DetectAction] = {}
        for i, w in enumerate(env.wingmen):
            owned = [tid for tid in sorted(w.owned) if not env.judged.get(tid)]
            if owned:
                acts[i] = self._track(env, i, owned)
                continue
            # no responsibility: request the best un-owned own-ship track
            req = self._best_request(env, i)
            if req is not None:
                acts[i] = req
            else:
                acts[i] = self._search(env, i)
        return acts

    # ------------------------------------------------------------------
    def _search(self, env, i: int) -> DetectAction:
        w = env.wingmen[i]
        cx, cy = _stale_cell_centroid(env, i)
        psi_ref = math.atan2(cy - w.y, cx - w.x)
        return DetectAction(u_psi=_u_psi(w, psi_ref, env.cfg),
                            u_v=_u_v(w, V_CRUISE, env.cfg),
                            u_xi=_u_xi(w, 0.0),
                            role=DetectRole.SEARCH,
                            gear=_gear_request(w, 60.0), pointer=0)

    def _track(self, env, i: int, owned: List[int]) -> DetectAction:
        w, cfg, t = env.wingmen[i], env.cfg, env.t
        # update-hungriest owned target first (own track age; missing = inf)
        def hunger(tid: int) -> float:
            tr = w.tracks.track_of(tid)
            return tr.age(t) if tr is not None else 1e9
        tid = max(owned, key=hunger)
        tr = w.tracks.track_of(tid)
        if tr is None:                      # lost the plot: re-acquire
            return self._search(env, i)
        px, py = tr.extrapolate(t + w.radar.frame_period())
        bearing, xi_ref, v_ref = _distance_loop(w, px, py, tr.vx, tr.vy, cfg)
        gear = _gear_request(w, 7.5 if (tr.state == "confirmed"
                                        and tr.sigma < 2.0) else 15.0)
        return DetectAction(u_psi=_u_psi(w, bearing, cfg),
                            u_v=_u_v(w, v_ref, cfg),
                            u_xi=_u_xi(w, xi_ref),
                            role=DetectRole.TRACK, gear=gear,
                            pointer=_slot_of(w, tid))

    def _best_request(self, env, i: int) -> Optional[DetectAction]:
        """Ask for the best un-owned track (predictive pointing so the C04
        gate passes at arbitration time)."""
        w, cfg, t = env.wingmen[i], env.cfg, env.t
        best, best_key = None, None
        for tr in w.tracks.slots:
            if tr is None or tr.state == "detected":
                continue  # I02 (V2.7): tentative (>=2 hits) is a legal basis
            tid = tr.target_id
            if env.judged.get(tid) or env.responsibility.get(tid) is not None:
                continue
            px, py = tr.extrapolate(t)
            d = math.hypot(px - w.x, py - w.y)
            if d > cfg.C04_RANGE:
                continue
            if px > cfg.X_TRACK + 10.0:
                continue  # M03 discipline: don't own what you can't progress
            key = (0 if cfg.zone_of(px) == ZONE_TRACKING else 1, d)
            if best_key is None or key < best_key:
                best, best_key = tr, key
        if best is None:
            return None
        px, py = best.extrapolate(t + 1.0)
        bearing, xi_ref, v_ref = _distance_loop(w, px, py, best.vx, best.vy, cfg)
        return DetectAction(u_psi=_u_psi(w, bearing, cfg),
                            u_v=_u_v(w, v_ref, cfg),
                            u_xi=_u_xi(w, xi_ref),
                            role=DetectRole.TRACK,
                            gear=_gear_request(w, 15.0),
                            pointer=_slot_of(w, best.target_id))


# ----------------------------------------------------------------------
class TwsPolicy(HeurPolicy):
    """Fixed TWS sensor programme (structural-failure contrast, plan §11).

    Owning: legal-gear cycle 2 x (30 deg / 2 s frame) + 1 x (15 deg / 1 s
    frame) == the "4 s search + 1 s fine scan" rhythm, regardless of track
    quality; motion still follows the reference laws (same motion, dumb
    sensor) so the failure is attributable to the sensor programme alone.
    """

    GEAR_CYCLE: Tuple[Tuple[float, float], ...] = ((30.0, 2.0), (30.0, 2.0),
                                                   (15.0, 1.0))

    def __init__(self) -> None:
        self._phase: Dict[int, List[float]] = {}   # wingman -> [idx, t_left]

    def _cycle_gear(self, w, i: int, dt: float) -> float:
        """Desired gear from the fixed 4+1 cycle.  The timer only runs
        while the desired gear is actually ENGAGED (C8 switches land late),
        and requests go through _gear_request so no E01 denial spam."""
        idx, t_left = self._phase.get(i, [0, 0.0])
        desired = self.GEAR_CYCLE[idx][0]
        if w.radar.gear == desired:
            t_left -= dt
            if t_left <= 0.0:
                idx = (idx + 1) % len(self.GEAR_CYCLE)
                t_left = self.GEAR_CYCLE[idx][1]
        self._phase[i] = [idx, t_left]
        return _gear_request(w, self.GEAR_CYCLE[idx][0])

    def _track(self, env, i: int, owned: List[int]) -> DetectAction:
        a = super()._track(env, i, owned)
        if a.role == DetectRole.TRACK:
            a.gear = self._cycle_gear(env.wingmen[i], i, env.cfg.dt)
        return a

    def _best_request(self, env, i: int) -> Optional[DetectAction]:
        a = super()._best_request(env, i)
        if a is not None:
            a.gear = _gear_request(env.wingmen[i], 15.0)
        return a


# ----------------------------------------------------------------------
class OraclePolicy:
    """Truth scheduler (ceiling): exact pointing, tightest legal gear."""

    def __init__(self) -> None:
        self._heur = HeurPolicy()

    def __call__(self, env) -> Dict[int, DetectAction]:
        cfg, t = env.cfg, env.t
        reds = {a.uid: a for a in env.reds.agents}
        acts: Dict[int, DetectAction] = {}
        for i, w in enumerate(env.wingmen):
            owned = [tid for tid in sorted(w.owned) if not env.judged.get(tid)]
            tid = None
            if owned:
                # nearest owned truth target (cheapest geometry to keep)
                tid = min(owned, key=lambda k: math.hypot(reds[k].x - w.x,
                                                          reds[k].y - w.y))
            else:
                # nearest un-owned, un-judged truth target ahead of cutoff
                cand = [a for a in env.reds.agents
                        if not env.judged.get(a.uid)
                        and env.responsibility.get(a.uid) is None
                        and cfg.X_CUTOFF < a.x <= cfg.X_TRACK + 10.0]
                                                   # M03 discipline (V2.7)
                if cand:
                    a = min(cand, key=lambda a: math.hypot(a.x - w.x, a.y - w.y))
                    tid = a.uid
            if tid is None:
                acts[i] = self._heur._search(env, i)
                continue
            r = reds[tid]
            tvx, tvy = r.speed * math.cos(r.heading), r.speed * math.sin(r.heading)
            bearing, xi_ref, v_ref = _distance_loop(w, r.x, r.y, tvx, tvy, cfg)
            slot = _slot_of(w, tid)
            tr = w.tracks.track_of(tid)
            if tr is None or tr.state != "confirmed":
                slot = 0          # I02: request only once confirmed
            # no own track yet: point exactly at truth (gear 15), request
            # once the track is confirmed
            gear = _gear_request(w, 7.5 if slot else 15.0)
            acts[i] = DetectAction(u_psi=_u_psi(w, bearing, cfg),
                                   u_v=_u_v(w, v_ref, cfg),
                                   u_xi=_u_xi(w, xi_ref),
                                   role=DetectRole.TRACK, gear=gear,
                                   pointer=slot)
        return acts


BASELINES = {"heur": HeurPolicy, "tws": TwsPolicy, "oracle": OraclePolicy}
