"""Per-aircraft track file for the MUMT detection scenario (plan V2.4 §3.3).

Single-source tracking only -- the V2.0 boundary removes all cross-platform
fusion (no COP, no cross-source correlation).  Each wingman maintains its own
track file fed exclusively by its own radar plots.

Simulation simplifications (documented, plan §3.3 note):
  - plots carry the truth target id: with P_fa = 0 (no clutter module;
    reserved as an extension in plan §9.4) single-source hit-to-track
    association is exact;
  - no plot-noise term: estimation uncertainty lives exclusively in the
    covariance bookkeeping (sigma x0.8 on update, +q per second inflation,
    capped at sigma_ref = 4 km).

State machine: detected -> tentative -> confirmed -> coast -> dropped
  - tentative : >= 2 hits of the same target within a 6.5 s window
  - confirmed : after tentative, 5 s of continuous refresh with every gap
                <= min(2*T_frame, 1.6 s)
  - coast     : age > 8 s (features down-weighted in observation)
  - dropped   : age > 13 s (slot freed)

Velocity estimate: first-order finite difference with smoothing
V <- 0.5 * (p_new - p_old)/dt + 0.5 * V (engineering initial value,
not in plan registry -- recorded here for calibration at pilot stage).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .detect_config import DetectSimConfig

STATE_ORDER = ("detected", "tentative", "confirmed", "coast")


@dataclass
class Track:
    slot: int
    target_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    sigma: float = 4.0          # km, 1-sigma position uncertainty
    state: str = "detected"
    last_hit_t: float = 0.0
    first_hit_t: float = 0.0
    hits_in_window: List[float] = field(default_factory=list)
    confirm_progress: float = 0.0
    stable_progress: int = 0    # consecutive seconds meeting age standard (M03)
    judged: bool = False        # stable-window判定完成 (permanent once set)

    def age(self, t: float) -> float:
        return t - self.last_hit_t

    def extrapolate(self, t: float) -> Tuple[float, float]:
        dt = t - self.last_hit_t
        return self.x + self.vx * dt, self.y + self.vy * dt


class TrackFile:
    """Own-ship track book: fixed slot array, no cross-source data."""

    def __init__(self, cfg: DetectSimConfig):
        self.cfg = cfg
        self.slots: List[Optional[Track]] = [None] * cfg.TRACK_SLOTS
        self._by_target: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def _alloc(self, target_id: int, x: float, y: float, t: float) -> Optional[Track]:
        for i, tr in enumerate(self.slots):
            if tr is None:
                tr = Track(slot=i, target_id=target_id, x=x, y=y,
                           sigma=self.cfg.SIGMA_REF, state="detected",
                           last_hit_t=t, first_hit_t=t, hits_in_window=[t])
                self.slots[i] = tr
                self._by_target[target_id] = i
                return tr
        return None  # slots exhausted: plot discarded (capacity pressure)

    def _drop(self, idx: int) -> None:
        tr = self.slots[idx]
        if tr is not None:
            self._by_target.pop(tr.target_id, None)
            self.slots[idx] = None

    # ------------------------------------------------------------------
    def update_with_hits(self, hits: List[Tuple[int, float, float]], t: float,
                         t_frame: float) -> None:
        """Ingest this step's plots: (target_id, x, y) truth-id-tagged."""
        for tid, x, y in hits:
            idx = self._by_target.get(tid)
            if idx is None:
                self._alloc(tid, x, y, t)
                continue
            tr = self.slots[idx]
            gap = t - tr.last_hit_t
            # velocity finite difference with 0.5 smoothing
            if gap > 0:
                nvx, nvy = (x - tr.x) / gap, (y - tr.y) / gap
                tr.vx = 0.5 * nvx + 0.5 * tr.vx
                tr.vy = 0.5 * nvy + 0.5 * tr.vy
            tr.x, tr.y = x, y
            tr.last_hit_t = t
            tr.hits_in_window.append(t)
            tr.sigma = max(1e-3, tr.sigma * self.cfg.COV_SHRINK)
            # state transitions
            if tr.state == "detected":
                recent = [h for h in tr.hits_in_window if t - h <= self.cfg.TRACK_INIT_WINDOW]
                if len(recent) >= self.cfg.TRACK_INIT_HITS:
                    tr.state = "tentative"
            elif tr.state == "tentative":
                max_gap = min(2.0 * t_frame, self.cfg.TRACK_CONFIRM_GAP_MAX)
                if gap <= max_gap:
                    tr.confirm_progress += gap
                else:
                    tr.confirm_progress = 0.0
                if tr.confirm_progress >= self.cfg.TRACK_CONFIRM_TIME:
                    tr.state = "confirmed"
            if tr.state == "coast":
                tr.state = "confirmed" if tr.confirm_progress >= self.cfg.TRACK_CONFIRM_TIME else "tentative"

    def tick(self, t: float, dt: float) -> None:
        """Age all tracks; inflate covariance; coast/drop transitions."""
        for i, tr in enumerate(self.slots):
            if tr is None:
                continue
            tr.sigma = min(self.cfg.SIGMA_REF,
                           math.sqrt(tr.sigma ** 2 + self.cfg.COV_Q_KM2_S * dt))
            a = tr.age(t)
            if a > self.cfg.AGE_DROP:
                self._drop(i)
            elif a > self.cfg.AGE_DEGRADE and tr.state in ("tentative", "confirmed"):
                tr.state = "coast"

    # ------------------------------------------------------------------
    def track_of(self, target_id: int) -> Optional[Track]:
        idx = self._by_target.get(target_id)
        return self.slots[idx] if idx is not None else None

    def n_tracks(self) -> int:
        return sum(1 for tr in self.slots if tr is not None)

    def confirmed_ids(self) -> List[int]:
        return [tr.target_id for tr in self.slots if tr is not None and tr.state == "confirmed"]

    def features(self, own_xy: Tuple[float, float], t: float) -> List[List[float]]:
        """Per-slot observation features (plan §7.1 track block, 12 dims/slot):

        [dx/R_max, dy/R_max, sin psi_j, cos psi_j, V_j/(0.95Ma),
         age/13, sigma/sigma_ref, state one-hot x4, slot_mask]
        Angles encoded sin/cos; geometry relative to own ship (V1.23 encoding
        discipline).  Empty slots return zeros (mask 0).
        """
        out: List[List[float]] = []
        r_max = self.cfg.R_MAX
        v_norm = 0.95 * 0.308  # km/s
        for tr in self.slots:
            if tr is None:
                out.append([0.0] * 12)
                continue
            ex, ey = tr.extrapolate(t)
            dx, dy = (ex - own_xy[0]) / r_max, (ey - own_xy[1]) / r_max
            spd = math.hypot(tr.vx, tr.vy)
            psi = math.atan2(tr.vy, tr.vx) if spd > 1e-9 else 0.0
            onehot = [1.0 if tr.state == s else 0.0 for s in STATE_ORDER]
            age_feat = min(1.0, tr.age(t) / self.cfg.AGE_DROP)
            if tr.age(t) > self.cfg.AGE_DEGRADE:
                age_feat = min(1.0, age_feat)  # down-weight handled consumer-side
            out.append([dx, dy, math.sin(psi), math.cos(psi),
                        min(1.0, spd / v_norm),
                        age_feat, min(1.0, tr.sigma / self.cfg.SIGMA_REF),
                        *onehot, 1.0])
        return out
