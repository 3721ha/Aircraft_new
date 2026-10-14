"""Sector-scan radar model for the MUMT detection scenario (plan V2.4 §3).

Mechanics (§3.1):
  - five gears: half-angle beta in {7.5, 15, 30, 45, 60} deg,
    frame period T_frame in {0.5, 1, 2, 3, 4} s;
  - instantaneous FOV half-width 7.5 deg, beam dwell 0.5 s;
  - beam positions per frame: n = 2*beta/15, offsets evenly spaced so the
    outer beam edges reach +/- beta (relative to sector axis xi);
  - sector axis xi relative to nose, |xi| <= 60 deg; V1.22-C7 clamp: usable
    half-width = 60 - |xi| is enforced by the environment via gear masking.

Hit determination (§3.2):
  - a target is "swept" in a dwell when its bearing relative to the beam
    boresight is within FOV half-width and slant range <= R_max_eff;
  - off-boresight attenuation: R_max_eff = R_max * sqrt(cos(theta_off)),
    theta_off = bearing relative to aircraft nose;
  - per swept target per dwell: Bernoulli(P_d) hit, P_d from the range
    ladder evaluated at r = d / R_max_eff (0.9 / 0.6 / 0.3 / 0).

Plots are position-perfect: V2.4 §3.3 keeps no plot-noise term; estimation
uncertainty lives exclusively in the track-file covariance bookkeeping
(sigma shrinks x0.8 on update, inflates q=0.25 km^2/s between updates).

Decision step dt = 1 s = two dwell slots.  The radar advances its beam phase
by 2 slots per environment step and returns the set of swept-and-hit
target ids plus their plot positions.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .detect_config import DetectSimConfig


def wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def beam_offsets(beta_deg: float, fov_half_deg: float) -> List[float]:
    """Beam-centre offsets (deg) within one frame for gear half-angle beta.

    n = round(2*beta / (2*fov_half)); offsets evenly spaced so outer beam
    edges touch +/- beta.
    """
    step = 2.0 * fov_half_deg
    n = max(1, int(round(2.0 * beta_deg / step)))
    if n == 1:
        return [0.0]
    edge = beta_deg - fov_half_deg
    return [-edge + i * (2.0 * edge / (n - 1)) for i in range(n)]


@dataclass
class RadarState:
    """Per-aircraft radar state (plan §3.1 machine)."""

    cfg: DetectSimConfig
    gear: float = 60.0                    # current half-angle gear, deg
    axis_deg: float = 0.0                 # sector axis xi relative to nose
    phase: int = 0                        # dwell index within current frame
    frame_progress: float = 0.0           # s elapsed in current frame
    last_switch_step: int = -10_000       # env step of last gear switch
    on: bool = True                       # V2.3: radar on by default

    # ------------------------------------------------------------------
    def n_positions(self) -> int:
        return len(beam_offsets(self.gear, self.cfg.FOV_HALF_DEG))

    def frame_period(self) -> float:
        return self.cfg.GEAR_FRAME[self.gear]

    # ------------------------------------------------------------------
    def switch_gear(self, new_gear: float, step: int) -> bool:
        """C8 dwell constraint: switch allowed only AT a frame boundary and
        after >= 1 full frame in the current gear; the in-progress frame is
        voided on switch.  (V2.6 fix: frame_progress is now monotonic since
        the last switch -- the old modulo form wrapped below the period and
        permanently denied every switch after the first.)"""
        if new_gear not in self.cfg.GEAR_FRAME:
            return False
        if new_gear == self.gear:
            return True
        if self.last_switch_step >= 0 and \
                (self.frame_progress < self.frame_period() or self.phase != 0):
            return False  # in-progress frame would be voided mid-way: reject
        self.gear = new_gear
        self.phase = 0
        self.frame_progress = 0.0
        self.last_switch_step = step
        return True

    def switch_ready(self) -> bool:
        """True iff a gear switch requested now would be honoured (C8)."""
        return self.phase == 0 and self.frame_progress >= self.frame_period()

    def trim_axis(self, u_xi: float) -> None:
        """Continuous axis micro-adjust (plan §6.1): u_xi in [-1,1] -> deg."""
        u = max(-1.0, min(1.0, u_xi))
        self.axis_deg = self._clamp_axis(self.axis_deg + u * self.cfg.AXIS_TRIM_SCALE_DEG)

    def _clamp_axis(self, xi: float) -> float:
        return max(-self.cfg.SECTOR_AXIS_MAX_DEG, min(self.cfg.SECTOR_AXIS_MAX_DEG, xi))

    # ------------------------------------------------------------------
    def dwell_boresights(self, heading_rad: float) -> List[float]:
        """Absolute boresight angles (rad) of the two dwell slots of the next
        1 s decision step, advancing the phase."""
        offs = beam_offsets(self.gear, self.cfg.FOV_HALF_DEG)
        slots: List[float] = []
        for _ in range(int(round(1.0 / self.cfg.BEAM_DWELL))):  # 2 dwells per step
            b = heading_rad + math.radians(self.axis_deg + offs[self.phase])
            slots.append(b)
            self.phase = (self.phase + 1) % len(offs)
            self.frame_progress = (self.frame_progress + self.cfg.BEAM_DWELL) % self.frame_period()
        return slots

    # ------------------------------------------------------------------
    def pd(self, d_km: float, theta_off_rad: float) -> float:
        """Bernoulli hit probability: range ladder on r = d / R_max_eff."""
        r_max_eff = self.cfg.R_MAX * math.sqrt(max(0.0, math.cos(abs(theta_off_rad))))
        if r_max_eff <= 1e-9:
            return 0.0
        r = d_km / r_max_eff
        for r_edge, p in self.cfg.PD_TIERS:
            if r <= r_edge:
                return p
        return 0.0

    def swept(self, d_km: float, theta_off_rad: float, boresights: List[float],
              target_bearing_rad: float) -> bool:
        """Whether the target fell inside any dwell beam this step and inside
        the (attenuated) range."""
        r_max_eff = self.cfg.R_MAX * math.sqrt(max(0.0, math.cos(abs(theta_off_rad))))
        if d_km > r_max_eff:
            return False
        half = math.radians(self.cfg.FOV_HALF_DEG)
        return any(abs(wrap_pi(target_bearing_rad - b)) <= half for b in boresights)

    # ------------------------------------------------------------------
    def scan_dwell(self, heading_rad: float, own_xy: Tuple[float, float],
                   targets: Dict[int, Tuple[float, float]], rng) -> Tuple[List[int], float]:
        """Advance exactly one 0.5 s dwell; per-dwell Bernoulli draw (plan §3.2).

        Returns (hit ids, absolute boresight used).  The environment calls this
        at 0.5 s substeps so track ages carry dwell resolution -- the appendix-A
        window anchors (0.211 / 0.005) are defined at this granularity.
        """
        if not self.on:
            return [], heading_rad
        offs = beam_offsets(self.gear, self.cfg.FOV_HALF_DEG)
        b = heading_rad + math.radians(self.axis_deg + offs[self.phase])
        self.phase = (self.phase + 1) % len(offs)
        self.frame_progress += self.cfg.BEAM_DWELL   # monotonic since switch
        # (phase == 0 marks a frame boundary; C8 reads both)
        hits: List[int] = []
        half = math.radians(self.cfg.FOV_HALF_DEG)
        for tid, (tx, ty) in targets.items():
            dx, dy = tx - own_xy[0], ty - own_xy[1]
            d = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx)
            theta_off = abs(wrap_pi(bearing - heading_rad))
            if d > self.cfg.R_MAX * math.sqrt(max(0.0, math.cos(theta_off))):
                continue
            if abs(wrap_pi(bearing - b)) > half:
                continue
            if rng.random() < self.pd(d, theta_off):
                hits.append(tid)
        return hits, b

    def _scan(self, heading_rad: float, own_xy: Tuple[float, float],
              targets: Dict[int, Tuple[float, float]], rng) -> Tuple[List[int], List[float]]:
        """Advance one 1 s step (two dwells); return (hit ids, boresights used)."""
        hits: List[int] = []
        bores: List[float] = []
        for _ in range(int(round(1.0 / self.cfg.BEAM_DWELL))):
            h, b = self.scan_dwell(heading_rad, own_xy, targets, rng)
            hits.extend(h)
            bores.append(b)
        return hits, bores

    def scan_step(self, heading_rad: float, own_xy: Tuple[float, float],
                  targets: Dict[int, Tuple[float, float]], rng) -> List[int]:
        """Advance one decision step; return ids of targets with a Bernoulli hit.

        targets: {target_id: (x, y)} truth positions (km) -- the radar never
        sees more than this: no id list beyond geometry.
        """
        return self._scan(heading_rad, own_xy, targets, rng)[0]
