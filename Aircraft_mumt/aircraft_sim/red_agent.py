"""Scripted red force for the MUMT detection scenario (plan V2.4 §4).

- Ingress geometry: staggered pairs after Sambhoos, Bowman & Llinas
  (Information Fusion 12, 2011, §6.1 Fig. 6) -- "pairs from left and right
  45 deg and center"; quantified as lanes y in {30, 90} (K=4), +60 (K=6),
  +{15,105} (K=8), intra-pair x-stagger 8 km (plan §4.3).
- Manoeuvre library (plan §4.2, equal-probability mixture, resample every
  20-40 s): straight patrol / S-snake (+-30 deg, 20 s period) / random turn
  (Poisson mean 30 s, +-45..90 deg) / waypoint cruise (forward cone: west of
  current position, east of the cutoff line) / **regain (V2.6): turn back
  east +-30 deg for 15-25 s, then resume** -- the tactical "pump" rhythm;
  the duty servo is suspended during regain and compensates afterwards.
- **Same-altitude separation (V2.6)**: predicted 3 s pairwise min distance
  below RED_SEP_WARN_KM triggers a lateral heading bias; the hard floor
  RED_SEP_HARD_KM = 2 km mirrors blue S02 (frozen post-cutoff agents count
  as static obstacles).
- Westward duty (V2.2 hard requirement): each agent is assigned a fixed
  duty target in [0.60, 0.70] at spawn (frozen per seed).  A per-agent
  cumulative servo keeps actual westward progress on the target line:
  behind schedule -> pull heading to due west; ahead -> weave east-bound
  legs inside the lane.  This is the scripted-calibration layer (not a
  learned behaviour), guaranteeing cutoff arrival t_cut in [730, 851] s,
  i.e. before the 900 s timeout -- the failure terminal is a real race.
- Boundary suppression (V1.21): within 5 km of the airspace edge the
  outward heading component is reflected inward.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .detect_config import DetectSimConfig

WEST = math.pi  # heading of due west (+x = east)
EAST = 0.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class RedAgent:
    uid: int
    x: float
    y: float
    heading: float = WEST
    speed: float = 0.231        # km/s, set from cfg on spawn
    lane_y: float = 30.0
    maneuver: str = "patrol"
    maneuver_until: float = 0.0
    snake_phase: float = 0.0
    turn_target: float = WEST
    waypoint: Optional[Tuple[float, float]] = None
    duty_target: float = 0.65   # sampled in [DUTY_MIN, DUTY_MAX] at spawn
    west_done: float = 0.0      # cumulative westward progress, km
    weave_sign: float = 1.0
    crossed_cutoff: bool = False
    t_cutoff: Optional[float] = None

    def pos(self) -> Tuple[float, float]:
        return self.x, self.y


class RedFleet:
    SERVO_TOL_KM = 2.0          # cumulative progress deadband
    WEAVE_ANGLE_DEG = 45.0      # east-bound weave legs inside lane

    def __init__(self, cfg: DetectSimConfig, k: Optional[int] = None, seed: Optional[int] = None):
        self.cfg = cfg
        self.k = k or cfg.K_RED
        self.rng = random.Random(seed if seed is not None else cfg.seed)
        self.agents: List[RedAgent] = []
        self._regain_count = 0
        self._spawn()
        self.duty_log: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------
    def _spawn(self) -> None:
        cfg = self.cfg
        lanes = cfg.red_lane_ys(self.k)
        v = cfg.RED_CRUISE_MA * 0.308
        uid = 0
        li = 0
        while uid < self.k:
            lane = lanes[li % len(lanes)]
            for s in range(2):  # pair with 8 km x-stagger + lateral sub-lanes
                if uid >= self.k:
                    break
                x0 = cfg.RED_X0 + s * cfg.RED_STAGGER_KM
                sub = lane + (cfg.RED_PAIR_SUBLANE_KM if s == 0
                              else -cfg.RED_PAIR_SUBLANE_KM)
                a = RedAgent(uid=uid, x=x0,
                             y=sub, lane_y=sub, speed=v)
                # per-agent duty floor so that even the staggered trailer
                # crosses the cutoff by t <= 810 s (90 s margin absorbing
                # servo transient AND separation-escape detours, V2.6):
                # duty >= dist / (v * 810)
                dist = x0 - cfg.X_CUTOFF
                floor = max(cfg.RED_DUTY_MIN, dist / (v * 810.0))
                a.duty_target = self.rng.uniform(floor, cfg.RED_DUTY_MAX)
                a.weave_sign = self.rng.choice([-1.0, 1.0])
                self.agents.append(a)
                uid += 1
            li += 1

    # ------------------------------------------------------------------
    def _resample(self, a: RedAgent, t: float) -> None:
        r = self.rng
        a.maneuver = r.choice(["patrol", "snake", "turn", "waypoint", "regain"])
        a.maneuver_until = t + r.uniform(*self.cfg.RED_MANEUVER_RESAMPLE)
        if a.maneuver == "snake":
            a.snake_phase = r.uniform(0, 2 * math.pi)
        elif a.maneuver == "turn":
            delta = math.radians(r.uniform(45, 90)) * r.choice([-1, 1])
            a.turn_target = _wrap(a.heading + delta)
        elif a.maneuver == "waypoint":
            a.waypoint = self._sample_waypoint(a)
        elif a.maneuver == "regain":
            # pump rhythm (V2.6): east-bound leg, duty servo suspended.
            # Gated on schedule slack: only allowed when AHEAD of the duty
            # line (err < -TOL), so it spends margin instead of creating a
            # deficit the servo cannot repay before the cutoff race ends.
            v_cruise = self.cfg.RED_CRUISE_MA * 0.308
            err = a.duty_target * v_cruise * t - a.west_done
            if err > -self.SERVO_TOL_KM:
                a.maneuver = r.choice(["patrol", "snake", "turn", "waypoint"])
                if a.maneuver == "snake":
                    a.snake_phase = r.uniform(0, 2 * math.pi)
                elif a.maneuver == "turn":
                    delta = math.radians(r.uniform(45, 90)) * r.choice([-1, 1])
                    a.turn_target = _wrap(a.heading + delta)
                elif a.maneuver == "waypoint":
                    a.waypoint = self._sample_waypoint(a)
                return
            self._regain_count += 1
            a.maneuver_until = t + r.uniform(*self.cfg.RED_REGAIN_T)
            a.turn_target = _wrap(EAST + r.uniform(-1, 1)
                                  * math.radians(self.cfg.RED_REGAIN_SPREAD_DEG))

    def _sample_waypoint(self, a: RedAgent) -> Tuple[float, float]:
        """Forward cone: west of current x, east of cutoff + margin; lateral
        inside own sub-lane +-10 km (boundary-clamped)."""
        cfg, r = self.cfg, self.rng
        x_lo = cfg.X_CUTOFF + 10.0
        x_hi = max(x_lo + 5.0, a.x - 10.0)
        wx = r.uniform(x_lo, x_hi)
        y_lo = max(a.lane_y - 10.0, cfg.RED_EDGE_SUPPRESS_KM)
        y_hi = min(a.lane_y + 10.0, cfg.L_Y - cfg.RED_EDGE_SUPPRESS_KM)
        return wx, r.uniform(y_lo, y_hi)

    def _maneuver_heading(self, a: RedAgent, t: float) -> float:
        cfg = self.cfg
        if a.maneuver == "snake":
            amp = math.radians(cfg.RED_SNAKE_HALF_ANGLE_DEG)
            return WEST + amp * math.sin(2 * math.pi * t / cfg.RED_SNAKE_PERIOD + a.snake_phase)
        if a.maneuver == "turn":
            return a.turn_target
        if a.maneuver == "waypoint":
            if a.waypoint is None:
                a.waypoint = self._sample_waypoint(a)
            dx, dy = a.waypoint[0] - a.x, a.waypoint[1] - a.y
            if math.hypot(dx, dy) < 2.0:
                a.waypoint = self._sample_waypoint(a)
                dx, dy = a.waypoint[0] - a.x, a.waypoint[1] - a.y
            return math.atan2(dy, dx)
        if a.maneuver == "regain":
            return a.turn_target
        return WEST  # patrol

    def _duty_servo_heading(self, a: RedAgent, t: float, base: float) -> float:
        """Cumulative westward-progress servo; overrides the manoeuvre heading
        outside the deadband."""
        v_cruise = self.cfg.RED_CRUISE_MA * 0.308
        expected = a.duty_target * v_cruise * t
        err = expected - a.west_done
        if err > self.SERVO_TOL_KM:
            return WEST  # behind schedule: transit hard
        if err < -self.SERVO_TOL_KM:
            # ahead: weave east-bound leg, alternate sign to stay in lane
            if abs(a.y - a.lane_y) > 12.0:
                a.weave_sign = -1.0 if a.y > a.lane_y else 1.0
            ang = math.radians(self.WEAVE_ANGLE_DEG)
            return _wrap(EAST + a.weave_sign * ang)
        return base

    # ------------------------------------------------------------------
    def _sep_predict(self, a: RedAgent, b: RedAgent, h: float,
                     committed: Dict[int, float]) -> float:
        """Min predicted distance (km) to b over a 12 s window if a flies h.
        Uses b's COMMITTED heading for this step when available (agents are
        processed in uid order), else b's current heading."""
        b_speed = 0.0 if b.crossed_cutoff else b.speed   # frozen = static
        b_h = committed.get(b.uid, b.heading)
        best = math.inf
        for tau in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
            dx = (a.x + a.speed * math.cos(h) * tau) \
               - (b.x + b_speed * math.cos(b_h) * tau)
            dy = (a.y + a.speed * math.sin(h) * tau) \
               - (b.y + b_speed * math.sin(b_h) * tau)
            best = min(best, math.hypot(dx, dy))
        return best

    def _separation_heading(self, a: RedAgent, h: float,
                            committed: Dict[int, float]) -> float:
        """Same-altitude deconfliction (V2.6): if the predicted 10 s min
        distance to any other red drops below RED_SEP_WARN_KM, evaluate the
        two candidate headings h +- bias by forward simulation against ALL
        other reds and pick the one with the LARGER predicted min distance
        (ties: nearer to current heading).  Agents commit headings in uid
        order within a step, so a lower-uid aircraft's escape is already
        visible to higher-uid aircraft -- no priority convention needed and
        no "both dodge to the same side" failure.  Bias escalates
        45 -> 60 -> 90 deg as predicted min distance falls below
        WARN / 4 km / HARD.  Frozen post-cutoff agents count as static
        obstacles."""
        cfg = self.cfg
        others = [b for b in self.agents if b is not a]
        worst = min((self._sep_predict(a, b, h, committed) for b in others),
                    default=math.inf)
        if worst >= cfg.RED_SEP_WARN_KM:
            return h
        if worst < cfg.RED_SEP_HARD_KM:
            bias_deg = 90.0
        elif worst < 4.0:
            bias_deg = 60.0
        else:
            bias_deg = cfg.RED_SEP_BIAS_DEG
        bias = math.radians(bias_deg)
        cand = (_wrap(h + bias), _wrap(h - bias))
        scored = [(min(self._sep_predict(a, b, hh, committed) for b in others),
                   -abs(_wrap(hh - a.heading)), hh)
                  for hh in cand]
        return max(scored)[2]

    # ------------------------------------------------------------------
    def step(self, t: float, dt: float) -> None:
        cfg = self.cfg
        omega = math.radians(cfg.RED_OMEGA_DEG)
        committed: Dict[int, float] = {}   # headings committed this step (uid order)
        for a in self.agents:
            if a.crossed_cutoff:
                continue
            if t >= a.maneuver_until:
                self._resample(a, t)
            h = self._maneuver_heading(a, t)
            if a.maneuver != "regain":          # servo suspended during regain
                h = self._duty_servo_heading(a, t, h)
            h = self._separation_heading(a, h, committed)  # safety outranks the servo
            # boundary suppression: reflect outward components (V1.21)
            sy, cx = math.sin(h), math.cos(h)
            if (a.y < cfg.RED_EDGE_SUPPRESS_KM and sy < 0) or \
               (a.y > cfg.L_Y - cfg.RED_EDGE_SUPPRESS_KM and sy > 0):
                sy = -sy
            if a.x > cfg.L_X - cfg.RED_EDGE_SUPPRESS_KM and cx > 0:
                cx = -cx
            h = math.atan2(sy, cx)
            dh = _wrap(h - a.heading)
            max_dh = omega * dt
            a.heading = _wrap(a.heading + max(-max_dh, min(max_dh, dh)))
            committed[a.uid] = a.heading
            dx_west = -a.speed * math.cos(a.heading) * dt
            a.west_done += dx_west
            a.x += a.speed * math.cos(a.heading) * dt
            a.y += a.speed * math.sin(a.heading) * dt
            a.y = max(0.0, min(cfg.L_Y, a.y))
            if a.x < cfg.X_CUTOFF:
                a.crossed_cutoff = True
                a.t_cutoff = t
        if int(t) % 60 == 0 and t > 0:
            v_cruise = cfg.RED_CRUISE_MA * 0.308
            duty = sum(a.west_done for a in self.agents) / (v_cruise * t * len(self.agents))
            self.duty_log.append((t, duty))

    def min_separation(self) -> float:
        """Current min pairwise distance (km) across all red agents."""
        best = math.inf
        for i, a in enumerate(self.agents):
            for b in self.agents[i + 1:]:
                best = min(best, math.hypot(a.x - b.x, a.y - b.y))
        return best

    # ------------------------------------------------------------------
    def run_episode(self) -> Dict:
        """Fast 900 s rollout for duty calibration (V2.2 hard requirement).

        min_separation is tracked only until the FIRST cutoff crossing: in
        DetectEnv the episode terminates at that moment, so post-crossing
        frozen-aircraft geometry is a rollout artifact, not a training
        condition."""
        t = 0.0
        dt = self.cfg.dt
        min_sep = math.inf
        first_crossed = False
        while t < self.cfg.horizon:
            self.step(t, dt)
            if not first_crossed:
                if any(a.crossed_cutoff for a in self.agents):
                    first_crossed = True
                else:
                    min_sep = min(min_sep, self.min_separation())
            t += dt
        times = [a.t_cutoff for a in self.agents]
        per_agent_duty = [a.west_done / (a.duty_target and (a.t_cutoff or self.cfg.horizon) * 1.0)
                          for a in self.agents]
        v_cruise = self.cfg.RED_CRUISE_MA * 0.308
        duties = [a.west_done / (v_cruise * (a.t_cutoff or self.cfg.horizon)) for a in self.agents]
        return {
            "t_cutoff": times,
            "all_crossed": all(v is not None for v in times),
            "max_t_cutoff": max((v for v in times if v is not None), default=None),
            "min_t_cutoff": min((v for v in times if v is not None), default=None),
            "duty_per_agent": duties,
            "duty_mean": sum(duties) / len(duties) if duties else 0.0,
            "duty_min": min(duties) if duties else 0.0,
            "duty_max": max(duties) if duties else 0.0,
            "min_separation": min_sep,
            "regain_count": self._regain_count,
        }
