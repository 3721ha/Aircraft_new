"""MUMT cooperative detection scenario configuration (plan V2.4).

Single source of truth for every parameter in plan document
``MUMT协同探测场景与规则库设计方案V2.0-聚焦版.md`` (content version V2.4),
section 10 parameter registry.  Units: distances in km, times in s,
angles in rad (config stores deg where the plan table is in deg),
speeds internally in km/s (Mach converted via A_MACH_KMS).

Nothing in this module mutates global state; the legacy ``SimConfig``
stays untouched so the original 70 unit tests keep passing.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --- unit conversion -------------------------------------------------------
A_MACH_MS = 308.0                      # speed of sound, m/s (plan §2)
A_MACH_KMS = A_MACH_MS / 1000.0        # km/s
G0 = 9.81                              # m/s^2, turn-rate physical check (plan §2 note)


def ma_to_kms(ma: float) -> float:
    return ma * A_MACH_KMS


# --- zone ids --------------------------------------------------------------
ZONE_BEHIND = "behind_cutoff"          # x < X_CUTOFF: red crossing => failure
ZONE_TRACKING = "tracking"             # [X_CUTOFF, X_TRACK]
ZONE_DETECTION = "detection"           # (X_TRACK, L_X]


@dataclass(frozen=True)
class DetectSimConfig:
    """Frozen parameter set; instantiate once per experiment and pass around."""

    # ---- episode (plan §1.1/§9.4) ----
    dt: float = 1.0                    # decision step, s
    horizon: int = 900                 # 15 min episode

    # ---- airspace & functional zones (plan §1.1) ----
    L_X: float = 160.0                 # km, west->east extent
    L_Y: float = 120.0                 # km, lateral extent
    X_CUTOFF: float = 40.0             # km, deadline line
    X_TRACK: float = 110.0             # km, tracking/detection zone boundary
    AGE_STD_TRACK: float = 2.0         # s, tracking-zone update standard
    AGE_STD_DETECT: float = 5.0        # s, detection-zone update standard
    STABLE_WINDOW: int = 30            # s, strict zero-gap stable tracking (V2.2)

    # ---- blue force (plan §1.2/§2) ----
    N_WINGMEN: int = 3
    BLUE_V_MIN_MA: float = 0.4
    BLUE_V_MAX_MA: float = 0.95
    OMEGA_CRUISE_DEG: float = 3.0      # deg/s
    OMEGA_MANEUVER_DEG: float = 6.0    # deg/s
    ACCEL_MAX_MA_S: float = 0.1        # Ma/s
    K_PSI: float = 1.0                 # first-order heading response gain (legacy default)
    K_V_MS: float = 30.0               # m/s per unit u_V per step (plan §10)
    MANNED_V_MIN_MA: float = 0.5
    MANNED_V_MAX_MA: float = 0.9
    MANNED_OMEGA_DEG: float = 3.0

    # ---- red force (plan §4) ----
    K_RED: int = 4                     # baseline; sweep {2,4,6,8}
    RED_CRUISE_MA: float = 0.75
    RED_MANEUVER_MA_RANGE: Tuple[float, float] = (0.6, 0.9)
    RED_OMEGA_DEG: float = 8.0         # V2.6: 3->8 deg/s; 0.75 Ma @ 8 deg/s ~ 3.2 g,
                                       # inside fighter envelope (3 deg/s was
                                       # airliner standard-rate, too weak for
                                       # same-altitude deconfliction)
    RED_DUTY_MIN: float = 0.70         # V2.7: band raised 0.60-0.70 -> 0.70-0.80
                                       # (G2: HEUR floor flat at 100%; ~12% shorter
                                       # dwell puts pressure on scripted baselines)
    RED_DUTY_MAX: float = 0.80
    RED_STAGGER_KM: float = 8.0        # intra-pair stagger (literature quant.)
    RED_X0: float = 158.0              # km, entry x
    RED_MANEUVER_RESAMPLE: Tuple[int, int] = (20, 40)   # s
    RED_TURN_POISSON_MEAN: float = 30.0                  # s
    RED_SNAKE_HALF_ANGLE_DEG: float = 30.0
    RED_SNAKE_PERIOD: float = 20.0     # s
    RED_EDGE_SUPPRESS_KM: float = 5.0  # boundary suppression band (V1.21)
    # regain (pump) manoeuvre: transit west -> turn back east -> resume (V2.6)
    RED_REGAIN_T: Tuple[float, float] = (12.0, 20.0)   # s east-bound leg
    RED_REGAIN_SPREAD_DEG: float = 30.0                # east +- spread
    # same-altitude separation keeping (V2.6): warn band + hard floor
    RED_SEP_WARN_KM: float = 18.0      # 12 s predicted min distance triggers bias
    RED_SEP_HARD_KM: float = 2.0       # never-breach floor (mirrors blue S02)
    RED_SEP_BIAS_DEG: float = 45.0     # lateral bias while separating
    RED_PAIR_SUBLANE_KM: float = 4.0   # intra-pair lateral split (combat spread, V2.6)

    # ---- radar (plan §3.1/§3.2) ----
    R_MAX: float = 80.0                # km
    FOV_HALF_DEG: float = 7.5          # instantaneous beam half-width
    BEAM_DWELL: float = 0.5            # s per beam position
    # gear half-angle deg -> full-frame period s
    GEAR_FRAME: Dict[float, float] = field(default_factory=lambda: {
        7.5: 0.5, 15.0: 1.0, 30.0: 2.0, 45.0: 3.0, 60.0: 4.0})
    GEARS_SEARCH: Tuple[float, ...] = (45.0, 60.0)      # SEARCH role mask
    GEARS_TRACK: Tuple[float, ...] = (7.5, 15.0, 30.0)  # TRACK role mask
    PD_TIERS: Tuple[Tuple[float, float], ...] = ((0.5, 0.9), (0.8, 0.6), (1.0, 0.3))
    SECTOR_AXIS_MAX_DEG: float = 60.0  # xi clamp; usable half-width = 60 - |xi|
    AXIS_TRIM_SCALE_DEG: float = 15.0  # u_xi normalisation (plan §6.1 mapping)

    # ---- track file (plan §3.3) ----
    TRACK_INIT_HITS: int = 2           # hits within INIT_WINDOW to go tentative
    TRACK_INIT_WINDOW: float = 6.5     # s
    TRACK_CONFIRM_TIME: float = 5.0    # s continuous refresh
    TRACK_CONFIRM_GAP_MAX: float = 1.6 # s cap: gap <= min(2*T_frame, 1.6)
    AGE_DEGRADE: float = 8.0           # s, observation down-weight
    AGE_DROP: float = 13.0             # s, track dropped
    COV_SHRINK: float = 0.8            # per update
    COV_Q_KM2_S: float = 0.25          # (0.5 km)^2 / s inflation
    SIGMA_REF: float = 4.0             # km cap

    # ---- MOR/MAR (plan §1.3) ----
    MAR: float = 40.0                  # km
    MOR: float = 60.0                  # km
    MAR_MOR_VARIANTS: Tuple[Tuple[float, float], ...] = ((40.0, 60.0), (30.0, 50.0), (20.0, 40.0))
    K_SIGMA: float = 3.0               # shield inflation: d_eff = d_est - K_SIGMA*sigma (V2.1)
    MOR_GRADIENT_SPAN: float = 20.0    # km, (MOR - d)/span shaping

    # ---- separation / boundary (plan §8.1) ----
    D_SAFE: float = 2.0                # km, predicted 3 s min separation
    D_SAFE_LOOKAHEAD: float = 3.0      # s
    BOUNDARY_PREDICT: float = 5.0      # s, S03 turn-back trigger

    # ---- responsibility & link (plan §5/§6.2/§8.2) ----
    LINK_RATE_HZ: float = 1.0
    LINK_DELAY_S: float = 1.0
    T_SCAN_REQ: float = 120.0          # s, C02 coverage period (V2.7 G2:
                                       # 20 s physically infeasible; scope cut to
                                       # reachable DETECTION-zone cells)
    HANDOFF_MIN_INTERVAL: float = 30.0 # s, C03 (sweep {10,20,30})
    HANDOFF_INTERVAL_SWEEP: Tuple[float, ...] = (10.0, 20.0, 30.0)
    C04_RANGE: float = 80.0            # km capability gate
    C04_RESPONSE: float = 5.0          # s
    T_DWELL: float = 5.0               # s, role dwell
    APPROVAL_DELAY_INIT: float = 0.0   # s, first-round training (10.13 decision)
    LINK_DROP_RATE: float = 0.30       # F01: 10 s window uplink loss
    LINK_DROP_WINDOW: float = 10.0     # s
    UPLINK_LOSS_P: float = 0.05        # per-second Bernoulli uplink loss (baseline; raised in degraded-link variants)
    E03_AREA_PER_KM: float = 5.0       # km^2/km anti-loiter threshold
    N_SUBZONES: int = 3                # lateral lanes, 40 km each

    # ---- TRACK role geometry (plan §6.1) ----
    TRACK_SWEET_DIST: float = 62.0     # km, d* (centre of penalty-free [60,64])
    TRACK_K_D: float = 2.5e-3          # 1/s, distance loop gain (V1.19)

    # ---- manned CAP anchor (V2.6: fixed anchor, no centroid following) ----
    CAP_X: float = 15.0                # km, anchor centre (entry band)
    CAP_Y: float = 60.0                # km
    CAP_LEG_Y: float = 10.0            # racetrack half-length, north-south
    CAP_LEG_X: float = 5.0             # racetrack half-width, east-west

    # ---- arbitration scoring (plan §5.2) ----
    ARB_WEIGHTS: Tuple[float, ...] = (1.0, 1.0, 0.5, 0.5, 0.3)
    ARB_HYSTERESIS: float = 0.2        # delta_hyst

    # ---- rewards (plan §7.3) ----
    W1_TRACK_TICK: float = 0.02        # per s per responsibility target, age<=2s
    W2_DETECT_TICK: float = 0.005      # per s per target in own subzone, age<=5s
    W3_STABLE: float = 10.0            # one-off per stable-window completion
    W4_SUCCESS: float = 50.0           # per episode
    W5_MOR_GRAD: float = 0.02          # per s per pair, *(MOR-d)/span
    W6_MAR: float = 0.1                # per s per pair inside MAR
    W7_HANDOFF: float = 0.5            # per handoff event
    W8_LAYER: Tuple[float, ...] = (10.0, 7.5, 10.0, 5.0, 2.5, 1.25)  # P0..P5 (V1.20 map)
    W8_SCALE: float = 1.0              # penalty-curriculum knob (gate-6): TRAINING ONLY,
                                       # scales the w8 reward contribution; violation
                                       # reporting and eval (always 1.0) unaffected
    M05_GRACE: float = 5.0             # s after first confirmation before the
                                       # ownership duty binds (aligned w/ C04 5 s)
    W9_FAILURE: float = 50.0           # per episode
    STEP_COST: float = 0.005

    # ---- RL interface (plan §7.1/§9.1) ----
    OBS_DIM: int = 138                 # 17 + 8*12 + 2*6 + 4 + 6 + 3 (V2.3)
    TRACK_SLOTS: int = 8
    NET_HIDDEN: int = 128
    TOTAL_STEPS: int = 1_500_000       # per run
    N_ENVS: int = 16
    N_SEEDS: int = 10                  # 6 for extension references
    BC_EPOCHS: Tuple[int, int] = (50, 100)
    CONVERGE_WINDOW: int = 100         # episodes sliding platform

    seed: int = 7

    # ------------------------------------------------------------------
    def zone_of(self, x: float) -> str:
        """Functional zone of an x coordinate (plan §1.1)."""
        if x < self.X_CUTOFF:
            return ZONE_BEHIND
        if x <= self.X_TRACK:
            return ZONE_TRACKING
        return ZONE_DETECTION

    def age_standard(self, x: float) -> float:
        """Update standard (s) applying to a target at x."""
        return self.AGE_STD_TRACK if self.zone_of(x) == ZONE_TRACKING else self.AGE_STD_DETECT

    def pd_at(self, d_km: float) -> float:
        """Detection-probability tier at slant range (plan §3.2 ladder)."""
        r = d_km / self.R_MAX
        for r_edge, p in self.PD_TIERS:
            if r <= r_edge:
                return p
        return 0.0

    def red_lane_ys(self, k: int | None = None) -> List[float]:
        """Ingress pair lanes (plan §4.3): K=4 -> {30,90}; K=6 adds 60; K=8 adds
        {15,105}; K=12 (V2.7 E2 sweep cap) -> six lanes at 20 km spacing."""
        k = k or self.K_RED
        if k >= 12:
            return [10.0, 30.0, 50.0, 70.0, 90.0, 110.0]
        lanes = [30.0, 90.0]
        if k >= 6:
            lanes.append(60.0)
        if k >= 8:
            lanes.extend([15.0, 105.0])
        return sorted(lanes)

    def subzone_bounds(self, idx: int) -> Tuple[float, float]:
        """Lateral [y0, y1) of responsibility subzone idx (plan §6.2)."""
        w = self.L_Y / self.N_SUBZONES
        return idx * w, (idx + 1) * w
