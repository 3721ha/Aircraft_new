"""Observation encoder (138-dim, plan V2.4 §7.1) + action codec (§7.2).

Encoding discipline (V1.23, plan §7.1 preamble): circular quantities as
sin/cos, geometry relative to own ship, zero absolute coordinates.

Block layout (total 138 = 17 + 8*12 + 2*6 + 4 + 6 + 3):

    [0:17)    own state: boundary d/40 + inward-normal sin/cos (3),
              sin psi / cos psi (2), V/(0.95 Ma) (1), gear one-hot (5),
              xi/60 (1), role one-hot (2), role_age/T_dwell (1),
              radar on (1), radiate duty window (1, constant 1.0 in V2.4)
    [17:113)  own track cluster, 8 slots x 12 (track.TrackFile.features);
              slots with age > AGE_DEGRADE are down-weighted x0.5 here
    [113:125) teammates x2: dx/R, dy/R, sin psi, cos psi, V/(0.95 Ma),
              has-pointer flag
    [125:129) manned: dx/R, dy/R, V/(0.95 Ma), link state
    [129:135) assignment table: owned/8, subzone one-hot (3),
              coverage margin (T_scan_req - t_since)/T_scan_req, table age/2
    [135:138) area posture: detection-zone est/16, tracking-zone est/16,
              judged/16
"""
import math
from typing import Dict, List

import numpy as np

from .detect_config import A_MACH_KMS, DetectSimConfig
from .detect_models import DetectAction, DetectRole

GEAR_ORDER = (7.5, 15.0, 30.0, 45.0, 60.0)
V_NORM = 0.95 * A_MACH_KMS
AGE_DOWNWEIGHT = 0.5          # age > 8 s feature scaling (plan §7.1, V1.23)


class DetectObservationEncoder:
    dimension = 138

    def __init__(self, cfg: DetectSimConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    def encode(self, obs: dict) -> np.ndarray:
        cfg = self.cfg
        w = obs["self"]                    # BlueWingman
        out: List[float] = []

        # -- own state (17) --
        d_bound = min(w.x, cfg.L_X - w.x, w.y, cfg.L_Y - w.y)
        # inward normal of the nearest boundary (absolute angle)
        candidates = [(w.x, math.pi), (cfg.L_X - w.x, 0.0),
                      (w.y, math.pi / 2), (cfg.L_Y - w.y, -math.pi / 2)]
        _, n_ang = min(candidates, key=lambda c: c[0])
        rel = n_ang - w.psi
        out += [min(1.0, d_bound / 40.0), math.sin(rel), math.cos(rel),
                math.sin(w.psi), math.cos(w.psi),
                min(1.0, w.v / V_NORM)]
        out += [1.0 if abs(w.radar.gear - g) < 1e-6 else 0.0 for g in GEAR_ORDER]
        out += [w.radar.axis_deg / cfg.SECTOR_AXIS_MAX_DEG]
        out += [1.0 if w.role == DetectRole.SEARCH else 0.0,
                1.0 if w.role == DetectRole.TRACK else 0.0]
        out += [min(1.0, w.role_age / cfg.T_DWELL), 1.0, 1.0]

        # -- own track cluster (8 x 12) with age>8 s down-weight --
        feats = obs["tracks"]
        for slot in feats:
            if slot[11] > 0 and slot[5] * cfg.AGE_DROP > cfg.AGE_DEGRADE:
                out += [v * AGE_DOWNWEIGHT for v in slot]
            else:
                out += slot

        # -- teammates (2 x 6) --
        for k in sorted(obs["teammates"]):
            tm = obs["teammates"][k]
            out += [tm["dx"] / cfg.R_MAX, tm["dy"] / cfg.R_MAX,
                    math.sin(tm["psi"]), math.cos(tm["psi"]),
                    min(1.0, tm["v"] / V_NORM), tm["has_pointer"]]

        # -- manned (4) --
        mn = obs["manned"]
        out += [mn["dx"] / cfg.R_MAX, mn["dy"] / cfg.R_MAX,
                min(1.0, mn["v"] / V_NORM), mn["link"]]

        # -- assignment table (6) --
        cov_age = obs["coverage_max_age"]
        out += [len(w.owned) / cfg.TRACK_SLOTS]
        out += [1.0 if w.subzone == s else 0.0 for s in range(cfg.N_SUBZONES)]
        out += [max(-2.0, (cfg.T_SCAN_REQ - cov_age) / cfg.T_SCAN_REQ),
                obs["mission"]["table_age"] / 2.0]

        # -- area posture (3) --
        ms = obs["mission"]
        out += [ms["n_detect_est"] / 16.0, ms["n_track_est"] / 16.0,
                ms["n_judged"] / 16.0]

        vec = np.asarray(out, dtype=np.float32)
        assert vec.shape == (self.dimension,), f"obs dim {vec.shape} != 138"
        return vec


class DetectActionCodec:
    """Hybrid action codec: continuous 3 + discrete (role 2, gear 5, pointer 9).

    P3 consumes this from the policy heads; the env applies interface masking
    (role-gear, link-degraded requests) regardless.
    """

    CONT_DIM = 3
    N_ROLE = 2
    N_GEAR = 5
    N_POINTER = 9

    def to_action(self, cont, role_idx: int, gear_idx: int, pointer_idx: int) -> DetectAction:
        u = [max(-1.0, min(1.0, float(c))) for c in cont[:3]]
        role = DetectRole.SEARCH if int(role_idx) == 0 else DetectRole.TRACK
        gear = GEAR_ORDER[int(gear_idx) % self.N_GEAR]
        pointer = max(0, min(self.N_POINTER - 1, int(pointer_idx)))
        return DetectAction(u[0], u[1], u[2], role, gear, pointer)

    def heads(self, action: DetectAction) -> Dict[str, int]:
        return {"role": 0 if action.role == DetectRole.SEARCH else 1,
                "gear": GEAR_ORDER.index(action.gear) if action.gear in GEAR_ORDER else 4,
                "pointer": max(0, min(self.N_POINTER - 1, action.pointer))}
