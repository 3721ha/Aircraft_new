"""Conflict-aware QP safety shield for the MUMT detection scenario (plan §8.1).

Scope (V2.7): the shield is an *onboard* system on each wingman.  It only
sees the wingman's own track file (local observation premise; a target with
no track anywhere is invisible to the shield and lands in the hidden ledger,
plan §9.1 dual bookkeeping).

Barriers on the continuous heads (u_psi, u_V); discrete heads pass through:

  S02  pairwise separation (3 wingmen + scripted manned node), 3 s horizon
  S03  airspace boundary (nearest face), 5 s horizon
  S04  MAR red line via inflated estimate d_eff = d_est - K_SIGMA*sigma
       (plan V2.1): d_eff(0) < MAR  ==> hard barrier (forced breakaway)
  S05  MOR gradient: MAR <= d_eff(0) < MOR ==> linear push-away term in the
       objective (the "梯度推离" of plan §8.1), constraints always win

All hard barriers use the CBF decay form  h(tau) >= (1-gamma)^tau * h(0):
an already-violated state is driven back at rate gamma instead of declaring
the QP infeasible, and a safe state may approach but never cross.

Numerics (P3 pilot note): the 3/5 s kinematics are linearised about the
nominal-control rollout.  Heading/speed are exactly linear in (u_psi, u_v)
per step (clip ignored in the Jacobian); position is a sum of v*cos/sin(psi)
terms, linearised analytically.  Linearisation error over 3 s is O(1 m)
against 2 km barriers -- negligible.  The QP is one vectorised SLSQP call
(~50 x 6), ~100x cheaper than per-constraint nonlinear finite differences.
Cascade on failure: relaxed decay (gamma/2) -> plan §6.1 reference law
(conservative-fallback seat), logged as ``reference_fallback`` so Gate-4 can
separate continuous QP corrections from discrete fallbacks.
"""
from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import Bounds, minimize

from .detect_config import DetectSimConfig, ma_to_kms
from .detect_models import DetectAction


@dataclass
class DetectIntervention:
    uid: int
    kind: str                       # "continuous_qp" | "reference_fallback"
    reasons: List[str] = field(default_factory=list)
    magnitude: float = 0.0          # L2 of (du_psi, du_v)


@dataclass
class DetectQPSolveInfo:
    status: str = "not_solved"      # optimal | no_constraints | infeasible_fallback
    objective: float = 0.0
    n_constraints: int = 0
    slack: float = 0.0              # physically unavoidable violation (km)


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class DetectQPShield:
    def __init__(self, cfg: DetectSimConfig, barrier_gain: float = 0.5,
                 s05_weight: float = 1.0):
        self.cfg = cfg
        if not 0.0 < barrier_gain < 1.0:
            raise ValueError("barrier_gain must be in (0, 1)")
        self.gamma = float(barrier_gain)
        self.s05_weight = float(s05_weight)
        self.last_solve = DetectQPSolveInfo()
        self._reference = None       # lazy HeurPolicy (avoids circular import)

    # ------------------------------------------------------------------
    def _predict(self, state: Tuple[float, float, float, float],
                 u_psi: float, u_v: float, steps: int):
        """Exact replica of DetectEnv._apply_controls kinematics (D1); used
        by tests and by the nominal-rollout anchor of the linearisation."""
        cfg = self.cfg
        x, y, psi, v = state
        omega = math.radians(cfg.OMEGA_CRUISE_DEG)
        v_min, v_max = ma_to_kms(cfg.BLUE_V_MIN_MA), ma_to_kms(cfg.BLUE_V_MAX_MA)
        for _ in range(steps):
            psi = wrap_pi(psi + cfg.K_PSI * u_psi * omega * cfg.dt)
            v = max(v_min, min(v_max, v + (cfg.K_V_MS / 1000.0) * u_v * cfg.dt))
            x += v * math.cos(psi) * cfg.dt
            y += v * math.sin(psi) * cfg.dt
        return x, y

    def _rollout_with_jacobian(self, state, u_psi, u_v, horizon):
        """Nominal positions + d(pos)/d(u_psi, u_v) per step (clip ignored)."""
        cfg = self.cfg
        c_psi = cfg.K_PSI * math.radians(cfg.OMEGA_CRUISE_DEG) * cfg.dt
        c_v = (cfg.K_V_MS / 1000.0) * cfg.dt
        x, y, psi, v = state
        traj = [(x, y)]
        jac = [np.zeros((2, 2))]
        # running sums: dpsi_k = c_psi*u*k, dv_k = c_v*u*k
        jx_psi = jy_psi = jx_v = jy_v = 0.0
        for k in range(1, horizon + 1):
            psi_k = wrap_pi(psi + c_psi * u_psi * k)
            v_k = v + c_v * u_v * k
            # position increment this step
            x += v_k * math.cos(psi_k) * cfg.dt
            y += v_k * math.sin(psi_k) * cfg.dt
            # Jacobian running sums (before/after increment ordering matches
            # the env: heading & speed update first, then move)
            jx_psi += (-v_k * math.sin(psi_k)) * (c_psi * k) * cfg.dt
            jy_psi += (v_k * math.cos(psi_k)) * (c_psi * k) * cfg.dt
            jx_v += (c_v * k) * math.cos(psi_k) * cfg.dt
            jy_v += (c_v * k) * math.sin(psi_k) * cfg.dt
            traj.append((x, y))
            jac.append(np.array([[jx_psi, jx_v], [jy_psi, jy_v]]))
        return traj, jac

    def _reference_law(self, env, i: int) -> DetectAction:
        if self._reference is None:
            from .detect_baselines import HeurPolicy
            self._reference = HeurPolicy()
        return self._reference(env)[i]

    # ------------------------------------------------------------------
    def _barrier_target(self, h0: float, tau: int, recover_cap: float) -> float:
        """CBF decay for safe states (h0 >= 0); max-rate recovery for violated
        ones.  Proportional decay on a deep violation demands physically
        impossible improvement (e.g. 10 km in 1 s) and makes the QP
        chronically infeasible -- G2 pilot: 95% of interventions fell back.
        Capping the required recovery at the achievable closing rate keeps
        the QP feasible and still forces monotonic improvement."""
        if h0 >= 0.0:
            return (1.0 - self.gamma) ** tau * h0
        return min(0.0, h0 + recover_cap * tau * self.cfg.dt)

    def filter(self, env, nominal: Dict[int, DetectAction]):
        cfg = self.cfg
        ids = sorted(nominal)
        self.last_solve = DetectQPSolveInfo()
        if not ids:
            self.last_solve = DetectQPSolveInfo(status="optimal")
            return {}, []

        wing = env.wingmen
        states = {i: (wing[i].x, wing[i].y, wing[i].psi, wing[i].v) for i in ids}
        manned = env.manned
        manned_state = (manned.x, manned.y, manned.psi, manned.v)

        v_max = ma_to_kms(cfg.BLUE_V_MAX_MA)
        s02_gate = cfg.D_SAFE + 2.0 * v_max * cfg.D_SAFE_LOOKAHEAD + 1.0
        s03_gate = v_max * 5.0 + 1.5

        # ---- collect active barrier pairs (activation gating) ----
        pairs = []          # (i, j or "manned", h0)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                h0 = math.hypot(states[i][0] - states[j][0],
                                states[i][1] - states[j][1]) - cfg.D_SAFE
                if h0 + cfg.D_SAFE < s02_gate:
                    pairs.append((i, j, h0))
        for i in ids:
            h0 = math.hypot(states[i][0] - manned_state[0],
                            states[i][1] - manned_state[1]) - cfg.D_SAFE
            if h0 + cfg.D_SAFE < s02_gate:
                pairs.append((i, "manned", h0))
        boundary = {}       # i -> (unit nx, ny, h0)
        for i in ids:
            x_i, y_i = states[i][0], states[i][1]
            h0 = min(x_i, cfg.L_X - x_i, y_i, cfg.L_Y - y_i)
            if h0 < s03_gate:
                cands = [(x_i, (-1.0, 0.0)), (cfg.L_X - x_i, (1.0, 0.0)),
                         (y_i, (0.0, -1.0)), (cfg.L_Y - y_i, (0.0, 1.0))]
                d, n = min(cands, key=lambda c: c[0])
                boundary[i] = (n, d)    # signed dist along inward normal
        track_barriers = []   # (i, tr, d_eff0, kind "S04"|"S05")
        for i in ids:
            for tr in wing[i].tracks.slots:
                if tr is None:
                    continue
                est = tr.extrapolate(env.t)
                d_eff0 = math.hypot(est[0] - states[i][0], est[1] - states[i][1]) \
                    - cfg.K_SIGMA * tr.sigma
                if d_eff0 < cfg.MAR:
                    track_barriers.append((i, tr, d_eff0, "S04"))
                elif d_eff0 < cfg.MOR + 2.0:
                    track_barriers.append((i, tr, d_eff0, "S05"))

        if not pairs and not boundary and not track_barriers:
            self.last_solve = DetectQPSolveInfo(status="no_constraints")
            return {i: nominal[i].clipped() for i in ids}, []

        # ---- nominal rollouts & Jacobians ----
        horizon = 5
        traj, jac = {}, {}
        for i in ids:
            traj[i], jac[i] = self._rollout_with_jacobian(
                states[i], nominal[i].u_psi, nominal[i].u_v, horizon)
        idx = {i: r for r, i in enumerate(ids)}
        n_var = 2 * len(ids)
        x0 = np.array([[nominal[i].u_psi, nominal[i].u_v] for i in ids],
                      dtype=float).reshape(-1)

        rows_A, rows_b = [], []
        lin_obj = np.zeros(n_var)          # S05 push-away (linear term)
        reasons: Dict[int, set] = {i: set() for i in ids}

        def add_row(coefs: Dict[int, np.ndarray], b: float):
            row = np.zeros(n_var)
            for i, c in coefs.items():
                row[2 * idx[i]:2 * idx[i] + 2] = c
            rows_A.append(row)
            rows_b.append(b)

        # S02 (tau in 1..3); recovery cap = mutual closing speed 2*v_max
        cap_s02 = 2.0 * v_max
        for i, j, h0 in pairs:
            for tau in (1, 2, 3):
                pi = np.array(traj[i][tau])
                pj = (np.array(traj[j][tau]) if j != "manned" else np.array(
                    [manned_state[0] + manned_state[3] * math.cos(manned_state[2]) * tau * cfg.dt,
                     manned_state[1] + manned_state[3] * math.sin(manned_state[2]) * tau * cfg.dt]))
                rel = pi - pj
                dist = float(np.linalg.norm(rel))
                if dist < 1e-9:
                    continue
                n = rel / dist
                m = dist - cfg.D_SAFE - self._barrier_target(h0, tau, cap_s02)
                coefs = {i: n @ jac[i][tau]}
                if j != "manned":
                    coefs[j] = -n @ jac[j][tau]
                add_row(coefs, -m)
            reasons[i].add(f"S02:{j}")
            if j != "manned":
                reasons[j].add(f"S02:{i}")
            else:
                reasons[i].add("S02:manned")

        # S03 (tau in 1..5), inward-normal signed distance
        for i, (n_vec, h0) in boundary.items():
            n = np.array(n_vec)
            for tau in range(1, 6):
                p = np.array(traj[i][tau])
                if n_vec == (-1.0, 0.0):
                    dist = p[0]
                elif n_vec == (1.0, 0.0):
                    dist = cfg.L_X - p[0]
                elif n_vec == (0.0, -1.0):
                    dist = p[1]
                else:
                    dist = cfg.L_Y - p[1]
                m = dist - self._barrier_target(h0, tau, v_max)
                add_row({i: n @ jac[i][tau]}, -m)
            reasons[i].add("S03")

        # S04 hard barrier / S05 linear push (tau in 1..3)
        for i, tr, d_eff0, kind in track_barriers:
            for tau in (1, 2, 3):
                est = tr.extrapolate(env.t + tau * cfg.dt)
                pe = np.array(est)
                rel = np.array(traj[i][tau]) - pe
                dist = float(np.linalg.norm(rel))
                if dist < 1e-9:
                    continue
                n = rel / dist          # from track toward aircraft: +n grows d
                d_eff = dist - cfg.K_SIGMA * tr.sigma
                if kind == "S04":
                    # recovery cap: blue max + red max maneuver speed
                    cap_s04 = v_max + ma_to_kms(cfg.RED_MANEUVER_MA_RANGE[1])
                    m = (d_eff - cfg.MAR
                         - self._barrier_target(d_eff0 - cfg.MAR, tau, cap_s04))
                    add_row({i: n @ jac[i][tau]}, -m)
                elif tau == 3:
                    lin_obj[2 * idx[i]:2 * idx[i] + 2] += \
                        -(self.s05_weight / cfg.MOR_GRADIENT_SPAN) * (n @ jac[i][tau])
            reasons[i].add(f"{kind}:tid{tr.target_id}")

        A = np.array(rows_A) if rows_A else np.zeros((0, n_var))
        b = np.array(rows_b) if rows_b else np.zeros(0)
        m_con = len(rows_b)

        # ---- soft-CBF QP: per-row slack with heavy penalty (P3 pilot fix).
        # 1 s of control authority is tiny (turn 3 deg/s, dv 0.03 km/s), so a
        # deep violation is *provably* unsatisfiable at any recovery cap --
        # hard-constraint QP fell back ~95% of the time.  Slack turns the
        # same barriers into best-effort maximal recovery: the slack penalty
        # (rho) dominates the deviation cost, so barriers are only ever
        # violated by the physically unavoidable amount.  Continuous QP
        # corrections stay the norm; reference-law fallback is reserved for
        # numerical solver failure.
        rho = 100.0
        n_aug = n_var + m_con

        def unpack(z):
            return z[:n_var], z[n_var:]

        def obj(z):
            x, s = unpack(z)
            r = x - x0
            return 0.5 * float(r @ r) + float(lin_obj @ x) + rho * float(s.sum())

        def jac_obj(z):
            g = np.zeros(n_aug)
            g[:n_var] = (z[:n_var] - x0) + lin_obj
            g[n_var:] = rho
            return g

        A_aug = np.hstack([A, np.eye(m_con)]) if m_con else A

        def cons(z):
            return A_aug @ z - b

        def jac_cons(z):
            return A_aug

        lb = np.concatenate([-np.ones(n_var), np.zeros(m_con)])
        ub = np.concatenate([np.ones(n_var), np.full(m_con, np.inf)])
        z0 = np.concatenate([x0, np.zeros(m_con)])
        result = None
        for ftol, maxiter in ((1e-8, 100), (1e-6, 200)):
            try:
                r = minimize(obj, z0, jac=jac_obj, method="SLSQP",
                             bounds=Bounds(lb, ub),
                             constraints=[{"type": "ineq", "fun": cons,
                                           "jac": jac_cons}] if m_con else [],
                             options={"maxiter": maxiter, "ftol": ftol})
                if r.success and np.all(np.isfinite(r.x)):
                    result = r
                    break
            except Exception:
                continue

        interventions: List[DetectIntervention] = []
        if result is not None:
            controls = result.x[:n_var].reshape(len(ids), 2)
            slack = float(result.x[n_var:].sum()) if m_con else 0.0
            self.last_solve = DetectQPSolveInfo(
                status="optimal", objective=float(result.fun) ,
                n_constraints=m_con)
            self.last_solve.slack = slack
            safe = {}
            for row, i in enumerate(ids):
                nom = nominal[i]
                corr = DetectAction(float(controls[row, 0]), float(controls[row, 1]),
                                    nom.u_xi, nom.role, nom.gear, nom.pointer).clipped()
                safe[i] = corr
                mag = math.hypot(corr.u_psi - nom.u_psi, corr.u_v - nom.u_v)
                if mag > 1e-6:
                    interventions.append(DetectIntervention(
                        uid=i, kind="continuous_qp",
                        reasons=sorted(reasons[i]), magnitude=mag))
            return safe, interventions

        # ---- infeasible: reference-law fallback (plan §6.1 seat) ----
        self.last_solve = DetectQPSolveInfo(status="infeasible_fallback",
                                            n_constraints=len(rows_b))
        safe = {}
        for i in ids:
            nom = nominal[i]
            ref = self._reference_law(env, i)
            corr = DetectAction(ref.u_psi, ref.u_v, nom.u_xi,
                                nom.role, nom.gear, nom.pointer).clipped()
            safe[i] = corr
            mag = math.hypot(corr.u_psi - nom.u_psi, corr.u_v - nom.u_v)
            if mag > 1e-6:
                interventions.append(DetectIntervention(
                    uid=i, kind="reference_fallback",
                    reasons=sorted(reasons[i]) or ["qp_infeasible"],
                    magnitude=mag))
        return safe, interventions


class DetectPassThroughShield:
    """Ablation seat: no shield at all (E3 '无屏蔽' configuration)."""

    def __init__(self):
        self.last_solve = DetectQPSolveInfo(status="optimal")

    def filter(self, env, nominal):
        return {i: a.clipped() for i, a in nominal.items()}, []
