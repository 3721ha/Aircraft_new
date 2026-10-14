"""Trainable-baseline helpers with no access to the proposed rule graph."""

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np
from scipy.optimize import Bounds, minimize

from .models import Action
from .shield import Intervention, SafetyShield


class PassThroughShield:
    """Preserve the shield API while executing the learned policy unchanged."""

    def filter(self, observations: Dict[int, dict], nominal: Dict[int, Action]):
        return {uid: action.clipped() for uid, action in nominal.items()}, []


@dataclass
class CBFQPSolveInfo:
    status: str = "not_solved"
    objective: float = 0.0
    minimum_barrier: float = 0.0
    infeasibility_reason: str | None = None


class PhysicalCBFQPShield:
    """Short-horizon deterministic CBF-QP for physical constraints only.

    The filter enforces flight-envelope and pairwise-separation barriers on
    the next decision step. It intentionally has no belief propagation,
    temporal-rule state, semantic-mode correction, or conflict arbitration.
    """

    def __init__(self, config, barrier_gain: float = 1.0, lookahead_steps: int = 5):
        self.cfg = getattr(config, "cfg", config)
        if not 0.0 < barrier_gain <= 1.0:
            raise ValueError("barrier_gain must be in (0, 1]")
        if lookahead_steps < 1:
            raise ValueError("lookahead_steps must be positive")
        self.barrier_gain = float(barrier_gain)
        self.lookahead_steps = int(lookahead_steps)
        self.last_solve = CBFQPSolveInfo()

    @staticmethod
    def _predict(observation: dict, controls, dt: float, steps: int = 1):
        own = observation["self"]
        turn, climb, acceleration = controls
        heading = float(own["heading"])
        climb_angle = 0.0
        speed = float(own["speed"])
        position = np.asarray(own["position"], dtype=float)
        for _ in range(steps):
            heading += turn * 0.22 * dt
            climb_angle = max(-0.45, min(0.45, climb_angle + climb * 0.08 * dt))
            speed = max(0.0, speed + acceleration * 12.0 * dt)
            position = position + np.asarray([
                speed * math.cos(climb_angle) * math.cos(heading) * dt,
                speed * math.cos(climb_angle) * math.sin(heading) * dt,
                speed * math.sin(climb_angle) * dt,
            ])
        return position, speed

    def filter(self, observations: Dict[int, dict], nominal: Dict[int, Action]):
        self.last_solve = CBFQPSolveInfo()
        ids = sorted(nominal)
        if not ids:
            self.last_solve = CBFQPSolveInfo(status="optimal")
            return {}, []
        nominal_controls = np.asarray([
            [nominal[uid].turn, nominal[uid].climb, nominal[uid].acceleration]
            for uid in ids
        ], dtype=float)
        x0 = nominal_controls.reshape(-1)

        def objective(x):
            residual = x - x0
            return 0.5 * float(np.dot(residual, residual))

        constraints = []
        current_positions = {
            uid: np.asarray(observations[uid]["self"]["position"], dtype=float)
            for uid in ids
        }
        for row, uid in enumerate(ids):
            observation = observations[uid]

            def min_speed(x, row=row, observation=observation):
                return self._predict(observation, x[row * 3:row * 3 + 3], self.cfg.dt)[1] - self.cfg.min_speed

            def max_speed(x, row=row, observation=observation):
                return self.cfg.max_speed - self._predict(observation, x[row * 3:row * 3 + 3], self.cfg.dt)[1]

            def min_altitude(x, row=row, observation=observation):
                return self._predict(observation, x[row * 3:row * 3 + 3], self.cfg.dt)[0][2] - self.cfg.min_altitude

            def max_altitude(x, row=row, observation=observation):
                return self.cfg.max_altitude - self._predict(observation, x[row * 3:row * 3 + 3], self.cfg.dt)[0][2]

            constraints.extend({"type": "ineq", "fun": fun} for fun in (min_speed, max_speed, min_altitude, max_altitude))

        required_squared = self.cfg.min_separation ** 2
        for left, uid in enumerate(ids):
            for right in range(left + 1, len(ids)):
                vid = ids[right]
                current_h = float(np.dot(current_positions[vid] - current_positions[uid], current_positions[vid] - current_positions[uid]) - required_squared)

                for prediction_step in range(1, self.lookahead_steps + 1):
                    def separation_barrier(
                        x,
                        left=left,
                        right=right,
                        uid=uid,
                        vid=vid,
                        current_h=current_h,
                        prediction_step=prediction_step,
                    ):
                        pi = self._predict(
                            observations[uid], x[left * 3:left * 3 + 3], self.cfg.dt, prediction_step
                        )[0]
                        pj = self._predict(
                            observations[vid], x[right * 3:right * 3 + 3], self.cfg.dt, prediction_step
                        )[0]
                        next_h = float(np.dot(pj - pi, pj - pi) - required_squared)
                        target_h = (1.0 - self.barrier_gain) ** prediction_step * current_h
                        return next_h - target_h

                    constraints.append({"type": "ineq", "fun": separation_barrier})

        result = None
        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=Bounds(-np.ones_like(x0), np.ones_like(x0)),
                constraints=constraints,
                options={"maxiter": 80, "ftol": 1e-7},
            )
        except Exception as exc:  # pragma: no cover - solver boundary
            self.last_solve = CBFQPSolveInfo(status="infeasible_fallback", infeasibility_reason=f"solver_error:{exc.__class__.__name__}")
        if result is not None and result.success and np.all(np.isfinite(result.x)):
            controls = result.x.reshape(len(ids), 3)
            barriers = [float(item["fun"](result.x)) for item in constraints]
            self.last_solve = CBFQPSolveInfo(
                status="optimal",
                objective=float(result.fun),
                minimum_barrier=min(barriers, default=0.0),
            )
        else:
            controls = nominal_controls
            if self.last_solve.status == "not_solved":
                self.last_solve = CBFQPSolveInfo(
                    status="infeasible_fallback",
                    infeasibility_reason="no_short_horizon_physical_feasible_residual",
                )

        safe = {}
        interventions = []
        for row, uid in enumerate(ids):
            original = nominal[uid]
            corrected = Action(
                original.mode,
                float(controls[row, 0]),
                float(controls[row, 1]),
                float(controls[row, 2]),
                original.target,
                original.target_kind,
            ).clipped()
            safe[uid] = corrected
            distance = SafetyShield._distance(original, corrected)
            if distance > 1e-8:
                interventions.append(Intervention(uid, original, corrected, ["instantaneous_physical_cbf"], distance))
        return safe, interventions


# Backward-compatible name for early experiment scripts.
InstantaneousCBFQPShield = PhysicalCBFQPShield
