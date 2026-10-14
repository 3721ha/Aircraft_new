"""JSBSim-backed multi-aircraft validation environment.

Each friendly aircraft owns an independent six-degree-of-freedom JSBSim FDM
instance. High-level semantic actions are tracked at the research simulator's
decision rate and converted to bounded attitude/throttle commands by a simple
inner-loop controller. The class preserves AircraftEnv's observation, rule,
communication and logging contracts so the policy and shield are unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .config import SimConfig
from .env import AircraftEnv
from .models import Action, Mode, StepResult


FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M
EARTH_RADIUS_M = 6_371_000.0


class JSBSimAircraftEnv(AircraftEnv):
    """Multi-instance JSBSim environment using the bundled GPL F-16 model."""

    backend_name = "jsbsim_f16_6dof"

    def __init__(
        self,
        n_aircraft: int = 3,
        config: Optional[SimConfig] = None,
        *,
        aircraft_model: str = "f16",
        integration_hz: int = 120,
        origin_latitude_deg: float = 35.0,
        origin_longitude_deg: float = 110.0,
    ):
        super().__init__(n_aircraft, config)
        if integration_hz < 20:
            raise ValueError("integration_hz must be at least 20 Hz")
        try:
            import jsbsim
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise RuntimeError("JSBSimAircraftEnv requires the optional 'jsbsim' package") from exc
        self._jsbsim = jsbsim
        self.aircraft_model = aircraft_model
        self.integration_hz = int(integration_hz)
        self.integration_dt = 1.0 / self.integration_hz
        self.origin_latitude_deg = float(origin_latitude_deg)
        self.origin_longitude_deg = float(origin_longitude_deg)
        self._fdms = []
        self._initial_positions = []
        self._fdm_reference = []

    @property
    def dynamics_metadata(self) -> dict:
        return {
            "backend": self.backend_name,
            "aircraft_model": self.aircraft_model,
            "integration_hz": self.integration_hz,
            "decision_dt": self.cfg.dt,
            "model_source": "JSBSim bundled aircraft XML",
            "model_license": "GPL as declared by aircraft/f16/f16.xml",
            "truth_access_for_policy": False,
        }

    def _local_to_geodetic(self, north_m: float, east_m: float):
        latitude = self.origin_latitude_deg + math.degrees(north_m / EARTH_RADIUS_M)
        longitude = self.origin_longitude_deg + math.degrees(
            east_m / (EARTH_RADIUS_M * math.cos(math.radians(self.origin_latitude_deg)))
        )
        return latitude, longitude

    def _geodetic_to_local(self, latitude_deg: float, longitude_deg: float):
        north = math.radians(latitude_deg - self.origin_latitude_deg) * EARTH_RADIUS_M
        east = math.radians(longitude_deg - self.origin_longitude_deg) * EARTH_RADIUS_M * math.cos(math.radians(self.origin_latitude_deg))
        return north, east

    def _make_fdm(self, state):
        fdm = self._jsbsim.FGFDMExec(None)
        fdm.set_debug_level(0)
        if not fdm.load_model(self.aircraft_model):
            raise RuntimeError(f"unable to load JSBSim aircraft model '{self.aircraft_model}'")
        fdm.set_dt(self.integration_dt)
        latitude, longitude = self._local_to_geodetic(state.position[0], state.position[1])
        initial = {
            "ic/h-sl-ft": state.position[2] * M_TO_FT,
            "ic/vt-fps": state.speed * M_TO_FT,
            "ic/psi-true-deg": math.degrees(state.heading) % 360.0,
            "ic/phi-deg": math.degrees(state.roll),
            "ic/theta-deg": math.degrees(state.pitch),
            "ic/lat-geod-deg": latitude,
            "ic/long-gc-deg": longitude,
        }
        for key, value in initial.items():
            fdm[key] = value
        if not fdm.run_ic():
            raise RuntimeError(f"JSBSim initial condition failed for aircraft {state.uid}")
        fdm["gear/gear-cmd-norm"] = 0.0
        fdm["fcs/throttle-cmd-norm"] = 0.5
        fdm["fcs/aileron-cmd-norm"] = 0.0
        fdm["fcs/elevator-cmd-norm"] = 0.0
        fdm["fcs/rudder-cmd-norm"] = 0.0
        return fdm

    def reset(self, scenario: Optional[dict] = None) -> dict:
        observations = super().reset(scenario)
        self._initial_positions = [tuple(state.position) for state in self.aircraft]
        self._fdms = [self._make_fdm(state) for state in self.aircraft]
        self._fdm_reference = [
            (
                float(fdm["position/from-start-neu-n-ft"]) * FT_TO_M,
                float(fdm["position/from-start-neu-e-ft"]) * FT_TO_M,
                float(fdm["position/h-sl-ft"]) * FT_TO_M,
            )
            for fdm in self._fdms
        ]
        for uid in range(self.n):
            self._sync_state(uid)
        return self.observe()

    @staticmethod
    def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, value))

    def _apply_inner_loop(self, fdm, action: Action, uid: int):
        roll_deg = float(fdm["attitude/phi-deg"])
        gamma_deg = math.degrees(math.atan2(-float(fdm["velocities/v-down-fps"]), max(1e-6, math.hypot(float(fdm["velocities/v-north-fps"]), float(fdm["velocities/v-east-fps"])))))
        p_deg_s = math.degrees(float(fdm["velocities/p-rad_sec"]))
        q_deg_s = math.degrees(float(fdm["velocities/q-rad_sec"]))
        r_deg_s = math.degrees(float(fdm["velocities/r-rad_sec"]))
        true_speed_mps = max(1.0, float(fdm["velocities/vtrue-fps"]) * FT_TO_M)
        desired_heading_rate = 0.22 * action.turn
        target_roll = math.degrees(math.atan(desired_heading_rate * true_speed_mps / 9.80665))
        target_roll = self._clip(target_roll, -75.0, 75.0)
        target_gamma = 12.0 * action.climb
        if action.mode == Mode.RECOVER:
            target_gamma += 8.0 if uid % 2 == 0 else -8.0
            target_gamma = self._clip(target_gamma, -15.0, 15.0)
        aileron = self._clip(0.035 * (target_roll - roll_deg) - 0.015 * p_deg_s)
        # Negative elevator command pitches the bundled F-16 model upward.
        elevator = self._clip(-0.035 * (target_gamma - gamma_deg) + 0.010 * q_deg_s)
        rudder = self._clip(-0.015 * r_deg_s, -0.35, 0.35)
        throttle = self._clip(0.52 + 0.30 * action.acceleration, 0.15, 0.92)
        fdm["fcs/aileron-cmd-norm"] = aileron
        fdm["fcs/elevator-cmd-norm"] = elevator
        fdm["fcs/rudder-cmd-norm"] = rudder
        fdm["fcs/throttle-cmd-norm"] = throttle
        return {"aileron": aileron, "elevator": elevator, "rudder": rudder, "throttle": throttle}

    def _sync_state(self, uid: int):
        fdm = self._fdms[uid]
        state = self.aircraft[uid]
        initial = self._initial_positions[uid]
        reference = self._fdm_reference[uid]
        north = initial[0] + float(fdm["position/from-start-neu-n-ft"]) * FT_TO_M - reference[0]
        east = initial[1] + float(fdm["position/from-start-neu-e-ft"]) * FT_TO_M - reference[1]
        values = [
            north,
            east,
            initial[2] + float(fdm["position/h-sl-ft"]) * FT_TO_M - reference[2],
            float(fdm["velocities/vtrue-fps"]) * FT_TO_M,
            float(fdm["attitude/psi-rad"]),
            float(fdm["attitude/phi-rad"]),
            float(fdm["attitude/theta-rad"]),
            float(fdm["aero/alpha-rad"]),
            float(fdm["accelerations/Nz"]),
        ]
        if not all(math.isfinite(value) for value in values):
            state.alive = False
            return
        state.position = (values[0], values[1], values[2])
        state.speed = values[3]
        state.heading = values[4] % (2.0 * math.pi)
        state.roll = values[5]
        state.pitch = values[6]
        state.angle_of_attack = values[7]
        state.load_factor = values[8]
        horizontal = math.hypot(float(fdm["velocities/v-north-fps"]), float(fdm["velocities/v-east-fps"]))
        state.climb_angle = math.atan2(-float(fdm["velocities/v-down-fps"]), max(1e-6, horizontal))

    def _update_semantics_before_dynamics(self, actions: Dict[int, Action]):
        for i, state in enumerate(self.aircraft):
            if not state.alive:
                continue
            action = actions[i]
            previous_role = state.role
            state.role = action.mode
            previous_mode_age = state.mode_age
            state.dwell_violation = previous_role != action.mode and 0 < previous_mode_age < self.cfg.mode_dwell_steps
            state.mode_age = 1 if previous_role != action.mode else state.mode_age + 1
            state.handover_confirmed = previous_role == action.mode or state.intent_shared or action.mode in (Mode.RECOVER, Mode.EXIT)
            state.hazard_previous = state.hazard_active
            state.intent_shared = bool(state.c2_connected)
            state.data_age = 0 if not state.sensor_degraded else state.data_age + 1
            state.task_progress = min(1.0, state.task_progress + (self.cfg.task_progress_rate if action.mode in (Mode.SUPPORT, Mode.COVER, Mode.CRITICAL) else 0.002))
            if action.mode in (Mode.SUPPORT, Mode.CRITICAL):
                state.last_task_action_step = self.step_count
                state.assessment_due = True
            elif state.assessment_due and action.mode in (Mode.HOLD, Mode.RECOVER, Mode.EXIT):
                state.assessment_due = False

    def _update_semantics_after_dynamics(self, actions: Dict[int, Action]):
        for i, state in enumerate(self.aircraft):
            if not state.alive:
                continue
            action = actions[i]
            effort = abs(action.turn) + abs(action.climb) + 0.25 * abs(action.acceleration)
            state.energy = max(0.0, min(1.0, state.energy - 0.006 * effort + 0.0015))
            state.resource = max(0.0, state.resource - self.cfg.resource_drain * (1.5 if action.mode in (Mode.SUPPORT, Mode.CRITICAL) else 1.0))
            if state.resource < self.cfg.low_resource:
                state.capability = max(0.2, state.resource / max(self.cfg.low_resource, 1e-6))
            if state.c2_connected and self.rng.random() < self.cfg.communication_drop * 0.02:
                state.c2_connected = False
            if not state.c2_connected and self.rng.random() < 0.03:
                state.c2_connected = True
            state.last_message_step = self.step_count if state.c2_connected else state.last_message_step
            state.warning_pending = self.risk_of(i) > self.cfg.support_risk_threshold
            state.warning_age = 0 if state.c2_connected else state.warning_age + (1 if state.warning_pending else 0)
            if (state.warning_pending or state.support_requested) and state.pending_support_since is None:
                state.pending_support_since = self.step_count
            if not state.warning_pending and not state.support_requested:
                state.pending_support_since = None
            envelope_margin = min(
                state.speed - self.cfg.min_speed,
                self.cfg.max_speed - state.speed,
                state.position[2] - self.cfg.min_altitude,
                self.cfg.max_altitude - state.position[2],
                self.cfg.max_abs_aoa_deg - abs(math.degrees(state.angle_of_attack)),
                state.load_factor - self.cfg.min_load_factor,
                self.cfg.max_load_factor - state.load_factor,
            )
            state.hazard_active = envelope_margin < self.cfg.hysteresis_margin or self.risk_of(i) > 0.7
            state.safe_dwell = state.safe_dwell + 1 if not state.hazard_active else 0
            state.intervention_window.append(0)
            if len(state.intervention_window) > self.cfg.replan_intervention_window:
                state.intervention_window.pop(0)

    def step(self, actions: Dict[int, Action]) -> StepResult:
        actions = {i: actions.get(i, Action()).clipped() for i in range(self.n)}
        self._update_semantics_before_dynamics(actions)
        substeps = max(1, int(round(self.cfg.dt * self.integration_hz)))
        applied_controls = {}
        for _ in range(substeps):
            for i, fdm in enumerate(self._fdms):
                if not self.aircraft[i].alive:
                    continue
                applied_controls[i] = self._apply_inner_loop(fdm, actions[i], i)
                if not fdm.run():
                    self.aircraft[i].alive = False
        for uid in range(self.n):
            if self.aircraft[uid].alive:
                self._sync_state(uid)
        self._update_semantics_after_dynamics(actions)
        self.step_count += 1
        self.communication_log.append({"step": self.step_count, "connected": [a.c2_connected for a in self.aircraft], "intent_shared": [a.intent_shared for a in self.aircraft]})
        truth = self.truth_state()
        reward = self._reward(truth, actions)
        done = self.step_count >= self.cfg.horizon or not any(a.alive for a in self.aircraft)
        return StepResult(self.observe(), reward, done, {"truth": truth, "actions": actions, "dynamics": self.dynamics_metadata, "inner_loop_controls": applied_controls})
