import math
import random
from typing import Dict, List, Optional, Tuple

from .config import SimConfig
from .models import Action, AircraftState, Mode, StepResult


def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class AircraftEnv:
    """Fast decision-level environment with hidden truth and noisy local observations."""

    def __init__(self, n_aircraft: int = 3, config: Optional[SimConfig] = None):
        self.cfg = config or SimConfig()
        self.n = n_aircraft
        self.rng = random.Random(self.cfg.seed)
        self.step_count = 0
        self.aircraft: List[AircraftState] = []
        self.opponent = (7000.0, 0.0, 5000.0)
        self.mission_active = True
        self.coverage_required = self.cfg.min_coverage
        self.communication_log: List[dict] = []

    def reset(self, scenario: Optional[dict] = None) -> dict:
        scenario = scenario or {}
        self.step_count = 0
        self.mission_active = True
        self.communication_log = []
        default_positions = [(-2000.0 + i * 1400.0, i * 500.0, 5000.0 + i * 300.0) for i in range(self.n)]
        positions = list(scenario.get("positions") or default_positions)
        resources = list(scenario.get("resources") or [1.0] * self.n)
        energies = list(scenario.get("energies") or [0.8] * self.n)
        capabilities = list(scenario.get("capabilities") or [1.0] * self.n)
        if len(positions) < self.n or len(resources) < self.n or len(energies) < self.n or len(capabilities) < self.n:
            raise ValueError(f"scenario requires positions/resources/energies for all {self.n} aircraft")
        self.aircraft = [AircraftState(uid=i, position=positions[i], heading=0.0 if i % 2 == 0 else math.pi, resource=resources[i], energy=energies[i], capability=capabilities[i]) for i in range(self.n)]
        self.opponent = tuple(scenario.get("opponent", self.opponent))
        for i, a in enumerate(self.aircraft):
            a.c2_connected = not scenario.get("c2_lost", {}).get(i, False)
            a.sensor_degraded = scenario.get("sensor_degraded", {}).get(i, False)
            a.id_confidence = scenario.get("id_confidence", {}).get(i, 0.95)
            a.authorization_valid = scenario.get("authorization", {}).get(i, True)
            a.intent_shared = not scenario.get("intent_unsent", {}).get(i, False)
            a.data_age = int(scenario.get("data_age", {}).get(i, 0))
            a.task_progress = float(scenario.get("task_progress", {}).get(i, 0.0))
            a.handover_confirmed = True
            a.source_disagreement = float(scenario.get("source_disagreement", {}).get(i, 0.0))
            a.support_requested = bool(scenario.get("support_requested", {}).get(i, False))
            a.pending_support_since = 0 if a.support_requested else None
            a.task_authority = bool(scenario.get("task_authority", {}).get(i, i == 0))
            a.hazard_active = False
            a.hazard_previous = False
            a.safe_dwell = 0
            a.warning_pending = False
            a.warning_age = 0
            a.dwell_violation = False
        return self.observe()

    def _observation_for(self, i: int) -> dict:
        a = self.aircraft[i]
        def noisy(v, sigma): return v + self.rng.gauss(0.0, sigma)
        own = {"position": tuple(noisy(v, self.cfg.noise_position) for v in a.position), "speed": noisy(a.speed, self.cfg.noise_speed), "energy": max(0.0, min(1.0, noisy(a.energy, 0.05))), "resource": max(0.0, min(1.0, noisy(a.resource, 0.05))), "capability": max(0.0, min(1.0, noisy(a.capability, 0.03))), "heading": a.heading, "roll": a.roll, "pitch": a.pitch, "angle_of_attack": a.angle_of_attack, "load_factor": a.load_factor, "role": a.role.value, "mode_age": a.mode_age, "dwell_violation": a.dwell_violation, "c2_connected": a.c2_connected, "sensor_degraded": a.sensor_degraded, "id_confidence": max(0.0, min(1.0, noisy(a.id_confidence, 0.04))), "authorization_valid": a.authorization_valid, "intent_shared": a.intent_shared, "data_age": a.data_age, "task_progress": a.task_progress, "handover_confirmed": a.handover_confirmed, "source_disagreement": a.source_disagreement, "warning_pending": a.warning_pending, "warning_age": a.warning_age, "support_requested": a.support_requested, "assessment_due": a.assessment_due, "replan_requested": a.replan_requested, "task_authority": a.task_authority, "hazard_active": a.hazard_active, "hazard_previous": a.hazard_previous, "safe_dwell": a.safe_dwell}
        teammates = []
        for j, b in enumerate(self.aircraft):
            if j == i or not b.alive or self.rng.random() < self.cfg.sensor_drop:
                continue
            teammates.append({
                "uid": j,
                "relative_position": tuple(
                    noisy(b.position[k] - a.position[k], self.cfg.noise_position)
                    for k in range(3)
                ),
                "role": b.role.value,
                "risk": max(0.0, min(1.0, noisy(self.risk_of(j), 0.08))),
                # Public status fields are available only when this teammate
                # survives the observer's sensor/communication dropout above.
                "speed": noisy(b.speed, self.cfg.noise_speed),
                "heading": b.heading,
                "energy": max(0.0, min(1.0, noisy(b.energy, 0.05))),
                "resource": max(0.0, min(1.0, noisy(b.resource, 0.05))),
                "capability": max(0.0, min(1.0, noisy(b.capability, 0.03))),
                "c2_connected": b.c2_connected,
                "sensor_degraded": b.sensor_degraded,
                "id_confidence": max(
                    0.0, min(1.0, noisy(b.id_confidence, 0.04))
                ),
                "authorization_valid": b.authorization_valid,
                "intent_shared": b.intent_shared,
                "data_age": b.data_age,
                "task_progress": b.task_progress,
                "handover_confirmed": b.handover_confirmed,
                "source_disagreement": b.source_disagreement,
                "warning_pending": b.warning_pending,
                "warning_age": b.warning_age,
                "support_requested": b.support_requested,
                "assessment_due": b.assessment_due,
                "replan_requested": b.replan_requested,
                "task_authority": b.task_authority,
                "hazard_active": b.hazard_active,
                "hazard_previous": b.hazard_previous,
                "safe_dwell": b.safe_dwell,
                "mode_age": b.mode_age,
                "dwell_violation": b.dwell_violation,
            })
        own["communication_load"] = len(teammates) / max(1, self.n - 1)
        return {"self": own, "teammates": teammates, "mission": {"active": self.mission_active, "step": self.step_count, "task_deadline": self.cfg.task_deadline, "task_progress": a.task_progress, "risk_budget": max(0.0, min(1.0, 0.5 * a.energy + 0.5 * a.resource)), "opponent_bearing": math.atan2(self.opponent[1] - a.position[1], self.opponent[0] - a.position[0]) + self.rng.gauss(0, 0.08)}}

    def observe(self) -> Dict[int, dict]:
        return {i: self._observation_for(i) for i in range(self.n) if self.aircraft[i].alive}

    def risk_of(self, i: int) -> float:
        a = self.aircraft[i]
        separation = min((_dist(a.position, b.position) for j, b in enumerate(self.aircraft) if j != i and b.alive), default=99999.0)
        risk = max(0.0, (self.cfg.min_separation * 2.0 - separation) / (self.cfg.min_separation * 2.0))
        risk += max(0.0, (self.cfg.low_resource - a.resource) * 1.4)
        return min(1.0, risk)

    def step(self, actions: Dict[int, Action]) -> StepResult:
        actions = {i: actions.get(i, Action()).clipped() for i in range(self.n)}
        for i, a in enumerate(self.aircraft):
            if not a.alive:
                continue
            act = actions[i]
            previous_role = a.role
            a.role = act.mode
            previous_mode_age = a.mode_age
            a.dwell_violation = previous_role != act.mode and 0 < previous_mode_age < self.cfg.mode_dwell_steps
            a.mode_age = 1 if previous_role != act.mode else a.mode_age + 1
            a.handover_confirmed = previous_role == act.mode or a.intent_shared or act.mode in (Mode.RECOVER, Mode.EXIT)
            a.hazard_previous = a.hazard_active
            envelope_margin = min(a.speed - self.cfg.min_speed, self.cfg.max_speed - a.speed, a.position[2] - self.cfg.min_altitude, self.cfg.max_altitude - a.position[2])
            a.hazard_active = envelope_margin < self.cfg.hysteresis_margin or self.risk_of(i) > 0.7
            a.safe_dwell = a.safe_dwell + 1 if not a.hazard_active else 0
            a.intent_shared = bool(a.c2_connected)
            a.data_age = 0 if not a.sensor_degraded else a.data_age + 1
            a.task_progress = min(1.0, a.task_progress + (self.cfg.task_progress_rate if act.mode in (Mode.SUPPORT, Mode.COVER, Mode.CRITICAL) else 0.002))
            if act.mode in (Mode.SUPPORT, Mode.CRITICAL):
                a.last_task_action_step = self.step_count
                a.assessment_due = True
            # HOLD/RECOVER/EXIT represent a deliberate assessment or
            # fallback phase.  They close the pending assessment obligation;
            # without this transition M03 could remain permanently pending
            # even though the aircraft had already stopped task execution.
            elif a.assessment_due and act.mode in (Mode.HOLD, Mode.RECOVER, Mode.EXIT):
                a.assessment_due = False
            a.heading += act.turn * 0.22 * self.cfg.dt
            a.climb_angle = max(-0.45, min(0.45, a.climb_angle + act.climb * 0.08 * self.cfg.dt))
            a.speed = max(self.cfg.min_speed * 0.7, min(self.cfg.max_speed, a.speed + act.acceleration * 12.0 * self.cfg.dt))
            dx = a.speed * math.cos(a.climb_angle) * math.cos(a.heading) * self.cfg.dt
            dy = a.speed * math.cos(a.climb_angle) * math.sin(a.heading) * self.cfg.dt
            dz = a.speed * math.sin(a.climb_angle) * self.cfg.dt
            a.position = (a.position[0] + dx, a.position[1] + dy, max(0.0, a.position[2] + dz))
            a.energy = max(0.0, min(1.0, a.energy - 0.012 * (abs(act.turn) + abs(act.climb)) - 0.002 * abs(act.acceleration) + 0.003))
            a.resource = max(0.0, a.resource - self.cfg.resource_drain * (1.5 if act.mode in (Mode.SUPPORT, Mode.CRITICAL) else 1.0))
            if a.resource < self.cfg.low_resource:
                a.capability = max(0.2, a.resource / max(self.cfg.low_resource, 1e-6))
            if a.c2_connected and self.rng.random() < self.cfg.communication_drop * 0.02:
                a.c2_connected = False
            if not a.c2_connected and self.rng.random() < 0.03:
                a.c2_connected = True
            a.last_message_step = self.step_count if a.c2_connected else a.last_message_step
            a.warning_pending = self.risk_of(i) > self.cfg.support_risk_threshold
            a.warning_age = 0 if a.c2_connected else a.warning_age + (1 if a.warning_pending else 0)
            if (a.warning_pending or a.support_requested) and a.pending_support_since is None:
                a.pending_support_since = self.step_count
            if not a.warning_pending and not a.support_requested:
                a.pending_support_since = None
            a.intervention_window.append(0)
            if len(a.intervention_window) > self.cfg.replan_intervention_window:
                a.intervention_window.pop(0)
        self.step_count += 1
        self.communication_log.append({"step": self.step_count, "connected": [a.c2_connected for a in self.aircraft], "intent_shared": [a.intent_shared for a in self.aircraft]})
        truth = self.truth_state()
        reward = self._reward(truth, actions)
        done = self.step_count >= self.cfg.horizon or not any(a.alive for a in self.aircraft)
        return StepResult(self.observe(), reward, done, {"truth": truth, "actions": actions})

    def _reward(self, truth: dict, actions: Dict[int, Action]) -> float:
        reward = 0.2 * sum(a.capability for a in self.aircraft if a.alive)
        cover_count = sum(a.role == Mode.COVER for a in self.aircraft if a.alive)
        reward += 0.4 if cover_count >= self.coverage_required else -0.2
        reward -= 0.08 * max(0, cover_count - self.coverage_required)
        # The paper defines ``C_hold`` as the capability kept in HOLD/SUPPORT
        # reserve.  RECOVER/EXIT are fault responses, not available reserve,
        # so they must not inflate this term.
        reserve = sum(
            a.capability
            for a in self.aircraft
            if a.alive and a.role in (Mode.HOLD, Mode.SUPPORT)
        )
        reward -= 0.15 * max(0.0, self.cfg.reserve_capacity - reserve) / max(self.cfg.reserve_capacity, 1e-6)
        reward -= sum(max(0.0, self.cfg.min_separation - d) / self.cfg.min_separation for d in truth["separations"])
        reward -= sum(0.15 for a in self.aircraft if a.energy < self.cfg.min_energy)
        reward -= sum(0.1 for act in actions.values() if act.mode == Mode.CRITICAL)
        return reward

    def truth_state(self) -> dict:
        separations = [_dist(self.aircraft[i].position, self.aircraft[j].position) for i in range(self.n) for j in range(i + 1, self.n)]
        return {"step": self.step_count, "aircraft": [a.copy() for a in self.aircraft], "opponent": self.opponent, "separations": separations, "coverage": sum(a.capability for a in self.aircraft if a.alive and a.role == Mode.COVER), "mission_active": self.mission_active, "task_progress": sum(a.task_progress for a in self.aircraft) / max(1, self.n), "risk_budget": sum(0.5 * a.energy + 0.5 * a.resource for a in self.aircraft) / max(1, self.n)}
