"""Framework-neutral interfaces for MAPPO/PPO style policies."""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .models import Action, Mode, TargetKind


class ObservationEncoder:
    """Encode each variable-size local observation into a fixed-size float vector."""

    def __init__(self, n_aircraft: int, max_speed: float = 320.0, max_altitude: float = 12000.0):
        self.n = n_aircraft
        self.max_speed = max_speed
        self.max_altitude = max_altitude
        # A bounded scale keeps mode age numerically comparable to the other
        # normalized state features across simulator configurations.
        self._dwell_norm = 10.0
        self.mode_order = list(Mode)
        self.target_kind_order = [
            TargetKind.TEAMMATE,
            TargetKind.INTRUDER,
            TargetKind.PROTECTED_ASSET,
            TargetKind.RESOURCE,
        ]
        # Include temporal role state so a policy can learn the E02 minimum
        # dwell-time constraint from its local observation.
        # The final sixteen coordinates describe the active task from the
        # aircraft's local frame.  They are present (as zeros) in legacy
        # scenarios so every baseline uses one stable observation schema.
        self.dimension = 19 + max(0, n_aircraft - 1) * 7 + 16

    def encode(self, observation: dict, agent_id: int = 0) -> List[float]:
        own = observation["self"]
        px, py, pz = own["position"]
        result = [
            max(-1.0, min(1.0, px / 12000.0)), max(-1.0, min(1.0, py / 12000.0)),
            max(0.0, min(1.0, pz / self.max_altitude)), max(0.0, min(1.0, own["speed"] / self.max_speed)),
            own["energy"], own["resource"], math.sin(own["heading"]), math.cos(own["heading"]),
            float(own["c2_connected"]), own["id_confidence"],
            float(own.get("authorization_valid", True)),
            max(0.0, min(1.0, own.get("data_age", 0) / 20.0)),
            float(own.get("intent_shared", True)),
            max(0.0, min(1.0, own.get("task_progress", 0.0))),
            float(own.get("handover_confirmed", True)),
            self.mode_order.index(Mode(own.get("role", Mode.HOLD.value))) / max(1, len(self.mode_order) - 1),
            agent_id / max(1, self.n - 1),
            max(0.0, min(1.0, float(own.get("mode_age", 0)) /  self._dwell_norm)),
            float(own.get("dwell_violation", False)),
        ]
        by_uid = {item["uid"]: item for item in observation.get("teammates", [])}
        for uid in range(self.n):
            if uid == agent_id:
                continue
            item = by_uid.get(uid)
            if item is None:
                result.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            x, y, z = item["relative_position"]
            role_index = self.mode_order.index(Mode(item["role"]))
            result.extend([max(-1.0, min(1.0, x / 12000.0)), max(-1.0, min(1.0, y / 12000.0)), max(-1.0, min(1.0, z / self.max_altitude)), item["risk"], role_index / max(1, len(self.mode_order) - 1), 1.0, 0.0])
        mission = observation.get("mission", {})
        intruder_relative = tuple(mission.get("intruder_relative_position", (0.0, 0.0, 0.0)))
        intruder_velocity = tuple(mission.get("intruder_velocity", (0.0, 0.0, 0.0)))
        protected_point = tuple(mission.get("protected_point", own["position"]))
        protected_relative = tuple(
            protected_point[index] - own["position"][index] for index in range(3)
        )
        result.extend([
            float(mission.get("active", False)),
            max(0.0, min(1.0, mission.get("step", 0) / 1000.0)),
            math.sin(mission.get("opponent_bearing", 0.0)),
            max(-1.0, min(1.0, intruder_relative[0] / 12000.0)),
            max(-1.0, min(1.0, intruder_relative[1] / 12000.0)),
            max(-1.0, min(1.0, intruder_relative[2] / self.max_altitude)),
            max(-1.0, min(1.0, intruder_velocity[0] / self.max_speed)),
            max(-1.0, min(1.0, intruder_velocity[1] / self.max_speed)),
            max(-1.0, min(1.0, intruder_velocity[2] / self.max_speed)),
            max(-1.0, min(1.0, protected_relative[0] / 12000.0)),
            max(-1.0, min(1.0, protected_relative[1] / 12000.0)),
            max(-1.0, min(1.0, protected_relative[2] / self.max_altitude)),
            max(0.0, min(1.0, float(mission.get("task_progress", 0.0)))),
            float(mission.get("task_success", False)),
            float(mission.get("task_failure", False)),
            max(0.0, min(1.0, float(mission.get("lock_time", 0)) / 10.0)),
        ])
        return result


class ActionCodec:
    """Maps a compact policy vector to the simulator's semantic Action."""

    def __init__(self, n_aircraft: int):
        self.n_aircraft = n_aircraft
        self.mode_order = list(Mode)
        self.target_kind_order = [
            TargetKind.TEAMMATE,
            TargetKind.INTRUDER,
            TargetKind.PROTECTED_ASSET,
            TargetKind.RESOURCE,
        ]
        self.dimension = 6

    def decode(self, vector: Sequence[float]) -> Action:
        if len(vector) < 5:
            raise ValueError("action vector requires at least five values")
        mode_index = max(0, min(len(self.mode_order) - 1, int(round(vector[0]))))
        target = int(round(vector[4])) if self.n_aircraft > 1 else None
        target = None if target is None or target < 0 or target >= self.n_aircraft else target
        if len(vector) >= self.dimension:
            kind_index = max(0, min(len(self.target_kind_order) - 1, int(round(vector[5]))))
            target_kind: TargetKind | str = self.target_kind_order[kind_index]
        else:
            # Backwards-compatible legacy vectors have no target-type head.
            target_kind = TargetKind.NONE
        if target_kind not in (TargetKind.NONE, TargetKind.TEAMMATE):
            target = None
        return Action(
            self.mode_order[mode_index],
            float(vector[1]),
            float(vector[2]),
            float(vector[3]),
            target,
            target_kind,
        ).clipped()

    def encode(self, action: Action) -> List[float]:
        target_kind = action.target_kind
        if isinstance(target_kind, str):
            try:
                target_kind = TargetKind(target_kind)
            except ValueError:
                target_kind = TargetKind.NONE
        if target_kind in self.target_kind_order:
            kind_index = self.target_kind_order.index(target_kind)
        elif action.mode is Mode.CRITICAL:
            kind_index = self.target_kind_order.index(TargetKind.INTRUDER)
        elif action.mode is Mode.COVER:
            kind_index = self.target_kind_order.index(TargetKind.PROTECTED_ASSET)
        else:
            kind_index = self.target_kind_order.index(TargetKind.TEAMMATE)
        return [
            float(self.mode_order.index(action.mode)),
            action.turn,
            action.climb,
            action.acceleration,
            float(action.target if action.target is not None else -1),
            float(kind_index),
        ]


@dataclass
class Transition:
    observations: Dict[int, dict]
    encoded_observations: Dict[int, List[float]]
    nominal_actions: Dict[int, Action]
    safe_actions: Dict[int, Action]
    reward: float
    done: bool
    info: dict


class RolloutCollector:
    """Collect transitions while keeping nominal and shielded actions for intervention learning."""

    def __init__(self, env, shield, encoder: ObservationEncoder):
        self.env = env
        self.shield = shield
        self.encoder = encoder

    def collect(self, policy, scenario: dict | None = None, horizon: int | None = None) -> List[Transition]:
        observations = self.env.reset(scenario)
        transitions = []
        for _ in range(horizon or self.env.cfg.horizon):
            nominal = {i: policy(observations[i], i) for i in observations}
            safe, interventions = self.shield.filter(observations, nominal)
            result = self.env.step(safe)
            shield_info = {}
            if hasattr(self.shield, "last_solution"):
                shield_info = {"belief_report": self.shield.last_solution.belief_report,
                               "qp_solution": self.shield.last_solution}
            transitions.append(Transition(observations, {i: self.encoder.encode(obs, i) for i, obs in observations.items()}, nominal, safe, result.reward, result.done, {**result.info, "interventions": interventions, **shield_info}))
            observations = result.observations
            if result.done:
                break
        return transitions
