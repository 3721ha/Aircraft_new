from dataclasses import dataclass
import math
from typing import Dict

from .models import Action, Mode
from .rules import RuleMonitor, RuleReport


@dataclass
class Intervention:
    agent: int
    original: Action
    safe: Action
    reasons: list[str]
    distance: float


class SafetyShield:
    """Minimal residual shield for high-level discrete/continuous actions."""

    _modes = (Mode.HOLD, Mode.SUPPORT, Mode.COVER, Mode.RECOVER, Mode.EXIT, Mode.CRITICAL)

    def __init__(self, monitor: RuleMonitor):
        self.monitor = monitor

    @staticmethod
    def _distance(a: Action, b: Action) -> float:
        mode_cost = 0.0 if a.mode == b.mode else 1.0
        return mode_cost + abs(a.turn - b.turn) + abs(a.climb - b.climb) + abs(a.acceleration - b.acceleration)

    def filter(self, observations: Dict[int, dict], nominal: Dict[int, Action]) -> tuple[Dict[int, Action], list[Intervention]]:
        safe = {}
        interventions = []
        for i, action in nominal.items():
            candidates = [action.clipped()]
            for mode in self._modes:
                candidates.append(Action(mode, action.turn, action.climb, action.acceleration, action.target, action.target_kind).clipped())
            valid = [(self._distance(action, c), c) for c in candidates if self.monitor.evaluate_observation(observations[i], c).hard_safe]
            if not valid:
                selected = Action(Mode.RECOVER, 0.0, 0.0, -0.3)
                reasons = ["no_feasible_hard_safe_action"]
            else:
                _, selected = min(valid, key=lambda x: x[0])
                reasons = self.monitor.evaluate_observation(observations[i], action).violations
            safe[i] = selected
            distance = self._distance(action, selected)
            if distance > 1e-9:
                interventions.append(Intervention(i, action, selected, reasons, distance))
        return safe, interventions


class JointSafetyShield(SafetyShield):
    """Team-level residual arbitration for separation and coverage constraints.

    Local hard constraints are handled first. The joint pass then uses only team
    observations (never truth state) to choose the smallest team correction.
    """

    def filter(self, observations: Dict[int, dict], nominal: Dict[int, Action]) -> tuple[Dict[int, Action], list[Intervention]]:
        safe, interventions = super().filter(observations, nominal)
        # Resolve observed close pairs by assigning opposite turns. This is a
        # decision-level residual, not a substitute for a low-level collision controller.
        for i, observation in observations.items():
            for teammate in observation.get("teammates", []):
                j = teammate["uid"]
                if j not in safe or i >= j:
                    continue
                x, y, z = teammate["relative_position"]
                distance = math.sqrt(x * x + y * y + z * z)
                if distance < self.monitor.cfg.min_separation * 1.35:
                    left, right = (i, j) if y >= 0 else (j, i)
                    for agent, turn in ((left, 0.8), (right, -0.8)):
                        original = safe[agent]
                        corrected = Action(
                            Mode.RECOVER, turn=turn, climb=0.25,
                            acceleration=-0.25, target=original.target,
                            target_kind=original.target_kind,
                        )
                        if self._distance(original, corrected) > 1e-9:
                            interventions.append(Intervention(agent, nominal[agent], corrected, ["S02_joint_separation"], self._distance(nominal[agent], corrected)))
                        safe[agent] = corrected
        # Preserve coverage if no aircraft remains assigned to it and a capable
        # connected member is available. Coverage is soft relative to safety.
        if observations and not any(action.mode == Mode.COVER for action in safe.values()):
            candidates = [i for i, obs in observations.items() if obs["self"]["resource"] >= self.monitor.cfg.low_resource and obs["self"]["c2_connected"]]
            if candidates:
                agent = max(candidates, key=lambda i: observations[i]["self"]["resource"])
                original = safe[agent]
                corrected = Action(
                    Mode.COVER, original.turn, original.climb,
                    original.acceleration, original.target,
                    original.target_kind,
                )
                if self._distance(original, corrected) > 1e-9:
                    interventions.append(Intervention(agent, nominal[agent], corrected, ["C02_joint_coverage"], self._distance(nominal[agent], corrected)))
                safe[agent] = corrected
        return safe, interventions
