from dataclasses import dataclass, field
import math
from typing import Dict, List

from .config import SimConfig
from .models import Action, AircraftState, Mode


HARD_RULES = frozenset({
    "S01", "S02", "S03", "S04", "S05", "S06",
    "I01", "I02", "I03", "I04", "I05", "I06",
    "F01", "F02", "F03", "F04", "F06",
})


def rule_family(rule: str) -> str:
    """Return the rule identifier without an agent/pair instance suffix."""

    return rule.split(":", 1)[0].split("/", 1)[0]


def is_hard_rule(rule: str) -> bool:
    return rule_family(rule) in HARD_RULES


@dataclass
class RuleReport:
    violations: List[str] = field(default_factory=list)
    margins: Dict[str, float] = field(default_factory=dict)
    hard_safe: bool = True

    @property
    def satisfied(self) -> bool:
        return not self.violations


class RuleMonitor:
    """Shared rule semantics; truth mode evaluates state, belief mode evaluates observations."""

    def __init__(self, config: SimConfig):
        self.cfg = config
        self._safe_trend_deadlines: Dict[str, int] = {}
        self._last_truth_step = -1

    def evaluate_truth(self, truth: dict, actions: Dict[int, Action] | None = None) -> RuleReport:
        report = RuleReport()
        # A monitor may be reused after an environment reset.  Temporal
        # obligations from the preceding episode must never leak into it.
        if truth.get("step", 0) < self._last_truth_step:
            self._safe_trend_deadlines.clear()
        self._last_truth_step = truth.get("step", 0)
        aircraft: List[AircraftState] = truth["aircraft"]
        for i, a in enumerate(aircraft):
            env_margin = min(
                a.speed - self.cfg.min_speed,
                self.cfg.max_speed - a.speed,
                a.position[2] - self.cfg.min_altitude,
                self.cfg.max_altitude - a.position[2],
                self.cfg.max_abs_roll_deg - abs(math.degrees(a.roll)),
                self.cfg.max_abs_pitch_deg - abs(math.degrees(a.pitch)),
                self.cfg.max_abs_aoa_deg - abs(math.degrees(a.angle_of_attack)),
                a.load_factor - self.cfg.min_load_factor,
                self.cfg.max_load_factor - a.load_factor,
            )
            report.margins[f"S01/{i}"] = env_margin
            if env_margin < 0:
                report.violations.append(f"S01:{i}")
            report.margins[f"S04/{i}"] = a.energy - self.cfg.min_energy
            if a.energy < self.cfg.min_energy:
                report.violations.append(f"S04:{i}")
            if a.resource < self.cfg.low_resource and a.role not in (Mode.RECOVER, Mode.EXIT):
                report.violations.append(f"F04:{i}")
            if a.capability < self.cfg.capability_loss_threshold and a.role not in (Mode.RECOVER, Mode.EXIT):
                report.violations.append(f"F03:{i}")
                report.violations.append(f"M06:{i}")
            if actions and actions.get(i, Action()).mode == Mode.CRITICAL and a.id_confidence < self.cfg.id_confidence_threshold:
                report.violations.append(f"I02:{i}")
            if actions and actions.get(i, Action()).mode == Mode.CRITICAL and not a.authorization_valid:
                report.violations.append(f"I01:{i}")
            if a.source_disagreement > self.cfg.source_disagreement_threshold and actions and actions.get(i, Action()).mode in (Mode.SUPPORT, Mode.CRITICAL):
                report.violations.append(f"I04:{i}")
            if a.data_age > self.cfg.info_freshness_timeout and actions and actions.get(i, Action()).mode == Mode.CRITICAL:
                report.violations.append(f"I03:{i}")
            if not a.c2_connected and self.cfg.c2_timeout < truth["step"] - a.last_message_step:
                if a.role not in (Mode.RECOVER, Mode.EXIT):
                    report.violations.append(f"F01:{i}")
            risk_budget_i = 0.5 * a.energy + 0.5 * a.resource
            if (not a.c2_connected and a.task_authority and risk_budget_i >= self.cfg.reserve_capacity
                    and a.role not in (Mode.SUPPORT, Mode.COVER, Mode.CRITICAL, Mode.RECOVER, Mode.EXIT)):
                report.violations.append(f"M05:{i}")
            if actions and not a.handover_confirmed and actions.get(i, Action()).mode not in (Mode.RECOVER, Mode.EXIT):
                report.violations.append(f"C04:{i}")
            if actions and actions.get(i, Action()).mode in (Mode.SUPPORT, Mode.COVER, Mode.CRITICAL) and not a.intent_shared:
                report.violations.append(f"C05:{i}")
            if a.warning_pending and a.warning_age > self.cfg.warning_deadline and not a.c2_connected:
                report.violations.append(f"I05:{i}")
            if a.support_requested and actions and not any(other.mode == Mode.SUPPORT and other.target == i for other in actions.values()):
                pending_since = a.pending_support_since if a.pending_support_since is not None else truth["step"]
                if truth["step"] - pending_since > self.cfg.support_response_deadline:
                    report.violations.append(f"C03:{i}")
            if a.assessment_due and a.last_task_action_step is not None and truth["step"] - a.last_task_action_step > self.cfg.assessment_deadline:
                report.violations.append(f"M03:{i}")
            if a.replan_requested:
                report.violations.append(f"M04:{i}")
            if a.task_progress < 0.0:
                report.violations.append(f"M01:{i}")
        for idx, d in enumerate(truth["separations"]):
            report.margins[f"S02/{idx}"] = d - self.cfg.min_separation
            if d < self.cfg.min_separation:
                report.violations.append(f"S02:{idx}")
        for i, a in enumerate(aircraft):
            for j in range(i + 1, len(aircraft)):
                b = aircraft[j]
                if abs(a.source_disagreement - b.source_disagreement) > self.cfg.belief_disagreement_threshold and actions and actions.get(i, Action()).mode in (Mode.SUPPORT, Mode.COVER) and actions.get(j, Action()).mode in (Mode.SUPPORT, Mode.COVER):
                    report.violations.append(f"I06:{i}-{j}")
        # S03: estimate time-to-conflict from the current relative velocity.
        for i, a in enumerate(aircraft):
            va = (a.speed * math.cos(a.heading), a.speed * math.sin(a.heading), 0.0)
            for j in range(i + 1, len(aircraft)):
                b = aircraft[j]
                vb = (b.speed * math.cos(b.heading), b.speed * math.sin(b.heading), 0.0)
                rel_p = tuple(b.position[k] - a.position[k] for k in range(3))
                rel_v = tuple(vb[k] - va[k] for k in range(3))
                closing = sum(rel_p[k] * rel_v[k] for k in range(3)) < 0
                speed_sq = sum(v * v for v in rel_v)
                ttc = (-sum(rel_p[k] * rel_v[k] for k in range(3)) / speed_sq) if closing and speed_sq > 1e-9 else math.inf
                closest_distance = _distance(tuple(rel_p[k] + rel_v[k] * ttc for k in range(3))) if math.isfinite(ttc) else math.inf
                # Keep the offline oracle and online belief monitor on the
                # same scenario-defined prediction horizon.  A fixed literal
                # here would make a configured anticipatory conflict visible
                # to the shield but invisible to the truth evaluation.
                key = f"S03:{i}-{j}"
                unsafe_trend = ttc < self.cfg.belief_ttc_threshold and closest_distance < self.cfg.min_separation
                if unsafe_trend:
                    # S03 is a bounded eventuality: a predicted conflict
                    # creates an obligation to establish a safe trend within
                    # Delta, rather than an instantaneous violation.
                    deadline = self._safe_trend_deadlines.setdefault(
                        key, truth.get("step", 0) + max(1, self.cfg.safe_trend_window)
                    )
                    report.margins[f"S03/{i}-{j}"] = deadline - truth.get("step", 0)
                    if truth.get("step", 0) >= deadline:
                        report.violations.append(f"S03:{i}-{j}")
                else:
                    self._safe_trend_deadlines.pop(key, None)
        report.margins["C02"] = truth["coverage"] - self.cfg.min_coverage
        if truth["mission_active"] and truth["coverage"] < self.cfg.min_coverage:
            report.violations.append("C02")
        progress = truth.get("task_progress", 0.0)
        if truth.get("mission_active", False) and truth.get("step", 0) >= self.cfg.task_deadline and progress < 0.95:
            report.margins["M02"] = progress - 0.95
            report.violations.append("M02")
        if truth.get("mission_active", False) and truth.get("step", 0) >= self.cfg.replan_intervention_window and progress < self.cfg.task_progress_rate * self.cfg.replan_intervention_window * 0.5:
            report.margins["M04"] = progress
            report.violations.append("M04")
        authorities = sum(int(a.task_authority) for a in aircraft if a.alive)
        if truth.get("mission_active", False) and authorities != 1:
            report.margins["C01"] = float(1 - authorities)
            report.violations.append("C01")
        risk_budget = truth.get("risk_budget", 1.0)
        if risk_budget < self.cfg.reserve_capacity:
            report.margins["F06"] = risk_budget - self.cfg.reserve_capacity
            report.violations.append("F06")
        # Efficiency preferences are reported as soft violations.  They are
        # intentionally excluded from hard_safe and only diagnose excess
        # allocation, role churn, communication load, delayed resumption, and
        # team load imbalance.
        active_modes = [a for a in aircraft if a.alive and a.role in (Mode.SUPPORT, Mode.COVER, Mode.CRITICAL)]
        if len(active_modes) > self.cfg.min_coverage + int(self.cfg.overallocated_capacity_tolerance):
            report.violations.append("E01")
        for i, a in enumerate(aircraft):
            if a.dwell_violation and a.role not in (Mode.RECOVER, Mode.EXIT):
                report.violations.append(f"E02:{i}")
            if a.hazard_previous and not a.hazard_active and a.safe_dwell >= self.cfg.safety_dwell_steps and a.role in (Mode.HOLD, Mode.RECOVER, Mode.EXIT):
                report.violations.append(f"E05:{i}")
        capabilities = [a.capability for a in aircraft if a.alive]
        if capabilities and max(capabilities) - min(capabilities) > self.cfg.load_balance_tolerance:
            report.violations.append("E06")
        report.hard_safe = not any(is_hard_rule(v) for v in report.violations)
        return report
    def evaluate_observation(self, observation: dict, action: Action) -> RuleReport:
        report = RuleReport()
        own = observation["self"]
        if action.mode == Mode.CRITICAL and own["id_confidence"] < self.cfg.id_confidence_threshold:
            report.violations.append("I02")
        if own["resource"] < self.cfg.low_resource and action.mode not in (Mode.RECOVER, Mode.EXIT):
            report.violations.append("F04")
        for teammate in observation.get("teammates", []):
            x, y, z = teammate["relative_position"]
            d = (x * x + y * y + z * z) ** 0.5
            if d < self.cfg.min_separation:
                report.violations.append(f"S02:{teammate['uid']}")
        report.hard_safe = not any(is_hard_rule(v) for v in report.violations)
        return report


def _distance(vector) -> float:
    return math.sqrt(sum(value * value for value in vector))
