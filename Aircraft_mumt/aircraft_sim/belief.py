"""Online belief-state STL monitoring.

The monitor deliberately accepts only local observations.  It maintains a
small Gaussian belief (diagonal covariance) and reports calibrated-style
violation probabilities plus a conservative robustness lower bound.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

from .config import SimConfig
from .models import Action, Mode
from .rules import is_hard_rule


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


@dataclass
class BeliefAgent:
    position: List[float]
    position_var: List[float]
    speed: float
    speed_var: float
    energy: float
    resource: float
    capability: float
    heading: float
    id_confidence: float
    c2_connected: bool
    age: int = 0
    mode_age: int = 0
    dwell_violation: bool = False
    sensor_degraded: bool = False
    c2_age: int = 0
    authorization_valid: bool = True
    intent_shared: bool = True
    data_age: int = 0
    task_progress: float = 0.0
    handover_confirmed: bool = True
    source_disagreement: float = 0.0
    warning_pending: bool = False
    warning_age: int = 0
    support_requested: bool = False
    assessment_due: bool = False
    replan_requested: bool = False
    task_authority: bool = True
    hazard_active: bool = False
    safe_dwell: int = 0
    hazard_previous: bool = False


@dataclass
class BeliefState:
    agents: Dict[int, BeliefAgent] = field(default_factory=dict)
    step: int = 0
    history: List[dict] = field(default_factory=list)
    latest_observations: Dict[int, dict] = field(default_factory=dict)


@dataclass
class BeliefRuleReport:
    violation_probability: Dict[str, float] = field(default_factory=dict)
    robust_lower: Dict[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    hard_safe: bool = True
    belief: Optional[BeliefState] = None


class BeliefSTLMonitor:
    """Recursive filter and probabilistic monitor for P0/P1/P2 rules."""

    def __init__(self, config: SimConfig, kappa: float = 2.0, process_noise: float = 35.0, model_error: float = 10.0):
        self.cfg = config
        self.kappa = kappa
        self.process_noise = process_noise
        self.model_error = model_error
        self.state = BeliefState()

    def reset(self) -> None:
        self.state = BeliefState()

    def _update_agent(self, uid: int, observation: dict) -> BeliefAgent:
        own = observation["self"]
        position = list(own["position"])
        speed = float(own["speed"])
        if uid not in self.state.agents:
            belief = BeliefAgent(position, [self.cfg.noise_position ** 2] * 3, speed, self.cfg.noise_speed ** 2, own["energy"], own["resource"], own.get("capability", own["resource"]), own["heading"], own["id_confidence"], own["c2_connected"], mode_age=int(own.get("mode_age", 0)), dwell_violation=bool(own.get("dwell_violation", False)), sensor_degraded=bool(own.get("sensor_degraded", False)), c2_age=0 if own["c2_connected"] else 1, authorization_valid=bool(own.get("authorization_valid", True)), intent_shared=bool(own.get("intent_shared", True)), data_age=int(own.get("data_age", 0)), task_progress=float(own.get("task_progress", 0.0)), handover_confirmed=bool(own.get("handover_confirmed", True)), source_disagreement=float(own.get("source_disagreement", 0.0)), warning_pending=bool(own.get("warning_pending", False)), warning_age=int(own.get("warning_age", 0)), support_requested=bool(own.get("support_requested", False)), assessment_due=bool(own.get("assessment_due", False)), replan_requested=bool(own.get("replan_requested", False)), task_authority=bool(own.get("task_authority", True)), hazard_active=bool(own.get("hazard_active", False)), hazard_previous=bool(own.get("hazard_previous", False)), safe_dwell=int(own.get("safe_dwell", 0)))
        else:
            belief = self.state.agents[uid]
            # Constant-gain Gaussian update; the prior is propagated with a
            # bounded process variance to avoid overconfidence after packet loss.
            measurement_var = max(1.0, self.cfg.noise_position ** 2)
            for k in range(3):
                prior_var = belief.position_var[k] + self.process_noise ** 2
                gain = prior_var / (prior_var + measurement_var)
                belief.position[k] += gain * (position[k] - belief.position[k])
                belief.position_var[k] = max(1.0, (1.0 - gain) * prior_var)
            speed_var = belief.speed_var + self.cfg.noise_speed ** 2
            measurement_speed_var = max(1e-6, self.cfg.noise_speed ** 2)
            gain = speed_var / (speed_var + measurement_speed_var)
            belief.speed += gain * (speed - belief.speed)
            belief.speed_var = max(0.25, (1.0 - gain) * speed_var)
            belief.energy = 0.7 * belief.energy + 0.3 * own["energy"]
            belief.resource = 0.7 * belief.resource + 0.3 * own["resource"]
            belief.capability = 0.7 * belief.capability + 0.3 * own.get("capability", belief.resource)
            belief.heading = own["heading"]
            belief.id_confidence = 0.7 * belief.id_confidence + 0.3 * own["id_confidence"]
            belief.c2_connected = bool(own["c2_connected"])
            belief.c2_age = 0 if belief.c2_connected else belief.c2_age + 1
            belief.mode_age = int(own.get("mode_age", belief.mode_age))
            belief.dwell_violation = bool(own.get("dwell_violation", False))
            belief.sensor_degraded = bool(own.get("sensor_degraded", False))
            belief.authorization_valid = bool(own.get("authorization_valid", True))
            belief.intent_shared = bool(own.get("intent_shared", True))
            belief.data_age = int(own.get("data_age", 0))
            belief.task_progress = float(own.get("task_progress", 0.0))
            belief.handover_confirmed = bool(own.get("handover_confirmed", True))
            belief.source_disagreement = float(own.get("source_disagreement", 0.0))
            belief.warning_pending = bool(own.get("warning_pending", False))
            belief.warning_age = int(own.get("warning_age", 0))
            belief.support_requested = bool(own.get("support_requested", False))
            belief.assessment_due = bool(own.get("assessment_due", False))
            belief.replan_requested = bool(own.get("replan_requested", False))
            belief.task_authority = bool(own.get("task_authority", True))
            belief.hazard_active = bool(own.get("hazard_active", False))
            belief.safe_dwell = int(own.get("safe_dwell", 0))
            belief.hazard_previous = bool(own.get("hazard_previous", False))
            belief.age = 0
        self.state.agents[uid] = belief
        return belief

    def update(self, observations: Dict[int, dict]) -> BeliefState:
        for uid, observation in observations.items():
            self._update_agent(uid, observation)
        self.state.step += 1
        self.state.history.append(observations)
        self.state.latest_observations = observations
        if len(self.state.history) > 32:
            self.state.history.pop(0)
        for uid in self.state.agents:
            if uid not in observations:
                self.state.agents[uid].age += 1
                if not self.state.agents[uid].c2_connected:
                    self.state.agents[uid].c2_age += 1
        return self.state

    def _distance_stats(self, first: BeliefAgent, second: BeliefAgent):
        rel = [second.position[k] - first.position[k] for k in range(3)]
        distance = math.sqrt(sum(value * value for value in rel))
        if distance < 1e-6:
            return distance, math.sqrt(sum(first.position_var) + sum(second.position_var))
        variance = sum((rel[k] / distance) ** 2 * (first.position_var[k] + second.position_var[k]) for k in range(3))
        return distance, math.sqrt(max(1.0, variance))

    def evaluate(self, observations: Dict[int, dict], actions: Optional[Dict[int, Action]] = None) -> BeliefRuleReport:
        self.update(observations)
        return self.evaluate_current(actions)

    def evaluate_current(self, actions: Optional[Dict[int, Action]] = None) -> BeliefRuleReport:
        """Evaluate a candidate action against the current belief without updating it."""
        report = BeliefRuleReport(belief=self.state)
        actions = actions or {}
        for uid, belief in self.state.agents.items():
            speed_sigma = math.sqrt(belief.speed_var)
            speed_margin = min(belief.speed - self.cfg.min_speed, self.cfg.max_speed - belief.speed)
            speed_prob = _normal_cdf(-speed_margin / max(1e-6, speed_sigma))
            altitude_margin = min(belief.position[2] - self.cfg.min_altitude, self.cfg.max_altitude - belief.position[2])
            altitude_sigma = math.sqrt(belief.position_var[2])
            altitude_prob = _normal_cdf(-altitude_margin / max(1e-6, altitude_sigma))
            env_prob = max(speed_prob, altitude_prob)
            report.violation_probability[f"S01:{uid}"] = env_prob
            report.robust_lower[f"S01:{uid}"] = min(speed_margin, altitude_margin) - self.kappa * max(speed_sigma, altitude_sigma) - self.model_error
            if env_prob > 0.05:
                report.violations.append(f"S01:{uid}")
            recover_margin = min(speed_margin, altitude_margin) - self.kappa * max(speed_sigma, altitude_sigma)
            recovery_possible = recover_margin > -self.cfg.hysteresis_margin and belief.energy > self.cfg.min_energy * 0.8 and belief.resource > self.cfg.low_resource * 0.5
            report.violation_probability[f"S05:{uid}"] = 0.0 if recovery_possible else 1.0
            report.robust_lower[f"S05:{uid}"] = recover_margin if recovery_possible else -abs(recover_margin)
            if not recovery_possible:
                report.violations.append(f"S05:{uid}")
            if belief.hazard_previous and not belief.hazard_active and 0 < belief.safe_dwell < self.cfg.safety_dwell_steps:
                report.violation_probability[f"S06:{uid}"] = 1.0
                report.robust_lower[f"S06:{uid}"] = belief.safe_dwell - self.cfg.safety_dwell_steps
                report.violations.append(f"S06:{uid}")
            # F01/F02 are time-bounded degradation rules.  Age counts missed
            # local reports and therefore remains observable without truth.
            if not belief.c2_connected and belief.c2_age > self.cfg.c2_timeout and actions.get(uid, Action()).mode not in (Mode.RECOVER, Mode.EXIT):
                report.violation_probability[f"F01:{uid}"] = 1.0
                report.robust_lower[f"F01:{uid}"] = self.cfg.c2_timeout - belief.c2_age
                report.violations.append(f"F01:{uid}")
            if (not belief.c2_connected and belief.task_authority
                    and 0.5 * belief.energy + 0.5 * belief.resource >= self.cfg.reserve_capacity
                    and actions.get(uid, Action()).mode not in (Mode.SUPPORT, Mode.COVER, Mode.CRITICAL, Mode.RECOVER, Mode.EXIT)):
                report.violation_probability[f"M05:{uid}"] = 1.0
                report.robust_lower[f"M05:{uid}"] = -1.0
                report.violations.append(f"M05:{uid}")
            if belief.sensor_degraded and actions.get(uid, Action()).mode not in (Mode.RECOVER, Mode.EXIT, Mode.HOLD):
                report.violation_probability[f"F02:{uid}"] = 1.0
                report.robust_lower[f"F02:{uid}"] = -1.0
                report.violations.append(f"F02:{uid}")
            # Capability loss requires team reconfiguration or explicit
            # recovery; this is observable through the propagated capability
            # estimate and does not require truth-state access.
            if belief.capability < 0.5 and actions.get(uid, Action()).mode not in (Mode.RECOVER, Mode.EXIT):
                report.violation_probability[f"F03:{uid}"] = 1.0
                report.robust_lower[f"F03:{uid}"] = belief.capability - 0.5
                report.violations.append(f"F03:{uid}")
            energy_margin = belief.energy - self.cfg.min_energy
            energy_prob = 1.0 if energy_margin < 0 else 0.0
            report.violation_probability[f"S04:{uid}"] = energy_prob
            report.robust_lower[f"S04:{uid}"] = energy_margin
            if energy_prob > 0.05:
                report.violations.append(f"S04:{uid}")
            if actions.get(uid, Action()).mode == Mode.CRITICAL:
                auth_prob = 0.0 if belief.authorization_valid else 1.0
                report.violation_probability[f"I01:{uid}"] = auth_prob
                report.robust_lower[f"I01:{uid}"] = 1.0 if belief.authorization_valid else -1.0
                if auth_prob > 0.05:
                    report.violations.append(f"I01:{uid}")
                id_margin = belief.id_confidence - self.cfg.id_confidence_threshold
                report.violation_probability[f"I02:{uid}"] = _normal_cdf(-id_margin / 0.04)
                report.robust_lower[f"I02:{uid}"] = id_margin - self.kappa * 0.04
                if report.violation_probability[f"I02:{uid}"] > 0.05:
                    report.violations.append(f"I02:{uid}")
                freshness_margin = self.cfg.info_freshness_timeout - belief.data_age
                report.violation_probability[f"I03:{uid}"] = 1.0 if freshness_margin < 0 else 0.0
                report.robust_lower[f"I03:{uid}"] = freshness_margin
                if freshness_margin < 0:
                    report.violations.append(f"I03:{uid}")
            if actions.get(uid, Action()).mode in (Mode.SUPPORT, Mode.COVER, Mode.CRITICAL) and not belief.intent_shared:
                report.violation_probability[f"C05:{uid}"] = 1.0
                report.robust_lower[f"C05:{uid}"] = -1.0
                report.violations.append(f"C05:{uid}")
            if belief.source_disagreement > self.cfg.source_disagreement_threshold and actions.get(uid, Action()).mode in (Mode.SUPPORT, Mode.CRITICAL):
                report.violation_probability[f"I04:{uid}"] = 1.0
                report.robust_lower[f"I04:{uid}"] = self.cfg.source_disagreement_threshold - belief.source_disagreement
                report.violations.append(f"I04:{uid}")
            if belief.warning_pending and belief.warning_age > self.cfg.warning_deadline and not belief.c2_connected:
                report.violation_probability[f"I05:{uid}"] = 1.0
                report.robust_lower[f"I05:{uid}"] = self.cfg.warning_deadline - belief.warning_age
                report.violations.append(f"I05:{uid}")
            if belief.support_requested and not any(action.mode == Mode.SUPPORT and action.target == uid for action in actions.values()):
                report.violation_probability[f"C03:{uid}"] = 1.0
                report.robust_lower[f"C03:{uid}"] = -1.0
                report.violations.append(f"C03:{uid}")
            if belief.assessment_due:
                report.violation_probability[f"M03:{uid}"] = 1.0
                report.robust_lower[f"M03:{uid}"] = -1.0
                report.violations.append(f"M03:{uid}")
            if belief.replan_requested:
                report.violation_probability[f"M04:{uid}"] = 1.0
                report.robust_lower[f"M04:{uid}"] = -1.0
                report.violations.append(f"M04:{uid}")
            if actions.get(uid, Action()).mode not in (Mode.RECOVER, Mode.EXIT) and not belief.handover_confirmed:
                report.violation_probability[f"C04:{uid}"] = 1.0
                report.robust_lower[f"C04:{uid}"] = -1.0
                report.violations.append(f"C04:{uid}")
            risk_budget = 0.5 * belief.energy + 0.5 * belief.resource
            report.violation_probability[f"F06:{uid}"] = 1.0 if risk_budget < self.cfg.reserve_capacity else 0.0
            report.robust_lower[f"F06:{uid}"] = risk_budget - self.cfg.reserve_capacity
            if risk_budget < self.cfg.reserve_capacity and actions.get(uid, Action()).mode not in (Mode.RECOVER, Mode.EXIT):
                report.violations.append(f"F06:{uid}")
            if self.state.step >= self.cfg.task_deadline and belief.task_progress < 0.95:
                report.violation_probability[f"M02:{uid}"] = 1.0
                report.robust_lower[f"M02:{uid}"] = belief.task_progress - 0.95
                report.violations.append(f"M02:{uid}")
            if belief.resource < self.cfg.low_resource and actions.get(uid, Action()).mode not in (Mode.RECOVER, Mode.EXIT):
                report.violation_probability[f"F04:{uid}"] = 1.0
                report.robust_lower[f"F04:{uid}"] = belief.resource - self.cfg.low_resource
                report.violations.append(f"F04:{uid}")
            if belief.capability < self.cfg.capability_loss_threshold and actions.get(uid, Action()).mode not in (Mode.RECOVER, Mode.EXIT):
                # F03 is the hard fault-reconfiguration rule from the
                # outline.  Keep M06 as a diagnostic alias for compatibility,
                # but expose the hard family to belief safety gating so an
                # agent cannot continue a role after capability loss.
                report.violation_probability[f"F03:{uid}"] = 1.0
                report.robust_lower[f"F03:{uid}"] = belief.capability - self.cfg.capability_loss_threshold
                report.violations.append(f"F03:{uid}")
                report.violation_probability[f"M06:{uid}"] = 1.0
                report.robust_lower[f"M06:{uid}"] = belief.capability - self.cfg.capability_loss_threshold
                report.violations.append(f"M06:{uid}")
            previous_role = self.state.latest_observations.get(uid, {}).get("self", {}).get("role")
            if (previous_role is not None and actions.get(uid, Action()).mode.value != previous_role
                    and 0 < belief.mode_age < self.cfg.mode_dwell_steps
                    and actions.get(uid, Action()).mode not in (Mode.RECOVER, Mode.EXIT)):
                report.violation_probability[f"E02:{uid}"] = 1.0
                report.robust_lower[f"E02:{uid}"] = float(belief.mode_age - self.cfg.mode_dwell_steps)
                report.violations.append(f"E02:{uid}")
        ids = sorted(self.state.agents)
        observed_roles = []
        for uid, observation in self.state.latest_observations.items():
            observed_roles.append(observation.get("self", {}).get("role"))
            observed_roles.extend(item.get("role") for item in observation.get("teammates", []))
        has_cover = any(role == Mode.COVER.value for role in observed_roles) or any(action.mode == Mode.COVER for action in actions.values())
        c02_margin = 1.0 if has_cover else -1.0
        report.violation_probability["C02"] = 0.0 if has_cover else 1.0
        report.robust_lower["C02"] = c02_margin
        if not has_cover:
            report.violations.append("C02")
        for offset, uid in enumerate(ids):
            for vid in ids[offset + 1:]:
                distance, sigma = self._distance_stats(self.state.agents[uid], self.state.agents[vid])
                margin = distance - self.cfg.min_separation
                probability = _normal_cdf(-margin / max(1e-6, sigma))
                key = f"S02:{uid}-{vid}"
                report.violation_probability[key] = probability
                report.robust_lower[key] = margin - self.kappa * sigma - self.model_error
                if probability > 0.05:
                    report.violations.append(key)
                # Online S03: predict near-term time-to-conflict from the
                # belief means. A positive closing rate and TTC below the
                # horizon require an immediate safe trend.
                vi = (self.state.agents[uid].speed * math.cos(self.state.agents[uid].heading), self.state.agents[uid].speed * math.sin(self.state.agents[uid].heading), 0.0)
                vj = (self.state.agents[vid].speed * math.cos(self.state.agents[vid].heading), self.state.agents[vid].speed * math.sin(self.state.agents[vid].heading), 0.0)
                rel = [self.state.agents[vid].position[k] - self.state.agents[uid].position[k] for k in range(3)]
                relv = [vj[k] - vi[k] for k in range(3)]
                closing_rate = sum(rel[k] * relv[k] for k in range(3))
                speed_sq = sum(v * v for v in relv)
                ttc = -closing_rate / speed_sq if closing_rate < 0 and speed_sq > 1e-9 else math.inf
                closest_distance = math.sqrt(sum((rel[k] + relv[k] * ttc) ** 2 for k in range(3))) if math.isfinite(ttc) else math.inf
                if ttc < self.cfg.belief_ttc_threshold and closest_distance < self.cfg.min_separation:
                    skey = f"S03:{uid}-{vid}"
                    report.violation_probability[skey] = max(report.violation_probability.get(skey, 0.0), 1.0)
                    report.robust_lower[skey] = self.cfg.belief_ttc_threshold - ttc
                    report.violations.append(skey)
                disagreement = abs(self.state.agents[uid].source_disagreement - self.state.agents[vid].source_disagreement)
                if disagreement > self.cfg.belief_disagreement_threshold and actions.get(uid, Action()).mode in (Mode.SUPPORT, Mode.COVER) and actions.get(vid, Action()).mode in (Mode.SUPPORT, Mode.COVER):
                    ikey = f"I06:{uid}-{vid}"
                    report.violation_probability[ikey] = 1.0
                    report.robust_lower[ikey] = self.cfg.belief_disagreement_threshold - disagreement
                    report.violations.append(ikey)
        # C06: reserve enough uncommitted capability for a new task.  COVER
        # and SUPPORT consume commitment; HOLD/RECOVER remain available.
        reserve = sum(max(0.0, a.resource) for uid, a in self.state.agents.items()
                       if actions.get(uid, Action()).mode in (Mode.HOLD, Mode.RECOVER, Mode.EXIT))
        c06_margin = reserve - self.cfg.reserve_capacity
        report.violation_probability["C06"] = 1.0 if c06_margin < 0 else 0.0
        report.robust_lower["C06"] = c06_margin
        if c06_margin < 0:
            report.violations.append("C06")
        active_count = sum(action.mode in (Mode.SUPPORT, Mode.COVER, Mode.CRITICAL) for action in actions.values())
        if active_count > self.cfg.min_coverage + int(self.cfg.overallocated_capacity_tolerance):
            report.violation_probability["E01"] = 1.0
            report.robust_lower["E01"] = float(self.cfg.min_coverage + self.cfg.overallocated_capacity_tolerance - active_count)
            report.violations.append("E01")
        capabilities = [agent.capability for agent in self.state.agents.values()]
        if capabilities and max(capabilities) - min(capabilities) > self.cfg.load_balance_tolerance:
            report.violation_probability["E06"] = 1.0
            report.robust_lower["E06"] = self.cfg.load_balance_tolerance - (max(capabilities) - min(capabilities))
            report.violations.append("E06")
        for uid, belief in self.state.agents.items():
            action = actions.get(uid, Action())
            if belief.hazard_previous and not belief.hazard_active and belief.safe_dwell >= self.cfg.safety_dwell_steps and action.mode in (Mode.HOLD, Mode.RECOVER, Mode.EXIT):
                report.violation_probability[f"E05:{uid}"] = 1.0
                report.robust_lower[f"E05:{uid}"] = float(belief.safe_dwell - self.cfg.safety_dwell_steps)
                report.violations.append(f"E05:{uid}")
        authorities = sum(int(agent.task_authority) for agent in self.state.agents.values())
        if authorities != 1:
            report.violation_probability["C01"] = 1.0
            report.robust_lower["C01"] = float(1 - authorities)
            report.violations.append("C01")
        report.hard_safe = not any(is_hard_rule(name) for name in report.violations)
        return report
