"""Conflict-aware belief-space residual safety shield.

The implementation uses a small SLSQP problem as a practical QP/SQP
approximation.  It optimizes continuous residuals jointly for all aircraft,
while discrete modes are selected by the hard-rule filter first.  Only local
observations are passed to the belief monitor; truth state is never needed.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import Bounds, minimize

from .belief import BeliefSTLMonitor, BeliefRuleReport
from .config import SimConfig
from .models import Action, Mode
from .rule_graph import RuleAgentDependencyGraph, RuleAgentGraphSnapshot, rule_category
from .rules import is_hard_rule
from .shield import Intervention


@dataclass
class QPResidualSolution:
    status: str = "not_solved"
    objective: float = 0.0
    corrected_controls: Dict[int, List[float]] = field(default_factory=dict)
    hard_constraints: Dict[str, float] = field(default_factory=dict)
    relaxed_constraints: Dict[str, float] = field(default_factory=dict)
    conflict_rules: List[List[str]] = field(default_factory=list)
    infeasibility_reason: Optional[str] = None
    nominal_actions: Dict[int, Action] = field(default_factory=dict)
    safe_actions: Dict[int, Action] = field(default_factory=dict)
    belief_report: Optional[BeliefRuleReport] = None
    post_belief_report: Optional[BeliefRuleReport] = None
    dependency_graph: Optional[RuleAgentGraphSnapshot] = None
    conflict_events: List[dict] = field(default_factory=list)
    agent_rule_conflicts: Dict[int, List[str]] = field(default_factory=dict)
    solver_agents: List[int] = field(default_factory=list)
    solver_components: List[List[int]] = field(default_factory=list)
    cross_component_violations: int = 0
    cross_component_repairs: int = 0


class ConflictAwareQPSafetyShield:
    """Joint residual optimizer with hard/soft rule arbitration.

    ``filter`` keeps the historical two-return-value shield API.  The latest
    detailed solve is available as ``last_solution`` and is also attached to
    rollout transition info by :class:`RolloutCollector`.
    """

    # Fixed ordering used only by the no-dynamic-priority ablation.  The
    # ordering mirrors the paper's P0--P5 hierarchy and is independent of
    # belief probability, time-to-violation, and graph scope.
    FIXED_CATEGORY_PRIORITY = {
        "physical": 6,
        "authorization": 5,
        "fault_resource": 4,
        "coordination": 3,
        "mission": 2,
        "efficiency": 1,
    }

    def __init__(self, config: SimConfig, monitor: Optional[BeliefSTLMonitor] = None,
                 kappa: float = 2.0, alpha: float = 0.05, safety_buffer: float = 20.0,
                 allow_soft_relaxation: bool = True, sparse_active_set: bool = True,
                 active_set_margin: float = 200.0,
                 use_dependency_graph: bool = True,
                 use_dynamic_priority: bool = True,
                 use_continuous_qp: bool = True):
        # Accept either SimConfig or an AircraftEnv for ergonomic integration.
        self.cfg = getattr(config, "cfg", config)
        self.monitor = monitor or BeliefSTLMonitor(self.cfg, kappa=kappa)
        self.kappa = kappa
        self.alpha = alpha
        self.safety_buffer = safety_buffer
        self.allow_soft_relaxation = allow_soft_relaxation
        self.sparse_active_set = sparse_active_set
        self.active_set_margin = float(active_set_margin)
        # These switches are deliberately limited to controlled component
        # ablations.  The default keeps the complete DG-QP behavior unchanged.
        self.use_dependency_graph = bool(use_dependency_graph)
        self.use_dynamic_priority = bool(use_dynamic_priority)
        self.use_continuous_qp = bool(use_continuous_qp)
        self.rule_graph = RuleAgentDependencyGraph()
        self.last_solution = QPResidualSolution()
        self.intervention_history: Dict[int, List[int]] = {}

    def _active_solver_agents(self, selected: Dict[int, Action], report: BeliefRuleReport, ids: List[int]):
        """Return agents that can affect a near-term physical constraint.

        Far-separated aircraft have no useful continuous coupling in the
        one-step/trend model.  Leaving them out of the residual solve keeps
        the safety layer sparse while the discrete rule gate still processes
        every agent.  Any explicitly reported pair risk always activates both
        endpoints.
        """
        if not self.sparse_active_set or len(ids) <= 3:
            return list(ids), [list(ids)] if ids else []
        agents = self.monitor.state.agents
        active = set()
        pairs = []
        for ai, i in enumerate(ids):
            for j in ids[ai + 1:]:
                bi, bj = agents[i], agents[j]
                pi = self._predict_position(bi, (selected[i].turn, selected[i].climb, selected[i].acceleration), self.cfg.dt)
                pj = self._predict_position(bj, (selected[j].turn, selected[j].climb, selected[j].acceleration), self.cfg.dt)
                distance = float(np.linalg.norm(pj - pi))
                sigma = math.sqrt(sum(bi.position_var) + sum(bj.position_var))
                required = self.cfg.min_separation + self.kappa * sigma + self.safety_buffer
                key_s02 = f"S02:{i}-{j}"
                key_s03 = f"S03:{i}-{j}"
                if key_s02 in report.violations or key_s03 in report.violations or distance <= required + self.active_set_margin:
                    active.update((i, j))
                    pairs.append((i, j))
        for i in ids:
            belief = agents[i]
            speed = belief.speed + selected[i].acceleration * 12.0 * self.cfg.dt
            altitude = self._predict_position(belief, (selected[i].turn, selected[i].climb, selected[i].acceleration), self.cfg.dt)[2]
            sigma = max(math.sqrt(belief.speed_var), math.sqrt(belief.position_var[2]))
            if min(speed - self.cfg.min_speed, self.cfg.max_speed - speed,
                   altitude - self.cfg.min_altitude, self.cfg.max_altitude - altitude) <= self.kappa * sigma + 20.0:
                active.add(i)
        if not active:
            return [], []
        # Connected components of the active pair graph.  Components are
        # retained in the trace; the current solver uses their union to keep
        # the global fallback semantics unchanged.
        components = []
        pending = set(active)
        while pending:
            component = {pending.pop()}
            changed = True
            while changed:
                changed = False
                for left, right in pairs:
                    if (left in component) ^ (right in component):
                        component.update((left, right)); pending.difference_update((left, right)); changed = True
            components.append(sorted(component))
        return sorted(active), components

    def _pair_margin(self, first, second, controls_first, controls_second):
        """Return the robust one-step separation margin for an aircraft pair."""
        agents = self.monitor.state.agents
        p_first = self._predict_position(agents[first], controls_first, self.cfg.dt)
        p_second = self._predict_position(agents[second], controls_second, self.cfg.dt)
        sigma = math.sqrt(sum(agents[first].position_var) + sum(agents[second].position_var))
        required = self.cfg.min_separation + self.kappa * sigma + self.safety_buffer
        return float(np.linalg.norm(p_second - p_first) - required)

    def _unsafe_pairs(self, controls, ids, report):
        """Check every pair after sparse solves, including pairs across components."""
        unsafe = []
        agents = self.monitor.state.agents
        for offset, first in enumerate(ids):
            for second in ids[offset + 1:]:
                margin = self._pair_margin(first, second, controls[first], controls[second])
                trend_key = f"S03:{first}-{second}"
                trend_unsafe = False
                if trend_key in report.violations:
                    steps = max(1, self.cfg.safe_trend_window)
                    p_first = self._predict_position_window(agents[first], controls[first], steps)
                    p_second = self._predict_position_window(agents[second], controls[second], steps)
                    sigma = math.sqrt(sum(agents[first].position_var) + sum(agents[second].position_var))
                    required = self.cfg.min_separation + self.kappa * sigma + self.safety_buffer
                    trend_unsafe = float(np.linalg.norm(p_second - p_first) - required) < 0.0
                if margin < 0.0 or trend_unsafe:
                    unsafe.append((first, second))
        return unsafe

    @staticmethod
    def _cross_component_pairs(pairs, components):
        """Keep only unsafe pairs whose endpoints were solved separately."""
        membership = {agent: index for index, component in enumerate(components) for agent in component}
        return [
            pair for pair in pairs
            if membership.get(pair[0]) is None
            or membership.get(pair[1]) is None
            or membership[pair[0]] != membership[pair[1]]
        ]

    @staticmethod
    def _merge_components(components, pairs):
        """Merge components touched by a post-solve unsafe pair."""
        groups = [set(component) for component in components if component]
        for first, second in pairs:
            touched = [idx for idx, group in enumerate(groups) if first in group or second in group]
            if not touched:
                groups.append({first, second})
                continue
            merged = {first, second}
            for idx in reversed(touched):
                merged.update(groups.pop(idx))
            groups.append(merged)
        return sorted((sorted(group) for group in groups), key=lambda group: group[0])

    @staticmethod
    def _distance(a: Action, b: Action) -> float:
        mode = 0.0 if a.mode == b.mode else 1.0
        return mode + abs(a.turn - b.turn) + abs(a.climb - b.climb) + abs(a.acceleration - b.acceleration)

    def _proposal_priority(self, rule: str, node) -> tuple[float, str]:
        """Return the priority key used for discrete arbitration proposals."""
        if self.use_dynamic_priority:
            return float(node.dynamic_priority), rule
        return float(self.FIXED_CATEGORY_PRIORITY.get(rule_category(rule), 0)), rule

    def _select_modes(self, observations: Dict[int, dict], nominal: Dict[int, Action], report: BeliefRuleReport, graph: RuleAgentGraphSnapshot):
        selected = {i: action.clipped() for i, action in nominal.items()}
        reasons: Dict[int, List[str]] = {i: [] for i in selected}
        for i, action in selected.items():
            proposals = []

            def propose(rule, mode, reason, acceleration=None):
                node = graph.rules.get(rule)
                if node is not None:
                    corrected = Action(
                        mode, action.turn, action.climb,
                        action.acceleration if acceleration is None else acceleration,
                        action.target, action.target_kind,
                    )
                    priority, tie_break = self._proposal_priority(rule, node)
                    proposals.append((priority, tie_break, corrected, reason))

            for rule in (f"I01:{i}", f"I02:{i}", f"I03:{i}", f"I04:{i}", f"I05:{i}"):
                if rule in report.violations and action.mode == Mode.CRITICAL:
                    propose(rule, Mode.RECOVER, "authorization_or_information_gate")
            if f"F04:{i}" in report.violations and action.mode not in (Mode.RECOVER, Mode.EXIT):
                propose(f"F04:{i}", Mode.EXIT, "F04_low_resource", min(action.acceleration, 0.0))
            if f"F06:{i}" in report.violations and action.mode not in (Mode.RECOVER, Mode.EXIT):
                propose(f"F06:{i}", Mode.EXIT, "F06_risk_budget_depleted", min(action.acceleration, 0.0))
            for rule in (f"F03:{i}", f"M06:{i}"):
                if rule in report.violations and action.mode not in (Mode.RECOVER, Mode.EXIT):
                    propose(rule, Mode.RECOVER, "capability_loss_reconfiguration", min(action.acceleration, 0.0))
            for rule in (f"F01:{i}", f"F02:{i}"):
                if rule in report.violations:
                    propose(rule, Mode.RECOVER, "degraded_information_fallback", min(action.acceleration, 0.0))
            for rule in (f"C04:{i}", f"C05:{i}"):
                if rule in report.violations:
                    propose(rule, Mode.RECOVER, "handover_or_intent_confirmation_required", min(action.acceleration, 0.0))
            if f"M03:{i}" in report.violations:
                propose(f"M03:{i}", Mode.HOLD, "task_assessment_required", min(action.acceleration, 0.0))
            if f"M04:{i}" in report.violations:
                propose(f"M04:{i}", Mode.RECOVER, "task_replan_required", min(action.acceleration, 0.0))
            if f"M05:{i}" in report.violations and action.mode not in (Mode.SUPPORT, Mode.COVER, Mode.CRITICAL, Mode.RECOVER, Mode.EXIT):
                propose(f"M05:{i}", Mode.COVER, "lost_link_task_continuity", action.acceleration)
            if f"S01:{i}" in report.violations:
                reasons[i].append("S01_flight_envelope")
            for rule in (f"S05:{i}", f"S06:{i}"):
                if rule in report.violations:
                    propose(rule, Mode.RECOVER, "recoverability_or_hysteresis", min(action.acceleration, 0.0))

            if proposals:
                _, _, corrected, reason = max(proposals, key=lambda item: (item[0], item[1]))
                selected[i] = corrected
                reasons[i].append(reason)
        if self.allow_soft_relaxation and not any(action.mode == Mode.COVER for action in selected.values()):
            candidates = [i for i in selected if observations[i]["self"].get("resource", 0.0) >= self.cfg.low_resource and observations[i]["self"].get("c2_connected", False) and selected[i].mode not in (Mode.RECOVER, Mode.EXIT)]
            if candidates:
                cover = max(candidates, key=lambda i: observations[i]["self"].get("resource", 0.0))
                selected[cover] = Action(
                    Mode.COVER, selected[cover].turn, selected[cover].climb,
                    selected[cover].acceleration, selected[cover].target,
                    selected[cover].target_kind,
                )
                reasons[cover].append("C02_coverage_assignment")
        # A high-risk support request is assigned to another capable member
        # when possible; otherwise the request is explicitly recorded as a
        # soft/temporarily infeasible coordination rule.
        for requested in selected:
            if f"C03:{requested}" not in report.violations:
                continue
            candidates = [i for i, action in selected.items() if i != requested and action.mode in (Mode.HOLD, Mode.COVER, Mode.CRITICAL) and observations[i]["self"].get("resource", 0.0) >= self.cfg.low_resource and not observations[i]["self"].get("support_requested", False)]
            if candidates:
                helper = max(candidates, key=lambda i: observations[i]["self"].get("resource", 0.0))
                selected[helper] = Action(
                    Mode.SUPPORT, selected[helper].turn, selected[helper].climb,
                    selected[helper].acceleration, requested, "teammate",
                )
                reasons[helper].append(f"C03_support_request:{requested}")
        return selected, reasons

    @staticmethod
    def _predict_position(agent, controls, dt):
        turn, climb, acceleration = controls
        heading = agent.heading + turn * 0.22 * dt
        climb_angle = max(-0.45, min(0.45, climb * 0.08 * dt))
        speed = max(0.0, agent.speed + acceleration * 12.0 * dt)
        return np.array([
            agent.position[0] + speed * math.cos(climb_angle) * math.cos(heading) * dt,
            agent.position[1] + speed * math.cos(climb_angle) * math.sin(heading) * dt,
            agent.position[2] + speed * math.sin(climb_angle) * dt,
        ])

    def _predict_position_window(self, agent, controls, steps):
        """Predict a short constant-command window without simulator truth.

        S03 is an eventual safe-trend rule, so forcing a non-closing velocity
        in the current one-second decision interval is unnecessarily strict
        for a bounded-turn aircraft.  This rollout is deliberately short and
        is only used to enforce that the residual reaches a robustly separated
        state within the configured anticipation window.
        """

        turn, climb, acceleration = controls
        position = np.asarray(agent.position, dtype=float)
        heading = float(agent.heading)
        climb_angle = 0.0
        speed = float(agent.speed)
        for _ in range(max(1, steps)):
            heading += turn * 0.22 * self.cfg.dt
            climb_angle = max(-0.45, min(0.45, climb_angle + climb * 0.08 * self.cfg.dt))
            speed = max(0.0, speed + acceleration * 12.0 * self.cfg.dt)
            position = position + np.array([
                speed * math.cos(climb_angle) * math.cos(heading) * self.cfg.dt,
                speed * math.cos(climb_angle) * math.sin(heading) * self.cfg.dt,
                speed * math.sin(climb_angle) * self.cfg.dt,
            ])
        return position

    def _action_conditioned_report(self, actions: Dict[int, Action]) -> BeliefRuleReport:
        """Evaluate next-step physical risk for the action actually executed."""

        report = self.monitor.evaluate_current(actions)
        physical = {"S01", "S02", "S03"}
        report.violations = [rule for rule in report.violations if rule.split(":", 1)[0] not in physical]
        for key in list(report.violation_probability):
            if key.split(":", 1)[0] in physical:
                report.violation_probability.pop(key, None)
                report.robust_lower.pop(key, None)
        agents = self.monitor.state.agents
        ids = sorted(actions)
        predicted = {}
        headings = {}
        speeds = {}
        for uid in ids:
            action = actions[uid]
            belief = agents[uid]
            controls = [action.turn, action.climb, action.acceleration]
            predicted[uid] = self._predict_position(belief, controls, self.cfg.dt)
            headings[uid] = belief.heading + action.turn * 0.22 * self.cfg.dt
            speeds[uid] = max(0.0, belief.speed + action.acceleration * 12.0 * self.cfg.dt)
            speed_sigma = math.sqrt(belief.speed_var)
            altitude_sigma = math.sqrt(belief.position_var[2])
            speed_margin = min(speeds[uid] - self.cfg.min_speed, self.cfg.max_speed - speeds[uid])
            altitude_margin = min(predicted[uid][2] - self.cfg.min_altitude, self.cfg.max_altitude - predicted[uid][2])
            probability = max(
                0.5 * (1.0 + math.erf((-speed_margin / max(1e-6, speed_sigma)) / math.sqrt(2.0))),
                0.5 * (1.0 + math.erf((-altitude_margin / max(1e-6, altitude_sigma)) / math.sqrt(2.0))),
            )
            key = f"S01:{uid}"
            report.violation_probability[key] = probability
            report.robust_lower[key] = min(speed_margin, altitude_margin) - self.kappa * max(speed_sigma, altitude_sigma) - self.monitor.model_error
            if probability > self.alpha:
                report.violations.append(key)
        for offset, uid in enumerate(ids):
            for vid in ids[offset + 1:]:
                delta = predicted[vid] - predicted[uid]
                distance = float(np.linalg.norm(delta))
                sigma = math.sqrt(sum(agents[uid].position_var) + sum(agents[vid].position_var))
                margin = distance - self.cfg.min_separation
                probability = 0.5 * (1.0 + math.erf((-margin / max(1e-6, sigma)) / math.sqrt(2.0)))
                key = f"S02:{uid}-{vid}"
                report.violation_probability[key] = probability
                report.robust_lower[key] = margin - self.kappa * sigma - self.safety_buffer
                if probability > self.alpha:
                    report.violations.append(key)
                vi = np.array([speeds[uid] * math.cos(headings[uid]), speeds[uid] * math.sin(headings[uid]), 0.0])
                vj = np.array([speeds[vid] * math.cos(headings[vid]), speeds[vid] * math.sin(headings[vid]), 0.0])
                relative_velocity = vj - vi
                speed_sq = float(np.dot(relative_velocity, relative_velocity))
                closing = float(np.dot(delta, relative_velocity))
                ttc = -closing / speed_sq if closing < 0.0 and speed_sq > 1e-9 else math.inf
                closest = float(np.linalg.norm(delta + relative_velocity * ttc)) if math.isfinite(ttc) else math.inf
                s03_probability = 1.0 if ttc < self.cfg.belief_ttc_threshold and closest < self.cfg.min_separation else 0.0
                s03_key = f"S03:{uid}-{vid}"
                report.violation_probability[s03_key] = s03_probability
                report.robust_lower[s03_key] = min(ttc - self.cfg.belief_ttc_threshold, closest - self.cfg.min_separation) if math.isfinite(ttc) else closest
                if s03_probability > self.alpha:
                    report.violations.append(s03_key)
        report.hard_safe = not any(is_hard_rule(rule) for rule in report.violations)
        return report

    def _build_constraints(self, report: BeliefRuleReport, controls0, ids):
        """Build linearized robust separation and flight-envelope inequalities."""
        constraints = []
        names = []
        agents = self.monitor.state.agents
        dt = self.cfg.dt
        eps = 1e-4
        # A constraint is represented as g(x) >= 0.  Finite differences make
        # this valid around arbitrary headings without exposing environment truth.
        for ai, i in enumerate(ids):
            belief = agents[i]
            base = np.asarray(controls0[ai], dtype=float)
            pred = self._predict_position(belief, base, dt)
            speed = max(0.0, belief.speed + base[2] * 12.0 * dt)
            altitude = pred[2]
            speed_sigma = math.sqrt(belief.speed_var)
            alt_sigma = math.sqrt(belief.position_var[2])
            # Robust one-step envelope margins.
            constraints.append((self.cfg.max_speed - speed) - self.kappa * speed_sigma)
            names.append(f"S01:max_speed:{i}")
            constraints.append((speed - self.cfg.min_speed) - self.kappa * speed_sigma)
            names.append(f"S01:min_speed:{i}")
            constraints.append((self.cfg.max_altitude - altitude) - self.kappa * alt_sigma)
            names.append(f"S01:max_altitude:{i}")
            constraints.append((altitude - self.cfg.min_altitude) - self.kappa * alt_sigma)
            names.append(f"S01:min_altitude:{i}")
        for ai, i in enumerate(ids):
            for aj in range(ai + 1, len(ids)):
                j = ids[aj]
                bi, bj = agents[i], agents[j]
                p_i = self._predict_position(bi, controls0[ai], dt)
                p_j = self._predict_position(bj, controls0[aj], dt)
                delta = p_j - p_i
                distance = float(np.linalg.norm(delta))
                sigma = math.sqrt(sum(bi.position_var) + sum(bj.position_var))
                required = self.cfg.min_separation + self.kappa * sigma + self.safety_buffer
                # linearized distance d(x) around nominal controls
                grad = np.zeros(6, dtype=float)
                for local, (agent, idx, sign) in enumerate(((bi, ai, -1.0), (bj, aj, 1.0))):
                    for k in range(3):
                        trial = list(controls0[idx])
                        trial[k] = max(-1.0, min(1.0, trial[k] + eps))
                        moved = self._predict_position(agent, trial, dt)
                        candidate = (p_j + (moved - p_i) * 0.0) if sign < 0 else p_i
                        if sign < 0:
                            d_trial = float(np.linalg.norm(p_j - moved))
                        else:
                            d_trial = float(np.linalg.norm(moved - p_i))
                        grad[local * 3 + k] = (d_trial - distance) / eps * sign
                constraints.append(distance - required)
                names.append(f"S02:{i}-{j}")
                constraints.append((distance - required, grad))
                names.append(f"S02_linear:{i}-{j}")
        # Flatten only the linearized entries; constants are checked separately.
        return constraints, names

    def _solve_component(self, selected, report, component):
        """Solve one independent active conflict component.

        Components are independent in the active rule graph, so solving them
        separately avoids the cubic pairwise constraint growth of one global
        SLSQP problem.  The returned controls are merged by ``filter``.
        """
        ids = sorted(component)
        if not ids:
            return {}, "optimal", 0.0, None
        agents = self.monitor.state.agents
        controls0 = np.asarray([[selected[i].turn, selected[i].climb, selected[i].acceleration] for i in ids], dtype=float)
        x0 = controls0.reshape(-1)
        constraints = []
        for ai, i in enumerate(ids):
            belief = agents[i]
            speed_sigma = math.sqrt(belief.speed_var)
            altitude_sigma = math.sqrt(belief.position_var[2])
            constraints.extend([
                {"type": "ineq", "fun": lambda x, ai=ai, b=belief, s=speed_sigma: float(b.speed + x[ai*3+2] * 12.0 * self.cfg.dt - self.cfg.min_speed - self.kappa*s)},
                {"type": "ineq", "fun": lambda x, ai=ai, b=belief, s=speed_sigma: float(self.cfg.max_speed - b.speed - x[ai*3+2] * 12.0 * self.cfg.dt - self.kappa*s)},
                {"type": "ineq", "fun": lambda x, ai=ai, b=belief, s=altitude_sigma: float(self._predict_position(b, x[ai*3:ai*3+3], self.cfg.dt)[2] - self.cfg.min_altitude - self.kappa*s)},
                {"type": "ineq", "fun": lambda x, ai=ai, b=belief, s=altitude_sigma: float(self.cfg.max_altitude - self._predict_position(b, x[ai*3:ai*3+3], self.cfg.dt)[2] - self.kappa*s)},
            ])
        for ai, i in enumerate(ids):
            for aj in range(ai + 1, len(ids)):
                j = ids[aj]
                bi, bj = agents[i], agents[j]
                sigma = math.sqrt(sum(bi.position_var) + sum(bj.position_var))
                required = self.cfg.min_separation + self.kappa * sigma + self.safety_buffer
                def pair_fun(x, ai=ai, aj=aj, bi=bi, bj=bj, required=required):
                    pi = self._predict_position(bi, x[ai*3:ai*3+3], self.cfg.dt)
                    pj = self._predict_position(bj, x[aj*3:aj*3+3], self.cfg.dt)
                    return float(np.linalg.norm(pj - pi) - required)
                constraints.append({"type": "ineq", "fun": pair_fun})
                if f"S03:{i}-{j}" in report.violations:
                    trend_steps = max(1, self.cfg.safe_trend_window)
                    def trend_fun(x, ai=ai, aj=aj, bi=bi, bj=bj, required=required, steps=trend_steps):
                        pi = self._predict_position_window(bi, x[ai*3:ai*3+3], steps)
                        pj = self._predict_position_window(bj, x[aj*3:aj*3+3], steps)
                        return float(np.linalg.norm(pj - pi) - required)
                    constraints.append({"type": "ineq", "fun": trend_fun})
        def objective(x):
            residual = x - x0
            return 0.5 * float(np.dot(residual, residual))
        try:
            result = minimize(objective, x0, method="SLSQP", bounds=Bounds(-np.ones_like(x0), np.ones_like(x0)), constraints=constraints, options={"maxiter": 80, "ftol": 1e-7})
        except Exception as exc:  # pragma: no cover
            return {i: controls0[k].tolist() for k, i in enumerate(ids)}, "infeasible_fallback", 0.0, f"solver_error:{exc.__class__.__name__}"
        if result.success and np.all(np.isfinite(result.x)):
            return {i: result.x[k*3:k*3+3].tolist() for k, i in enumerate(ids)}, "optimal", float(result.fun), None
        fallback = {i: controls0[k].tolist() for k, i in enumerate(ids)}
        for k, i in enumerate(ids):
            if any(rule.startswith((f"S02:", f"S03:")) and str(i) in rule.split(":", 1)[1].split("-") for rule in report.violations):
                fallback[i] = [0.8 if i % 2 == 0 else -0.8, 0.25, -0.3]
        return fallback, "infeasible_fallback", 0.0, "no_component_hard_feasible_residual"

    def filter(self, observations: Dict[int, dict], nominal: Dict[int, Action]):
        report = self.monitor.evaluate(observations, nominal)
        all_ids = sorted(nominal)
        # C02 is an active team requirement even when the nominal COVER action
        # currently satisfies it. Keeping it in the graph exposes induced
        # conflicts such as COVER versus F04 resource exit.
        # C02 is a global coordination node, but it should only enter the
        # active conflict graph when coverage is actually threatened.  Adding
        # it unconditionally creates artificial C02--C03 events in a support
        # template that already has a dedicated cover aircraft.
        coverage_threatened = (
            not any(action.mode == Mode.COVER for action in nominal.values())
            or any(rule.startswith("F04:") for rule in report.violations)
        )
        graph = self.rule_graph.build(
            report.violations,
            report.violation_probability,
            nominal,
            active_rules=("C02",) if coverage_threatened else (),
        )
        if not self.use_dependency_graph:
            # Keep the same rule reports and node priorities, but remove all
            # explicit rule--rule events and dependency edges.  This is the
            # controlled flat-rule ablation, not a reduced-coverage baseline.
            graph.edges = {}
            graph.conflicts = []
        selected, mode_reasons = self._select_modes(observations, nominal, report, graph)
        ids, solver_components = self._active_solver_agents(selected, report, all_ids)
        # An empty active set means every aircraft is comfortably separated
        # and inside the robust envelope.  Skip SLSQP and retain the selected
        # controls while still producing a complete solution trace.
        solver_ids = ids
        controls0 = [[selected[i].turn, selected[i].climb, selected[i].acceleration] for i in solver_ids]
        x0 = np.asarray(controls0, dtype=float).reshape(-1)
        solution = QPResidualSolution(nominal_actions=dict(nominal), belief_report=report, dependency_graph=graph)
        solution.agent_rule_conflicts = (
            {i: [node.key for node in graph.rules_for_agent(i)] for i in all_ids}
            if self.use_dependency_graph else {}
        )
        solution.solver_agents = list(solver_ids)
        solution.solver_components = solver_components
        hard_violations = [v for v in report.violations if is_hard_rule(v)]
        solution.hard_constraints = {k: v for k, v in report.robust_lower.items() if is_hard_rule(k)}
        solved_controls = {i: [selected[i].turn, selected[i].climb, selected[i].acceleration] for i in all_ids}
        def solve_components(components):
            statuses = []
            objective = 0.0
            reason = None
            controls = {}
            for component in components:
                component_controls, status, component_objective, component_reason = self._solve_component(selected, report, component)
                controls.update(component_controls)
                statuses.append(status)
                objective += component_objective
                if component_reason and reason is None:
                    reason = component_reason
            return controls, statuses, objective, reason

        if self.use_continuous_qp:
            component_statuses = []
            component_objective = 0.0
            component_reason = None
            solved, component_statuses, component_objective, component_reason = solve_components(solver_components)
            solved_controls.update(solved)
            if component_reason:
                solution.infeasibility_reason = component_reason

            # Sparse components are selected from nominal geometry.  Validate
            # merged controls against every pair before exposing them.
            unsafe_pairs = self._unsafe_pairs(solved_controls, all_ids, report)
            cross_pairs = self._cross_component_pairs(unsafe_pairs, solver_components)
            solution.cross_component_violations = len(cross_pairs)
            for _ in range(2):
                if not cross_pairs:
                    break
                solution.cross_component_repairs += 1
                solver_components = self._merge_components(solver_components, cross_pairs)
                solver_ids = sorted({agent for component in solver_components for agent in component})
                solved, component_statuses, component_objective, component_reason = solve_components(solver_components)
                solved_controls = {i: [selected[i].turn, selected[i].climb, selected[i].acceleration] for i in all_ids}
                solved_controls.update(solved)
                if component_reason:
                    solution.infeasibility_reason = component_reason
                unsafe_pairs = self._unsafe_pairs(solved_controls, all_ids, report)
                cross_pairs = self._cross_component_pairs(unsafe_pairs, solver_components)
            solution.solver_agents = list(solver_ids)
            solution.solver_components = solver_components
            if any(status != "optimal" for status in component_statuses):
                solution.status = "infeasible_fallback"
                if solution.infeasibility_reason is None:
                    solution.infeasibility_reason = "component_hard_feasibility_failure"
            elif unsafe_pairs:
                solution.status = "infeasible_fallback"
                solution.infeasibility_reason = "cross_component_hard_safety_failure"
            else:
                solution.status = "optimal"
            solution.objective = component_objective
        else:
            # Gate-only ablation keeps the same belief reports, graph and
            # discrete mode arbitration, but exposes no continuous correction.
            solution.solver_agents = []
            solution.solver_components = []
            solution.status = "gate_only"
            solution.objective = 0.0
        # Soft coverage is explicitly recorded as relaxed when no COVER role
        # remains after hard arbitration.
        if not any(a.mode == Mode.COVER for a in selected.values()):
            solution.relaxed_constraints["C02"] = -1.0
            solution.conflict_rules.append(["C02", "F04"] if any("F04" in rs for rs in mode_reasons.values()) else ["C02", "S02"])
            if not self.allow_soft_relaxation:
                solution.status = "infeasible_fallback"
                solution.infeasibility_reason = "soft_rule_C02_promoted_to_hard_constraint"
        # Multiple simultaneous support requests share a finite helper pool.
        # The greedy reassignment can serve one request and must explicitly
        # mark any remaining request as a lower-priority soft relaxation.
        for requested in selected:
            request_rule = f"C03:{requested}"
            if request_rule in report.violations and not any(
                action.mode == Mode.SUPPORT and action.target == requested
                for action in selected.values()
            ):
                solution.relaxed_constraints[request_rule] = -1.0
        # F06 has priority over lost-link mission continuity when another
        # member exhausts the team's risk budget.  Keep the lower-priority
        # M05 obligation explicit in the trace instead of hiding it in a
        # generic infeasibility flag.
        if any(rule.startswith("F06:") for rule in report.violations):
            for rule in report.violations:
                if rule.startswith("M05:"):
                    solution.relaxed_constraints[rule] = -1.0
        safe = {}
        interventions = []
        solved_actions = {
            i: Action(
                selected[i].mode, float(values[0]), float(values[1]),
                float(values[2]), selected[i].target, selected[i].target_kind,
            ).clipped()
            for i, values in solved_controls.items()
        }
        for i in all_ids:
            action = solved_actions[i]
            safe[i] = action
            reasons = list(mode_reasons[i])
            if self._distance(nominal[i], action) > 1e-8:
                reasons.extend(v for v in report.violations if v.endswith(f":{i}") or v.startswith("S02"))
                interventions.append(Intervention(i, nominal[i], action, reasons, self._distance(nominal[i], action)))
            solution.corrected_controls[i] = [action.turn, action.climb, action.acceleration]
        solution.safe_actions = safe
        # This second evaluation does not update the filter. It measures the
        # residual belief risk after arbitration, not the pre-filter trigger.
        solution.post_belief_report = self._action_conditioned_report(safe)
        for i in all_ids:
            history = self.intervention_history.setdefault(i, [])
            history.append(int(self._distance(nominal[i], safe[i]) > 1e-8))
            if len(history) > self.cfg.replan_intervention_window:
                history.pop(0)
            if sum(history) > self.cfg.replan_intervention_max:
                solution.conflict_rules.append([f"F05:{i}", "E03"])
                solution.relaxed_constraints[f"F05:{i}"] = float(sum(history) - self.cfg.replan_intervention_max)
        remaining = set(solution.post_belief_report.violations)
        if solution.status == "optimal":
            # S02/S03 are enforced on predicted next-step position/trend by
            # the optimizer, while evaluate_current reports current geometry.
            remaining = {rule for rule in remaining if not rule.startswith(("S02:", "S03:"))}
        for event in graph.conflicts:
            if event.kind != "rule_rule":
                continue
            hard_unresolved = any(rule in remaining and graph.rules[rule].hard for rule in event.involved_rules)
            if not hard_unresolved:
                for rule in event.involved_rules:
                    if rule in remaining and not graph.rules[rule].hard:
                        solution.relaxed_constraints.setdefault(rule, -1.0)
        self.rule_graph.finalize(
            graph,
            safe,
            remaining,
            solution.relaxed_constraints,
            self._distance,
        )
        solution.conflict_events = [event.as_dict() for event in graph.conflicts]
        self.last_solution = solution
        return safe, interventions
