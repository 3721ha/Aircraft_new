"""Explicit rule-agent dependency graph and conflict records.

The graph makes the arbitration surface in note1 inspectable: which agents
affect a rule, which simultaneous requirements conflict, and whose action was
changed. It consumes belief-monitor rule keys such as ``I02:0`` and
``S02:0-1``; it never reads simulator truth.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Optional, Tuple

from .models import Action


CATEGORY_PRIORITY = {
    "physical": 4.0,
    "fault_resource": 4.0,
    "authorization": 3.0,
    "coordination": 2.0,
    "mission": 1.0,
    "efficiency": 0.5,
}


def rule_family(rule_key: str) -> str:
    return rule_key.split(":", 1)[0]


def rule_category(rule_key: str) -> str:
    prefix = rule_family(rule_key)[:1]
    if prefix == "S":
        return "physical"
    if prefix == "F":
        return "fault_resource"
    if prefix == "I":
        return "authorization"
    if prefix == "C":
        return "coordination"
    if prefix == "M":
        return "mission"
    return "efficiency"


def affected_agents(rule_key: str, all_agents: Iterable[int]) -> Tuple[int, ...]:
    if ":" not in rule_key:
        return tuple(sorted(all_agents))
    suffix = rule_key.split(":", 1)[1]
    parsed = []
    for token in suffix.split("-"):
        if token.isdigit():
            parsed.append(int(token))
    return tuple(sorted(set(parsed))) or tuple(sorted(all_agents))


@dataclass(frozen=True)
class RuleNode:
    key: str
    family: str
    category: str
    hard: bool
    affected_agents: Tuple[int, ...]
    violation_probability: float
    dynamic_priority: float

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "family": self.family,
            "category": self.category,
            "hard": self.hard,
            "affected_agents": list(self.affected_agents),
            "violation_probability": self.violation_probability,
            "dynamic_priority": self.dynamic_priority,
        }


@dataclass
class ConflictEvent:
    event_id: str
    kind: str
    involved_agents: Tuple[int, ...]
    involved_rules: Tuple[str, ...]
    rule_priorities: Dict[str, float]
    nominal_actions: Dict[int, Action]
    candidate_resolutions: List[str] = field(default_factory=list)
    selected_actions: Dict[int, Action] = field(default_factory=dict)
    relaxed_rules: Tuple[str, ...] = ()
    residual_by_agent: Dict[int, float] = field(default_factory=dict)
    resolved: Optional[bool] = None
    resolution_reason: str = ""

    def as_dict(self) -> dict:
        def action_dict(action: Action) -> dict:
            return {
                "mode": action.mode.value,
                "turn": action.turn,
                "climb": action.climb,
                "acceleration": action.acceleration,
                "target": action.target,
            }

        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "involved_agents": list(self.involved_agents),
            "involved_rules": list(self.involved_rules),
            "rule_priorities": self.rule_priorities,
            "nominal_actions": {str(i): action_dict(a) for i, a in self.nominal_actions.items()},
            "candidate_resolutions": self.candidate_resolutions,
            "selected_actions": {str(i): action_dict(a) for i, a in self.selected_actions.items()},
            "relaxed_rules": list(self.relaxed_rules),
            "residual_by_agent": {str(i): value for i, value in self.residual_by_agent.items()},
            "resolved": self.resolved,
            "resolution_reason": self.resolution_reason,
        }


@dataclass
class RuleAgentGraphSnapshot:
    agents: Tuple[int, ...]
    rules: Dict[str, RuleNode]
    edges: Dict[str, Tuple[int, ...]]
    conflicts: List[ConflictEvent]

    def rules_for_agent(self, agent: int) -> List[RuleNode]:
        nodes = [node for node in self.rules.values() if agent in node.affected_agents]
        return sorted(nodes, key=lambda node: (-node.dynamic_priority, node.key))

    def as_dict(self) -> dict:
        return {
            "agents": list(self.agents),
            "rules": {key: node.as_dict() for key, node in self.rules.items()},
            "edges": {key: list(agents) for key, agents in self.edges.items()},
            "conflicts": [event.as_dict() for event in self.conflicts],
        }


class RuleAgentDependencyGraph:
    """Build the active bipartite graph from belief-rule violations."""

    # These pairs instantiate the representative conflicts in note1. Matching
    # is symmetric and family based so per-agent keys remain traceable.
    CONFLICT_PAIRS = {
        frozenset(("C02", "F04")),
        frozenset(("C02", "F06")),
        frozenset(("C02", "F01")),
        frozenset(("C02", "S02")),
        frozenset(("C02", "S03")),
        frozenset(("C02", "C03")),
        frozenset(("C03", "S02")),
        frozenset(("C03", "S03")),
        frozenset(("C03", "F04")),
        frozenset(("C03", "F01")),
        frozenset(("F01", "C05")),
        frozenset(("F03", "E02")),
        frozenset(("F06", "M05")),
        frozenset(("M02", "I02")),
        frozenset(("M02", "I03")),
        frozenset(("M02", "F04")),
        frozenset(("M04", "S05")),
        frozenset(("M04", "F04")),
        frozenset(("E03", "S01")),
        frozenset(("E03", "S02")),
    }

    HARD_PREFIXES = ("S", "I", "F")

    @staticmethod
    def _priority(rule_key: str, probability: float, scope: int, ttv: Optional[float] = None) -> float:
        base = CATEGORY_PRIORITY[rule_category(rule_key)]
        urgency = 1.0 + (1.0 / max(1.0, ttv)) if ttv is not None and math.isfinite(ttv) else 1.0
        uncertainty = 1.0 + max(0.0, min(1.0, probability))
        scope_factor = 1.0 + 0.15 * max(0, scope - 1)
        return base * urgency * uncertainty * scope_factor

    def build(self, violations: Iterable[str], probabilities: Dict[str, float], nominal: Dict[int, Action], active_rules: Iterable[str] = ()) -> RuleAgentGraphSnapshot:
        agents = tuple(sorted(nominal))
        violation_set = set(violations)
        unique = sorted(violation_set | set(active_rules))
        rules: Dict[str, RuleNode] = {}
        for key in unique:
            impacted = affected_agents(key, agents)
            probability = float(probabilities.get(key, 1.0))
            category = rule_category(key)
            rules[key] = RuleNode(
                key=key,
                family=rule_family(key),
                category=category,
                hard=key.startswith(self.HARD_PREFIXES),
                affected_agents=impacted,
                violation_probability=probability,
                dynamic_priority=self._priority(key, probability, len(impacted)),
            )

        conflicts: List[ConflictEvent] = []
        for index, node in enumerate(node for node in rules.values() if node.key in violation_set):
            conflicts.append(ConflictEvent(
                event_id=f"action-rule-{index}",
                kind="action_rule",
                involved_agents=node.affected_agents,
                involved_rules=(node.key,),
                rule_priorities={node.key: node.dynamic_priority},
                nominal_actions={i: nominal[i] for i in node.affected_agents if i in nominal},
                candidate_resolutions=["retain_nominal", "change_affected_agent", "joint_reassignment", "safe_fallback"],
            ))

        nodes = list(rules.values())
        event_index = len(conflicts)
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1:]:
                shared_resource_conflict = left.family == right.family == "C03"
                shared_task_conflict = frozenset((left.family, right.family)) == frozenset(("F06", "M05"))
                if not shared_resource_conflict and not shared_task_conflict and frozenset((left.family, right.family)) not in self.CONFLICT_PAIRS:
                    continue
                if not shared_resource_conflict and not shared_task_conflict and not set(left.affected_agents).intersection(right.affected_agents):
                    continue
                # C03 requests compete for a common helper even when the
                # requesting aircraft are different; expose the full team so
                # the selected helper is visible in the arbitration trace.
                involved = tuple(sorted(set(agents) if (shared_resource_conflict or shared_task_conflict) else set(left.affected_agents) | set(right.affected_agents)))
                conflicts.append(ConflictEvent(
                    event_id=f"rule-rule-{event_index}",
                    kind="rule_rule",
                    involved_agents=involved,
                    involved_rules=(left.key, right.key),
                    rule_priorities={left.key: left.dynamic_priority, right.key: right.dynamic_priority},
                    nominal_actions={i: nominal[i] for i in involved if i in nominal},
                    candidate_resolutions=["change_lower_cost_agent", "reassign_team_role", "relax_lower_priority_rule", "safe_fallback"],
                ))
                event_index += 1

        return RuleAgentGraphSnapshot(agents, rules, {key: node.affected_agents for key, node in rules.items()}, conflicts)

    @staticmethod
    def finalize(snapshot: RuleAgentGraphSnapshot, safe: Dict[int, Action], post_violations: Iterable[str], relaxed_rules: Iterable[str], distance) -> None:
        remaining = set(post_violations)
        relaxed = set(relaxed_rules)
        for event in snapshot.conflicts:
            event.selected_actions = {i: safe[i] for i in event.involved_agents if i in safe}
            event.residual_by_agent = {
                i: float(distance(event.nominal_actions[i], safe[i]))
                for i in event.involved_agents if i in event.nominal_actions and i in safe
            }
            event.relaxed_rules = tuple(sorted(rule for rule in event.involved_rules if rule_family(rule) in relaxed or rule in relaxed))
            unresolved = [
                rule for rule in event.involved_rules
                if rule in remaining and rule not in relaxed and rule_family(rule) not in relaxed
            ]
            event.resolved = not unresolved
            if event.relaxed_rules:
                event.resolution_reason = "lower_priority_rule_relaxed"
            elif event.resolved and any(value > 1e-8 for value in event.residual_by_agent.values()):
                event.resolution_reason = "minimum_residual_action_change"
            elif event.resolved:
                event.resolution_reason = "nominal_action_already_acceptable"
            else:
                event.resolution_reason = "unresolved_or_infeasible"
