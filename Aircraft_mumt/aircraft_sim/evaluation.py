"""Frozen-scenario evaluation for baselines and shielded policies."""

from collections import Counter
from dataclasses import dataclass, field
import math
import time
from typing import Callable, Dict, Iterable, Optional

from .metrics import EpisodeMetrics
from .rules import is_hard_rule, rule_family
from .stl import STLMonitor


@dataclass
class EvaluationResult:
    policy: str
    episodes: int
    mean_reward: float
    joint_satisfaction_rate: float
    mean_intervention_rate: float
    mean_intervention_distance: float
    violation_rate: Dict[str, float]
    mean_first_violation_step: Dict[str, float]
    initial_infeasible_rate: float = 0.0
    online_hard_violation_rate: float = 0.0
    post_shield_hard_violation_rate: float = 0.0
    post_truth_hard_violation_rate: float = 0.0
    qp_infeasible_rate: float = 0.0
    conflict_rate: float = 0.0
    replan_trigger_rate: float = 0.0
    mean_belief_risk: float = 0.0
    mean_pre_filter_belief_risk: float = 0.0
    mean_post_filter_belief_risk: float = 0.0
    # Wall-clock time to construct the nominal action, measured for every
    # policy. This is separate from the safety-layer solve time below.
    mean_actor_time_ms: float = 0.0
    mean_online_time_ms: float = 0.0
    tail_violation_rate: float = 0.0
    risk_calibration_brier: float = 0.0
    initially_feasible_episodes: int = 0
    conditional_joint_satisfaction_rate: float = 0.0
    rule_conflict_event_rate: float = 0.0
    rule_rule_conflict_step_rate: float = 0.0
    rule_rule_resolution_rate: float = 0.0
    mean_conflict_residual: float = 0.0
    mean_solver_agents: float = 0.0
    mean_solver_components: float = 0.0
    max_solver_component_size: float = 0.0
    cross_component_violation_rate: float = 0.0
    cross_component_repair_rate: float = 0.0
    episode_records: list = field(default_factory=list, repr=False)

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "episodes": self.episodes,
            "mean_reward": round(self.mean_reward, 5),
            "joint_satisfaction_rate": round(self.joint_satisfaction_rate, 5),
            "mean_intervention_rate": round(self.mean_intervention_rate, 5),
            "mean_intervention_distance": round(self.mean_intervention_distance, 5),
            "violation_rate": {k: round(v, 5) for k, v in sorted(self.violation_rate.items())},
            "mean_first_violation_step": {k: round(v, 3) for k, v in sorted(self.mean_first_violation_step.items())},
            "initial_infeasible_rate": round(self.initial_infeasible_rate, 5),
            "online_hard_violation_rate": round(self.online_hard_violation_rate, 5),
            "post_shield_hard_violation_rate": round(self.post_shield_hard_violation_rate, 5),
            "post_truth_hard_violation_rate": round(self.post_truth_hard_violation_rate, 5),
            "qp_infeasible_rate": round(self.qp_infeasible_rate, 5),
            "conflict_rate": round(self.conflict_rate, 5),
            "replan_trigger_rate": round(self.replan_trigger_rate, 5),
            "mean_belief_risk": round(self.mean_belief_risk, 5),
            "mean_pre_filter_belief_risk": round(self.mean_pre_filter_belief_risk, 5),
            "mean_post_filter_belief_risk": round(self.mean_post_filter_belief_risk, 5),
            "mean_actor_time_ms": round(self.mean_actor_time_ms, 5),
            "mean_online_time_ms": round(self.mean_online_time_ms, 5),
            "tail_violation_rate": round(self.tail_violation_rate, 5),
            "risk_calibration_brier": round(self.risk_calibration_brier, 5),
            "initially_feasible_episodes": self.initially_feasible_episodes,
            "conditional_joint_satisfaction_rate": round(self.conditional_joint_satisfaction_rate, 5),
            "rule_conflict_event_rate": round(self.rule_conflict_event_rate, 5),
            "rule_rule_conflict_step_rate": round(self.rule_rule_conflict_step_rate, 5),
            "rule_rule_resolution_rate": round(self.rule_rule_resolution_rate, 5),
            "mean_conflict_residual": round(self.mean_conflict_residual, 5),
            "mean_solver_agents": round(self.mean_solver_agents, 5),
            "mean_solver_components": round(self.mean_solver_components, 5),
            "max_solver_component_size": round(self.max_solver_component_size, 5),
            "cross_component_violation_rate": round(self.cross_component_violation_rate, 5),
            "cross_component_repair_rate": round(self.cross_component_repair_rate, 5),
        }


def evaluate_policy(env, monitor, policy: Callable, scenarios: Iterable[dict], shield=None, horizon: Optional[int] = None, name: str = "policy") -> EvaluationResult:
    scenarios = list(scenarios)
    rewards = []
    joint_ok = 0
    intervention_rates = []
    intervention_distances = []
    violation_episodes = Counter()
    first_steps = {}
    initial_infeasible = 0
    initially_feasible = 0
    conditional_joint_ok = 0
    online_unsafe_steps = 0
    post_shield_unsafe_steps = 0
    post_truth_unsafe_steps = 0
    total_steps = 0
    qp_infeasible_steps = 0
    conflict_steps = 0
    replan_steps = 0
    pre_filter_belief_risks = []
    post_filter_belief_risks = []
    actor_times = []
    online_times = []
    episode_violation_severity = []
    calibration_errors = []
    graph_conflict_events = 0
    graph_rule_rule_events = 0
    graph_resolved_rule_rule_events = 0
    graph_rule_rule_steps = 0
    graph_residuals = []
    solver_agent_counts = []
    solver_component_counts = []
    solver_component_sizes = []
    cross_component_violations = 0
    cross_component_repairs = 0
    episode_records = []
    for episode, scenario in enumerate(scenarios):
        # Reset both environment randomness and state so all baselines see frozen noise.
        env.rng.seed(env.cfg.seed + episode)
        if shield is not None and hasattr(shield, "monitor") and hasattr(shield.monitor, "reset"):
            shield.monitor.reset()
        observations = env.reset(scenario)
        initial_report = monitor.evaluate_truth(env.truth_state())
        initial_infeasible += int(not initial_report.hard_safe)
        if initial_report.hard_safe:
            initially_feasible += 1
        metrics = EpisodeMetrics()
        truths, actions_history = [], []
        episode_severity = []
        calibration_pairs = []
        for _ in range(horizon or env.cfg.horizon):
            actor_started = time.perf_counter()
            if hasattr(policy, "act_joint"):
                nominal = policy.act_joint(observations, deterministic=True)
            else:
                nominal = {i: policy(observations[i], i) for i in observations}
            actor_times.append((time.perf_counter() - actor_started) * 1000.0)
            if shield is None:
                actions, interventions = nominal, []
            else:
                started = time.perf_counter()
                actions, interventions = shield.filter(observations, nominal)
                online_times.append((time.perf_counter() - started) * 1000.0)
                if hasattr(shield, "last_solution"):
                    solution = shield.last_solution
                    solver_agent_counts.append(len(solution.solver_agents))
                    solver_component_counts.append(len(solution.solver_components))
                    solver_component_sizes.extend(len(component) for component in solution.solver_components)
                    cross_component_violations += int(solution.cross_component_violations > 0)
                    cross_component_repairs += int(solution.cross_component_repairs > 0)
                    qp_infeasible_steps += int(solution.status == "infeasible_fallback")
                    conflict_steps += int(any(not any(rule.startswith("F05:") for rule in group) for group in solution.conflict_rules))
                    replan_steps += int(any(any(rule.startswith("F05:") for rule in group) for group in solution.conflict_rules))
                    if solution.belief_report is not None:
                        report = solution.belief_report
                        online_unsafe_steps += int(not report.hard_safe)
                        pre_filter_belief_risks.append(max(
                            (value for key, value in report.violation_probability.items() if is_hard_rule(key)),
                            default=0.0,
                        ))
                    if solution.post_belief_report is not None:
                        post_shield_unsafe_steps += int(not solution.post_belief_report.hard_safe)
                        post_filter_belief_risks.append(max(
                            (value for key, value in solution.post_belief_report.violation_probability.items() if is_hard_rule(key)),
                            default=0.0,
                        ))
                    events = getattr(solution, "conflict_events", [])
                    graph_conflict_events += len(events)
                    rule_rule = [event for event in events if event.get("kind") == "rule_rule"]
                    graph_rule_rule_events += len(rule_rule)
                    graph_rule_rule_steps += int(bool(rule_rule))
                    graph_resolved_rule_rule_events += sum(event.get("resolved") is True for event in rule_rule)
                    for event in events:
                        graph_residuals.extend(float(value) for value in event.get("residual_by_agent", {}).values())
                elif hasattr(shield, "last_solve"):
                    # Baseline optimizers expose only generic solve status;
                    # they do not manufacture belief or conflict-graph data.
                    qp_infeasible_steps += int(shield.last_solve.status == "infeasible_fallback")
            result = env.step(actions)
            total_steps += 1
            truth_report = monitor.evaluate_truth(result.info["truth"], actions)
            # This is the executed-action truth outcome.  Keep it separate
            # from post_shield_unsafe_steps, which is a belief-space residual
            # risk and is unavailable for non-belief baselines.
            post_truth_unsafe_steps += int(not truth_report.hard_safe)
            if shield is not None and hasattr(shield, "last_solution") and shield.last_solution.post_belief_report is not None:
                post_probabilities = shield.last_solution.post_belief_report.violation_probability
                predicted_by_rule = {}
                for key, probability in post_probabilities.items():
                    if is_hard_rule(key):
                        family = rule_family(key)
                        predicted_by_rule[family] = max(predicted_by_rule.get(family, 0.0), float(probability))
                predicted = max(predicted_by_rule.values(), default=0.0)
                actual = 1.0 if not truth_report.hard_safe else 0.0
                calibration_errors.append((predicted - actual) ** 2)
                calibration_pairs.append({
                    "predicted": float(predicted),
                    "actual": float(actual),
                    "prediction_stage": "post_filter_executed_action",
                    "predicted_by_rule": predicted_by_rule,
                    "actual_rules": sorted({rule_family(name) for name in truth_report.violations if is_hard_rule(name)}),
                })
            episode_severity.append(float(len(truth_report.violations)))
            metrics.record(result.reward, truth_report, interventions)
            truths.append(result.info["truth"])
            actions_history.append(actions)
            observations = result.observations
            if result.done:
                break
        trajectory = STLMonitor(monitor).evaluate(truths, actions_history)
        episode_violation_severity.append(sum(episode_severity))
        rewards.append(metrics.reward)
        joint_ok += int(trajectory.joint_hard_satisfied)
        conditional_joint_ok += int(initial_report.hard_safe and trajectory.joint_hard_satisfied)
        intervention_rates.append(metrics.intervention_count / max(1, metrics.steps))
        intervention_distances.append(metrics.intervention_distance / max(1, metrics.intervention_count))
        for violation in trajectory.first_violation:
            violation_episodes[violation] += 1
        for rule, step in trajectory.first_violation.items():
            first_steps.setdefault(rule, []).append(step)
        episode_records.append({
            "episode": episode,
            "initial_hard_safe": initial_report.hard_safe,
            "reward": metrics.reward,
            "steps": metrics.steps,
            "joint_hard_satisfied": trajectory.joint_hard_satisfied,
            "violations": trajectory.first_violation,
            "intervention_count": metrics.intervention_count,
            "intervention_rate": metrics.intervention_count / max(1, metrics.steps),
            "intervention_distance": metrics.intervention_distance,
        })
        if calibration_pairs:
            episode_records[-1]["calibration_pairs"] = calibration_pairs
    count = max(1, len(scenarios))
    denom = max(1, total_steps)
    ordered_severity = sorted(episode_violation_severity, reverse=True)
    tail_count = max(1, int(math.ceil(0.2 * len(ordered_severity)))) if ordered_severity else 1
    tail_rate = sum(value > 0 for value in ordered_severity[:tail_count]) / tail_count if ordered_severity else 0.0
    mean_pre_risk = sum(pre_filter_belief_risks) / max(1, len(pre_filter_belief_risks))
    mean_post_risk = sum(post_filter_belief_risks) / max(1, len(post_filter_belief_risks))
    return EvaluationResult(
        policy=name,
        episodes=len(scenarios),
        mean_reward=sum(rewards) / count,
        joint_satisfaction_rate=joint_ok / count,
        mean_intervention_rate=sum(intervention_rates) / count,
        mean_intervention_distance=sum(intervention_distances) / count,
        violation_rate={rule: value / count for rule, value in violation_episodes.items()},
        mean_first_violation_step={rule: sum(values) / len(values) for rule, values in first_steps.items()},
        initial_infeasible_rate=initial_infeasible / count,
        online_hard_violation_rate=online_unsafe_steps / denom,
        post_shield_hard_violation_rate=post_shield_unsafe_steps / denom,
        post_truth_hard_violation_rate=post_truth_unsafe_steps / denom,
        qp_infeasible_rate=qp_infeasible_steps / denom,
        conflict_rate=conflict_steps / denom,
        replan_trigger_rate=replan_steps / denom,
        mean_belief_risk=mean_post_risk,
        mean_pre_filter_belief_risk=mean_pre_risk,
        mean_post_filter_belief_risk=mean_post_risk,
        mean_actor_time_ms=sum(actor_times) / max(1, len(actor_times)),
        mean_online_time_ms=sum(online_times) / max(1, len(online_times)),
        tail_violation_rate=tail_rate,
        risk_calibration_brier=sum(calibration_errors) / max(1, len(calibration_errors)),
        initially_feasible_episodes=initially_feasible,
        conditional_joint_satisfaction_rate=conditional_joint_ok / max(1, initially_feasible),
        rule_conflict_event_rate=graph_conflict_events / denom,
        rule_rule_conflict_step_rate=graph_rule_rule_steps / denom,
        rule_rule_resolution_rate=graph_resolved_rule_rule_events / max(1, graph_rule_rule_events),
        mean_conflict_residual=sum(graph_residuals) / max(1, len(graph_residuals)),
        mean_solver_agents=sum(solver_agent_counts) / max(1, len(solver_agent_counts)),
        mean_solver_components=sum(solver_component_counts) / max(1, len(solver_component_counts)),
        max_solver_component_size=max(solver_component_sizes, default=0),
        cross_component_violation_rate=cross_component_violations / max(1, len(solver_agent_counts)),
        cross_component_repair_rate=cross_component_repairs / max(1, len(solver_agent_counts)),
        episode_records=episode_records,
    )


def pareto_front(results: Iterable[EvaluationResult]) -> list[dict]:
    """Return non-dominated points: maximize reward/safety, minimize intervention."""
    results = list(results)
    front = []
    for candidate in results:
        dominated = False
        for other in results:
            no_worse = (other.mean_reward >= candidate.mean_reward and other.joint_satisfaction_rate >= candidate.joint_satisfaction_rate and other.mean_intervention_rate <= candidate.mean_intervention_rate)
            strictly = (other.mean_reward > candidate.mean_reward or other.joint_satisfaction_rate > candidate.joint_satisfaction_rate or other.mean_intervention_rate < candidate.mean_intervention_rate)
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            front.append({"policy": candidate.policy, "reward": candidate.mean_reward, "joint_satisfaction_rate": candidate.joint_satisfaction_rate, "intervention_rate": candidate.mean_intervention_rate})
    return front
