"""Small deterministic policies used for smoke tests and dataset generation."""

from .models import Action, Mode


def rule_agnostic_policy(observation: dict, agent_id: int = 0) -> Action:
    """A deliberately simple nominal policy; safety behavior must come from the shield."""
    return Action(Mode.CRITICAL if agent_id == 0 else (Mode.COVER if agent_id == 1 else Mode.SUPPORT), turn=0.1, acceleration=0.2)


def conservative_policy(observation: dict, agent_id: int = 0) -> Action:
    own = observation["self"]
    if own["resource"] < 0.25 or not own["c2_connected"]:
        return Action(Mode.RECOVER, acceleration=-0.2)
    if agent_id == 0 and own["id_confidence"] > 0.8:
        return Action(Mode.CRITICAL, turn=0.05)
    return Action(Mode.COVER if agent_id == 1 else Mode.HOLD, turn=0.03)


def fixed_penalty_policy(observation: dict, agent_id: int = 0) -> Action:
    """A fixed-penalty style baseline with no explicit shield."""
    own = observation["self"]
    if own["resource"] < 0.28 or not own["c2_connected"]:
        return Action(Mode.RECOVER, acceleration=-0.25)
    if own["id_confidence"] < 0.78:
        return Action(Mode.HOLD, turn=-0.04)
    if agent_id == 0:
        return Action(Mode.CRITICAL, turn=0.04, acceleration=0.05)
    return Action(Mode.COVER if agent_id == 1 else Mode.SUPPORT, turn=0.02)


def stl_robustness_policy(observation: dict, agent_id: int = 0) -> Action:
    """Greedy robustness-margin baseline using only the current observation."""
    own = observation["self"]
    teammates = observation.get("teammates", [])
    close = any(sum(v * v for v in item["relative_position"]) ** 0.5 < 1.25 * 800.0 for item in teammates)
    if close:
        return Action(Mode.RECOVER, turn=0.55 if agent_id % 2 == 0 else -0.55, climb=0.15, acceleration=-0.2)
    if own.get("resource", 1.0) < 0.25 or own.get("risk_budget", 1.0) < 0.3:
        return Action(Mode.EXIT, acceleration=-0.2)
    if agent_id == 1:
        return Action(Mode.COVER, turn=0.01)
    return Action(Mode.HOLD, turn=0.01)


def lagrangian_policy(observation: dict, agent_id: int = 0) -> Action:
    """Constraint-price baseline: raise conservative behavior as local risk grows."""
    own = observation["self"]
    risk = max((item.get("risk", 0.0) for item in observation.get("teammates", [])), default=0.0)
    if risk > 0.6 or own.get("resource", 1.0) < 0.22:
        return Action(Mode.RECOVER, turn=0.45 if agent_id % 2 == 0 else -0.45, climb=0.1, acceleration=-0.2)
    if agent_id == 1:
        return Action(Mode.COVER, turn=0.02)
    return Action(Mode.SUPPORT if risk > 0.3 else Mode.HOLD, turn=0.02)
