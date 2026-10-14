from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .rules import RuleReport


@dataclass
class EpisodeMetrics:
    reward: float = 0.0
    steps: int = 0
    intervention_count: int = 0
    intervention_distance: float = 0.0
    violations: Counter = field(default_factory=Counter)

    def record(self, reward: float, report: RuleReport, interventions: Iterable) -> None:
        self.reward += reward
        self.steps += 1
        self.violations.update(report.violations)
        events = list(interventions)
        self.intervention_count += len(events)
        self.intervention_distance += sum(event.distance for event in events)

    def summary(self) -> dict:
        return {
            "reward": round(self.reward, 4),
            "steps": self.steps,
            "intervention_rate": round(self.intervention_count / max(1, self.steps), 4),
            "mean_intervention_distance": round(self.intervention_distance / max(1, self.intervention_count), 4),
            "violations": dict(self.violations),
            "hard_safe_episode": not any(key.startswith(("S01", "S02", "S03", "S04", "S05", "S06", "I01", "I02", "I03", "I04", "I05", "I06", "F01", "F04", "F06")) for key in self.violations),
        }
