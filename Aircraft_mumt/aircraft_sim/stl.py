"""Trajectory-level robustness summaries for the hard rules in the outline."""

import math
from dataclasses import dataclass
from typing import Iterable, List

from .rules import RuleMonitor, RuleReport


@dataclass
class TrajectoryReport:
    robustness: dict
    first_violation: dict
    joint_hard_satisfied: bool


class STLMonitor:
    """Computes conservative min-robustness for always-style rules."""

    def __init__(self, monitor: RuleMonitor):
        self.monitor = monitor

    def evaluate(self, truths: Iterable[dict], actions: Iterable[dict] | None = None) -> TrajectoryReport:
        truths = list(truths)
        actions = list(actions) if actions is not None else [None] * len(truths)
        margins = {}
        first = {}
        hard_ok = True
        for t, (truth, action) in enumerate(zip(truths, actions)):
            report: RuleReport = self.monitor.evaluate_truth(truth, action)
            for name, value in report.margins.items():
                margins[name] = min(margins.get(name, math.inf), value)
            for violation in report.violations:
                rule = violation.split(":", 1)[0]
                first.setdefault(rule, t)
            hard_ok = hard_ok and report.hard_safe
        # Rules with no explicit margin still get a useful Boolean robustness value.
        for rule in set(first):
            margins.setdefault(rule, -1.0)
        return TrajectoryReport(margins, first, hard_ok)
