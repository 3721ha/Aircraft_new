"""Post-hoc rule-wise risk calibration utilities.

Calibration is deliberately separated from :class:`BeliefSTLMonitor`'s raw
probabilities. A calibrated value may be used for reporting or analysis, but
hard safety gating continues to use the conservative raw monitor output.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass
class RuleRiskCalibrator:
    mapping: Dict[str, Dict[int, float]]

    @staticmethod
    def fit(samples: Iterable[Tuple[str, float, float]]) -> "RuleRiskCalibrator":
        buckets: Dict[str, Dict[int, list]] = {}
        for rule, predicted, actual in samples:
            bucket = min(9, max(0, int(float(predicted) * 10.0)))
            count, positives = buckets.setdefault(rule, {}).setdefault(bucket, [0, 0.0])
            buckets[rule][bucket] = [count + 1, positives + float(actual)]
        mapping = {
            rule: {bucket: (values[1] + 1.0) / (values[0] + 2.0) for bucket, values in bins.items()}
            for rule, bins in buckets.items()
        }
        return RuleRiskCalibrator(mapping)

    def transform_probability(self, rule: str, probability: float) -> float:
        bucket = min(9, max(0, int(float(probability) * 10.0)))
        return float(self.mapping.get(rule, {}).get(bucket, probability))

    def transform_report(self, violation_probability: Dict[str, float]) -> Dict[str, float]:
        return {key: self.transform_probability(key.split(":", 1)[0], value) for key, value in violation_probability.items()}
