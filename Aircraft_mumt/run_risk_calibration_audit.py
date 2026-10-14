"""Evaluate belief-risk calibration on held-out final-protocol checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from aircraft_sim import (
    AircraftEnv,
    ConflictAwareQPSafetyShield,
    MAPPOConfig,
    MAPPOPolicy,
    PassThroughShield,
    RuleMonitor,
    RuleRiskCalibrator,
    SimConfig,
    evaluate_policy,
)
from run_paper_experiments import make_scenarios


def _checkpoint(root: Path, seed: int) -> Path:
    return root / "checkpoints" / "proposed_belief_stl_conflict_qp" / f"seed_{seed}.pt"


def _pairs(root: Path, seeds: list[int], episodes: int, horizon: int) -> tuple[list[dict], list[tuple[str, float, float]]]:
    aggregate, by_rule = [], []
    for seed in seeds:
        config = SimConfig(seed=seed, horizon=horizon)
        env = AircraftEnv(3, config)
        policy = MAPPOPolicy(env, PassThroughShield(), config=MAPPOConfig(), seed=seed)
        policy.model.load_state_dict(torch.load(_checkpoint(root, seed), map_location="cpu", weights_only=True))
        shield = ConflictAwareQPSafetyShield(config)
        result = evaluate_policy(
            env,
            RuleMonitor(config),
            lambda observation, uid, policy=policy: policy.act(observation, uid, deterministic=True),
            make_scenarios(3, seed, episodes),
            shield=shield,
            horizon=horizon,
            name="proposed_calibration",
        )
        for episode in result.episode_records:
            for pair in episode.get("calibration_pairs", []):
                aggregate.append({"seed": seed, **pair})
                actual_rules = set(pair.get("actual_rules", []))
                for rule, probability in pair.get("predicted_by_rule", {}).items():
                    by_rule.append((rule, float(probability), float(rule in actual_rules)))
    return aggregate, by_rule


def _brier_ece(pairs: list[dict]) -> dict:
    if not pairs:
        return {"count": 0, "brier": 0.0, "mean_predicted": 0.0, "empirical_rate": 0.0, "ece": 0.0, "bins": []}
    bins = []
    brier = sum((float(p["predicted"]) - float(p["actual"])) ** 2 for p in pairs) / len(pairs)
    mean_predicted = sum(float(p["predicted"]) for p in pairs) / len(pairs)
    empirical = sum(float(p["actual"]) for p in pairs) / len(pairs)
    ece = 0.0
    for index in range(10):
        lower, upper = index / 10.0, (index + 1) / 10.0
        # Keep bins disjoint.  The previous final-bin condition only checked
        # the upper bound, so every low-probability sample was counted again
        # in the 0.9-1.0 bin and inflated ECE.
        selected = [
            p
            for p in pairs
            if lower <= float(p["predicted"]) < upper
            or (index == 9 and lower <= float(p["predicted"]) <= upper)
        ]
        if selected:
            predicted = sum(float(p["predicted"]) for p in selected) / len(selected)
            observed = sum(float(p["actual"]) for p in selected) / len(selected)
            ece += len(selected) / len(pairs) * abs(predicted - observed)
        else:
            predicted = observed = 0.0
        bins.append({"lower": lower, "upper": upper, "count": len(selected), "mean_predicted": predicted, "empirical_rate": observed})
    return {"count": len(pairs), "brier": brier, "mean_predicted": mean_predicted, "empirical_rate": empirical, "ece": ece, "bins": bins}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final-protocol belief risk calibration")
    parser.add_argument("--proposed-root", type=Path, required=True)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55, 66])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[77, 88, 99, 111])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train_pairs, train_rule_pairs = _pairs(args.proposed_root, args.train_seeds, args.episodes, args.horizon)
    test_pairs, test_rule_pairs = _pairs(args.proposed_root, args.test_seeds, args.episodes, args.horizon)
    calibrator = RuleRiskCalibrator.fit(train_rule_pairs)

    calibrated_pairs = []
    for pair in test_pairs:
        transformed = {rule: calibrator.transform_probability(rule, probability) for rule, probability in pair.get("predicted_by_rule", {}).items()}
        calibrated_pairs.append({**pair, "predicted": max(transformed.values(), default=float(pair["predicted"]))})

    train_summary = _brier_ece(train_pairs)
    test_summary = _brier_ece(test_pairs)
    calibrated_summary = _brier_ece(calibrated_pairs)
    by_rule = defaultdict(list)
    calibrated_rule_pairs = []
    for rule, probability, actual in test_rule_pairs:
        by_rule[rule].append({"predicted": probability, "actual": actual})
        calibrated_rule_pairs.append({"predicted": calibrator.transform_probability(rule, probability), "actual": actual})

    payload = {
        "metadata": {
            "train_seeds": args.train_seeds,
            "test_seeds": args.test_seeds,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "prediction_stage": "post_filter_executed_action",
            "checkpoint_root": str(args.proposed_root),
        },
        "train_raw": train_summary,
        "test_raw": test_summary,
        "test_calibrated": calibrated_summary,
        "rule_test_raw": {rule: _brier_ece(values) for rule, values in sorted(by_rule.items())},
        "rule_test_calibrated": _brier_ece(calibrated_rule_pairs),
        "calibrator_mapping": calibrator.mapping,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "train_pairs.json").write_text(json.dumps(train_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "test_pairs.json").write_text(json.dumps(test_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "calibrated_test_pairs.json").write_text(json.dumps(calibrated_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "train_pairs": len(train_pairs), "test_pairs": len(test_pairs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
