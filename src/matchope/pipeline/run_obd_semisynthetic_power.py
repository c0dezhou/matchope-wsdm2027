#!/usr/bin/env python3
from __future__ import annotations

"""Calibrate seven-day OBD resolution with a known matching effect.

The simulation keeps the released OBD rows, day labels, 34-action uniform
random logger, and the exact endpoint affinity policy.  It replaces clicks by
Bernoulli draws from an outcome surface whose matching value is fixed by
construction.  The target policy never reads an evaluation outcome.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from matchope.eval.matching_value import estimate_from_scores
from matchope.pipeline.run_obd_matching_gate import (
    affinity_limit_policy,
    cluster_summary,
    load_random_men,
)
from matchope.pipeline.run_kuairand_matching_gate import (
    enumerated_cluster_sign_flip_interval,
)


def _parse_effect_grid(raw: str) -> tuple[float, ...]:
    effects = tuple(float(token.strip()) for token in raw.split(",") if token.strip())
    if not effects or any(effect < 0.0 for effect in effects):
        raise ValueError("effect grid must contain non-negative values")
    if len(set(effects)) != len(effects):
        raise ValueError("effect grid must not contain duplicates")
    return effects


def _day_codes(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(days)
    if labels.ndim != 1:
        raise ValueError("days must be one-dimensional")
    unique, codes = np.unique(labels, return_inverse=True)
    if unique.size < 2:
        raise ValueError("the power calibration requires at least two days")
    return unique, codes.astype(np.int64, copy=False)


def leave_one_day_policy_marginals(
    policy: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """Return one policy marginal per held-out day."""

    target = np.asarray(policy, dtype=np.float64)
    if target.ndim != 2 or not np.isfinite(target).all():
        raise ValueError("policy must be a finite row-by-action matrix")
    _, codes = _day_codes(days)
    if codes.shape != (target.shape[0],):
        raise ValueError("days must align with policy rows")
    total = target.sum(axis=0)
    marginals = np.empty((int(codes.max()) + 1, target.shape[1]), dtype=np.float64)
    for day_code in range(marginals.shape[0]):
        held_out = codes == day_code
        training_rows = int((~held_out).sum())
        if training_rows == 0:
            raise ValueError("every held-out day needs training rows")
        marginals[day_code] = (total - target[held_out].sum(axis=0)) / training_rows
    return marginals


def cross_fitted_logged_action_means(
    *,
    actions: np.ndarray,
    rewards: np.ndarray,
    days: np.ndarray,
    n_actions: int,
    smoothing: float,
) -> np.ndarray:
    """Estimate action means outside each day and align them to logged actions."""

    action = np.asarray(actions, dtype=np.int64)
    outcome = np.asarray(rewards, dtype=np.float64)
    _, codes = _day_codes(days)
    if action.shape != outcome.shape or action.shape != codes.shape:
        raise ValueError("actions, rewards, and days must align")
    if n_actions < 2 or np.any(action < 0) or np.any(action >= n_actions):
        raise ValueError("actions must lie in the declared action space")
    if not np.isfinite(outcome).all():
        raise ValueError("rewards must be finite")
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive")

    aligned = np.empty(action.size, dtype=np.float64)
    for day_code in range(int(codes.max()) + 1):
        held_out = codes == day_code
        training = ~held_out
        counts = np.bincount(action[training], minlength=n_actions).astype(np.float64)
        reward_sums = np.bincount(
            action[training],
            weights=outcome[training],
            minlength=n_actions,
        ).astype(np.float64)
        training_mean = float(outcome[training].mean())
        action_means = (reward_sums + smoothing * training_mean) / (
            counts + smoothing
        )
        aligned[held_out] = action_means[action[held_out]]
    return aligned


def semisynthetic_outcome_parameters(
    *,
    policy: np.ndarray,
    days: np.ndarray,
    observed_rewards: np.ndarray,
    target_matching_value: float,
) -> dict[str, object]:
    """Build an action-invariant day baseline plus a calibrated interaction.

    The outcome surface is ``q_ia = b_day(i) + gamma * policy_ia``.  With the
    pooled policy marginal as reference, its matching value is exactly
    ``gamma * mean(sum((policy - marginal) * policy, axis=1))``.  The day
    baseline cancels from the contrast.
    """

    target = np.asarray(policy, dtype=np.float64)
    rewards = np.asarray(observed_rewards, dtype=np.float64)
    unique_days, codes = _day_codes(days)
    if target.ndim != 2 or rewards.shape != (target.shape[0],):
        raise ValueError("policy, rewards, and days must align")
    if not np.isfinite(target).all() or not np.isfinite(rewards).all():
        raise ValueError("policy and rewards must be finite")
    if target_matching_value < 0.0 or not np.isfinite(target_matching_value):
        raise ValueError("target matching value must be finite and non-negative")

    marginal = target.mean(axis=0)
    coefficient = float(np.mean(np.sum((target - marginal) * target, axis=1)))
    if coefficient <= 0.0:
        raise ValueError("the target policy has no user-specific variation")
    interaction_scale = float(target_matching_value / coefficient)
    day_baselines = np.asarray(
        [rewards[codes == day_code].mean() for day_code in range(unique_days.size)],
        dtype=np.float64,
    )
    minimum_probability = float(day_baselines.min())
    maximum_probability = float(
        day_baselines.max() + interaction_scale * target.max()
    )
    if minimum_probability < 0.0 or maximum_probability > 1.0:
        raise ValueError("requested effect produces an invalid Bernoulli probability")
    return {
        "target_matching_value": float(target_matching_value),
        "matching_coefficient": coefficient,
        "interaction_scale": interaction_scale,
        "day_baselines": day_baselines,
        "day_codes": codes,
        "minimum_outcome_probability": minimum_probability,
        "maximum_outcome_probability": maximum_probability,
        "mean_logger_outcome_probability": float(
            day_baselines[codes].mean() + interaction_scale / target.shape[1]
        ),
    }


def evaluate_semisynthetic_power(
    feedback: dict[str, np.ndarray],
    *,
    target_matching_values: Sequence[float],
    repetitions: int,
    seed: int,
    outcome_smoothing: float = 20.0,
) -> tuple[list[dict[str, float | int]], dict[str, object]]:
    """Estimate row-i.i.d. and day-cluster power at known matching values."""

    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if outcome_smoothing <= 0.0:
        raise ValueError("outcome_smoothing must be positive")
    effects = tuple(float(value) for value in target_matching_values)
    if not effects or any(value < 0.0 or not np.isfinite(value) for value in effects):
        raise ValueError("target matching values must be finite and non-negative")
    if 0.0 not in effects:
        raise ValueError(
            "the effect grid must include zero for held-out null calibration"
        )

    affinity = np.asarray(feedback["affinity"], dtype=np.float64)
    observed_rewards = np.asarray(feedback["rewards"], dtype=np.float64)
    days = np.asarray(feedback["days"])
    pscore = np.asarray(feedback["pscore"], dtype=np.float64)
    policy = affinity_limit_policy(affinity)
    n_rows, n_actions = policy.shape
    if observed_rewards.shape != days.shape or days.shape != pscore.shape:
        raise ValueError("feedback arrays must align")
    logging_probability = 1.0 / n_actions
    if not np.allclose(pscore, logging_probability, atol=1e-12, rtol=1e-12):
        raise ValueError("the calibration requires OBD's uniform random logger")

    unique_days, codes = _day_codes(days)
    held_out_marginals = leave_one_day_policy_marginals(policy, days)
    parameter_rows = [
        semisynthetic_outcome_parameters(
            policy=policy,
            days=days,
            observed_rewards=observed_rewards,
            target_matching_value=effect,
        )
        for effect in effects
    ]
    rng = np.random.default_rng(seed)
    records: list[dict[str, list[float] | int | float]] = []
    for effect, parameters in zip(effects, parameter_rows, strict=True):
        records.append(
            {
                "target_matching_value": effect,
                "interaction_scale": float(parameters["interaction_scale"]),
                "minimum_outcome_probability": float(
                    parameters["minimum_outcome_probability"]
                ),
                "maximum_outcome_probability": float(
                    parameters["maximum_outcome_probability"]
                ),
                "mean_logger_outcome_probability": float(
                    parameters["mean_logger_outcome_probability"]
                ),
                "estimates": [],
                "iid_standard_errors": [],
                "day_standard_errors": [],
                "day_mdes": [],
                "iid_positive_detections": 0,
                "day_positive_detections": 0,
                "day_negative_detections": 0,
                "day_coverages": 0,
                "day_sign_flip_coverages": 0,
            }
        )

    row_index = np.arange(n_rows)
    day_baselines = np.asarray(parameter_rows[0]["day_baselines"], dtype=np.float64)
    for _ in range(repetitions):
        actions = rng.integers(0, n_actions, size=n_rows)
        outcome_uniforms = rng.random(n_rows)
        selected_policy = policy[row_index, actions]
        selected_reference = held_out_marginals[codes, actions]
        contrast_weight = (selected_policy - selected_reference) / logging_probability
        for record in records:
            probabilities = day_baselines[codes] + float(
                record["interaction_scale"]
            ) * selected_policy
            rewards = (outcome_uniforms < probabilities).astype(np.float64)
            action_means = cross_fitted_logged_action_means(
                actions=actions,
                rewards=rewards,
                days=days,
                n_actions=n_actions,
                smoothing=outcome_smoothing,
            )
            scores = contrast_weight * (rewards - action_means)
            iid = estimate_from_scores(scores)
            day = cluster_summary(scores, days)
            day_sign_flip = enumerated_cluster_sign_flip_interval(
                scores,
                days,
                prefix="day",
            )
            estimate = iid.value
            lower = float(day["day_cluster_ci_lower"])
            upper = float(day["day_cluster_ci_upper"])
            target_value = float(record["target_matching_value"])
            cast_estimates = record["estimates"]
            cast_iid_se = record["iid_standard_errors"]
            cast_day_se = record["day_standard_errors"]
            cast_day_mde = record["day_mdes"]
            assert isinstance(cast_estimates, list)
            assert isinstance(cast_iid_se, list)
            assert isinstance(cast_day_se, list)
            assert isinstance(cast_day_mde, list)
            cast_estimates.append(estimate)
            cast_iid_se.append(iid.standard_error)
            cast_day_se.append(float(day["day_cluster_standard_error"]))
            cast_day_mde.append(float(day["day_cluster_t_mde_80pct"]))
            record["iid_positive_detections"] = int(
                record["iid_positive_detections"]
            ) + int(iid.ci_lower > 0.0)
            record["day_positive_detections"] = int(
                record["day_positive_detections"]
            ) + int(lower > 0.0)
            record["day_negative_detections"] = int(
                record["day_negative_detections"]
            ) + int(upper < 0.0)
            record["day_coverages"] = int(record["day_coverages"]) + int(
                lower <= target_value <= upper
            )
            record["day_sign_flip_coverages"] = int(
                record["day_sign_flip_coverages"]
            ) + int(
                float(day_sign_flip["day_sign_flip_ci_lower"])
                <= target_value
                <= float(day_sign_flip["day_sign_flip_ci_upper"])
            )

    rows: list[dict[str, float | int]] = []
    estimate_arrays = {
        float(record["target_matching_value"]): np.asarray(
            record["estimates"], dtype=np.float64
        )
        for record in records
    }
    calibration_repetitions = repetitions // 2
    evaluation_repetitions = repetitions - calibration_repetitions
    null_critical_value = float(
        np.quantile(
            estimate_arrays[0.0][:calibration_repetitions],
            0.95,
            method="higher",
        )
    )
    for record in records:
        estimates = np.asarray(record.pop("estimates"), dtype=np.float64)
        iid_standard_errors = np.asarray(
            record.pop("iid_standard_errors"), dtype=np.float64
        )
        day_standard_errors = np.asarray(
            record.pop("day_standard_errors"), dtype=np.float64
        )
        day_mdes = np.asarray(record.pop("day_mdes"), dtype=np.float64)
        target_value = float(record["target_matching_value"])
        iid_detections = int(record.pop("iid_positive_detections"))
        day_detections = int(record.pop("day_positive_detections"))
        day_negative_detections = int(record.pop("day_negative_detections"))
        day_coverages = int(record.pop("day_coverages"))
        day_sign_flip_coverages = int(
            record.pop("day_sign_flip_coverages")
        )
        rows.append(
            {
                "target_matching_value": target_value,
                "interaction_scale": float(record["interaction_scale"]),
                "minimum_outcome_probability": float(
                    record["minimum_outcome_probability"]
                ),
                "maximum_outcome_probability": float(
                    record["maximum_outcome_probability"]
                ),
                "mean_logger_outcome_probability": float(
                    record["mean_logger_outcome_probability"]
                ),
                "repetitions": repetitions,
                "mean_estimate": float(estimates.mean()),
                "bias": float(estimates.mean() - target_value),
                "empirical_standard_deviation": float(estimates.std(ddof=1)),
                "mean_iid_standard_error": float(iid_standard_errors.mean()),
                "mean_day_cluster_standard_error": float(day_standard_errors.mean()),
                "mean_day_cluster_mde_80pct": float(day_mdes.mean()),
                "iid_positive_detection_rate": iid_detections / repetitions,
                "day_positive_detection_rate": day_detections / repetitions,
                "day_negative_detection_rate": (
                    day_negative_detections / repetitions
                ),
                "day_interval_coverage": day_coverages / repetitions,
                "day_sign_flip_sensitivity_coverage": (
                    day_sign_flip_coverages / repetitions
                ),
                "day_sign_flip_draws": 128,
                "heldout_null_critical_value": null_critical_value,
                "heldout_null_calibrated_positive_detection_rate": float(
                    np.mean(
                        estimates[calibration_repetitions:] > null_critical_value
                    )
                ),
            }
        )
    protocol: dict[str, object] = {
        "rows": n_rows,
        "actions": n_actions,
        "days": int(unique_days.size),
        "policy": "uniform over maximum-affinity actions",
        "logger": (
            "resampled uniform random action with probability "
            f"1/{n_actions}"
        ),
        "outcome_surface": "q_ia = observed_day_ctr_i + gamma * policy_ia",
        "target_policy_training": "none; fixed public affinity transform",
        "common_random_numbers": (
            "actions and Bernoulli uniforms shared across the effect grid "
            "within each repetition"
        ),
        "reference": (
            "pooled endpoint-policy marginal; estimated leave-one-day-out "
            "in each repetition"
        ),
        "outcome_nuisance": "smoothed action mean estimated leave-one-day-out",
        "interval": "95% cluster-t interval over seven released days",
        "repetitions": repetitions,
        "null_calibration_repetitions": calibration_repetitions,
        "heldout_power_repetitions": evaluation_repetitions,
        "heldout_null_critical_value": null_critical_value,
        "seed": seed,
        "outcome_smoothing": outcome_smoothing,
        "matching_coefficient": float(parameter_rows[0]["matching_coefficient"]),
        "effect_grid": list(effects),
    }
    return rows, protocol


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--position", type=int, default=1)
    parser.add_argument(
        "--effect-grid",
        default="0,0.003,0.006,0.009,0.012",
    )
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--outcome-smoothing", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    feedback = load_random_men(args.csv, position=args.position)
    rows, protocol = evaluate_semisynthetic_power(
        feedback,
        target_matching_values=_parse_effect_grid(args.effect_grid),
        repetitions=args.repetitions,
        seed=args.seed,
        outcome_smoothing=args.outcome_smoothing,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    payload = {**protocol, "results": rows}
    text = json.dumps(payload, indent=2)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
