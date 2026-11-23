#!/usr/bin/env python3
from __future__ import annotations

"""Run a minimal matching-value gate on randomized Open Bandit data.

The target policy is a transparent softmax transformation of OBD's 34
user--item affinity fields.  Day-wise cross-fitting estimates the policy
marginal and the action--position outcome marginal. No temperature is selected
on the evaluation outcome; the requested grid is reported as a sensitivity
curve. A dense fixed path and the exact infinite-strength endpoint audit the
tested score path. Both row-i.i.d. and seven-day cluster uncertainty are
shown because the former can conceal temporal concentration of a small effect.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t

from matchope.eval.matching_value import (
    centered_matching_dr_scores,
    estimate_from_scores,
)


AFFINITY_COLUMNS = tuple(f"user-item_affinity_{index}" for index in range(34))


def affinity_policy(affinity: np.ndarray, *, inverse_temperature: float) -> np.ndarray:
    """Convert OBD affinity counts into a stochastic policy."""

    values = np.asarray(affinity, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("affinity must be a finite row-by-action matrix")
    if inverse_temperature < 0.0 or not np.isfinite(inverse_temperature):
        raise ValueError("inverse_temperature must be finite and non-negative")
    logits = inverse_temperature * values
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def affinity_limit_policy(affinity: np.ndarray) -> np.ndarray:
    """Return the exact infinite-strength limit of the affinity softmax."""

    values = np.asarray(affinity, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("affinity must be a finite row-by-action matrix")
    maximizers = values == values.max(axis=1, keepdims=True)
    return maximizers / maximizers.sum(axis=1, keepdims=True)


def day_cross_fitted_marginals(
    *,
    policy: np.ndarray,
    actions: np.ndarray,
    positions: np.ndarray,
    rewards: np.ndarray,
    days: np.ndarray,
    outcome_smoothing: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate policy and outcome marginals outside each held-out day."""

    target = np.asarray(policy, dtype=np.float64)
    action = np.asarray(actions, dtype=np.int64)
    position = np.asarray(positions, dtype=np.int64)
    outcome = np.asarray(rewards, dtype=np.float64)
    day = np.asarray(days)
    if (
        target.ndim != 2
        or min(target.shape) < 1
        or not np.isfinite(target).all()
        or np.any(target < 0.0)
        or not np.allclose(target.sum(axis=1), 1.0)
    ):
        raise ValueError("policy must be a finite row-stochastic matrix")
    n_rows, n_actions = target.shape
    if not (
        action.shape == position.shape == outcome.shape == day.shape == (n_rows,)
    ):
        raise ValueError("row-level arrays must align with policy")
    if outcome_smoothing <= 0.0:
        raise ValueError("outcome_smoothing must be positive")
    marginal_policy = np.empty_like(target)
    marginal_outcome = np.empty_like(target)
    maximum_marginal_shift = 0.0
    unique_days = np.unique(day)
    unique_positions = np.unique(position)
    if unique_days.size < 2:
        raise ValueError("day cross-fitting requires at least two days")
    for held_out_day in unique_days:
        held_out = day == held_out_day
        training = ~held_out
        for slot in unique_positions:
            train_slot = training & (position == slot)
            held_out_slot = held_out & (position == slot)
            if not train_slot.any() or not held_out_slot.any():
                raise ValueError("every day and position must occur in train and holdout")
            policy_mean = target[train_slot].mean(axis=0)
            marginal_policy[held_out_slot] = policy_mean
            held_out_mean = target[held_out_slot].mean(axis=0)
            maximum_marginal_shift = max(
                maximum_marginal_shift,
                float(np.max(np.abs(held_out_mean - policy_mean))),
            )
            counts = np.bincount(action[train_slot], minlength=n_actions).astype(float)
            reward_sums = np.bincount(
                action[train_slot],
                weights=outcome[train_slot],
                minlength=n_actions,
            ).astype(float)
            slot_mean = float(outcome[train_slot].mean())
            q_mean = (
                reward_sums + outcome_smoothing * slot_mean
            ) / (counts + outcome_smoothing)
            marginal_outcome[held_out_slot] = q_mean
    return marginal_policy, marginal_outcome, maximum_marginal_shift


def cluster_summary(
    scores: np.ndarray,
    clusters: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Cluster-robust uncertainty for a sample mean."""

    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(clusters)
    if values.ndim != 1 or labels.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("scores and clusters must be aligned finite vectors")
    unique, codes = np.unique(labels, return_inverse=True)
    n_clusters = unique.size
    if n_clusters < 2:
        raise ValueError("cluster inference requires at least two clusters")
    estimate = float(values.mean())
    centered_cluster_sums = np.bincount(
        codes,
        weights=values - estimate,
        minlength=n_clusters,
    )
    standard_error = float(
        np.sqrt(
            n_clusters
            / (n_clusters - 1)
            * np.sum(np.square(centered_cluster_sums))
        )
        / values.size
    )
    critical = float(t.ppf(1.0 - alpha / 2.0, df=n_clusters - 1))
    target_power = 0.80
    power_quantile = float(t.ppf(target_power, df=n_clusters - 1))
    cluster_means = np.asarray(
        [values[codes == code].mean() for code in range(n_clusters)]
    )
    return {
        "day_clusters": int(n_clusters),
        "day_cluster_standard_error": standard_error,
        "day_cluster_ci_lower": estimate - critical * standard_error,
        "day_cluster_ci_upper": estimate + critical * standard_error,
        "day_cluster_t_mde_80pct": (
            (critical + power_quantile) * standard_error
        ),
        "positive_day_count": int(np.sum(cluster_means > 0.0)),
        "minimum_day_estimate": float(cluster_means.min()),
        "maximum_day_estimate": float(cluster_means.max()),
    }


def reference_context_ipw_sensitivity(
    *,
    policy: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    logging_pscore: np.ndarray,
    days: np.ndarray,
) -> dict[str, float]:
    """Compare fixed-window and date-specific empirical references by IPW."""

    target = np.asarray(policy, dtype=np.float64)
    action = np.asarray(actions, dtype=np.int64)
    outcome = np.asarray(rewards, dtype=np.float64)
    pscore = np.asarray(logging_pscore, dtype=np.float64)
    day = np.asarray(days)
    n_rows, n_actions = target.shape
    if not (
        action.shape == outcome.shape == pscore.shape == day.shape == (n_rows,)
    ):
        raise ValueError("row-level arrays must align with policy")
    if np.any(action < 0) or np.any(action >= n_actions):
        raise ValueError("actions contain an invalid action")
    if not np.isfinite(outcome).all():
        raise ValueError("rewards must be finite")
    if np.any(pscore <= 0.0) or not np.isfinite(pscore).all():
        raise ValueError("logging_pscore must be finite and positive")

    fixed_reference = np.broadcast_to(target.mean(axis=0), target.shape)
    date_reference = np.empty_like(target)
    for label in np.unique(day):
        mask = day == label
        date_reference[mask] = target[mask].mean(axis=0)

    row = np.arange(n_rows)

    def summarize(reference: np.ndarray) -> tuple[float, float]:
        contrast = target - reference
        scores = contrast[row, action] / pscore * outcome
        mean_tv = float(np.mean(0.5 * np.abs(contrast).sum(axis=1)))
        return float(scores.mean()), mean_tv

    fixed_value, fixed_tv = summarize(fixed_reference)
    date_value, date_tv = summarize(date_reference)
    return {
        "fixed_window_reference_ipw_matching_value": fixed_value,
        "date_reference_ipw_matching_value": date_value,
        "date_minus_fixed_window_ipw_matching_value": date_value - fixed_value,
        "fixed_window_reference_mean_tv": fixed_tv,
        "date_reference_mean_tv": date_tv,
    }


def load_random_men(
    csv_path: Path,
    *,
    position: int,
) -> dict[str, np.ndarray]:
    """Load one position as a deployable single-slot allocation problem."""

    if position not in {1, 2, 3}:
        raise ValueError("position must be one of 1, 2, or 3")

    columns = [
        "timestamp",
        "item_id",
        "position",
        "click",
        "propensity_score",
        *AFFINITY_COLUMNS,
    ]
    frame = pd.read_csv(
        csv_path,
        usecols=columns,
        dtype={
            "item_id": np.int16,
            "position": np.int8,
            "click": np.float32,
            "propensity_score": np.float64,
            **{column: np.float32 for column in AFFINITY_COLUMNS},
        },
    )
    frame = frame.loc[frame["position"] == position].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no rows found for position={position}")
    return {
        "actions": frame["item_id"].to_numpy(dtype=np.int64),
        "positions": frame["position"].to_numpy(dtype=np.int64) - 1,
        "rewards": frame["click"].to_numpy(dtype=np.float64),
        "pscore": frame["propensity_score"].to_numpy(dtype=np.float64),
        "days": frame["timestamp"].str.slice(0, 10).to_numpy(),
        "affinity": frame[list(AFFINITY_COLUMNS)].to_numpy(dtype=np.float64),
    }


def _evaluate_affinity_policy(
    feedback: dict[str, np.ndarray],
    *,
    policy: np.ndarray,
    outcome_smoothing: float,
) -> tuple[dict[str, float | int], np.ndarray]:
    actions = feedback["actions"]
    positions = feedback["positions"]
    rewards = feedback["rewards"]
    pscore = feedback["pscore"]
    days = feedback["days"]
    affinity = feedback["affinity"]
    marginal_policy, marginal_outcome, maximum_shift = day_cross_fitted_marginals(
        policy=policy,
        actions=actions,
        positions=positions,
        rewards=rewards,
        days=days,
        outcome_smoothing=outcome_smoothing,
    )
    scores = centered_matching_dr_scores(
        policy,
        marginal_policy,
        actions,
        rewards,
        pscore,
        marginal_outcome,
        marginal_outcome,
    )
    estimate = estimate_from_scores(scores)
    any_affinity = np.any(affinity != 0.0, axis=1)
    return (
        {
            "rows": int(actions.size),
            "clicks": int(rewards.sum()),
            "ctr": float(rewards.mean()),
            "rows_with_any_affinity": int(any_affinity.sum()),
            "affinity_row_fraction": float(any_affinity.mean()),
            "mean_policy_to_marginal_tv": float(
                np.mean(0.5 * np.abs(policy - marginal_policy).sum(axis=1))
            ),
            "maximum_cross_day_action_marginal_shift": maximum_shift,
            "matching_value": estimate.value,
            "iid_standard_error": estimate.standard_error,
            "iid_ci_lower": estimate.ci_lower,
            "iid_ci_upper": estimate.ci_upper,
            "iid_mde_80pct": estimate.minimum_detectable_effect,
            **reference_context_ipw_sensitivity(
                policy=policy,
                actions=actions,
                rewards=rewards,
                logging_pscore=pscore,
                days=days,
            ),
            **cluster_summary(scores, days),
        },
        scores,
    )


def evaluate_affinity_grid(
    feedback: dict[str, np.ndarray],
    *,
    inverse_temperatures: Sequence[float],
    outcome_smoothing: float = 20.0,
) -> list[dict[str, float | int]]:
    """Evaluate a prespecified affinity-policy sensitivity grid."""

    affinity = feedback["affinity"]
    rows: list[dict[str, float | int]] = []
    for inverse_temperature in inverse_temperatures:
        policy = affinity_policy(
            affinity,
            inverse_temperature=float(inverse_temperature),
        )
        evaluation, _ = _evaluate_affinity_policy(
            feedback,
            policy=policy,
            outcome_smoothing=outcome_smoothing,
        )
        row = {
            "inverse_temperature": float(inverse_temperature),
            **evaluation,
        }
        rows.append(row)
    return rows


def evaluate_affinity_path(
    feedback: dict[str, np.ndarray],
    *,
    inverse_temperatures: Sequence[float],
    outcome_smoothing: float = 20.0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate a dense finite path and its exact analytic endpoint."""

    affinity = feedback["affinity"]
    rows: list[dict[str, object]] = []
    for inverse_temperature in inverse_temperatures:
        strength = float(inverse_temperature)
        policy = affinity_policy(affinity, inverse_temperature=strength)
        evaluation, _ = _evaluate_affinity_policy(
            feedback,
            policy=policy,
            outcome_smoothing=outcome_smoothing,
        )
        rows.append(
            {
                "policy_strength": format(strength, ".12g"),
                "inverse_temperature": strength,
                "analytic_endpoint": False,
                **evaluation,
            }
        )

    endpoint_evaluation, _ = _evaluate_affinity_policy(
        feedback,
        policy=affinity_limit_policy(affinity),
        outcome_smoothing=outcome_smoothing,
    )
    rows.append(
        {
            "policy_strength": "infinity",
            "inverse_temperature": None,
            "analytic_endpoint": True,
            **endpoint_evaluation,
        }
    )
    observed_t = np.asarray(
        [
            float(row["matching_value"])
            / float(row["day_cluster_standard_error"])
            for row in rows
        ]
    )
    day_critical = float(t.ppf(0.975, df=int(rows[0]["day_clusters"]) - 1))
    for index, row in enumerate(rows):
        row.update(
            {
                "absolute_day_t_statistic": float(abs(observed_t[index])),
                "pointwise_day_interval_excludes_zero": (
                    float(row["day_cluster_ci_lower"]) > 0.0
                    or float(row["day_cluster_ci_upper"]) < 0.0
                ),
            }
        )
    maximum_index = int(np.argmax(np.abs(observed_t)))
    path_tvs = np.asarray(
        [float(row["mean_policy_to_marginal_tv"]) for row in rows]
    )
    maximum_tv_index = int(np.argmax(path_tvs))
    summary: dict[str, object] = {
        "policies": len(rows),
        "finite_path_points": len(rows) - 1,
        "analytic_endpoint_included": True,
        "day_t_critical_95": day_critical,
        "maximum_absolute_day_t_statistic": float(
            abs(observed_t[maximum_index])
        ),
        "maximum_absolute_day_t_policy_strength": rows[maximum_index][
            "policy_strength"
        ],
        "maximum_path_tv": float(path_tvs[maximum_tv_index]),
        "maximum_path_tv_policy_strength": rows[maximum_tv_index][
            "policy_strength"
        ],
        "analytic_endpoint_tv": float(path_tvs[-1]),
        "any_pointwise_day_interval_excludes_zero": any(
            bool(row["pointwise_day_interval_excludes_zero"]) for row in rows
        ),
    }
    return rows, summary


def _parse_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        raise ValueError("inverse-temperature grid must not be empty")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-path-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--position", type=int, default=1)
    parser.add_argument("--inverse-temperature-grid", default="0.25,0.5,1.0,2.0")
    parser.add_argument("--path-log10-min", type=float, default=-3.0)
    parser.add_argument("--path-log10-max", type=float, default=3.0)
    parser.add_argument("--path-log10-points", type=int, default=61)
    parser.add_argument("--outcome-smoothing", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    feedback = load_random_men(args.csv, position=args.position)
    main_grid = _parse_grid(args.inverse_temperature_grid)
    rows = evaluate_affinity_grid(
        feedback,
        inverse_temperatures=main_grid,
        outcome_smoothing=args.outcome_smoothing,
    )
    if args.path_log10_points < 2 or args.path_log10_min >= args.path_log10_max:
        raise ValueError("path log grid requires at least two ordered points")
    path_grid = tuple(
        sorted(
            {
                *(
                    float(value)
                    for value in np.logspace(
                        args.path_log10_min,
                        args.path_log10_max,
                        args.path_log10_points,
                    )
                ),
                *(float(value) for value in main_grid),
            }
        )
    )
    path_rows, path_summary = evaluate_affinity_path(
        feedback,
        inverse_temperatures=path_grid,
        outcome_smoothing=args.outcome_smoothing,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_path_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(path_rows[0]))
        writer.writeheader()
        writer.writerows(path_rows)
    payload = {
        "policy": "softmax(inverse_temperature * user_item_affinity)",
        "position": args.position,
        "evaluation_unit": "single-slot contextual allocation",
        "temperature_selection": "none; prespecified sensitivity grid",
        "analytic_endpoint": (
            "uniform over all maximum-affinity actions in each row"
        ),
        "path_audit": (
            "fixed dense log-temperature path plus the analytic endpoint; "
            "every point is reported with its pointwise day-cluster interval"
        ),
        "shared_context": (
            "position in the primary fixed-window estimand; constant after "
            "restricting the analysis to position 1"
        ),
        "reference_context_sensitivity": (
            "known-propensity IPW comparison of the fixed-window reference "
            "with a date-specific reference"
        ),
        "day_cluster_mde": (
            "approximate two-sided 80% MDE using t quantiles with day-cluster "
            "degrees of freedom"
        ),
        "path_grid": {
            "log10_min": args.path_log10_min,
            "log10_max": args.path_log10_max,
            "log10_points": args.path_log10_points,
            "finite_policy_count_after_displayed_grid_union": len(path_grid),
        },
        "nuisance_protocol": "leave-one-day-out by position",
        "rows": rows,
        "endpoint_row": path_rows[-1],
        "path_summary": path_summary,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
