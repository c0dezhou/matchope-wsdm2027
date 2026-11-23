#!/usr/bin/env python3
from __future__ import annotations

"""Calibrate matching-value inference on the real KuaiRand evaluation design.

The user, date, shared-context, and category structure is kept fixed.  Logged
categories and binary rewards are regenerated from a known full-support logger.
The outcome mean combines additive user and context-category terms with a
policy-aligned interaction whose finite-design matching value is known.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from matchope.eval.matching_value import estimate_from_scores
from matchope.pipeline.run_kuairand_matching_gate import (
    _context_action_counts,
    _parse_numbers,
    assign_user_folds,
    calibrate_policy_to_tv,
    cluster_inference,
    cross_fitted_scores,
    fit_frozen_history_policy,
    load_kuairand_category_data,
    two_way_cluster_inference,
)


def build_additive_probability_components(
    standard: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    n_users: int,
    n_actions: int,
    outcome: str,
    logger_probability: np.ndarray | None = None,
    user_effect_bound: float = 0.02,
    context_action_effect_bound: float = 0.025,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Construct bounded empirical additive outcome components."""

    if not 0.0 < user_effect_bound < 0.25:
        raise ValueError("user_effect_bound must lie in (0, 0.25)")
    if not 0.0 < context_action_effect_bound < 0.25:
        raise ValueError("context_action_effect_bound must lie in (0, 0.25)")
    required = {"user_index", "action", outcome}
    if not required.issubset(standard) or not required.issubset(evaluation):
        raise ValueError("standard and evaluation frames miss required columns")

    standard_user = standard["user_index"].to_numpy(dtype=np.int64)
    standard_reward = standard[outcome].to_numpy(dtype=np.float64)
    user_count = np.bincount(standard_user, minlength=n_users).astype(np.float64)
    user_sum = np.bincount(
        standard_user,
        weights=standard_reward,
        minlength=n_users,
    )
    if np.any(user_count == 0.0):
        raise ValueError("every evaluation user must have preperiod outcomes")
    user_rate = user_sum / user_count
    user_scale = float(user_rate.std())
    if user_scale <= 0.0:
        raise ValueError("preperiod user rates have no variation")
    user_raw = np.tanh((user_rate - user_rate.mean()) / user_scale)
    evaluation_user = evaluation["user_index"].to_numpy(dtype=np.int64)
    evaluation_user_count = np.bincount(
        evaluation_user,
        minlength=n_users,
    )
    user_raw -= np.average(user_raw, weights=evaluation_user_count)
    user_effect = user_effect_bound * user_raw / np.max(np.abs(user_raw))

    context = evaluation["context_index"].to_numpy(dtype=np.int64)
    action = evaluation["action"].to_numpy(dtype=np.int64)
    reward = evaluation[outcome].to_numpy(dtype=np.float64)
    n_contexts = int(context.max()) + 1
    flat = context * n_actions + action
    cell_count = np.bincount(
        flat,
        minlength=n_contexts * n_actions,
    ).reshape(n_contexts, n_actions)
    cell_sum = np.bincount(
        flat,
        weights=reward,
        minlength=n_contexts * n_actions,
    ).reshape(n_contexts, n_actions)
    if np.any(cell_count == 0):
        raise ValueError("every context-category cell must be observed")
    cell_rate = cell_sum / cell_count
    cell_scale = float(cell_rate.std())
    if cell_scale <= 0.0:
        raise ValueError("context-category rates have no variation")
    context_action_raw = np.tanh((cell_rate - cell_rate.mean()) / cell_scale)
    if logger_probability is None:
        context_action_weight = cell_count
    else:
        logger = np.asarray(logger_probability, dtype=np.float64)
        if logger.shape != cell_count.shape:
            raise ValueError("logger_probability must align with context cells")
        context_count = np.bincount(context, minlength=n_contexts)
        context_action_weight = context_count[:, None] * logger
    context_action_raw -= np.average(
        context_action_raw,
        weights=context_action_weight,
    )
    context_action_effect = (
        context_action_effect_bound
        * context_action_raw
        / np.max(np.abs(context_action_raw))
    )
    return float(reward.mean()), user_effect, context_action_effect


def policy_aligned_interaction_shape(policy_by_user: np.ndarray) -> np.ndarray:
    """Return a bounded user-action interaction aligned with policy deviations."""

    policy = np.asarray(policy_by_user, dtype=np.float64)
    if policy.ndim != 2 or not np.isfinite(policy).all():
        raise ValueError("policy_by_user must be a finite matrix")
    if np.any(policy <= 0.0) or not np.allclose(
        policy.sum(axis=1),
        1.0,
        atol=1e-10,
    ):
        raise ValueError("policy_by_user must contain positive probabilities")
    return (policy > policy.mean(axis=0, keepdims=True)).astype(np.float64)


def additive_matching_truth(
    policy_by_user: np.ndarray,
    user_indices: np.ndarray,
    contexts: np.ndarray,
    *,
    intercept: float,
    user_effect: np.ndarray,
    context_action_effect: np.ndarray,
    interaction_shape: np.ndarray | None = None,
    interaction_coefficient: float = 0.0,
) -> float:
    """Evaluate finite-design truth for additive plus interaction outcomes."""

    policy = np.asarray(policy_by_user, dtype=np.float64)
    user = np.asarray(user_indices, dtype=np.int64)
    context = np.asarray(contexts, dtype=np.int64)
    if user.shape != context.shape or user.ndim != 1:
        raise ValueError("user_indices and contexts must be aligned vectors")
    if policy.ndim != 2 or user_effect.shape != (policy.shape[0],):
        raise ValueError("policy and user effects must align")
    if context_action_effect.ndim != 2 or context_action_effect.shape[1] != policy.shape[1]:
        raise ValueError("context-action effects must align with policy actions")
    if np.any(user < 0) or np.any(user >= policy.shape[0]):
        raise ValueError("user index is out of bounds")
    if np.any(context < 0) or np.any(context >= context_action_effect.shape[0]):
        raise ValueError("context index is out of bounds")
    if interaction_shape is None:
        interaction = np.zeros_like(policy)
    else:
        interaction = np.asarray(interaction_shape, dtype=np.float64)
        if interaction.shape != policy.shape or not np.isfinite(interaction).all():
            raise ValueError("interaction_shape must align with policy")
    if not np.isfinite(interaction_coefficient):
        raise ValueError("interaction_coefficient must be finite")

    total = 0.0
    for context_index in np.unique(context):
        rows = context == context_index
        row_policy = policy[user[rows]]
        reference = row_policy.mean(axis=0)
        q = (
            intercept
            + user_effect[user[rows], None]
            + context_action_effect[context_index][None, :]
            + interaction_coefficient * interaction[user[rows]]
        )
        total += float(np.sum((row_policy - reference[None, :]) * q))
    return total / user.size


def empirical_context_logger(
    evaluation: pd.DataFrame,
    *,
    n_actions: int,
    base_probability: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    """Freeze a positive context logger from the original randomized log."""

    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive")
    context = evaluation["context_index"].to_numpy(dtype=np.int64)
    action = evaluation["action"].to_numpy(dtype=np.int64)
    n_contexts = int(context.max()) + 1
    counts = _context_action_counts(
        context,
        action,
        n_contexts=n_contexts,
        n_actions=n_actions,
    ).astype(np.float64)
    base = np.asarray(base_probability, dtype=np.float64)
    if base.shape != (n_actions,) or np.any(base <= 0.0):
        raise ValueError("base_probability must be positive and aligned")
    return (counts + smoothing * base[None, :]) / (
        counts.sum(axis=1, keepdims=True) + smoothing
    )


def sample_contextual_actions(
    contexts: np.ndarray,
    logger_probability: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one categorical action per fixed evaluation row."""

    context = np.asarray(contexts, dtype=np.int64)
    logger = np.asarray(logger_probability, dtype=np.float64)
    if context.ndim != 1 or logger.ndim != 2:
        raise ValueError("contexts and logger_probability have invalid shapes")
    actions = np.empty(context.size, dtype=np.int16)
    labels = np.arange(logger.shape[1])
    for context_index in np.unique(context):
        rows = context == context_index
        actions[rows] = rng.choice(
            labels,
            size=int(rows.sum()),
            p=logger[context_index],
        )
    return actions


def semisynthetic_zero_coverage(
    standard: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    policy_by_user: np.ndarray,
    history_signal: np.ndarray,
    base_probability: np.ndarray,
    user_fold: np.ndarray,
    repetitions: int,
    seed: int,
    propensity_smoothing: float,
    outcome_smoothing: float,
    outcome_ridge: float,
    source_outcome: str,
    target_matching_value: float = 0.0,
) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    """Run repeated known-truth calibration on a fixed KuaiRand design."""

    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if target_matching_value < 0.0 or not np.isfinite(target_matching_value):
        raise ValueError("target_matching_value must be finite and non-negative")
    n_users, n_actions = policy_by_user.shape
    logger = empirical_context_logger(
        evaluation,
        n_actions=n_actions,
        base_probability=base_probability,
        smoothing=propensity_smoothing,
    )
    empirical_outcome_mean, user_effect, context_action_effect = (
        build_additive_probability_components(
            standard,
            evaluation,
            n_users=n_users,
            n_actions=n_actions,
            outcome=source_outcome,
            logger_probability=logger,
        )
    )
    user = evaluation["user_index"].to_numpy(dtype=np.int64)
    context = evaluation["context_index"].to_numpy(dtype=np.int64)
    day = evaluation["date"].to_numpy(dtype=np.int64)
    raw_interaction_shape = policy_aligned_interaction_shape(policy_by_user)
    raw_logger_interaction_mean = float(
        np.mean(
            np.sum(
                logger[context] * raw_interaction_shape[user],
                axis=1,
            )
        )
    )
    # Removing a global action-constant from the interaction does not change
    # matching value.  It keeps the empirical source rate as the literal DGP
    # intercept while making the interaction logger-mean-zero.
    interaction_shape = (
        raw_interaction_shape - raw_logger_interaction_mean
    )
    unit_interaction_truth = additive_matching_truth(
        policy_by_user,
        user,
        context,
        intercept=0.0,
        user_effect=np.zeros_like(user_effect),
        context_action_effect=np.zeros_like(context_action_effect),
        interaction_shape=interaction_shape,
        interaction_coefficient=1.0,
    )
    if unit_interaction_truth <= 0.0:
        raise RuntimeError("policy-aligned interaction has non-positive truth")
    interaction_coefficient = target_matching_value / unit_interaction_truth
    logger_interaction_mean = float(
        np.mean(
            np.sum(
                logger[context] * interaction_shape[user],
                axis=1,
            )
        )
    )
    intercept = empirical_outcome_mean
    truth = additive_matching_truth(
        policy_by_user,
        user,
        context,
        intercept=intercept,
        user_effect=user_effect,
        context_action_effect=context_action_effect,
        interaction_shape=interaction_shape,
        interaction_coefficient=interaction_coefficient,
    )
    if abs(truth - target_matching_value) > 1e-12:
        raise RuntimeError("semisynthetic truth does not match its target")
    minimum_probability = 1.0
    maximum_probability = 0.0
    for context_index in np.unique(context):
        rows = context == context_index
        probability = (
            intercept
            + user_effect[user[rows], None]
            + context_action_effect[context_index][None, :]
            + interaction_coefficient * interaction_shape[user[rows]]
        )
        minimum_probability = min(minimum_probability, float(probability.min()))
        maximum_probability = max(maximum_probability, float(probability.max()))
    if minimum_probability <= 0.0 or maximum_probability >= 1.0:
        raise RuntimeError("semisynthetic outcome probability left (0, 1)")
    methods = (
        "centered_iid",
        "centered_user_cluster",
        "centered_day_cluster",
        "centered_two_way_cluster",
        "separate_user_cluster",
        "ipw_user_cluster",
    )
    estimates = {method: [] for method in methods}
    standard_errors = {method: [] for method in methods}
    coverages = {method: [] for method in methods}
    failures = {method: 0 for method in methods}
    maximum_weights: list[float] = []
    rng = np.random.default_rng(seed)
    simulated = evaluation.copy()
    for _ in range(repetitions):
        action = sample_contextual_actions(context, logger, rng=rng)
        probability = (
            intercept
            + user_effect[user]
            + context_action_effect[context, action]
            + interaction_coefficient * interaction_shape[user, action]
        )
        if np.any(probability <= 0.0) or np.any(probability >= 1.0):
            raise RuntimeError("semisynthetic outcome probability left (0, 1)")
        simulated["action"] = action
        simulated["semisynthetic_outcome"] = rng.binomial(1, probability).astype(
            np.int8
        )
        scores = cross_fitted_scores(
            simulated,
            policy_by_user=policy_by_user,
            history_signal=history_signal,
            base_probability=base_probability,
            user_fold=user_fold,
            propensity_smoothing=propensity_smoothing,
            outcome_smoothing=outcome_smoothing,
            outcome_ridge=outcome_ridge,
            outcome="semisynthetic_outcome",
        )
        maximum_weights.append(scores.maximum_logged_contrast_weight)
        score_methods = {
            "centered_user_cluster": scores.centered_scores,
            "centered_day_cluster": scores.centered_scores,
            "centered_two_way_cluster": scores.centered_scores,
            "separate_user_cluster": scores.separate_dr_scores,
            "ipw_user_cluster": scores.ipw_scores,
        }
        iid = estimate_from_scores(scores.centered_scores)
        summaries: dict[str, tuple[float, float, float, float]] = {
            "centered_iid": (
                iid.value,
                iid.standard_error,
                iid.ci_lower,
                iid.ci_upper,
            )
        }
        for method, score in score_methods.items():
            try:
                if method == "centered_day_cluster":
                    summary = cluster_inference(score, day, prefix="interval")
                elif method == "centered_two_way_cluster":
                    summary = two_way_cluster_inference(
                        score,
                        user,
                        day,
                        prefix="interval",
                    )
                else:
                    summary = cluster_inference(score, user, prefix="interval")
                summaries[method] = (
                    float(score.mean()),
                    float(summary["interval_standard_error"]),
                    float(summary["interval_ci_lower"]),
                    float(summary["interval_ci_upper"]),
                )
            except ValueError:
                failures[method] += 1
        for method, (estimate, standard_error, lower, upper) in summaries.items():
            estimates[method].append(estimate)
            standard_errors[method].append(standard_error)
            coverages[method].append(lower <= truth <= upper)

    rows: list[dict[str, float | int | str]] = []
    for method in methods:
        estimate_array = np.asarray(estimates[method], dtype=np.float64)
        valid = int(estimate_array.size)
        coverage = float(np.mean(coverages[method])) if valid else float("nan")
        empirical_sd = (
            float(estimate_array.std(ddof=1)) if valid > 1 else float("nan")
        )
        mean_standard_error = (
            float(np.mean(standard_errors[method])) if valid else float("nan")
        )
        rows.append(
            {
                "method": method,
                "rows": int(len(evaluation)),
                "users": int(np.unique(user).size),
                "days": int(np.unique(day).size),
                "contexts": int(np.unique(context).size),
                "actions": int(n_actions),
                "repetitions": int(repetitions),
                "valid_repetitions": valid,
                "variance_failures": int(failures[method]),
                "true_matching_value": float(truth),
                "target_matching_value": float(target_matching_value),
                "mean_estimate": float(estimate_array.mean()) if valid else float("nan"),
                "bias": float(estimate_array.mean() - truth) if valid else float("nan"),
                "empirical_standard_deviation": empirical_sd,
                "bias_mcse": (
                    empirical_sd / np.sqrt(valid) if valid > 1 else float("nan")
                ),
                "mean_standard_error": mean_standard_error,
                "mean_se_to_empirical_sd": (
                    mean_standard_error / empirical_sd
                    if valid > 1 and empirical_sd > 0.0
                    else float("nan")
                ),
                "coverage_95pct": coverage,
                "coverage_mcse": (
                    float(np.sqrt(coverage * (1.0 - coverage) / valid))
                    if valid
                    else float("nan")
                ),
            }
        )
    dgp = {
        "outcome_intercept": float(intercept),
        "empirical_source_outcome_mean": float(empirical_outcome_mean),
        "minimum_user_effect": float(user_effect.min()),
        "maximum_user_effect": float(user_effect.max()),
        "minimum_context_action_effect": float(context_action_effect.min()),
        "maximum_context_action_effect": float(context_action_effect.max()),
        "interaction_coefficient": float(interaction_coefficient),
        "unit_interaction_matching_value": float(unit_interaction_truth),
        "logger_interaction_mean": logger_interaction_mean,
        "uncentered_logger_interaction_mean": raw_logger_interaction_mean,
        "minimum_outcome_probability": minimum_probability,
        "maximum_outcome_probability": maximum_probability,
        "logger_weighted_outcome_mean": float(empirical_outcome_mean),
        "minimum_logger_probability": float(logger.min()),
        "maximum_logger_probability": float(logger.max()),
        "maximum_logged_contrast_weight_mean": float(np.mean(maximum_weights)),
        "true_matching_value": float(truth),
    }
    return rows, dgp


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--category-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tab", type=int, default=1)
    parser.add_argument("--minimum-history-events", type=int, default=10)
    parser.add_argument("--minimum-items-per-category", type=int, default=30)
    parser.add_argument("--shared-context-hour-width", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-tv", type=float, default=0.04)
    parser.add_argument("--smoothing-grid", default="1,5,20,100")
    parser.add_argument("--validation-days", type=int, default=4)
    parser.add_argument("--propensity-smoothing", type=float, default=1.0)
    parser.add_argument("--outcome-smoothing", type=float, default=20.0)
    parser.add_argument("--outcome-ridge", type=float, default=1.0)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--effect-multiplier-grid", default="0,0.5,1,2")
    parser.add_argument("--reference-matching-effect", type=float, default=0.000966)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    standard, evaluation, mapping, data_audit = load_kuairand_category_data(
        args.data_dir,
        category_metadata_path=args.category_metadata,
        tab=args.tab,
        minimum_history_events=args.minimum_history_events,
        minimum_items_per_category=args.minimum_items_per_category,
        shared_context_hour_width=args.shared_context_hour_width,
    )
    user_ids = np.sort(evaluation["user_id"].unique()).astype(np.int64)
    frozen = fit_frozen_history_policy(
        standard,
        user_ids=user_ids,
        outcome="long_view",
        n_actions=len(mapping.labels),
        smoothing_grid=_parse_numbers(args.smoothing_grid),
        validation_days=args.validation_days,
    )
    beta, policy, achieved_tv = calibrate_policy_to_tv(
        frozen.signal,
        frozen.global_rate,
        mapping.base_probability,
        target_tv=args.target_tv,
    )
    user_fold = assign_user_folds(len(user_ids), folds=args.folds, seed=args.seed)
    rows: list[dict[str, object]] = []
    dgps: list[dict[str, float]] = []
    effect_multipliers = _parse_numbers(args.effect_multiplier_grid)
    if any(multiplier < 0.0 for multiplier in effect_multipliers):
        raise ValueError("effect multipliers must be non-negative")
    for multiplier in effect_multipliers:
        effect_rows, dgp = semisynthetic_zero_coverage(
            standard,
            evaluation,
            policy_by_user=policy,
            history_signal=frozen.signal,
            base_probability=mapping.base_probability,
            user_fold=user_fold,
            repetitions=args.repetitions,
            seed=args.seed + 1,
            propensity_smoothing=args.propensity_smoothing,
            outcome_smoothing=args.outcome_smoothing,
            outcome_ridge=args.outcome_ridge,
            source_outcome="long_view",
            target_matching_value=(
                float(multiplier) * args.reference_matching_effect
            ),
        )
        for row in effect_rows:
            rows.append({"effect_multiplier": float(multiplier), **row})
        dgps.append({"effect_multiplier": float(multiplier), **dgp})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "kuairand_semisynthetic_coverage.csv", rows)
    payload = {
        "protocol": {
            "seed": args.seed,
            "repetitions": args.repetitions,
            "target_tv": args.target_tv,
            "achieved_policy_tv": achieved_tv,
            "policy_inverse_temperature": beta,
            "selected_history_smoothing": frozen.selected_smoothing,
            "folds": args.folds,
            "source_outcome_for_dgp_shape": "long_view",
            "outcome_model": (
                "real-rate additive user plus context-category probability, "
                "with a known policy-aligned interaction"
            ),
            "evaluation_outcomes_used_for_dgp_shape": True,
            "evaluation_outcomes_used_for_policy_or_target_selection": False,
            "reference_matching_effect": args.reference_matching_effect,
            "effect_multipliers": list(effect_multipliers),
        },
        "design": {
            "evaluation_rows": data_audit["evaluation_rows"],
            "evaluation_users": data_audit["evaluation_users"],
            "evaluation_dates": data_audit["evaluation_dates"],
            "evaluation_shared_contexts": data_audit[
                "evaluation_shared_contexts"
            ],
            "category_count": data_audit["category_count"],
        },
        "dgp": dgps,
        "coverage": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
