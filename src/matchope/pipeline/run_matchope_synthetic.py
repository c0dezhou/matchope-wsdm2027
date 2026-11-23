#!/usr/bin/env python3
from __future__ import annotations

"""Run mechanism and detectability checks for matching-value evaluation."""

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from matchope.eval.matching_value import (
    centered_matching_dr_scores,
    contrast_dr_scores,
    contrast_identification_interval,
    contrast_support_certificate,
    equality_restricted_matching_dr_scores,
    equality_restricted_projection_coefficients,
    estimate_from_scores,
    marginalize_policy,
    matching_value,
    separate_value_difference_interval,
    support_adapted_matching_dr_scores,
)
from matchope.pipeline.run_kuairand_matching_gate import (
    cluster_inference,
    enumerated_cluster_sign_flip_interval,
    two_way_cluster_inference,
)


def _softmax_last_axis(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def mechanism_scenarios() -> list[dict[str, float | str | bool | None]]:
    """Return deterministic examples for the paper's main logical cases."""

    rows: list[dict[str, float | str | bool | None]] = []

    additive_policy = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.2, 0.2],
        ]
    )
    additive_q = (
        np.asarray([0.2, -0.1, 0.4, 0.0])[:, None]
        + np.asarray([0.3, 0.7, -0.2])[None, :]
    )
    rows.append(
        {
            "scenario": "additive_outcome",
            "finding": "interaction absence forces matching value to zero",
            "matching_value": matching_value(
                additive_policy,
                additive_q,
                np.zeros(4, dtype=int),
            ),
            "target_value": float(np.mean(np.sum(additive_policy * additive_q, axis=1))),
            "reference_value": float(
                np.mean(
                    np.sum(
                        marginalize_policy(
                            additive_policy,
                            np.zeros(4, dtype=int),
                        )
                        * additive_q,
                        axis=1,
                    )
                )
            ),
            "point_identified": True,
            "direct_bound_width": 0.0,
            "separate_bound_width": 0.0,
            "actionable_gain": 0.0,
        }
    )

    cancellation_policy = np.asarray(
        [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]]
    )
    cancellation_reference = np.asarray(
        [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]]
    )
    cancellation_logging = np.asarray(
        [[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]]
    )
    cancellation_q = np.asarray([[0.9, 0.7, 0.1], [0.1, 0.2, 0.8]])
    cancellation_observed_q = cancellation_q.copy()
    cancellation_observed_q[cancellation_logging == 0.0] = np.nan
    cancellation_certificate = contrast_support_certificate(
        cancellation_policy,
        cancellation_reference,
        cancellation_logging,
    )
    cancellation_direct = contrast_identification_interval(
        cancellation_policy,
        cancellation_reference,
        cancellation_logging,
        cancellation_observed_q,
        outcome_lower=0.0,
        outcome_upper=1.0,
    )
    cancellation_separate = separate_value_difference_interval(
        cancellation_policy,
        cancellation_reference,
        cancellation_logging,
        cancellation_observed_q,
        outcome_lower=0.0,
        outcome_upper=1.0,
    )
    rows.append(
        {
            "scenario": "unsupported_common_mass_cancels",
            "finding": "contrast is identified although both values are not",
            "matching_value": float(
                np.mean(
                    np.sum(
                        (cancellation_policy - cancellation_reference)
                        * cancellation_q,
                        axis=1,
                    )
                )
            ),
            "target_value": float(
                np.mean(np.sum(cancellation_policy * cancellation_q, axis=1))
            ),
            "reference_value": float(
                np.mean(np.sum(cancellation_reference * cancellation_q, axis=1))
            ),
            "point_identified": cancellation_certificate.point_identified,
            "direct_bound_width": cancellation_direct.width,
            "separate_bound_width": cancellation_separate.width,
            "actionable_gain": None,
        }
    )

    union_policy = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    union_reference = np.full((2, 2), 0.5)
    union_logging = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    union_q = np.asarray([[0.8, 0.1], [0.2, 0.7]])
    union_observed_q = union_q.copy()
    union_observed_q[union_logging == 0.0] = np.nan
    union_certificate = contrast_support_certificate(
        union_policy,
        union_reference,
        union_logging,
    )
    union_direct = contrast_identification_interval(
        union_policy,
        union_reference,
        union_logging,
        union_observed_q,
        outcome_lower=0.0,
        outcome_upper=1.0,
    )
    union_separate = separate_value_difference_interval(
        union_policy,
        union_reference,
        union_logging,
        union_observed_q,
        outcome_lower=0.0,
        outcome_upper=1.0,
    )
    rows.append(
        {
            "scenario": "marginal_reference_breaks_support",
            "finding": "target is supported but the matching contrast is only bounded",
            "matching_value": float(
                np.mean(np.sum((union_policy - union_reference) * union_q, axis=1))
            ),
            "target_value": float(np.mean(np.sum(union_policy * union_q, axis=1))),
            "reference_value": float(
                np.mean(np.sum(union_reference * union_q, axis=1))
            ),
            "point_identified": union_certificate.point_identified,
            "direct_bound_width": union_direct.width,
            "separate_bound_width": union_separate.width,
            "actionable_gain": None,
        }
    )

    actionable_policy = np.asarray([[0.6, 0.4, 0.0], [0.4, 0.6, 0.0]])
    actionable_q = np.asarray([[1.0, 0.0, 0.9], [0.0, 1.0, 0.9]])
    actionable_reference = marginalize_policy(
        actionable_policy,
        np.zeros(2, dtype=int),
    )
    target_value = float(np.mean(np.sum(actionable_policy * actionable_q, axis=1)))
    reference_value = float(
        np.mean(np.sum(actionable_reference * actionable_q, axis=1))
    )
    best_shared_value = float(actionable_q.mean(axis=0).max())
    rows.append(
        {
            "scenario": "positive_matching_negative_actionable",
            "finding": "matching works but loses to the best shared action",
            "matching_value": target_value - reference_value,
            "target_value": target_value,
            "reference_value": reference_value,
            "point_identified": True,
            "direct_bound_width": 0.0,
            "separate_bound_width": 0.0,
            "actionable_gain": target_value - best_shared_value,
        }
    )
    return rows


def detectability_frontier(
    *,
    sample_sizes: Sequence[int],
    interaction_strengths: Sequence[float],
    repetitions: int,
    seed: int,
) -> list[dict[str, float | int]]:
    """Estimate detection probability under controlled interaction strength."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for n_rows in sample_sizes:
        if n_rows < 2:
            raise ValueError("sample sizes must be at least two")
        for gamma in interaction_strengths:
            if gamma < 0.0 or 0.1 + gamma > 1.0:
                raise ValueError("interaction strengths must keep Bernoulli means in [0, 1]")
            estimates: list[float] = []
            standard_errors: list[float] = []
            mdes: list[float] = []
            positive_detections = 0
            for _ in range(repetitions):
                user = rng.integers(0, 2, size=n_rows)
                policy = np.where(
                    user[:, None] == np.arange(2)[None, :],
                    0.8,
                    0.2,
                )
                marginal_policy = np.full((n_rows, 2), 0.5)
                q = 0.1 + gamma * (
                    user[:, None] == np.arange(2)[None, :]
                )
                marginal_q = np.full((n_rows, 2), 0.1 + gamma / 2.0)
                actions = rng.integers(0, 2, size=n_rows)
                rewards = rng.binomial(1, q[np.arange(n_rows), actions])
                scores = centered_matching_dr_scores(
                    policy,
                    marginal_policy,
                    actions,
                    rewards,
                    np.full(n_rows, 0.5),
                    q,
                    marginal_q,
                )
                estimate = estimate_from_scores(scores)
                estimates.append(estimate.value)
                standard_errors.append(estimate.standard_error)
                mdes.append(estimate.minimum_detectable_effect)
                positive_detections += int(estimate.ci_lower > 0.0)
            rows.append(
                {
                    "sample_size": int(n_rows),
                    "interaction_strength": float(gamma),
                    "true_matching_value": float(0.3 * gamma),
                    "mean_estimate": float(np.mean(estimates)),
                    "empirical_standard_deviation": float(np.std(estimates, ddof=1)),
                    "mean_standard_error": float(np.mean(standard_errors)),
                    "mean_mde_80pct": float(np.mean(mdes)),
                    "positive_detection_rate": positive_detections / repetitions,
                    "repetitions": int(repetitions),
                }
            )
    return rows


def support_cancellation_coverage(
    *,
    sample_sizes: Sequence[int],
    repetitions: int,
    seed: int,
    folds: int = 5,
) -> list[dict[str, float | int]]:
    """Check point inference when common unsupported policy mass cancels.

    Every user has one structurally unsupported action.  The frozen target
    assigns that action exactly its population marginal probability, so the
    target/reference contrast is zero at every support hole even though both
    absolute values place positive mass there.  The marginal policy and the
    supported outcome marginal are re-estimated on each training fold.
    """

    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if folds < 2:
        raise ValueError("folds must be at least two")
    n_users = 3
    n_actions = 3
    policy_by_user = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 2.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0],
            [1.0 / 2.0, 1.0 / 6.0, 1.0 / 3.0],
        ]
    )
    support_by_user = ~np.eye(n_users, n_actions, dtype=bool)
    outcome_mean = np.asarray(
        [
            [0.95, 0.80, 0.10],
            [0.20, 0.05, 0.90],
            [0.70, 0.30, 0.40],
        ]
    )
    population_reference = np.broadcast_to(
        policy_by_user.mean(axis=0),
        policy_by_user.shape,
    )
    if np.any(
        np.abs(policy_by_user - population_reference)[~support_by_user] > 1e-12
    ):
        raise RuntimeError("support-cancellation DGP violates contrast overlap")
    true_matching_value = matching_value(
        policy_by_user,
        outcome_mean,
        np.zeros(n_users, dtype=np.int64),
    )
    critical = 1.959963984540054
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for sample_size in sample_sizes:
        if sample_size < max(100, 4 * folds * n_users):
            raise ValueError("sample sizes are too small for cross-fitted coverage")
        estimates: list[float] = []
        standard_errors: list[float] = []
        coverages: list[bool] = []
        for _ in range(repetitions):
            user = rng.integers(n_users, size=sample_size)
            action_offset = rng.integers(1, n_actions, size=sample_size)
            action = (user + action_offset) % n_actions
            reward = rng.binomial(
                1,
                outcome_mean[user, action],
            ).astype(np.float64)
            fold = np.arange(sample_size, dtype=np.int64) % folds
            rng.shuffle(fold)
            scores = np.empty(sample_size, dtype=np.float64)
            for held_out_fold in range(folds):
                training = fold != held_out_fold
                held_out = ~training
                training_user = user[training]
                marginal_policy = policy_by_user[training_user].mean(axis=0)
                supported_outcome_marginal = np.where(
                    support_by_user[training_user],
                    outcome_mean[training_user],
                    0.0,
                ).mean(axis=0)
                held_out_user = user[held_out]
                held_out_q = outcome_mean[held_out_user].copy()
                held_out_support = support_by_user[held_out_user]
                held_out_q[~held_out_support] = np.nan
                scores[held_out] = support_adapted_matching_dr_scores(
                    policy_by_user[held_out_user],
                    np.broadcast_to(marginal_policy, (held_out.sum(), n_actions)),
                    held_out_support,
                    action[held_out],
                    reward[held_out],
                    np.full(held_out.sum(), 0.5),
                    held_out_q,
                    np.broadcast_to(
                        supported_outcome_marginal,
                        (held_out.sum(), n_actions),
                    ),
                )
            estimate = float(scores.mean())
            standard_error = float(scores.std(ddof=1) / np.sqrt(sample_size))
            estimates.append(estimate)
            standard_errors.append(standard_error)
            coverages.append(
                estimate - critical * standard_error
                <= true_matching_value
                <= estimate + critical * standard_error
            )
        estimate_array = np.asarray(estimates)
        rows.append(
            {
                "sample_size": int(sample_size),
                "folds": int(folds),
                "repetitions": int(repetitions),
                "true_matching_value": float(true_matching_value),
                "mean_estimate": float(estimate_array.mean()),
                "bias": float(estimate_array.mean() - true_matching_value),
                "empirical_standard_deviation": float(estimate_array.std(ddof=1)),
                "mean_standard_error": float(np.mean(standard_errors)),
                "coverage_95pct": float(np.mean(coverages)),
                "unsupported_target_mass": float(1.0 / 3.0),
                "unsupported_reference_mass": float(1.0 / 3.0),
                "unsupported_contrast_mass": 0.0,
            }
        )
    return rows


def equality_restricted_efficiency_grid(
    *,
    sample_sizes: Sequence[int],
    repetitions: int,
    seed: int,
    folds: int = 5,
) -> list[dict[str, float | int | str]]:
    """Compare unrestricted and equality-restricted efficiency bounds.

    One action has a structural hole for one user type.  Exact contrast overlap
    fixes that action's population marginal probability, excluding one user-mix
    tangent direction.  Both estimators use the same support-adapted outcome
    nuisances; the restricted estimator additionally removes the conditional
    projection onto the excluded direction.
    """

    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if folds < 2:
        raise ValueError("folds must be at least two")
    n_users = 4
    n_actions = 3
    policy_by_user = np.asarray(
        [
            [0.3, 0.5, 0.2],
            [0.6, 0.2, 0.2],
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
        ]
    )
    population_reference = policy_by_user.mean(axis=0)
    support_by_user = np.ones_like(policy_by_user, dtype=bool)
    support_by_user[0, 0] = False
    hole_actions_by_user = np.broadcast_to(
        np.asarray([True, False, False]),
        policy_by_user.shape,
    )
    logging_by_user = np.asarray(
        [
            [0.0, 0.5, 0.5],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ]
    )
    outcome_mean = np.asarray(
        [
            [0.4, 0.9, 0.1],
            [0.8, 0.2, 0.4],
            [0.3, 0.6, 0.7],
            [0.2, 0.5, 0.8],
        ]
    )
    if abs(policy_by_user[0, 0] - population_reference[0]) > 1e-12:
        raise RuntimeError("restricted-efficiency DGP violates contrast overlap")
    true_matching_value = matching_value(
        policy_by_user,
        outcome_mean,
        np.zeros(n_users, dtype=np.int64),
    )
    methods = ("support_adapted_unrestricted", "equality_restricted_efficient")
    critical = 1.959963984540054
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for sample_size in sample_sizes:
        if sample_size < max(100, 4 * folds * n_users):
            raise ValueError("sample sizes are too small for cross-fitted efficiency")
        estimates = {method: [] for method in methods}
        standard_errors = {method: [] for method in methods}
        coverages = {method: [] for method in methods}
        for _ in range(repetitions):
            user = rng.integers(n_users, size=sample_size)
            action = rng.integers(n_actions, size=sample_size)
            first_user = user == 0
            action[first_user] = rng.integers(1, n_actions, size=first_user.sum())
            reward = rng.binomial(
                1,
                outcome_mean[user, action],
            ).astype(np.float64)
            fold = np.arange(sample_size, dtype=np.int64) % folds
            rng.shuffle(fold)
            repetition_scores = {
                method: np.empty(sample_size, dtype=np.float64)
                for method in methods
            }
            for held_out_fold in range(folds):
                training = fold != held_out_fold
                held_out = ~training
                training_user = user[training]
                training_support = support_by_user[training_user]
                training_q = outcome_mean[training_user].copy()
                training_q[~training_support] = np.nan
                supported_outcome_marginal = np.where(
                    training_support,
                    outcome_mean[training_user],
                    0.0,
                ).mean(axis=0)
                training_reference = np.broadcast_to(
                    population_reference,
                    (training.sum(), n_actions),
                )
                projection_coefficient = equality_restricted_projection_coefficients(
                    policy_by_user[training_user],
                    training_reference,
                    training_support,
                    hole_actions_by_user[training_user],
                    training_q,
                    np.broadcast_to(
                        supported_outcome_marginal,
                        (training.sum(), n_actions),
                    ),
                    np.zeros(training.sum(), dtype=np.int64),
                )[0]

                held_out_user = user[held_out]
                held_out_support = support_by_user[held_out_user]
                held_out_q = outcome_mean[held_out_user].copy()
                held_out_q[~held_out_support] = np.nan
                held_out_reference = np.broadcast_to(
                    population_reference,
                    (held_out.sum(), n_actions),
                )
                held_out_marginal_q = np.broadcast_to(
                    supported_outcome_marginal,
                    (held_out.sum(), n_actions),
                )
                held_out_pscore = logging_by_user[held_out_user, action[held_out]]
                unrestricted = support_adapted_matching_dr_scores(
                    policy_by_user[held_out_user],
                    held_out_reference,
                    held_out_support,
                    action[held_out],
                    reward[held_out],
                    held_out_pscore,
                    held_out_q,
                    held_out_marginal_q,
                )
                efficient = equality_restricted_matching_dr_scores(
                    policy_by_user[held_out_user],
                    held_out_reference,
                    held_out_support,
                    hole_actions_by_user[held_out_user],
                    np.broadcast_to(
                        projection_coefficient,
                        (held_out.sum(), n_actions),
                    ),
                    action[held_out],
                    reward[held_out],
                    held_out_pscore,
                    held_out_q,
                    held_out_marginal_q,
                )
                repetition_scores["support_adapted_unrestricted"][held_out] = (
                    unrestricted
                )
                repetition_scores["equality_restricted_efficient"][held_out] = efficient
            for method, scores in repetition_scores.items():
                estimate = float(scores.mean())
                standard_error = float(scores.std(ddof=1) / np.sqrt(sample_size))
                estimates[method].append(estimate)
                standard_errors[method].append(standard_error)
                coverages[method].append(
                    estimate - critical * standard_error
                    <= true_matching_value
                    <= estimate + critical * standard_error
                )
        unrestricted_variance = float(
            np.var(estimates["support_adapted_unrestricted"], ddof=1)
        )
        for method in methods:
            estimate_array = np.asarray(estimates[method])
            empirical_variance = float(estimate_array.var(ddof=1))
            rows.append(
                {
                    "method": method,
                    "sample_size": int(sample_size),
                    "folds": int(folds),
                    "repetitions": int(repetitions),
                    "true_matching_value": float(true_matching_value),
                    "mean_estimate": float(estimate_array.mean()),
                    "bias": float(estimate_array.mean() - true_matching_value),
                    "empirical_standard_deviation": float(
                        np.sqrt(empirical_variance)
                    ),
                    "mean_standard_error": float(np.mean(standard_errors[method])),
                    "coverage_95pct": float(np.mean(coverages[method])),
                    "variance_ratio_to_unrestricted": float(
                        empirical_variance / unrestricted_variance
                    ),
                    "unsupported_target_mass": 0.075,
                    "unsupported_reference_mass": 0.075,
                    "unsupported_contrast_mass": 0.0,
                }
            )
    return rows


def nuisance_coverage_grid(
    *,
    sample_sizes: Sequence[int],
    repetitions: int,
    seed: int,
    folds: int = 5,
    n_strata: int = 20,
) -> list[dict[str, float | int | str]]:
    """Check centered-score coverage with estimated and misspecified nuisances.

    The logging policy is known and uniform.  ``q_misspecified`` omits the user
    variable only from the outcome regression; the residual correction remains
    valid.  The single-marginal cases deliberately pool either ``bar_pi`` or
    ``bar_q`` across shared-context strata.  ``both_marginals_misspecified`` is
    a negative control: the product-bias term need not vanish when both
    conditional marginal functions converge to the wrong limits.
    """

    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if folds < 2:
        raise ValueError("folds must be at least two")
    if n_strata < 2:
        raise ValueError("n_strata must be at least two")
    n_users = 2
    n_actions = 3
    dgp_rng = np.random.default_rng(seed + 104_729)
    user_one_probability = 0.15 + 0.70 * np.arange(n_strata) / (n_strata - 1)
    user_probability = np.column_stack(
        (1.0 - user_one_probability, user_one_probability)
    )
    outcome_mean = dgp_rng.uniform(
        0.10,
        0.80,
        size=(n_strata, n_users, n_actions),
    )
    target_policy = _softmax_last_axis(
        3.0 * outcome_mean
        + dgp_rng.normal(0.0, 0.30, size=(n_strata, 1, n_actions))
    )
    true_marginal_policy = np.einsum(
        "cu,cua->ca",
        user_probability,
        target_policy,
    )
    true_matching_value = float(
        np.mean(
            np.einsum(
                "cu,cua,cua->c",
                user_probability,
                target_policy - true_marginal_policy[:, None, :],
                outcome_mean,
            )
        )
    )
    configurations = {
        "all_estimated_well_specified": (False, False, False),
        "q_misspecified": (True, False, False),
        "bar_pi_misspecified": (False, True, False),
        "bar_q_misspecified": (False, False, True),
        "both_marginals_misspecified": (False, True, True),
    }
    critical = 1.959963984540054
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for sample_size in sample_sizes:
        if sample_size < max(200, folds * n_strata * n_users * n_actions):
            raise ValueError("sample sizes are too small for the nuisance cell design")
        estimates: dict[tuple[str, str], list[float]] = {}
        standard_errors: dict[tuple[str, str], list[float]] = {}
        coverages: dict[tuple[str, str], list[bool]] = {}
        for scenario in configurations:
            for method in ("centered_score", "uncentered_dr"):
                key = (scenario, method)
                estimates[key] = []
                standard_errors[key] = []
                coverages[key] = []
        for _ in range(repetitions):
            shared_context = rng.integers(n_strata, size=sample_size)
            user = (
                rng.random(sample_size)
                < user_one_probability[shared_context]
            ).astype(np.int64)
            action = rng.integers(n_actions, size=sample_size)
            reward = rng.binomial(
                1,
                outcome_mean[shared_context, user, action],
            ).astype(np.float64)
            fold = rng.integers(folds, size=sample_size)
            repetition_scores = {
                key: np.empty(sample_size, dtype=np.float64)
                for key in estimates
            }
            for held_out_fold in range(folds):
                training = fold != held_out_fold
                held_out = ~training
                if not training.any() or not held_out.any():
                    raise RuntimeError("random fold assignment produced an empty fold")

                outcome_sum = np.zeros(
                    (n_strata, n_users, n_actions),
                    dtype=np.float64,
                )
                outcome_count = np.zeros_like(outcome_sum)
                np.add.at(
                    outcome_sum,
                    (
                        shared_context[training],
                        user[training],
                        action[training],
                    ),
                    reward[training],
                )
                np.add.at(
                    outcome_count,
                    (
                        shared_context[training],
                        user[training],
                        action[training],
                    ),
                    1.0,
                )
                q_correct = (outcome_sum + 1.0) / (outcome_count + 2.0)

                marginal_outcome_sum = np.zeros(
                    (n_strata, n_actions),
                    dtype=np.float64,
                )
                marginal_outcome_count = np.zeros_like(marginal_outcome_sum)
                np.add.at(
                    marginal_outcome_sum,
                    (shared_context[training], action[training]),
                    reward[training],
                )
                np.add.at(
                    marginal_outcome_count,
                    (shared_context[training], action[training]),
                    1.0,
                )
                marginal_q_correct = (
                    marginal_outcome_sum + 1.0
                ) / (marginal_outcome_count + 2.0)
                q_misspecified = np.broadcast_to(
                    marginal_q_correct[:, None, :],
                    q_correct.shape,
                )

                stratum_count = np.bincount(
                    shared_context[training],
                    minlength=n_strata,
                ).astype(np.float64)
                if np.any(stratum_count == 0.0):
                    raise RuntimeError("a training fold omitted a shared-context stratum")
                marginal_policy_correct = np.zeros(
                    (n_strata, n_actions),
                    dtype=np.float64,
                )
                np.add.at(
                    marginal_policy_correct,
                    shared_context[training],
                    target_policy[
                        shared_context[training],
                        user[training],
                    ],
                )
                marginal_policy_correct /= stratum_count[:, None]
                marginal_policy_misspecified = np.broadcast_to(
                    target_policy[
                        shared_context[training],
                        user[training],
                    ].mean(axis=0),
                    marginal_policy_correct.shape,
                )

                global_outcome_sum = np.bincount(
                    action[training],
                    weights=reward[training],
                    minlength=n_actions,
                ).astype(np.float64)
                global_outcome_count = np.bincount(
                    action[training],
                    minlength=n_actions,
                ).astype(np.float64)
                marginal_q_misspecified = np.broadcast_to(
                    (global_outcome_sum + 1.0) / (global_outcome_count + 2.0),
                    marginal_q_correct.shape,
                )

                held_out_context = shared_context[held_out]
                held_out_user = user[held_out]
                held_out_policy = target_policy[held_out_context, held_out_user]
                held_out_action = action[held_out]
                held_out_reward = reward[held_out]
                held_out_pscore = np.full(held_out_action.size, 1.0 / n_actions)
                for scenario, (
                    misspecify_q,
                    misspecify_bar_pi,
                    misspecify_bar_q,
                ) in configurations.items():
                    q_nuisance = (
                        q_misspecified if misspecify_q else q_correct
                    )[held_out_context, held_out_user]
                    marginal_policy_nuisance = (
                        marginal_policy_misspecified
                        if misspecify_bar_pi
                        else marginal_policy_correct
                    )[held_out_context]
                    marginal_q_nuisance = (
                        marginal_q_misspecified
                        if misspecify_bar_q
                        else marginal_q_correct
                    )[held_out_context]
                    centered = centered_matching_dr_scores(
                        held_out_policy,
                        marginal_policy_nuisance,
                        held_out_action,
                        held_out_reward,
                        held_out_pscore,
                        q_nuisance,
                        marginal_q_nuisance,
                    )
                    uncentered = contrast_dr_scores(
                        held_out_policy - marginal_policy_nuisance,
                        held_out_action,
                        held_out_reward,
                        held_out_pscore,
                        q_nuisance,
                    )
                    repetition_scores[(scenario, "centered_score")][held_out] = (
                        centered
                    )
                    repetition_scores[(scenario, "uncentered_dr")][held_out] = (
                        uncentered
                    )
            for key, scores in repetition_scores.items():
                estimate = float(scores.mean())
                standard_error = float(scores.std(ddof=1) / np.sqrt(sample_size))
                estimates[key].append(estimate)
                standard_errors[key].append(standard_error)
                coverages[key].append(
                    estimate - critical * standard_error
                    <= true_matching_value
                    <= estimate + critical * standard_error
                )
        for scenario in configurations:
            for method in ("centered_score", "uncentered_dr"):
                key = (scenario, method)
                estimate_array = np.asarray(estimates[key])
                rows.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "sample_size": int(sample_size),
                        "folds": int(folds),
                        "strata": int(n_strata),
                        "repetitions": int(repetitions),
                        "true_matching_value": true_matching_value,
                        "mean_estimate": float(estimate_array.mean()),
                        "bias": float(estimate_array.mean() - true_matching_value),
                        "rmse": float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        estimate_array - true_matching_value
                                    )
                                )
                            )
                        ),
                        "empirical_standard_deviation": float(
                            estimate_array.std(ddof=1)
                        ),
                        "mean_standard_error": float(
                            np.mean(standard_errors[key])
                        ),
                        "coverage_95pct": float(np.mean(coverages[key])),
                    }
                )
    return rows


def cluster_coverage_grid(
    *,
    repetitions: int,
    seed: int,
    n_users: int = 200,
    n_days: int = 20,
    rows_per_user_day: int = 1,
    interaction_strength: float = 0.1,
) -> list[dict[str, float | int | str]]:
    """Calibrate iid and multiway cluster intervals for matching scores.

    The target policy favors the action matching a binary user type.  Random
    user and day slopes perturb that same interaction, inducing dependence in
    the matching score rather than only adding outcome intercepts that cancel
    from the contrast.  Oracle outcome nuisances isolate interval calibration
    from nuisance-estimation error, which is tested separately by
    :func:`nuisance_coverage_grid`.
    """

    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if n_users < 20 or n_users % 2:
        raise ValueError("n_users must be even and at least twenty")
    if n_days < 5 or rows_per_user_day < 1:
        raise ValueError("cluster dimensions are too small")
    if interaction_strength <= 0.0 or not np.isfinite(interaction_strength):
        raise ValueError("interaction_strength must be finite and positive")

    user = np.repeat(
        np.arange(n_users, dtype=np.int64),
        n_days * rows_per_user_day,
    )
    day = np.tile(
        np.repeat(
            np.arange(n_days, dtype=np.int64),
            rows_per_user_day,
        ),
        n_users,
    )
    user_type = user % 2
    n_rows = user.size
    action_labels = np.arange(2, dtype=np.int64)
    target_policy = np.where(
        user_type[:, None] == action_labels[None, :],
        0.8,
        0.2,
    )
    reference_policy = np.full_like(target_policy, 0.5)
    outcome_mean = 0.2 + interaction_strength * (
        user_type[:, None] == action_labels[None, :]
    )
    marginal_outcome = np.full_like(
        outcome_mean,
        0.2 + interaction_strength / 2.0,
    )
    true_matching_value = matching_value(
        target_policy,
        outcome_mean,
        np.zeros(n_rows, dtype=np.int64),
    )
    regimes = {
        "iid": (0.0, 0.0),
        "user_cluster": (0.15, 0.0),
        "day_cluster": (0.0, 0.15),
        "user_and_day_cluster": (0.15, 0.15),
    }
    methods = ("iid", "user", "day", "two_way")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for regime, (user_slope_sd, day_slope_sd) in regimes.items():
        estimates = {method: [] for method in methods}
        standard_errors = {method: [] for method in methods}
        coverages = {method: [] for method in methods}
        failures = {method: 0 for method in methods}
        for _ in range(repetitions):
            user_slope = rng.normal(0.0, user_slope_sd, size=n_users)
            day_slope = rng.normal(0.0, day_slope_sd, size=n_days)
            action = rng.integers(0, 2, size=n_rows)
            matched = action == user_type
            reward = (
                outcome_mean[np.arange(n_rows), action]
                + (user_slope[user] + day_slope[day]) * matched
                + rng.normal(0.0, 0.4, size=n_rows)
            )
            scores = centered_matching_dr_scores(
                target_policy,
                reference_policy,
                action,
                reward,
                np.full(n_rows, 0.5),
                outcome_mean,
                marginal_outcome,
            )
            iid = estimate_from_scores(scores)
            summaries: dict[str, tuple[float, float, float]] = {
                "iid": (iid.standard_error, iid.ci_lower, iid.ci_upper),
            }
            for method, labels in (("user", user), ("day", day)):
                summary = cluster_inference(scores, labels, prefix="interval")
                summaries[method] = (
                    float(summary["interval_standard_error"]),
                    float(summary["interval_ci_lower"]),
                    float(summary["interval_ci_upper"]),
                )
            try:
                summary = two_way_cluster_inference(
                    scores,
                    user,
                    day,
                    prefix="interval",
                )
                summaries["two_way"] = (
                    float(summary["interval_standard_error"]),
                    float(summary["interval_ci_lower"]),
                    float(summary["interval_ci_upper"]),
                )
            except ValueError:
                failures["two_way"] += 1

            estimate = float(scores.mean())
            for method, (standard_error, lower, upper) in summaries.items():
                estimates[method].append(estimate)
                standard_errors[method].append(standard_error)
                coverages[method].append(
                    lower <= true_matching_value <= upper
                )

        for method in methods:
            valid_repetitions = len(estimates[method])
            if valid_repetitions == 0:
                mean_estimate = float("nan")
                bias = float("nan")
                empirical_sd = float("nan")
                mean_se = float("nan")
                coverage = float("nan")
                coverage_mcse = float("nan")
            else:
                estimate_array = np.asarray(estimates[method])
                mean_estimate = float(estimate_array.mean())
                bias = mean_estimate - true_matching_value
                empirical_sd = float(estimate_array.std(ddof=1))
                mean_se = float(np.mean(standard_errors[method]))
                coverage = float(np.mean(coverages[method]))
                coverage_mcse = float(
                    np.sqrt(coverage * (1.0 - coverage) / valid_repetitions)
                )
            rows.append(
                {
                    "regime": regime,
                    "method": method,
                    "rows": int(n_rows),
                    "users": int(n_users),
                    "days": int(n_days),
                    "rows_per_user_day": int(rows_per_user_day),
                    "repetitions": int(repetitions),
                    "valid_repetitions": int(valid_repetitions),
                    "variance_failures": int(failures[method]),
                    "true_matching_value": float(true_matching_value),
                    "mean_estimate": mean_estimate,
                    "bias": bias,
                    "empirical_standard_deviation": empirical_sd,
                    "mean_standard_error": mean_se,
                    "coverage_95pct": coverage,
                    "coverage_mcse": coverage_mcse,
                    "user_slope_sd": float(user_slope_sd),
                    "day_slope_sd": float(day_slope_sd),
                }
            )
    return rows


def nested_context_day_coverage_grid(
    *,
    day_counts: Sequence[int],
    repetitions: int,
    seed: int,
    n_users: int = 200,
    total_rows: int = 12_000,
    contexts_per_day: int = 3,
) -> list[dict[str, float | int | str]]:
    """Stress-test few-day inference with contexts nested inside days.

    The total event count is fixed while the number of independent days
    changes.  Sparse Bernoulli outcomes mimic the scale at which clustering
    matters in the logged studies.  Symmetric and skewed day shocks separate
    the cluster-t small-G problem from the extra symmetry assumption used by
    the enumerated sign-flip diagnostic.
    """

    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if n_users < 20 or n_users % 2:
        raise ValueError("n_users must be even and at least twenty")
    if total_rows < 1_000 or contexts_per_day < 1:
        raise ValueError("nested day design is too small")
    days_grid = tuple(sorted({int(value) for value in day_counts}))
    if not days_grid or days_grid[0] < 2 or days_grid[-1] > 40:
        raise ValueError("day counts must lie between two and forty")

    regimes = (
        "symmetric_day_cluster",
        "skewed_day_cluster",
        "symmetric_user_and_day_cluster",
    )
    core_methods = ("iid", "user", "day", "two_way")
    rows: list[dict[str, float | int | str]] = []
    for n_days in days_grid:
        design_rng = np.random.default_rng(seed + 10_000 + n_days)
        row_index = np.arange(total_rows, dtype=np.int64)
        day = row_index % n_days
        time_bin = (row_index // n_days) % contexts_per_day
        context = day * contexts_per_day + time_bin
        user = design_rng.integers(0, n_users, size=total_rows)
        user_type = user % 2
        action_labels = np.arange(2, dtype=np.int64)
        target_policy = np.where(
            user_type[:, None] == action_labels[None, :],
            0.8,
            0.2,
        )
        reference_policy = marginalize_policy(target_policy, context)
        outcome_mean = 0.08 + 0.02 * (
            user_type[:, None] == action_labels[None, :]
        )
        marginal_outcome = np.empty_like(outcome_mean)
        for context_index in np.unique(context):
            selected = context == context_index
            marginal_outcome[selected] = outcome_mean[selected].mean(axis=0)
        true_matching_value = float(
            np.mean(
                np.sum(
                    (target_policy - reference_policy) * outcome_mean,
                    axis=1,
                )
            )
        )
        rng = np.random.default_rng(seed + n_days)
        for regime in regimes:
            methods = list(core_methods)
            if n_days == 7:
                methods.append("day_sign_flip")
            estimates = {method: [] for method in methods}
            standard_errors = {method: [] for method in core_methods}
            coverages = {method: [] for method in methods}
            failures = {method: 0 for method in methods}
            for _ in range(repetitions):
                if regime == "skewed_day_cluster":
                    day_slope = rng.choice(
                        np.asarray([-0.02, 0.08]),
                        size=n_days,
                        p=np.asarray([0.8, 0.2]),
                    )
                    user_slope = np.zeros(n_users)
                else:
                    day_slope = rng.choice(
                        np.asarray([-0.04, 0.04]),
                        size=n_days,
                    )
                    user_slope = (
                        rng.choice(
                            np.asarray([-0.01, 0.01]),
                            size=n_users,
                        )
                        if regime == "symmetric_user_and_day_cluster"
                        else np.zeros(n_users)
                    )
                action = rng.integers(0, 2, size=total_rows)
                matched = action == user_type
                probability = (
                    outcome_mean[row_index, action]
                    + (day_slope[day] + user_slope[user]) * matched
                )
                reward = rng.binomial(1, probability).astype(np.float64)
                scores = centered_matching_dr_scores(
                    target_policy,
                    reference_policy,
                    action,
                    reward,
                    np.full(total_rows, 0.5),
                    outcome_mean,
                    marginal_outcome,
                )
                iid = estimate_from_scores(scores)
                summaries: dict[str, tuple[float, float, float]] = {
                    "iid": (iid.standard_error, iid.ci_lower, iid.ci_upper),
                }
                for method, labels in (("user", user), ("day", day)):
                    summary = cluster_inference(scores, labels, prefix="interval")
                    summaries[method] = (
                        float(summary["interval_standard_error"]),
                        float(summary["interval_ci_lower"]),
                        float(summary["interval_ci_upper"]),
                    )
                try:
                    summary = two_way_cluster_inference(
                        scores,
                        user,
                        day,
                        prefix="interval",
                    )
                    summaries["two_way"] = (
                        float(summary["interval_standard_error"]),
                        float(summary["interval_ci_lower"]),
                        float(summary["interval_ci_upper"]),
                    )
                except ValueError:
                    failures["two_way"] += 1
                estimate = float(scores.mean())
                for method, (standard_error, lower, upper) in summaries.items():
                    estimates[method].append(estimate)
                    standard_errors[method].append(standard_error)
                    coverages[method].append(
                        lower <= true_matching_value <= upper
                    )
                if n_days == 7:
                    sign_flip = enumerated_cluster_sign_flip_interval(
                        scores,
                        day,
                        prefix="interval",
                    )
                    estimates["day_sign_flip"].append(estimate)
                    coverages["day_sign_flip"].append(
                        float(sign_flip["interval_sign_flip_ci_lower"])
                        <= true_matching_value
                        <= float(sign_flip["interval_sign_flip_ci_upper"])
                    )

            for method in methods:
                estimate_array = np.asarray(estimates[method])
                coverage = float(np.mean(coverages[method]))
                rows.append(
                    {
                        "regime": regime,
                        "method": method,
                        "rows": int(total_rows),
                        "users": int(np.unique(user).size),
                        "days": int(n_days),
                        "contexts": int(np.unique(context).size),
                        "contexts_per_day": int(contexts_per_day),
                        "repetitions": int(repetitions),
                        "valid_repetitions": int(estimate_array.size),
                        "true_matching_value": true_matching_value,
                        "mean_estimate": float(estimate_array.mean()),
                        "bias": float(
                            estimate_array.mean() - true_matching_value
                        ),
                        "empirical_standard_deviation": float(
                            estimate_array.std(ddof=1)
                        ),
                        "mean_standard_error": (
                            float(np.mean(standard_errors[method]))
                            if method in standard_errors
                            else float("nan")
                        ),
                        "coverage_95pct_or_sensitivity": coverage,
                        "coverage_mcse": float(
                            np.sqrt(
                                coverage
                                * (1.0 - coverage)
                                / estimate_array.size
                            )
                        ),
                        "variance_failures": int(failures[method]),
                        "interval_role": (
                            "sign_flip_sensitivity_assumes_symmetric_clusters"
                            if method == "day_sign_flip"
                            else "nominal_95pct_interval"
                        ),
                        "sign_flip_symmetry_holds_by_construction": bool(
                            regime != "skewed_day_cluster"
                        ),
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: object) -> object:
    """Represent unavailable non-finite results as JSON null."""

    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--nuisance-repetitions", type=int, default=None)
    parser.add_argument("--cluster-repetitions", type=int, default=None)
    parser.add_argument("--nested-cluster-repetitions", type=int, default=None)
    parser.add_argument("--nuisance-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    mechanisms = mechanism_scenarios()
    frontier = detectability_frontier(
        sample_sizes=(1_000, 5_000, 20_000),
        interaction_strengths=(0.0, 0.02, 0.05, 0.10),
        repetitions=args.repetitions,
        seed=args.seed,
    )
    nuisance = nuisance_coverage_grid(
        sample_sizes=(5_000, 20_000),
        repetitions=(
            args.repetitions
            if args.nuisance_repetitions is None
            else args.nuisance_repetitions
        ),
        seed=args.seed + 1,
        folds=args.nuisance_folds,
    )
    support_cancellation = support_cancellation_coverage(
        sample_sizes=(1_000, 5_000),
        repetitions=args.repetitions,
        seed=args.seed + 2,
        folds=args.nuisance_folds,
    )
    restricted_efficiency = equality_restricted_efficiency_grid(
        sample_sizes=(1_000, 5_000),
        repetitions=args.repetitions,
        seed=args.seed + 3,
        folds=args.nuisance_folds,
    )
    cluster_coverage = cluster_coverage_grid(
        repetitions=(
            args.repetitions
            if args.cluster_repetitions is None
            else args.cluster_repetitions
        ),
        seed=args.seed + 4,
    )
    nested_cluster_coverage = nested_context_day_coverage_grid(
        day_counts=(7, 10, 20, 40),
        repetitions=(
            args.repetitions
            if args.nested_cluster_repetitions is None
            else args.nested_cluster_repetitions
        ),
        seed=args.seed + 5,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "mechanism_scenarios.csv", mechanisms)
    _write_csv(args.output_dir / "detectability_frontier.csv", frontier)
    _write_csv(args.output_dir / "nuisance_coverage.csv", nuisance)
    _write_csv(
        args.output_dir / "support_cancellation_coverage.csv",
        support_cancellation,
    )
    _write_csv(
        args.output_dir / "equality_restricted_efficiency.csv",
        restricted_efficiency,
    )
    _write_csv(args.output_dir / "cluster_coverage.csv", cluster_coverage)
    _write_csv(
        args.output_dir / "nested_context_day_coverage.csv",
        nested_cluster_coverage,
    )
    payload = {
        "protocol": {
            "seed": args.seed,
            "repetitions": args.repetitions,
            "outcome": "Bernoulli",
            "logging_policy": "uniform over two actions",
            "target_policy": "0.8 on the user-matched action",
            "nuisance_cross_fitting_folds": args.nuisance_folds,
            "nuisance_coverage_repetitions": (
                args.repetitions
                if args.nuisance_repetitions is None
                else args.nuisance_repetitions
            ),
            "cluster_coverage_repetitions": (
                args.repetitions
                if args.cluster_repetitions is None
                else args.cluster_repetitions
            ),
            "nested_context_day_coverage_repetitions": (
                args.repetitions
                if args.nested_cluster_repetitions is None
                else args.nested_cluster_repetitions
            ),
        },
        "mechanism_scenarios": mechanisms,
        "detectability_frontier": frontier,
        "nuisance_coverage": nuisance,
        "support_cancellation_coverage": support_cancellation,
        "equality_restricted_efficiency": restricted_efficiency,
        "cluster_coverage": cluster_coverage,
        "nested_context_day_coverage": nested_cluster_coverage,
    }
    serialized = json.dumps(_json_ready(payload), indent=2, allow_nan=False)
    (args.output_dir / "summary.json").write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
