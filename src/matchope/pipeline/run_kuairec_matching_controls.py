#!/usr/bin/env python3
from __future__ import annotations

"""Run common-exposure and positive controls for the KuaiRec matching study.

The learned policies in the main study are equally far from their own
fold-conditioned marginal policies.  Their marginals can still differ.  This
module constructs a score-only stress test in which interaction MF and
user-kNN have both the same fold-conditioned action marginal and the same
mean total-variation distance from that marginal. The main comparison uses
the midpoint of their original marginals. A fixed path between the two
endpoints checks whether that reference choice changes the result. A
constructed shared-score control adds only the fold-specific item-quality
score to the already fitted interaction score. Common-exposure projection
should remove that shared action term exactly.

A strength-scale sensitivity keeps the same common marginal but replaces
equal TV by equal mean KL from that marginal. Because the projected marginal
is fixed, this KL is the conditional mutual information between user and
action under the policy.

For a fixed score strength, the common marginal is imposed by the minimum-KL
projection of the score policy.  The score strength is then calibrated to the
requested TV.  Candidate outcomes are read only after both calibrated
policies have been fixed.  A separate positive control uses the held-out
outcomes as its score and then follows the same projection and TV calibration.
It is not a learned or deployable policy, and it is not claimed to be the best
policy among all policies that meet the two constraints.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.special import logsumexp

from matchope.pipeline.run_kuairec_matching_learned import (
    _evaluate_frozen_score,
    calibrate_inverse_temperature_to_personalization_tv,
    construct_cross_fitted_knn_policy_split,
    construct_cross_fitted_learned_policy_split,
)
from matchope.pipeline.run_kuairec_matching_oracle import (
    _softmax,
    load_preference_matrix,
)


def fold_conditioned_marginal(
    policy: np.ndarray,
    evaluation_fold_ids: np.ndarray,
) -> np.ndarray:
    """Return one action marginal per contiguous evaluation fold."""

    probabilities = np.asarray(policy, dtype=np.float64)
    fold_ids = np.asarray(evaluation_fold_ids, dtype=np.int64)
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise ValueError("policy must be a finite matrix")
    if fold_ids.shape != (probabilities.shape[0],):
        raise ValueError("evaluation_fold_ids must align with policy rows")
    labels = np.unique(fold_ids)
    if not np.array_equal(labels, np.arange(labels.size)):
        raise ValueError("evaluation fold IDs must be contiguous from zero")
    if np.any(probabilities <= 0.0):
        raise ValueError("policy probabilities must be strictly positive")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("policy rows must sum to one")
    return np.vstack(
        [probabilities[fold_ids == fold_id].mean(axis=0) for fold_id in labels]
    )


def minimum_kl_exposure_projection(
    score: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_marginal: np.ndarray,
    inverse_temperature: float,
    tolerance: float = 1e-10,
    maximum_iterations: int = 200,
) -> tuple[np.ndarray, float, int]:
    """Project a score policy to a fixed fold marginal by action intercepts."""

    score_matrix = np.asarray(score, dtype=np.float64)
    fold_ids = np.asarray(evaluation_fold_ids, dtype=np.int64)
    marginal = np.asarray(target_marginal, dtype=np.float64)
    if score_matrix.ndim != 2 or not np.isfinite(score_matrix).all():
        raise ValueError("score must be a finite matrix")
    if fold_ids.shape != (score_matrix.shape[0],):
        raise ValueError("evaluation_fold_ids must align with score rows")
    labels = np.unique(fold_ids)
    if not np.array_equal(labels, np.arange(labels.size)):
        raise ValueError("evaluation fold IDs must be contiguous from zero")
    if marginal.shape != (labels.size, score_matrix.shape[1]):
        raise ValueError("target_marginal must have one row per fold")
    if not np.isfinite(marginal).all() or np.any(marginal <= 0.0):
        raise ValueError("target_marginal must be finite and strictly positive")
    if not np.allclose(marginal.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("target marginal rows must sum to one")
    if inverse_temperature < 0.0 or not np.isfinite(inverse_temperature):
        raise ValueError("inverse_temperature must be finite and non-negative")
    if tolerance <= 0.0 or maximum_iterations <= 0:
        raise ValueError("projection controls must be positive")

    projected = np.empty_like(score_matrix)
    maximum_error = 0.0
    iterations_used = 0
    for fold_id in labels:
        fold_mask = fold_ids == fold_id
        logits = inverse_temperature * score_matrix[fold_mask]
        action_intercepts = np.zeros(score_matrix.shape[1], dtype=np.float64)
        fold_error = float("inf")
        for iteration in range(1, maximum_iterations + 1):
            log_policy = logits + action_intercepts[None, :]
            log_policy -= logsumexp(log_policy, axis=1, keepdims=True)
            log_marginal = logsumexp(log_policy, axis=0) - np.log(
                int(fold_mask.sum())
            )
            fold_error = float(
                np.max(np.abs(np.exp(log_marginal) - marginal[fold_id]))
            )
            if fold_error <= tolerance:
                break
            action_intercepts += np.log(marginal[fold_id]) - log_marginal
        if fold_error > tolerance:
            raise ValueError("minimum-KL exposure projection did not converge")
        projected[fold_mask] = np.exp(log_policy)
        maximum_error = max(maximum_error, fold_error)
        iterations_used = max(iterations_used, iteration)
    return projected, maximum_error, iterations_used


def mean_tv_to_fold_marginal(
    policy: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_marginal: np.ndarray,
) -> float:
    probabilities = np.asarray(policy, dtype=np.float64)
    fold_ids = np.asarray(evaluation_fold_ids, dtype=np.int64)
    marginal = np.asarray(target_marginal, dtype=np.float64)
    if probabilities.ndim != 2 or fold_ids.shape != (probabilities.shape[0],):
        raise ValueError("policy and fold IDs must align")
    labels = np.unique(fold_ids)
    if not np.array_equal(labels, np.arange(labels.size)):
        raise ValueError("evaluation fold IDs must be contiguous from zero")
    if marginal.shape != (labels.size, probabilities.shape[1]):
        raise ValueError("target marginal must have one aligned row per fold")
    return float(
        np.mean(0.5 * np.abs(probabilities - marginal[fold_ids]).sum(axis=1))
    )


def mean_kl_to_fold_marginal(
    policy: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_marginal: np.ndarray,
) -> float:
    """Return mean user-level KL from a fixed fold marginal, in nats."""

    probabilities = np.asarray(policy, dtype=np.float64)
    fold_ids = np.asarray(evaluation_fold_ids, dtype=np.int64)
    marginal = np.asarray(target_marginal, dtype=np.float64)
    if probabilities.ndim != 2 or fold_ids.shape != (probabilities.shape[0],):
        raise ValueError("policy and fold IDs must align")
    labels = np.unique(fold_ids)
    if not np.array_equal(labels, np.arange(labels.size)):
        raise ValueError("evaluation fold IDs must be contiguous from zero")
    if marginal.shape != (labels.size, probabilities.shape[1]):
        raise ValueError("target marginal must have one aligned row per fold")
    if np.any(probabilities <= 0.0) or np.any(marginal <= 0.0):
        raise ValueError("policy and target marginal must be strictly positive")
    marginal_rows = marginal[fold_ids]
    return float(
        np.mean(
            np.sum(
                probabilities * (np.log(probabilities) - np.log(marginal_rows)),
                axis=1,
            )
        )
    )


def _calibrate_common_exposure_strength(
    score: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_marginal: np.ndarray,
    target_strength: float,
    initial_upper: float,
    strength_measure: Callable[..., float],
    strength_name: str,
    strength_tolerance: float,
    projection_tolerance: float = 1e-10,
    maximum_iterations: int = 35,
) -> tuple[float, np.ndarray, float, float, int]:
    """Calibrate a scalar strength measure on the projected score path."""

    if target_strength <= 0.0 or not np.isfinite(target_strength):
        raise ValueError(f"target {strength_name} must be finite and positive")
    if initial_upper <= 0.0 or not np.isfinite(initial_upper):
        raise ValueError("initial_upper must be finite and positive")
    if strength_tolerance <= 0.0 or maximum_iterations <= 0:
        raise ValueError("calibration controls must be positive")

    lower = 0.0
    lower_strength = 0.0
    upper = float(initial_upper)
    policy, marginal_error, projection_iterations = minimum_kl_exposure_projection(
        score,
        evaluation_fold_ids=evaluation_fold_ids,
        target_marginal=target_marginal,
        inverse_temperature=upper,
        tolerance=projection_tolerance,
    )
    achieved_strength = strength_measure(
        policy,
        evaluation_fold_ids=evaluation_fold_ids,
        target_marginal=target_marginal,
    )
    while achieved_strength < target_strength and upper < 1024.0:
        if achieved_strength + strength_tolerance < lower_strength:
            raise ValueError(
                f"common-exposure {strength_name} path is not monotone"
            )
        lower = upper
        lower_strength = achieved_strength
        upper *= 2.0
        policy, marginal_error, projection_iterations = (
            minimum_kl_exposure_projection(
                score,
                evaluation_fold_ids=evaluation_fold_ids,
                target_marginal=target_marginal,
                inverse_temperature=upper,
                tolerance=projection_tolerance,
            )
        )
        achieved_strength = strength_measure(
            policy,
            evaluation_fold_ids=evaluation_fold_ids,
            target_marginal=target_marginal,
        )
    if achieved_strength + strength_tolerance < lower_strength:
        raise ValueError(f"common-exposure {strength_name} path is not monotone")
    if achieved_strength < target_strength:
        raise ValueError(
            f"target {strength_name} is not reachable after exposure calibration"
        )

    calibrated = upper
    for _ in range(maximum_iterations):
        calibrated = 0.5 * (lower + upper)
        policy, marginal_error, projection_iterations = (
            minimum_kl_exposure_projection(
                score,
                evaluation_fold_ids=evaluation_fold_ids,
                target_marginal=target_marginal,
                inverse_temperature=calibrated,
                tolerance=projection_tolerance,
            )
        )
        achieved_strength = strength_measure(
            policy,
            evaluation_fold_ids=evaluation_fold_ids,
            target_marginal=target_marginal,
        )
        if abs(achieved_strength - target_strength) <= strength_tolerance:
            break
        if achieved_strength < target_strength:
            lower = calibrated
        else:
            upper = calibrated
    if abs(achieved_strength - target_strength) > strength_tolerance:
        raise ValueError(
            f"common-exposure {strength_name} calibration did not converge"
        )
    return (
        float(calibrated),
        policy,
        float(achieved_strength),
        float(marginal_error),
        int(projection_iterations),
    )


def calibrate_common_exposure_policy(
    score: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_marginal: np.ndarray,
    target_tv: float,
    initial_upper: float,
    tv_tolerance: float = 1e-7,
    projection_tolerance: float = 1e-10,
    maximum_iterations: int = 35,
) -> tuple[float, np.ndarray, float, float, int]:
    """Calibrate score strength after imposing a common marginal and TV."""

    if not 0.0 < target_tv < 1.0 or not np.isfinite(target_tv):
        raise ValueError("target_tv must be finite and lie strictly between 0 and 1")
    return _calibrate_common_exposure_strength(
        score,
        evaluation_fold_ids=evaluation_fold_ids,
        target_marginal=target_marginal,
        target_strength=target_tv,
        initial_upper=initial_upper,
        strength_measure=mean_tv_to_fold_marginal,
        strength_name="TV",
        strength_tolerance=tv_tolerance,
        projection_tolerance=projection_tolerance,
        maximum_iterations=maximum_iterations,
    )


def calibrate_common_exposure_policy_to_kl(
    score: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_marginal: np.ndarray,
    target_kl: float,
    initial_upper: float,
    kl_tolerance: float = 1e-8,
    projection_tolerance: float = 1e-10,
    maximum_iterations: int = 35,
) -> tuple[float, np.ndarray, float, float, int]:
    """Calibrate mean KL from the common marginal instead of TV."""

    return _calibrate_common_exposure_strength(
        score,
        evaluation_fold_ids=evaluation_fold_ids,
        target_marginal=target_marginal,
        target_strength=target_kl,
        initial_upper=initial_upper,
        strength_measure=mean_kl_to_fold_marginal,
        strength_name="KL",
        strength_tolerance=kl_tolerance,
        projection_tolerance=projection_tolerance,
        maximum_iterations=maximum_iterations,
    )


def policy_tv_quantiles(
    policy: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_marginal: np.ndarray,
) -> tuple[float, float, float]:
    """Return user-level TV quantiles for a calibrated policy."""

    probabilities = np.asarray(policy, dtype=np.float64)
    fold_ids = np.asarray(evaluation_fold_ids, dtype=np.int64)
    marginal = np.asarray(target_marginal, dtype=np.float64)
    row_tv = 0.5 * np.abs(probabilities - marginal[fold_ids]).sum(axis=1)
    p10, p50, p90 = np.quantile(row_tv, [0.1, 0.5, 0.9])
    return float(p10), float(p50), float(p90)


def fixed_common_marginal_from_scores(
    factor_score: np.ndarray,
    knn_score: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    anchor_tv: float,
    common_marginal_weight: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """Build one score-only common marginal at a fixed anchor TV."""

    weight = float(common_marginal_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("common_marginal_weight must lie between zero and one")
    factor_beta = calibrate_inverse_temperature_to_personalization_tv(
        factor_score,
        evaluation_fold_ids=evaluation_fold_ids,
        target_tv=anchor_tv,
    )
    knn_beta = calibrate_inverse_temperature_to_personalization_tv(
        knn_score,
        evaluation_fold_ids=evaluation_fold_ids,
        target_tv=anchor_tv,
    )
    factor_marginal = fold_conditioned_marginal(
        _softmax(factor_score, factor_beta),
        evaluation_fold_ids,
    )
    knn_marginal = fold_conditioned_marginal(
        _softmax(knn_score, knn_beta),
        evaluation_fold_ids,
    )
    common_marginal = (
        (1.0 - weight) * factor_marginal + weight * knn_marginal
    )
    return common_marginal, float(factor_beta), float(knn_beta)


def evaluate_fixed_common_marginal_strength_path(
    factor_score: np.ndarray,
    knn_score: np.ndarray,
    utility: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_marginal: np.ndarray,
    target_tvs: Sequence[float],
    initial_factor_upper: float,
    initial_knn_upper: float,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    """Sweep policy strength while keeping one common marginal fixed."""

    factor = np.asarray(factor_score, dtype=np.float64)
    knn = np.asarray(knn_score, dtype=np.float64)
    outcome = np.asarray(utility, dtype=np.float64)
    fold_ids = np.asarray(evaluation_fold_ids, dtype=np.int64)
    marginal = np.asarray(target_marginal, dtype=np.float64)
    if factor.shape != knn.shape or factor.shape != outcome.shape:
        raise ValueError("factor scores, kNN scores, and outcomes must align")
    reference_rows = marginal[fold_ids]
    reference_value = float(np.mean(np.sum(reference_rows * outcome, axis=1)))
    rows: list[dict[str, float]] = []
    contributions: list[np.ndarray] = []
    for target_tv in sorted({float(value) for value in target_tvs}):
        factor_beta, factor_policy, factor_tv, factor_error, _ = (
            calibrate_common_exposure_policy(
                factor,
                evaluation_fold_ids=fold_ids,
                target_marginal=marginal,
                target_tv=target_tv,
                initial_upper=initial_factor_upper,
            )
        )
        knn_beta, knn_policy, knn_tv, knn_error, _ = (
            calibrate_common_exposure_policy(
                knn,
                evaluation_fold_ids=fold_ids,
                target_marginal=marginal,
                target_tv=target_tv,
                initial_upper=initial_knn_upper,
            )
        )
        factor_value = float(np.mean(np.sum(factor_policy * outcome, axis=1)))
        knn_value = float(np.mean(np.sum(knn_policy * outcome, axis=1)))
        paired = np.sum((knn_policy - factor_policy) * outcome, axis=1)
        factor_p10, factor_p50, factor_p90 = policy_tv_quantiles(
            factor_policy,
            evaluation_fold_ids=fold_ids,
            target_marginal=marginal,
        )
        knn_p10, knn_p50, knn_p90 = policy_tv_quantiles(
            knn_policy,
            evaluation_fold_ids=fold_ids,
            target_marginal=marginal,
        )
        rows.append(
            {
                "target_tv": target_tv,
                "factor_inverse_temperature": factor_beta,
                "knn_inverse_temperature": knn_beta,
                "factor_actual_tv": factor_tv,
                "knn_actual_tv": knn_tv,
                "factor_marginal_error": factor_error,
                "knn_marginal_error": knn_error,
                "reference_value": reference_value,
                "factor_matching_value": factor_value - reference_value,
                "knn_matching_value": knn_value - reference_value,
                "delta_matching_knn_minus_factor": knn_value - factor_value,
                "factor_tv_p10": factor_p10,
                "factor_tv_p50": factor_p50,
                "factor_tv_p90": factor_p90,
                "knn_tv_p10": knn_p10,
                "knn_tv_p50": knn_p50,
                "knn_tv_p90": knn_p90,
            }
        )
        contributions.append(paired)
    return rows, contributions


def evaluate_common_exposure_kl_sensitivity(
    factor_score: np.ndarray,
    knn_score: np.ndarray,
    utility: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_tvs: Sequence[float],
    common_marginal_weight: float = 0.5,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    """Replace equal TV by equal conditional mutual information.

    For each main-study TV setting, the common marginal is held fixed. We
    first obtain the two equal-TV projected policies and set the KL target to
    the mean of their user-level KL divergences from that marginal. Both
    frozen score policies are then recalibrated to this common target. The
    target selection reads scores but not candidate outcomes.
    """

    factor = np.asarray(factor_score, dtype=np.float64)
    knn = np.asarray(knn_score, dtype=np.float64)
    outcome = np.asarray(utility, dtype=np.float64)
    fold_ids = np.asarray(evaluation_fold_ids, dtype=np.int64)
    if factor.shape != knn.shape or factor.shape != outcome.shape:
        raise ValueError("factor scores, kNN scores, and outcomes must align")
    if fold_ids.shape != (outcome.shape[0],):
        raise ValueError("evaluation fold IDs must align with outcomes")
    weight = float(common_marginal_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("common_marginal_weight must lie between zero and one")

    rows: list[dict[str, float]] = []
    contributions: list[np.ndarray] = []
    for target_tv in sorted({float(value) for value in target_tvs}):
        factor_own_beta = calibrate_inverse_temperature_to_personalization_tv(
            factor,
            evaluation_fold_ids=fold_ids,
            target_tv=target_tv,
        )
        knn_own_beta = calibrate_inverse_temperature_to_personalization_tv(
            knn,
            evaluation_fold_ids=fold_ids,
            target_tv=target_tv,
        )
        factor_marginal = fold_conditioned_marginal(
            _softmax(factor, factor_own_beta),
            fold_ids,
        )
        knn_marginal = fold_conditioned_marginal(
            _softmax(knn, knn_own_beta),
            fold_ids,
        )
        common_marginal = (
            (1.0 - weight) * factor_marginal + weight * knn_marginal
        )
        (
            factor_tv_beta,
            factor_tv_policy,
            factor_actual_tv,
            _,
            _,
        ) = calibrate_common_exposure_policy(
            factor,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
            target_tv=target_tv,
            initial_upper=factor_own_beta,
        )
        (
            knn_tv_beta,
            knn_tv_policy,
            knn_actual_tv,
            _,
            _,
        ) = calibrate_common_exposure_policy(
            knn,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
            target_tv=target_tv,
            initial_upper=knn_own_beta,
        )
        factor_tv_kl = mean_kl_to_fold_marginal(
            factor_tv_policy,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
        )
        knn_tv_kl = mean_kl_to_fold_marginal(
            knn_tv_policy,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
        )
        target_kl = 0.5 * (factor_tv_kl + knn_tv_kl)
        (
            factor_kl_beta,
            factor_kl_policy,
            factor_actual_kl,
            factor_marginal_error,
            _,
        ) = calibrate_common_exposure_policy_to_kl(
            factor,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
            target_kl=target_kl,
            initial_upper=factor_tv_beta,
        )
        (
            knn_kl_beta,
            knn_kl_policy,
            knn_actual_kl,
            knn_marginal_error,
            _,
        ) = calibrate_common_exposure_policy_to_kl(
            knn,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
            target_kl=target_kl,
            initial_upper=knn_tv_beta,
        )
        factor_kl_tv = mean_tv_to_fold_marginal(
            factor_kl_policy,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
        )
        knn_kl_tv = mean_tv_to_fold_marginal(
            knn_kl_policy,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
        )
        paired = np.sum((knn_kl_policy - factor_kl_policy) * outcome, axis=1)
        rows.append(
            {
                "source_target_tv": target_tv,
                "common_marginal_weight": weight,
                "target_mean_kl_nats": target_kl,
                "tv_factor_actual_tv": factor_actual_tv,
                "tv_knn_actual_tv": knn_actual_tv,
                "tv_factor_mean_kl_nats": factor_tv_kl,
                "tv_knn_mean_kl_nats": knn_tv_kl,
                "kl_factor_inverse_temperature": factor_kl_beta,
                "kl_knn_inverse_temperature": knn_kl_beta,
                "kl_factor_actual_mean_kl_nats": factor_actual_kl,
                "kl_knn_actual_mean_kl_nats": knn_actual_kl,
                "kl_factor_actual_tv": factor_kl_tv,
                "kl_knn_actual_tv": knn_kl_tv,
                "kl_factor_marginal_error": factor_marginal_error,
                "kl_knn_marginal_error": knn_marginal_error,
                "kl_delta_matching_knn_minus_factor": float(paired.mean()),
            }
        )
        contributions.append(paired)
    return rows, contributions


def _evaluate_common_exposure_controls(
    factor_score: np.ndarray,
    knn_score: np.ndarray,
    utility: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_tvs: Sequence[float],
    include_positive_control: bool,
    common_marginal_weight: float = 0.5,
) -> tuple[list[dict[str, float | int]], list[np.ndarray]]:
    """Evaluate the controls and keep paired user-level value differences."""

    factor = np.asarray(factor_score, dtype=np.float64)
    knn = np.asarray(knn_score, dtype=np.float64)
    outcome = np.asarray(utility, dtype=np.float64)
    fold_ids = np.asarray(evaluation_fold_ids, dtype=np.int64)
    if factor.shape != knn.shape or factor.shape != outcome.shape:
        raise ValueError("factor scores, kNN scores, and outcomes must align")
    if fold_ids.shape != (outcome.shape[0],):
        raise ValueError("evaluation fold IDs must align with outcomes")
    targets = tuple(sorted({float(value) for value in target_tvs}))
    if not targets:
        raise ValueError("at least one target TV is required")
    weight = float(common_marginal_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("common_marginal_weight must lie between zero and one")

    rows: list[dict[str, float | int]] = []
    user_contributions: list[np.ndarray] = []
    for target_tv in targets:
        factor_beta = calibrate_inverse_temperature_to_personalization_tv(
            factor,
            evaluation_fold_ids=fold_ids,
            target_tv=target_tv,
        )
        knn_beta = calibrate_inverse_temperature_to_personalization_tv(
            knn,
            evaluation_fold_ids=fold_ids,
            target_tv=target_tv,
        )
        factor_policy = _softmax(factor, factor_beta)
        knn_policy = _softmax(knn, knn_beta)
        factor_marginal = fold_conditioned_marginal(factor_policy, fold_ids)
        knn_marginal = fold_conditioned_marginal(knn_policy, fold_ids)
        common_marginal = (1.0 - weight) * factor_marginal + weight * knn_marginal

        (
            common_factor_beta,
            common_factor_policy,
            common_factor_tv,
            factor_marginal_error,
            factor_projection_iterations,
        ) = calibrate_common_exposure_policy(
            factor,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
            target_tv=target_tv,
            initial_upper=factor_beta,
        )
        (
            common_knn_beta,
            common_knn_policy,
            common_knn_tv,
            knn_marginal_error,
            knn_projection_iterations,
        ) = calibrate_common_exposure_policy(
            knn,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
            target_tv=target_tv,
            initial_upper=knn_beta,
        )
        factor_metrics = _evaluate_frozen_score(
            factor,
            outcome,
            evaluation_fold_ids=fold_ids,
            inverse_temperature=factor_beta,
        )
        knn_metrics = _evaluate_frozen_score(
            knn,
            outcome,
            evaluation_fold_ids=fold_ids,
            inverse_temperature=knn_beta,
        )
        cross_fitted_shared_value = float(
            factor_metrics["cross_fitted_shared_action_value"]
        )
        if not np.isclose(
            cross_fitted_shared_value,
            float(knn_metrics["cross_fitted_shared_action_value"]),
            atol=1e-12,
        ):
            raise ValueError("shared-action comparator differs across policy families")
        common_reference_rows = common_marginal[fold_ids]
        common_reference_value = float(
            np.mean(np.sum(common_reference_rows * outcome, axis=1))
        )
        common_factor_value = float(
            np.mean(np.sum(common_factor_policy * outcome, axis=1))
        )
        common_knn_value = float(
            np.mean(np.sum(common_knn_policy * outcome, axis=1))
        )
        common_policy_pair_tv = float(
            np.mean(
                0.5
                * np.abs(common_knn_policy - common_factor_policy).sum(axis=1)
            )
        )
        common_policy_pair_maximum_tv = float(
            np.max(
                0.5
                * np.abs(common_knn_policy - common_factor_policy).sum(axis=1)
            )
        )
        factor_tv_p10, factor_tv_p50, factor_tv_p90 = policy_tv_quantiles(
            common_factor_policy,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
        )
        knn_tv_p10, knn_tv_p50, knn_tv_p90 = policy_tv_quantiles(
            common_knn_policy,
            evaluation_fold_ids=fold_ids,
            target_marginal=common_marginal,
        )

        positive_control_fields: dict[str, float | int] = {}
        if include_positive_control:
            own_positive_control_beta = (
                calibrate_inverse_temperature_to_personalization_tv(
                outcome,
                evaluation_fold_ids=fold_ids,
                target_tv=target_tv,
            )
            )
            (
                common_positive_control_beta,
                common_positive_control_policy,
                common_positive_control_tv,
                common_positive_control_marginal_error,
                common_positive_control_projection_iterations,
            ) = calibrate_common_exposure_policy(
                outcome,
                evaluation_fold_ids=fold_ids,
                target_marginal=common_marginal,
                target_tv=target_tv,
                initial_upper=own_positive_control_beta,
            )
            common_positive_control_value = float(
                np.mean(
                    np.sum(common_positive_control_policy * outcome, axis=1)
                )
            )
            common_positive_control_matching = (
                common_positive_control_value - common_reference_value
            )
            positive_control_fields = {
                "common_positive_control_inverse_temperature": (
                    common_positive_control_beta
                ),
                "common_positive_control_actual_tv": common_positive_control_tv,
                "common_positive_control_marginal_error": (
                    common_positive_control_marginal_error
                ),
                "common_positive_control_projection_iterations": (
                    common_positive_control_projection_iterations
                ),
                "common_positive_control_policy_value": (
                    common_positive_control_value
                ),
                "common_positive_control_matching_value": (
                    common_positive_control_matching
                ),
                "common_factor_fraction_of_positive_control": float(
                    (common_factor_value - common_reference_value)
                    / common_positive_control_matching
                ),
                "common_knn_fraction_of_positive_control": float(
                    (common_knn_value - common_reference_value)
                    / common_positive_control_matching
                ),
                "absolute_common_delta_fraction_of_positive_control": float(
                    abs(common_knn_value - common_factor_value)
                    / common_positive_control_matching
                ),
            }
        original_matching_delta = float(
            knn_metrics["matching_value"] - factor_metrics["matching_value"]
        )
        common_matching_delta = common_knn_value - common_factor_value
        paired_user_contribution = np.sum(
            (common_knn_policy - common_factor_policy) * outcome,
            axis=1,
        )
        if not np.isclose(
            paired_user_contribution.mean(),
            common_matching_delta,
            atol=1e-12,
        ):
            raise ValueError("paired user contributions do not recover the value gap")
        user_contributions.append(paired_user_contribution)
        rows.append(
            {
                "target_tv": target_tv,
                "common_marginal_weight": weight,
                "own_reference_marginal_tv": float(
                    np.mean(
                        0.5
                        * np.abs(knn_marginal - factor_marginal).sum(axis=1)
                    )
                ),
                "own_factor_inverse_temperature": factor_beta,
                "own_knn_inverse_temperature": knn_beta,
                "own_factor_matching_value": float(
                    factor_metrics["matching_value"]
                ),
                "own_knn_matching_value": float(knn_metrics["matching_value"]),
                "cross_fitted_shared_action_value": cross_fitted_shared_value,
                "cross_fitted_shared_unique_actions": int(
                    factor_metrics["cross_fitted_shared_unique_actions"]
                ),
                "own_factor_actionable_gain_vs_cross_fitted_shared": float(
                    factor_metrics["actionable_gain_vs_cross_fitted_shared"]
                ),
                "own_knn_actionable_gain_vs_cross_fitted_shared": float(
                    knn_metrics["actionable_gain_vs_cross_fitted_shared"]
                ),
                "own_delta_matching_knn_minus_factor": original_matching_delta,
                "common_factor_inverse_temperature": common_factor_beta,
                "common_knn_inverse_temperature": common_knn_beta,
                "common_factor_actual_tv": common_factor_tv,
                "common_knn_actual_tv": common_knn_tv,
                "common_factor_marginal_error": factor_marginal_error,
                "common_knn_marginal_error": knn_marginal_error,
                "common_factor_projection_iterations": (
                    factor_projection_iterations
                ),
                "common_knn_projection_iterations": knn_projection_iterations,
                "common_reference_value": common_reference_value,
                "common_factor_policy_value": common_factor_value,
                "common_knn_policy_value": common_knn_value,
                "common_factor_matching_value": (
                    common_factor_value - common_reference_value
                ),
                "common_knn_matching_value": (
                    common_knn_value - common_reference_value
                ),
                "common_factor_actionable_gain_vs_cross_fitted_shared": (
                    common_factor_value - cross_fitted_shared_value
                ),
                "common_knn_actionable_gain_vs_cross_fitted_shared": (
                    common_knn_value - cross_fitted_shared_value
                ),
                "common_delta_matching_knn_minus_factor": (
                    common_matching_delta
                ),
                "common_delta_exposure_knn_minus_factor": 0.0,
                "common_policy_pair_tv": common_policy_pair_tv,
                "common_policy_pair_maximum_tv": (
                    common_policy_pair_maximum_tv
                ),
                "common_factor_tv_p10": factor_tv_p10,
                "common_factor_tv_p50": factor_tv_p50,
                "common_factor_tv_p90": factor_tv_p90,
                "common_knn_tv_p10": knn_tv_p10,
                "common_knn_tv_p50": knn_tv_p50,
                "common_knn_tv_p90": knn_tv_p90,
                **positive_control_fields,
            }
        )
    return rows, user_contributions


def evaluate_common_exposure_controls(
    factor_score: np.ndarray,
    knn_score: np.ndarray,
    utility: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_tvs: Sequence[float],
    include_positive_control: bool = True,
    common_marginal_weight: float = 0.5,
) -> list[dict[str, float | int]]:
    """Evaluate own-reference, common-exposure, and positive-control comparisons."""

    rows, _ = _evaluate_common_exposure_controls(
        factor_score,
        knn_score,
        utility,
        evaluation_fold_ids=evaluation_fold_ids,
        target_tvs=target_tvs,
        include_positive_control=include_positive_control,
        common_marginal_weight=common_marginal_weight,
    )
    return rows


def evaluate_common_exposure_details(
    factor_score: np.ndarray,
    knn_score: np.ndarray,
    utility: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_tvs: Sequence[float],
    common_marginal_weight: float = 0.5,
) -> tuple[list[dict[str, float | int]], list[np.ndarray]]:
    """Also return each user's paired common-exposure value difference."""

    return _evaluate_common_exposure_controls(
        factor_score,
        knn_score,
        utility,
        evaluation_fold_ids=evaluation_fold_ids,
        target_tvs=target_tvs,
        include_positive_control=True,
        common_marginal_weight=common_marginal_weight,
    )


def paired_user_bootstrap_interval(
    contributions_by_split: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
    chunk_size: int = 256,
) -> tuple[float, float, float]:
    """Bootstrap users after averaging each user's paired gap over splits."""

    values = np.asarray(contributions_by_split, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("contributions must have shape (splits, users)")
    if not np.isfinite(values).all():
        raise ValueError("contributions must be finite")
    if replicates < 1 or chunk_size < 1:
        raise ValueError("bootstrap controls must be positive")

    per_user = values.mean(axis=0)
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        sampled = rng.integers(
            0,
            per_user.size,
            size=(stop - start, per_user.size),
        )
        bootstrap_means[start:stop] = per_user[sampled].mean(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(per_user.mean()), float(low), float(high)


def paired_user_bootstrap_path(
    contributions_by_weight_split: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
    chunk_size: int = 256,
) -> dict[str, np.ndarray | float]:
    """Bootstrap one user sample across a path of common marginals.

    Split-specific contributions are averaged first. The same resampled users
    are then used at every path point. The simultaneous band uses the largest
    absolute centered bootstrap deviation over the full path.
    """

    values = np.asarray(contributions_by_weight_split, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1 or values.shape[2] < 2:
        raise ValueError("contributions must have shape (weights, splits, users)")
    if not np.isfinite(values).all():
        raise ValueError("contributions must be finite")
    if replicates < 1 or chunk_size < 1:
        raise ValueError("bootstrap controls must be positive")

    per_weight_user = values.mean(axis=1)
    point = per_weight_user.mean(axis=1)
    bootstrap_means = np.empty(
        (replicates, per_weight_user.shape[0]),
        dtype=np.float64,
    )
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        sampled = rng.integers(
            0,
            per_weight_user.shape[1],
            size=(stop - start, per_weight_user.shape[1]),
        )
        bootstrap_means[start:stop] = per_weight_user[:, sampled].mean(
            axis=2
        ).T
    pointwise_low, pointwise_high = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
        axis=0,
    )
    maximum_deviation = np.max(
        np.abs(bootstrap_means - point[None, :]),
        axis=1,
    )
    simultaneous_critical = float(np.percentile(maximum_deviation, 95.0))
    return {
        "point": point,
        "pointwise_low": pointwise_low,
        "pointwise_high": pointwise_high,
        "simultaneous_low": point - simultaneous_critical,
        "simultaneous_high": point + simultaneous_critical,
        "simultaneous_critical": simultaneous_critical,
    }


def _reference_path_status(
    point: np.ndarray,
    simultaneous_low: np.ndarray,
    simultaneous_high: np.ndarray,
) -> str:
    if float(np.max(simultaneous_high)) < 0.0:
        return "reference_robust_negative"
    if float(np.min(simultaneous_low)) > 0.0:
        return "reference_robust_positive"
    if float(np.min(point)) <= 0.0 <= float(np.max(point)):
        return "point_estimate_changes_sign"
    return "unresolved_over_reference_path"


def summarize_control_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["outcome"]), float(row["target_tv"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for (outcome, target_tv), group in sorted(grouped.items()):
        def values(name: str) -> np.ndarray:
            return np.asarray([float(row[name]) for row in group])

        own_delta = values("own_delta_matching_knn_minus_factor")
        common_delta = values("common_delta_matching_knn_minus_factor")
        summaries.append(
            {
                "outcome": outcome,
                "target_tv": target_tv,
                "splits": len(group),
                "own_reference_marginal_tv_mean": float(
                    values("own_reference_marginal_tv").mean()
                ),
                "own_delta_matching_mean": float(own_delta.mean()),
                "own_delta_matching_min": float(own_delta.min()),
                "own_delta_matching_max": float(own_delta.max()),
                "own_negative_splits": int(np.sum(own_delta < 0.0)),
                "common_delta_matching_mean": float(common_delta.mean()),
                "common_delta_matching_min": float(common_delta.min()),
                "common_delta_matching_max": float(common_delta.max()),
                "common_negative_splits": int(np.sum(common_delta < 0.0)),
                "common_positive_control_matching_value_mean": float(
                    values("common_positive_control_matching_value").mean()
                ),
                "common_factor_fraction_of_positive_control_mean": float(
                    values("common_factor_fraction_of_positive_control").mean()
                ),
                "common_knn_fraction_of_positive_control_mean": float(
                    values("common_knn_fraction_of_positive_control").mean()
                ),
                "cross_fitted_shared_action_value_mean": float(
                    values("cross_fitted_shared_action_value").mean()
                ),
                "common_factor_actionable_upper_bound_mean": float(
                    values(
                        "common_factor_actionable_gain_vs_cross_fitted_shared"
                    ).mean()
                ),
                "common_knn_actionable_upper_bound_mean": float(
                    values(
                        "common_knn_actionable_gain_vs_cross_fitted_shared"
                    ).mean()
                ),
                "common_factor_actionable_upper_bound_max": float(
                    values(
                        "common_factor_actionable_gain_vs_cross_fitted_shared"
                    ).max()
                ),
                "common_knn_actionable_upper_bound_max": float(
                    values(
                        "common_knn_actionable_gain_vs_cross_fitted_shared"
                    ).max()
                ),
                "absolute_common_delta_fraction_of_positive_control_mean": float(
                    values(
                        "absolute_common_delta_fraction_of_positive_control"
                    ).mean()
                ),
                "maximum_common_marginal_error": float(
                    max(
                        values("common_factor_marginal_error").max(),
                        values("common_knn_marginal_error").max(),
                    )
                ),
                "maximum_common_tv_error": float(
                    max(
                        np.abs(
                            values("common_factor_actual_tv") - target_tv
                        ).max(),
                        np.abs(values("common_knn_actual_tv") - target_tv).max(),
                        np.abs(
                            values("common_positive_control_actual_tv")
                            - target_tv
                        ).max(),
                    )
                ),
                "maximum_common_positive_control_marginal_error": float(
                    values("common_positive_control_marginal_error").max()
                ),
                "common_policy_pair_tv_mean": float(
                    values("common_policy_pair_tv").mean()
                ),
                "shared_quality_own_delta_matching_mean": float(
                    values("shared_quality_own_delta_matching").mean()
                ),
                "shared_quality_common_delta_matching_max_abs": float(
                    np.abs(
                        values("shared_quality_common_delta_matching")
                    ).max()
                ),
                "shared_quality_common_policy_pair_tv_max": float(
                    values("shared_quality_common_policy_pair_tv").max()
                ),
            }
        )
    return summaries


def summarize_fixed_strength_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize each outcome's fixed-marginal strength path."""

    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["outcome"]), float(row["target_tv"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for (outcome, target_tv), group in sorted(grouped.items()):
        def values(name: str) -> np.ndarray:
            return np.asarray([float(row[name]) for row in group])

        delta = values("delta_matching_knn_minus_factor")
        summaries.append(
            {
                "outcome": outcome,
                "target_tv": target_tv,
                "splits": len(group),
                "delta_matching_mean": float(delta.mean()),
                "delta_matching_min": float(delta.min()),
                "delta_matching_max": float(delta.max()),
                "negative_splits": int(np.sum(delta < 0.0)),
                "factor_matching_value_mean": float(
                    values("factor_matching_value").mean()
                ),
                "knn_matching_value_mean": float(
                    values("knn_matching_value").mean()
                ),
                "factor_tv_p10_mean": float(values("factor_tv_p10").mean()),
                "factor_tv_p50_mean": float(values("factor_tv_p50").mean()),
                "factor_tv_p90_mean": float(values("factor_tv_p90").mean()),
                "knn_tv_p10_mean": float(values("knn_tv_p10").mean()),
                "knn_tv_p50_mean": float(values("knn_tv_p50").mean()),
                "knn_tv_p90_mean": float(values("knn_tv_p90").mean()),
                "maximum_marginal_error": float(
                    max(
                        values("factor_marginal_error").max(),
                        values("knn_marginal_error").max(),
                    )
                ),
                "maximum_tv_error": float(
                    max(
                        np.abs(values("factor_actual_tv") - target_tv).max(),
                        np.abs(values("knn_actual_tv") - target_tv).max(),
                    )
                ),
            }
        )
    return summaries


def summarize_kl_sensitivity_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize equal-information comparisons across fixed splits."""

    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["outcome"]), float(row["source_target_tv"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for (outcome, target_tv), group in sorted(grouped.items()):
        def values(name: str) -> np.ndarray:
            return np.asarray([float(row[name]) for row in group])

        delta = values("kl_delta_matching_knn_minus_factor")
        summaries.append(
            {
                "outcome": outcome,
                "source_target_tv": target_tv,
                "splits": len(group),
                "target_mean_kl_nats_mean": float(
                    values("target_mean_kl_nats").mean()
                ),
                "target_mean_kl_nats_min": float(
                    values("target_mean_kl_nats").min()
                ),
                "target_mean_kl_nats_max": float(
                    values("target_mean_kl_nats").max()
                ),
                "kl_delta_matching_mean": float(delta.mean()),
                "kl_delta_matching_min": float(delta.min()),
                "kl_delta_matching_max": float(delta.max()),
                "kl_negative_splits": int(np.sum(delta < 0.0)),
                "kl_factor_actual_tv_mean": float(
                    values("kl_factor_actual_tv").mean()
                ),
                "kl_knn_actual_tv_mean": float(
                    values("kl_knn_actual_tv").mean()
                ),
                "maximum_kl_error": float(
                    max(
                        np.abs(
                            values("kl_factor_actual_mean_kl_nats")
                            - values("target_mean_kl_nats")
                        ).max(),
                        np.abs(
                            values("kl_knn_actual_mean_kl_nats")
                            - values("target_mean_kl_nats")
                        ).max(),
                    )
                ),
                "maximum_marginal_error": float(
                    max(
                        values("kl_factor_marginal_error").max(),
                        values("kl_knn_marginal_error").max(),
                    )
                ),
            }
        )
    return summaries


def run_common_exposure_factor_sensitivity(
    utility: np.ndarray,
    *,
    outcome: str,
    split_seeds: Sequence[int],
    user_folds: int,
    history_action_count: int,
    rank_grid: Sequence[int],
    ridge_penalty_grid: Sequence[float],
    primary_rank: int,
    primary_ridge_penalty: float,
    neighbor_fraction_grid: Sequence[float],
    inner_validation_fraction: float,
    target_tv: float,
) -> list[dict[str, object]]:
    """Vary MF rank and ridge while keeping the primary exposure target fixed."""

    ranks = tuple(sorted({int(value) for value in rank_grid}))
    ridges = tuple(sorted({float(value) for value in ridge_penalty_grid}))
    seeds = tuple(int(value) for value in split_seeds)
    if not outcome:
        raise ValueError("outcome must not be empty")
    if not ranks or ranks[0] <= 0:
        raise ValueError("rank_grid must contain positive integers")
    if not ridges or ridges[0] <= 0.0 or not np.isfinite(ridges).all():
        raise ValueError("ridge_penalty_grid must contain finite positive values")
    if primary_rank not in ranks or float(primary_ridge_penalty) not in ridges:
        raise ValueError("the sensitivity grid must contain the primary MF setup")
    if not seeds or not 0.0 < target_tv < 1.0:
        raise ValueError("split seeds and a target TV in (0, 1) are required")

    rows: list[dict[str, object]] = []
    for split_seed in seeds:
        primary_factor = construct_cross_fitted_learned_policy_split(
            utility,
            user_folds=user_folds,
            history_action_count=history_action_count,
            rank=primary_rank,
            ridge_penalty=primary_ridge_penalty,
            split_seed=split_seed,
        )
        knn = construct_cross_fitted_knn_policy_split(
            utility,
            user_folds=user_folds,
            history_action_count=history_action_count,
            neighbor_fraction_grid=neighbor_fraction_grid,
            inner_validation_fraction=inner_validation_fraction,
            split_seed=split_seed,
        )
        if not np.array_equal(
            primary_factor.evaluation_user_indices,
            knn.evaluation_user_indices,
        ) or not np.array_equal(
            primary_factor.candidate_action_indices,
            knn.candidate_action_indices,
        ):
            raise ValueError("primary MF and kNN splits are not aligned")
        primary_marginal, _, knn_upper = fixed_common_marginal_from_scores(
            primary_factor.candidate_interaction_score,
            knn.candidate_score,
            evaluation_fold_ids=primary_factor.evaluation_fold_ids,
            anchor_tv=target_tv,
        )

        for rank in ranks:
            for ridge_penalty in ridges:
                if (
                    rank == primary_rank
                    and ridge_penalty == float(primary_ridge_penalty)
                ):
                    factor = primary_factor
                else:
                    factor = construct_cross_fitted_learned_policy_split(
                        utility,
                        user_folds=user_folds,
                        history_action_count=history_action_count,
                        rank=rank,
                        ridge_penalty=ridge_penalty,
                        split_seed=split_seed,
                    )
                if not np.array_equal(
                    factor.evaluation_user_indices,
                    primary_factor.evaluation_user_indices,
                ) or not np.array_equal(
                    factor.candidate_action_indices,
                    primary_factor.candidate_action_indices,
                ):
                    raise ValueError("MF sensitivity splits are not aligned")
                factor_upper = (
                    calibrate_inverse_temperature_to_personalization_tv(
                        factor.candidate_interaction_score,
                        evaluation_fold_ids=factor.evaluation_fold_ids,
                        target_tv=target_tv,
                    )
                )
                comparison_rows, _ = evaluate_fixed_common_marginal_strength_path(
                    factor.candidate_interaction_score,
                    knn.candidate_score,
                    factor.candidate_utility,
                    evaluation_fold_ids=factor.evaluation_fold_ids,
                    target_marginal=primary_marginal,
                    target_tvs=(target_tv,),
                    initial_factor_upper=max(factor_upper, 1.0),
                    initial_knn_upper=max(knn_upper, 1.0),
                )
                comparison = comparison_rows[0]
                rows.append(
                    {
                        "outcome": outcome,
                        "split_seed": split_seed,
                        "factor_rank": rank,
                        "factor_ridge_penalty": ridge_penalty,
                        "is_primary_factor_configuration": bool(
                            rank == primary_rank
                            and ridge_penalty == float(primary_ridge_penalty)
                        ),
                        "target_tv": target_tv,
                        "common_marginal_rule": (
                            "primary_MF_kNN_midpoint_fixed_within_split"
                        ),
                        "factor_history_rmse": factor.history_rmse,
                        **comparison,
                    }
                )
    return rows


def run_common_exposure_knn_sensitivity(
    utility: np.ndarray,
    *,
    outcome: str,
    split_seeds: Sequence[int],
    user_folds: int,
    history_action_count: int,
    factor_rank: int,
    factor_ridge_penalty: float,
    neighbor_fraction_grid: Sequence[float],
    inner_validation_fraction: float,
    target_tv: float,
) -> list[dict[str, object]]:
    """Vary fixed kNN neighborhoods while keeping the primary exposure fixed."""

    fractions = tuple(sorted({float(value) for value in neighbor_fraction_grid}))
    seeds = tuple(int(value) for value in split_seeds)
    if not fractions or fractions[0] <= 0.0 or fractions[-1] > 1.0:
        raise ValueError("neighbor fractions must lie in (0, 1]")
    if not seeds or not 0.0 < target_tv < 1.0:
        raise ValueError("split seeds and a target TV in (0, 1) are required")

    rows: list[dict[str, object]] = []
    for split_seed in seeds:
        factor = construct_cross_fitted_learned_policy_split(
            utility,
            user_folds=user_folds,
            history_action_count=history_action_count,
            rank=factor_rank,
            ridge_penalty=factor_ridge_penalty,
            split_seed=split_seed,
        )
        knn = construct_cross_fitted_knn_policy_split(
            utility,
            user_folds=user_folds,
            history_action_count=history_action_count,
            neighbor_fraction_grid=fractions,
            inner_validation_fraction=inner_validation_fraction,
            split_seed=split_seed,
        )
        if not np.array_equal(
            factor.evaluation_user_indices,
            knn.evaluation_user_indices,
        ) or not np.array_equal(
            factor.candidate_action_indices,
            knn.candidate_action_indices,
        ):
            raise ValueError("MF and kNN sensitivity splits are not aligned")
        primary_marginal, factor_upper, _ = fixed_common_marginal_from_scores(
            factor.candidate_interaction_score,
            knn.candidate_score,
            evaluation_fold_ids=factor.evaluation_fold_ids,
            anchor_tv=target_tv,
        )

        for neighbor_fraction in fractions:
            knn_score = knn.candidate_scores_by_neighbor_fraction[
                neighbor_fraction
            ]
            knn_upper = calibrate_inverse_temperature_to_personalization_tv(
                knn_score,
                evaluation_fold_ids=factor.evaluation_fold_ids,
                target_tv=target_tv,
            )
            comparison_rows, _ = evaluate_fixed_common_marginal_strength_path(
                factor.candidate_interaction_score,
                knn_score,
                factor.candidate_utility,
                evaluation_fold_ids=factor.evaluation_fold_ids,
                target_marginal=primary_marginal,
                target_tvs=(target_tv,),
                initial_factor_upper=max(factor_upper, 1.0),
                initial_knn_upper=max(knn_upper, 1.0),
            )
            neighbor_counts = [
                record.outer_neighbor_count
                for record in knn.selection_records
                if record.neighbor_fraction == neighbor_fraction
            ]
            rows.append(
                {
                    "outcome": outcome,
                    "split_seed": split_seed,
                    "neighbor_fraction": neighbor_fraction,
                    "is_inner_selected_configuration": bool(
                        np.all(knn.selected_neighbor_fractions == neighbor_fraction)
                    ),
                    "outer_neighbor_count_min": min(neighbor_counts),
                    "outer_neighbor_count_max": max(neighbor_counts),
                    "factor_rank": factor_rank,
                    "factor_ridge_penalty": factor_ridge_penalty,
                    "target_tv": target_tv,
                    "common_marginal_rule": (
                        "primary_MF_nested-selected-kNN_midpoint_fixed_within_split"
                    ),
                    **comparison_rows[0],
                }
            )
    return rows


def summarize_common_exposure_factor_sensitivity(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize the common-exposure rank/ridge grid across splits."""

    grouped: dict[tuple[int, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["factor_rank"]), float(row["factor_ridge_penalty"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for (rank, ridge_penalty), group in sorted(grouped.items()):
        delta = np.asarray(
            [float(row["delta_matching_knn_minus_factor"]) for row in group]
        )
        summaries.append(
            {
                "outcome": str(group[0]["outcome"]),
                "factor_rank": rank,
                "factor_ridge_penalty": ridge_penalty,
                "is_primary_factor_configuration": bool(
                    group[0]["is_primary_factor_configuration"]
                ),
                "target_tv": float(group[0]["target_tv"]),
                "splits": len(group),
                "delta_matching_mean": float(delta.mean()),
                "delta_matching_min": float(delta.min()),
                "delta_matching_max": float(delta.max()),
                "negative_splits": int(np.sum(delta < 0.0)),
                "positive_splits": int(np.sum(delta > 0.0)),
                "factor_matching_value_mean": float(
                    np.mean(
                        [float(row["factor_matching_value"]) for row in group]
                    )
                ),
                "knn_matching_value_mean": float(
                    np.mean(
                        [float(row["knn_matching_value"]) for row in group]
                    )
                ),
                "maximum_marginal_error": float(
                    max(
                        max(float(row["factor_marginal_error"]) for row in group),
                        max(float(row["knn_marginal_error"]) for row in group),
                    )
                ),
                "maximum_tv_error": float(
                    max(
                        max(
                            abs(
                                float(row["factor_actual_tv"])
                                - float(row["target_tv"])
                            )
                            for row in group
                        ),
                        max(
                            abs(
                                float(row["knn_actual_tv"])
                                - float(row["target_tv"])
                            )
                            for row in group
                        ),
                    )
                ),
            }
        )
    return summaries


def summarize_common_exposure_knn_sensitivity(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize the fixed-neighborhood kNN grid across splits."""

    grouped: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(float(row["neighbor_fraction"]), []).append(row)
    summaries: list[dict[str, object]] = []
    for neighbor_fraction, group in sorted(grouped.items()):
        delta = np.asarray(
            [float(row["delta_matching_knn_minus_factor"]) for row in group]
        )
        target_tv = float(group[0]["target_tv"])
        summaries.append(
            {
                "outcome": str(group[0]["outcome"]),
                "neighbor_fraction": neighbor_fraction,
                "is_inner_selected_configuration": bool(
                    all(
                        bool(row["is_inner_selected_configuration"])
                        for row in group
                    )
                ),
                "outer_neighbor_count_min": min(
                    int(row["outer_neighbor_count_min"]) for row in group
                ),
                "outer_neighbor_count_max": max(
                    int(row["outer_neighbor_count_max"]) for row in group
                ),
                "factor_rank": int(group[0]["factor_rank"]),
                "factor_ridge_penalty": float(group[0]["factor_ridge_penalty"]),
                "target_tv": target_tv,
                "splits": len(group),
                "delta_matching_mean": float(delta.mean()),
                "delta_matching_min": float(delta.min()),
                "delta_matching_max": float(delta.max()),
                "negative_splits": int(np.sum(delta < 0.0)),
                "positive_splits": int(np.sum(delta > 0.0)),
                "factor_matching_value_mean": float(
                    np.mean(
                        [float(row["factor_matching_value"]) for row in group]
                    )
                ),
                "knn_matching_value_mean": float(
                    np.mean([float(row["knn_matching_value"]) for row in group])
                ),
                "maximum_marginal_error": float(
                    max(
                        max(float(row["factor_marginal_error"]) for row in group),
                        max(float(row["knn_marginal_error"]) for row in group),
                    )
                ),
                "maximum_tv_error": float(
                    max(
                        max(
                            abs(float(row["factor_actual_tv"]) - target_tv)
                            for row in group
                        ),
                        max(
                            abs(float(row["knn_actual_tv"]) - target_tv)
                            for row in group
                        ),
                    )
                ),
            }
        )
    return summaries


def _parse_numbers(
    raw: str,
    *,
    cast: type[int] | type[float],
) -> tuple[int, ...] | tuple[float, ...]:
    values = tuple(cast(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        raise ValueError("numeric grid must not be empty")
    return values


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-reference-path-csv", type=Path, required=True)
    parser.add_argument(
        "--output-reference-path-summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-strength-path-csv", type=Path, required=True)
    parser.add_argument(
        "--output-strength-path-summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-kl-sensitivity-csv", type=Path, required=True)
    parser.add_argument(
        "--output-kl-sensitivity-summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-factor-sensitivity-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-factor-sensitivity-summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-knn-sensitivity-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-knn-sensitivity-summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--split-seeds", default="11,23,37,41,53,67,79,83,97,101")
    parser.add_argument("--user-folds", type=int, default=5)
    parser.add_argument("--history-action-count", type=int, default=750)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    parser.add_argument("--factor-rank-sensitivity-grid", default="8,16,32")
    parser.add_argument("--factor-ridge-sensitivity-grid", default="1,10,100")
    parser.add_argument("--factor-sensitivity-tv", type=float, default=0.04)
    parser.add_argument("--neighbor-fraction-grid", default="0.01,0.05,0.2,0.5,1")
    parser.add_argument("--inner-validation-fraction", type=float, default=0.2)
    parser.add_argument("--target-tv-grid", default="0.02,0.04,0.07")
    parser.add_argument(
        "--strength-path-tv-grid",
        default="0.02,0.04,0.07,0.1,0.2,0.3,0.5",
    )
    parser.add_argument("--strength-path-anchor-tv", type=float, default=0.04)
    parser.add_argument(
        "--reference-marginal-weight-grid",
        default="0,0.25,0.5,0.75,1",
    )
    parser.add_argument("--user-bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--user-bootstrap-seed", type=int, default=20260815)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    preferences = load_preference_matrix(args.csv)
    common_actions = np.all(np.isfinite(preferences.utility), axis=0)
    outcome_surfaces = {
        "capped_watch_ratio": preferences.utility[:, common_actions],
        "complete_watch": (
            preferences.ranking_score[:, common_actions] >= 1.0
        ).astype(np.float64),
    }
    split_seeds = _parse_numbers(args.split_seeds, cast=int)
    neighbor_fractions = _parse_numbers(args.neighbor_fraction_grid, cast=float)
    factor_sensitivity_ranks = _parse_numbers(
        args.factor_rank_sensitivity_grid,
        cast=int,
    )
    factor_sensitivity_ridges = _parse_numbers(
        args.factor_ridge_sensitivity_grid,
        cast=float,
    )
    target_tvs = _parse_numbers(args.target_tv_grid, cast=float)
    strength_path_tvs = _parse_numbers(args.strength_path_tv_grid, cast=float)
    if not 0.0 < args.strength_path_anchor_tv < 1.0:
        raise ValueError("strength_path_anchor_tv must lie between zero and one")
    reference_weights = tuple(
        sorted(
            {
                float(value)
                for value in _parse_numbers(
                    args.reference_marginal_weight_grid,
                    cast=float,
                )
            }
        )
    )
    if any(not 0.0 <= weight <= 1.0 for weight in reference_weights):
        raise ValueError("reference marginal weights must lie between zero and one")
    if 0.5 not in reference_weights:
        raise ValueError("reference marginal weight grid must contain 0.5")
    if args.user_bootstrap_replicates < 1:
        raise ValueError("user_bootstrap_replicates must be positive")
    rows: list[dict[str, object]] = []
    user_contribution_groups: dict[tuple[str, float], list[np.ndarray]] = {}
    reference_path_rows: list[dict[str, object]] = []
    reference_path_contributions: dict[
        tuple[str, float, float], list[np.ndarray]
    ] = {}
    strength_path_rows: list[dict[str, object]] = []
    strength_path_contributions: dict[
        tuple[str, float], list[np.ndarray]
    ] = {}
    kl_sensitivity_rows: list[dict[str, object]] = []
    kl_sensitivity_contributions: dict[
        tuple[str, float], list[np.ndarray]
    ] = {}
    for outcome, surface in outcome_surfaces.items():
        for split_seed in split_seeds:
            factor_split = construct_cross_fitted_learned_policy_split(
                surface,
                user_folds=args.user_folds,
                history_action_count=args.history_action_count,
                rank=args.rank,
                ridge_penalty=args.ridge_penalty,
                split_seed=int(split_seed),
            )
            knn_split = construct_cross_fitted_knn_policy_split(
                surface,
                user_folds=args.user_folds,
                history_action_count=args.history_action_count,
                neighbor_fraction_grid=neighbor_fractions,
                inner_validation_fraction=args.inner_validation_fraction,
                split_seed=int(split_seed),
            )
            (
                fixed_marginal,
                anchor_factor_beta,
                anchor_knn_beta,
            ) = fixed_common_marginal_from_scores(
                factor_split.candidate_interaction_score,
                knn_split.candidate_score,
                evaluation_fold_ids=factor_split.evaluation_fold_ids,
                anchor_tv=args.strength_path_anchor_tv,
            )
            fixed_rows, fixed_contributions = (
                evaluate_fixed_common_marginal_strength_path(
                    factor_split.candidate_interaction_score,
                    knn_split.candidate_score,
                    factor_split.candidate_utility,
                    evaluation_fold_ids=factor_split.evaluation_fold_ids,
                    target_marginal=fixed_marginal,
                    target_tvs=strength_path_tvs,
                    initial_factor_upper=max(anchor_factor_beta, 1.0),
                    initial_knn_upper=max(anchor_knn_beta, 1.0),
                )
            )
            for fixed_row, fixed_contribution in zip(
                fixed_rows,
                fixed_contributions,
            ):
                aligned = np.empty(surface.shape[0], dtype=np.float64)
                aligned[factor_split.evaluation_user_indices] = fixed_contribution
                fixed_tv = float(fixed_row["target_tv"])
                strength_path_contributions.setdefault(
                    (outcome, fixed_tv),
                    [],
                ).append(aligned)
                strength_path_rows.append(
                    {
                        "outcome": outcome,
                        "split_seed": int(split_seed),
                        "anchor_tv": args.strength_path_anchor_tv,
                        "common_marginal_rule": (
                            "midpoint_of_own_marginals_at_anchor_tv"
                        ),
                        **fixed_row,
                    }
                )
            control_rows, user_contributions = evaluate_common_exposure_details(
                factor_split.candidate_interaction_score,
                knn_split.candidate_score,
                factor_split.candidate_utility,
                evaluation_fold_ids=factor_split.evaluation_fold_ids,
                target_tvs=target_tvs,
            )
            split_kl_rows, split_kl_contributions = (
                evaluate_common_exposure_kl_sensitivity(
                    factor_split.candidate_interaction_score,
                    knn_split.candidate_score,
                    factor_split.candidate_utility,
                    evaluation_fold_ids=factor_split.evaluation_fold_ids,
                    target_tvs=target_tvs,
                )
            )
            for kl_row, kl_contribution in zip(
                split_kl_rows,
                split_kl_contributions,
            ):
                aligned = np.empty(surface.shape[0], dtype=np.float64)
                aligned[factor_split.evaluation_user_indices] = kl_contribution
                key = (outcome, float(kl_row["source_target_tv"]))
                kl_sensitivity_contributions.setdefault(key, []).append(aligned)
                kl_sensitivity_rows.append(
                    {
                        "outcome": outcome,
                        "split_seed": int(split_seed),
                        **kl_row,
                    }
                )
            shared_quality_rows = evaluate_common_exposure_controls(
                factor_split.candidate_interaction_score,
                factor_split.candidate_interaction_score
                + factor_split.candidate_item_quality,
                factor_split.candidate_utility,
                evaluation_fold_ids=factor_split.evaluation_fold_ids,
                target_tvs=target_tvs,
                include_positive_control=False,
            )
            if [row["target_tv"] for row in control_rows] != [
                row["target_tv"] for row in shared_quality_rows
            ]:
                raise ValueError("common-exposure control grids are not aligned")
            for row, user_contribution in zip(
                control_rows,
                user_contributions,
            ):
                aligned = np.empty(surface.shape[0], dtype=np.float64)
                aligned[factor_split.evaluation_user_indices] = user_contribution
                key = (outcome, float(row["target_tv"]))
                user_contribution_groups.setdefault(key, []).append(aligned)
                path_key = (outcome, float(row["target_tv"]), 0.5)
                reference_path_contributions.setdefault(path_key, []).append(
                    aligned
                )
                reference_path_rows.append(
                    {
                        "outcome": outcome,
                        "split_seed": int(split_seed),
                        "target_tv": row["target_tv"],
                        "common_marginal_weight": 0.5,
                        "common_delta_matching_knn_minus_factor": row[
                            "common_delta_matching_knn_minus_factor"
                        ],
                        "common_factor_actual_tv": row[
                            "common_factor_actual_tv"
                        ],
                        "common_knn_actual_tv": row["common_knn_actual_tv"],
                        "common_factor_marginal_error": row[
                            "common_factor_marginal_error"
                        ],
                        "common_knn_marginal_error": row[
                            "common_knn_marginal_error"
                        ],
                    }
                )
            for weight in reference_weights:
                if weight == 0.5:
                    continue
                path_rows, path_contributions = _evaluate_common_exposure_controls(
                    factor_split.candidate_interaction_score,
                    knn_split.candidate_score,
                    factor_split.candidate_utility,
                    evaluation_fold_ids=factor_split.evaluation_fold_ids,
                    target_tvs=target_tvs,
                    include_positive_control=False,
                    common_marginal_weight=weight,
                )
                for path_row, contribution in zip(
                    path_rows,
                    path_contributions,
                ):
                    aligned = np.empty(surface.shape[0], dtype=np.float64)
                    aligned[factor_split.evaluation_user_indices] = contribution
                    path_key = (
                        outcome,
                        float(path_row["target_tv"]),
                        weight,
                    )
                    reference_path_contributions.setdefault(
                        path_key,
                        [],
                    ).append(aligned)
                    reference_path_rows.append(
                        {
                            "outcome": outcome,
                            "split_seed": int(split_seed),
                            "target_tv": path_row["target_tv"],
                            "common_marginal_weight": weight,
                            "common_delta_matching_knn_minus_factor": path_row[
                                "common_delta_matching_knn_minus_factor"
                            ],
                            "common_factor_actual_tv": path_row[
                                "common_factor_actual_tv"
                            ],
                            "common_knn_actual_tv": path_row[
                                "common_knn_actual_tv"
                            ],
                            "common_factor_marginal_error": path_row[
                                "common_factor_marginal_error"
                            ],
                            "common_knn_marginal_error": path_row[
                                "common_knn_marginal_error"
                            ],
                        }
                    )
            rows.extend(
                {
                    "outcome": outcome,
                    "split_seed": int(split_seed),
                    "evaluation_users": int(surface.shape[0]),
                    "history_actions": args.history_action_count,
                    "candidate_actions": int(
                        factor_split.candidate_action_indices.size
                    ),
                    "reference_stratum": "outer_fold",
                    "common_marginal_rule": (
                        "mean_of_own_marginals_at_target_tv"
                    ),
                    "common_policy_rule": (
                        "minimum_kl_action_intercepts_then_tv_calibration"
                    ),
                    "calibration_signal": (
                        "policy_scores_only_no_candidate_outcomes"
                    ),
                    **row,
                    "shared_quality_own_reference_marginal_tv": (
                        shared_row["own_reference_marginal_tv"]
                    ),
                    "shared_quality_own_delta_matching": (
                        shared_row["own_delta_matching_knn_minus_factor"]
                    ),
                    "shared_quality_common_delta_matching": (
                        shared_row["common_delta_matching_knn_minus_factor"]
                    ),
                    "shared_quality_common_policy_pair_tv": (
                        shared_row["common_policy_pair_tv"]
                    ),
                }
                for row, shared_row in zip(control_rows, shared_quality_rows)
            )

    summaries = summarize_control_rows(rows)
    bootstrap_rng = np.random.default_rng(args.user_bootstrap_seed)
    for summary in summaries:
        key = (str(summary["outcome"]), float(summary["target_tv"]))
        contribution_matrix = np.vstack(user_contribution_groups[key])
        point, low, high = paired_user_bootstrap_interval(
            contribution_matrix,
            replicates=args.user_bootstrap_replicates,
            rng=bootstrap_rng,
        )
        if not np.isclose(
            point,
            float(summary["common_delta_matching_mean"]),
            atol=1e-12,
        ):
            raise ValueError("user-level and split-level means do not agree")
        summary.update(
            {
                "common_delta_user_bootstrap_low": low,
                "common_delta_user_bootstrap_high": high,
                "user_bootstrap_replicates": args.user_bootstrap_replicates,
                "user_bootstrap_seed": args.user_bootstrap_seed,
            }
        )
    reference_path_summaries: list[dict[str, object]] = []
    path_rng = np.random.default_rng(args.user_bootstrap_seed)
    for outcome in sorted(outcome_surfaces):
        for target_tv in target_tvs:
            contribution_cube = np.stack(
                [
                    np.vstack(
                        reference_path_contributions[
                            (outcome, float(target_tv), weight)
                        ]
                    )
                    for weight in reference_weights
                ],
                axis=0,
            )
            path_bootstrap = paired_user_bootstrap_path(
                contribution_cube,
                replicates=args.user_bootstrap_replicates,
                rng=path_rng,
            )
            point = np.asarray(path_bootstrap["point"])
            pointwise_low = np.asarray(path_bootstrap["pointwise_low"])
            pointwise_high = np.asarray(path_bootstrap["pointwise_high"])
            simultaneous_low = np.asarray(path_bootstrap["simultaneous_low"])
            simultaneous_high = np.asarray(path_bootstrap["simultaneous_high"])
            status = _reference_path_status(
                point,
                simultaneous_low,
                simultaneous_high,
            )
            for weight_index, weight in enumerate(reference_weights):
                reference_path_summaries.append(
                    {
                        "outcome": outcome,
                        "target_tv": target_tv,
                        "common_marginal_weight": weight,
                        "splits": len(split_seeds),
                        "common_delta_matching_mean": point[weight_index],
                        "pointwise_bootstrap_low": pointwise_low[weight_index],
                        "pointwise_bootstrap_high": pointwise_high[weight_index],
                        "simultaneous_bootstrap_low": simultaneous_low[
                            weight_index
                        ],
                        "simultaneous_bootstrap_high": simultaneous_high[
                            weight_index
                        ],
                        "path_estimate_min": float(np.min(point)),
                        "path_estimate_max": float(np.max(point)),
                        "path_simultaneous_envelope_lower": float(
                            np.min(simultaneous_low)
                        ),
                        "path_simultaneous_envelope_upper": float(
                            np.max(simultaneous_high)
                        ),
                        "reference_path_status": status,
                        "user_bootstrap_replicates": (
                            args.user_bootstrap_replicates
                        ),
                        "user_bootstrap_seed": args.user_bootstrap_seed,
                    }
                )
    strength_path_summaries = summarize_fixed_strength_rows(
        strength_path_rows
    )
    strength_rng = np.random.default_rng(args.user_bootstrap_seed)
    for summary in strength_path_summaries:
        outcome = str(summary["outcome"])
        target_tv = float(summary["target_tv"])
        point, low, high = paired_user_bootstrap_interval(
            np.vstack(strength_path_contributions[(outcome, target_tv)]),
            replicates=args.user_bootstrap_replicates,
            rng=strength_rng,
        )
        if not np.isclose(
            point,
            float(summary["delta_matching_mean"]),
            atol=1e-12,
        ):
            raise ValueError("strength-path user and split means do not agree")
        summary.update(
            {
                "user_bootstrap_low": low,
                "user_bootstrap_high": high,
                "user_bootstrap_replicates": args.user_bootstrap_replicates,
                "user_bootstrap_seed": args.user_bootstrap_seed,
            }
        )
    kl_sensitivity_summaries = summarize_kl_sensitivity_rows(
        kl_sensitivity_rows
    )
    kl_rng = np.random.default_rng(args.user_bootstrap_seed)
    for summary in kl_sensitivity_summaries:
        key = (
            str(summary["outcome"]),
            float(summary["source_target_tv"]),
        )
        point, low, high = paired_user_bootstrap_interval(
            np.vstack(kl_sensitivity_contributions[key]),
            replicates=args.user_bootstrap_replicates,
            rng=kl_rng,
        )
        if not np.isclose(
            point,
            float(summary["kl_delta_matching_mean"]),
            atol=1e-12,
        ):
            raise ValueError("KL user-level and split-level means do not agree")
        summary.update(
            {
                "kl_delta_user_bootstrap_low": low,
                "kl_delta_user_bootstrap_high": high,
                "user_bootstrap_replicates": args.user_bootstrap_replicates,
                "user_bootstrap_seed": args.user_bootstrap_seed,
            }
        )
    factor_sensitivity_rows = run_common_exposure_factor_sensitivity(
        outcome_surfaces["complete_watch"],
        outcome="complete_watch",
        split_seeds=split_seeds,
        user_folds=args.user_folds,
        history_action_count=args.history_action_count,
        rank_grid=factor_sensitivity_ranks,
        ridge_penalty_grid=factor_sensitivity_ridges,
        primary_rank=args.rank,
        primary_ridge_penalty=args.ridge_penalty,
        neighbor_fraction_grid=neighbor_fractions,
        inner_validation_fraction=args.inner_validation_fraction,
        target_tv=args.factor_sensitivity_tv,
    )
    factor_sensitivity_summaries = (
        summarize_common_exposure_factor_sensitivity(
            factor_sensitivity_rows
        )
    )
    knn_sensitivity_rows = run_common_exposure_knn_sensitivity(
        outcome_surfaces["complete_watch"],
        outcome="complete_watch",
        split_seeds=split_seeds,
        user_folds=args.user_folds,
        history_action_count=args.history_action_count,
        factor_rank=args.rank,
        factor_ridge_penalty=args.ridge_penalty,
        neighbor_fraction_grid=neighbor_fractions,
        inner_validation_fraction=args.inner_validation_fraction,
        target_tv=args.factor_sensitivity_tv,
    )
    knn_sensitivity_summaries = summarize_common_exposure_knn_sensitivity(
        knn_sensitivity_rows
    )
    _write_rows(args.output_csv, rows)
    _write_rows(args.output_summary_csv, summaries)
    _write_rows(args.output_reference_path_csv, reference_path_rows)
    _write_rows(
        args.output_reference_path_summary_csv,
        reference_path_summaries,
    )
    _write_rows(args.output_strength_path_csv, strength_path_rows)
    _write_rows(
        args.output_strength_path_summary_csv,
        strength_path_summaries,
    )
    _write_rows(args.output_kl_sensitivity_csv, kl_sensitivity_rows)
    _write_rows(
        args.output_kl_sensitivity_summary_csv,
        kl_sensitivity_summaries,
    )
    _write_rows(args.output_factor_sensitivity_csv, factor_sensitivity_rows)
    _write_rows(
        args.output_factor_sensitivity_summary_csv,
        factor_sensitivity_summaries,
    )
    _write_rows(args.output_knn_sensitivity_csv, knn_sensitivity_rows)
    _write_rows(
        args.output_knn_sensitivity_summary_csv,
        knn_sensitivity_summaries,
    )
    payload = {
        "protocol": {
            "common_candidate_actions": int(common_actions.sum()),
            "evaluation_users": int(preferences.utility.shape[0]),
            "split_seeds": list(split_seeds),
            "target_tvs": list(target_tvs),
            "strength_path_tvs": list(strength_path_tvs),
            "strength_path_anchor_tv": args.strength_path_anchor_tv,
            "reference_marginal_weights": list(reference_weights),
            "reference_marginal_path": (
                "(1 - weight) * interaction-MF own marginal + weight * "
                "user-kNN own marginal"
            ),
            "common_marginal": (
                "within each outer fold, the mean of the two learned policies' "
                "own marginals at the requested own-reference TV"
            ),
            "common_policy": (
                "minimum-KL action-intercept projection to the common marginal; "
                "score strength calibrated to the requested common-exposure TV"
            ),
            "outcome_access": (
                "common-exposure policies use frozen scores only; candidate "
                "outcomes are read after calibration"
            ),
            "positive_control_role": (
                "uses held-out outcomes as scores, then applies the same "
                "common-marginal projection and TV calibration; it is not "
                "learned or deployable and is not claimed to be the best "
                "policy under the two constraints"
            ),
            "user_bootstrap": (
                "paired percentile bootstrap over users after averaging each "
                "user's frozen-policy value gap over split seeds"
            ),
            "reference_path_bootstrap": (
                "the same resampled users are used at every common-marginal "
                "weight; simultaneous bands use the maximum absolute centered "
                "bootstrap deviation over the path"
            ),
            "user_bootstrap_replicates": args.user_bootstrap_replicates,
            "user_bootstrap_seed": args.user_bootstrap_seed,
            "shared_quality_negative_control": (
                "constructed shared-score control equals the interaction-only "
                "MF score plus a "
                "fold-specific action-only item-quality term; common-exposure "
                "projection should remove this term exactly"
            ),
            "fixed_marginal_strength_path": (
                "for each outcome, one midpoint marginal is built at the anchor "
                "TV and held fixed while policy TV changes"
            ),
            "kl_strength_sensitivity": (
                "for each main TV cell, hold its common marginal fixed; set "
                "the outcome-free KL target to the mean KL of the two "
                "equal-TV projected policies, then recalibrate both frozen "
                "scores to equal mean KL from the marginal"
            ),
            "factor_hyperparameter_sensitivity": (
                "for complete watch, keep each split's primary MF-kNN "
                "midpoint marginal and TV fixed, then vary MF rank and ridge"
            ),
            "factor_rank_sensitivity_grid": list(factor_sensitivity_ranks),
            "factor_ridge_sensitivity_grid": list(
                factor_sensitivity_ridges
            ),
            "factor_sensitivity_tv": args.factor_sensitivity_tv,
            "knn_hyperparameter_sensitivity": (
                "for complete watch, keep each split's primary MF and "
                "nested-selected-kNN midpoint marginal and TV fixed, then "
                "evaluate every fixed neighborhood fraction"
            ),
            "knn_neighbor_fraction_sensitivity_grid": list(neighbor_fractions),
        },
        "summaries": summaries,
        "reference_path_summaries": reference_path_summaries,
        "strength_path_summaries": strength_path_summaries,
        "kl_sensitivity_summaries": kl_sensitivity_summaries,
        "factor_sensitivity_summaries": factor_sensitivity_summaries,
        "knn_sensitivity_summaries": knn_sensitivity_summaries,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
