#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate cross-fitted KuaiRec policies without candidate-outcome leakage.

The dense matrix is split along both users and actions.  For each user fold,
the remaining users teach item quality and item factors.  Held-out users reveal
only a disjoint history-action block used to fit their latent representations.
Their policies are frozen before outcomes on the candidate-action block are
accessed.  Rotating the held-out fold evaluates every user without training a
user's policy on that user's candidate outcomes.

This is a controlled warm-start recommendation protocol on a dense preference
surface.  It is not an online causal experiment, but unlike the oracle policy
it can test whether overall policy value and exposure-preserving matching value
rank independently learned policies differently.  The primary cross-family
comparison contrasts the existing interaction-only ridge-factorization head
with centered user-kNN; the standard biased factor head is reported as a
non-reversing control.  The kNN neighborhood is selected using only outer-
training users before an outer user's candidate outcomes are read.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

from matchope.pipeline.run_kuairec_matching_oracle import (
    _softmax,
    load_preference_matrix,
)


@dataclass(frozen=True)
class CrossFittedLearnedPolicySplit:
    split_seed: int
    user_folds: int
    minimum_training_users: int
    maximum_training_users: int
    evaluation_user_indices: np.ndarray
    evaluation_fold_ids: np.ndarray
    history_action_indices: np.ndarray
    candidate_action_indices: np.ndarray
    candidate_action_start: int
    candidate_item_quality: np.ndarray
    candidate_interaction_score: np.ndarray
    candidate_utility: np.ndarray
    history_rmse: float
    rank: int
    ridge_penalty: float


@dataclass(frozen=True)
class KNNSelectionRecord:
    outer_fold: int
    inner_training_users: int
    inner_validation_users: int
    neighbor_fraction: float
    inner_neighbor_count: int
    outer_neighbor_count: int
    validation_rmse: float
    selected: bool


@dataclass(frozen=True)
class CrossFittedKNNPolicySplit:
    split_seed: int
    user_folds: int
    minimum_training_users: int
    maximum_training_users: int
    evaluation_user_indices: np.ndarray
    evaluation_fold_ids: np.ndarray
    history_action_indices: np.ndarray
    candidate_action_indices: np.ndarray
    candidate_action_start: int
    candidate_score: np.ndarray
    candidate_scores_by_neighbor_fraction: dict[float, np.ndarray]
    candidate_utility: np.ndarray
    selected_neighbor_fractions: np.ndarray
    selected_neighbor_counts: np.ndarray
    selection_records: tuple[KNNSelectionRecord, ...]
    inner_validation_fraction: float


def _randomized_item_factors(
    interaction: np.ndarray,
    *,
    rank: int,
    rng: np.random.Generator,
    oversampling: int = 8,
    power_iterations: int = 2,
) -> np.ndarray:
    """Return scaled item factors from a deterministic randomized SVD."""

    matrix = np.asarray(interaction, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("interaction must be a finite matrix")
    maximum_rank = min(matrix.shape)
    if not 1 <= rank < maximum_rank:
        raise ValueError("rank must be positive and smaller than both dimensions")
    if oversampling < 0 or power_iterations < 0:
        raise ValueError("oversampling and power_iterations must be non-negative")
    sketch_width = min(maximum_rank, rank + oversampling)
    projection = rng.normal(size=(matrix.shape[1], sketch_width))
    sketch = matrix @ projection
    for _ in range(power_iterations):
        sketch = matrix @ (matrix.T @ sketch)
    basis, _ = np.linalg.qr(sketch, mode="reduced")
    _, singular_values, right_vectors = np.linalg.svd(
        basis.T @ matrix,
        full_matrices=False,
    )
    return right_vectors[:rank].T * singular_values[:rank]


def construct_cross_fitted_learned_policy_split(
    utility: np.ndarray,
    *,
    user_folds: int,
    history_action_count: int,
    rank: int,
    ridge_penalty: float,
    split_seed: int,
    candidate_action_start: int | None = None,
) -> CrossFittedLearnedPolicySplit:
    """Cross-fit low-rank scorers before exposing candidate outcomes."""

    surface = np.asarray(utility, dtype=np.float64)
    if surface.ndim != 2 or min(surface.shape) < 4 or not np.isfinite(surface).all():
        raise ValueError("utility must be a finite matrix with at least four rows/actions")
    n_users, n_actions = surface.shape
    if not 2 <= user_folds <= n_users // 2:
        raise ValueError("user_folds must leave at least two users in every fold")
    candidate_start = (
        history_action_count
        if candidate_action_start is None
        else int(candidate_action_start)
    )
    if not 2 <= history_action_count <= candidate_start <= n_actions - 2:
        raise ValueError(
            "history_action_count and candidate_action_start must define "
            "disjoint history/candidate blocks with at least two candidates"
        )
    if ridge_penalty <= 0.0 or not np.isfinite(ridge_penalty):
        raise ValueError("ridge_penalty must be finite and positive")

    rng = np.random.default_rng(split_seed)
    user_order = rng.permutation(n_users)
    action_order = rng.permutation(n_actions)
    history_actions = action_order[:history_action_count]
    candidate_actions = action_order[candidate_start:]
    evaluation_folds = tuple(np.array_split(user_order, user_folds))
    all_users = np.arange(n_users)
    evaluation_parts: list[np.ndarray] = []
    fold_id_parts: list[np.ndarray] = []
    quality_parts: list[np.ndarray] = []
    interaction_parts: list[np.ndarray] = []
    utility_parts: list[np.ndarray] = []
    squared_history_errors = 0.0
    history_observations = 0
    training_counts: list[int] = []
    for fold_id, evaluation_users in enumerate(evaluation_folds):
        training_mask = np.ones(n_users, dtype=bool)
        training_mask[evaluation_users] = False
        training_users = all_users[training_mask]
        training_counts.append(int(training_users.size))
        training_utility = surface[training_users]
        item_quality = training_utility.mean(axis=0)
        training_user_main = training_utility.mean(axis=1) - training_utility.mean()
        training_interaction = (
            training_utility - item_quality[None, :] - training_user_main[:, None]
        )
        item_factors = _randomized_item_factors(
            training_interaction,
            rank=rank,
            rng=np.random.default_rng(
                np.random.SeedSequence([int(split_seed), int(fold_id)])
            ),
        )
        history_design = np.column_stack(
            (np.ones(history_action_count), item_factors[history_actions])
        )
        penalty = ridge_penalty * np.eye(rank + 1)
        penalty[0, 0] = 0.0
        inverse = np.linalg.inv(history_design.T @ history_design + penalty)
        centered_history = (
            surface[np.ix_(evaluation_users, history_actions)]
            - item_quality[history_actions]
        )
        coefficients = centered_history @ history_design @ inverse
        fitted_history = coefficients @ history_design.T
        squared_history_errors += float(
            np.square(centered_history - fitted_history).sum()
        )
        history_observations += int(centered_history.size)
        evaluation_parts.append(evaluation_users)
        fold_id_parts.append(
            np.full(evaluation_users.size, fold_id, dtype=np.int64)
        )
        quality_parts.append(
            np.broadcast_to(
                item_quality[candidate_actions],
                (evaluation_users.size, candidate_actions.size),
            )
        )
        interaction_parts.append(
            coefficients[:, 1:] @ item_factors[candidate_actions].T
        )
        utility_parts.append(surface[np.ix_(evaluation_users, candidate_actions)])

    return CrossFittedLearnedPolicySplit(
        split_seed=int(split_seed),
        user_folds=int(user_folds),
        minimum_training_users=min(training_counts),
        maximum_training_users=max(training_counts),
        evaluation_user_indices=np.concatenate(evaluation_parts),
        evaluation_fold_ids=np.concatenate(fold_id_parts),
        history_action_indices=history_actions,
        candidate_action_indices=candidate_actions,
        candidate_action_start=candidate_start,
        candidate_item_quality=np.vstack(quality_parts),
        candidate_interaction_score=np.vstack(interaction_parts),
        candidate_utility=np.vstack(utility_parts),
        history_rmse=float(np.sqrt(squared_history_errors / history_observations)),
        rank=int(rank),
        ridge_penalty=float(ridge_penalty),
    )


def _centered_user_knn_predictions(
    training_utility: np.ndarray,
    query_history_utility: np.ndarray,
    *,
    history_action_indices: np.ndarray,
    candidate_action_indices: np.ndarray,
    neighbor_counts: Sequence[int],
) -> dict[int, np.ndarray]:
    """Predict candidates from centered cosine neighbors.

    The query argument contains history outcomes only.  Candidate outcomes for
    the query users therefore cannot enter either neighbor search or scoring.
    Row-main effects are estimated from history and added back for prediction
    RMSE; they are constant across actions and do not change the induced policy.
    """

    training = np.asarray(training_utility, dtype=np.float64)
    query_history = np.asarray(query_history_utility, dtype=np.float64)
    history_actions = np.asarray(history_action_indices, dtype=np.int64)
    candidate_actions = np.asarray(candidate_action_indices, dtype=np.int64)
    counts = tuple(sorted({int(value) for value in neighbor_counts}))
    if training.ndim != 2 or not np.isfinite(training).all():
        raise ValueError("training_utility must be a finite matrix")
    if query_history.ndim != 2 or not np.isfinite(query_history).all():
        raise ValueError("query_history_utility must be a finite matrix")
    if query_history.shape[1] != history_actions.size:
        raise ValueError("query history width must match history_action_indices")
    if not counts or counts[0] <= 0 or counts[-1] > training.shape[0]:
        raise ValueError("neighbor counts must be positive and fit the training cohort")
    if history_actions.size < 2 or candidate_actions.size < 2:
        raise ValueError("history and candidate blocks must each contain two actions")
    if np.intersect1d(history_actions, candidate_actions).size:
        raise ValueError("history and candidate action blocks must be disjoint")
    if min(history_actions.min(), candidate_actions.min()) < 0:
        raise ValueError("action indices must be non-negative")
    if max(history_actions.max(), candidate_actions.max()) >= training.shape[1]:
        raise ValueError("action indices exceed the training surface")

    item_quality = training.mean(axis=0)
    training_user_main = training.mean(axis=1) - training.mean()
    training_residual = (
        training - item_quality[None, :] - training_user_main[:, None]
    )
    query_centered = query_history - item_quality[history_actions][None, :]
    query_user_main = query_centered.mean(axis=1)
    query_residual = query_centered - query_user_main[:, None]
    training_history = training_residual[:, history_actions]
    query_norm = np.linalg.norm(query_residual, axis=1)
    training_norm = np.linalg.norm(training_history, axis=1)
    if np.any(query_norm <= 0.0) or np.any(training_norm <= 0.0):
        raise ValueError("centered kNN requires non-constant history profiles")
    similarities = (
        query_residual / query_norm[:, None]
    ) @ (training_history / training_norm[:, None]).T
    predictions: dict[int, np.ndarray] = {}
    candidate_residual = training_residual[:, candidate_actions]
    for neighbor_count in counts:
        indices = np.argpartition(
            similarities,
            -neighbor_count,
            axis=1,
        )[:, -neighbor_count:]
        top_similarities = np.take_along_axis(similarities, indices, axis=1)
        descending = np.argsort(top_similarities, axis=1)[:, ::-1]
        indices = np.take_along_axis(indices, descending, axis=1)
        weights = np.maximum(
            np.take_along_axis(top_similarities, descending, axis=1),
            0.0,
        )
        normalizers = weights.sum(axis=1)
        if np.any(normalizers <= 0.0):
            raise ValueError("selected kNN neighborhoods have no positive similarity")
        weights = weights / normalizers[:, None]
        weight_matrix = csr_matrix(
            (
                weights.ravel(),
                (
                    np.repeat(np.arange(query_history.shape[0]), neighbor_count),
                    indices.ravel(),
                ),
            ),
            shape=(query_history.shape[0], training.shape[0]),
        )
        predictions[neighbor_count] = (
            item_quality[candidate_actions][None, :]
            + query_user_main[:, None]
            + weight_matrix @ candidate_residual
        )
    return predictions


def construct_cross_fitted_knn_policy_split(
    utility: np.ndarray,
    *,
    user_folds: int,
    history_action_count: int,
    neighbor_fraction_grid: Sequence[float],
    inner_validation_fraction: float,
    split_seed: int,
    candidate_action_start: int | None = None,
) -> CrossFittedKNNPolicySplit:
    """Cross-fit user-kNN and select its neighborhood within outer training data."""

    surface = np.asarray(utility, dtype=np.float64)
    if surface.ndim != 2 or min(surface.shape) < 4 or not np.isfinite(surface).all():
        raise ValueError("utility must be a finite matrix with at least four rows/actions")
    n_users, n_actions = surface.shape
    if not 2 <= user_folds <= n_users // 2:
        raise ValueError("user_folds must leave at least two users in every fold")
    candidate_start = (
        history_action_count
        if candidate_action_start is None
        else int(candidate_action_start)
    )
    if not 2 <= history_action_count <= candidate_start <= n_actions - 2:
        raise ValueError(
            "history_action_count and candidate_action_start must define "
            "disjoint history/candidate blocks with at least two candidates"
        )
    if not 0.0 < inner_validation_fraction < 0.5:
        raise ValueError("inner_validation_fraction must lie strictly between 0 and 0.5")
    neighbor_fractions = tuple(
        sorted({float(value) for value in neighbor_fraction_grid})
    )
    if (
        not neighbor_fractions
        or neighbor_fractions[0] <= 0.0
        or neighbor_fractions[-1] > 1.0
        or not np.isfinite(neighbor_fractions).all()
    ):
        raise ValueError("neighbor fractions must be finite and lie in (0, 1]")
    if neighbor_fractions[-1] != 1.0:
        raise ValueError(
            "neighbor_fraction_grid must include 1.0 to close the selection boundary"
        )

    rng = np.random.default_rng(split_seed)
    user_order = rng.permutation(n_users)
    action_order = rng.permutation(n_actions)
    history_actions = action_order[:history_action_count]
    candidate_actions = action_order[candidate_start:]
    evaluation_folds = tuple(np.array_split(user_order, user_folds))
    all_users = np.arange(n_users)
    evaluation_parts: list[np.ndarray] = []
    fold_id_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    fixed_score_parts: dict[float, list[np.ndarray]] = {
        neighbor_fraction: [] for neighbor_fraction in neighbor_fractions
    }
    utility_parts: list[np.ndarray] = []
    selected_fractions: list[float] = []
    selected_counts: list[int] = []
    selection_records: list[KNNSelectionRecord] = []
    training_counts: list[int] = []
    for fold_id, evaluation_users in enumerate(evaluation_folds):
        training_mask = np.ones(n_users, dtype=bool)
        training_mask[evaluation_users] = False
        training_users = all_users[training_mask]
        training_counts.append(int(training_users.size))
        inner_rng = np.random.default_rng(
            np.random.SeedSequence([int(split_seed), int(fold_id), 2027])
        )
        inner_order = inner_rng.permutation(training_users)
        validation_count = max(
            2,
            int(round(inner_validation_fraction * training_users.size)),
        )
        inner_validation_users = inner_order[:validation_count]
        inner_training_users = inner_order[validation_count:]
        inner_neighbor_counts = {
            neighbor_fraction: max(
                1,
                int(round(neighbor_fraction * inner_training_users.size)),
            )
            for neighbor_fraction in neighbor_fractions
        }
        if len(set(inner_neighbor_counts.values())) != len(neighbor_fractions):
            raise ValueError(
                "neighbor fractions collapse to duplicate counts in an inner cohort"
            )
        inner_predictions = _centered_user_knn_predictions(
            surface[inner_training_users],
            surface[np.ix_(inner_validation_users, history_actions)],
            history_action_indices=history_actions,
            candidate_action_indices=candidate_actions,
            neighbor_counts=tuple(inner_neighbor_counts.values()),
        )
        validation_target = surface[
            np.ix_(inner_validation_users, candidate_actions)
        ]
        validation_rmses = {
            neighbor_fraction: float(
                np.sqrt(
                    np.mean(
                        np.square(
                            inner_predictions[
                                inner_neighbor_counts[neighbor_fraction]
                            ]
                            - validation_target
                        )
                    )
                )
            )
            for neighbor_fraction in neighbor_fractions
        }
        selected_neighbor_fraction = min(
            neighbor_fractions,
            key=lambda neighbor_fraction: (
                validation_rmses[neighbor_fraction],
                neighbor_fraction,
            ),
        )
        selected_neighbor_count = max(
            1,
            int(round(selected_neighbor_fraction * training_users.size)),
        )
        selected_fractions.append(selected_neighbor_fraction)
        selected_counts.append(selected_neighbor_count)
        selection_records.extend(
            KNNSelectionRecord(
                outer_fold=fold_id,
                inner_training_users=int(inner_training_users.size),
                inner_validation_users=int(inner_validation_users.size),
                neighbor_fraction=neighbor_fraction,
                inner_neighbor_count=inner_neighbor_counts[neighbor_fraction],
                outer_neighbor_count=max(
                    1,
                    int(round(neighbor_fraction * training_users.size)),
                ),
                validation_rmse=validation_rmses[neighbor_fraction],
                selected=neighbor_fraction == selected_neighbor_fraction,
            )
            for neighbor_fraction in neighbor_fractions
        )
        outer_neighbor_counts = {
            neighbor_fraction: max(
                1,
                int(round(neighbor_fraction * training_users.size)),
            )
            for neighbor_fraction in neighbor_fractions
        }
        outer_predictions = _centered_user_knn_predictions(
            surface[training_users],
            surface[np.ix_(evaluation_users, history_actions)],
            history_action_indices=history_actions,
            candidate_action_indices=candidate_actions,
            neighbor_counts=tuple(outer_neighbor_counts.values()),
        )
        evaluation_parts.append(evaluation_users)
        fold_id_parts.append(
            np.full(evaluation_users.size, fold_id, dtype=np.int64)
        )
        score_parts.append(outer_predictions[selected_neighbor_count])
        for neighbor_fraction, neighbor_count in outer_neighbor_counts.items():
            fixed_score_parts[neighbor_fraction].append(
                outer_predictions[neighbor_count]
            )
        utility_parts.append(surface[np.ix_(evaluation_users, candidate_actions)])

    return CrossFittedKNNPolicySplit(
        split_seed=int(split_seed),
        user_folds=int(user_folds),
        minimum_training_users=min(training_counts),
        maximum_training_users=max(training_counts),
        evaluation_user_indices=np.concatenate(evaluation_parts),
        evaluation_fold_ids=np.concatenate(fold_id_parts),
        history_action_indices=history_actions,
        candidate_action_indices=candidate_actions,
        candidate_action_start=candidate_start,
        candidate_score=np.vstack(score_parts),
        candidate_scores_by_neighbor_fraction={
            neighbor_fraction: np.vstack(parts)
            for neighbor_fraction, parts in fixed_score_parts.items()
        },
        candidate_utility=np.vstack(utility_parts),
        selected_neighbor_fractions=np.asarray(
            selected_fractions,
            dtype=np.float64,
        ),
        selected_neighbor_counts=np.asarray(selected_counts, dtype=np.int64),
        selection_records=tuple(selection_records),
        inner_validation_fraction=float(inner_validation_fraction),
    )


def evaluate_policy_family(
    split: CrossFittedLearnedPolicySplit,
    *,
    outcome: str,
    quality_weights: Sequence[float],
    inverse_temperatures: Sequence[float],
) -> list[dict[str, float | int | str]]:
    """Evaluate policies that trade shared item quality against learned matching."""

    if not outcome:
        raise ValueError("outcome must not be empty")
    rows: list[dict[str, float | int | str]] = []
    utility = split.candidate_utility
    for inverse_temperature in inverse_temperatures:
        if inverse_temperature <= 0.0 or not np.isfinite(inverse_temperature):
            raise ValueError("inverse temperatures must be finite and positive")
        for quality_weight in quality_weights:
            if quality_weight < 0.0 or not np.isfinite(quality_weight):
                raise ValueError("quality weights must be finite and non-negative")
            score = (
                quality_weight * split.candidate_item_quality
                + split.candidate_interaction_score
            )
            metrics = _evaluate_frozen_score(
                score,
                utility,
                evaluation_fold_ids=split.evaluation_fold_ids,
                inverse_temperature=float(inverse_temperature),
            )
            rows.append(
                {
                    "outcome": outcome,
                    "split_seed": split.split_seed,
                    "user_folds": split.user_folds,
                    "minimum_training_users": split.minimum_training_users,
                    "maximum_training_users": split.maximum_training_users,
                    "evaluation_users": int(split.evaluation_user_indices.size),
                    "history_actions": int(split.history_action_indices.size),
                    "candidate_actions": int(split.candidate_action_indices.size),
                    "candidate_action_start": split.candidate_action_start,
                    "reference_stratum": "outer_fold",
                    "rank": split.rank,
                    "ridge_penalty": split.ridge_penalty,
                    "history_rmse": split.history_rmse,
                    "quality_weight": float(quality_weight),
                    "inverse_temperature": float(inverse_temperature),
                    **metrics,
                }
            )
    return rows


def _evaluate_frozen_score(
    score: np.ndarray,
    utility: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    inverse_temperature: float,
) -> dict[str, float]:
    score_matrix = np.asarray(score, dtype=np.float64)
    outcome_matrix = np.asarray(utility, dtype=np.float64)
    if score_matrix.shape != outcome_matrix.shape or score_matrix.ndim != 2:
        raise ValueError("score and utility must be aligned matrices")
    if not np.isfinite(score_matrix).all() or not np.isfinite(outcome_matrix).all():
        raise ValueError("score and utility must be finite")
    fold_ids = np.asarray(evaluation_fold_ids, dtype=np.int64)
    if fold_ids.ndim != 1 or fold_ids.size != score_matrix.shape[0]:
        raise ValueError("evaluation_fold_ids must align with score rows")
    if np.unique(fold_ids).size < 2:
        raise ValueError("fold-conditioned evaluation requires at least two folds")
    if inverse_temperature <= 0.0 or not np.isfinite(inverse_temperature):
        raise ValueError("inverse_temperature must be finite and positive")
    cross_fitted_shared_values = np.empty(outcome_matrix.shape[0], dtype=np.float64)
    selected_shared_actions: list[int] = []
    for fold_id in np.unique(fold_ids):
        evaluation_mask = fold_ids == fold_id
        training_mask = ~evaluation_mask
        if not bool(np.any(training_mask)):
            raise ValueError("each evaluation fold must leave shared-policy training rows")
        selected_action = int(
            np.argmax(outcome_matrix[training_mask].mean(axis=0))
        )
        selected_shared_actions.append(selected_action)
        cross_fitted_shared_values[evaluation_mask] = outcome_matrix[
            evaluation_mask, selected_action
        ]
    policy = _softmax(score_matrix, float(inverse_temperature))
    row_reference = np.empty_like(policy)
    for fold_id in np.unique(fold_ids):
        fold_mask = fold_ids == fold_id
        row_reference[fold_mask] = policy[fold_mask].mean(axis=0)
    target_value = float(np.mean(np.sum(policy * outcome_matrix, axis=1)))
    reference_value = float(np.mean(np.sum(row_reference * outcome_matrix, axis=1)))
    cross_fitted_shared_value = float(cross_fitted_shared_values.mean())
    return {
        "overall_policy_value": target_value,
        "reference_policy_value": reference_value,
        "matching_value": target_value - reference_value,
        "uniform_value": float(outcome_matrix.mean()),
        "best_shared_value": float(outcome_matrix.mean(axis=0).max()),
        "actionable_gain_vs_best_shared": target_value
        - float(outcome_matrix.mean(axis=0).max()),
        "cross_fitted_shared_action_value": cross_fitted_shared_value,
        "actionable_gain_vs_cross_fitted_shared": (
            target_value - cross_fitted_shared_value
        ),
        "cross_fitted_shared_unique_actions": len(set(selected_shared_actions)),
        "mean_policy_to_reference_tv": float(
            np.mean(0.5 * np.abs(policy - row_reference).sum(axis=1))
        ),
    }


def _mean_policy_to_reference_tv(
    score: np.ndarray,
    evaluation_fold_ids: np.ndarray,
    inverse_temperature: float,
) -> float:
    policy = _softmax(score, inverse_temperature)
    row_reference = np.empty_like(policy)
    for fold_id in np.unique(evaluation_fold_ids):
        fold_mask = evaluation_fold_ids == fold_id
        row_reference[fold_mask] = policy[fold_mask].mean(axis=0)
    return float(np.mean(0.5 * np.abs(policy - row_reference).sum(axis=1)))


def calibrate_inverse_temperature_to_personalization_tv(
    score: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_tv: float,
    tolerance: float = 1e-7,
    maximum_iterations: int = 30,
) -> float:
    """Calibrate policy strength from scores alone, without outcome access."""

    score_matrix = np.asarray(score, dtype=np.float64)
    fold_ids = np.asarray(evaluation_fold_ids, dtype=np.int64)
    if score_matrix.ndim != 2 or not np.isfinite(score_matrix).all():
        raise ValueError("score must be a finite matrix")
    if fold_ids.ndim != 1 or fold_ids.size != score_matrix.shape[0]:
        raise ValueError("evaluation_fold_ids must align with score rows")
    if np.unique(fold_ids).size < 2:
        raise ValueError("TV calibration requires at least two folds")
    if not 0.0 < target_tv < 1.0 or not np.isfinite(target_tv):
        raise ValueError("target_tv must be finite and lie strictly between 0 and 1")
    if tolerance <= 0.0 or maximum_iterations <= 0:
        raise ValueError("calibration controls must be positive")

    lower = 0.0
    upper = 1.0
    upper_tv = _mean_policy_to_reference_tv(score_matrix, fold_ids, upper)
    while upper_tv < target_tv and upper < 1024.0:
        lower = upper
        upper *= 2.0
        upper_tv = _mean_policy_to_reference_tv(score_matrix, fold_ids, upper)
    if upper_tv < target_tv:
        raise ValueError("target_tv is not reachable below inverse temperature 1024")

    calibrated = upper
    calibrated_tv = upper_tv
    for _ in range(maximum_iterations):
        calibrated = 0.5 * (lower + upper)
        calibrated_tv = _mean_policy_to_reference_tv(
            score_matrix,
            fold_ids,
            calibrated,
        )
        if abs(calibrated_tv - target_tv) <= tolerance:
            break
        if calibrated_tv < target_tv:
            lower = calibrated
        else:
            upper = calibrated
    if abs(calibrated_tv - target_tv) > tolerance:
        raise ValueError("TV calibration did not converge")
    return float(calibrated)


def evaluate_cross_family_policies(
    factor_split: CrossFittedLearnedPolicySplit,
    knn_split: CrossFittedKNNPolicySplit,
    *,
    outcome: str,
    factor_quality_weights: Sequence[float],
    inverse_temperatures: Sequence[float],
) -> list[dict[str, object]]:
    """Evaluate the fixed factor heads and nested-selected neighborhood model."""

    if not outcome:
        raise ValueError("outcome must not be empty")
    quality_weights = tuple(float(value) for value in factor_quality_weights)
    if quality_weights != (0.0, 1.0):
        raise ValueError(
            "cross-family protocol requires factor quality weights (0, 1)"
        )
    if not np.array_equal(
        factor_split.evaluation_user_indices,
        knn_split.evaluation_user_indices,
    ):
        raise ValueError("factor and kNN evaluation users are not aligned")
    if not np.array_equal(
        factor_split.candidate_action_indices,
        knn_split.candidate_action_indices,
    ):
        raise ValueError("factor and kNN candidate actions are not aligned")
    if not np.allclose(
        factor_split.candidate_utility,
        knn_split.candidate_utility,
    ):
        raise ValueError("factor and kNN candidate outcomes are not aligned")
    family_specs = (
        (
            "interaction_ridge_mf",
            f"rank{factor_split.rank}_ridge{factor_split.ridge_penalty:g}_quality0",
            factor_split.candidate_interaction_score,
            "fixed_before_cross_family_probe",
            0.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        ),
        (
            "standard_ridge_mf",
            f"rank{factor_split.rank}_ridge{factor_split.ridge_penalty:g}_quality1",
            factor_split.candidate_item_quality
            + factor_split.candidate_interaction_score,
            "fixed_before_cross_family_probe",
            1.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        ),
        (
            "centered_user_knn",
            "inner_rmse_selected_neighbors",
            knn_split.candidate_score,
            "outer_training_inner_validation_rmse",
            float("nan"),
            float(knn_split.selected_neighbor_fractions.min()),
            float(knn_split.selected_neighbor_fractions.max()),
            float(knn_split.selected_neighbor_fractions.mean()),
            float(knn_split.selected_neighbor_counts.min()),
            float(knn_split.selected_neighbor_counts.max()),
            float(knn_split.selected_neighbor_counts.mean()),
        ),
    )
    rows: list[dict[str, object]] = []
    for inverse_temperature in inverse_temperatures:
        for (
            family,
            variant,
            score,
            selection_rule,
            factor_quality_weight,
            neighbor_fraction_minimum,
            neighbor_fraction_maximum,
            neighbor_fraction_mean,
            neighbor_minimum,
            neighbor_maximum,
            neighbor_mean,
        ) in family_specs:
            metrics = _evaluate_frozen_score(
                score,
                factor_split.candidate_utility,
                evaluation_fold_ids=factor_split.evaluation_fold_ids,
                inverse_temperature=float(inverse_temperature),
            )
            rows.append(
                {
                    "outcome": outcome,
                    "split_seed": factor_split.split_seed,
                    "policy_family": family,
                    "policy_variant": variant,
                    "selection_rule": selection_rule,
                    "user_folds": factor_split.user_folds,
                    "evaluation_users": int(
                        factor_split.evaluation_user_indices.size
                    ),
                    "history_actions": int(
                        factor_split.history_action_indices.size
                    ),
                    "candidate_actions": int(
                        factor_split.candidate_action_indices.size
                    ),
                    "candidate_action_start": factor_split.candidate_action_start,
                    "reference_stratum": "outer_fold",
                    "inverse_temperature": float(inverse_temperature),
                    "factor_quality_weight": (
                        factor_quality_weight
                    ),
                    "selected_neighbor_fraction_min": neighbor_fraction_minimum,
                    "selected_neighbor_fraction_max": neighbor_fraction_maximum,
                    "selected_neighbor_fraction_mean": neighbor_fraction_mean,
                    "selected_neighbor_count_min": neighbor_minimum,
                    "selected_neighbor_count_max": neighbor_maximum,
                    "selected_neighbor_count_mean": neighbor_mean,
                    **metrics,
                }
            )
    return rows


def evaluate_personalization_tv_matched_policies(
    factor_split: CrossFittedLearnedPolicySplit,
    knn_split: CrossFittedKNNPolicySplit,
    *,
    outcome: str,
    target_tvs: Sequence[float],
) -> list[dict[str, object]]:
    """Compare policy families at equal fold-conditioned personalization TV."""

    if not outcome:
        raise ValueError("outcome must not be empty")
    targets = tuple(sorted({float(value) for value in target_tvs}))
    if not targets:
        raise ValueError("at least one target personalization TV is required")
    if not np.array_equal(
        factor_split.evaluation_user_indices,
        knn_split.evaluation_user_indices,
    ):
        raise ValueError("factor and kNN evaluation users are not aligned")
    if not np.array_equal(
        factor_split.evaluation_fold_ids,
        knn_split.evaluation_fold_ids,
    ):
        raise ValueError("factor and kNN evaluation folds are not aligned")
    if not np.array_equal(
        factor_split.candidate_action_indices,
        knn_split.candidate_action_indices,
    ):
        raise ValueError("factor and kNN candidate actions are not aligned")
    if not np.allclose(
        factor_split.candidate_utility,
        knn_split.candidate_utility,
    ):
        raise ValueError("factor and kNN candidate outcomes are not aligned")

    family_scores = {
        "interaction_ridge_mf": factor_split.candidate_interaction_score,
        "standard_ridge_mf": (
            factor_split.candidate_item_quality
            + factor_split.candidate_interaction_score
        ),
        "centered_user_knn": knn_split.candidate_score,
    }
    rows: list[dict[str, object]] = []
    for target_tv in targets:
        calibrated: dict[str, tuple[float, dict[str, float]]] = {}
        for family, score in family_scores.items():
            inverse_temperature = (
                calibrate_inverse_temperature_to_personalization_tv(
                    score,
                    evaluation_fold_ids=factor_split.evaluation_fold_ids,
                    target_tv=target_tv,
                )
            )
            calibrated[family] = (
                inverse_temperature,
                _evaluate_frozen_score(
                    score,
                    factor_split.candidate_utility,
                    evaluation_fold_ids=factor_split.evaluation_fold_ids,
                    inverse_temperature=inverse_temperature,
                ),
            )
        knn_beta, knn = calibrated["centered_user_knn"]
        interaction_beta, interaction = calibrated["interaction_ridge_mf"]
        standard_beta, standard = calibrated["standard_ridge_mf"]
        delta_overall = knn["overall_policy_value"] - interaction[
            "overall_policy_value"
        ]
        delta_reference = knn["reference_policy_value"] - interaction[
            "reference_policy_value"
        ]
        delta_matching = knn["matching_value"] - interaction["matching_value"]
        standard_delta_overall = knn["overall_policy_value"] - standard[
            "overall_policy_value"
        ]
        standard_delta_reference = knn["reference_policy_value"] - standard[
            "reference_policy_value"
        ]
        standard_delta_matching = knn["matching_value"] - standard[
            "matching_value"
        ]
        rows.append(
            {
                "outcome": outcome,
                "split_seed": factor_split.split_seed,
                "target_policy_to_reference_tv": target_tv,
                "calibration_signal": "policy_scores_only_no_candidate_outcomes",
                "user_folds": factor_split.user_folds,
                "evaluation_users": int(factor_split.evaluation_user_indices.size),
                "history_actions": int(factor_split.history_action_indices.size),
                "candidate_actions": int(
                    factor_split.candidate_action_indices.size
                ),
                "candidate_action_start": factor_split.candidate_action_start,
                "reference_stratum": "outer_fold",
                "selected_neighbor_fraction_min": float(
                    knn_split.selected_neighbor_fractions.min()
                ),
                "selected_neighbor_fraction_max": float(
                    knn_split.selected_neighbor_fractions.max()
                ),
                "interaction_inverse_temperature": interaction_beta,
                "standard_inverse_temperature": standard_beta,
                "knn_inverse_temperature": knn_beta,
                "interaction_actual_tv": interaction[
                    "mean_policy_to_reference_tv"
                ],
                "standard_actual_tv": standard["mean_policy_to_reference_tv"],
                "knn_actual_tv": knn["mean_policy_to_reference_tv"],
                "interaction_overall_policy_value": interaction[
                    "overall_policy_value"
                ],
                "interaction_reference_policy_value": interaction[
                    "reference_policy_value"
                ],
                "interaction_matching_value": interaction["matching_value"],
                "standard_overall_policy_value": standard[
                    "overall_policy_value"
                ],
                "standard_reference_policy_value": standard[
                    "reference_policy_value"
                ],
                "standard_matching_value": standard["matching_value"],
                "knn_overall_policy_value": knn["overall_policy_value"],
                "knn_reference_policy_value": knn["reference_policy_value"],
                "knn_matching_value": knn["matching_value"],
                "delta_overall_knn_minus_interaction_mf": delta_overall,
                "delta_reference_knn_minus_interaction_mf": delta_reference,
                "delta_matching_knn_minus_interaction_mf": delta_matching,
                "interaction_decomposition_error": (
                    delta_overall - delta_reference - delta_matching
                ),
                "interaction_pair_ranking_reversal": bool(
                    delta_overall * delta_matching < 0.0
                ),
                "delta_overall_knn_minus_standard_mf": standard_delta_overall,
                "delta_reference_knn_minus_standard_mf": (
                    standard_delta_reference
                ),
                "delta_matching_knn_minus_standard_mf": standard_delta_matching,
                "standard_decomposition_error": (
                    standard_delta_overall
                    - standard_delta_reference
                    - standard_delta_matching
                ),
                "standard_pair_ranking_reversal": bool(
                    standard_delta_overall * standard_delta_matching < 0.0
                ),
            }
        )
    return rows


def summarize_cross_family_reversals(
    policy_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return primary and negative-control cross-family decompositions."""

    grouped: dict[tuple[str, int, float], dict[str, dict[str, object]]] = {}
    for row in policy_rows:
        key = (
            str(row["outcome"]),
            int(row["split_seed"]),
            float(row["inverse_temperature"]),
        )
        family = str(row["policy_family"])
        if family in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate policy family {family!r} for {key}")
        grouped[key][family] = row
    expected_families = {
        "interaction_ridge_mf",
        "standard_ridge_mf",
        "centered_user_knn",
    }
    summaries: list[dict[str, object]] = []
    for (outcome, split_seed, inverse_temperature), family_rows in sorted(
        grouped.items()
    ):
        if set(family_rows) != expected_families:
            raise ValueError(
                "cross-family comparison requires both factor heads and user-kNN"
            )
        knn = family_rows["centered_user_knn"]
        interaction = family_rows["interaction_ridge_mf"]
        standard = family_rows["standard_ridge_mf"]
        delta_overall = float(knn["overall_policy_value"]) - float(
            interaction["overall_policy_value"]
        )
        delta_reference = float(knn["reference_policy_value"]) - float(
            interaction["reference_policy_value"]
        )
        delta_matching = float(knn["matching_value"]) - float(
            interaction["matching_value"]
        )
        standard_delta_overall = float(knn["overall_policy_value"]) - float(
            standard["overall_policy_value"]
        )
        standard_delta_reference = float(knn["reference_policy_value"]) - float(
            standard["reference_policy_value"]
        )
        standard_delta_matching = float(knn["matching_value"]) - float(
            standard["matching_value"]
        )
        best_overall = max(
            family_rows,
            key=lambda family: float(family_rows[family]["overall_policy_value"]),
        )
        best_matching = max(
            family_rows,
            key=lambda family: float(family_rows[family]["matching_value"]),
        )
        summaries.append(
            {
                "outcome": outcome,
                "split_seed": split_seed,
                "inverse_temperature": inverse_temperature,
                "delta_overall_knn_minus_interaction_mf": delta_overall,
                "delta_reference_knn_minus_interaction_mf": delta_reference,
                "delta_matching_knn_minus_interaction_mf": delta_matching,
                "interaction_decomposition_error": delta_overall
                - delta_reference
                - delta_matching,
                "interaction_pair_overall_preferred": (
                    "centered_user_knn"
                    if delta_overall > 0.0
                    else "interaction_ridge_mf"
                ),
                "interaction_pair_matching_preferred": (
                    "centered_user_knn"
                    if delta_matching > 0.0
                    else "interaction_ridge_mf"
                ),
                "interaction_pair_ranking_reversal": bool(
                    delta_overall * delta_matching < 0.0
                ),
                "delta_overall_knn_minus_standard_mf": standard_delta_overall,
                "delta_reference_knn_minus_standard_mf": standard_delta_reference,
                "delta_matching_knn_minus_standard_mf": standard_delta_matching,
                "standard_decomposition_error": standard_delta_overall
                - standard_delta_reference
                - standard_delta_matching,
                "standard_pair_ranking_reversal": bool(
                    standard_delta_overall * standard_delta_matching < 0.0
                ),
                "bank_overall_best_family": best_overall,
                "bank_matching_best_family": best_matching,
                "bank_ranking_reversal": best_overall != best_matching,
            }
        )
    return summaries


def annotate_history_sensitivity(
    summary_rows: Sequence[dict[str, object]],
    selection_rows: Sequence[dict[str, object]],
    *,
    history_action_count: int,
    candidate_action_start: int,
) -> list[dict[str, object]]:
    """Attach selection diagnostics to a fixed-candidate history comparison."""

    selected_by_split: dict[tuple[str, int], list[float]] = {}
    for row in selection_rows:
        if bool(row["selected"]):
            key = (str(row["outcome"]), int(row["split_seed"]))
            selected_by_split.setdefault(key, []).append(
                float(row["neighbor_fraction"])
            )
    annotated: list[dict[str, object]] = []
    for row in summary_rows:
        key = (str(row["outcome"]), int(row["split_seed"]))
        selected_fractions = selected_by_split.get(key, [])
        if not selected_fractions:
            raise ValueError(f"missing selected kNN fractions for {key}")
        annotated.append(
            {
                "history_action_count": int(history_action_count),
                "candidate_action_start": int(candidate_action_start),
                "unused_action_gap": int(
                    candidate_action_start - history_action_count
                ),
                "selected_neighbor_fraction_min": min(selected_fractions),
                "selected_neighbor_fraction_max": max(selected_fractions),
                "selected_neighbor_fraction_mean": float(
                    np.mean(selected_fractions)
                ),
                "selected_full_fraction_folds": sum(
                    fraction == 1.0 for fraction in selected_fractions
                ),
                **row,
            }
        )
    return annotated


def summarize_ranking_reversals(
    policy_rows: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Summarize agreement between overall-value and matching-value rankings."""

    grouped: dict[
        tuple[str, int, float],
        list[dict[str, float | int | str]],
    ] = {}
    for row in policy_rows:
        key = (
            str(row["outcome"]),
            int(row["split_seed"]),
            float(row["inverse_temperature"]),
        )
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, float | int | str]] = []
    for (outcome, split_seed, inverse_temperature), rows in sorted(grouped.items()):
        if len(rows) < 2:
            raise ValueError("ranking comparison requires at least two policies")
        rows = sorted(rows, key=lambda row: float(row["quality_weight"]))
        overall = np.asarray([row["overall_policy_value"] for row in rows])
        matching = np.asarray([row["matching_value"] for row in rows])
        disagreements = 0
        comparable_pairs = 0
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                overall_sign = np.sign(overall[left] - overall[right])
                matching_sign = np.sign(matching[left] - matching[right])
                if overall_sign != 0.0 and matching_sign != 0.0:
                    comparable_pairs += 1
                    disagreements += int(overall_sign != matching_sign)
        correlation = float(spearmanr(overall, matching).statistic)
        summaries.append(
            {
                "outcome": outcome,
                "split_seed": split_seed,
                "inverse_temperature": inverse_temperature,
                "policies": len(rows),
                "overall_best_quality_weight": float(
                    rows[int(np.argmax(overall))]["quality_weight"]
                ),
                "matching_best_quality_weight": float(
                    rows[int(np.argmax(matching))]["quality_weight"]
                ),
                "spearman_overall_vs_matching": correlation,
                "pairwise_ranking_disagreement": disagreements / comparable_pairs,
            }
        )
    return summaries


def run_learned_policy_splits(
    utility: np.ndarray,
    *,
    outcome: str,
    split_seeds: Sequence[int],
    user_folds: int,
    history_action_count: int,
    rank: int,
    ridge_penalty: float,
    quality_weights: Sequence[float],
    inverse_temperatures: Sequence[float],
    candidate_action_start: int | None = None,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    """Fit and evaluate the complete policy family across fixed random splits."""

    if not split_seeds:
        raise ValueError("at least one split seed is required")
    rows: list[dict[str, float | int | str]] = []
    for split_seed in split_seeds:
        split = construct_cross_fitted_learned_policy_split(
            utility,
            user_folds=user_folds,
            history_action_count=history_action_count,
            rank=rank,
            ridge_penalty=ridge_penalty,
            split_seed=int(split_seed),
            candidate_action_start=candidate_action_start,
        )
        rows.extend(
            evaluate_policy_family(
                split,
                outcome=outcome,
                quality_weights=quality_weights,
                inverse_temperatures=inverse_temperatures,
            )
        )
    return rows, summarize_ranking_reversals(rows)


def run_cross_family_policy_splits(
    utility: np.ndarray,
    *,
    outcome: str,
    split_seeds: Sequence[int],
    user_folds: int,
    history_action_count: int,
    rank: int,
    ridge_penalty: float,
    factor_quality_weights: Sequence[float],
    neighbor_fraction_grid: Sequence[float],
    inner_validation_fraction: float,
    inverse_temperatures: Sequence[float],
    candidate_action_start: int | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Run the cross-family comparison with training-only kNN selection."""

    if not split_seeds:
        raise ValueError("at least one split seed is required")
    policy_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for split_seed in split_seeds:
        factor_split = construct_cross_fitted_learned_policy_split(
            utility,
            user_folds=user_folds,
            history_action_count=history_action_count,
            rank=rank,
            ridge_penalty=ridge_penalty,
            split_seed=int(split_seed),
            candidate_action_start=candidate_action_start,
        )
        knn_split = construct_cross_fitted_knn_policy_split(
            utility,
            user_folds=user_folds,
            history_action_count=history_action_count,
            neighbor_fraction_grid=neighbor_fraction_grid,
            inner_validation_fraction=inner_validation_fraction,
            split_seed=int(split_seed),
            candidate_action_start=candidate_action_start,
        )
        policy_rows.extend(
            evaluate_cross_family_policies(
                factor_split,
                knn_split,
                outcome=outcome,
                factor_quality_weights=factor_quality_weights,
                inverse_temperatures=inverse_temperatures,
            )
        )
        selection_rows.extend(
            {
                "outcome": outcome,
                "split_seed": int(split_seed),
                "outer_fold": record.outer_fold,
                "inner_training_users": record.inner_training_users,
                "inner_validation_users": record.inner_validation_users,
                "neighbor_fraction": record.neighbor_fraction,
                "inner_neighbor_count": record.inner_neighbor_count,
                "outer_neighbor_count": record.outer_neighbor_count,
                "validation_rmse": record.validation_rmse,
                "selected": record.selected,
            }
            for record in knn_split.selection_records
        )
    return (
        policy_rows,
        summarize_cross_family_reversals(policy_rows),
        selection_rows,
    )


def run_personalization_tv_matched_policy_splits(
    utility: np.ndarray,
    *,
    outcome: str,
    split_seeds: Sequence[int],
    user_folds: int,
    history_action_count: int,
    rank: int,
    ridge_penalty: float,
    neighbor_fraction_grid: Sequence[float],
    inner_validation_fraction: float,
    target_tvs: Sequence[float],
    candidate_action_start: int | None = None,
) -> list[dict[str, object]]:
    """Run the score-only policy-strength calibration across fixed splits."""

    if not split_seeds:
        raise ValueError("at least one split seed is required")
    rows: list[dict[str, object]] = []
    for split_seed in split_seeds:
        factor_split = construct_cross_fitted_learned_policy_split(
            utility,
            user_folds=user_folds,
            history_action_count=history_action_count,
            rank=rank,
            ridge_penalty=ridge_penalty,
            split_seed=int(split_seed),
            candidate_action_start=candidate_action_start,
        )
        knn_split = construct_cross_fitted_knn_policy_split(
            utility,
            user_folds=user_folds,
            history_action_count=history_action_count,
            neighbor_fraction_grid=neighbor_fraction_grid,
            inner_validation_fraction=inner_validation_fraction,
            split_seed=int(split_seed),
            candidate_action_start=candidate_action_start,
        )
        rows.extend(
            evaluate_personalization_tv_matched_policies(
                factor_split,
                knn_split,
                outcome=outcome,
                target_tvs=target_tvs,
            )
        )
    return rows


def run_factor_hyperparameter_sensitivity(
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
    inverse_temperatures: Sequence[float],
    knn_policy_rows: Sequence[dict[str, object]],
    candidate_action_start: int | None = None,
) -> list[dict[str, object]]:
    """Compare a fixed MF grid with the already nested-selected kNN policy.

    The grid is reported in full.  It is a sensitivity analysis, not a second
    model-selection stage, so no rank/ridge configuration is chosen from the
    resulting candidate-outcome values.
    """

    ranks = tuple(sorted({int(value) for value in rank_grid}))
    ridge_penalties = tuple(sorted({float(value) for value in ridge_penalty_grid}))
    temperatures = tuple(float(value) for value in inverse_temperatures)
    seeds = tuple(int(value) for value in split_seeds)
    if not ranks or ranks[0] <= 0:
        raise ValueError("rank_grid must contain positive integers")
    if (
        not ridge_penalties
        or ridge_penalties[0] <= 0.0
        or not np.isfinite(ridge_penalties).all()
    ):
        raise ValueError("ridge_penalty_grid must contain finite positive values")
    if primary_rank not in ranks or float(primary_ridge_penalty) not in ridge_penalties:
        raise ValueError(
            "factor sensitivity grid must include the primary configuration"
        )
    if not seeds or not temperatures:
        raise ValueError("factor sensitivity requires seeds and temperatures")

    knn_lookup: dict[tuple[int, float], dict[str, object]] = {}
    for row in knn_policy_rows:
        if (
            str(row["outcome"]) != outcome
            or str(row["policy_family"]) != "centered_user_knn"
        ):
            continue
        key = (int(row["split_seed"]), float(row["inverse_temperature"]))
        if key in knn_lookup:
            raise ValueError(f"duplicate nested-selected kNN policy for {key}")
        knn_lookup[key] = row
    expected_keys = {
        (split_seed, inverse_temperature)
        for split_seed in seeds
        for inverse_temperature in temperatures
    }
    if set(knn_lookup) != expected_keys:
        raise ValueError("nested-selected kNN rows do not match the sensitivity grid")

    rows: list[dict[str, object]] = []
    for rank in ranks:
        for ridge_penalty in ridge_penalties:
            for split_seed in seeds:
                factor_split = construct_cross_fitted_learned_policy_split(
                    utility,
                    user_folds=user_folds,
                    history_action_count=history_action_count,
                    rank=rank,
                    ridge_penalty=ridge_penalty,
                    split_seed=split_seed,
                    candidate_action_start=candidate_action_start,
                )
                for inverse_temperature in temperatures:
                    knn = knn_lookup[(split_seed, inverse_temperature)]
                    if (
                        int(knn["user_folds"]) != factor_split.user_folds
                        or int(knn["evaluation_users"])
                        != factor_split.evaluation_user_indices.size
                        or int(knn["history_actions"])
                        != factor_split.history_action_indices.size
                        or int(knn["candidate_actions"])
                        != factor_split.candidate_action_indices.size
                        or int(knn["candidate_action_start"])
                        != factor_split.candidate_action_start
                        or str(knn["reference_stratum"]) != "outer_fold"
                    ):
                        raise ValueError(
                            "factor and reused kNN protocols are not aligned"
                        )
                    factor = _evaluate_frozen_score(
                        factor_split.candidate_interaction_score,
                        factor_split.candidate_utility,
                        evaluation_fold_ids=factor_split.evaluation_fold_ids,
                        inverse_temperature=inverse_temperature,
                    )
                    delta_overall = float(knn["overall_policy_value"]) - float(
                        factor["overall_policy_value"]
                    )
                    delta_reference = float(knn["reference_policy_value"]) - float(
                        factor["reference_policy_value"]
                    )
                    delta_matching = float(knn["matching_value"]) - float(
                        factor["matching_value"]
                    )
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
                            "factor_variant": "interaction_ridge_mf_quality0",
                            "factor_grid_role": "full_sensitivity_no_selection",
                            "knn_selection_rule": (
                                "reused_outer_training_inner_validation_rmse"
                            ),
                            "user_folds": factor_split.user_folds,
                            "evaluation_users": int(
                                factor_split.evaluation_user_indices.size
                            ),
                            "history_actions": int(
                                factor_split.history_action_indices.size
                            ),
                            "candidate_actions": int(
                                factor_split.candidate_action_indices.size
                            ),
                            "candidate_action_start": (
                                factor_split.candidate_action_start
                            ),
                            "reference_stratum": "outer_fold",
                            "inverse_temperature": inverse_temperature,
                            "factor_history_rmse": factor_split.history_rmse,
                            "selected_neighbor_fraction_min": float(
                                knn["selected_neighbor_fraction_min"]
                            ),
                            "selected_neighbor_fraction_max": float(
                                knn["selected_neighbor_fraction_max"]
                            ),
                            "selected_neighbor_fraction_mean": float(
                                knn["selected_neighbor_fraction_mean"]
                            ),
                            "factor_overall_policy_value": float(
                                factor["overall_policy_value"]
                            ),
                            "factor_reference_policy_value": float(
                                factor["reference_policy_value"]
                            ),
                            "factor_matching_value": float(
                                factor["matching_value"]
                            ),
                            "knn_overall_policy_value": float(
                                knn["overall_policy_value"]
                            ),
                            "knn_reference_policy_value": float(
                                knn["reference_policy_value"]
                            ),
                            "knn_matching_value": float(knn["matching_value"]),
                            "delta_overall_knn_minus_interaction_mf": delta_overall,
                            "delta_reference_knn_minus_interaction_mf": (
                                delta_reference
                            ),
                            "delta_matching_knn_minus_interaction_mf": delta_matching,
                            "decomposition_error": (
                                delta_overall - delta_reference - delta_matching
                            ),
                            "overall_preferred": (
                                "centered_user_knn"
                                if delta_overall > 0.0
                                else "interaction_ridge_mf"
                            ),
                            "matching_preferred": (
                                "centered_user_knn"
                                if delta_matching > 0.0
                                else "interaction_ridge_mf"
                            ),
                            "ranking_reversal": bool(
                                delta_overall * delta_matching < 0.0
                            ),
                        }
                    )
    return rows


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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-policy-csv", type=Path, required=True)
    parser.add_argument("--output-ranking-csv", type=Path, required=True)
    parser.add_argument("--output-cross-family-policy-csv", type=Path, required=True)
    parser.add_argument("--output-cross-family-ranking-csv", type=Path, required=True)
    parser.add_argument("--output-knn-selection-csv", type=Path, required=True)
    parser.add_argument("--output-history-sensitivity-csv", type=Path, required=True)
    parser.add_argument("--output-factor-sensitivity-csv", type=Path, required=True)
    parser.add_argument(
        "--output-personalization-tv-matched-csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--split-seeds", default="11,23,37,41,53,67,79,83,97,101")
    parser.add_argument("--user-folds", type=int, default=5)
    parser.add_argument("--history-action-count", type=int, default=750)
    parser.add_argument("--history-sensitivity-counts", default="250,500,750")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    parser.add_argument("--factor-rank-sensitivity-grid", default="8,16,32")
    parser.add_argument("--factor-ridge-sensitivity-grid", default="1,10,100")
    parser.add_argument("--quality-weight-grid", default="0,0.5,1,2,4,8")
    parser.add_argument("--cross-family-factor-quality-weights", default="0,1")
    parser.add_argument("--neighbor-fraction-grid", default="0.01,0.05,0.2,0.5,1")
    parser.add_argument("--inner-validation-fraction", type=float, default=0.2)
    parser.add_argument("--inverse-temperature-grid", default="2,4,8")
    parser.add_argument("--target-personalization-tv-grid", default="0.02,0.04,0.07")
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
    quality_weights = _parse_numbers(args.quality_weight_grid, cast=float)
    cross_family_factor_quality_weights = _parse_numbers(
        args.cross_family_factor_quality_weights,
        cast=float,
    )
    neighbor_fractions = _parse_numbers(args.neighbor_fraction_grid, cast=float)
    history_sensitivity_counts = _parse_numbers(
        args.history_sensitivity_counts,
        cast=int,
    )
    factor_sensitivity_ranks = _parse_numbers(
        args.factor_rank_sensitivity_grid,
        cast=int,
    )
    factor_sensitivity_ridges = _parse_numbers(
        args.factor_ridge_sensitivity_grid,
        cast=float,
    )
    target_personalization_tvs = _parse_numbers(
        args.target_personalization_tv_grid,
        cast=float,
    )
    if (
        args.history_action_count not in history_sensitivity_counts
        or min(history_sensitivity_counts) < 2
        or max(history_sensitivity_counts) > args.history_action_count
    ):
        raise ValueError(
            "history sensitivity counts must include and not exceed the primary "
            "history_action_count"
        )
    inverse_temperatures = _parse_numbers(
        args.inverse_temperature_grid,
        cast=float,
    )
    rows: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []
    cross_family_rows: list[dict[str, object]] = []
    cross_family_summaries: list[dict[str, object]] = []
    knn_selection_rows: list[dict[str, object]] = []
    for outcome, surface in outcome_surfaces.items():
        outcome_rows, outcome_summaries = run_learned_policy_splits(
            surface,
            outcome=outcome,
            split_seeds=split_seeds,
            user_folds=args.user_folds,
            history_action_count=args.history_action_count,
            rank=args.rank,
            ridge_penalty=args.ridge_penalty,
            quality_weights=quality_weights,
            inverse_temperatures=inverse_temperatures,
            candidate_action_start=args.history_action_count,
        )
        rows.extend(outcome_rows)
        summaries.extend(outcome_summaries)
        (
            outcome_cross_family_rows,
            outcome_cross_family_summaries,
            outcome_knn_selection_rows,
        ) = run_cross_family_policy_splits(
            surface,
            outcome=outcome,
            split_seeds=split_seeds,
            user_folds=args.user_folds,
            history_action_count=args.history_action_count,
            rank=args.rank,
            ridge_penalty=args.ridge_penalty,
            factor_quality_weights=cross_family_factor_quality_weights,
            neighbor_fraction_grid=neighbor_fractions,
            inner_validation_fraction=args.inner_validation_fraction,
            inverse_temperatures=inverse_temperatures,
            candidate_action_start=args.history_action_count,
        )
        cross_family_rows.extend(outcome_cross_family_rows)
        cross_family_summaries.extend(outcome_cross_family_summaries)
        knn_selection_rows.extend(outcome_knn_selection_rows)
    personalization_tv_matched_rows: list[dict[str, object]] = []
    for outcome, surface in outcome_surfaces.items():
        personalization_tv_matched_rows.extend(
            run_personalization_tv_matched_policy_splits(
                surface,
                outcome=outcome,
                split_seeds=split_seeds,
                user_folds=args.user_folds,
                history_action_count=args.history_action_count,
                rank=args.rank,
                ridge_penalty=args.ridge_penalty,
                neighbor_fraction_grid=neighbor_fractions,
                inner_validation_fraction=args.inner_validation_fraction,
                target_tvs=target_personalization_tvs,
                candidate_action_start=args.history_action_count,
            )
        )
    history_sensitivity_rows: list[dict[str, object]] = []
    primary_outcome = "capped_watch_ratio"
    for history_action_count in history_sensitivity_counts:
        if history_action_count == args.history_action_count:
            sensitivity_summaries = [
                row
                for row in cross_family_summaries
                if row["outcome"] == primary_outcome
            ]
            sensitivity_selection = [
                row
                for row in knn_selection_rows
                if row["outcome"] == primary_outcome
            ]
        else:
            (
                _,
                sensitivity_summaries,
                sensitivity_selection,
            ) = run_cross_family_policy_splits(
                outcome_surfaces[primary_outcome],
                outcome=primary_outcome,
                split_seeds=split_seeds,
                user_folds=args.user_folds,
                history_action_count=history_action_count,
                rank=args.rank,
                ridge_penalty=args.ridge_penalty,
                factor_quality_weights=cross_family_factor_quality_weights,
                neighbor_fraction_grid=neighbor_fractions,
                inner_validation_fraction=args.inner_validation_fraction,
                inverse_temperatures=inverse_temperatures,
                candidate_action_start=args.history_action_count,
            )
        history_sensitivity_rows.extend(
            annotate_history_sensitivity(
                sensitivity_summaries,
                sensitivity_selection,
                history_action_count=history_action_count,
                candidate_action_start=args.history_action_count,
            )
        )
    factor_sensitivity_rows = run_factor_hyperparameter_sensitivity(
        outcome_surfaces[primary_outcome],
        outcome=primary_outcome,
        split_seeds=split_seeds,
        user_folds=args.user_folds,
        history_action_count=args.history_action_count,
        rank_grid=factor_sensitivity_ranks,
        ridge_penalty_grid=factor_sensitivity_ridges,
        primary_rank=args.rank,
        primary_ridge_penalty=args.ridge_penalty,
        inverse_temperatures=inverse_temperatures,
        knn_policy_rows=cross_family_rows,
        candidate_action_start=args.history_action_count,
    )
    _write_rows(args.output_policy_csv, rows)
    _write_rows(args.output_ranking_csv, summaries)
    _write_rows(args.output_cross_family_policy_csv, cross_family_rows)
    _write_rows(args.output_cross_family_ranking_csv, cross_family_summaries)
    _write_rows(args.output_knn_selection_csv, knn_selection_rows)
    _write_rows(args.output_history_sensitivity_csv, history_sensitivity_rows)
    _write_rows(args.output_factor_sensitivity_csv, factor_sensitivity_rows)
    _write_rows(
        args.output_personalization_tv_matched_csv,
        personalization_tv_matched_rows,
    )
    payload = {
        "protocol": {
            "training_signal": "other-user folds plus held-out-user history block",
            "held_out_signal": "each user's candidate outcomes",
            "reference_marginalization": (
                "within outer fold so every reference mixes users scored by "
                "the same trained policy"
            ),
            "common_candidate_actions": int(common_actions.sum()),
            "evaluation_user_count": int(preferences.utility.shape[0]),
            "user_folds": args.user_folds,
            "history_action_count": args.history_action_count,
            "candidate_action_start": args.history_action_count,
            "history_sensitivity_counts": list(history_sensitivity_counts),
            "history_sensitivity_candidate_block": (
                "fixed at the primary candidate_action_start"
            ),
            "candidate_action_count": int(common_actions.sum())
            - args.history_action_count,
            "rank": args.rank,
            "ridge_penalty": args.ridge_penalty,
            "factor_rank_sensitivity_grid": list(factor_sensitivity_ranks),
            "factor_ridge_sensitivity_grid": list(factor_sensitivity_ridges),
            "factor_sensitivity_role": "full grid; no result-based selection",
            "cross_family_factor_quality_weights": list(
                cross_family_factor_quality_weights
            ),
            "knn_neighbor_fraction_grid": list(neighbor_fractions),
            "knn_inner_validation_fraction": args.inner_validation_fraction,
            "knn_selection_target": "outer-training inner-validation candidate RMSE",
            "outcomes": {
                "capped_watch_ratio": "clip(watch_ratio, 0, 1)",
                "complete_watch": "1[watch_ratio >= 1]",
            },
            "split_seeds": list(split_seeds),
            "quality_weights": list(quality_weights),
            "inverse_temperatures": list(inverse_temperatures),
            "target_personalization_tvs": list(target_personalization_tvs),
            "personalization_tv_calibration": (
                "fold-conditioned policy-score TV; no candidate outcomes"
            ),
        },
        "ranking_summaries": summaries,
        "cross_family_summaries": cross_family_summaries,
        "knn_selection": knn_selection_rows,
        "history_sensitivity": history_sensitivity_rows,
        "factor_hyperparameter_sensitivity": factor_sensitivity_rows,
        "personalization_tv_matched": personalization_tv_matched_rows,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
