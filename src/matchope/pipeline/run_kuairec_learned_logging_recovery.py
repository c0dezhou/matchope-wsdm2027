#!/usr/bin/env python3
from __future__ import annotations

"""Recover a learned-policy KuaiRec comparison from simulated sparse logs.

The target policies are the same cross-fitted MF and user-kNN policies used in
the dense KuaiRec comparison.  They are first calibrated to the same exposure
and TV.  We then sample actions from a known popularity-biased logger and draw
Bernoulli outcomes from the held-out dense matrix.  This connects the dense
comparison to the logged-feedback estimator without treating KuaiRec as a live
causal experiment.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import norm

from matchope.pipeline.run_kuairec_matching_controls import (
    calibrate_common_exposure_policy,
    fold_conditioned_marginal,
)
from matchope.pipeline.run_kuairec_matching_learned import (
    calibrate_inverse_temperature_to_personalization_tv,
    construct_cross_fitted_knn_policy_split,
    construct_cross_fitted_learned_policy_split,
)
from matchope.pipeline.run_kuairec_matching_oracle import (
    _softmax,
    load_preference_matrix,
)


@dataclass(frozen=True)
class LearnedLoggingDesign:
    factor_policy: np.ndarray
    knn_policy: np.ndarray
    reference_policy: np.ndarray
    outcome: np.ndarray
    outcome_prediction: np.ndarray
    user_context: np.ndarray
    logging_policy: np.ndarray
    logging_inverse_temperature: float
    target_tv: float
    factor_matching_value: float
    knn_matching_value: float
    matching_gap: float


def build_learned_logging_design(
    factor_score: np.ndarray,
    knn_score: np.ndarray,
    outcome: np.ndarray,
    outcome_prediction: np.ndarray,
    *,
    evaluation_fold_ids: np.ndarray,
    target_tv: float,
    logging_inverse_temperature: float,
) -> LearnedLoggingDesign:
    """Freeze common-exposure policies and a known popularity-biased logger."""

    factor = np.asarray(factor_score, dtype=np.float64)
    knn = np.asarray(knn_score, dtype=np.float64)
    utility = np.asarray(outcome, dtype=np.float64)
    q_hat = np.asarray(outcome_prediction, dtype=np.float64)
    context = np.asarray(evaluation_fold_ids, dtype=np.int64)
    if factor.shape != knn.shape or factor.shape != utility.shape:
        raise ValueError("factor scores, kNN scores, and outcomes must align")
    if q_hat.shape != utility.shape:
        raise ValueError("outcome predictions must align with outcomes")
    if context.shape != (utility.shape[0],):
        raise ValueError("evaluation fold IDs must align with users")
    if not np.isfinite(utility).all() or not np.isfinite(q_hat).all():
        raise ValueError("outcomes and outcome predictions must be finite")
    if not 0.0 < target_tv < 1.0:
        raise ValueError("target_tv must lie between zero and one")
    if logging_inverse_temperature < 0.0 or not np.isfinite(
        logging_inverse_temperature
    ):
        raise ValueError("logging temperature must be finite and non-negative")

    factor_beta = calibrate_inverse_temperature_to_personalization_tv(
        factor,
        evaluation_fold_ids=context,
        target_tv=target_tv,
    )
    knn_beta = calibrate_inverse_temperature_to_personalization_tv(
        knn,
        evaluation_fold_ids=context,
        target_tv=target_tv,
    )
    factor_marginal = fold_conditioned_marginal(
        _softmax(factor, factor_beta),
        context,
    )
    knn_marginal = fold_conditioned_marginal(
        _softmax(knn, knn_beta),
        context,
    )
    common_marginal = 0.5 * (factor_marginal + knn_marginal)
    _, factor_policy, _, _, _ = calibrate_common_exposure_policy(
        factor,
        evaluation_fold_ids=context,
        target_marginal=common_marginal,
        target_tv=target_tv,
        initial_upper=factor_beta,
    )
    _, knn_policy, _, _, _ = calibrate_common_exposure_policy(
        knn,
        evaluation_fold_ids=context,
        target_marginal=common_marginal,
        target_tv=target_tv,
        initial_upper=knn_beta,
    )

    labels = np.unique(context)
    if not np.array_equal(labels, np.arange(labels.size)):
        raise ValueError("evaluation fold IDs must be consecutive from zero")
    logging_policy = np.empty_like(common_marginal)
    for label in labels:
        context_mean = utility[context == label].mean(axis=0)
        if logging_inverse_temperature == 0.0:
            logging_policy[label] = 1.0 / utility.shape[1]
        else:
            logging_policy[label] = _softmax(
                context_mean[None, :],
                logging_inverse_temperature,
            )[0]

    reference_rows = common_marginal[context]
    factor_matching = float(
        np.mean(np.sum((factor_policy - reference_rows) * utility, axis=1))
    )
    knn_matching = float(
        np.mean(np.sum((knn_policy - reference_rows) * utility, axis=1))
    )
    return LearnedLoggingDesign(
        factor_policy=factor_policy,
        knn_policy=knn_policy,
        reference_policy=common_marginal,
        outcome=utility,
        outcome_prediction=np.clip(q_hat, 1e-4, 1.0 - 1e-4),
        user_context=context,
        logging_policy=logging_policy,
        logging_inverse_temperature=float(logging_inverse_temperature),
        target_tv=float(target_tv),
        factor_matching_value=factor_matching,
        knn_matching_value=knn_matching,
        matching_gap=knn_matching - factor_matching,
    )


def _sample_logged_data(
    design: LearnedLoggingDesign,
    *,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_users, _ = design.outcome.shape
    users = rng.integers(n_users, size=sample_size)
    contexts = design.user_context[users]
    actions = np.empty(sample_size, dtype=np.int64)
    for context in np.unique(design.user_context):
        selected = contexts == context
        actions[selected] = rng.choice(
            design.outcome.shape[1],
            size=int(selected.sum()),
            p=design.logging_policy[context],
        )
    means = design.outcome[users, actions]
    rewards = rng.binomial(1, means).astype(np.float64)
    order = rng.permutation(sample_size)
    evaluation_folds = np.empty(sample_size, dtype=np.int64)
    evaluation_folds[order] = np.arange(sample_size) % 5
    return users, actions, rewards, evaluation_folds


def _estimate_scores(
    design: LearnedLoggingDesign,
    *,
    users: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    evaluation_folds: np.ndarray,
) -> dict[str, np.ndarray]:
    sample_size = users.size
    n_users, _ = design.outcome.shape
    contexts = design.user_context[users]
    scores = {
        "centered_factor": np.empty(sample_size, dtype=np.float64),
        "centered_knn": np.empty(sample_size, dtype=np.float64),
        "ipw_factor": np.empty(sample_size, dtype=np.float64),
        "ipw_knn": np.empty(sample_size, dtype=np.float64),
    }
    for held_out_fold in range(5):
        training = evaluation_folds != held_out_fold
        held_out = ~training
        training_counts = np.bincount(users[training], minlength=n_users).astype(
            np.float64
        )
        factor_marginal = np.empty_like(design.reference_policy)
        knn_marginal = np.empty_like(design.reference_policy)
        outcome_marginal = np.empty_like(design.reference_policy)
        for context in np.unique(design.user_context):
            context_users = np.flatnonzero(design.user_context == context)
            weights = training_counts[context_users]
            total = float(weights.sum())
            if total <= 0.0:
                raise RuntimeError("a training fold omitted a shared context")
            factor_marginal[context] = (
                weights @ design.factor_policy[context_users]
            ) / total
            knn_marginal[context] = (
                weights @ design.knn_policy[context_users]
            ) / total
            outcome_marginal[context] = (
                weights @ design.outcome_prediction[context_users]
            ) / total

        factor_direct = np.sum(
            (
                design.factor_policy
                - factor_marginal[design.user_context]
            )
            * (
                design.outcome_prediction
                - outcome_marginal[design.user_context]
            ),
            axis=1,
        )
        knn_direct = np.sum(
            (design.knn_policy - knn_marginal[design.user_context])
            * (
                design.outcome_prediction
                - outcome_marginal[design.user_context]
            ),
            axis=1,
        )
        held_users = users[held_out]
        held_actions = actions[held_out]
        held_contexts = contexts[held_out]
        pscore = design.logging_policy[held_contexts, held_actions]
        residual = (
            rewards[held_out]
            - design.outcome_prediction[held_users, held_actions]
        )
        factor_contrast = (
            design.factor_policy[held_users, held_actions]
            - factor_marginal[held_contexts, held_actions]
        )
        knn_contrast = (
            design.knn_policy[held_users, held_actions]
            - knn_marginal[held_contexts, held_actions]
        )
        scores["centered_factor"][held_out] = (
            factor_direct[held_users] + factor_contrast / pscore * residual
        )
        scores["centered_knn"][held_out] = (
            knn_direct[held_users] + knn_contrast / pscore * residual
        )
        scores["ipw_factor"][held_out] = (
            factor_contrast / pscore * rewards[held_out]
        )
        scores["ipw_knn"][held_out] = (
            knn_contrast / pscore * rewards[held_out]
        )
    return scores


def run_learned_logging_recovery(
    design: LearnedLoggingDesign,
    *,
    sample_sizes: Sequence[int],
    repetitions: int,
    seed: int,
) -> list[dict[str, object]]:
    """Estimate both learned-policy values and their gap from simulated logs."""

    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    critical = float(norm.ppf(0.975))
    rng = np.random.default_rng(seed)
    truths = {
        "factor_matching": design.factor_matching_value,
        "knn_matching": design.knn_matching_value,
        "knn_minus_factor": design.matching_gap,
    }
    rows: list[dict[str, object]] = []
    for sample_size in sample_sizes:
        if sample_size < 100:
            raise ValueError("sample sizes must be at least 100")
        estimates = {
            (method, estimand): []
            for method in ("centered_score", "ipw")
            for estimand in truths
        }
        standard_errors = {key: [] for key in estimates}
        coverages = {key: [] for key in estimates}
        correct_signs = {key: [] for key in estimates}
        for _ in range(repetitions):
            users, actions, rewards, folds = _sample_logged_data(
                design,
                sample_size=int(sample_size),
                rng=rng,
            )
            scores = _estimate_scores(
                design,
                users=users,
                actions=actions,
                rewards=rewards,
                evaluation_folds=folds,
            )
            score_sets = {
                ("centered_score", "factor_matching"): scores["centered_factor"],
                ("centered_score", "knn_matching"): scores["centered_knn"],
                ("centered_score", "knn_minus_factor"): (
                    scores["centered_knn"] - scores["centered_factor"]
                ),
                ("ipw", "factor_matching"): scores["ipw_factor"],
                ("ipw", "knn_matching"): scores["ipw_knn"],
                ("ipw", "knn_minus_factor"): (
                    scores["ipw_knn"] - scores["ipw_factor"]
                ),
            }
            for key, score in score_sets.items():
                truth = truths[key[1]]
                estimate = float(score.mean())
                standard_error = float(score.std(ddof=1) / np.sqrt(sample_size))
                estimates[key].append(estimate)
                standard_errors[key].append(standard_error)
                coverages[key].append(
                    estimate - critical * standard_error
                    <= truth
                    <= estimate + critical * standard_error
                )
                correct_signs[key].append(
                    np.sign(estimate) == np.sign(truth)
                )
        for method in ("centered_score", "ipw"):
            for estimand, truth in truths.items():
                key = (method, estimand)
                estimate_array = np.asarray(estimates[key])
                rows.append(
                    {
                        "method": method,
                        "estimand": estimand,
                        "sample_size": int(sample_size),
                        "repetitions": int(repetitions),
                        "target_tv": design.target_tv,
                        "logging_inverse_temperature": (
                            design.logging_inverse_temperature
                        ),
                        "minimum_logging_probability": float(
                            design.logging_policy.min()
                        ),
                        "true_value": float(truth),
                        "mean_estimate": float(estimate_array.mean()),
                        "bias": float(estimate_array.mean() - truth),
                        "rmse": float(
                            np.sqrt(np.mean(np.square(estimate_array - truth)))
                        ),
                        "empirical_standard_deviation": float(
                            estimate_array.std(ddof=1)
                        ),
                        "mean_standard_error": float(
                            np.mean(standard_errors[key])
                        ),
                        "coverage_95pct": float(np.mean(coverages[key])),
                        "correct_sign_rate": float(np.mean(correct_signs[key])),
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--user-folds", type=int, default=5)
    parser.add_argument("--history-action-count", type=int, default=750)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    parser.add_argument("--neighbor-fraction-grid", default="0.01,0.05,0.2,0.5,1")
    parser.add_argument("--inner-validation-fraction", type=float, default=0.2)
    parser.add_argument("--target-tv", type=float, default=0.04)
    parser.add_argument("--logging-inverse-temperature", type=float, default=4.0)
    parser.add_argument("--sample-size-grid", default="50000,200000")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    preferences = load_preference_matrix(args.csv)
    common_actions = np.all(np.isfinite(preferences.utility), axis=0)
    complete_watch = (
        preferences.ranking_score[:, common_actions] >= 1.0
    ).astype(np.float64)
    factor_split = construct_cross_fitted_learned_policy_split(
        complete_watch,
        user_folds=args.user_folds,
        history_action_count=args.history_action_count,
        rank=args.rank,
        ridge_penalty=args.ridge_penalty,
        split_seed=args.split_seed,
    )
    knn_split = construct_cross_fitted_knn_policy_split(
        complete_watch,
        user_folds=args.user_folds,
        history_action_count=args.history_action_count,
        neighbor_fraction_grid=_parse_numbers(
            args.neighbor_fraction_grid,
            cast=float,
        ),
        inner_validation_fraction=args.inner_validation_fraction,
        split_seed=args.split_seed,
    )
    if not np.array_equal(
        factor_split.evaluation_user_indices,
        knn_split.evaluation_user_indices,
    ) or not np.array_equal(
        factor_split.evaluation_fold_ids,
        knn_split.evaluation_fold_ids,
    ):
        raise ValueError("learned policy rows are not aligned")
    design = build_learned_logging_design(
        factor_split.candidate_interaction_score,
        knn_split.candidate_score,
        factor_split.candidate_utility,
        knn_split.candidate_score,
        evaluation_fold_ids=factor_split.evaluation_fold_ids,
        target_tv=args.target_tv,
        logging_inverse_temperature=args.logging_inverse_temperature,
    )
    rows = run_learned_logging_recovery(
        design,
        sample_sizes=_parse_numbers(args.sample_size_grid, cast=int),
        repetitions=args.repetitions,
        seed=args.seed,
    )
    for row in rows:
        row["split_seed"] = args.split_seed
        row["outcome"] = "complete_watch"
    _write_rows(args.output_csv, rows)
    payload = {
        "protocol": {
            "outcome": "complete_watch",
            "split_seed": args.split_seed,
            "target_tv": args.target_tv,
            "logging_inverse_temperature": args.logging_inverse_temperature,
            "sample_sizes": list(_parse_numbers(args.sample_size_grid, cast=int)),
            "repetitions": args.repetitions,
            "seed": args.seed,
            "outcome_model": (
                "cross-fitted user-kNN candidate prediction clipped to [1e-4, 1-1e-4]"
            ),
            "logger": "known context-specific softmax of candidate popularity",
            "role": (
                "semisynthetic bridge from the dense learned-policy comparison "
                "to sparse logged-feedback estimation"
            ),
        },
        "truth": {
            "factor_matching": design.factor_matching_value,
            "knn_matching": design.knn_matching_value,
            "knn_minus_factor": design.matching_gap,
        },
        "results": rows,
    }
    text = json.dumps(payload, indent=2, allow_nan=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
