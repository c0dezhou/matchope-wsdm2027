#!/usr/bin/env python3
from __future__ import annotations

"""Audit KuaiRec eligibility and recover matching value from simulated logs.

KuaiRec's small matrix is used as a dense preference surface, not as causal
potential-outcome ground truth.  The outcome is capped watch ratio in [0, 1].
Only videos observed for every user are eligible.  This gives the marginalized
reference one shared, feasible candidate set without treating blocked/missing
videos as zero utility or introducing an arbitrary user partition.

The target is an oracle softmax of the dense preference surface.  It is a
positive-control policy for estimator recovery, not a learned deployable
recommender.  Logged actions and Bernoulli outcomes are then simulated under a
known uniform logging design.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from matchope.eval.matching_value import (
    contrast_identification_interval,
    contrast_support_certificate,
    separate_value_difference_interval,
)


@dataclass(frozen=True)
class PreferenceMatrix:
    user_ids: np.ndarray
    video_ids: np.ndarray
    utility: np.ndarray
    ranking_score: np.ndarray
    observed_pairs: int


@dataclass(frozen=True)
class OracleCohort:
    user_indices: np.ndarray
    action_indices: np.ndarray
    utility: np.ndarray
    target_policy: np.ndarray
    marginal_policy: np.ndarray
    marginal_utility: np.ndarray
    centered_direct: np.ndarray


def load_preference_matrix(csv_path: Path) -> PreferenceMatrix:
    """Load unique user-video watch ratios into a dense matrix with NaN holes."""

    frame = pd.read_csv(
        csv_path,
        usecols=["user_id", "video_id", "watch_ratio"],
        dtype={"user_id": np.int32, "video_id": np.int32, "watch_ratio": np.float32},
    )
    if frame.empty:
        raise ValueError("KuaiRec small matrix is empty")
    if frame.duplicated(["user_id", "video_id"]).any():
        raise ValueError("KuaiRec small matrix contains duplicate user-video pairs")
    if not np.isfinite(frame["watch_ratio"].to_numpy()).all():
        raise ValueError("watch_ratio contains non-finite values")

    user_ids, user_index = np.unique(
        frame["user_id"].to_numpy(),
        return_inverse=True,
    )
    video_ids, video_index = np.unique(
        frame["video_id"].to_numpy(),
        return_inverse=True,
    )
    shape = (user_ids.size, video_ids.size)
    raw_watch_ratio = frame["watch_ratio"].to_numpy(dtype=np.float32)
    ranking_score = np.full(shape, np.nan, dtype=np.float32)
    ranking_score[user_index, video_index] = raw_watch_ratio
    utility = np.full(shape, np.nan, dtype=np.float32)
    utility[user_index, video_index] = np.clip(
        raw_watch_ratio,
        0.0,
        1.0,
    )
    return PreferenceMatrix(
        user_ids=user_ids,
        video_ids=video_ids,
        utility=utility,
        ranking_score=ranking_score,
        observed_pairs=int(frame.shape[0]),
    )


def _softmax(values: np.ndarray, inverse_temperature: float) -> np.ndarray:
    if inverse_temperature <= 0.0 or not np.isfinite(inverse_temperature):
        raise ValueError("inverse_temperature must be finite and positive")
    logits = inverse_temperature * np.asarray(values, dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def construct_oracle_cohorts(
    utility: np.ndarray,
    *,
    cohort_size: int,
    inverse_temperature: float,
) -> list[OracleCohort]:
    """Build fixed-eligibility strata and their oracle target/reference policies."""

    surface = np.asarray(utility, dtype=np.float64)
    if surface.ndim != 2 or min(surface.shape) < 2:
        raise ValueError("utility must be a two-dimensional user-action surface")
    if cohort_size < 2:
        raise ValueError("cohort_size must be at least two")
    cohorts: list[OracleCohort] = []
    for start in range(0, surface.shape[0], cohort_size):
        user_indices = np.arange(start, min(start + cohort_size, surface.shape[0]))
        if user_indices.size < 2:
            raise ValueError("the final eligibility cohort contains only one user")
        eligible = np.all(np.isfinite(surface[user_indices]), axis=0)
        action_indices = np.flatnonzero(eligible)
        if action_indices.size < 2:
            raise ValueError(
                "an eligibility cohort has fewer than two commonly observed actions"
            )
        cohort_utility = surface[np.ix_(user_indices, action_indices)]
        target = _softmax(cohort_utility, inverse_temperature)
        marginal_policy = target.mean(axis=0)
        marginal_utility = cohort_utility.mean(axis=0)
        contrast = target - marginal_policy
        centered_direct = np.sum(
            contrast * (cohort_utility - marginal_utility),
            axis=1,
        )
        cohorts.append(
            OracleCohort(
                user_indices=user_indices,
                action_indices=action_indices,
                utility=cohort_utility,
                target_policy=target,
                marginal_policy=marginal_policy,
                marginal_utility=marginal_utility,
                centered_direct=centered_direct,
            )
        )
    return cohorts


def oracle_matching_value(cohorts: Sequence[OracleCohort]) -> float:
    """Return matching value under a uniform distribution over retained users."""

    if not cohorts:
        raise ValueError("at least one cohort is required")
    total_users = sum(cohort.user_indices.size for cohort in cohorts)
    return float(
        sum(cohort.centered_direct.sum() for cohort in cohorts) / total_users
    )


def eligibility_audit(
    preferences: PreferenceMatrix,
    cohorts: Sequence[OracleCohort],
    *,
    cohort_size: int,
    inverse_temperature: float,
) -> dict[str, float | int]:
    """Summarize original density and the fixed-candidate cohort restriction."""

    total_pairs = int(preferences.utility.size)
    eligible_actions = np.asarray(
        [cohort.action_indices.size for cohort in cohorts],
        dtype=np.int64,
    )
    retained_pairs = int(
        sum(
            cohort.user_indices.size * cohort.action_indices.size
            for cohort in cohorts
        )
    )
    globally_common = int(np.all(np.isfinite(preferences.utility), axis=0).sum())
    return {
        "users": int(preferences.user_ids.size),
        "videos": int(preferences.video_ids.size),
        "observed_pairs": preferences.observed_pairs,
        "total_pairs": total_pairs,
        "observed_density": preferences.observed_pairs / total_pairs,
        "missing_pairs": total_pairs - preferences.observed_pairs,
        "globally_common_videos": globally_common,
        "cohort_size": cohort_size,
        "cohorts": len(cohorts),
        "minimum_common_videos": int(eligible_actions.min()),
        "median_common_videos": float(np.median(eligible_actions)),
        "maximum_common_videos": int(eligible_actions.max()),
        "retained_pairs": retained_pairs,
        "retained_pair_fraction": retained_pairs / total_pairs,
        "inverse_temperature": inverse_temperature,
        "oracle_matching_value": oracle_matching_value(cohorts),
    }


def simulate_logging_scores(
    cohorts: Sequence[OracleCohort],
    *,
    sample_size: int,
    rng: np.random.Generator,
    logging_inverse_temperature: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate centered-DR and IPS scores under popularity-biased logging."""

    if sample_size < 2:
        raise ValueError("sample_size must be at least two")
    if logging_inverse_temperature < 0.0 or not np.isfinite(
        logging_inverse_temperature
    ):
        raise ValueError(
            "logging_inverse_temperature must be finite and non-negative"
        )
    total_users = sum(cohort.user_indices.size for cohort in cohorts)
    sampled_users = rng.integers(total_users, size=sample_size)
    centered_dr = np.empty(sample_size, dtype=np.float64)
    ips = np.empty(sample_size, dtype=np.float64)
    offset = 0
    for cohort in cohorts:
        cohort_users = cohort.user_indices.size
        selected = (sampled_users >= offset) & (sampled_users < offset + cohort_users)
        local_users = sampled_users[selected] - offset
        n_selected = int(selected.sum())
        if n_selected:
            n_actions = cohort.action_indices.size
            logging_probability = _softmax(
                cohort.marginal_utility[None, :],
                logging_inverse_temperature,
            )[0] if logging_inverse_temperature > 0.0 else np.full(
                n_actions,
                1.0 / n_actions,
            )
            local_actions = rng.choice(
                n_actions,
                size=n_selected,
                p=logging_probability,
            )
            means = cohort.utility[local_users, local_actions]
            rewards = rng.binomial(1, means).astype(np.float64)
            contrast = (
                cohort.target_policy[local_users, local_actions]
                - cohort.marginal_policy[local_actions]
            )
            pscore = logging_probability[local_actions]
            correction = contrast / pscore * (rewards - means)
            centered_dr[selected] = cohort.centered_direct[local_users] + correction
            ips[selected] = contrast / pscore * rewards
        offset += cohort_users
    return centered_dr, ips


def simulate_uniform_logging_scores(
    cohorts: Sequence[OracleCohort],
    *,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for the uniform logging design."""

    return simulate_logging_scores(
        cohorts,
        sample_size=sample_size,
        rng=rng,
        logging_inverse_temperature=0.0,
    )


def run_recovery_grid(
    cohorts: Sequence[OracleCohort],
    *,
    sample_sizes: Sequence[int],
    repetitions: int,
    seed: int,
    logging_inverse_temperatures: Sequence[float] = (0.0,),
) -> list[dict[str, float | int | str]]:
    """Measure finite-sample recovery against the exact dense oracle value."""

    if repetitions < 2:
        raise ValueError("repetitions must be at least two")
    truth = oracle_matching_value(cohorts)
    critical = float(norm.ppf(0.975))
    rng = np.random.default_rng(seed)
    results: list[dict[str, float | int | str]] = []
    for logging_inverse_temperature in logging_inverse_temperatures:
        if logging_inverse_temperature < 0.0 or not np.isfinite(
            logging_inverse_temperature
        ):
            raise ValueError("logging temperatures must be finite and non-negative")
        for sample_size in sample_sizes:
            if sample_size < 2:
                raise ValueError("all sample sizes must be at least two")
            method_estimates = {"centered_dr": [], "ips_contrast": []}
            method_standard_errors = {"centered_dr": [], "ips_contrast": []}
            method_coverages = {"centered_dr": [], "ips_contrast": []}
            method_detections = {"centered_dr": [], "ips_contrast": []}
            minimum_logging_probability: float | None = None
            for _ in range(repetitions):
                centered_dr, ips = simulate_logging_scores(
                    cohorts,
                    sample_size=sample_size,
                    rng=rng,
                    logging_inverse_temperature=float(
                        logging_inverse_temperature
                    ),
                )
                if minimum_logging_probability is None:
                    minimum_logging_probability = min(
                        float(
                            _softmax(
                                cohort.marginal_utility[None, :],
                                float(logging_inverse_temperature),
                            ).min()
                        )
                        if logging_inverse_temperature > 0.0
                        else 1.0 / cohort.action_indices.size
                        for cohort in cohorts
                    )
                for method, scores in (
                    ("centered_dr", centered_dr),
                    ("ips_contrast", ips),
                ):
                    estimate = float(scores.mean())
                    standard_error = float(scores.std(ddof=1) / np.sqrt(sample_size))
                    lower = estimate - critical * standard_error
                    upper = estimate + critical * standard_error
                    method_estimates[method].append(estimate)
                    method_standard_errors[method].append(standard_error)
                    method_coverages[method].append(lower <= truth <= upper)
                    method_detections[method].append(lower > 0.0)
            for method in method_estimates:
                estimates = np.asarray(method_estimates[method])
                results.append(
                    {
                        "method": method,
                        "sample_size": int(sample_size),
                        "repetitions": repetitions,
                        "logging_inverse_temperature": float(
                            logging_inverse_temperature
                        ),
                        "minimum_logging_probability": minimum_logging_probability,
                        "oracle_matching_value": truth,
                        "mean_estimate": float(estimates.mean()),
                        "bias": float(estimates.mean() - truth),
                        "rmse": float(
                            np.sqrt(np.mean(np.square(estimates - truth)))
                        ),
                        "mean_standard_error": float(
                            np.mean(method_standard_errors[method])
                        ),
                        "coverage_95pct": float(
                            np.mean(method_coverages[method])
                        ),
                        "positive_detection_rate": float(
                            np.mean(method_detections[method])
                        ),
                    }
                )
    return results


def run_popularity_support_degradation(
    cohort: OracleCohort,
    *,
    supported_action_fractions: Sequence[float],
    logging_inverse_temperature: float,
) -> list[dict[str, float | int | bool]]:
    """Remove increasingly unpopular actions from structural logging support."""

    if logging_inverse_temperature < 0.0 or not np.isfinite(
        logging_inverse_temperature
    ):
        raise ValueError("logging_inverse_temperature must be finite and non-negative")
    n_users, n_actions = cohort.utility.shape
    popularity_order = np.argsort(-cohort.marginal_utility)
    target = cohort.target_policy
    reference = np.broadcast_to(cohort.marginal_policy, target.shape)
    truth = float(np.mean(np.sum((target - reference) * cohort.utility, axis=1)))
    rows: list[dict[str, float | int | bool]] = []
    for supported_fraction in supported_action_fractions:
        if not 0.0 < supported_fraction <= 1.0:
            raise ValueError("supported action fractions must lie in (0, 1]")
        n_supported = max(1, int(np.ceil(supported_fraction * n_actions)))
        supported_actions = popularity_order[:n_supported]
        logging_row = np.zeros(n_actions, dtype=np.float64)
        if logging_inverse_temperature == 0.0:
            logging_row[supported_actions] = 1.0 / n_supported
        else:
            probabilities = _softmax(
                cohort.marginal_utility[supported_actions][None, :],
                logging_inverse_temperature,
            )[0]
            logging_row[supported_actions] = probabilities
        logging = np.broadcast_to(logging_row, target.shape)
        certificate = contrast_support_certificate(target, reference, logging)
        direct = contrast_identification_interval(
            target,
            reference,
            logging,
            cohort.utility,
            outcome_lower=0.0,
            outcome_upper=1.0,
        )
        separate = separate_value_difference_interval(
            target,
            reference,
            logging,
            cohort.utility,
            outcome_lower=0.0,
            outcome_upper=1.0,
        )
        rows.append(
            {
                "users": n_users,
                "actions": n_actions,
                "supported_actions": n_supported,
                "supported_action_fraction": n_supported / n_actions,
                "logging_inverse_temperature": logging_inverse_temperature,
                "oracle_matching_value": truth,
                "contrast_point_identified": certificate.point_identified,
                "affected_row_fraction": certificate.affected_row_fraction,
                "unsupported_target_mass": certificate.unsupported_target_mass,
                "unsupported_reference_mass": certificate.unsupported_reference_mass,
                "unsupported_contrast_mass": certificate.unsupported_contrast_mass,
                "direct_lower": direct.lower,
                "direct_upper": direct.upper,
                "direct_width": direct.width,
                "separate_value_width": separate.width,
                "oracle_inside_direct_interval": (
                    direct.lower - 1e-12 <= truth <= direct.upper + 1e-12
                ),
            }
        )
    return rows


def run_support_stress_cases(
    cohort: OracleCohort,
    ranking_score: np.ndarray,
    *,
    common_mass: float = 0.30,
    unsupported_action_fraction: float = 0.25,
    sparse_top_k: int = 50,
) -> list[dict[str, float | int | str | bool]]:
    """Construct two support reversals on a real dense preference surface."""

    scores = np.asarray(ranking_score, dtype=np.float64)
    if scores.shape != cohort.utility.shape or not np.isfinite(scores).all():
        raise ValueError("ranking_score must be finite and aligned with the cohort")
    if not 0.0 < common_mass < 1.0:
        raise ValueError("common_mass must lie in (0, 1)")
    if not 0.0 < unsupported_action_fraction < 1.0:
        raise ValueError("unsupported_action_fraction must lie in (0, 1)")
    n_users, n_actions = cohort.utility.shape
    n_unsupported = int(np.floor(unsupported_action_fraction * n_actions))
    if not 1 <= n_unsupported < n_actions - 1:
        raise ValueError("unsupported action block leaves fewer than two supported actions")
    if not 1 <= sparse_top_k < n_actions:
        raise ValueError("sparse_top_k must be smaller than the action count")

    def summarize(
        case: str,
        target: np.ndarray,
        reference: np.ndarray,
        logging: np.ndarray,
    ) -> dict[str, float | int | str | bool]:
        truth = float(np.mean(np.sum((target - reference) * cohort.utility, axis=1)))
        certificate = contrast_support_certificate(target, reference, logging)
        direct = contrast_identification_interval(
            target,
            reference,
            logging,
            cohort.utility,
            outcome_lower=0.0,
            outcome_upper=1.0,
        )
        separate = separate_value_difference_interval(
            target,
            reference,
            logging,
            cohort.utility,
            outcome_lower=0.0,
            outcome_upper=1.0,
        )
        unsupported = logging == 0.0
        target_point_identified = not bool(np.any(unsupported & (target > 1e-12)))
        reference_point_identified = not bool(
            np.any(unsupported & (reference > 1e-12))
        )
        return {
            "case": case,
            "users": n_users,
            "actions": n_actions,
            "oracle_matching_value": truth,
            "target_value_point_identified": target_point_identified,
            "reference_value_point_identified": reference_point_identified,
            "contrast_point_identified": certificate.point_identified,
            "unsupported_target_mass": certificate.unsupported_target_mass,
            "unsupported_reference_mass": certificate.unsupported_reference_mass,
            "unsupported_contrast_mass": certificate.unsupported_contrast_mass,
            "direct_lower": direct.lower,
            "direct_upper": direct.upper,
            "direct_width": direct.width,
            "separate_value_width": separate.width,
            "oracle_inside_direct_interval": (
                direct.lower - 1e-12 <= truth <= direct.upper + 1e-12
            ),
        }

    unsupported_actions = np.arange(n_unsupported)
    supported_actions = np.arange(n_unsupported, n_actions)
    personalized_supported = _softmax(
        cohort.utility[:, supported_actions],
        inverse_temperature=2.0,
    )
    common_target = np.zeros((n_users, n_actions), dtype=np.float64)
    common_target[:, unsupported_actions] = common_mass / n_unsupported
    common_target[:, supported_actions] = (
        1.0 - common_mass
    ) * personalized_supported
    common_reference = np.broadcast_to(
        common_target.mean(axis=0),
        common_target.shape,
    ).copy()
    common_reference[:, unsupported_actions] = common_target[
        0,
        unsupported_actions,
    ]
    common_logging = np.zeros_like(common_target)
    common_logging[:, supported_actions] = 1.0 / supported_actions.size

    top_indices = np.argpartition(-scores, sparse_top_k - 1, axis=1)[
        :, :sparse_top_k
    ]
    sparse_target = np.zeros((n_users, n_actions), dtype=np.float64)
    np.put_along_axis(
        sparse_target,
        top_indices,
        np.full_like(top_indices, 1.0 / sparse_top_k, dtype=np.float64),
        axis=1,
    )
    sparse_reference = np.broadcast_to(
        sparse_target.mean(axis=0),
        sparse_target.shape,
    ).copy()
    sparse_logging = sparse_target.copy()

    return [
        summarize(
            "common_unsupported_allocation_cancels",
            common_target,
            common_reference,
            common_logging,
        ),
        summarize(
            "marginal_comparator_breaks_target_support",
            sparse_target,
            sparse_reference,
            sparse_logging,
        ),
    ]


def _parse_numbers(raw: str, *, cast: type[int] | type[float]) -> tuple[int, ...] | tuple[float, ...]:
    values = tuple(cast(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        raise ValueError("numeric grid must not be empty")
    return values


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-audit-csv", type=Path, required=True)
    parser.add_argument("--output-recovery-csv", type=Path, required=True)
    parser.add_argument("--output-support-csv", type=Path, required=True)
    parser.add_argument("--output-degradation-csv", type=Path, default=None)
    parser.add_argument("--inverse-temperature-grid", default="1.0,2.0,4.0")
    parser.add_argument("--sample-size-grid", default="1000,10000,50000")
    parser.add_argument("--logging-temperature-grid", default="0.0")
    parser.add_argument(
        "--supported-action-fraction-grid",
        default="1.0,0.9,0.75,0.5,0.25,0.1",
    )
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    preferences = load_preference_matrix(args.csv)
    cohort_size = int(preferences.user_ids.size)
    temperatures = _parse_numbers(args.inverse_temperature_grid, cast=float)
    sample_sizes = _parse_numbers(args.sample_size_grid, cast=int)
    logging_temperatures = _parse_numbers(
        args.logging_temperature_grid,
        cast=float,
    )
    audits: list[dict[str, object]] = []
    recovery: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] | None = None
    for inverse_temperature in temperatures:
        cohorts = construct_oracle_cohorts(
            preferences.utility,
            cohort_size=cohort_size,
            inverse_temperature=float(inverse_temperature),
        )
        audits.append(
            eligibility_audit(
                preferences,
                cohorts,
                cohort_size=cohort_size,
                inverse_temperature=float(inverse_temperature),
            )
        )
        rows = run_recovery_grid(
            cohorts,
            sample_sizes=sample_sizes,
            repetitions=args.repetitions,
            seed=args.seed,
            logging_inverse_temperatures=logging_temperatures,
        )
        for row in rows:
            row["inverse_temperature"] = float(inverse_temperature)
        recovery.extend(rows)
        if support_rows is None and len(cohorts) == 1:
            cohort = cohorts[0]
            cohort_ranking_score = preferences.ranking_score[
                np.ix_(cohort.user_indices, cohort.action_indices)
            ]
            support_rows = run_support_stress_cases(
                cohort,
                cohort_ranking_score,
            )
    _write_rows(args.output_audit_csv, audits)
    _write_rows(args.output_recovery_csv, recovery)
    if support_rows is None:
        raise ValueError("support stress test requires one global eligibility cohort")
    _write_rows(args.output_support_csv, support_rows)
    if args.output_degradation_csv is not None:
        degradation_cohorts = construct_oracle_cohorts(
            preferences.utility,
            cohort_size=cohort_size,
            inverse_temperature=float(temperatures[-1]),
        )
        degradation_rows = run_popularity_support_degradation(
            degradation_cohorts[0],
            supported_action_fractions=_parse_numbers(
                args.supported_action_fraction_grid,
                cast=float,
            ),
            logging_inverse_temperature=float(logging_temperatures[-1]),
        )
        _write_rows(args.output_degradation_csv, degradation_rows)


if __name__ == "__main__":
    main()
