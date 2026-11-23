from __future__ import annotations

import numpy as np

from matchope.pipeline.run_kuairec_matching_oracle import (
    PreferenceMatrix,
    construct_oracle_cohorts,
    eligibility_audit,
    oracle_matching_value,
    run_popularity_support_degradation,
    run_recovery_grid,
    run_support_stress_cases,
)


def test_additive_surface_has_zero_oracle_matching_value() -> None:
    user_effect = np.asarray([0.0, 0.1, 0.2, 0.3])[:, None]
    action_effect = np.asarray([0.1, 0.2, 0.3])[None, :]
    cohorts = construct_oracle_cohorts(
        user_effect + action_effect,
        cohort_size=2,
        inverse_temperature=2.0,
    )

    assert abs(oracle_matching_value(cohorts)) < 1e-14


def test_eligibility_is_intersected_within_each_cohort() -> None:
    utility = np.asarray(
        [
            [0.0, 0.2, np.nan, 0.4],
            [0.1, 0.3, 0.5, 0.6],
            [0.2, np.nan, 0.7, 0.8],
            [0.3, 0.4, 0.8, 0.9],
        ]
    )
    cohorts = construct_oracle_cohorts(
        utility,
        cohort_size=2,
        inverse_temperature=2.0,
    )

    np.testing.assert_array_equal(cohorts[0].action_indices, [0, 1, 3])
    np.testing.assert_array_equal(cohorts[1].action_indices, [0, 2, 3])
    audit = eligibility_audit(
        PreferenceMatrix(
            user_ids=np.arange(4),
            video_ids=np.arange(4),
            utility=utility,
            ranking_score=utility,
            observed_pairs=14,
        ),
        cohorts,
        cohort_size=2,
        inverse_temperature=2.0,
    )
    assert audit["globally_common_videos"] == 2
    assert audit["minimum_common_videos"] == 3


def test_uniform_logging_recovers_positive_oracle_value() -> None:
    utility = np.asarray(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ]
    )
    cohorts = construct_oracle_cohorts(
        utility,
        cohort_size=4,
        inverse_temperature=3.0,
    )
    truth = oracle_matching_value(cohorts)
    rows = run_recovery_grid(
        cohorts,
        sample_sizes=[50_000],
        repetitions=20,
        seed=7,
    )

    assert truth > 0.0
    centered = next(row for row in rows if row["method"] == "centered_dr")
    assert abs(centered["bias"]) < 0.01
    assert centered["positive_detection_rate"] == 1.0


def test_real_surface_support_stress_has_both_reversals() -> None:
    rng = np.random.default_rng(12)
    utility = rng.uniform(size=(20, 30))
    cohorts = construct_oracle_cohorts(
        utility,
        cohort_size=20,
        inverse_temperature=2.0,
    )
    common, marginal_break = run_support_stress_cases(
        cohorts[0],
        utility,
        sparse_top_k=4,
    )

    assert not common["target_value_point_identified"]
    assert not common["reference_value_point_identified"]
    assert common["contrast_point_identified"]
    assert common["direct_width"] == 0.0
    assert common["separate_value_width"] > 0.0
    assert marginal_break["target_value_point_identified"]
    assert not marginal_break["reference_value_point_identified"]
    assert not marginal_break["contrast_point_identified"]
    assert marginal_break["direct_width"] > 0.0
    assert marginal_break["oracle_inside_direct_interval"]


def test_popularity_logging_recovery_and_gradual_support_degradation() -> None:
    rng = np.random.default_rng(22)
    utility = rng.uniform(size=(20, 30))
    cohorts = construct_oracle_cohorts(
        utility,
        cohort_size=20,
        inverse_temperature=2.0,
    )
    recovery = run_recovery_grid(
        cohorts,
        sample_sizes=(30_000,),
        repetitions=20,
        seed=8,
        logging_inverse_temperatures=(3.0,),
    )
    centered = next(row for row in recovery if row["method"] == "centered_dr")
    degradation = run_popularity_support_degradation(
        cohorts[0],
        supported_action_fractions=(1.0, 0.75, 0.5),
        logging_inverse_temperature=3.0,
    )

    assert abs(centered["bias"]) < 0.01
    assert degradation[0]["contrast_point_identified"]
    assert degradation[0]["direct_width"] == 0.0
    assert not degradation[1]["contrast_point_identified"]
    assert degradation[1]["direct_width"] > 0.0
    assert degradation[2]["direct_width"] >= degradation[1]["direct_width"]
    assert all(row["oracle_inside_direct_interval"] for row in degradation)
