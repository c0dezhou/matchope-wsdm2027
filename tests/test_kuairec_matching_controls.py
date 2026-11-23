from __future__ import annotations

import numpy as np
import pytest

from matchope.pipeline.run_kuairec_matching_controls import (
    calibrate_common_exposure_policy,
    calibrate_common_exposure_policy_to_kl,
    evaluate_common_exposure_controls,
    evaluate_common_exposure_details,
    evaluate_common_exposure_kl_sensitivity,
    evaluate_fixed_common_marginal_strength_path,
    fixed_common_marginal_from_scores,
    fold_conditioned_marginal,
    mean_kl_to_fold_marginal,
    mean_tv_to_fold_marginal,
    minimum_kl_exposure_projection,
    paired_user_bootstrap_interval,
    paired_user_bootstrap_path,
    summarize_common_exposure_factor_sensitivity,
    summarize_common_exposure_knn_sensitivity,
    summarize_fixed_strength_rows,
)


def test_minimum_kl_projection_hits_each_fold_marginal() -> None:
    score = np.asarray(
        [
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
    )
    folds = np.asarray([0, 0, 1, 1])
    target = np.asarray([[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]])

    policy, error, iterations = minimum_kl_exposure_projection(
        score,
        evaluation_fold_ids=folds,
        target_marginal=target,
        inverse_temperature=2.0,
    )

    np.testing.assert_allclose(policy.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        fold_conditioned_marginal(policy, folds),
        target,
        atol=1e-9,
    )
    assert error <= 1e-10
    assert iterations > 0


def test_common_exposure_calibration_hits_requested_tv() -> None:
    score = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.5, 0.0],
            [0.0, 1.5],
        ]
    )
    folds = np.asarray([0, 0, 1, 1])
    target = np.asarray([[0.5, 0.5], [0.5, 0.5]])

    _, policy, achieved, marginal_error, _ = calibrate_common_exposure_policy(
        score,
        evaluation_fold_ids=folds,
        target_marginal=target,
        target_tv=0.2,
        initial_upper=1.0,
    )

    assert achieved == pytest.approx(0.2, abs=1e-7)
    assert mean_tv_to_fold_marginal(
        policy,
        evaluation_fold_ids=folds,
        target_marginal=target,
    ) == pytest.approx(0.2, abs=1e-7)
    np.testing.assert_allclose(
        fold_conditioned_marginal(policy, folds),
        target,
        atol=1e-9,
    )
    assert marginal_error <= 1e-10


def test_common_exposure_calibration_hits_requested_kl() -> None:
    score = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.5, 0.0],
            [0.0, 1.5],
        ]
    )
    folds = np.asarray([0, 0, 1, 1])
    target = np.asarray([[0.5, 0.5], [0.5, 0.5]])

    _, policy, achieved, marginal_error, _ = (
        calibrate_common_exposure_policy_to_kl(
            score,
            evaluation_fold_ids=folds,
            target_marginal=target,
            target_kl=0.1,
            initial_upper=1.0,
        )
    )

    assert achieved == pytest.approx(0.1, abs=1e-8)
    assert mean_kl_to_fold_marginal(
        policy,
        evaluation_fold_ids=folds,
        target_marginal=target,
    ) == pytest.approx(0.1, abs=1e-8)
    np.testing.assert_allclose(
        fold_conditioned_marginal(policy, folds),
        target,
        atol=1e-9,
    )
    assert marginal_error <= 1e-10


def test_kl_sensitivity_equalizes_information() -> None:
    factor_score = np.asarray(
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]] * 2
    )
    knn_score = np.asarray(
        [[0.0, 1.0, -1.0], [1.0, 0.0, -1.0]] * 2
    )
    utility = np.asarray(
        [[1.0, 0.0, 0.2], [0.0, 1.0, 0.2]] * 2
    )

    rows, contributions = evaluate_common_exposure_kl_sensitivity(
        factor_score,
        knn_score,
        utility,
        evaluation_fold_ids=np.asarray([0, 0, 1, 1]),
        target_tvs=(0.1,),
    )

    row = rows[0]
    assert row["kl_factor_actual_mean_kl_nats"] == pytest.approx(
        row["target_mean_kl_nats"], abs=1e-8
    )
    assert row["kl_knn_actual_mean_kl_nats"] == pytest.approx(
        row["target_mean_kl_nats"], abs=1e-8
    )
    assert row["kl_factor_marginal_error"] <= 1e-10
    assert row["kl_knn_marginal_error"] <= 1e-10
    assert contributions[0].mean() == pytest.approx(
        row["kl_delta_matching_knn_minus_factor"], abs=1e-12
    )


def test_common_exposure_comparison_has_zero_exposure_delta() -> None:
    factor_score = np.asarray(
        [
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
        ]
    )
    knn_score = np.asarray(
        [
            [0.0, 1.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [1.0, 0.0, -1.0],
        ]
    )
    utility = np.asarray(
        [
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.2],
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.2],
        ]
    )

    row = evaluate_common_exposure_controls(
        factor_score,
        knn_score,
        utility,
        evaluation_fold_ids=np.asarray([0, 0, 1, 1]),
        target_tvs=(0.1,),
    )[0]

    assert row["common_delta_exposure_knn_minus_factor"] == 0.0
    assert row["common_factor_actual_tv"] == pytest.approx(0.1, abs=1e-7)
    assert row["common_knn_actual_tv"] == pytest.approx(0.1, abs=1e-7)
    assert row["common_delta_matching_knn_minus_factor"] < 0.0
    assert row["common_positive_control_matching_value"] > 0.0
    assert row["common_positive_control_actual_tv"] == pytest.approx(
        0.1, abs=1e-7
    )
    assert row["common_positive_control_marginal_error"] <= 1e-10
    assert row["cross_fitted_shared_action_value"] == pytest.approx(0.5)
    assert row["cross_fitted_shared_unique_actions"] == 1
    assert row["common_factor_actionable_gain_vs_cross_fitted_shared"] > 0.0
    assert row["common_knn_actionable_gain_vs_cross_fitted_shared"] < 0.0
    assert row["common_factor_tv_p10"] <= row["common_factor_tv_p50"]
    assert row["common_factor_tv_p50"] <= row["common_factor_tv_p90"]
    assert row["common_knn_tv_p10"] <= row["common_knn_tv_p50"]
    assert row["common_knn_tv_p50"] <= row["common_knn_tv_p90"]

    detail_rows, contributions = evaluate_common_exposure_details(
        factor_score,
        knn_score,
        utility,
        evaluation_fold_ids=np.asarray([0, 0, 1, 1]),
        target_tvs=(0.1,),
    )
    assert len(contributions) == 1
    assert contributions[0].mean() == pytest.approx(
        detail_rows[0]["common_delta_matching_knn_minus_factor"],
        abs=1e-12,
    )


def test_fixed_common_marginal_strength_path_keeps_one_reference() -> None:
    factor_score = np.asarray(
        [[2.0, 0.0], [0.0, 2.0], [1.5, 0.0], [0.0, 1.5]]
    )
    knn_score = np.asarray(
        [[0.0, 1.5], [1.5, 0.0], [0.0, 2.0], [2.0, 0.0]]
    )
    utility = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    folds = np.asarray([0, 0, 1, 1])
    marginal, factor_beta, knn_beta = fixed_common_marginal_from_scores(
        factor_score,
        knn_score,
        evaluation_fold_ids=folds,
        anchor_tv=0.1,
    )

    rows, contributions = evaluate_fixed_common_marginal_strength_path(
        factor_score,
        knn_score,
        utility,
        evaluation_fold_ids=folds,
        target_marginal=marginal,
        target_tvs=(0.05, 0.1, 0.2),
        initial_factor_upper=factor_beta,
        initial_knn_upper=knn_beta,
    )

    assert [row["target_tv"] for row in rows] == [0.05, 0.1, 0.2]
    assert len(contributions) == 3
    assert len({row["reference_value"] for row in rows}) == 1
    for row, contribution in zip(rows, contributions):
        assert contribution.mean() == pytest.approx(
            row["delta_matching_knn_minus_factor"],
            abs=1e-12,
        )


def test_fixed_strength_summary_keeps_outcomes_separate() -> None:
    factor_score = np.asarray(
        [[2.0, 0.0], [0.0, 2.0], [1.5, 0.0], [0.0, 1.5]]
    )
    knn_score = np.asarray(
        [[0.0, 1.5], [1.5, 0.0], [0.0, 2.0], [2.0, 0.0]]
    )
    utility = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    folds = np.asarray([0, 0, 1, 1])
    marginal, factor_beta, knn_beta = fixed_common_marginal_from_scores(
        factor_score,
        knn_score,
        evaluation_fold_ids=folds,
        anchor_tv=0.1,
    )
    rows, _ = evaluate_fixed_common_marginal_strength_path(
        factor_score,
        knn_score,
        utility,
        evaluation_fold_ids=folds,
        target_marginal=marginal,
        target_tvs=(0.1,),
        initial_factor_upper=factor_beta,
        initial_knn_upper=knn_beta,
    )
    mixed = [
        {"outcome": outcome, **rows[0]}
        for outcome in ("capped_watch_ratio", "complete_watch")
    ]

    summaries = summarize_fixed_strength_rows(mixed)

    assert [summary["outcome"] for summary in summaries] == [
        "capped_watch_ratio",
        "complete_watch",
    ]
    assert all(summary["splits"] == 1 for summary in summaries)


def test_factor_sensitivity_summary_keeps_each_configuration_separate() -> None:
    rows = []
    for rank, ridge, deltas in (
        (8, 1.0, (-0.2, -0.1)),
        (16, 10.0, (-0.3, 0.1)),
    ):
        for split_seed, delta in zip((11, 23), deltas):
            rows.append(
                {
                    "outcome": "complete_watch",
                    "split_seed": split_seed,
                    "factor_rank": rank,
                    "factor_ridge_penalty": ridge,
                    "is_primary_factor_configuration": rank == 16,
                    "target_tv": 0.04,
                    "delta_matching_knn_minus_factor": delta,
                    "factor_matching_value": 0.3 - delta,
                    "knn_matching_value": 0.3,
                    "factor_marginal_error": 1e-12,
                    "knn_marginal_error": 2e-12,
                    "factor_actual_tv": 0.04,
                    "knn_actual_tv": 0.04,
                }
            )

    summaries = summarize_common_exposure_factor_sensitivity(rows)

    configurations = [
        (row["factor_rank"], row["factor_ridge_penalty"])
        for row in summaries
    ]
    assert configurations == [
        (8, 1.0),
        (16, 10.0),
    ]
    assert summaries[0]["negative_splits"] == 2
    assert summaries[1]["negative_splits"] == 1


def test_knn_sensitivity_summary_keeps_each_neighborhood_separate() -> None:
    rows = []
    for neighbor_fraction, deltas in (
        (0.05, (-0.2, -0.1)),
        (0.2, (-0.3, 0.1)),
    ):
        for split_seed, delta in zip((11, 23), deltas):
            rows.append(
                {
                    "outcome": "complete_watch",
                    "split_seed": split_seed,
                    "neighbor_fraction": neighbor_fraction,
                    "is_inner_selected_configuration": neighbor_fraction == 0.2,
                    "outer_neighbor_count_min": int(100 * neighbor_fraction),
                    "outer_neighbor_count_max": int(101 * neighbor_fraction),
                    "factor_rank": 16,
                    "factor_ridge_penalty": 10.0,
                    "target_tv": 0.04,
                    "delta_matching_knn_minus_factor": delta,
                    "factor_matching_value": 0.3 - delta,
                    "knn_matching_value": 0.3,
                    "factor_marginal_error": 1e-12,
                    "knn_marginal_error": 2e-12,
                    "factor_actual_tv": 0.04,
                    "knn_actual_tv": 0.04,
                }
            )

    summaries = summarize_common_exposure_knn_sensitivity(rows)

    assert [row["neighbor_fraction"] for row in summaries] == [0.05, 0.2]
    assert summaries[0]["negative_splits"] == 2
    assert summaries[1]["negative_splits"] == 1
    assert not summaries[0]["is_inner_selected_configuration"]
    assert summaries[1]["is_inner_selected_configuration"]


def test_common_marginal_weight_changes_the_reference_path() -> None:
    factor_score = np.asarray(
        [[3.0, 0.0, -1.0], [0.0, 1.0, -1.0]] * 2
    )
    knn_score = np.asarray(
        [[0.0, 3.0, -1.0], [1.0, 0.0, -1.0]] * 2
    )
    utility = np.asarray(
        [[1.0, 0.0, 0.2], [0.5, 0.0, 0.2]] * 2
    )
    folds = np.asarray([0, 0, 1, 1])

    factor_endpoint = evaluate_common_exposure_controls(
        factor_score,
        knn_score,
        utility,
        evaluation_fold_ids=folds,
        target_tvs=(0.1,),
        include_positive_control=False,
        common_marginal_weight=0.0,
    )[0]
    knn_endpoint = evaluate_common_exposure_controls(
        factor_score,
        knn_score,
        utility,
        evaluation_fold_ids=folds,
        target_tvs=(0.1,),
        include_positive_control=False,
        common_marginal_weight=1.0,
    )[0]

    assert factor_endpoint["common_marginal_weight"] == 0.0
    assert knn_endpoint["common_marginal_weight"] == 1.0
    assert factor_endpoint["common_reference_value"] != pytest.approx(
        knn_endpoint["common_reference_value"]
    )


def test_common_exposure_projection_removes_shared_action_score() -> None:
    score = np.asarray(
        [
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
            [1.5, 0.0, -0.5],
            [0.0, 1.5, -0.5],
        ]
    )
    shifted = score + np.asarray([0.3, -0.2, 0.05])[None, :]
    outcome = np.asarray(
        [
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.2],
            [0.8, 0.1, 0.3],
            [0.1, 0.8, 0.3],
        ]
    )

    row = evaluate_common_exposure_controls(
        score,
        shifted,
        outcome,
        evaluation_fold_ids=np.asarray([0, 0, 1, 1]),
        target_tvs=(0.1,),
    )[0]

    assert row["own_reference_marginal_tv"] > 0.0
    assert abs(row["common_delta_matching_knn_minus_factor"]) < 2e-7
    assert row["common_policy_pair_tv"] < 2e-7
    assert row["common_policy_pair_maximum_tv"] < 2e-7


def test_paired_user_bootstrap_averages_splits_before_resampling() -> None:
    contributions = np.asarray(
        [
            [-0.4, -0.2, 0.0, 0.2],
            [-0.2, -0.2, -0.2, -0.2],
        ]
    )

    point, low, high = paired_user_bootstrap_interval(
        contributions,
        replicates=2_000,
        rng=np.random.default_rng(7),
    )

    assert point == pytest.approx(-0.15)
    assert low < point < high


def test_paired_user_bootstrap_path_returns_a_simultaneous_band() -> None:
    contributions = np.asarray(
        [
            [[-0.4, -0.2, 0.0, 0.2], [-0.2, -0.2, -0.2, -0.2]],
            [[-0.3, -0.1, 0.1, 0.3], [-0.1, -0.1, -0.1, -0.1]],
        ]
    )

    summary = paired_user_bootstrap_path(
        contributions,
        replicates=2_000,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_allclose(summary["point"], [-0.15, -0.05])
    assert np.all(
        summary["simultaneous_low"] <= summary["pointwise_low"] + 1e-12
    )
    assert np.all(
        summary["simultaneous_high"] >= summary["pointwise_high"] - 1e-12
    )
