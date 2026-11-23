from __future__ import annotations

import numpy as np
import pytest

from matchope.pipeline.run_kuairec_matching_learned import (
    CrossFittedLearnedPolicySplit,
    _centered_user_knn_predictions,
    _evaluate_frozen_score,
    construct_cross_fitted_knn_policy_split,
    construct_cross_fitted_learned_policy_split,
    evaluate_personalization_tv_matched_policies,
    evaluate_policy_family,
    run_factor_hyperparameter_sensitivity,
    summarize_cross_family_reversals,
    summarize_ranking_reversals,
)


def test_policy_training_does_not_access_evaluation_candidate_outcomes() -> None:
    rng = np.random.default_rng(7)
    utility = rng.uniform(size=(10, 12))
    original = construct_cross_fitted_learned_policy_split(
        utility,
        user_folds=5,
        history_action_count=7,
        rank=2,
        ridge_penalty=1.0,
        split_seed=19,
    )
    evaluation_row = 0
    evaluation_user = original.evaluation_user_indices[evaluation_row]
    perturbed_utility = utility.copy()
    perturbed_utility[
        np.ix_(
            np.asarray([evaluation_user]),
            original.candidate_action_indices,
        )
    ] = 1.0 - original.candidate_utility[evaluation_row]
    perturbed = construct_cross_fitted_learned_policy_split(
        perturbed_utility,
        user_folds=5,
        history_action_count=7,
        rank=2,
        ridge_penalty=1.0,
        split_seed=19,
    )

    np.testing.assert_allclose(
        original.candidate_item_quality[evaluation_row],
        perturbed.candidate_item_quality[evaluation_row],
    )
    np.testing.assert_allclose(
        original.candidate_interaction_score[evaluation_row],
        perturbed.candidate_interaction_score[evaluation_row],
    )
    assert not np.allclose(
        original.candidate_utility[evaluation_row],
        perturbed.candidate_utility[evaluation_row],
    )


def test_cross_fitting_evaluates_every_user_once() -> None:
    utility = np.random.default_rng(13).uniform(size=(11, 14))

    split = construct_cross_fitted_learned_policy_split(
        utility,
        user_folds=5,
        history_action_count=6,
        rank=2,
        ridge_penalty=1.0,
        split_seed=29,
    )

    np.testing.assert_array_equal(
        np.sort(split.evaluation_user_indices),
        np.arange(utility.shape[0]),
    )
    assert split.candidate_utility.shape == (11, 8)
    assert split.candidate_item_quality.shape == (11, 8)
    assert split.minimum_training_users == 8
    assert split.maximum_training_users == 9


def test_overall_and_matching_value_can_reverse_policy_ranking() -> None:
    item_quality = np.asarray([0.1, 0.1, 0.9])
    interaction = np.asarray(
        [
            [0.6, 0.0, 0.0],
            [0.0, 0.6, 0.0],
            [0.6, 0.0, 0.0],
            [0.0, 0.6, 0.0],
        ]
    )
    split = CrossFittedLearnedPolicySplit(
        split_seed=3,
        user_folds=2,
        minimum_training_users=2,
        maximum_training_users=2,
        evaluation_user_indices=np.arange(4),
        evaluation_fold_ids=np.repeat(np.arange(2), 2),
        history_action_indices=np.arange(2),
        candidate_action_indices=np.arange(3),
        candidate_action_start=2,
        candidate_item_quality=np.broadcast_to(item_quality, (4, 3)),
        candidate_interaction_score=interaction,
        candidate_utility=np.clip(item_quality + interaction, 0.0, 1.0),
        history_rmse=0.1,
        rank=2,
        ridge_penalty=1.0,
    )
    rows = evaluate_policy_family(
        split,
        outcome="test_utility",
        quality_weights=(0.0, 8.0),
        inverse_temperatures=(4.0,),
    )
    summary = summarize_ranking_reversals(rows)[0]

    assert summary["outcome"] == "test_utility"
    assert rows[1]["overall_policy_value"] > rows[0]["overall_policy_value"]
    assert rows[1]["matching_value"] < rows[0]["matching_value"]
    assert summary["pairwise_ranking_disagreement"] == 1.0
    assert np.isclose(summary["spearman_overall_vs_matching"], -1.0)


def test_knn_outer_fold_does_not_access_its_candidate_outcomes() -> None:
    utility = np.random.default_rng(31).uniform(size=(18, 16))
    original = construct_cross_fitted_knn_policy_split(
        utility,
        user_folds=3,
        history_action_count=8,
        neighbor_fraction_grid=(0.25, 0.5, 1.0),
        inner_validation_fraction=0.25,
        split_seed=43,
    )
    fold_id = 0
    fold_rows = original.evaluation_fold_ids == fold_id
    evaluation_users = original.evaluation_user_indices[fold_rows]
    perturbed_utility = utility.copy()
    perturbed_utility[np.ix_(evaluation_users, original.candidate_action_indices)] = (
        1.0
        - original.candidate_utility[fold_rows]
    )
    perturbed = construct_cross_fitted_knn_policy_split(
        perturbed_utility,
        user_folds=3,
        history_action_count=8,
        neighbor_fraction_grid=(0.25, 0.5, 1.0),
        inner_validation_fraction=0.25,
        split_seed=43,
    )

    np.testing.assert_allclose(
        original.candidate_score[fold_rows],
        perturbed.candidate_score[fold_rows],
    )
    assert set(original.candidate_scores_by_neighbor_fraction) == {
        0.25,
        0.5,
        1.0,
    }
    for neighbor_fraction, candidate_score in (
        original.candidate_scores_by_neighbor_fraction.items()
    ):
        np.testing.assert_allclose(
            candidate_score[fold_rows],
            perturbed.candidate_scores_by_neighbor_fraction[neighbor_fraction][
                fold_rows
            ],
        )
    np.testing.assert_allclose(
        original.candidate_score[fold_rows],
        original.candidate_scores_by_neighbor_fraction[
            original.selected_neighbor_fractions[fold_id]
        ][fold_rows],
    )
    assert (
        original.selected_neighbor_counts[fold_id]
        == perturbed.selected_neighbor_counts[fold_id]
    )
    assert not np.allclose(
        original.candidate_utility[fold_rows],
        perturbed.candidate_utility[fold_rows],
    )


def test_knn_prediction_for_one_k_is_invariant_to_the_requested_grid() -> None:
    utility = np.random.default_rng(37).uniform(size=(12, 10))
    history_actions = np.arange(5)
    candidate_actions = np.arange(5, 10)
    query_history = utility[:3, :5]
    training = utility[3:]

    single = _centered_user_knn_predictions(
        training,
        query_history,
        history_action_indices=history_actions,
        candidate_action_indices=candidate_actions,
        neighbor_counts=(3,),
    )[3]
    grid = _centered_user_knn_predictions(
        training,
        query_history,
        history_action_indices=history_actions,
        candidate_action_indices=candidate_actions,
        neighbor_counts=(1, 3, 9),
    )[3]

    np.testing.assert_array_equal(single, grid)


def test_history_sensitivity_can_hold_candidate_actions_fixed() -> None:
    utility = np.random.default_rng(41).uniform(size=(16, 20))
    short_history = construct_cross_fitted_learned_policy_split(
        utility,
        user_folds=4,
        history_action_count=4,
        candidate_action_start=8,
        rank=2,
        ridge_penalty=1.0,
        split_seed=47,
    )
    long_history = construct_cross_fitted_learned_policy_split(
        utility,
        user_folds=4,
        history_action_count=8,
        candidate_action_start=8,
        rank=2,
        ridge_penalty=1.0,
        split_seed=47,
    )

    np.testing.assert_array_equal(
        short_history.candidate_action_indices,
        long_history.candidate_action_indices,
    )
    np.testing.assert_array_equal(
        short_history.history_action_indices,
        long_history.history_action_indices[:4],
    )


def test_knn_inner_validation_selects_one_rmse_winner_per_outer_fold() -> None:
    split = construct_cross_fitted_knn_policy_split(
        np.random.default_rng(47).uniform(size=(20, 18)),
        user_folds=4,
        history_action_count=9,
        neighbor_fraction_grid=(0.2, 0.5, 1.0),
        inner_validation_fraction=0.25,
        split_seed=53,
    )

    for fold_id in range(split.user_folds):
        records = [
            record
            for record in split.selection_records
            if record.outer_fold == fold_id
        ]
        selected = [record for record in records if record.selected]
        assert len(records) == 3
        assert len(selected) == 1
        assert selected[0].neighbor_fraction == min(
            records,
            key=lambda record: (record.validation_rmse, record.neighbor_fraction),
        ).neighbor_fraction


def test_fold_conditioned_reference_does_not_call_model_drift_matching() -> None:
    score = np.asarray(
        [
            [10.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [0.0, 10.0],
        ]
    )
    utility = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    metrics = _evaluate_frozen_score(
        score,
        utility,
        evaluation_fold_ids=np.asarray([0, 0, 1, 1]),
        inverse_temperature=1.0,
    )

    assert abs(metrics["matching_value"]) < 1e-14
    assert metrics["cross_fitted_shared_action_value"] == pytest.approx(0.0)
    assert metrics["cross_fitted_shared_unique_actions"] == 2
    assert metrics["actionable_gain_vs_cross_fitted_shared"] > 0.0


def test_cross_family_summary_is_an_exact_exposure_matching_decomposition() -> None:
    common = {
        "outcome": "test_utility",
        "split_seed": 7,
        "inverse_temperature": 4.0,
    }
    rows = [
        {
            **common,
            "policy_family": "interaction_ridge_mf",
            "overall_policy_value": 0.70,
            "reference_policy_value": 0.66,
            "matching_value": 0.04,
        },
        {
            **common,
            "policy_family": "standard_ridge_mf",
            "overall_policy_value": 0.73,
            "reference_policy_value": 0.70,
            "matching_value": 0.03,
        },
        {
            **common,
            "policy_family": "centered_user_knn",
            "overall_policy_value": 0.72,
            "reference_policy_value": 0.70,
            "matching_value": 0.02,
        },
    ]

    summary = summarize_cross_family_reversals(rows)[0]

    assert np.isclose(
        summary["delta_overall_knn_minus_interaction_mf"],
        0.02,
    )
    assert np.isclose(
        summary["delta_reference_knn_minus_interaction_mf"],
        0.04,
    )
    assert np.isclose(
        summary["delta_matching_knn_minus_interaction_mf"],
        -0.02,
    )
    assert abs(float(summary["interaction_decomposition_error"])) < 1e-14
    assert summary["interaction_pair_overall_preferred"] == "centered_user_knn"
    assert (
        summary["interaction_pair_matching_preferred"]
        == "interaction_ridge_mf"
    )
    assert summary["interaction_pair_ranking_reversal"] is True
    assert summary["standard_pair_ranking_reversal"] is False
    assert summary["bank_overall_best_family"] == "standard_ridge_mf"
    assert summary["bank_matching_best_family"] == "interaction_ridge_mf"
    assert summary["bank_ranking_reversal"] is True


def test_factor_sensitivity_reports_full_grid_without_selecting_a_winner() -> None:
    utility = np.random.default_rng(59).uniform(size=(12, 16))
    knn_policy_rows = [
        {
            "outcome": "test_utility",
            "split_seed": 61,
            "policy_family": "centered_user_knn",
            "user_folds": 3,
            "evaluation_users": 12,
            "history_actions": 6,
            "candidate_actions": 8,
            "candidate_action_start": 8,
            "reference_stratum": "outer_fold",
            "inverse_temperature": 2.0,
            "selected_neighbor_fraction_min": 0.5,
            "selected_neighbor_fraction_max": 0.5,
            "selected_neighbor_fraction_mean": 0.5,
            "overall_policy_value": 0.60,
            "reference_policy_value": 0.55,
            "matching_value": 0.05,
        }
    ]

    rows = run_factor_hyperparameter_sensitivity(
        utility,
        outcome="test_utility",
        split_seeds=(61,),
        user_folds=3,
        history_action_count=6,
        candidate_action_start=8,
        rank_grid=(2, 3),
        ridge_penalty_grid=(1.0, 2.0),
        primary_rank=2,
        primary_ridge_penalty=1.0,
        inverse_temperatures=(2.0,),
        knn_policy_rows=knn_policy_rows,
    )

    assert len(rows) == 4
    assert sum(bool(row["is_primary_factor_configuration"]) for row in rows) == 1
    assert {row["factor_grid_role"] for row in rows} == {
        "full_sensitivity_no_selection"
    }
    assert max(abs(float(row["decomposition_error"])) for row in rows) < 1e-14


def test_policy_families_can_be_compared_at_equal_personalization_tv() -> None:
    utility = np.random.default_rng(47).uniform(size=(20, 18))
    factor_split = construct_cross_fitted_learned_policy_split(
        utility,
        user_folds=4,
        history_action_count=9,
        rank=2,
        ridge_penalty=1.0,
        split_seed=53,
    )
    knn_split = construct_cross_fitted_knn_policy_split(
        utility,
        user_folds=4,
        history_action_count=9,
        neighbor_fraction_grid=(0.2, 0.5, 1.0),
        inner_validation_fraction=0.25,
        split_seed=53,
    )

    rows = evaluate_personalization_tv_matched_policies(
        factor_split,
        knn_split,
        outcome="test_utility",
        target_tvs=(0.01, 0.02),
    )

    assert len(rows) == 2
    for row in rows:
        target = float(row["target_policy_to_reference_tv"])
        for field in (
            "interaction_actual_tv",
            "standard_actual_tv",
            "knn_actual_tv",
        ):
            assert abs(float(row[field]) - target) <= 1e-7
        assert abs(float(row["interaction_decomposition_error"])) < 1e-14
        assert abs(float(row["standard_decomposition_error"])) < 1e-14
        assert row["calibration_signal"] == (
            "policy_scores_only_no_candidate_outcomes"
        )
