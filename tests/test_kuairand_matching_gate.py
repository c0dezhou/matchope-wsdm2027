from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from matchope.pipeline.run_kuairand_matching_gate import (
    _load_candidate_categories,
    assign_user_folds,
    audit_contrast_overlap,
    build_first_level_category_mapping,
    calibrate_policy_to_tv,
    cluster_inference,
    enumerated_cluster_sign_flip_interval,
    cross_fitted_scores,
    resolution_status,
    tilted_category_policy,
    two_way_cluster_inference,
)


def test_candidate_category_loader_stops_after_covering_pool(tmp_path) -> None:
    path = tmp_path / "categories.csv"
    pd.DataFrame(
        {
            "final_video_id": [0, 1, 2, 99],
            "first_level_category_id": [1, 2, 2, 7],
            "first_level_category_name": ["a", "b", "b", "unused"],
        }
    ).to_csv(path, index=False)

    selected = _load_candidate_categories(path, np.asarray([0, 1, 2]))

    assert selected["final_video_id"].tolist() == [0, 1, 2]


def test_candidate_category_loader_rejects_missing_pool_item(tmp_path) -> None:
    path = tmp_path / "categories.csv"
    pd.DataFrame(
        {
            "final_video_id": [0],
            "first_level_category_id": [1],
            "first_level_category_name": ["a"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="misses 1 candidate videos"):
        _load_candidate_categories(path, np.asarray([0, 1]))


def test_first_level_mapping_is_metadata_only_and_merges_rare_categories() -> None:
    metadata = pd.DataFrame(
        {
            "final_video_id": np.arange(7),
            "first_level_category_id": [1, 1, 1, 2, 2, np.nan, 3],
            "first_level_category_name": [
                "music",
                "music",
                "music",
                "games",
                "games",
                np.nan,
                "news",
            ],
        }
    )

    mapping = build_first_level_category_mapping(
        metadata,
        minimum_items_per_category=2,
    )

    assert mapping.raw_category_count == 4
    assert mapping.merged_rare_category_count == 2
    assert mapping.labels == ("1:music", "2:games", "__OTHER_RARE__")
    assert mapping.item_counts.tolist() == [3, 2, 2]
    assert mapping.base_probability.sum() == pytest.approx(1.0)
    assert mapping.video_to_action[0] == mapping.video_to_action[1]
    assert mapping.video_to_action[5] == mapping.video_to_action[6]


def test_contrast_overlap_audit_uses_signed_unsupported_difference() -> None:
    policy = np.asarray(
        [
            [0.8, 0.2],
            [0.2, 0.8],
        ]
    )
    users = np.asarray([0, 1, 0, 1])
    contexts = np.asarray([0, 0, 1, 1])
    support = np.asarray([[True, False], [True, True]])

    audit = audit_contrast_overlap(
        policy,
        event_user_index=users,
        event_context_index=contexts,
        structural_support=support,
    )

    assert audit["unsupported_context_action_cells"] == 1
    assert audit["event_weighted_target_unsupported_mass"] == pytest.approx(0.25)
    assert audit["event_weighted_reference_unsupported_mass"] == pytest.approx(0.25)
    assert audit["maximum_absolute_unsupported_contrast"] == pytest.approx(0.3)
    assert audit["contrast_overlap_holds"] is False


def test_contrast_overlap_audit_accepts_design_known_full_support() -> None:
    audit = audit_contrast_overlap(
        np.asarray([[0.8, 0.2], [0.2, 0.8]]),
        event_user_index=np.asarray([0, 1]),
        event_context_index=np.asarray([0, 0]),
        structural_support=np.asarray([[True, True]]),
    )

    assert audit["unsupported_context_action_cells"] == 0
    assert audit["event_weighted_absolute_unsupported_contrast"] == 0.0
    assert audit["contrast_overlap_holds"] is True


def test_contrast_overlap_audit_accepts_exact_cancellation_in_a_hole() -> None:
    audit = audit_contrast_overlap(
        np.asarray([[0.7, 0.1, 0.2], [0.1, 0.7, 0.2]]),
        event_user_index=np.asarray([0, 1]),
        event_context_index=np.asarray([0, 0]),
        structural_support=np.asarray([[True, True, False]]),
    )

    assert audit["unsupported_context_action_cells"] == 1
    assert audit["event_weighted_target_unsupported_mass"] == pytest.approx(0.2)
    assert audit["event_weighted_reference_unsupported_mass"] == pytest.approx(0.2)
    assert audit["event_weighted_absolute_unsupported_contrast"] == 0.0
    assert audit["contrast_overlap_holds"] is True


def test_tv_calibration_does_not_use_any_reward_input() -> None:
    signal = np.asarray(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.3, 0.7],
        ]
    )
    global_rate = np.asarray([0.5, 0.5])
    base = np.asarray([0.6, 0.4])

    beta, policy, achieved = calibrate_policy_to_tv(
        signal,
        global_rate,
        base,
        target_tv=0.1,
    )

    assert beta > 0.0
    assert achieved == pytest.approx(0.1)
    np.testing.assert_allclose(policy.sum(axis=1), 1.0)
    assert policy[0, 0] > policy[2, 0]


def test_shared_signal_produces_a_shared_policy() -> None:
    signal = np.tile([0.4, 0.6], (5, 1))
    policy = tilted_category_policy(
        signal,
        np.asarray([0.4, 0.6]),
        np.asarray([0.3, 0.7]),
        inverse_temperature=100.0,
    )

    np.testing.assert_allclose(policy, np.tile([0.3, 0.7], (5, 1)))


def test_user_folds_are_deterministic_balanced_and_user_level() -> None:
    first = assign_user_folds(103, folds=5, seed=17)
    second = assign_user_folds(103, folds=5, seed=17)

    np.testing.assert_array_equal(first, second)
    counts = np.bincount(first, minlength=5)
    assert counts.max() - counts.min() <= 1


def test_cross_fitted_category_score_recovers_randomized_matching_value() -> None:
    rng = np.random.default_rng(11)
    n_users = 400
    n_dates = 2
    rows_per_user_date = 30
    user = np.repeat(np.arange(n_users), n_dates * rows_per_user_date)
    date_index = np.tile(
        np.repeat(np.arange(n_dates), rows_per_user_date), n_users
    )
    action = rng.integers(0, 2, size=user.size)
    user_type = user % 2
    q = np.where(action == user_type, 0.8, 0.2)
    reward = rng.binomial(1, q)
    policy = np.where(
        np.arange(2)[None, :] == (np.arange(n_users) % 2)[:, None],
        0.8,
        0.2,
    )
    history_signal = np.where(
        np.arange(2)[None, :] == (np.arange(n_users) % 2)[:, None],
        0.8,
        0.2,
    )
    evaluation = pd.DataFrame(
        {
            "user_index": user,
            "user_id": user,
            "action": action,
            "context_index": date_index,
            "date": 20220422 + date_index,
            "long_view": reward,
        }
    )

    scores = cross_fitted_scores(
        evaluation,
        policy_by_user=policy,
        history_signal=history_signal,
        base_probability=np.asarray([0.5, 0.5]),
        user_fold=assign_user_folds(n_users, folds=5, seed=7),
        propensity_smoothing=1.0,
        outcome_smoothing=10.0,
        outcome_ridge=1.0,
        outcome="long_view",
    )

    assert scores.centered_scores.mean() == pytest.approx(0.18, abs=0.015)
    assert scores.ipw_scores.mean() == pytest.approx(0.18, abs=0.02)
    assert scores.minimum_logging_pscore > 0.45
    assert scores.maximum_logging_pscore < 0.55


def test_cross_fitted_score_accepts_fixed_logging_probability() -> None:
    rng = np.random.default_rng(21)
    n_users = 100
    rows_per_context = 20
    user = np.repeat(np.arange(n_users), 2 * rows_per_context)
    context = np.tile(np.repeat(np.arange(2), rows_per_context), n_users)
    action = rng.choice(2, size=user.size, p=[0.7, 0.3])
    reward = rng.binomial(1, np.where(action == user % 2, 0.7, 0.3))
    evaluation = pd.DataFrame(
        {
            "user_index": user,
            "user_id": user,
            "action": action,
            "context_index": context,
            "date": 20220422 + context,
            "long_view": reward,
        }
    )
    policy = np.where(
        np.arange(2)[None, :] == (np.arange(n_users) % 2)[:, None],
        0.7,
        0.3,
    )

    scores = cross_fitted_scores(
        evaluation,
        policy_by_user=policy,
        history_signal=policy,
        base_probability=np.asarray([0.7, 0.3]),
        user_fold=assign_user_folds(n_users, folds=5, seed=9),
        propensity_smoothing=1.0,
        fixed_logging_probability=np.asarray([0.7, 0.3]),
        outcome_smoothing=10.0,
        outcome_ridge=1.0,
        outcome="long_view",
    )

    assert scores.minimum_logging_pscore == pytest.approx(0.3)
    assert scores.maximum_logging_pscore == pytest.approx(0.7)


def test_cross_fitted_shared_policy_has_exactly_zero_matching_scores() -> None:
    evaluation = pd.DataFrame(
        {
            "user_index": np.repeat(np.arange(10), 4),
            "user_id": np.repeat(np.arange(10), 4),
            "action": np.tile([0, 1, 0, 1], 10),
            "context_index": np.tile([0, 0, 1, 1], 10),
            "date": np.tile([20220422, 20220422, 20220423, 20220423], 10),
            "long_view": np.tile([0, 1, 1, 0], 10),
        }
    )
    policy = np.tile([0.4, 0.6], (10, 1))

    scores = cross_fitted_scores(
        evaluation,
        policy_by_user=policy,
        history_signal=np.tile([0.3, 0.7], (10, 1)),
        base_probability=np.asarray([0.5, 0.5]),
        user_fold=assign_user_folds(10, folds=2, seed=3),
        propensity_smoothing=1.0,
        outcome_smoothing=2.0,
        outcome_ridge=1.0,
        outcome="long_view",
    )

    np.testing.assert_allclose(scores.centered_scores, 0.0, atol=1e-15)
    np.testing.assert_allclose(scores.separate_dr_scores, 0.0, atol=1e-15)
    np.testing.assert_allclose(scores.ipw_scores, 0.0, atol=1e-15)


def test_cluster_inference_uses_cluster_sums_not_cluster_means() -> None:
    scores = np.asarray([1.0, 3.0, 2.0, -2.0, 0.0])
    clusters = np.asarray([0, 0, 1, 2, 2])

    result = cluster_inference(scores, clusters, prefix="user")

    estimate = scores.mean()
    centered_sums = np.asarray(
        [
            (scores[:2] - estimate).sum(),
            scores[2] - estimate,
            (scores[3:] - estimate).sum(),
        ]
    )
    expected = np.sqrt(3.0 / 2.0 * np.square(centered_sums).sum()) / len(scores)
    assert result["user_standard_error"] == pytest.approx(expected)


def test_enumerated_cluster_sign_flip_interval_uses_all_patterns() -> None:
    scores = np.asarray([0.0, 2.0, 1.0, 3.0, -1.0, 1.0])
    days = np.asarray([0, 0, 1, 1, 2, 2])

    result = enumerated_cluster_sign_flip_interval(scores, days, prefix="day")

    assert result["day_sign_flip_draws"] == 8
    assert result["day_sign_flip_ci_lower"] < scores.mean()
    assert result["day_sign_flip_ci_upper"] > scores.mean()
    assert result["day_sign_flip_assumption"] == "independent_symmetric_clusters"


def test_enumerated_cluster_sign_flip_interval_rejects_too_many_clusters() -> None:
    scores = np.arange(21, dtype=np.float64)

    with pytest.raises(ValueError, match="between 2 and 20"):
        enumerated_cluster_sign_flip_interval(scores, np.arange(21), prefix="day")


def test_two_way_cluster_inference_adds_axes_and_subtracts_intersections() -> None:
    scores = np.asarray([0.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    users = np.asarray([0, 0, 1, 1, 2, 2])
    days = np.asarray([0, 1, 0, 1, 0, 1])

    result = two_way_cluster_inference(
        scores,
        users,
        days,
        prefix="both",
    )

    centered = scores - scores.mean()

    def variance(labels: np.ndarray) -> float:
        _, codes = np.unique(labels, return_inverse=True)
        groups = int(codes.max()) + 1
        sums = np.bincount(codes, weights=centered)
        return groups / (groups - 1) * np.square(sums).sum() / scores.size**2

    intersection = np.asarray([f"{u}:{d}" for u, d in zip(users, days)])
    expected = np.sqrt(variance(users) + variance(days) - variance(intersection))
    assert result["both_standard_error"] == pytest.approx(expected)
    assert result["both_degrees_of_freedom"] == 1


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (0.002, 0.003, "positive_practically_relevant"),
        (-0.003, -0.002, "negative_practically_relevant"),
        (-0.0005, 0.0007, "practically_equivalent"),
        (0.0001, 0.0008, "positive_but_submargin"),
        (-0.002, 0.002, "inconclusive"),
    ],
)
def test_resolution_status_separates_sign_and_equivalence(
    lower: float,
    upper: float,
    expected: str,
) -> None:
    assert (
        resolution_status(lower, upper, equivalence_margin=0.001) == expected
    )
