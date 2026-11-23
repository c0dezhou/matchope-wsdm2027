from __future__ import annotations

import numpy as np
import pytest

from matchope.eval.matching_value import (
    centered_matching_dr_scores,
    contrast_identification_interval,
    contrast_support_certificate,
    equality_restricted_matching_dr_scores,
    equality_restricted_projection_coefficients,
    estimate_from_scores,
    marginalize_policy,
    matching_value,
    normal_minimum_detectable_effect,
    separate_value_difference_interval,
    support_adapted_matching_dr_scores,
    two_sided_normal_power,
)


def test_additive_outcome_has_zero_matching_value() -> None:
    policy = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.2, 0.2],
        ]
    )
    user_effect = np.asarray([0.2, -0.1, 0.4, 0.0])[:, None]
    item_effect = np.asarray([0.3, 0.7, -0.2])[None, :]

    value = matching_value(
        policy,
        user_effect + item_effect,
        np.zeros(4, dtype=int),
    )

    assert value == pytest.approx(0.0, abs=1e-15)


def test_policy_marginal_preserves_action_mass_within_strata() -> None:
    policy = np.asarray(
        [[0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.7, 0.3]]
    )
    strata = np.asarray([0, 0, 1, 1])

    reference = marginalize_policy(policy, strata)

    np.testing.assert_allclose(reference[:2], [[0.6, 0.4], [0.6, 0.4]])
    np.testing.assert_allclose(reference[2:], [[0.4, 0.6], [0.4, 0.6]])
    for stratum in (0, 1):
        rows = strata == stratum
        np.testing.assert_allclose(policy[rows].mean(axis=0), reference[rows][0])


def test_policy_marginal_rejects_varying_candidate_sets_inside_stratum() -> None:
    policy = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    eligibility = np.asarray([[True, False], [False, True]])

    with pytest.raises(ValueError, match="eligibility varies"):
        marginalize_policy(
            policy,
            np.zeros(2, dtype=int),
            eligibility=eligibility,
        )


def test_contrast_can_be_identified_when_neither_value_is() -> None:
    policy = np.asarray([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]])
    reference = np.asarray([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    logging = np.asarray([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])
    observed_q = np.asarray([[np.nan, 0.7, 0.1], [np.nan, 0.2, 0.8]])

    certificate = contrast_support_certificate(policy, reference, logging)
    direct = contrast_identification_interval(
        policy,
        reference,
        logging,
        observed_q,
        outcome_lower=0.0,
        outcome_upper=1.0,
    )

    assert certificate.point_identified
    assert certificate.unsupported_target_mass == pytest.approx(0.5)
    assert certificate.unsupported_reference_mass == pytest.approx(0.5)
    assert certificate.unsupported_contrast_mass == pytest.approx(0.0)
    assert direct.width == pytest.approx(0.0)


def test_signed_contrast_bounds_are_tighter_than_separate_value_bounds() -> None:
    policy = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    reference = np.full((2, 2), 0.5)
    logging = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    observed_q = np.asarray([[0.8, np.nan], [np.nan, 0.7]])

    direct = contrast_identification_interval(
        policy,
        reference,
        logging,
        observed_q,
        outcome_lower=0.0,
        outcome_upper=1.0,
    )
    separate = separate_value_difference_interval(
        policy,
        reference,
        logging,
        observed_q,
        outcome_lower=0.0,
        outcome_upper=1.0,
    )

    assert not contrast_support_certificate(policy, reference, logging).point_identified
    assert direct.width == pytest.approx(0.5)
    assert separate.width == pytest.approx(0.5)
    assert direct.lower == pytest.approx(-0.125)
    assert direct.upper == pytest.approx(0.375)


def test_contrast_cancellation_strictly_tightens_separate_bounds() -> None:
    policy = np.asarray([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]])
    reference = np.asarray([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
    logging = np.asarray([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])
    observed_q = np.asarray([[np.nan, 0.7, 0.1], [np.nan, 0.2, 0.8]])

    direct = contrast_identification_interval(
        policy,
        reference,
        logging,
        observed_q,
        outcome_lower=0.0,
        outcome_upper=1.0,
    )
    separate = separate_value_difference_interval(
        policy,
        reference,
        logging,
        observed_q,
        outcome_lower=0.0,
        outcome_upper=1.0,
    )

    assert direct.width == pytest.approx(0.0)
    assert separate.width == pytest.approx(1.0)


def test_centered_dr_score_recovers_matching_value_in_expectation() -> None:
    rng = np.random.default_rng(7)
    n = 300_000
    user = rng.integers(0, 2, size=n)
    policy = np.where(user[:, None] == np.arange(2)[None, :], 0.8, 0.2)
    marginal_policy = np.full((n, 2), 0.5)
    q = 0.1 + 0.2 * (user[:, None] == np.arange(2)[None, :])
    marginal_q = np.full((n, 2), 0.2)
    actions = rng.integers(0, 2, size=n)
    rewards = rng.binomial(1, q[np.arange(n), actions])
    scores = centered_matching_dr_scores(
        policy,
        marginal_policy,
        actions,
        rewards,
        np.full(n, 0.5),
        q,
        marginal_q,
    )

    assert scores.mean() == pytest.approx(0.06, abs=0.002)


def test_support_adapted_score_reduces_to_centered_score_under_full_support() -> None:
    policy = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    reference = np.full((2, 2), 0.5)
    actions = np.asarray([0, 1])
    rewards = np.asarray([1.0, 0.0])
    pscore = np.full(2, 0.5)
    q = np.asarray([[0.7, 0.1], [0.2, 0.8]])
    marginal_q = np.asarray([[0.45, 0.45], [0.45, 0.45]])

    full_support = support_adapted_matching_dr_scores(
        policy,
        reference,
        np.ones_like(policy, dtype=bool),
        actions,
        rewards,
        pscore,
        q,
        marginal_q,
    )
    centered = centered_matching_dr_scores(
        policy,
        reference,
        actions,
        rewards,
        pscore,
        q,
        marginal_q,
    )

    np.testing.assert_allclose(full_support, centered)


def test_support_adapted_score_ignores_unidentified_outcomes_at_cancelled_holes() -> None:
    policy = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 2.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0],
            [1.0 / 2.0, 1.0 / 6.0, 1.0 / 3.0],
        ]
    )
    reference = np.full((3, 3), 1.0 / 3.0)
    support = ~np.eye(3, dtype=bool)
    q = np.asarray(
        [
            [np.nan, 0.80, 0.10],
            [0.20, np.nan, 0.90],
            [0.70, 0.30, np.nan],
        ]
    )
    supported_marginal_q = np.broadcast_to(
        np.nan_to_num(q, nan=0.0).mean(axis=0),
        q.shape,
    )
    actions = np.asarray([1, 2, 0])
    rewards = np.asarray([0.80, 0.90, 0.70])

    scores = support_adapted_matching_dr_scores(
        policy,
        reference,
        support,
        actions,
        rewards,
        np.full(3, 0.5),
        q,
        supported_marginal_q,
    )

    assert scores.mean() == pytest.approx(0.10)


def test_support_adapted_score_rejects_logged_action_outside_design_support() -> None:
    policy = np.full((2, 2), 0.5)
    support = np.asarray([[False, True], [True, False]])

    with pytest.raises(ValueError, match="outside declared structural support"):
        support_adapted_matching_dr_scores(
            policy,
            policy,
            support,
            np.asarray([0, 1]),
            np.asarray([0.0, 1.0]),
            np.ones(2),
            np.asarray([[np.nan, 0.2], [0.8, np.nan]]),
            np.zeros((2, 2)),
        )


def test_centered_score_matches_a_finite_model_pathwise_derivative() -> None:
    user_probability = np.asarray([0.4, 0.6])
    user_score = np.asarray([1.5, -1.0])
    policy_by_user = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    logging_by_user = np.asarray([[0.6, 0.4], [0.3, 0.7]])
    q_by_user = np.asarray([[0.7, 0.2], [0.1, 0.8]])
    outcome_logit_score = np.asarray([[0.4, -0.2], [0.3, 0.5]])
    reference = user_probability @ policy_by_user
    marginal_q = user_probability @ q_by_user
    true_value = matching_value(
        policy_by_user,
        q_by_user,
        np.zeros(2, dtype=np.int64),
        sample_weight=user_probability,
    )

    policies: list[np.ndarray] = []
    references: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    pscores: list[float] = []
    predictions: list[np.ndarray] = []
    marginal_predictions: list[np.ndarray] = []
    observation_scores: list[float] = []
    observation_weights: list[float] = []
    for user in range(2):
        for action in range(2):
            for reward in (0.0, 1.0):
                reward_probability = (
                    q_by_user[user, action]
                    if reward == 1.0
                    else 1.0 - q_by_user[user, action]
                )
                policies.append(policy_by_user[user])
                references.append(reference)
                actions.append(action)
                rewards.append(reward)
                pscores.append(logging_by_user[user, action])
                predictions.append(q_by_user[user])
                marginal_predictions.append(marginal_q)
                observation_scores.append(
                    user_score[user]
                    + outcome_logit_score[user, action]
                    * (reward - q_by_user[user, action])
                )
                observation_weights.append(
                    user_probability[user]
                    * logging_by_user[user, action]
                    * reward_probability
                )
    estimating_scores = centered_matching_dr_scores(
        np.asarray(policies),
        np.asarray(references),
        np.asarray(actions),
        np.asarray(rewards),
        np.asarray(pscores),
        np.asarray(predictions),
        np.asarray(marginal_predictions),
    )
    influence_function = estimating_scores - true_value
    influence_derivative = float(
        np.dot(
            np.asarray(observation_weights),
            influence_function * np.asarray(observation_scores),
        )
    )

    def path_value(step: float) -> float:
        tilted_user_probability = user_probability * (1.0 + step * user_score)
        logits = np.log(q_by_user / (1.0 - q_by_user))
        tilted_q = 1.0 / (
            1.0 + np.exp(-(logits + step * outcome_logit_score))
        )
        return matching_value(
            policy_by_user,
            tilted_q,
            np.zeros(2, dtype=np.int64),
            sample_weight=tilted_user_probability,
        )

    step = 1e-5
    numerical_derivative = (path_value(step) - path_value(-step)) / (2.0 * step)

    assert influence_derivative == pytest.approx(numerical_derivative, abs=1e-9)


def test_support_adapted_marginal_nuisance_bias_is_exactly_second_order() -> None:
    policy_by_user = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 2.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0],
            [1.0 / 2.0, 1.0 / 6.0, 1.0 / 3.0],
        ]
    )
    support_by_user = ~np.eye(3, dtype=bool)
    q_by_user = np.asarray(
        [[0.95, 0.80, 0.10], [0.20, 0.05, 0.90], [0.70, 0.30, 0.40]]
    )
    true_reference = policy_by_user.mean(axis=0)
    true_supported_marginal_q = np.where(
        support_by_user,
        q_by_user,
        0.0,
    ).mean(axis=0)
    perturbation = 0.02 * np.asarray([1.0, -1.0, 0.0])
    estimated_reference = true_reference + perturbation
    estimated_supported_marginal_q = true_supported_marginal_q + perturbation
    true_value = matching_value(
        policy_by_user,
        q_by_user,
        np.zeros(3, dtype=np.int64),
    )

    user = np.repeat(np.arange(3), 2)
    action = np.concatenate([np.flatnonzero(row) for row in support_by_user])
    q = q_by_user[user].copy()
    support = support_by_user[user]
    q[~support] = np.nan
    reward = q_by_user[user, action]
    scores = support_adapted_matching_dr_scores(
        policy_by_user[user],
        np.broadcast_to(estimated_reference, (user.size, 3)),
        support,
        action,
        reward,
        np.full(user.size, 0.5),
        q,
        np.broadcast_to(estimated_supported_marginal_q, (user.size, 3)),
    )

    expected_bias = float(np.dot(perturbation, perturbation))
    assert scores.mean() - true_value == pytest.approx(expected_bias, abs=1e-15)


def test_support_adapted_score_matches_supported_functional_pathwise_derivative() -> None:
    user_probability = np.full(3, 1.0 / 3.0)
    user_score = np.asarray([1.0, -0.5, -0.5])
    policy_by_user = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 2.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0],
            [1.0 / 2.0, 1.0 / 6.0, 1.0 / 3.0],
        ]
    )
    support_by_user = ~np.eye(3, dtype=bool)
    q_by_user = np.asarray(
        [[0.95, 0.80, 0.10], [0.20, 0.05, 0.90], [0.70, 0.30, 0.40]]
    )
    outcome_logit_score = np.asarray(
        [[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1], [0.6, -0.2, 0.3]]
    )
    reference = user_probability @ policy_by_user
    supported_marginal_q = user_probability @ np.where(
        support_by_user,
        q_by_user,
        0.0,
    )
    true_supported_value = float(
        np.sum(
            user_probability[:, None]
            * support_by_user
            * (policy_by_user - reference)
            * q_by_user
        )
    )

    policies: list[np.ndarray] = []
    references: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    predictions: list[np.ndarray] = []
    marginal_predictions: list[np.ndarray] = []
    observation_scores: list[float] = []
    observation_weights: list[float] = []
    for user in range(3):
        for action in np.flatnonzero(support_by_user[user]):
            for reward in (0.0, 1.0):
                reward_probability = (
                    q_by_user[user, action]
                    if reward == 1.0
                    else 1.0 - q_by_user[user, action]
                )
                policies.append(policy_by_user[user])
                references.append(reference)
                supports.append(support_by_user[user])
                actions.append(int(action))
                rewards.append(reward)
                q_row = q_by_user[user].copy()
                q_row[~support_by_user[user]] = np.nan
                predictions.append(q_row)
                marginal_predictions.append(supported_marginal_q)
                observation_scores.append(
                    user_score[user]
                    + outcome_logit_score[user, action]
                    * (reward - q_by_user[user, action])
                )
                observation_weights.append(
                    user_probability[user] * 0.5 * reward_probability
                )
    estimating_scores = support_adapted_matching_dr_scores(
        np.asarray(policies),
        np.asarray(references),
        np.asarray(supports),
        np.asarray(actions),
        np.asarray(rewards),
        np.full(len(actions), 0.5),
        np.asarray(predictions),
        np.asarray(marginal_predictions),
    )
    influence_derivative = float(
        np.dot(
            np.asarray(observation_weights),
            (estimating_scores - true_supported_value)
            * np.asarray(observation_scores),
        )
    )

    def path_value(step: float) -> float:
        tilted_user_probability = user_probability * (1.0 + step * user_score)
        logits = np.log(q_by_user / (1.0 - q_by_user))
        tilted_q = 1.0 / (
            1.0 + np.exp(-(logits + step * outcome_logit_score))
        )
        tilted_reference = tilted_user_probability @ policy_by_user
        return float(
            np.sum(
                tilted_user_probability[:, None]
                * support_by_user
                * (policy_by_user - tilted_reference)
                * tilted_q
            )
        )

    step = 1e-5
    numerical_derivative = (path_value(step) - path_value(-step)) / (2.0 * step)

    assert influence_derivative == pytest.approx(numerical_derivative, abs=1e-9)


def test_equality_restricted_projection_is_canonical_and_reduces_variance() -> None:
    user_probability = np.full(4, 0.25)
    user_score = np.asarray([1.0, 1.0, -7.0, 5.0])
    policy_by_user = np.asarray(
        [
            [0.3, 0.5, 0.2],
            [0.6, 0.2, 0.2],
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
        ]
    )
    support_by_user = np.ones_like(policy_by_user, dtype=bool)
    support_by_user[0, 0] = False
    hole_actions = np.broadcast_to(
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
    q_by_user = np.asarray(
        [
            [0.4, 0.9, 0.1],
            [0.8, 0.2, 0.4],
            [0.3, 0.6, 0.7],
            [0.2, 0.5, 0.8],
        ]
    )
    outcome_logit_score = np.asarray(
        [
            [0.1, -0.2, 0.4],
            [0.3, 0.5, -0.1],
            [-0.4, 0.2, 0.6],
            [0.5, -0.3, 0.2],
        ]
    )
    reference = user_probability @ policy_by_user
    supported_marginal_q = user_probability @ np.where(
        support_by_user,
        q_by_user,
        0.0,
    )
    q_for_fit = q_by_user.copy()
    q_for_fit[~support_by_user] = np.nan
    coefficients = equality_restricted_projection_coefficients(
        policy_by_user,
        np.broadcast_to(reference, policy_by_user.shape),
        support_by_user,
        hole_actions,
        q_for_fit,
        np.broadcast_to(supported_marginal_q, policy_by_user.shape),
        np.zeros(4, dtype=np.int64),
        sample_weight=user_probability,
    )
    true_value = matching_value(
        policy_by_user,
        q_by_user,
        np.zeros(4, dtype=np.int64),
        sample_weight=user_probability,
    )

    policies: list[np.ndarray] = []
    references: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    constrained_actions: list[np.ndarray] = []
    projection_coefficients: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    pscores: list[float] = []
    predictions: list[np.ndarray] = []
    marginal_predictions: list[np.ndarray] = []
    observation_scores: list[float] = []
    observation_weights: list[float] = []
    for user in range(4):
        for action in np.flatnonzero(support_by_user[user]):
            for reward in (0.0, 1.0):
                reward_probability = (
                    q_by_user[user, action]
                    if reward == 1.0
                    else 1.0 - q_by_user[user, action]
                )
                policies.append(policy_by_user[user])
                references.append(reference)
                supports.append(support_by_user[user])
                constrained_actions.append(hole_actions[user])
                projection_coefficients.append(coefficients[user])
                actions.append(int(action))
                rewards.append(reward)
                pscores.append(logging_by_user[user, action])
                q_row = q_by_user[user].copy()
                q_row[~support_by_user[user]] = np.nan
                predictions.append(q_row)
                marginal_predictions.append(supported_marginal_q)
                observation_scores.append(
                    user_score[user]
                    + outcome_logit_score[user, action]
                    * (reward - q_by_user[user, action])
                )
                observation_weights.append(
                    user_probability[user]
                    * logging_by_user[user, action]
                    * reward_probability
                )
    score_arguments = (
        np.asarray(policies),
        np.asarray(references),
        np.asarray(supports),
        np.asarray(actions),
        np.asarray(rewards),
        np.asarray(pscores),
        np.asarray(predictions),
        np.asarray(marginal_predictions),
    )
    unrestricted_scores = support_adapted_matching_dr_scores(*score_arguments)
    restricted_scores = equality_restricted_matching_dr_scores(
        np.asarray(policies),
        np.asarray(references),
        np.asarray(supports),
        np.asarray(constrained_actions),
        np.asarray(projection_coefficients),
        np.asarray(actions),
        np.asarray(rewards),
        np.asarray(pscores),
        np.asarray(predictions),
        np.asarray(marginal_predictions),
    )
    weights = np.asarray(observation_weights)
    restricted_derivative = float(
        np.dot(
            weights,
            (restricted_scores - true_value) * np.asarray(observation_scores),
        )
    )

    def path_value(step: float) -> float:
        tilted_user_probability = user_probability * (1.0 + step * user_score)
        logits = np.log(q_by_user / (1.0 - q_by_user))
        tilted_q = 1.0 / (
            1.0 + np.exp(-(logits + step * outcome_logit_score))
        )
        return matching_value(
            policy_by_user,
            tilted_q,
            np.zeros(4, dtype=np.int64),
            sample_weight=tilted_user_probability,
        )

    step = 1e-5
    numerical_derivative = (path_value(step) - path_value(-step)) / (2.0 * step)
    unrestricted_variance = float(
        np.dot(weights, np.square(unrestricted_scores - true_value))
    )
    restricted_variance = float(
        np.dot(weights, np.square(restricted_scores - true_value))
    )

    assert restricted_derivative == pytest.approx(numerical_derivative, abs=1e-9)
    assert restricted_variance < unrestricted_variance


def test_equality_restricted_score_requires_exact_hole_certificate() -> None:
    policy = np.asarray([[0.5, 0.5], [0.8, 0.2]])
    noisy_reference = np.asarray([[0.49, 0.51], [0.49, 0.51]])
    support = np.asarray([[False, True], [True, True]])
    hole_actions = np.asarray([[True, False], [True, False]])

    with pytest.raises(ValueError, match="does not cancel"):
        equality_restricted_projection_coefficients(
            policy,
            noisy_reference,
            support,
            hole_actions,
            np.asarray([[np.nan, 0.3], [0.7, 0.2]]),
            np.zeros((2, 2)),
            np.zeros(2, dtype=np.int64),
        )


def test_mde_hits_requested_power_and_score_summary_is_consistent() -> None:
    mde = normal_minimum_detectable_effect(
        standard_error=2.0,
        alpha=0.05,
        target_power=0.80,
    )
    estimate = estimate_from_scores(np.asarray([-1.0, 0.0, 1.0, 2.0]))

    assert mde == pytest.approx(5.60317, abs=1e-5)
    assert two_sided_normal_power(mde, 2.0) == pytest.approx(0.80, abs=1e-12)
    assert estimate.value == pytest.approx(0.5)
    assert estimate.minimum_detectable_effect > 0.0
