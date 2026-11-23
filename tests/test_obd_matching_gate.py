from __future__ import annotations

import numpy as np

from matchope.eval.matching_value import matching_value
from matchope.pipeline.run_obd_matching_gate import (
    affinity_limit_policy,
    affinity_policy,
    cluster_summary,
    day_cross_fitted_marginals,
    reference_context_ipw_sensitivity,
)
from matchope.pipeline.run_obd_semisynthetic_power import (
    cross_fitted_logged_action_means,
    evaluate_semisynthetic_power,
    leave_one_day_policy_marginals,
    semisynthetic_outcome_parameters,
)


def test_affinity_policy_is_uniform_at_zero_and_prefers_larger_counts() -> None:
    affinity = np.asarray([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])

    uniform = affinity_policy(affinity, inverse_temperature=0.0)
    targeted = affinity_policy(affinity, inverse_temperature=1.0)

    np.testing.assert_allclose(uniform, np.full((2, 3), 1.0 / 3.0))
    assert targeted[0, 1] > targeted[0, 0]
    np.testing.assert_allclose(targeted[1], np.full(3, 1.0 / 3.0))


def test_affinity_limit_policy_splits_ties_exactly() -> None:
    affinity = np.asarray([[1.0, 3.0, 3.0], [0.0, 0.0, 0.0]])

    policy = affinity_limit_policy(affinity)

    np.testing.assert_allclose(policy[0], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(policy[1], np.full(3, 1.0 / 3.0))


def test_day_cross_fitting_uses_other_day_policy_margins() -> None:
    policy = np.asarray(
        [[0.8, 0.2], [0.6, 0.4], [0.2, 0.8], [0.4, 0.6]]
    )
    marginal_policy, marginal_q, shift = day_cross_fitted_marginals(
        policy=policy,
        actions=np.asarray([0, 1, 0, 1]),
        positions=np.zeros(4, dtype=int),
        rewards=np.asarray([1.0, 0.0, 0.0, 1.0]),
        days=np.asarray([0, 0, 1, 1]),
        outcome_smoothing=1.0,
    )

    np.testing.assert_allclose(marginal_policy[:2], [[0.3, 0.7], [0.3, 0.7]])
    np.testing.assert_allclose(marginal_policy[2:], [[0.7, 0.3], [0.7, 0.3]])
    assert np.isfinite(marginal_q).all()
    assert shift > 0.0


def test_cluster_summary_reflects_between_day_variation() -> None:
    summary = cluster_summary(
        np.asarray([1.0, 1.0, -1.0, -1.0]),
        np.asarray([0, 0, 1, 1]),
    )

    assert summary["day_clusters"] == 2
    assert summary["day_cluster_standard_error"] > 0.0
    assert summary["day_cluster_ci_lower"] < 0.0 < summary["day_cluster_ci_upper"]
    assert summary["day_cluster_t_mde_80pct"] > 0.0


def test_reference_context_sensitivity_conditions_on_date() -> None:
    policy = np.asarray(
        [[0.8, 0.2], [0.6, 0.4], [0.2, 0.8], [0.4, 0.6]]
    )
    summary = reference_context_ipw_sensitivity(
        policy=policy,
        actions=np.zeros(4, dtype=int),
        rewards=np.asarray([1.0, 0.0, 0.0, 0.0]),
        logging_pscore=np.full(4, 0.5),
        days=np.asarray([0, 0, 1, 1]),
    )

    assert summary["fixed_window_reference_mean_tv"] > 0.0
    assert summary["date_reference_mean_tv"] > 0.0
    assert not np.isclose(
        summary["fixed_window_reference_ipw_matching_value"],
        summary["date_reference_ipw_matching_value"],
    )


def test_semisynthetic_outcome_has_declared_matching_value() -> None:
    policy = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.1, 0.7],
        ]
    )
    days = np.asarray([0, 0, 1, 1, 2, 2])
    observed = np.asarray([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    parameters = semisynthetic_outcome_parameters(
        policy=policy,
        days=days,
        observed_rewards=observed,
        target_matching_value=0.02,
    )
    outcome_surface = (
        np.asarray(parameters["day_baselines"])[
            np.asarray(parameters["day_codes"]), None
        ]
        + float(parameters["interaction_scale"]) * policy
    )

    value = matching_value(policy, outcome_surface, np.zeros(policy.shape[0]))

    np.testing.assert_allclose(value, 0.02, atol=1e-12)


def test_semisynthetic_helpers_cross_fit_by_day() -> None:
    policy = np.asarray(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.2, 0.8],
            [0.4, 0.6],
        ]
    )
    days = np.asarray([0, 0, 1, 1])
    marginals = leave_one_day_policy_marginals(policy, days)
    aligned_q = cross_fitted_logged_action_means(
        actions=np.asarray([0, 1, 0, 1]),
        rewards=np.asarray([1.0, 0.0, 0.0, 1.0]),
        days=days,
        n_actions=2,
        smoothing=1.0,
    )

    np.testing.assert_allclose(marginals, [[0.3, 0.7], [0.7, 0.3]])
    np.testing.assert_allclose(aligned_q, [0.25, 0.75, 0.75, 0.25])


def test_semisynthetic_power_returns_finite_calibration_rows() -> None:
    affinity = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    feedback = {
        "affinity": affinity,
        "rewards": np.zeros(6),
        "days": np.asarray([0, 0, 1, 1, 2, 2]),
        "pscore": np.full(6, 0.5),
    }

    rows, protocol = evaluate_semisynthetic_power(
        feedback,
        target_matching_values=(0.0, 0.05),
        repetitions=20,
        seed=7,
        outcome_smoothing=1.0,
    )

    assert len(rows) == 2
    assert protocol["days"] == 3
    assert rows[0]["interaction_scale"] == 0.0
    for row in rows:
        assert all(np.isfinite(float(value)) for value in row.values())
        assert 0.0 <= row["day_positive_detection_rate"] <= 1.0
        assert 0.0 <= row["day_interval_coverage"] <= 1.0
