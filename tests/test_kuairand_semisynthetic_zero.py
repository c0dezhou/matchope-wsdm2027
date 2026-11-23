from __future__ import annotations

import numpy as np

from matchope.pipeline.run_kuairand_semisynthetic_zero import (
    additive_matching_truth,
    policy_aligned_interaction_shape,
    sample_contextual_actions,
)


def test_additive_surface_has_exact_zero_matching_value() -> None:
    policy = np.asarray(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.1, 0.9],
        ]
    )
    user = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
    context = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    user_effect = np.asarray([-0.05, 0.02, 0.04, -0.01])
    context_action_effect = np.asarray([[-0.1, 0.08], [0.03, -0.07]])

    truth = additive_matching_truth(
        policy,
        user,
        context,
        intercept=0.5,
        user_effect=user_effect,
        context_action_effect=context_action_effect,
    )

    assert abs(truth) < 1e-14


def test_contextual_action_sampler_tracks_each_logger_row() -> None:
    contexts = np.repeat(np.asarray([0, 1]), 20_000)
    logger = np.asarray([[0.8, 0.2], [0.25, 0.75]])

    actions = sample_contextual_actions(
        contexts,
        logger,
        rng=np.random.default_rng(17),
    )

    assert abs(actions[contexts == 0].mean() - 0.2) < 0.015
    assert abs(actions[contexts == 1].mean() - 0.75) < 0.015


def test_policy_aligned_interaction_can_set_known_nonzero_truth() -> None:
    policy = np.asarray(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.1, 0.9],
        ]
    )
    user = np.arange(4)
    context = np.zeros(4, dtype=np.int64)
    shape = policy_aligned_interaction_shape(policy)
    additive = {
        "intercept": 0.1,
        "user_effect": np.zeros(4),
        "context_action_effect": np.zeros((1, 2)),
        "interaction_shape": shape,
    }
    unit_truth = additive_matching_truth(
        policy,
        user,
        context,
        interaction_coefficient=1.0,
        **additive,
    )
    target = 0.002

    truth = additive_matching_truth(
        policy,
        user,
        context,
        interaction_coefficient=target / unit_truth,
        **additive,
    )

    assert unit_truth > 0.0
    assert abs(truth - target) < 1e-14
