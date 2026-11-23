from __future__ import annotations

import numpy as np

from matchope.pipeline.run_kuairec_learned_logging_recovery import (
    build_learned_logging_design,
    run_learned_logging_recovery,
)


def test_learned_logging_recovery_recovers_policy_gap() -> None:
    factor_score = np.asarray(
        [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]]
    )
    knn_score = np.asarray(
        [[0.0, 2.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]
    )
    outcome = np.asarray(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
    )
    folds = np.asarray([0, 0, 1, 1])
    design = build_learned_logging_design(
        factor_score,
        knn_score,
        outcome,
        outcome,
        evaluation_fold_ids=folds,
        target_tv=0.1,
        logging_inverse_temperature=2.0,
    )
    rows = run_learned_logging_recovery(
        design,
        sample_sizes=(20_000,),
        repetitions=30,
        seed=17,
    )

    centered_gap = next(
        row
        for row in rows
        if row["method"] == "centered_score"
        and row["estimand"] == "knn_minus_factor"
    )
    assert design.matching_gap < 0.0
    assert abs(float(centered_gap["bias"])) < 0.01
    assert float(centered_gap["correct_sign_rate"]) >= 0.8
