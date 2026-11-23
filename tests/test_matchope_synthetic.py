from __future__ import annotations

import json

from matchope.pipeline.run_matchope_synthetic import (
    _json_ready,
    cluster_coverage_grid,
    detectability_frontier,
    equality_restricted_efficiency_grid,
    mechanism_scenarios,
    nested_context_day_coverage_grid,
    nuisance_coverage_grid,
    support_cancellation_coverage,
)


def test_json_ready_uses_null_for_unavailable_numeric_results() -> None:
    payload = {"available": 1.0, "unavailable": float("nan")}

    assert json.dumps(_json_ready(payload), allow_nan=False) == (
        '{"available": 1.0, "unavailable": null}'
    )


def test_nested_day_grid_keeps_contexts_inside_days() -> None:
    rows = nested_context_day_coverage_grid(
        day_counts=(7, 10),
        repetitions=30,
        seed=59,
        n_users=40,
        total_rows=1_200,
    )

    assert {int(row["contexts"]) for row in rows if row["days"] == 7} == {21}
    assert {int(row["contexts"]) for row in rows if row["days"] == 10} == {30}
    assert any(row["method"] == "day_sign_flip" for row in rows if row["days"] == 7)
    assert all(row["method"] != "day_sign_flip" for row in rows if row["days"] == 10)
    assert all(
        row["valid_repetitions"] + row["variance_failures"] == 30
        for row in rows
    )


def test_cluster_coverage_requires_the_matching_cluster_dimension() -> None:
    rows = cluster_coverage_grid(
        repetitions=100,
        seed=53,
        n_users=80,
        n_days=10,
    )
    by_design = {
        (str(row["regime"]), str(row["method"])): row
        for row in rows
    }

    iid = by_design[("iid", "iid")]
    day_iid = by_design[("day_cluster", "iid")]
    day_clustered = by_design[("day_cluster", "day")]
    both_iid = by_design[("user_and_day_cluster", "iid")]
    both_two_way = by_design[("user_and_day_cluster", "two_way")]

    assert iid["coverage_95pct"] >= 0.8
    assert day_clustered["coverage_95pct"] >= 0.8
    assert day_clustered["coverage_95pct"] > day_iid["coverage_95pct"] + 0.2
    assert both_two_way["coverage_95pct"] >= 0.8
    assert both_two_way["coverage_95pct"] > both_iid["coverage_95pct"] + 0.2
    assert both_two_way["variance_failures"] == 0


def test_mechanism_scenarios_cover_the_claim_boundary_cases() -> None:
    rows = {row["scenario"]: row for row in mechanism_scenarios()}

    assert abs(float(rows["additive_outcome"]["matching_value"])) < 1e-14
    assert rows["unsupported_common_mass_cancels"]["point_identified"] is True
    assert (
        rows["unsupported_common_mass_cancels"]["direct_bound_width"]
        < rows["unsupported_common_mass_cancels"]["separate_bound_width"]
    )
    assert rows["marginal_reference_breaks_support"]["point_identified"] is False
    assert rows["unsupported_common_mass_cancels"]["actionable_gain"] is None
    assert rows["marginal_reference_breaks_support"]["actionable_gain"] is None
    assert rows["positive_matching_negative_actionable"]["matching_value"] > 0.0
    assert rows["positive_matching_negative_actionable"]["actionable_gain"] < 0.0
    json.dumps(list(rows.values()), allow_nan=False)


def test_detectability_frontier_recovers_null_and_strong_signal() -> None:
    rows = detectability_frontier(
        sample_sizes=(20_000,),
        interaction_strengths=(0.0, 0.1),
        repetitions=80,
        seed=19,
    )
    null, signal = rows

    assert null["positive_detection_rate"] <= 0.1
    assert signal["positive_detection_rate"] >= 0.8
    assert abs(signal["mean_estimate"] - signal["true_matching_value"]) < 0.005


def test_nuisance_grid_has_valid_case_and_joint_misspecification_control() -> None:
    rows = nuisance_coverage_grid(
        sample_sizes=(2_000,),
        repetitions=40,
        seed=31,
        n_strata=8,
    )
    centered = {
        row["scenario"]: row
        for row in rows
        if row["method"] == "centered_score"
    }

    assert {int(row["folds"]) for row in rows} == {5}
    assert abs(centered["all_estimated_well_specified"]["bias"]) < 0.01
    assert abs(centered["q_misspecified"]["bias"]) < 0.01
    assert centered["all_estimated_well_specified"]["coverage_95pct"] >= 0.75
    assert abs(centered["both_marginals_misspecified"]["bias"]) > 0.01
    assert (
        centered["both_marginals_misspecified"]["coverage_95pct"]
        < centered["all_estimated_well_specified"]["coverage_95pct"]
    )


def test_support_cancellation_has_calibrated_point_inference() -> None:
    rows = support_cancellation_coverage(
        sample_sizes=(2_000,),
        repetitions=80,
        seed=43,
    )
    row = rows[0]

    assert row["unsupported_target_mass"] > 0.0
    assert row["unsupported_reference_mass"] > 0.0
    assert row["unsupported_contrast_mass"] == 0.0
    assert abs(row["bias"]) < 0.01
    assert row["coverage_95pct"] >= 0.85


def test_equality_restriction_reduces_variance_without_losing_coverage() -> None:
    rows = equality_restricted_efficiency_grid(
        sample_sizes=(2_000,),
        repetitions=120,
        seed=47,
    )
    by_method = {row["method"]: row for row in rows}
    unrestricted = by_method["support_adapted_unrestricted"]
    efficient = by_method["equality_restricted_efficient"]

    assert abs(efficient["bias"]) < 0.01
    assert efficient["coverage_95pct"] >= 0.85
    assert efficient["empirical_standard_deviation"] < unrestricted[
        "empirical_standard_deviation"
    ]
    assert efficient["variance_ratio_to_unrestricted"] < 1.0
