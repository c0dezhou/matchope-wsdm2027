from __future__ import annotations

"""Identification and inference utilities for personalization matching value.

The target policy is ``policy(a | c, u)``.  Its matching-free reference is the
policy averaged over the personalization variables within shared-context
strata ``c``.  Candidate eligibility must therefore be constant inside a
stratum; otherwise averaging can create infeasible actions.

The routines here deliberately separate three objects:

* the policy contrast, which defines the estimand;
* design support, which determines identification; and
* finite-sample scores, which determine detectability.

They do not infer structural support from a finite action count.  Callers must
provide logging probabilities (or a design-derived support matrix).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass(frozen=True)
class ContrastSupportCertificate:
    """Design-support summary for a signed policy contrast."""

    n_rows: int
    n_actions: int
    point_identified: bool
    affected_row_fraction: float
    unsupported_cell_count: int
    unsupported_target_mass: float
    unsupported_reference_mass: float
    unsupported_contrast_mass: float
    unsupported_positive_contrast_mass: float
    unsupported_negative_contrast_mass: float
    support_tolerance: float
    contrast_tolerance: float


@dataclass(frozen=True)
class IdentificationInterval:
    """Bounds for a policy contrast under a bounded outcome."""

    lower: float
    upper: float
    width: float
    supported_component: float
    unsupported_contrast_mass: float


@dataclass(frozen=True)
class ContrastEstimate:
    """Normal-approximation inference from row-level contrast scores."""

    value: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    alpha: float
    target_power: float
    minimum_detectable_effect: float
    n_rows: int


def _as_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    return matrix


def _as_probability_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = _as_matrix(values, name=name)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"each row of {name} must sum to one")
    return matrix


def _normalized_weights(sample_weight: np.ndarray | None, n_rows: int) -> np.ndarray:
    if sample_weight is None:
        return np.full(n_rows, 1.0 / n_rows, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (n_rows,):
        raise ValueError("sample_weight must have one entry per row")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("sample_weight must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("sample_weight must have positive total mass")
    return weights / total


def _stratum_codes(shared_strata: np.ndarray, n_rows: int) -> np.ndarray:
    strata = np.asarray(shared_strata)
    if strata.ndim != 1 or strata.shape[0] != n_rows:
        raise ValueError("shared_strata must be one-dimensional with one entry per row")
    if strata.dtype.kind in {"f", "c"} and not np.isfinite(strata).all():
        raise ValueError("shared_strata must not contain non-finite values")
    _, codes = np.unique(strata, return_inverse=True)
    return codes.astype(np.int64, copy=False)


def stratified_mean(
    values: np.ndarray,
    shared_strata: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Return the weighted within-stratum mean aligned to every input row."""

    matrix = _as_matrix(values, name="values")
    if not np.isfinite(matrix).all():
        raise ValueError("values must contain only finite values")
    n_rows, n_columns = matrix.shape
    codes = _stratum_codes(shared_strata, n_rows)
    weights = _normalized_weights(sample_weight, n_rows)
    n_strata = int(codes.max()) + 1
    totals = np.zeros((n_strata, n_columns), dtype=np.float64)
    masses = np.zeros(n_strata, dtype=np.float64)
    np.add.at(totals, codes, matrix * weights[:, None])
    np.add.at(masses, codes, weights)
    means = totals / masses[:, None]
    return means[codes]


def marginalize_policy(
    policy: np.ndarray,
    shared_strata: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    eligibility: np.ndarray | None = None,
) -> np.ndarray:
    """Average a target policy over users inside shared-context strata.

    ``eligibility`` is optional only when every action is feasible for every
    row.  When supplied, it must be identical for all rows in a stratum.  This
    prevents the marginal reference from assigning an action that is feasible
    for a donor user but impossible for the recipient.
    """

    probabilities = _as_probability_matrix(policy, name="policy")
    codes = _stratum_codes(shared_strata, probabilities.shape[0])
    if eligibility is not None:
        feasible = np.asarray(eligibility, dtype=bool)
        if feasible.shape != probabilities.shape:
            raise ValueError("eligibility must have the same shape as policy")
        if np.any(probabilities[~feasible] > 1e-12):
            raise ValueError("policy assigns positive mass to an ineligible action")
        for code in range(int(codes.max()) + 1):
            rows = np.flatnonzero(codes == code)
            if not np.all(feasible[rows] == feasible[rows[0]]):
                raise ValueError(
                    "candidate eligibility varies inside a shared-context stratum; "
                    "refine shared_strata before marginalizing the policy"
                )
    reference = stratified_mean(
        probabilities,
        shared_strata,
        sample_weight=sample_weight,
    )
    return _as_probability_matrix(reference, name="marginal_policy")


def matching_value(
    policy: np.ndarray,
    outcome_surface: np.ndarray,
    shared_strata: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    eligibility: np.ndarray | None = None,
) -> float:
    """Compute the empirical-oracle value of matching at fixed action margins."""

    probabilities = _as_probability_matrix(policy, name="policy")
    outcomes = _as_matrix(outcome_surface, name="outcome_surface")
    if outcomes.shape != probabilities.shape or not np.isfinite(outcomes).all():
        raise ValueError("outcome_surface must be finite and aligned with policy")
    reference = marginalize_policy(
        probabilities,
        shared_strata,
        sample_weight=sample_weight,
        eligibility=eligibility,
    )
    weights = _normalized_weights(sample_weight, probabilities.shape[0])
    return float(np.sum(weights[:, None] * (probabilities - reference) * outcomes))


def contrast_support_certificate(
    policy: np.ndarray,
    reference_policy: np.ndarray,
    logging_probability: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    support_tolerance: float = 0.0,
    contrast_tolerance: float = 1e-12,
) -> ContrastSupportCertificate:
    """Certify whether every nonzero contrast component has logging support."""

    if support_tolerance < 0.0 or contrast_tolerance < 0.0:
        raise ValueError("support and contrast tolerances must be non-negative")
    target = _as_probability_matrix(policy, name="policy")
    reference = _as_probability_matrix(reference_policy, name="reference_policy")
    logging = _as_probability_matrix(logging_probability, name="logging_probability")
    if target.shape != reference.shape or target.shape != logging.shape:
        raise ValueError("policy, reference_policy, and logging_probability must align")
    weights = _normalized_weights(sample_weight, target.shape[0])
    unsupported = logging <= support_tolerance
    contrast = target - reference
    active_hole = unsupported & (np.abs(contrast) > contrast_tolerance)
    row_affected = active_hole.any(axis=1)
    positive = np.where(unsupported, np.maximum(contrast, 0.0), 0.0)
    negative = np.where(unsupported, np.maximum(-contrast, 0.0), 0.0)

    def weighted_mass(values: np.ndarray) -> float:
        return float(np.sum(weights[:, None] * values))

    positive_mass = weighted_mass(positive)
    negative_mass = weighted_mass(negative)
    return ContrastSupportCertificate(
        n_rows=int(target.shape[0]),
        n_actions=int(target.shape[1]),
        point_identified=not bool(active_hole.any()),
        affected_row_fraction=float(np.sum(weights[row_affected])),
        unsupported_cell_count=int(active_hole.sum()),
        unsupported_target_mass=weighted_mass(np.where(unsupported, target, 0.0)),
        unsupported_reference_mass=weighted_mass(
            np.where(unsupported, reference, 0.0)
        ),
        unsupported_contrast_mass=positive_mass + negative_mass,
        unsupported_positive_contrast_mass=positive_mass,
        unsupported_negative_contrast_mass=negative_mass,
        support_tolerance=float(support_tolerance),
        contrast_tolerance=float(contrast_tolerance),
    )


def contrast_identification_interval(
    policy: np.ndarray,
    reference_policy: np.ndarray,
    logging_probability: np.ndarray,
    supported_outcome: np.ndarray,
    *,
    outcome_lower: float,
    outcome_upper: float,
    sample_weight: np.ndarray | None = None,
    support_tolerance: float = 0.0,
    contrast_tolerance: float = 1e-12,
) -> IdentificationInterval:
    """Sharp box-constraint bounds for the unsupported contrast component."""

    if not np.isfinite([outcome_lower, outcome_upper]).all():
        raise ValueError("outcome bounds must be finite")
    if outcome_lower > outcome_upper:
        raise ValueError("outcome_lower must not exceed outcome_upper")
    target = _as_probability_matrix(policy, name="policy")
    reference = _as_probability_matrix(reference_policy, name="reference_policy")
    logging = _as_probability_matrix(logging_probability, name="logging_probability")
    outcomes = _as_matrix(supported_outcome, name="supported_outcome")
    if not (target.shape == reference.shape == logging.shape == outcomes.shape):
        raise ValueError("all policy, logging, and outcome arrays must align")
    weights = _normalized_weights(sample_weight, target.shape[0])
    contrast = target - reference
    supported = logging > support_tolerance
    needed_supported = supported & (np.abs(contrast) > contrast_tolerance)
    if not np.isfinite(outcomes[needed_supported]).all():
        raise ValueError("supported_outcome is missing on an active supported contrast")
    supported_values = np.where(needed_supported, contrast * outcomes, 0.0)
    supported_component = float(np.sum(weights[:, None] * supported_values))
    unsupported = ~supported
    positive = np.where(unsupported, np.maximum(contrast, 0.0), 0.0)
    negative = np.where(unsupported, np.maximum(-contrast, 0.0), 0.0)
    positive_mass = float(np.sum(weights[:, None] * positive))
    negative_mass = float(np.sum(weights[:, None] * negative))
    lower = (
        supported_component
        + outcome_lower * positive_mass
        - outcome_upper * negative_mass
    )
    upper = (
        supported_component
        + outcome_upper * positive_mass
        - outcome_lower * negative_mass
    )
    return IdentificationInterval(
        lower=float(lower),
        upper=float(upper),
        width=float(upper - lower),
        supported_component=supported_component,
        unsupported_contrast_mass=positive_mass + negative_mass,
    )


def separate_value_difference_interval(
    policy: np.ndarray,
    reference_policy: np.ndarray,
    logging_probability: np.ndarray,
    supported_outcome: np.ndarray,
    *,
    outcome_lower: float,
    outcome_upper: float,
    sample_weight: np.ndarray | None = None,
    support_tolerance: float = 0.0,
) -> IdentificationInterval:
    """Bound two values separately, then subtract their intervals.

    This is a valid but generally looser baseline because it does not cancel
    common unsupported policy mass before applying the outcome bounds.
    """

    target = _as_probability_matrix(policy, name="policy")
    reference = _as_probability_matrix(reference_policy, name="reference_policy")
    logging = _as_probability_matrix(logging_probability, name="logging_probability")
    outcomes = _as_matrix(supported_outcome, name="supported_outcome")
    if not (target.shape == reference.shape == logging.shape == outcomes.shape):
        raise ValueError("all policy, logging, and outcome arrays must align")
    weights = _normalized_weights(sample_weight, target.shape[0])
    supported = logging > support_tolerance
    if not np.isfinite(outcomes[supported]).all():
        raise ValueError("supported_outcome must be finite wherever logging has support")

    def value_bounds(candidate: np.ndarray) -> tuple[float, float, float]:
        observed = float(
            np.sum(weights[:, None] * np.where(supported, candidate * outcomes, 0.0))
        )
        missing_mass = float(
            np.sum(weights[:, None] * np.where(~supported, candidate, 0.0))
        )
        return (
            observed + outcome_lower * missing_mass,
            observed + outcome_upper * missing_mass,
            missing_mass,
        )

    target_lower, target_upper, target_missing = value_bounds(target)
    reference_lower, reference_upper, reference_missing = value_bounds(reference)
    lower = target_lower - reference_upper
    upper = target_upper - reference_lower
    supported_component = float(
        np.sum(
            weights[:, None]
            * np.where(supported, (target - reference) * outcomes, 0.0)
        )
    )
    return IdentificationInterval(
        lower=float(lower),
        upper=float(upper),
        width=float(upper - lower),
        supported_component=supported_component,
        unsupported_contrast_mass=target_missing + reference_missing,
    )


def contrast_dr_scores(
    policy_contrast: np.ndarray,
    logged_actions: np.ndarray,
    rewards: np.ndarray,
    logging_pscore: np.ndarray,
    outcome_prediction: np.ndarray,
) -> np.ndarray:
    """Row scores for a fixed, externally constructed policy contrast."""

    contrast = _as_matrix(policy_contrast, name="policy_contrast")
    predictions = _as_matrix(outcome_prediction, name="outcome_prediction")
    if contrast.shape != predictions.shape:
        raise ValueError("policy_contrast and outcome_prediction must align")
    if not np.isfinite(contrast).all() or not np.isfinite(predictions).all():
        raise ValueError("contrast and outcome predictions must be finite")
    n_rows, n_actions = contrast.shape
    actions = np.asarray(logged_actions, dtype=np.int64)
    outcomes = np.asarray(rewards, dtype=np.float64)
    pscore = np.asarray(logging_pscore, dtype=np.float64)
    if actions.shape != (n_rows,) or outcomes.shape != (n_rows,) or pscore.shape != (n_rows,):
        raise ValueError("logged arrays must have one entry per policy row")
    if np.any(actions < 0) or np.any(actions >= n_actions):
        raise ValueError("logged_actions contains an invalid action")
    if not np.isfinite(outcomes).all() or not np.isfinite(pscore).all() or np.any(pscore <= 0.0):
        raise ValueError("rewards must be finite and logging_pscore must be positive")
    row = np.arange(n_rows)
    direct = np.sum(contrast * predictions, axis=1)
    correction = (
        contrast[row, actions]
        / pscore
        * (outcomes - predictions[row, actions])
    )
    return direct + correction


def centered_matching_dr_scores(
    policy: np.ndarray,
    marginal_policy: np.ndarray,
    logged_actions: np.ndarray,
    rewards: np.ndarray,
    logging_pscore: np.ndarray,
    outcome_prediction: np.ndarray,
    marginal_outcome_prediction: np.ndarray,
) -> np.ndarray:
    """Orthogonal row scores when the marginal comparator is data-dependent.

    The first term estimates the conditional covariance

    ``sum_a (policy - E[policy|C]) * (q - E[q|C])``.

    The residual correction handles bandit feedback.  With cross-fitted
    nuisances, errors in the two conditional marginal functions enter as a
    product rather than as first-order bias.  If the marginal policy was built
    once on a separate reference population and is treated as fixed, use
    :func:`contrast_dr_scores` instead.
    """

    target = _as_probability_matrix(policy, name="policy")
    reference = _as_probability_matrix(marginal_policy, name="marginal_policy")
    prediction = _as_matrix(outcome_prediction, name="outcome_prediction")
    marginal_prediction = _as_matrix(
        marginal_outcome_prediction,
        name="marginal_outcome_prediction",
    )
    if not (
        target.shape
        == reference.shape
        == prediction.shape
        == marginal_prediction.shape
    ):
        raise ValueError("policy and outcome nuisance arrays must align")
    contrast = target - reference
    centered_prediction = prediction - marginal_prediction
    n_rows, n_actions = target.shape
    actions = np.asarray(logged_actions, dtype=np.int64)
    outcomes = np.asarray(rewards, dtype=np.float64)
    pscore = np.asarray(logging_pscore, dtype=np.float64)
    if actions.shape != (n_rows,) or outcomes.shape != (n_rows,) or pscore.shape != (n_rows,):
        raise ValueError("logged arrays must have one entry per policy row")
    if np.any(actions < 0) or np.any(actions >= n_actions):
        raise ValueError("logged_actions contains an invalid action")
    if not np.isfinite(outcomes).all() or not np.isfinite(pscore).all() or np.any(pscore <= 0.0):
        raise ValueError("rewards must be finite and logging_pscore must be positive")
    row = np.arange(n_rows)
    direct = np.sum(contrast * centered_prediction, axis=1)
    correction = (
        contrast[row, actions]
        / pscore
        * (outcomes - prediction[row, actions])
    )
    return direct + correction


def support_adapted_matching_dr_scores(
    policy: np.ndarray,
    marginal_policy: np.ndarray,
    structural_support: np.ndarray,
    logged_actions: np.ndarray,
    rewards: np.ndarray,
    logging_pscore: np.ndarray,
    outcome_prediction: np.ndarray,
    supported_marginal_outcome_prediction: np.ndarray,
) -> np.ndarray:
    """Orthogonal scores for the supported matching-value component.

    Let ``S(a, c, u)`` be the design-known structural-support indicator and
    ``p(a | c) = E[policy(a | c, U) | C=c]``.  This function estimates

    ``E[sum_a S_a * (policy_a - p_a) * q_a]``

    with the centered nuisance

    ``R_a(c) = E[S_a * q_a(c, U) | C=c]``.

    The supported component equals the full matching value when the policy
    contrast is zero off support.  Callers must establish that identification
    condition from the logging/candidate design, for example with
    :func:`contrast_support_certificate`; the condition cannot be inferred
    from finite zero action counts.  If it fails, use
    :func:`contrast_identification_interval` for the full matching value.

    Outcome predictions may be non-finite off structural support because those
    cells are never read.  The marginal policy and supported marginal outcome
    nuisance may be estimated on training folds and need not satisfy the
    population cancellation equality exactly in finite samples.
    """

    target = _as_probability_matrix(policy, name="policy")
    reference = _as_probability_matrix(marginal_policy, name="marginal_policy")
    prediction = _as_matrix(outcome_prediction, name="outcome_prediction")
    marginal_supported_prediction = _as_matrix(
        supported_marginal_outcome_prediction,
        name="supported_marginal_outcome_prediction",
    )
    if not (
        target.shape
        == reference.shape
        == prediction.shape
        == marginal_supported_prediction.shape
    ):
        raise ValueError("policy and outcome nuisance arrays must align")
    support = np.asarray(structural_support, dtype=bool)
    if support.shape != target.shape:
        raise ValueError("structural_support must align with policy")
    if not np.isfinite(prediction[support]).all():
        raise ValueError("outcome_prediction must be finite on structural support")
    if not np.isfinite(marginal_supported_prediction).all():
        raise ValueError(
            "supported_marginal_outcome_prediction must contain only finite values"
        )

    n_rows, n_actions = target.shape
    actions = np.asarray(logged_actions, dtype=np.int64)
    outcomes = np.asarray(rewards, dtype=np.float64)
    pscore = np.asarray(logging_pscore, dtype=np.float64)
    if actions.shape != (n_rows,) or outcomes.shape != (n_rows,) or pscore.shape != (n_rows,):
        raise ValueError("logged arrays must have one entry per policy row")
    if np.any(actions < 0) or np.any(actions >= n_actions):
        raise ValueError("logged_actions contains an invalid action")
    if not np.isfinite(outcomes).all() or not np.isfinite(pscore).all() or np.any(pscore <= 0.0):
        raise ValueError("rewards must be finite and logging_pscore must be positive")
    row = np.arange(n_rows)
    if not support[row, actions].all():
        raise ValueError("a logged action lies outside declared structural support")

    contrast = target - reference
    supported_prediction = np.where(support, prediction, 0.0)
    direct = np.sum(
        contrast * (supported_prediction - marginal_supported_prediction),
        axis=1,
    )
    correction = (
        contrast[row, actions]
        / pscore
        * (outcomes - prediction[row, actions])
    )
    return direct + correction


def equality_restricted_projection_coefficients(
    policy: np.ndarray,
    certified_marginal_policy: np.ndarray,
    structural_support: np.ndarray,
    hole_action_indicator: np.ndarray,
    outcome_prediction: np.ndarray,
    supported_marginal_outcome_prediction: np.ndarray,
    shared_strata: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    contrast_tolerance: float = 1e-12,
    rcond: float = 1e-12,
) -> np.ndarray:
    """Fit the conditional projection for the contrast-overlap efficiency bound.

    ``hole_action_indicator`` marks actions that have positive structural-hole
    probability in a shared-context stratum.  For each such action, the
    equality-restricted model fixes its marginal policy probability to the
    common target-policy value on the holes.  ``certified_marginal_policy``
    must encode that design certificate exactly; an ordinary noisy marginal
    policy estimate is not sufficient for the restricted efficiency claim.

    The returned coefficient is the conditional least-squares coefficient

    ``pinv(E[D D.T | C]) E[D F | C]``,

    where ``D = J * (policy - certified_marginal_policy)`` and ``F`` is the
    direct component of the support-adapted influence function.  A stable
    population-rank condition is still required for asymptotic use.
    """

    if contrast_tolerance < 0.0:
        raise ValueError("contrast_tolerance must be non-negative")
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be finite and positive")
    target = _as_probability_matrix(policy, name="policy")
    reference = _as_probability_matrix(
        certified_marginal_policy,
        name="certified_marginal_policy",
    )
    prediction = _as_matrix(outcome_prediction, name="outcome_prediction")
    marginal_supported_prediction = _as_matrix(
        supported_marginal_outcome_prediction,
        name="supported_marginal_outcome_prediction",
    )
    if not (
        target.shape
        == reference.shape
        == prediction.shape
        == marginal_supported_prediction.shape
    ):
        raise ValueError("policy and outcome nuisance arrays must align")
    support = np.asarray(structural_support, dtype=bool)
    hole_actions = np.asarray(hole_action_indicator, dtype=bool)
    if support.shape != target.shape or hole_actions.shape != target.shape:
        raise ValueError("support and hole-action indicators must align with policy")
    if np.any((~support) & (~hole_actions)):
        raise ValueError("every structural hole must mark its action as constrained")
    contrast = target - reference
    if np.any(np.abs(contrast[~support]) > contrast_tolerance):
        raise ValueError(
            "certified_marginal_policy does not cancel the contrast at every hole"
        )
    if not np.isfinite(prediction[support]).all():
        raise ValueError("outcome_prediction must be finite on structural support")
    if not np.isfinite(marginal_supported_prediction).all():
        raise ValueError(
            "supported_marginal_outcome_prediction must contain only finite values"
        )

    n_rows, n_actions = target.shape
    codes = _stratum_codes(shared_strata, n_rows)
    weights = _normalized_weights(sample_weight, n_rows)
    supported_prediction = np.where(support, prediction, 0.0)
    direct_signal = np.sum(
        contrast * (supported_prediction - marginal_supported_prediction),
        axis=1,
    )
    constrained_contrast = hole_actions * contrast
    coefficients = np.zeros_like(target)
    for code in range(int(codes.max()) + 1):
        rows = np.flatnonzero(codes == code)
        if not np.all(hole_actions[rows] == hole_actions[rows[0]]):
            raise ValueError("hole_action_indicator must be constant inside a stratum")
        if not np.allclose(
            reference[rows],
            reference[rows[0]],
            atol=contrast_tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                "certified_marginal_policy must be constant inside a stratum"
            )
        stratum_weights = weights[rows]
        stratum_weights = stratum_weights / stratum_weights.sum()
        design = constrained_contrast[rows]
        gram = (design * stratum_weights[:, None]).T @ design
        cross_moment = design.T @ (stratum_weights * direct_signal[rows])
        coefficient = np.linalg.pinv(gram, rcond=rcond) @ cross_moment
        coefficient[~hole_actions[rows[0]]] = 0.0
        coefficients[rows] = coefficient
    return coefficients


def equality_restricted_matching_dr_scores(
    policy: np.ndarray,
    certified_marginal_policy: np.ndarray,
    structural_support: np.ndarray,
    hole_action_indicator: np.ndarray,
    projection_coefficient: np.ndarray,
    logged_actions: np.ndarray,
    rewards: np.ndarray,
    logging_pscore: np.ndarray,
    outcome_prediction: np.ndarray,
    supported_marginal_outcome_prediction: np.ndarray,
    *,
    contrast_tolerance: float = 1e-12,
) -> np.ndarray:
    """Efficient scores in the equality-restricted contrast-overlap model.

    This subtracts the conditional projection of the support-adapted influence
    function onto user-distribution directions excluded by exact contrast
    overlap.  The marginal policy must be a design-certified equality value on
    every action with structural holes.  If only a noisy unconstrained estimate
    is available, use :func:`support_adapted_matching_dr_scores` instead.
    """

    if contrast_tolerance < 0.0:
        raise ValueError("contrast_tolerance must be non-negative")
    target = _as_probability_matrix(policy, name="policy")
    reference = _as_probability_matrix(
        certified_marginal_policy,
        name="certified_marginal_policy",
    )
    support = np.asarray(structural_support, dtype=bool)
    hole_actions = np.asarray(hole_action_indicator, dtype=bool)
    coefficient = _as_matrix(
        projection_coefficient,
        name="projection_coefficient",
    )
    if not (
        target.shape
        == reference.shape
        == support.shape
        == hole_actions.shape
        == coefficient.shape
    ):
        raise ValueError("restricted-score arrays must align with policy")
    if not np.isfinite(coefficient).all():
        raise ValueError("projection_coefficient must contain only finite values")
    if np.any((~support) & (~hole_actions)):
        raise ValueError("every structural hole must mark its action as constrained")
    contrast = target - reference
    if np.any(np.abs(contrast[~support]) > contrast_tolerance):
        raise ValueError(
            "certified_marginal_policy does not cancel the contrast at every hole"
        )
    base_scores = support_adapted_matching_dr_scores(
        target,
        reference,
        support,
        logged_actions,
        rewards,
        logging_pscore,
        outcome_prediction,
        supported_marginal_outcome_prediction,
    )
    constrained_contrast = hole_actions * contrast
    projection = np.sum(coefficient * constrained_contrast, axis=1)
    return base_scores - projection


def two_sided_normal_power(
    effect: float,
    standard_error: float,
    *,
    alpha: float = 0.05,
) -> float:
    """Power of a two-sided normal test at a declared standard error."""

    if standard_error < 0.0 or not np.isfinite(standard_error):
        raise ValueError("standard_error must be finite and non-negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if standard_error == 0.0:
        return 1.0 if effect != 0.0 else alpha
    standardized = abs(float(effect)) / standard_error
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    return float(
        norm.cdf(-critical - standardized)
        + 1.0
        - norm.cdf(critical - standardized)
    )


def normal_minimum_detectable_effect(
    standard_error: float,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> float:
    """Smallest absolute effect attaining the target two-sided normal power."""

    if standard_error < 0.0 or not np.isfinite(standard_error):
        raise ValueError("standard_error must be finite and non-negative")
    if not 0.0 < alpha < 1.0 or not alpha < target_power < 1.0:
        raise ValueError("require 0 < alpha < target_power < 1")
    if standard_error == 0.0:
        return 0.0

    def objective(effect: float) -> float:
        return (
            two_sided_normal_power(effect, standard_error, alpha=alpha)
            - target_power
        )

    upper = standard_error
    while objective(upper) < 0.0:
        upper *= 2.0
    return float(brentq(objective, 0.0, upper))


def estimate_from_scores(
    scores: np.ndarray,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> ContrastEstimate:
    """Summarize a contrast estimate, confidence interval, and resolution."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite vector with at least two rows")
    estimate = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    return ContrastEstimate(
        value=estimate,
        standard_error=standard_error,
        ci_lower=estimate - critical * standard_error,
        ci_upper=estimate + critical * standard_error,
        alpha=float(alpha),
        target_power=float(target_power),
        minimum_detectable_effect=normal_minimum_detectable_effect(
            standard_error,
            alpha=alpha,
            target_power=target_power,
        ),
        n_rows=int(values.size),
    )


__all__ = [
    "ContrastEstimate",
    "ContrastSupportCertificate",
    "IdentificationInterval",
    "centered_matching_dr_scores",
    "contrast_dr_scores",
    "contrast_identification_interval",
    "contrast_support_certificate",
    "estimate_from_scores",
    "equality_restricted_matching_dr_scores",
    "equality_restricted_projection_coefficients",
    "marginalize_policy",
    "matching_value",
    "normal_minimum_detectable_effect",
    "separate_value_difference_interval",
    "stratified_mean",
    "support_adapted_matching_dr_scores",
    "two_sided_normal_power",
]
