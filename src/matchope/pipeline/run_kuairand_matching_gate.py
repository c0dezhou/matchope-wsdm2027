#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate category-level matching value on KuaiRand random interventions.

The random-intervention log is used only for evaluation.  A frozen warm-start
policy is learned from the preceding standard-log window, then evaluated on a
compound action: choose a predeclared video category and retain the logger's
within-category item sampler.  This avoids claiming that the released rows
contain per-event item eligibility snapshots.

The primary estimand averages over random-insertion opportunities in the
released 17-day window.  Date and an outcome-independent time-of-day bin form
the shared context, user history is the personalization variable, and
uncertainty is clustered by user.  Day-cluster uncertainty is reported only
as a temporal-transport diagnostic.
"""

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import t

from matchope.eval.matching_value import (
    centered_matching_dr_scores,
    estimate_from_scores,
    normal_minimum_detectable_effect,
)


OFFICIAL_DATASET_URL = "https://kuairand.com/"
OFFICIAL_CATEGORY_URL = (
    "https://zenodo.org/records/18159199/files/kuairand_video_categories.csv"
)
CATEGORY_SOURCE = "official KuaiRand first_level_category_id/name"
SUPPORTED_OUTCOMES = ("long_view", "is_click")
PER_1000 = 1_000.0


@dataclass(frozen=True)
class CategoryMapping:
    video_to_action: dict[int, int]
    labels: tuple[str, ...]
    item_counts: np.ndarray
    base_probability: np.ndarray
    raw_category_count: int
    merged_rare_category_count: int
    source: str


@dataclass(frozen=True)
class FrozenHistoryPolicy:
    user_ids: np.ndarray
    signal: np.ndarray
    global_rate: np.ndarray
    selected_smoothing: float
    validation_rows: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class CrossFitScores:
    centered_scores: np.ndarray
    separate_dr_scores: np.ndarray
    ipw_scores: np.ndarray
    logging_pscore: np.ndarray
    fold_ids: np.ndarray
    mean_policy_to_reference_tv: float
    maximum_policy_to_reference_tv: float
    maximum_logged_contrast_weight: float
    minimum_logging_pscore: float
    maximum_logging_pscore: float
    maximum_fold_marginal_shift: float


def _parse_numbers(raw: str, *, cast: type = float) -> tuple:
    values = tuple(cast(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        raise ValueError("numeric grid must not be empty")
    return values


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - provenance, not cryptographic security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_first_level_category_mapping(
    video_categories: pd.DataFrame,
    *,
    minimum_items_per_category: int,
) -> CategoryMapping:
    """Build a metadata-only, outcome-independent category mapping."""

    if minimum_items_per_category < 1:
        raise ValueError("minimum_items_per_category must be positive")
    required = {
        "final_video_id",
        "first_level_category_id",
        "first_level_category_name",
    }
    missing = sorted(required - set(video_categories.columns))
    if missing:
        raise ValueError(f"category metadata is missing columns: {missing}")
    frame = video_categories[list(required)].copy()
    if frame["final_video_id"].duplicated().any():
        raise ValueError("category metadata has duplicate final_video_id rows")
    known = frame["first_level_category_id"].notna()
    raw = pd.Series("__UNKNOWN__", index=frame.index, dtype="string")
    raw.loc[known] = (
        frame.loc[known, "first_level_category_id"].astype(int).astype(str)
        + ":"
        + frame.loc[known, "first_level_category_name"]
        .astype("string")
        .fillna("UNKNOWN")
        .str.strip()
        .replace("", "UNKNOWN")
    )
    raw_counts = raw.value_counts()
    rare = set(raw_counts[raw_counts < minimum_items_per_category].index)
    merged = raw.where(~raw.isin(rare), "__OTHER_RARE__")
    labels = tuple(sorted(str(value) for value in merged.unique()))
    label_to_action = {label: index for index, label in enumerate(labels)}
    action = merged.map(label_to_action).to_numpy(dtype=np.int64)
    item_counts = np.bincount(action, minlength=len(labels)).astype(np.int64)
    if np.any(item_counts < 1):
        raise ValueError("category mapping produced an empty category")
    return CategoryMapping(
        video_to_action={
            int(video_id): int(category)
            for video_id, category in zip(
                frame["final_video_id"].to_numpy(), action, strict=True
            )
        },
        labels=labels,
        item_counts=item_counts,
        base_probability=item_counts / item_counts.sum(),
        raw_category_count=int(raw_counts.size),
        merged_rare_category_count=int(len(rare)),
        source=CATEGORY_SOURCE,
    )


def _load_candidate_categories(
    path: Path,
    candidate_video_ids: np.ndarray,
) -> pd.DataFrame:
    """Read only enough of the sorted official file to cover the candidate pool."""

    required = [
        "final_video_id",
        "first_level_category_id",
        "first_level_category_name",
    ]
    candidate = set(int(value) for value in candidate_video_ids)
    selected: list[pd.DataFrame] = []
    covered: set[int] = set()
    for chunk in pd.read_csv(path, usecols=required, chunksize=100_000):
        rows = chunk.loc[chunk["final_video_id"].isin(candidate)]
        if not rows.empty:
            selected.append(rows)
            covered.update(int(value) for value in rows["final_video_id"])
        if covered == candidate:
            break
    missing = sorted(candidate - covered)
    if missing:
        raise ValueError(
            f"official category metadata misses {len(missing)} candidate videos: "
            f"{missing[:10]}"
        )
    return pd.concat(selected, ignore_index=True)


def _attach_actions(frame: pd.DataFrame, mapping: CategoryMapping) -> pd.DataFrame:
    attached = frame.copy()
    attached["action"] = attached["video_id"].map(mapping.video_to_action)
    if attached["action"].isna().any():
        missing = int(attached["action"].isna().sum())
        raise ValueError(f"{missing} log rows refer to videos without categories")
    attached["action"] = attached["action"].astype(np.int16)
    return attached


def load_kuairand_category_data(
    data_dir: Path,
    *,
    category_metadata_path: Path,
    tab: int,
    minimum_history_events: int,
    minimum_items_per_category: int,
    shared_context_hour_width: int,
) -> tuple[pd.DataFrame, pd.DataFrame, CategoryMapping, dict[str, object]]:
    """Load pre-period training rows and held-out randomized evaluation rows."""

    if minimum_history_events < 1:
        raise ValueError("minimum_history_events must be positive")
    if shared_context_hour_width < 1 or 24 % shared_context_hour_width != 0:
        raise ValueError("shared_context_hour_width must be a positive divisor of 24")
    random_path = data_dir / "log_random_4_22_to_5_08_pure.csv"
    standard_path = data_dir / "log_standard_4_08_to_4_21_pure.csv"
    for path in (random_path, standard_path, category_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    columns = [
        "user_id",
        "video_id",
        "date",
        "hourmin",
        "tab",
        "is_rand",
        *SUPPORTED_OUTCOMES,
    ]
    random_all = pd.read_csv(
        random_path,
        usecols=columns,
        dtype={
            "user_id": np.int32,
            "video_id": np.int32,
            "date": np.int32,
            "hourmin": np.int16,
            "tab": np.int8,
            "is_rand": np.int8,
            "long_view": np.int8,
            "is_click": np.int8,
        },
    )
    if not bool(random_all["is_rand"].eq(1).all()):
        raise ValueError("random-intervention file contains non-random rows")
    candidate_video_ids = np.sort(random_all["video_id"].unique())
    categories = _load_candidate_categories(
        category_metadata_path,
        candidate_video_ids,
    )
    mapping = build_first_level_category_mapping(
        categories,
        minimum_items_per_category=minimum_items_per_category,
    )
    standard = pd.read_csv(
        standard_path,
        usecols=columns,
        dtype={
            "user_id": np.int32,
            "video_id": np.int32,
            "date": np.int32,
            "hourmin": np.int16,
            "tab": np.int8,
            "is_rand": np.int8,
            "long_view": np.int8,
            "is_click": np.int8,
        },
    )
    random = random_all.loc[random_all["tab"] == tab].copy()
    standard = standard.loc[standard["tab"] == tab].copy()
    history_counts = standard.groupby("user_id").size()
    warm_users = history_counts.index[
        history_counts >= minimum_history_events
    ].to_numpy(dtype=np.int64)
    warm_set = set(int(value) for value in warm_users)
    standard = standard.loc[standard["user_id"].isin(warm_set)].copy()
    random = random.loc[random["user_id"].isin(warm_set)].copy()
    if standard.empty or random.empty:
        raise ValueError("warm-start and tab filters produced an empty dataset")
    standard = _attach_actions(standard, mapping)
    random = _attach_actions(random, mapping)
    user_ids = np.sort(random["user_id"].unique())
    user_to_index = {int(user): index for index, user in enumerate(user_ids)}
    standard = standard.loc[standard["user_id"].isin(user_to_index)].copy()
    for frame in (standard, random):
        frame["user_index"] = (
            frame["user_id"].map(user_to_index).astype(np.int32)
        )
    dates = np.sort(random["date"].unique())
    random["hour"] = random["hourmin"].astype(np.int64) // 100
    if np.any((random["hour"] < 0) | (random["hour"] > 23)):
        raise ValueError("hourmin contains an hour outside [0, 23]")
    random["hour_bin"] = random["hour"] // shared_context_hour_width
    context = pd.MultiIndex.from_frame(random[["date", "hour_bin"]])
    context_codes, context_labels = pd.factorize(context, sort=True)
    random["context_index"] = context_codes.astype(np.int16)

    item_counts = random_all["video_id"].value_counts()
    expected = len(random_all) / int(item_counts.size)
    first_date = random_all.groupby("video_id")["date"].min()
    last_date = random_all.groupby("video_id")["date"].max()
    item_uniformity = {
        "random_rows_all_tabs": int(len(random_all)),
        "random_users_all_tabs": int(random_all["user_id"].nunique()),
        "random_items": int(item_counts.size),
        "expected_rows_per_item_under_fixed_uniform_pool": float(expected),
        "minimum_item_rows": int(item_counts.min()),
        "median_item_rows": float(item_counts.median()),
        "maximum_item_rows": int(item_counts.max()),
        "item_count_pearson_chisquare_per_df": float(
            np.sum(np.square(item_counts.to_numpy() - expected) / expected)
            / (item_counts.size - 1)
        ),
        "items_present_on_first_day": int((first_date == random_all["date"].min()).sum()),
        "items_present_on_last_day": int((last_date == random_all["date"].max()).sum()),
        "item_level_propensity_conclusion": (
            "released frequencies are inconsistent with blindly assigning 1/7583 "
            "to every row; exact-item OPE is not used"
        ),
    }

    category_by_context = pd.crosstab(
        [random["date"], random["hour_bin"]], random["action"]
    ).reindex(
        columns=np.arange(len(mapping.labels)),
        fill_value=0,
    )
    empirical = category_by_context.div(category_by_context.sum(axis=1), axis=0)
    metadata_probability = mapping.base_probability
    daily_tv = 0.5 * np.abs(
        empirical.to_numpy() - metadata_probability[None, :]
    ).sum(axis=1)
    audit = {
        "paths": {
            "random": str(random_path),
            "standard_preperiod": str(standard_path),
            "category_metadata": str(category_metadata_path),
        },
        "md5": {
            "random": _md5(random_path),
            "standard_preperiod": _md5(standard_path),
            "category_metadata": _md5(category_metadata_path),
        },
        "tab": int(tab),
        "minimum_history_events": int(minimum_history_events),
        "evaluation_rows": int(len(random)),
        "evaluation_users": int(random["user_id"].nunique()),
        "evaluation_dates": int(len(dates)),
        "evaluation_shared_contexts": int(len(context_labels)),
        "shared_context": f"date x {shared_context_hour_width}-hour bin",
        "preperiod_rows": int(len(standard)),
        "preperiod_users": int(standard["user_id"].nunique()),
        "category_count": int(len(mapping.labels)),
        "category_source": CATEGORY_SOURCE,
        "category_metadata_official_url": OFFICIAL_CATEGORY_URL,
        "category_metadata_file_size_bytes": int(
            category_metadata_path.stat().st_size
        ),
        "category_labels": list(mapping.labels),
        "category_item_counts": mapping.item_counts.tolist(),
        "zero_context_category_cells": int(
            (category_by_context.to_numpy() == 0).sum()
        ),
        "minimum_context_category_rows": int(
            category_by_context.to_numpy().min()
        ),
        "maximum_context_category_tv_from_metadata_mass": float(daily_tv.max()),
        "item_uniformity": item_uniformity,
    }
    return standard.reset_index(drop=True), random.reset_index(drop=True), mapping, audit


def audit_contrast_overlap(
    policy_by_user: np.ndarray,
    *,
    event_user_index: np.ndarray,
    event_context_index: np.ndarray,
    structural_support: np.ndarray,
    tolerance: float = 1e-12,
) -> dict[str, float | int | bool]:
    """Audit signed contrast overlap against design-known support.

    The population marginal reference is formed within each shared context
    using the evaluation event weights. Structural support must come from the
    randomization/candidate design; this function never infers it from zero
    sample counts.
    """

    policy = np.asarray(policy_by_user, dtype=np.float64)
    users = np.asarray(event_user_index, dtype=np.int64)
    contexts = np.asarray(event_context_index, dtype=np.int64)
    support = np.asarray(structural_support, dtype=bool)
    if policy.ndim != 2 or not np.isfinite(policy).all():
        raise ValueError("policy_by_user must be a finite matrix")
    if np.any(policy < 0.0) or not np.allclose(
        policy.sum(axis=1), 1.0, atol=1e-10
    ):
        raise ValueError("policy_by_user rows must be probabilities")
    if users.shape != contexts.shape or users.ndim != 1 or users.size == 0:
        raise ValueError("event user and context indices must be aligned vectors")
    if users.min() < 0 or users.max() >= policy.shape[0]:
        raise ValueError("event user index lies outside policy_by_user")
    if contexts.min() < 0 or contexts.max() >= support.shape[0]:
        raise ValueError("event context index lies outside structural_support")
    if support.shape[1] != policy.shape[1]:
        raise ValueError("support actions must align with policy actions")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")

    maximum_unsupported_contrast = 0.0
    target_unsupported_mass_sum = 0.0
    reference_unsupported_mass_sum = 0.0
    absolute_unsupported_contrast_sum = 0.0
    for context in np.unique(contexts):
        row_mask = contexts == context
        context_policy = policy[users[row_mask]]
        reference = context_policy.mean(axis=0)
        unsupported = ~support[context]
        if not bool(np.any(unsupported)):
            continue
        unsupported_contrast = (
            context_policy[:, unsupported] - reference[unsupported][None, :]
        )
        maximum_unsupported_contrast = max(
            maximum_unsupported_contrast,
            float(np.max(np.abs(unsupported_contrast))),
        )
        target_unsupported_mass_sum += float(
            context_policy[:, unsupported].sum()
        )
        reference_unsupported_mass_sum += float(
            row_mask.sum() * reference[unsupported].sum()
        )
        absolute_unsupported_contrast_sum += float(
            np.abs(unsupported_contrast).sum()
        )

    rows = int(users.size)
    return {
        "shared_contexts": int(support.shape[0]),
        "actions": int(support.shape[1]),
        "structural_context_action_cells": int(support.size),
        "unsupported_context_action_cells": int((~support).sum()),
        "event_weighted_target_unsupported_mass": (
            target_unsupported_mass_sum / rows
        ),
        "event_weighted_reference_unsupported_mass": (
            reference_unsupported_mass_sum / rows
        ),
        "event_weighted_absolute_unsupported_contrast": (
            absolute_unsupported_contrast_sum / rows
        ),
        "maximum_absolute_unsupported_contrast": maximum_unsupported_contrast,
        "contrast_overlap_tolerance": float(tolerance),
        "contrast_overlap_holds": bool(
            maximum_unsupported_contrast <= tolerance
        ),
    }


def _history_signal(
    frame: pd.DataFrame,
    *,
    outcome: str,
    n_users: int,
    n_actions: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    if smoothing <= 0.0:
        raise ValueError("history smoothing must be positive")
    user = frame["user_index"].to_numpy(dtype=np.int64)
    action = frame["action"].to_numpy(dtype=np.int64)
    reward = frame[outcome].to_numpy(dtype=np.float64)
    flat = user * n_actions + action
    counts = np.bincount(flat, minlength=n_users * n_actions).reshape(
        n_users, n_actions
    )
    sums = np.bincount(
        flat,
        weights=reward,
        minlength=n_users * n_actions,
    ).reshape(n_users, n_actions)
    action_counts = np.bincount(action, minlength=n_actions).astype(np.float64)
    action_sums = np.bincount(
        action, weights=reward, minlength=n_actions
    ).astype(np.float64)
    global_rate = (action_sums + 0.5) / (action_counts + 1.0)
    signal = (sums + smoothing * global_rate[None, :]) / (
        counts + smoothing
    )
    return signal, global_rate


def fit_frozen_history_policy(
    standard: pd.DataFrame,
    *,
    user_ids: np.ndarray,
    outcome: str,
    n_actions: int,
    smoothing_grid: Sequence[float],
    validation_days: int,
) -> FrozenHistoryPolicy:
    """Select smoothing only on the pre-period temporal validation tail."""

    if outcome not in SUPPORTED_OUTCOMES:
        raise ValueError(f"unsupported outcome: {outcome}")
    unique_dates = np.sort(standard["date"].unique())
    if validation_days < 1 or validation_days >= unique_dates.size:
        raise ValueError("validation_days must leave at least one training date")
    cutoff = unique_dates[-validation_days]
    train = standard.loc[standard["date"] < cutoff]
    validation = standard.loc[standard["date"] >= cutoff]
    if train.empty or validation.empty:
        raise ValueError("pre-period temporal split is empty")
    validation_action = validation["action"].to_numpy(dtype=np.int64)
    validation_user = validation["user_index"].to_numpy(dtype=np.int64)
    validation_reward = validation[outcome].to_numpy(dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    for smoothing in smoothing_grid:
        signal, _ = _history_signal(
            train,
            outcome=outcome,
            n_users=len(user_ids),
            n_actions=n_actions,
            smoothing=float(smoothing),
        )
        prediction = np.clip(
            signal[validation_user, validation_action], 1e-6, 1.0 - 1e-6
        )
        log_loss = float(
            -np.mean(
                validation_reward * np.log(prediction)
                + (1.0 - validation_reward) * np.log(1.0 - prediction)
            )
        )
        rows.append(
            {
                "smoothing": float(smoothing),
                "validation_log_loss": log_loss,
                "validation_rows": int(len(validation)),
                "cutoff_date": int(cutoff),
            }
        )
    selected = min(rows, key=lambda row: (row["validation_log_loss"], row["smoothing"]))
    signal, global_rate = _history_signal(
        standard,
        outcome=outcome,
        n_users=len(user_ids),
        n_actions=n_actions,
        smoothing=float(selected["smoothing"]),
    )
    return FrozenHistoryPolicy(
        user_ids=np.asarray(user_ids, dtype=np.int64),
        signal=signal,
        global_rate=global_rate,
        selected_smoothing=float(selected["smoothing"]),
        validation_rows=tuple(rows),
    )


def tilted_category_policy(
    signal: np.ndarray,
    global_rate: np.ndarray,
    base_probability: np.ndarray,
    *,
    inverse_temperature: float,
) -> np.ndarray:
    """Exponential tilt of a metadata-only category-mass base policy."""

    values = np.asarray(signal, dtype=np.float64)
    center = np.asarray(global_rate, dtype=np.float64)
    base = np.asarray(base_probability, dtype=np.float64)
    if values.ndim != 2 or center.shape != (values.shape[1],):
        raise ValueError("signal and global_rate do not align")
    if base.shape != center.shape or np.any(base <= 0.0):
        raise ValueError("base_probability must be positive and aligned")
    if inverse_temperature < 0.0 or not np.isfinite(inverse_temperature):
        raise ValueError("inverse_temperature must be finite and non-negative")
    logits = np.log(base)[None, :] + inverse_temperature * (
        values - center[None, :]
    )
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability


def calibrate_policy_to_tv(
    signal: np.ndarray,
    global_rate: np.ndarray,
    base_probability: np.ndarray,
    *,
    target_tv: float,
) -> tuple[float, np.ndarray, float]:
    """Choose policy strength without reading randomized evaluation outcomes."""

    if not 0.0 < target_tv < 1.0:
        raise ValueError("target_tv must lie in (0, 1)")

    def evaluate(beta: float) -> tuple[float, np.ndarray]:
        policy = tilted_category_policy(
            signal,
            global_rate,
            base_probability,
            inverse_temperature=beta,
        )
        reference = policy.mean(axis=0)
        tv = float(np.mean(0.5 * np.abs(policy - reference).sum(axis=1)))
        return tv, policy

    upper = 1.0
    upper_tv, _ = evaluate(upper)
    while upper_tv < target_tv and upper < 1e6:
        upper *= 2.0
        upper_tv, _ = evaluate(upper)
    if upper_tv < target_tv:
        raise ValueError(
            f"target TV {target_tv} is unattainable; maximum reached {upper_tv}"
        )
    beta = float(brentq(lambda value: evaluate(value)[0] - target_tv, 0.0, upper))
    achieved, policy = evaluate(beta)
    return beta, policy, achieved


def assign_user_folds(
    n_users: int,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    if folds < 2 or folds > n_users:
        raise ValueError("folds must be between 2 and n_users")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_users)
    assignment = np.empty(n_users, dtype=np.int16)
    assignment[order] = np.arange(n_users) % folds
    return assignment


def _context_action_counts(
    contexts: np.ndarray,
    actions: np.ndarray,
    *,
    n_contexts: int,
    n_actions: int,
) -> np.ndarray:
    flat = contexts * n_actions + actions
    return np.bincount(flat, minlength=n_contexts * n_actions).reshape(
        n_contexts, n_actions
    )


def _fit_outcome_nuisance(
    *,
    user_indices: np.ndarray,
    actions: np.ndarray,
    contexts: np.ndarray,
    rewards: np.ndarray,
    signal: np.ndarray,
    n_contexts: int,
    n_actions: int,
    smoothing: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a shared-context intercept plus category history calibration."""

    if smoothing <= 0.0 or ridge <= 0.0:
        raise ValueError("outcome nuisance smoothing and ridge must be positive")
    observed_signal = signal[user_indices, actions]
    count = np.bincount(actions, minlength=n_actions).astype(np.float64)
    sum_x = np.bincount(
        actions, weights=observed_signal, minlength=n_actions
    ).astype(np.float64)
    sum_y = np.bincount(
        actions, weights=rewards, minlength=n_actions
    ).astype(np.float64)
    sum_xx = np.bincount(
        actions, weights=np.square(observed_signal), minlength=n_actions
    ).astype(np.float64)
    sum_xy = np.bincount(
        actions,
        weights=observed_signal * rewards,
        minlength=n_actions,
    ).astype(np.float64)
    x_mean = np.divide(sum_x, count, out=np.zeros_like(sum_x), where=count > 0)
    y_mean = np.divide(sum_y, count, out=np.zeros_like(sum_y), where=count > 0)
    numerator = sum_xy - count * x_mean * y_mean
    denominator = sum_xx - count * np.square(x_mean) + ridge
    slope = np.clip(numerator / denominator, -5.0, 5.0)
    residual = rewards - slope[actions] * (observed_signal - x_mean[actions])
    context_action = contexts * n_actions + actions
    cell_count = np.bincount(
        context_action, minlength=n_contexts * n_actions
    ).reshape(n_contexts, n_actions)
    cell_sum = np.bincount(
        context_action,
        weights=residual,
        minlength=n_contexts * n_actions,
    ).reshape(n_contexts, n_actions)
    intercept = (cell_sum + smoothing * y_mean[None, :]) / (
        cell_count + smoothing
    )
    return intercept, slope, x_mean


def cross_fitted_scores(
    evaluation: pd.DataFrame,
    *,
    policy_by_user: np.ndarray,
    history_signal: np.ndarray,
    base_probability: np.ndarray,
    user_fold: np.ndarray,
    propensity_smoothing: float,
    fixed_logging_probability: np.ndarray | None = None,
    outcome_smoothing: float,
    outcome_ridge: float,
    outcome: str,
) -> CrossFitScores:
    """Cross-fit logger, policy-marginal, and outcome nuisances by user."""

    if propensity_smoothing <= 0.0:
        raise ValueError("propensity_smoothing must be positive")
    fixed_logger: np.ndarray | None = None
    if fixed_logging_probability is not None:
        fixed_logger = np.asarray(fixed_logging_probability, dtype=np.float64)
        if (
            fixed_logger.shape != np.asarray(base_probability).shape
            or np.any(fixed_logger <= 0.0)
            or not np.isclose(fixed_logger.sum(), 1.0)
        ):
            raise ValueError(
                "fixed_logging_probability must be positive and sum to one"
            )
    user = evaluation["user_index"].to_numpy(dtype=np.int64)
    action = evaluation["action"].to_numpy(dtype=np.int64)
    context = evaluation["context_index"].to_numpy(dtype=np.int64)
    reward = evaluation[outcome].to_numpy(dtype=np.float64)
    row_fold = user_fold[user]
    n_rows = len(evaluation)
    n_contexts = int(context.max()) + 1
    n_actions = policy_by_user.shape[1]
    centered = np.empty(n_rows, dtype=np.float64)
    separate_dr = np.empty(n_rows, dtype=np.float64)
    ipw = np.empty(n_rows, dtype=np.float64)
    pscore_all = np.empty(n_rows, dtype=np.float64)
    tv_all = np.empty(n_rows, dtype=np.float64)
    logged_weight_all = np.empty(n_rows, dtype=np.float64)
    maximum_shift = 0.0
    for fold in np.unique(row_fold):
        held_out = row_fold == fold
        training = ~held_out
        train_user = user[training]
        train_action = action[training]
        train_context = context[training]
        train_reward = reward[training]
        held_user = user[held_out]
        held_action = action[held_out]
        held_context = context[held_out]

        if fixed_logger is None:
            logger_count = _context_action_counts(
                train_context,
                train_action,
                n_contexts=n_contexts,
                n_actions=n_actions,
            ).astype(np.float64)
            logger_probability = (
                logger_count + propensity_smoothing * base_probability[None, :]
            ) / (logger_count.sum(axis=1, keepdims=True) + propensity_smoothing)
        else:
            logger_probability = np.broadcast_to(
                fixed_logger[None, :], (n_contexts, n_actions)
            )
        pscore = logger_probability[held_context, held_action]

        marginal_policy = np.empty((n_contexts, n_actions), dtype=np.float64)
        held_out_policy_mean = np.empty_like(marginal_policy)
        for context_index in range(n_contexts):
            train_rows = training & (context == context_index)
            held_rows = held_out & (context == context_index)
            if not train_rows.any() or not held_rows.any():
                raise ValueError(
                    "every user fold must cover every shared-context stratum"
                )
            marginal_policy[context_index] = policy_by_user[user[train_rows]].mean(
                axis=0
            )
            held_out_policy_mean[context_index] = policy_by_user[user[held_rows]].mean(
                axis=0
            )
        maximum_shift = max(
            maximum_shift,
            float(np.max(np.abs(held_out_policy_mean - marginal_policy))),
        )

        intercept, slope, x_mean = _fit_outcome_nuisance(
            user_indices=train_user,
            actions=train_action,
            contexts=train_context,
            rewards=train_reward,
            signal=history_signal,
            n_contexts=n_contexts,
            n_actions=n_actions,
            smoothing=outcome_smoothing,
            ridge=outcome_ridge,
        )
        marginal_outcome = np.empty((n_contexts, n_actions), dtype=np.float64)
        for context_index in range(n_contexts):
            rows = training & (context == context_index)
            prediction = intercept[context_index][None, :] + slope[None, :] * (
                history_signal[user[rows]] - x_mean[None, :]
            )
            marginal_outcome[context_index] = np.clip(
                prediction, 1e-4, 1.0 - 1e-4
            ).mean(axis=0)

        held_policy = policy_by_user[held_user]
        held_reference = marginal_policy[held_context]
        held_prediction = intercept[held_context] + slope[None, :] * (
            history_signal[held_user] - x_mean[None, :]
        )
        held_prediction = np.clip(held_prediction, 1e-4, 1.0 - 1e-4)
        held_marginal_outcome = marginal_outcome[held_context]
        held_reward = reward[held_out]
        centered[held_out] = centered_matching_dr_scores(
            held_policy,
            held_reference,
            held_action,
            held_reward,
            pscore,
            held_prediction,
            held_marginal_outcome,
        )
        row = np.arange(held_action.size)
        logged_contrast = (
            held_policy[row, held_action] - held_reference[row, held_action]
        )
        separate_dr[held_out] = np.sum(
            (held_policy - held_reference) * held_prediction,
            axis=1,
        ) + logged_contrast / pscore * (
            held_reward - held_prediction[row, held_action]
        )
        ipw[held_out] = logged_contrast / pscore * held_reward
        pscore_all[held_out] = pscore
        tv_all[held_out] = 0.5 * np.abs(
            held_policy - held_reference
        ).sum(axis=1)
        logged_weight_all[held_out] = np.abs(logged_contrast / pscore)

    return CrossFitScores(
        centered_scores=centered,
        separate_dr_scores=separate_dr,
        ipw_scores=ipw,
        logging_pscore=pscore_all,
        fold_ids=row_fold,
        mean_policy_to_reference_tv=float(tv_all.mean()),
        maximum_policy_to_reference_tv=float(tv_all.max()),
        maximum_logged_contrast_weight=float(logged_weight_all.max()),
        minimum_logging_pscore=float(pscore_all.min()),
        maximum_logging_pscore=float(pscore_all.max()),
        maximum_fold_marginal_shift=float(maximum_shift),
    )


def cluster_inference(
    scores: np.ndarray,
    clusters: np.ndarray,
    *,
    prefix: str,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(clusters)
    if values.ndim != 1 or labels.shape != values.shape:
        raise ValueError("scores and clusters must be aligned vectors")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    unique, codes = np.unique(labels, return_inverse=True)
    n_clusters = int(unique.size)
    if n_clusters < 2:
        raise ValueError("cluster inference requires at least two clusters")
    estimate = float(values.mean())
    cluster_sums = np.bincount(
        codes,
        weights=values - estimate,
        minlength=n_clusters,
    )
    standard_error = float(
        np.sqrt(n_clusters / (n_clusters - 1) * np.square(cluster_sums).sum())
        / values.size
    )
    critical = float(t.ppf(1.0 - alpha / 2.0, df=n_clusters - 1))
    absolute_sum = float(np.abs(cluster_sums).sum())
    maximum_share = (
        float(np.abs(cluster_sums).max() / absolute_sum)
        if absolute_sum > 0.0
        else 0.0
    )
    return {
        f"{prefix}_clusters": n_clusters,
        f"{prefix}_standard_error": standard_error,
        f"{prefix}_ci_lower": estimate - critical * standard_error,
        f"{prefix}_ci_upper": estimate + critical * standard_error,
        f"{prefix}_normal_mde_80pct": normal_minimum_detectable_effect(
            standard_error
        ),
        f"{prefix}_maximum_absolute_influence_share": maximum_share,
    }


def enumerated_cluster_sign_flip_interval(
    scores: np.ndarray,
    clusters: np.ndarray,
    *,
    prefix: str,
    alpha: float = 0.05,
) -> dict[str, float | int | str]:
    """Enumerated Rademacher wild-cluster sensitivity interval for a score mean.

    The interval flips every combination of centered cluster sums.  It is a
    small-cluster diagnostic, not a replacement for the cluster-t interval:
    its justification needs approximate symmetry across independent clusters.
    """

    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(clusters)
    if values.ndim != 1 or labels.shape != values.shape:
        raise ValueError("scores and clusters must be aligned vectors")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    unique, codes = np.unique(labels, return_inverse=True)
    n_clusters = int(unique.size)
    if not 2 <= n_clusters <= 20:
        raise ValueError("enumerated sign flips require between 2 and 20 clusters")

    estimate = float(values.mean())
    cluster_sums = np.bincount(
        codes,
        weights=values - estimate,
        minlength=n_clusters,
    )
    combinations = np.arange(1 << n_clusters, dtype=np.uint32)[:, None]
    bit_positions = np.arange(n_clusters, dtype=np.uint32)[None, :]
    multipliers = 2.0 * ((combinations >> bit_positions) & 1) - 1.0
    perturbations = (
        np.sqrt(n_clusters / (n_clusters - 1))
        * (multipliers @ cluster_sums)
        / values.size
    )
    lower_quantile, upper_quantile = np.quantile(
        perturbations,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return {
        f"{prefix}_sign_flip_draws": int(1 << n_clusters),
        f"{prefix}_sign_flip_ci_lower": estimate - float(upper_quantile),
        f"{prefix}_sign_flip_ci_upper": estimate - float(lower_quantile),
        f"{prefix}_sign_flip_assumption": "independent_symmetric_clusters",
    }


def two_way_cluster_inference(
    scores: np.ndarray,
    first_clusters: np.ndarray,
    second_clusters: np.ndarray,
    *,
    prefix: str,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Cameron--Gelbach--Miller two-way cluster inference for a score mean."""

    values = np.asarray(scores, dtype=np.float64)
    first = np.asarray(first_clusters)
    second = np.asarray(second_clusters)
    if (
        values.ndim != 1
        or first.shape != values.shape
        or second.shape != values.shape
    ):
        raise ValueError("scores and both cluster vectors must be aligned")
    estimate = float(values.mean())
    centered = values - estimate

    def cluster_variance(labels: np.ndarray) -> tuple[float, int]:
        _, codes = np.unique(labels, return_inverse=True)
        groups = int(codes.max()) + 1
        if groups < 2:
            raise ValueError("two-way inference requires at least two clusters per axis")
        sums = np.bincount(codes, weights=centered, minlength=groups)
        variance = groups / (groups - 1) * np.square(sums).sum() / values.size**2
        return float(variance), groups

    first_variance, first_count = cluster_variance(first)
    second_variance, second_count = cluster_variance(second)
    intersection = pd.MultiIndex.from_arrays([first, second])
    intersection_variance, intersection_count = cluster_variance(intersection)
    variance = first_variance + second_variance - intersection_variance
    if variance <= 0.0:
        raise ValueError("two-way cluster variance is non-positive")
    standard_error = float(np.sqrt(variance))
    degrees_of_freedom = min(first_count, second_count) - 1
    critical = float(t.ppf(1.0 - alpha / 2.0, df=degrees_of_freedom))
    return {
        f"{prefix}_first_clusters": first_count,
        f"{prefix}_second_clusters": second_count,
        f"{prefix}_intersection_clusters": intersection_count,
        f"{prefix}_degrees_of_freedom": degrees_of_freedom,
        f"{prefix}_standard_error": standard_error,
        f"{prefix}_ci_lower": estimate - critical * standard_error,
        f"{prefix}_ci_upper": estimate + critical * standard_error,
        f"{prefix}_normal_mde_80pct": normal_minimum_detectable_effect(
            standard_error
        ),
    }


def resolution_status(
    ci_lower: float,
    ci_upper: float,
    *,
    equivalence_margin: float,
) -> str:
    if equivalence_margin <= 0.0:
        raise ValueError("equivalence_margin must be positive")
    if ci_lower > equivalence_margin:
        return "positive_practically_relevant"
    if ci_upper < -equivalence_margin:
        return "negative_practically_relevant"
    if ci_lower > 0.0:
        return "positive_but_submargin"
    if ci_upper < 0.0:
        return "negative_but_submargin"
    if ci_lower >= -equivalence_margin and ci_upper <= equivalence_margin:
        return "practically_equivalent"
    return "inconclusive"


def temporal_diagnostics(
    evaluation: pd.DataFrame,
    scores: np.ndarray,
    *,
    outcome: str,
    target_tv: float,
) -> list[dict[str, float | int | str]]:
    dates = evaluation["date"].to_numpy()
    rows: list[dict[str, float | int | str]] = []
    total_sum = float(np.sum(scores))
    total_n = len(scores)
    for date in np.unique(dates):
        mask = dates == date
        leave_n = total_n - int(mask.sum())
        rows.append(
            {
                "outcome": outcome,
                "target_tv": float(target_tv),
                "date": int(date),
                "rows": int(mask.sum()),
                "day_mean_score": float(np.mean(scores[mask])),
                "leave_one_day_out_estimate": float(
                    (total_sum - float(np.sum(scores[mask]))) / leave_n
                ),
            }
        )
    return rows


def permutation_placebo(
    evaluation: pd.DataFrame,
    *,
    policy_by_user: np.ndarray,
    pscore: np.ndarray,
    rewards: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[list[dict[str, float | int]], float, float]:
    """Break user-policy binding while preserving the policy bank."""

    if repetitions < 1:
        raise ValueError("placebo repetitions must be positive")
    user = evaluation["user_index"].to_numpy(dtype=np.int64)
    action = evaluation["action"].to_numpy(dtype=np.int64)
    context = evaluation["context_index"].to_numpy(dtype=np.int64)
    n_contexts = int(context.max()) + 1
    n_users, n_actions = policy_by_user.shape
    user_context_count = np.zeros((n_contexts, n_users), dtype=np.float64)
    np.add.at(user_context_count, (context, user), 1.0)

    def ipw_value(candidate: np.ndarray) -> float:
        marginal = user_context_count @ candidate
        marginal /= user_context_count.sum(axis=1, keepdims=True)
        delta = candidate[user, action] - marginal[context, action]
        return float(np.mean(delta / pscore * rewards))

    observed = ipw_value(policy_by_user)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        candidate = policy_by_user[rng.permutation(n_users)]
        values[index] = ipw_value(candidate)
        rows.append(
            {
                "permutation": index,
                "matching_ipw": float(values[index]),
            }
        )
    p_value = float(
        (1 + np.sum(np.abs(values) >= abs(observed))) / (repetitions + 1)
    )
    return rows, observed, p_value


def _save_figure(
    results: pd.DataFrame,
    day_results: pd.DataFrame,
    *,
    output_pdf: Path,
    output_png: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for outcome, label, marker in (
        ("long_view", "Long view", "o"),
        ("is_click", "Click", "s"),
    ):
        rows = results.loc[results["outcome"] == outcome].sort_values("target_tv")
        axes[0].errorbar(
            rows["target_tv"],
            PER_1000 * rows["matching_value"],
            yerr=np.vstack(
                [
                    PER_1000
                    * (rows["matching_value"] - rows["user_ci_lower"]),
                    PER_1000
                    * (rows["user_ci_upper"] - rows["matching_value"]),
                ]
            ),
            marker=marker,
            capsize=3,
            label=label,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Pre-set policy TV")
    axes[0].set_ylabel("Matching value per 1,000")
    axes[0].set_title("Random-log estimates (user-cluster 95% CI)")
    axes[0].legend()

    primary = day_results.loc[day_results["outcome"] == "long_view"].copy()
    target = float(primary["target_tv"].median())
    primary = primary.loc[np.isclose(primary["target_tv"], target)]
    axes[1].plot(
        primary["date"].astype(str),
        PER_1000 * primary["day_mean_score"],
        marker="o",
        label="Day mean",
    )
    axes[1].plot(
        primary["date"].astype(str),
        PER_1000 * primary["leave_one_day_out_estimate"],
        marker="s",
        label="Leave one day out",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    shown_ticks = np.arange(0, primary.shape[0], 3)
    axes[1].set_xticks(
        shown_ticks,
        labels=primary["date"].astype(str).iloc[shown_ticks],
        rotation=45,
        ha="right",
    )
    axes[1].set_ylabel("Matching score per 1,000")
    axes[1].set_title(f"Long-view day check at TV={target:.2f}")
    axes[1].legend()
    figure.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_protocol(
    *,
    data_dir: Path,
    category_metadata_path: Path,
    output_dir: Path,
    tab: int,
    minimum_history_events: int,
    minimum_items_per_category: int,
    shared_context_hour_width: int,
    folds: int,
    target_tvs: Sequence[float],
    primary_target_tv: float,
    smoothing_grid: Sequence[float],
    validation_days: int,
    propensity_smoothing: float,
    outcome_smoothing: float,
    outcome_ridge: float,
    equivalence_margin: float,
    placebo_repetitions: int,
    seed: int,
) -> dict[str, object]:
    standard, evaluation, mapping, data_audit = load_kuairand_category_data(
        data_dir,
        category_metadata_path=category_metadata_path,
        tab=tab,
        minimum_history_events=minimum_history_events,
        minimum_items_per_category=minimum_items_per_category,
        shared_context_hour_width=shared_context_hour_width,
    )
    user_ids = np.sort(evaluation["user_id"].unique()).astype(np.int64)
    user_fold = assign_user_folds(len(user_ids), folds=folds, seed=seed)
    all_results: list[dict[str, object]] = []
    all_days: list[dict[str, object]] = []
    policy_audit: dict[str, object] = {}
    primary_by_outcome: dict[
        str, tuple[np.ndarray, np.ndarray, CrossFitScores]
    ] = {}
    for outcome in SUPPORTED_OUTCOMES:
        frozen = fit_frozen_history_policy(
            standard,
            user_ids=user_ids,
            outcome=outcome,
            n_actions=len(mapping.labels),
            smoothing_grid=smoothing_grid,
            validation_days=validation_days,
        )
        policy_audit[outcome] = {
            "selected_smoothing": frozen.selected_smoothing,
            "preperiod_validation": list(frozen.validation_rows),
        }
        for target_tv in target_tvs:
            beta, policy, calibrated_tv = calibrate_policy_to_tv(
                frozen.signal,
                frozen.global_rate,
                mapping.base_probability,
                target_tv=float(target_tv),
            )
            scores = cross_fitted_scores(
                evaluation,
                policy_by_user=policy,
                history_signal=frozen.signal,
                base_probability=mapping.base_probability,
                user_fold=user_fold,
                propensity_smoothing=propensity_smoothing,
                outcome_smoothing=outcome_smoothing,
                outcome_ridge=outcome_ridge,
                outcome=outcome,
            )
            iid = estimate_from_scores(scores.centered_scores)
            user_cluster = cluster_inference(
                scores.centered_scores,
                evaluation["user_id"].to_numpy(),
                prefix="user",
            )
            day_cluster = cluster_inference(
                scores.centered_scores,
                evaluation["date"].to_numpy(),
                prefix="day",
            )
            day_sign_flip = enumerated_cluster_sign_flip_interval(
                scores.centered_scores,
                evaluation["date"].to_numpy(),
                prefix="day",
            )
            user_day_cluster = two_way_cluster_inference(
                scores.centered_scores,
                evaluation["user_id"].to_numpy(),
                evaluation["date"].to_numpy(),
                prefix="user_day",
            )
            row = {
                "outcome": outcome,
                "target_tv": float(target_tv),
                "calibrated_equal_user_tv": calibrated_tv,
                "inverse_temperature": beta,
                "rows": int(len(evaluation)),
                "users": int(len(user_ids)),
                "dates": int(evaluation["date"].nunique()),
                "outcome_mean": float(evaluation[outcome].mean()),
                "matching_value": float(iid.value),
                "iid_standard_error": float(iid.standard_error),
                "iid_ci_lower": float(iid.ci_lower),
                "iid_ci_upper": float(iid.ci_upper),
                "iid_mde_80pct": float(iid.minimum_detectable_effect),
                **user_cluster,
                **day_cluster,
                **day_sign_flip,
                **user_day_cluster,
                "resolution_status": resolution_status(
                    float(user_cluster["user_ci_lower"]),
                    float(user_cluster["user_ci_upper"]),
                    equivalence_margin=equivalence_margin,
                ),
                "equivalence_margin": equivalence_margin,
                "separate_dr_matching_value": float(
                    scores.separate_dr_scores.mean()
                ),
                "ipw_matching_value": float(scores.ipw_scores.mean()),
                "mean_policy_to_reference_tv": scores.mean_policy_to_reference_tv,
                "maximum_policy_to_reference_tv": scores.maximum_policy_to_reference_tv,
                "maximum_logged_contrast_weight": scores.maximum_logged_contrast_weight,
                "minimum_logging_pscore": scores.minimum_logging_pscore,
                "maximum_logging_pscore": scores.maximum_logging_pscore,
                "maximum_fold_marginal_shift": scores.maximum_fold_marginal_shift,
            }
            all_results.append(row)
            all_days.extend(
                temporal_diagnostics(
                    evaluation,
                    scores.centered_scores,
                    outcome=outcome,
                    target_tv=float(target_tv),
                )
            )
            if np.isclose(target_tv, primary_target_tv):
                primary_by_outcome[outcome] = (policy, frozen.signal, scores)

    if set(primary_by_outcome) != set(SUPPORTED_OUTCOMES):
        raise ValueError("primary_target_tv must appear in target_tvs")
    structural_category_support = np.tile(
        mapping.item_counts > 0,
        (int(evaluation["context_index"].max()) + 1, 1),
    )
    contrast_overlap_audit: dict[str, object] = {
        "action_level": "first_level_category_within_fixed_random_video_pool",
        "support_source": (
            "published uniform fixed-pool random-intervention design and "
            "candidate-pool category metadata; finite context-category zero "
            "counts are not used to define structural support"
        ),
        "category_pool_minimum_items": int(mapping.item_counts.min()),
        "category_pool_maximum_items": int(mapping.item_counts.max()),
        "outcomes_at_primary_tv": {
            outcome: audit_contrast_overlap(
                primary_by_outcome[outcome][0],
                event_user_index=evaluation["user_index"].to_numpy(),
                event_context_index=evaluation["context_index"].to_numpy(),
                structural_support=structural_category_support,
            )
            for outcome in SUPPORTED_OUTCOMES
        },
        "exact_video_status": "not_certified",
        "exact_video_reason": (
            "although the published design describes uniform draws from a "
            "fixed 7583-video pool, released per-video frequencies are "
            "inconsistent with assigning every row propensity 1/7583 and the "
            "release has no row-level eligibility metadata; the weaker pooled-"
            "category support certificate is not promoted to exact-video OPE"
        ),
    }
    primary_policy, _, primary_scores = primary_by_outcome["long_view"]
    primary_reward = evaluation["long_view"].to_numpy(dtype=np.float64)
    placebo_rows, placebo_observed, placebo_p = permutation_placebo(
        evaluation,
        policy_by_user=primary_policy,
        pscore=primary_scores.logging_pscore,
        rewards=primary_reward,
        repetitions=placebo_repetitions,
        seed=seed + 10_007,
    )
    for row in placebo_rows:
        row["outcome"] = "long_view"
        row["target_tv"] = float(primary_target_tv)

    propensity_rows: list[dict[str, object]] = []
    for outcome in SUPPORTED_OUTCOMES:
        policy, history_signal, estimated_scores = primary_by_outcome[outcome]
        fixed_scores = cross_fitted_scores(
            evaluation,
            policy_by_user=policy,
            history_signal=history_signal,
            base_probability=mapping.base_probability,
            user_fold=user_fold,
            propensity_smoothing=propensity_smoothing,
            fixed_logging_probability=mapping.base_probability,
            outcome_smoothing=outcome_smoothing,
            outcome_ridge=outcome_ridge,
            outcome=outcome,
        )
        for propensity_model, scores in (
            ("fold_excluded_context_frequency", estimated_scores),
            ("candidate_pool_category_mass", fixed_scores),
        ):
            user_cluster = cluster_inference(
                scores.centered_scores,
                evaluation["user_id"].to_numpy(),
                prefix="user",
            )
            user_day_cluster = two_way_cluster_inference(
                scores.centered_scores,
                evaluation["user_id"].to_numpy(),
                evaluation["date"].to_numpy(),
                prefix="user_day",
            )
            propensity_rows.append(
                {
                    "outcome": outcome,
                    "target_tv": float(primary_target_tv),
                    "propensity_model": propensity_model,
                    "matching_value": float(scores.centered_scores.mean()),
                    "user_ci_lower": float(user_cluster["user_ci_lower"]),
                    "user_ci_upper": float(user_cluster["user_ci_upper"]),
                    "user_day_ci_lower": float(
                        user_day_cluster["user_day_ci_lower"]
                    ),
                    "user_day_ci_upper": float(
                        user_day_cluster["user_day_ci_upper"]
                    ),
                    "separate_dr_matching_value": float(
                        scores.separate_dr_scores.mean()
                    ),
                    "ipw_matching_value": float(scores.ipw_scores.mean()),
                    "minimum_logging_pscore": scores.minimum_logging_pscore,
                    "maximum_logging_pscore": scores.maximum_logging_pscore,
                    "maximum_logged_contrast_weight": (
                        scores.maximum_logged_contrast_weight
                    ),
                }
            )

    results_path = output_dir / "kuairand_category_matching_results.csv"
    days_path = output_dir / "kuairand_category_day_sensitivity.csv"
    placebo_path = output_dir / "kuairand_category_permutation_placebo.csv"
    propensity_path = output_dir / "kuairand_category_propensity_sensitivity.csv"
    _write_rows(results_path, all_results)
    _write_rows(days_path, all_days)
    _write_rows(placebo_path, placebo_rows)
    _write_rows(propensity_path, propensity_rows)
    results_frame = pd.DataFrame(all_results)
    days_frame = pd.DataFrame(all_days)
    _save_figure(
        results_frame,
        days_frame,
        output_pdf=output_dir / "kuairand_category_matching.pdf",
        output_png=output_dir / "kuairand_category_matching.png",
    )

    payload: dict[str, object] = {
        "protocol": {
            "dataset": "KuaiRand-Pure",
            "official_source": OFFICIAL_DATASET_URL,
            "evaluation_unit": "random video-insertion opportunity",
            "action": (
                "official first-level video category, rare categories merged by "
                "candidate-pool item count; logger within-category item sampler retained"
            ),
            "estimand_population": (
                "event-weighted warm-start users in tab 1 during the released "
                "2022-04-22 to 2022-05-08 random-intervention window"
            ),
            "shared_context": f"date x {shared_context_hour_width}-hour bin",
            "temporal_scope": (
                "fixed-window inference; day-cluster results diagnose but do not "
                "establish transport to future dates"
            ),
            "primary_outcome": "long_view",
            "secondary_outcome": "is_click",
            "primary_target_tv": float(primary_target_tv),
            "target_tv_grid": [float(value) for value in target_tvs],
            "target_training": (
                "history-smoothed user-category outcome rates from the preceding "
                "standard-log window only"
            ),
            "target_strength_selection": (
                "calibrated to equal-user TV without randomized evaluation outcomes"
            ),
            "nuisance_cross_fitting": f"{folds} folds split by user",
            "logger_propensity": (
                "cross-fitted shared-context-by-category assignment frequency "
                "with a candidate-pool category-mass Dirichlet prior"
            ),
            "logger_propensity_sensitivity": (
                "replace the cross-fitted context frequencies with fixed category "
                "mass from the released candidate-pool metadata"
            ),
            "primary_uncertainty": "user-cluster sandwich with t critical value",
            "dependence_sensitivity": (
                "date-cluster and user-by-date two-way cluster intervals; the "
                "two-way t critical value uses the smaller cluster-axis degrees of freedom; "
                "an exhaustive Rademacher sign-flip interval checks the 17-day "
                "approximation "
                "under independent symmetric day clusters"
            ),
            "equivalence_margin": float(equivalence_margin),
            "seed": int(seed),
        },
        "data_audit": data_audit,
        "category_mapping": {
            "source": mapping.source,
            "minimum_items_per_category": int(minimum_items_per_category),
            "raw_category_count": mapping.raw_category_count,
            "merged_rare_category_count": mapping.merged_rare_category_count,
            "final_category_count": len(mapping.labels),
        },
        "policy_audit": policy_audit,
        "contrast_overlap_audit": contrast_overlap_audit,
        "negative_control": {
            "shared_metadata_mass_policy_matching_value": 0.0,
            "reason": "policy is identical for every user, so policy-reference contrast is zero",
        },
        "permutation_placebo": {
            "repetitions": int(placebo_repetitions),
            "observed_full_marginal_ipw": float(placebo_observed),
            "two_sided_binding_p_value": float(placebo_p),
            "interpretation": (
                "diagnostic for user-policy binding, not a replacement for the "
                "user-cluster confidence interval"
            ),
        },
        "propensity_sensitivity": propensity_rows,
        "results": all_results,
        "outputs": {
            "results_csv": str(results_path),
            "day_sensitivity_csv": str(days_path),
            "permutation_placebo_csv": str(placebo_path),
            "propensity_sensitivity_csv": str(propensity_path),
            "figure_pdf": str(output_dir / "kuairand_category_matching.pdf"),
            "figure_png": str(output_dir / "kuairand_category_matching.png"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "kuairand_category_matching_protocol.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--category-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tab", type=int, default=1)
    parser.add_argument("--minimum-history-events", type=int, default=10)
    parser.add_argument("--minimum-items-per-category", type=int, default=30)
    parser.add_argument("--shared-context-hour-width", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-tv-grid", default="0.02,0.04,0.07")
    parser.add_argument("--primary-target-tv", type=float, default=0.04)
    parser.add_argument("--smoothing-grid", default="1,5,20,100")
    parser.add_argument("--validation-days", type=int, default=4)
    parser.add_argument("--propensity-smoothing", type=float, default=1.0)
    parser.add_argument("--outcome-smoothing", type=float, default=20.0)
    parser.add_argument("--outcome-ridge", type=float, default=1.0)
    parser.add_argument("--equivalence-margin", type=float, default=0.001)
    parser.add_argument("--placebo-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_protocol(
        data_dir=args.data_dir,
        category_metadata_path=args.category_metadata,
        output_dir=args.output_dir,
        tab=args.tab,
        minimum_history_events=args.minimum_history_events,
        minimum_items_per_category=args.minimum_items_per_category,
        shared_context_hour_width=args.shared_context_hour_width,
        folds=args.folds,
        target_tvs=_parse_numbers(args.target_tv_grid),
        primary_target_tv=args.primary_target_tv,
        smoothing_grid=_parse_numbers(args.smoothing_grid),
        validation_days=args.validation_days,
        propensity_smoothing=args.propensity_smoothing,
        outcome_smoothing=args.outcome_smoothing,
        outcome_ridge=args.outcome_ridge,
        equivalence_margin=args.equivalence_margin,
        placebo_repetitions=args.placebo_repetitions,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
