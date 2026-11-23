#!/usr/bin/env python3
from __future__ import annotations

"""Plot the compact empirical evidence chain for the WSDM manuscript."""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import NullFormatter


PER_1000 = 1_000.0


def _style_axis(axis: Axes) -> None:
    axis.grid(alpha=0.22, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_strength_path(
    axis: Axes,
    strength: pd.DataFrame,
    strength_summary: pd.DataFrame,
    *,
    title: str | None,
) -> None:
    strength_tvs = strength_summary["target_tv"].to_numpy(dtype=np.float64)
    for target_tv in strength_tvs:
        split_values = strength.loc[
            np.isclose(strength["target_tv"], target_tv),
            "delta_matching_knn_minus_factor",
        ] * PER_1000
        split_jitter = np.linspace(-0.035, 0.035, split_values.size)
        axis.scatter(
            target_tv * np.exp(split_jitter),
            split_values,
            color="tab:orange",
            marker="s",
            s=11,
            alpha=0.35,
            linewidths=0,
        )
    mean = PER_1000 * strength_summary["delta_matching_mean"].to_numpy()
    low = PER_1000 * strength_summary["user_bootstrap_low"].to_numpy()
    high = PER_1000 * strength_summary["user_bootstrap_high"].to_numpy()
    axis.errorbar(
        strength_tvs,
        mean,
        yerr=np.vstack((mean - low, high - mean)),
        color="tab:orange",
        marker="s",
        markersize=4.0,
        linewidth=1.3,
        capsize=2,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_xticks(
        strength_tvs,
        labels=[f"{value:g}" for value in strength_tvs],
    )
    axis.tick_params(axis="x", labelsize=8)
    axis.get_xaxis().set_minor_formatter(NullFormatter())
    if title is not None:
        axis.set_title(title)
    axis.set_xlabel("Target–common-reference TV (log scale)")
    axis.set_ylabel("kNN–MF gap per 1,000")
    axis.text(
        0.98,
        0.96,
        "Below zero: MF better",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color="tab:orange",
    )


def _plot_nuisance_check(
    axis: Axes,
    nuisance_subset: pd.DataFrame,
    *,
    title: str | None,
) -> None:
    method_styles = {
        "centered_score": ("Centered score", "o", "-"),
        "uncentered_dr": ("Uncentered DR", "s", "--"),
    }
    for method, (label, marker, linestyle) in method_styles.items():
        subset = nuisance_subset.loc[nuisance_subset["method"] == method]
        absolute_bias = PER_1000 * np.abs(subset["bias"])
        axis.plot(
            subset["sample_size"],
            absolute_bias,
            marker=marker,
            linestyle=linestyle,
            markersize=3.5,
            linewidth=1.3,
            label=label,
        )
        sample_sizes = subset["sample_size"].to_numpy()
        for sample_size, bias in zip(
            sample_sizes,
            absolute_bias,
            strict=True,
        ):
            axis.annotate(
                f"{bias:.2f}",
                xy=(sample_size, bias),
                xytext=(
                    7 if sample_size == sample_sizes.min() else -7,
                    -7 if method == "uncentered_dr" else 5,
                ),
                textcoords="offset points",
                ha="left" if sample_size == sample_sizes.min() else "right",
                va="top" if method == "uncentered_dr" else "bottom",
                fontsize=6.8,
            )
    axis.set_yscale("log")
    sample_sizes = sorted(nuisance_subset["sample_size"].unique())
    axis.set_xticks(
        sample_sizes,
        labels=[f"{value:,}" for value in sample_sizes],
    )
    if title is not None:
        axis.set_title(title)
    axis.set_xlabel("Sample size")
    axis.set_ylabel("Absolute bias per 1,000\n(log scale)")
    axis.legend(frameon=False)


def _plot_support_check(
    axis: Axes,
    degradation: pd.DataFrame,
    *,
    title: str | None,
) -> None:
    ordered = degradation.sort_values("supported_action_fraction")
    supported_percent = 100.0 * ordered["supported_action_fraction"]
    axis.plot(
        supported_percent,
        ordered["direct_width"],
        marker="o",
        markersize=3.5,
        linewidth=1.3,
        label="Direct contrast",
    )
    axis.plot(
        supported_percent,
        ordered["separate_value_width"],
        marker="s",
        markersize=3.2,
        linewidth=1.3,
        label="Separate values",
    )
    if title is not None:
        axis.set_title(title)
    axis.set_xlabel("Logging support (%)")
    axis.set_ylabel("Interval width")
    axis.legend(frameon=False)


def _plot_support_interval(
    axis: Axes,
    degradation: pd.DataFrame,
    *,
    title: str | None,
) -> None:
    ordered = degradation.sort_values("supported_action_fraction")
    supported_percent = 100.0 * ordered["supported_action_fraction"].to_numpy()
    lower = ordered["direct_lower"].to_numpy()
    upper = ordered["direct_upper"].to_numpy()
    truth = float(ordered["oracle_matching_value"].iloc[0])
    axis.fill_between(
        supported_percent,
        lower,
        upper,
        color="tab:blue",
        alpha=0.18,
        label="Direct identified set",
    )
    axis.plot(
        supported_percent,
        lower,
        color="tab:blue",
        marker="o",
        markersize=3,
    )
    axis.plot(
        supported_percent,
        upper,
        color="tab:blue",
        marker="o",
        markersize=3,
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(
        truth,
        color="0.35",
        linewidth=0.9,
        linestyle="--",
        label="Dense-matrix truth",
    )
    resolved = lower > 0.0
    axis.scatter(
        supported_percent[resolved],
        lower[resolved],
        s=24,
        facecolors="white",
        edgecolors="tab:blue",
        linewidths=1.0,
        zorder=4,
    )
    if title is not None:
        axis.set_title(title)
    axis.set_xlabel("Logging support (%)")
    axis.set_ylabel("Direct bound endpoints")
    axis.legend(frameon=False, fontsize=7.0, loc="lower right")


def _save_strength_path(
    strength: pd.DataFrame,
    strength_summary: pd.DataFrame,
    output_pdf: Path,
) -> None:
    """Create a standalone strength-path figure with no hidden panels."""

    figure, axis = plt.subplots(figsize=(3.35, 2.25))
    _plot_strength_path(axis, strength, strength_summary, title=None)
    _style_axis(axis)
    figure.subplots_adjust(left=0.21, right=0.98, top=0.98, bottom=0.20)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def _save_theory_checks(
    nuisance_subset: pd.DataFrame,
    degradation: pd.DataFrame,
    output_pdf: Path,
) -> None:
    """Create standalone estimation and support checks with no hidden panels."""

    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))
    _plot_nuisance_check(
        axes[0],
        nuisance_subset,
        title="(a) Centering cuts bias",
    )
    _plot_support_check(
        axes[1],
        degradation,
        title="(b) Direct bounds are narrower",
    )
    _plot_support_interval(
        axes[2],
        degradation,
        title="(c) Endpoints reveal the sign",
    )
    for axis in axes:
        _style_axis(axis)
        axis.title.set_fontsize(8.5)
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.88,
        bottom=0.20,
        wspace=0.38,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def _save_teaser(
    tv_matched: pd.DataFrame,
    target_tvs: np.ndarray,
    output_pdf: Path,
) -> None:
    """Save the total/exposure gap and matching gap on separate scales."""

    grouped = (
        tv_matched.groupby("target_policy_to_reference_tv", sort=True)[
            [
                "delta_overall_knn_minus_interaction_mf",
                "delta_reference_knn_minus_interaction_mf",
                "delta_matching_knn_minus_interaction_mf",
            ]
        ]
        .mean()
        .reindex(target_tvs)
        * PER_1000
    )
    x_positions = np.arange(target_tvs.size, dtype=np.float64)
    overall = grouped["delta_overall_knn_minus_interaction_mf"].to_numpy()
    exposure = grouped[
        "delta_reference_knn_minus_interaction_mf"
    ].to_numpy()
    matching = grouped[
        "delta_matching_knn_minus_interaction_mf"
    ].to_numpy()
    figure, (overall_axis, matching_axis) = plt.subplots(
        2,
        1,
        figsize=(3.35, 2.85),
        sharex=True,
        gridspec_kw={"height_ratios": (1.62, 1.08), "hspace": 0.36},
    )

    exposure_bars = overall_axis.bar(
        x_positions,
        exposure,
        width=0.50,
        color="tab:green",
        alpha=0.78,
        label="Exposure component",
        zorder=2,
    )
    total_points = overall_axis.scatter(
        x_positions,
        overall,
        color="black",
        marker="D",
        s=22,
        label="Total value gap",
        zorder=3,
    )
    overall_axis.axhline(0.0, color="black", linewidth=0.8)
    overall_axis.set_ylim(0.0, 1.12 * float(exposure.max()))
    overall_axis.set_yticks([0.0, 40.0, 80.0, 120.0])
    overall_axis.set_ylabel("Complete-watch gap\nper 1,000", fontsize=7.4)
    overall_axis.set_title(
        "(a) Total value gap is driven by exposure",
        loc="left",
        fontsize=8.1,
        fontweight="bold",
        pad=4,
    )
    overall_axis.legend(
        [total_points, exposure_bars],
        ["Total value gap", "Exposure component"],
        frameon=False,
        fontsize=6.8,
        loc="upper left",
        handlelength=1.2,
        borderaxespad=0.25,
    )
    overall_axis.tick_params(axis="x", which="both", length=0)

    matching_axis.scatter(
        x_positions,
        matching,
        color="tab:orange",
        s=24,
        zorder=3,
    )
    matching_axis.axhline(0.0, color="black", linewidth=0.8)
    matching_axis.set_ylim(-1.65, 0.15)
    matching_axis.set_yticks([-1.5, -1.0, -0.5, 0.0])
    matching_axis.set_xticks(
        x_positions,
        labels=[f"{value:g}" for value in target_tvs],
    )
    matching_axis.set_xlabel("Target–own-reference TV")
    matching_axis.set_ylabel(
        "kNN − MF matching gap\nper 1,000",
        fontsize=7.4,
    )
    matching_axis.set_title(
        "(b) Matching gap is negative",
        loc="left",
        fontsize=8.1,
        fontweight="bold",
        pad=4,
    )
    matching_axis.text(
        0.98,
        0.91,
        "negative = MF better",
        transform=matching_axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color="0.25",
    )
    for x_position, value in zip(x_positions, matching, strict=True):
        matching_axis.annotate(
            f"{value:.2f}".replace("-", "−"),
            xy=(x_position, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="tab:orange",
        )

    for axis in (overall_axis, matching_axis):
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.22, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.25, right=0.98, top=0.96, bottom=0.17)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def plot_main_results(
    *,
    personalization_tv_matched_csv: Path,
    common_exposure_csv: Path,
    common_exposure_summary_csv: Path,
    fixed_marginal_strength_csv: Path,
    fixed_marginal_strength_summary_csv: Path,
    nuisance_coverage_csv: Path,
    support_degradation_csv: Path,
    output_pdf: Path,
    output_png: Path | None = None,
    output_teaser_pdf: Path | None = None,
    output_strength_path_pdf: Path | None = None,
    output_theory_checks_pdf: Path | None = None,
) -> None:
    tv_matched = pd.read_csv(personalization_tv_matched_csv)
    common = pd.read_csv(common_exposure_csv)
    common_summary = pd.read_csv(common_exposure_summary_csv)
    strength = pd.read_csv(fixed_marginal_strength_csv)
    strength_summary = pd.read_csv(fixed_marginal_strength_summary_csv)
    nuisance = pd.read_csv(nuisance_coverage_csv)
    degradation = pd.read_csv(support_degradation_csv)
    required_tv_matched = {
        "outcome",
        "split_seed",
        "target_policy_to_reference_tv",
        "delta_overall_knn_minus_interaction_mf",
        "delta_reference_knn_minus_interaction_mf",
        "delta_matching_knn_minus_interaction_mf",
    }
    required_nuisance = {
        "scenario",
        "method",
        "sample_size",
        "bias",
    }
    required_degradation = {
        "supported_action_fraction",
        "oracle_matching_value",
        "direct_width",
        "separate_value_width",
        "direct_lower",
        "direct_upper",
    }
    required_common = {
        "outcome",
        "split_seed",
        "target_tv",
        "common_delta_matching_knn_minus_factor",
    }
    required_common_summary = {
        "outcome",
        "target_tv",
        "common_delta_matching_mean",
        "common_delta_user_bootstrap_low",
        "common_delta_user_bootstrap_high",
    }
    required_strength = {
        "outcome",
        "split_seed",
        "target_tv",
        "delta_matching_knn_minus_factor",
    }
    required_strength_summary = {
        "outcome",
        "target_tv",
        "delta_matching_mean",
        "user_bootstrap_low",
        "user_bootstrap_high",
    }
    for frame, required, name in (
        (tv_matched, required_tv_matched, "TV-matched ranking"),
        (common, required_common, "common-exposure split results"),
        (common_summary, required_common_summary, "common-exposure summary"),
        (strength, required_strength, "fixed-marginal strength results"),
        (
            strength_summary,
            required_strength_summary,
            "fixed-marginal strength summary",
        ),
        (nuisance, required_nuisance, "nuisance coverage"),
        (degradation, required_degradation, "support degradation"),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} table is missing columns: {sorted(missing)}")
    complete_watch_tv_matched = tv_matched.loc[
        tv_matched["outcome"] == "complete_watch"
    ]
    if complete_watch_tv_matched.empty:
        raise ValueError("TV-matched table has no complete_watch rows")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    target_tvs = np.asarray(
        sorted(common["target_tv"].unique()),
        dtype=np.float64,
    )
    strength = strength.loc[strength["outcome"] == "complete_watch"]
    strength_summary = (
        strength_summary.loc[strength_summary["outcome"] == "complete_watch"]
        .sort_values("target_tv")
    )
    _plot_strength_path(
        axes[0, 0],
        strength,
        strength_summary,
        title="(a) Gap over policy strength",
    )

    decomposition_fields = (
        (
            "delta_reference_knn_minus_interaction_mf",
            "Exposure-marginal value",
            "tab:green",
        ),
        (
            "delta_matching_knn_minus_interaction_mf",
            "Matching value",
            "tab:orange",
        ),
    )
    x_positions = np.arange(target_tvs.size, dtype=np.float64)
    bar_width = 0.32
    for index, (field, label, color) in enumerate(decomposition_fields):
        mean = (
            complete_watch_tv_matched.groupby(
                "target_policy_to_reference_tv",
                sort=True,
            )[field]
            .mean()
            .reindex(target_tvs)
            * PER_1000
        )
        axes[0, 1].bar(
            x_positions + (index - 0.5) * bar_width,
            mean,
            width=bar_width,
            color=color,
            alpha=0.82,
            label=label,
        )
    overall_mean = (
        complete_watch_tv_matched.groupby(
            "target_policy_to_reference_tv", sort=True
        )["delta_overall_knn_minus_interaction_mf"]
        .mean()
        .reindex(target_tvs)
        * PER_1000
    )
    axes[0, 1].scatter(
        x_positions,
        overall_mean,
        color="black",
        marker="D",
        s=22,
        label="Overall value (sum)",
        zorder=4,
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xticks(x_positions, [f"{value:g}" for value in target_tvs])
    axes[0, 1].set_title("(b) Exposure hides the matching loss")
    axes[0, 1].set_xlabel("Own-reference TV")
    axes[0, 1].set_ylabel("Value gap per 1,000")
    axes[0, 1].legend(frameon=False)
    matching_mean = (
        complete_watch_tv_matched.groupby(
            "target_policy_to_reference_tv", sort=True
        )["delta_matching_knn_minus_interaction_mf"]
        .mean()
        .reindex(target_tvs)
        * PER_1000
    )
    axes[0, 1].set_ylim(-18.0, 1.15 * float(overall_mean.max()))
    for index, value in enumerate(matching_mean):
        axes[0, 1].annotate(
            f"{value:.2f}",
            xy=(x_positions[index] + 0.5 * bar_width, value),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.0,
            color="tab:orange",
        )

    nuisance_subset = nuisance.loc[
        nuisance["scenario"] == "bar_pi_misspecified"
    ].sort_values("sample_size")
    method_labels = {
        "centered_score": "Centered score",
        "uncentered_dr": "Uncentered DR",
    }
    missing_nuisance_methods = set(method_labels).difference(
        nuisance_subset["method"].unique()
    )
    if nuisance_subset.empty or missing_nuisance_methods:
        raise ValueError(
            "nuisance coverage lacks the policy-marginal stress rows for "
            f"{sorted(missing_nuisance_methods)}"
        )
    _plot_nuisance_check(
        axes[1, 0],
        nuisance_subset,
        title="(c) Centering sharply reduces bias",
    )
    _plot_support_check(
        axes[1, 1],
        degradation,
        title="(d) Contrast bounds exploit cancellation",
    )

    for axis in axes.flat:
        _style_axis(axis)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_png, dpi=220, bbox_inches="tight")
    if output_teaser_pdf is not None:
        _save_teaser(
            complete_watch_tv_matched,
            target_tvs,
            output_teaser_pdf,
        )
    if output_strength_path_pdf is not None:
        _save_strength_path(strength, strength_summary, output_strength_path_pdf)
    if output_theory_checks_pdf is not None:
        _save_theory_checks(
            nuisance_subset,
            degradation,
            output_theory_checks_pdf,
        )
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--personalization-tv-matched-csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--common-exposure-csv", type=Path, required=True)
    parser.add_argument(
        "--common-exposure-summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--fixed-marginal-strength-csv", type=Path, required=True)
    parser.add_argument(
        "--fixed-marginal-strength-summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--nuisance-coverage-csv", type=Path, required=True)
    parser.add_argument("--support-degradation-csv", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--output-teaser-pdf", type=Path, default=None)
    parser.add_argument("--output-strength-path-pdf", type=Path, default=None)
    parser.add_argument("--output-theory-checks-pdf", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plot_main_results(
        personalization_tv_matched_csv=args.personalization_tv_matched_csv,
        common_exposure_csv=args.common_exposure_csv,
        common_exposure_summary_csv=args.common_exposure_summary_csv,
        fixed_marginal_strength_csv=args.fixed_marginal_strength_csv,
        fixed_marginal_strength_summary_csv=(
            args.fixed_marginal_strength_summary_csv
        ),
        nuisance_coverage_csv=args.nuisance_coverage_csv,
        support_degradation_csv=args.support_degradation_csv,
        output_pdf=args.output_pdf,
        output_png=args.output_png,
        output_teaser_pdf=args.output_teaser_pdf,
        output_strength_path_pdf=args.output_strength_path_pdf,
        output_theory_checks_pdf=args.output_theory_checks_pdf,
    )


if __name__ == "__main__":
    main()
