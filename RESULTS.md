# Paper figures and saved results

All paths below are relative to `research/wsdm2027_matchope/evidence/`.
Module names are relative to `matchope.pipeline`; complete commands are in
[REPRODUCE.md](REPRODUCE.md). The paper is
[Higher Policy Value Can Mask Worse User–Item Matching](https://doi.org/10.1145/3862322.3863623).

## Reading the numbers

CSV value estimates are generally in raw outcome units. The paper multiplies
value gaps and intervals by 1,000 when reporting effects per 1,000 exposures.
KuaiRec comparisons average ten fixed action partitions with split seeds
`11,23,37,41,53,67,79,83,97,101`. These partitions are sensitivity settings,
not ten independent datasets. Conditional user-bootstrap intervals average
each user's contribution across partitions before resampling users.

The KuaiRec kNN-minus-MF sign convention means that a negative matching gap
favors interaction-only MF. Own-reference policies have different exposure
distributions; common-exposure comparisons hold both exposure and mean TV fixed.
Do not mix these two comparisons when checking the paper's numbers.

## Main figures and tables

| Paper item | Saved source | Generating module |
| --- | --- | --- |
| Introductory exposure/matching decomposition figure | `kuairec_personalization_tv_matched_reversals.csv` → `main_empirical_teaser.pdf` | `run_kuairec_matching_learned`, then `plot_matchope_main_results` |
| KuaiRec kNN-minus-MF table, own-reference columns | `kuairec_personalization_tv_matched_reversals.csv`; average the delta columns by outcome and target TV | `run_kuairec_matching_learned` |
| Same table, common-exposure column and positive-control fractions | `kuairec_common_exposure_summary.csv`, backed by `kuairec_common_exposure_controls.csv` and `kuairec_common_exposure_protocol.json` | `run_kuairec_matching_controls` |
| Fixed-common-exposure strength-path figure | `kuairec_fixed_marginal_strength_path.csv` and its `_summary.csv` → `main_empirical_strength_path.pdf` | `run_kuairec_matching_controls`, then `plot_matchope_main_results` |
| Known-case figure: reference nuisance error | `nuisance_coverage.csv` → panel of `main_empirical_theory_checks.pdf` | `run_matchope_synthetic`, then `plot_matchope_main_results` |
| Known-case figure: support loss | `kuairec_popularity_support_degradation.csv` → panel of `main_empirical_theory_checks.pdf` | `run_kuairec_matching_oracle`, then `plot_matchope_main_results` |
| Constructed support cases on KuaiRec | `kuairec_support_stress.csv`, with panel eligibility in `kuairec_eligibility_audit.csv` | `run_kuairec_matching_oracle` |
| Randomized-log results: KuaiRand | `kuairand_category_matching/kuairand_category_matching_results.csv` and `kuairand_category_matching_protocol.json` in the same directory | `run_kuairand_matching_gate` |
| Randomized-log results: OBD | `obd_affinity_gate.csv`, `obd_affinity_path_audit.csv`, `obd_affinity_gate.json` | `run_obd_matching_gate` |
| Combined empirical overview | `main_empirical_results.pdf` and `.png`, using the KuaiRec CSVs above | `plot_matchope_main_results` |

The notation and evidence-role tables describe definitions and study design;
they are not generated statistical outputs. Figure/table numbering may change
between manuscript versions, so this mapping identifies items by their content.

## Supplement and sensitivity analyses

| Evidence | Saved files | Generating module |
| --- | --- | --- |
| Basic mechanisms, precision, overlap cancellation, restricted efficiency | `mechanism_scenarios.csv`, `detectability_frontier.csv`, `support_cancellation_coverage.csv`, `equality_restricted_efficiency.csv`, `summary.json` | `run_matchope_synthetic` |
| Clustering and nested-context coverage | `cluster_coverage.csv`, `nested_context_day_coverage.csv`, `summary.json` | `run_matchope_synthetic` |
| Uniform and popularity-biased KuaiRec log recovery | `kuairec_uniform_recovery.csv`, `kuairec_logging_recovery.csv` | `run_kuairec_matching_oracle` |
| Learned policy values, rankings, and cross-family reversals | `kuairec_learned_policy_values.csv`, `kuairec_learned_ranking_reversals.csv`, `kuairec_cross_family_policy_values.csv`, `kuairec_cross_family_reversals.csv`, `kuairec_learned_protocol.json` | `run_kuairec_matching_learned` |
| kNN selection, history size, and MF hyperparameters | `kuairec_knn_inner_selection.csv`, `kuairec_history_sensitivity.csv`, `kuairec_factor_hyperparameter_sensitivity.csv` | `run_kuairec_matching_learned` |
| Common marginal path | `kuairec_common_marginal_path.csv` and its `_summary.csv` | `run_kuairec_matching_controls` |
| Equal-KL sensitivity | `kuairec_common_exposure_kl_sensitivity.csv` and its `_summary.csv` | `run_kuairec_matching_controls` |
| Common-exposure MF rank/ridge and kNN neighborhood sensitivity | `kuairec_common_exposure_factor_sensitivity.csv`, `kuairec_common_exposure_knn_sensitivity.csv`, and their `_summary.csv` files | `run_kuairec_matching_controls` |
| Learned-policy sparse-logging recovery | `kuairec_learned_logging_recovery.csv` and `.json` | `run_kuairec_learned_logging_recovery` |
| KuaiRand day, permutation, propensity, and protocol checks | Files ending in `_day_sensitivity.csv`, `_permutation_placebo.csv`, `_propensity_sensitivity.csv`, `_protocol_sensitivity.csv` under `kuairand_category_matching/`; companion `.pdf`, `.png`, protocol JSON, and README | `run_kuairand_matching_gate` |
| KuaiRand known-truth calibration | `kuairand_semisynthetic_realrate/kuairand_semisynthetic_coverage.csv` and its `summary.json` | `run_kuairand_semisynthetic_zero` |
| OBD same-design power and null control | `obd_semisynthetic_power.csv` and `.json` | `run_obd_semisynthetic_power` |

## Saved report summaries

Two compact CSVs are retained for reading the reported results. The current
pipeline regenerates the detailed tables but does not regenerate these summaries:

- `nested_context_day_coverage_summary.csv` selects the `method=day` rows
  from `nested_context_day_coverage.csv` for each of the three regimes at
  7, 10, 20, and 40 days. It pivots `coverage_95pct_or_sensitivity` into the
  regime-specific coverage columns and carries over `contexts_per_day` and
  `repetitions`.
- `kuairand_semisynthetic_calibration_summary.csv` combines methods at each
  truth level from `kuairand_semisynthetic_realrate/kuairand_semisynthetic_coverage.csv`.
  User, day, and two-way coverages use the corresponding centered methods.
  Bias/MCSE is `bias / bias_mcse` for the centered user-cluster estimate;
  SE/SD is its `mean_se_to_empirical_sd`. The uncentered/centered SD ratio and
  IPW/centered variance ratio use the methods' empirical standard deviations
  (squaring the ratio for variance). Two-way variance failures are also retained.
  The reported truth labels are the target values; floating-point computed
  truths can differ in their last digits. Displayed ratios are rounded.

## Scope and interpretation

The saved material includes the capped-watch comparisons whose intervals cross
zero, hyperparameter sensitivities with different directions, OBD intervals
that cross zero, and settings with poor coverage. These are part of the evidence,
not files to discard when reproducing the headline results.

KuaiRand evaluates category-level interventions under the released randomization
design; it does not establish exact-video overlap. OBD's observed-log results
have only seven dates, and its simulated positive controls do not establish a
positive effect in the observed log. Bootstrap intervals, cluster intervals,
and semisynthetic Monte Carlo coverage answer different questions.

Recorded seeds and fixed package versions help reproduce computations. Floating
point libraries can change last digits, and newly rendered figure files can
differ in metadata. Compare numerical tables, definitions, and uncertainty
summaries rather than expecting figure files to be identical byte for byte.
