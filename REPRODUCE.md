# Reproduce the WSDM paper

Run every command from the repository root with Python 3.10. Install the fixed
dependencies and run the existing tests and the data-free example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python examples/two_users.py
```

The public code pipeline runs experiments and plots. It does not require LaTeX.
Paper PDFs, sources, and typesetting files are provided in the
[paper bundle](paper/MatchOPE_WSDM2027.zip). Compiling the paper is separate from
this pipeline. [RESULTS.md](RESULTS.md) maps outputs to the paper's figures and tables.

Experiment commands overwrite same-named files in
`research/wsdm2027_matchope/evidence/`. Use a separate checkout or copy for new
runs so that you can compare against the saved results. The two oracle commands
below intentionally share the audit and support-stress output paths; the later
command supplies the final files in a full pipeline run.

## Data

The raw data files are not included in this repository. [DATA.md](DATA.md)
documents exact input filenames, extraction paths, and the category prefix.

- KuaiRec: download `KuaiRec.zip` from
  [Zenodo record 18164998](https://zenodo.org/records/18164998) and place
  `small_matrix.csv` at
  `data/kuairec/small_matrix.csv`.
- KuaiRand-Pure: download `KuaiRand-Pure.tar.gz` from
  [Zenodo record 10439422](https://zenodo.org/records/10439422) and place the
  standard and random logs under
  `data/kuairand/KuaiRand-Pure/data/`. The category-file command is in
  `research/wsdm2027_matchope/evidence/kuairand_category_matching/README.md`.
- These KuaiRec and KuaiRand records, including the category metadata in
  [record 18159199](https://zenodo.org/records/18159199), list CC BY 4.0.
- Open Bandit Dataset: download the full release using the
  [official dataset instructions](https://github.com/st-tech/zr-obp/blob/master/obd/README.md)
  and place the random men's log at
  `data/open_bandit_dataset_full/open_bandit_dataset/random/men/men.csv`. The
  dataset is released under CC BY 4.0.

The saved JSON files record the seeds, grids, and checks used by the paper.

## One data-to-results entry point

The following command runs the fixed synthetic checks, policy fitting,
evaluation, and figures in 11 stages. All data locations are explicit;
the script does not infer or download a dataset.

```bash
bash scripts/run_matchope_paper_pipeline.sh \
  --kuairec-csv data/kuairec/small_matrix.csv \
  --kuairand-data-dir data/kuairand/KuaiRand-Pure/data \
  --kuairand-category-metadata data/kuairand/KuaiRand-Pure/data/kuairand_video_categories.prefix.csv \
  --obd-csv data/open_bandit_dataset_full/open_bandit_dataset/random/men/men.csv
```

Use `--dry-run` to print the complete command sequence without reading data or
writing results. The sections below expose the same stages separately.

## Synthetic checks

```bash
PYTHONPATH=src python -m matchope.pipeline.run_matchope_synthetic \
  --output-dir research/wsdm2027_matchope/evidence \
  --repetitions 300 \
  --nuisance-repetitions 500 \
  --nested-cluster-repetitions 500 \
  --seed 2027
```

The main mechanism and cluster grids use 300 repetitions. The nuisance and
7/10/20/40-day context-nesting grids use 500.

## KuaiRec oracle and support checks

First run the uniform-log recovery grid:

```bash
PYTHONPATH=src python -m matchope.pipeline.run_kuairec_matching_oracle \
  --csv data/kuairec/small_matrix.csv \
  --output-audit-csv research/wsdm2027_matchope/evidence/kuairec_eligibility_audit.csv \
  --output-recovery-csv research/wsdm2027_matchope/evidence/kuairec_uniform_recovery.csv \
  --output-support-csv research/wsdm2027_matchope/evidence/kuairec_support_stress.csv \
  --sample-size-grid 1000,10000,50000 \
  --logging-temperature-grid 0 \
  --repetitions 200 \
  --seed 20260814
```

Then run uneven logging and gradual support loss:

```bash
PYTHONPATH=src python -m matchope.pipeline.run_kuairec_matching_oracle \
  --csv data/kuairec/small_matrix.csv \
  --output-audit-csv research/wsdm2027_matchope/evidence/kuairec_eligibility_audit.csv \
  --output-recovery-csv research/wsdm2027_matchope/evidence/kuairec_logging_recovery.csv \
  --output-support-csv research/wsdm2027_matchope/evidence/kuairec_support_stress.csv \
  --output-degradation-csv research/wsdm2027_matchope/evidence/kuairec_popularity_support_degradation.csv \
  --sample-size-grid 1000,10000 \
  --logging-temperature-grid 0,2,4,8 \
  --repetitions 200 \
  --seed 20260814
```

## KuaiRec learned policies

```bash
PYTHONPATH=src python -m matchope.pipeline.run_kuairec_matching_learned \
  --csv data/kuairec/small_matrix.csv \
  --output-policy-csv research/wsdm2027_matchope/evidence/kuairec_learned_policy_values.csv \
  --output-ranking-csv research/wsdm2027_matchope/evidence/kuairec_learned_ranking_reversals.csv \
  --output-cross-family-policy-csv research/wsdm2027_matchope/evidence/kuairec_cross_family_policy_values.csv \
  --output-cross-family-ranking-csv research/wsdm2027_matchope/evidence/kuairec_cross_family_reversals.csv \
  --output-knn-selection-csv research/wsdm2027_matchope/evidence/kuairec_knn_inner_selection.csv \
  --output-history-sensitivity-csv research/wsdm2027_matchope/evidence/kuairec_history_sensitivity.csv \
  --output-factor-sensitivity-csv research/wsdm2027_matchope/evidence/kuairec_factor_hyperparameter_sensitivity.csv \
  --output-personalization-tv-matched-csv research/wsdm2027_matchope/evidence/kuairec_personalization_tv_matched_reversals.csv \
  --output-json research/wsdm2027_matchope/evidence/kuairec_learned_protocol.json
```

The next command gives interaction MF and kNN the same exposure marginal and
the same TV. It runs the invariance control, the paired user bootstrap, a
five-point marginal path, a fixed-marginal strength path through TV .50, and
an outcome-free sensitivity that replaces equal TV by equal mean KL from the
common marginal. It also varies MF rank and ridge and, separately, every fixed
kNN neighborhood fraction for the complete-watch comparison at common exposure
and TV .04.

```bash
PYTHONPATH=src python -m matchope.pipeline.run_kuairec_matching_controls \
  --csv data/kuairec/small_matrix.csv \
  --output-csv research/wsdm2027_matchope/evidence/kuairec_common_exposure_controls.csv \
  --output-summary-csv research/wsdm2027_matchope/evidence/kuairec_common_exposure_summary.csv \
  --output-reference-path-csv research/wsdm2027_matchope/evidence/kuairec_common_marginal_path.csv \
  --output-reference-path-summary-csv research/wsdm2027_matchope/evidence/kuairec_common_marginal_path_summary.csv \
  --output-strength-path-csv research/wsdm2027_matchope/evidence/kuairec_fixed_marginal_strength_path.csv \
  --output-strength-path-summary-csv research/wsdm2027_matchope/evidence/kuairec_fixed_marginal_strength_path_summary.csv \
  --output-kl-sensitivity-csv research/wsdm2027_matchope/evidence/kuairec_common_exposure_kl_sensitivity.csv \
  --output-kl-sensitivity-summary-csv research/wsdm2027_matchope/evidence/kuairec_common_exposure_kl_sensitivity_summary.csv \
  --output-factor-sensitivity-csv research/wsdm2027_matchope/evidence/kuairec_common_exposure_factor_sensitivity.csv \
  --output-factor-sensitivity-summary-csv research/wsdm2027_matchope/evidence/kuairec_common_exposure_factor_sensitivity_summary.csv \
  --output-knn-sensitivity-csv research/wsdm2027_matchope/evidence/kuairec_common_exposure_knn_sensitivity.csv \
  --output-knn-sensitivity-summary-csv research/wsdm2027_matchope/evidence/kuairec_common_exposure_knn_sensitivity_summary.csv \
  --output-json research/wsdm2027_matchope/evidence/kuairec_common_exposure_protocol.json \
  --user-bootstrap-replicates 20000 \
  --user-bootstrap-seed 20260815
```

The following semisynthetic check uses the split-11 complete-watch policies at
common exposure and TV .04. It samples sparse feedback from a known
popularity-biased logger and compares the centered score with IPW.

```bash
PYTHONPATH=src python -m matchope.pipeline.run_kuairec_learned_logging_recovery \
  --csv data/kuairec/small_matrix.csv \
  --output-csv research/wsdm2027_matchope/evidence/kuairec_learned_logging_recovery.csv \
  --output-json research/wsdm2027_matchope/evidence/kuairec_learned_logging_recovery.json \
  --sample-size-grid 50000,200000 \
  --repetitions 100 \
  --seed 20260821
```

## Randomized logs

Run KuaiRand with:

```bash
PYTHONPATH=src python -m matchope.pipeline.run_kuairand_matching_gate \
  --data-dir data/kuairand/KuaiRand-Pure/data \
  --category-metadata data/kuairand/KuaiRand-Pure/data/kuairand_video_categories.prefix.csv \
  --output-dir research/wsdm2027_matchope/evidence/kuairand_category_matching
```

Run the KuaiRand-scale known-truth calibration with:

```bash
PYTHONPATH=src python -m matchope.pipeline.run_kuairand_semisynthetic_zero \
  --data-dir data/kuairand/KuaiRand-Pure/data \
  --category-metadata data/kuairand/KuaiRand-Pure/data/kuairand_video_categories.prefix.csv \
  --output-dir research/wsdm2027_matchope/evidence/kuairand_semisynthetic_realrate \
  --effect-multiplier-grid 0,0.5,1,2 \
  --reference-matching-effect 0.000966 \
  --repetitions 500 \
  --seed 20260817
```

This calibration freezes the real row, user, date, context, category, fold,
and estimated-logger structure. The empirical long-view rate and released
outcomes shape a bounded probability surface; the policy and truth grid are
fixed before each new action/outcome log is drawn. The four effects use common
random numbers. The saved output directory contains the complete aggregate
coverage table and protocol JSON behind the four-row paper summary.

Run Open Bandit Dataset with:

```bash
PYTHONPATH=src python -m matchope.pipeline.run_obd_matching_gate \
  --csv data/open_bandit_dataset_full/open_bandit_dataset/random/men/men.csv \
  --inverse-temperature-grid 0.25,0.5,1,2,4,8,16 \
  --output-csv research/wsdm2027_matchope/evidence/obd_affinity_gate.csv \
  --output-path-csv research/wsdm2027_matchope/evidence/obd_affinity_path_audit.csv \
  --output-json research/wsdm2027_matchope/evidence/obd_affinity_gate.json
```

The OBD path audit adds 61 log-spaced strengths, the seven displayed grid
points, and the exact infinite-strength policy. Tied maximum-affinity
actions share probability equally at that endpoint.

Run the same-design OBD positive control with:

```bash
PYTHONPATH=src python -m matchope.pipeline.run_obd_semisynthetic_power \
  --csv data/open_bandit_dataset_full/open_bandit_dataset/random/men/men.csv \
  --output-csv research/wsdm2027_matchope/evidence/obd_semisynthetic_power.csv \
  --output-json research/wsdm2027_matchope/evidence/obd_semisynthetic_power.json \
  --effect-grid 0,0.003,0.006,0.009,0.012 \
  --repetitions 500 \
  --seed 20260816 \
  --outcome-smoothing 20
```

This control resamples the known uniform logger and Bernoulli outcomes while
holding the released rows, day labels, affinity policy, and injected matching
values fixed. The paper reports the seven-day cluster-$t$ and enumerated
sign-flip calibration. The saved empirical-null threshold is specific to this
simulated DGP and is not used for inference on the observed OBD log.

## Regenerate figures from saved results

This command uses only the included CSVs and does not need the raw datasets.
It regenerates five main figure files. Copy the repository first if you want
to preserve the original figure bytes for comparison.

```bash
PYTHONPATH=src python -m matchope.pipeline.plot_matchope_main_results \
  --personalization-tv-matched-csv research/wsdm2027_matchope/evidence/kuairec_personalization_tv_matched_reversals.csv \
  --common-exposure-csv research/wsdm2027_matchope/evidence/kuairec_common_exposure_controls.csv \
  --common-exposure-summary-csv research/wsdm2027_matchope/evidence/kuairec_common_exposure_summary.csv \
  --fixed-marginal-strength-csv research/wsdm2027_matchope/evidence/kuairec_fixed_marginal_strength_path.csv \
  --fixed-marginal-strength-summary-csv research/wsdm2027_matchope/evidence/kuairec_fixed_marginal_strength_path_summary.csv \
  --nuisance-coverage-csv research/wsdm2027_matchope/evidence/nuisance_coverage.csv \
  --support-degradation-csv research/wsdm2027_matchope/evidence/kuairec_popularity_support_degradation.csv \
  --output-pdf research/wsdm2027_matchope/evidence/main_empirical_results.pdf \
  --output-png research/wsdm2027_matchope/evidence/main_empirical_results.png \
  --output-teaser-pdf research/wsdm2027_matchope/evidence/main_empirical_teaser.pdf \
  --output-strength-path-pdf research/wsdm2027_matchope/evidence/main_empirical_strength_path.pdf \
  --output-theory-checks-pdf research/wsdm2027_matchope/evidence/main_empirical_theory_checks.pdf
```

The compact report files `nested_context_day_coverage_summary.csv` and
`kuairand_semisynthetic_calibration_summary.csv` are saved summaries. The
pipeline regenerates their detailed source tables but does not write these two
summaries. Their source rows and calculations are described in
[RESULTS.md](RESULTS.md#saved-report-summaries).

## Verification of this public snapshot

Checked on 2026-11-23 in a fresh Linux environment with Python 3.10.19 and
the versions in `requirements.txt` (NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3,
Matplotlib 3.10.7, pytest 8.4.2):

- The existing test suite passed: **86 passed**.
- The two-user example printed target 1.0, reference 0.5, matching 0.5.
- The pipeline dry run produced 11 stages; experiment arguments match the
  supplied research snapshot. Only the two paper-typesetting stages were removed.
- The plotting command generated all five requested figure files from the
  saved CSVs in a separate output directory.
- Local documentation links and the citation file were checked.

This check did not download the full external datasets or rerun every experiment.
It verifies installation, tests, the example, command organization, and plotting;
it is not an independent reproduction of every numerical result in the paper.
