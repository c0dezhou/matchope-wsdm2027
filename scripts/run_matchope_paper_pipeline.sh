#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KUAIREC_CSV=""
KUAIRAND_DATA_DIR=""
KUAIRAND_CATEGORY_METADATA=""
OBD_CSV=""
DRY_RUN=0

usage() {
  printf '%s\n' \
    "Usage:" \
    "  bash scripts/run_matchope_paper_pipeline.sh \\" \
    "    --kuairec-csv PATH \\" \
    "    --kuairand-data-dir PATH \\" \
    "    --kuairand-category-metadata PATH \\" \
    "    --obd-csv PATH [--dry-run]" \
    "" \
    "Runs the fixed synthetic, KuaiRec, KuaiRand, OBD, and figure stages." \
    "The four public-data locations are required; no data path is inferred."
}

run_cmd() {
  printf '[run]'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kuairec-csv)
      KUAIREC_CSV="$2"
      shift 2
      ;;
    --kuairand-data-dir)
      KUAIRAND_DATA_DIR="$2"
      shift 2
      ;;
    --kuairand-category-metadata)
      KUAIRAND_CATEGORY_METADATA="$2"
      shift 2
      ;;
    --obd-csv)
      OBD_CSV="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$KUAIREC_CSV" || -z "$KUAIRAND_DATA_DIR" || -z "$KUAIRAND_CATEGORY_METADATA" || -z "$OBD_CSV" ]]; then
  usage >&2
  exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  for required_file in "$KUAIREC_CSV" "$KUAIRAND_CATEGORY_METADATA" "$OBD_CSV"; do
    if [[ ! -f "$required_file" ]]; then
      printf 'Missing required file: %s\n' "$required_file" >&2
      exit 1
    fi
  done
  if [[ ! -d "$KUAIRAND_DATA_DIR" ]]; then
    printf 'Missing required directory: %s\n' "$KUAIRAND_DATA_DIR" >&2
    exit 1
  fi
fi

cd "$REPO_ROOT"
PYTHON=(env "PYTHONPATH=$REPO_ROOT/src" python -m)
EVIDENCE_DIR="research/wsdm2027_matchope/evidence"
KUAIRAND_OUTPUT_DIR="$EVIDENCE_DIR/kuairand_category_matching"

run_cmd "${PYTHON[@]}" matchope.pipeline.run_matchope_synthetic \
  --output-dir "$EVIDENCE_DIR" \
  --repetitions 300 \
  --nuisance-repetitions 500 \
  --nested-cluster-repetitions 500 \
  --seed 2027

run_cmd "${PYTHON[@]}" matchope.pipeline.run_kuairec_matching_oracle \
  --csv "$KUAIREC_CSV" \
  --output-audit-csv "$EVIDENCE_DIR/kuairec_eligibility_audit.csv" \
  --output-recovery-csv "$EVIDENCE_DIR/kuairec_uniform_recovery.csv" \
  --output-support-csv "$EVIDENCE_DIR/kuairec_support_stress.csv" \
  --sample-size-grid 1000,10000,50000 \
  --logging-temperature-grid 0 \
  --repetitions 200 \
  --seed 20260814

run_cmd "${PYTHON[@]}" matchope.pipeline.run_kuairec_matching_oracle \
  --csv "$KUAIREC_CSV" \
  --output-audit-csv "$EVIDENCE_DIR/kuairec_eligibility_audit.csv" \
  --output-recovery-csv "$EVIDENCE_DIR/kuairec_logging_recovery.csv" \
  --output-support-csv "$EVIDENCE_DIR/kuairec_support_stress.csv" \
  --output-degradation-csv "$EVIDENCE_DIR/kuairec_popularity_support_degradation.csv" \
  --sample-size-grid 1000,10000 \
  --logging-temperature-grid 0,2,4,8 \
  --repetitions 200 \
  --seed 20260814

run_cmd "${PYTHON[@]}" matchope.pipeline.run_kuairec_matching_learned \
  --csv "$KUAIREC_CSV" \
  --output-policy-csv "$EVIDENCE_DIR/kuairec_learned_policy_values.csv" \
  --output-ranking-csv "$EVIDENCE_DIR/kuairec_learned_ranking_reversals.csv" \
  --output-cross-family-policy-csv "$EVIDENCE_DIR/kuairec_cross_family_policy_values.csv" \
  --output-cross-family-ranking-csv "$EVIDENCE_DIR/kuairec_cross_family_reversals.csv" \
  --output-knn-selection-csv "$EVIDENCE_DIR/kuairec_knn_inner_selection.csv" \
  --output-history-sensitivity-csv "$EVIDENCE_DIR/kuairec_history_sensitivity.csv" \
  --output-factor-sensitivity-csv "$EVIDENCE_DIR/kuairec_factor_hyperparameter_sensitivity.csv" \
  --output-personalization-tv-matched-csv "$EVIDENCE_DIR/kuairec_personalization_tv_matched_reversals.csv" \
  --output-json "$EVIDENCE_DIR/kuairec_learned_protocol.json"

run_cmd "${PYTHON[@]}" matchope.pipeline.run_kuairec_matching_controls \
  --csv "$KUAIREC_CSV" \
  --output-csv "$EVIDENCE_DIR/kuairec_common_exposure_controls.csv" \
  --output-summary-csv "$EVIDENCE_DIR/kuairec_common_exposure_summary.csv" \
  --output-reference-path-csv "$EVIDENCE_DIR/kuairec_common_marginal_path.csv" \
  --output-reference-path-summary-csv "$EVIDENCE_DIR/kuairec_common_marginal_path_summary.csv" \
  --output-strength-path-csv "$EVIDENCE_DIR/kuairec_fixed_marginal_strength_path.csv" \
  --output-strength-path-summary-csv "$EVIDENCE_DIR/kuairec_fixed_marginal_strength_path_summary.csv" \
  --output-kl-sensitivity-csv "$EVIDENCE_DIR/kuairec_common_exposure_kl_sensitivity.csv" \
  --output-kl-sensitivity-summary-csv "$EVIDENCE_DIR/kuairec_common_exposure_kl_sensitivity_summary.csv" \
  --output-factor-sensitivity-csv "$EVIDENCE_DIR/kuairec_common_exposure_factor_sensitivity.csv" \
  --output-factor-sensitivity-summary-csv "$EVIDENCE_DIR/kuairec_common_exposure_factor_sensitivity_summary.csv" \
  --output-knn-sensitivity-csv "$EVIDENCE_DIR/kuairec_common_exposure_knn_sensitivity.csv" \
  --output-knn-sensitivity-summary-csv "$EVIDENCE_DIR/kuairec_common_exposure_knn_sensitivity_summary.csv" \
  --output-json "$EVIDENCE_DIR/kuairec_common_exposure_protocol.json" \
  --user-bootstrap-replicates 20000 \
  --user-bootstrap-seed 20260815

run_cmd "${PYTHON[@]}" matchope.pipeline.run_kuairec_learned_logging_recovery \
  --csv "$KUAIREC_CSV" \
  --output-csv "$EVIDENCE_DIR/kuairec_learned_logging_recovery.csv" \
  --output-json "$EVIDENCE_DIR/kuairec_learned_logging_recovery.json" \
  --sample-size-grid 50000,200000 \
  --repetitions 100 \
  --seed 20260821

run_cmd "${PYTHON[@]}" matchope.pipeline.run_kuairand_matching_gate \
  --data-dir "$KUAIRAND_DATA_DIR" \
  --category-metadata "$KUAIRAND_CATEGORY_METADATA" \
  --output-dir "$KUAIRAND_OUTPUT_DIR"

run_cmd "${PYTHON[@]}" matchope.pipeline.run_kuairand_semisynthetic_zero \
  --data-dir "$KUAIRAND_DATA_DIR" \
  --category-metadata "$KUAIRAND_CATEGORY_METADATA" \
  --output-dir "$EVIDENCE_DIR/kuairand_semisynthetic_realrate" \
  --effect-multiplier-grid 0,0.5,1,2 \
  --reference-matching-effect 0.000966 \
  --repetitions 500 \
  --seed 20260817

run_cmd "${PYTHON[@]}" matchope.pipeline.run_obd_matching_gate \
  --csv "$OBD_CSV" \
  --inverse-temperature-grid 0.25,0.5,1,2,4,8,16 \
  --output-csv "$EVIDENCE_DIR/obd_affinity_gate.csv" \
  --output-path-csv "$EVIDENCE_DIR/obd_affinity_path_audit.csv" \
  --output-json "$EVIDENCE_DIR/obd_affinity_gate.json"

run_cmd "${PYTHON[@]}" matchope.pipeline.run_obd_semisynthetic_power \
  --csv "$OBD_CSV" \
  --output-csv "$EVIDENCE_DIR/obd_semisynthetic_power.csv" \
  --output-json "$EVIDENCE_DIR/obd_semisynthetic_power.json" \
  --effect-grid 0,0.003,0.006,0.009,0.012 \
  --repetitions 500 \
  --seed 20260816 \
  --outcome-smoothing 20

run_cmd "${PYTHON[@]}" matchope.pipeline.plot_matchope_main_results \
  --personalization-tv-matched-csv "$EVIDENCE_DIR/kuairec_personalization_tv_matched_reversals.csv" \
  --common-exposure-csv "$EVIDENCE_DIR/kuairec_common_exposure_controls.csv" \
  --common-exposure-summary-csv "$EVIDENCE_DIR/kuairec_common_exposure_summary.csv" \
  --fixed-marginal-strength-csv "$EVIDENCE_DIR/kuairec_fixed_marginal_strength_path.csv" \
  --fixed-marginal-strength-summary-csv "$EVIDENCE_DIR/kuairec_fixed_marginal_strength_path_summary.csv" \
  --nuisance-coverage-csv "$EVIDENCE_DIR/nuisance_coverage.csv" \
  --support-degradation-csv "$EVIDENCE_DIR/kuairec_popularity_support_degradation.csv" \
  --output-pdf "$EVIDENCE_DIR/main_empirical_results.pdf" \
  --output-png "$EVIDENCE_DIR/main_empirical_results.png" \
  --output-teaser-pdf "$EVIDENCE_DIR/main_empirical_teaser.pdf" \
  --output-strength-path-pdf "$EVIDENCE_DIR/main_empirical_strength_path.pdf" \
  --output-theory-checks-pdf "$EVIDENCE_DIR/main_empirical_theory_checks.pdf"
