#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <bon_run_dir> <score_method> <cand_min> <cand_max> <pool_size> [gds_run_dir]" >&2
  exit 2
fi

BON_RUN_DIR=$1
SCORE_METHOD=$2
CAND_MIN=$3
CAND_MAX=$4
POOL_SIZE=$5
GDS_RUN_DIR=${6:-}

case "$SCORE_METHOD" in
  llm_judge)
    SUITE_ID=${SUITE_ID:-nosuite}
    SCORE_FILE=${SCORE_FILE:-"scores/llm_judge__${SUITE_ID}__{tag}.json"}
    ;;
  generated_tests)
    if [[ -z "$GDS_RUN_DIR" ]]; then
      echo "gds_run_dir is required for generated_tests" >&2
      exit 2
    fi
    SUITE_ID=${SUITE_ID:-$(basename "$GDS_RUN_DIR")}
    SCORE_FILE=${SCORE_FILE:-"scores/generated_tests__${SUITE_ID}.json"}
    ;;
  generated_tests_s_star|generated_tests_codemonkey_select|generated_tests_ours)
    if [[ -z "$GDS_RUN_DIR" ]]; then
      echo "gds_run_dir is required for $SCORE_METHOD" >&2
      exit 2
    fi
    SUITE_ID=${SUITE_ID:-$(basename "$GDS_RUN_DIR")}
    SCORE_FILE=${SCORE_FILE:-"scores/${SCORE_METHOD}__${SUITE_ID}__{tag}.json"}
    ;;
  *)
    echo "Unknown score method: $SCORE_METHOD" >&2
    exit 2
    ;;
esac

python bon_scores.py \
  "$BON_RUN_DIR" \
  --pool-size "$POOL_SIZE" \
  --score-method "$SCORE_METHOD" \
  --score-file "$SCORE_FILE" \
  --cand-min "$CAND_MIN" \
  --cand-max "$CAND_MAX"
