#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <bon_run_dir> <gds_run_dir> <cand_max> [cand_min] [pool_size]" >&2
  exit 2
fi

BON_RUN_DIR=$1
GDS_RUN_DIR=$2
CAND_MAX=$3
CAND_MIN=${4:-0}
POOL_SIZE=${5:-$((CAND_MAX - CAND_MIN))}

PROBLEMS_DIR=${PROBLEMS_DIR:-problems_v5}
EVAL_JOBS=${EVAL_JOBS:-200}
KLAYOUT_BIN=${KLAYOUT_BIN:-klayout}
REGEN=${REGEN:---regen}

if (( (CAND_MAX - CAND_MIN) % POOL_SIZE != 0 )); then
  echo "CAND_MAX - CAND_MIN must be divisible by POOL_SIZE" >&2
  exit 2
fi

for ((lo=CAND_MIN; lo<CAND_MAX; lo+=POOL_SIZE)); do
  hi=$((lo + POOL_SIZE))
  python score_generated_tests.py \
    "$BON_RUN_DIR" \
    --gds-run "$GDS_RUN_DIR" \
    --problems-dir "$PROBLEMS_DIR" \
    --eval-jobs "$EVAL_JOBS" \
    --klayout-bin "$KLAYOUT_BIN" \
    --cand-min "$lo" \
    --cand-max "$hi" \
    $REGEN
done
