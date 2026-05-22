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

MODEL=${MODEL:-openai/gpt-oss-120b}
BASE_URL=${BASE_URL:-https://api.openai.com/v1}
API_KEY=${API_KEY:-${OPENAI_API_KEY:-}}
PROBLEMS_DIR=${PROBLEMS_DIR:-problems_v5}
DOC_PATH=${DOC_PATH:-refs/klayout_docs.txt}
JUDGE_JOBS=${JUDGE_JOBS:-200}
TARGET_ADDITIONAL_TEST_CASES=${TARGET_ADDITIONAL_TEST_CASES:-8}
REASONING_EFFORT=${REASONING_EFFORT:-medium}
REGEN=${REGEN:---regen}

if (( (CAND_MAX - CAND_MIN) % POOL_SIZE != 0 )); then
  echo "CAND_MAX - CAND_MIN must be divisible by POOL_SIZE" >&2
  exit 2
fi

for ((lo=CAND_MIN; lo<CAND_MAX; lo+=POOL_SIZE)); do
  hi=$((lo + POOL_SIZE))
  python score_generated_tests_s_star.py \
    "$BON_RUN_DIR" \
    --gds-run "$GDS_RUN_DIR" \
    --problems-dir "$PROBLEMS_DIR" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --reasoning-effort "$REASONING_EFFORT" \
    --ctx-mode ic \
    --doc-path "$DOC_PATH" \
    --cand-min "$lo" \
    --cand-max "$hi" \
    --judge-jobs "$JUDGE_JOBS" \
    --target-additional-test-cases "$TARGET_ADDITIONAL_TEST_CASES" \
    $REGEN
done
