#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <bon_run_dir> <gds_run_dir> <cand_min> <cand_max> <pool_size>" >&2
  exit 2
fi

BON_RUN_DIR=$1
GDS_RUN_DIR=$2
CAND_MIN=$3
CAND_MAX=$4
POOL_SIZE=$5

MODEL=${MODEL:-openai/gpt-oss-120b}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
API_KEY=${API_KEY:-${OPENAI_API_KEY:-}}
PROBLEMS_DIR=${PROBLEMS_DIR:-problems}
DOC_PATH=${DOC_PATH:-refs/klayout_docs.txt}
JOBS=${JOBS:-200}
KLAYOUT_BIN=${KLAYOUT_BIN:-klayout}
PYTHON_BIN=${PYTHON_BIN:-python}
EXTRA_BUDGET=${EXTRA_BUDGET:-8}
EARLY_STOP=${EARLY_STOP:-1}
N_REPS=${N_REPS:-3}
REASONING_EFFORT=${REASONING_EFFORT:-medium}
REGEN=${REGEN:---regen}

if (( (CAND_MAX - CAND_MIN) % POOL_SIZE != 0 )); then
  echo "CAND_MAX - CAND_MIN must be divisible by POOL_SIZE" >&2
  exit 2
fi

for ((lo=CAND_MIN; lo<CAND_MAX; lo+=POOL_SIZE)); do
  hi=$((lo + POOL_SIZE))
  python score_generated_tests_ours.py \
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
    --jobs "$JOBS" \
    --klayout-bin "$KLAYOUT_BIN" \
    --python-bin "$PYTHON_BIN" \
    --extra-budget "$EXTRA_BUDGET" \
    --early-stop "$EARLY_STOP" \
    --n-reps "$N_REPS" \
    --method-key generated_tests_ours \
    $REGEN
done
