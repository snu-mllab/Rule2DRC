#!/usr/bin/env bash
set -euo pipefail

BON_N=${1:-${BON_N:-30}}
RUN_TS=${RUN_TS:-$(TZ=Asia/Seoul date +%y%m%d_%H%M%S)}

MODEL=${MODEL:-openai/gpt-oss-120b}
MODEL_NAME=${MODEL##*/}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
API_KEY=${API_KEY:-${OPENAI_API_KEY:-}}
PROBLEMS_DIR=${PROBLEMS_DIR:-problems}
DOC_PATH=${DOC_PATH:-refs/klayout_docs.txt}
JOBS=${JOBS:-200}
EVAL_JOBS=${EVAL_JOBS:-100}
REASONING_EFFORT=${REASONING_EFFORT:-medium}
OUTPUT_DIR=${OUTPUT_DIR:-bon${BON_N}_${MODEL_NAME}}

python bon.py \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --api-key "$API_KEY" \
  --output-dir "$OUTPUT_DIR" \
  --problems-dir "$PROBLEMS_DIR" \
  --reasoning-effort "$REASONING_EFFORT" \
  --ctx-mode ic \
  --doc-path "$DOC_PATH" \
  --bon-n "$BON_N" \
  --jobs "$JOBS" \
  --eval-jobs "$EVAL_JOBS" \
  --run-ts "$RUN_TS"
