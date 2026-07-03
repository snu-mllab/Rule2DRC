#!/usr/bin/env bash
# Evaluate a model on the Rule2DRC public set for the leaderboard.
#
# Generates one DRC script per problem in two settings (KLayout docs in
# context, and no context), evaluates them with KLayout, and prints the
# success/error rates plus a ready-to-paste leaderboard YAML row.
#
# Usage:
#   export OPENROUTER_API_KEY=sk-or-...
#   bash scripts/leaderboard/run_eval.sh <model> [reasoning-effort]
#
# Examples:
#   bash scripts/leaderboard/run_eval.sh openai/gpt-oss-120b medium
#   bash scripts/leaderboard/run_eval.sh qwen/qwen3-30b-a3b-instruct-2507
#
# Environment overrides:
#   BASE_URL      API endpoint (default: https://openrouter.ai/api/v1)
#   API_KEY       API key (default: $OPENROUTER_API_KEY)
#   JOBS          parallel workers (default: 16)
#   PROBLEMS_DIR  problems directory (default: problems)
#   MAX_NEW_TOKENS max completion tokens per call (default: 16384)
set -euo pipefail

MODEL=${1:?usage: run_eval.sh <model> [reasoning-effort]}
EFFORT=${2:-}

BASE_URL=${BASE_URL:-https://openrouter.ai/api/v1}
API_KEY=${API_KEY:-${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (or API_KEY)}}
JOBS=${JOBS:-16}
PROBLEMS_DIR=${PROBLEMS_DIR:-problems}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}

TS=$(date +%y%m%d_%H%M%S)
TAG="lb_$(echo "$MODEL" | tr '/:' '__')"
PROBLEMS_NAME=$(basename "$PROBLEMS_DIR")

COMMON=(
  --model "$MODEL"
  --base-url "$BASE_URL"
  --api-key "$API_KEY"
  --problems-dir "$PROBLEMS_DIR"
  --bon-n 1
  --jobs "$JOBS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --run-ts "$TS"
)
if [[ -n "$EFFORT" ]]; then
  COMMON+=(--reasoning-effort "$EFFORT")
fi

echo "== [1/3] Generating + evaluating WITH docs in context =="
python bon.py "${COMMON[@]}" \
  --ctx-mode ic --doc-path refs/klayout_docs.txt \
  --output-dir "$TAG"

echo "== [2/3] Generating + evaluating WITHOUT context =="
python bon.py "${COMMON[@]}" \
  --ctx-mode none \
  --output-dir "${TAG}_noctx"

WITH_DIR="out_drc/${PROBLEMS_NAME}/${TAG}_ic_klayout_docs_${TS}"
NOCTX_DIR="out_drc/${PROBLEMS_NAME}/${TAG}_noctx_${TS}"

echo "== [3/3] Summarizing =="
python scripts/leaderboard/summarize.py \
  --model "$MODEL" \
  --with-context "$WITH_DIR" \
  --without-context "$NOCTX_DIR"
