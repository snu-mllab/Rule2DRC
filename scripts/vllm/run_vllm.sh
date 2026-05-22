MODEL_PATH=$1
PORT=${2:-8000}

if [ $MODEL_PATH == "openai/gpt-oss-20b" ]; then
    CONFIG_FILE="scripts/vllm/gpt-oss-20b.yaml"
elif [ $MODEL_PATH == "openai/gpt-oss-120b" ]; then
    CONFIG_FILE="scripts/vllm/gpt-oss-120b.yaml"
elif [ $MODEL_PATH == "qwen/qwen3-30b-a3b-thinking-2507" ]; then
    CONFIG_FILE="scripts/vllm/qwen3-30b-a3b-thinking-2507.yaml"
elif [ $MODEL_PATH == "Qwen/Qwen3-30B-A3B-Instruct-2507" ]; then
    CONFIG_FILE="scripts/vllm/qwen3-30b-a3b-instruct-2507.yaml"
else
    echo "Unknown model: $MODEL_PATH"
    exit 1
fi

vllm serve $MODEL_PATH \
    --config $CONFIG_FILE \
    --api-key $OPENAI_API_KEY \
    --port $PORT
