#!/bin/bash

TEMPLATE_PATH=$1

export BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY=$OPENAI_UMASS_API_KEY
# Remove the .txt extension and just use the filename, remove parent folder
TEMPLATE_KEYWORD=${TEMPLATE_PATH%.txt}
TEMPLATE_KEYWORD=${TEMPLATE_KEYWORD##*/}
# Array of models to run
OPENAI_MODELS=(
    "gpt-4o"
    # "gpt-3.5-turbo"
    # "o3-mini"
)

# Function to run inference for a single model
run_inference() {
    local model=$1
    echo "Running inference for model: $model"
    LOG_DIR="logs/${model//\//_}"
    mkdir -p $LOG_DIR

    python prompt_variants_end_to_end.py \
        --model "$model" \
        --template-path "$TEMPLATE_PATH" \
        --max-tokens 4096 \
        &> $LOG_DIR/${TEMPLATE_KEYWORD}.log &
}

# Run all models in parallel
for model in "${OPENAI_MODELS[@]}"; do
    run_inference "$model" &
done
wait

# Run for the together ai models
TOGETHER_AI_MODELS=(
    "meta/llama3-70b-instruct"
    "mistralai/mixtral-8x22b-instruct-v0.1"
    "deepseek-ai/DeepSeek-R1"
)

export OPENAI_API_KEY=$TOGETHER_AI_API_KEY
export BASE_URL="https://api.together.xyz/v1"
for model in "${TOGETHER_AI_MODELS[@]}"; do
    echo "Running inference for model: $model"
    LOG_DIR="logs/${model//\//_}"
    mkdir -p $LOG_DIR
    run_inference "$model" &> $LOG_DIR/${TEMPLATE_KEYWORD}.log &
done
wait