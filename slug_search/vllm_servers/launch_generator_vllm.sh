#!/bin/bash

export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH="./slug_search/vllm_servers/generator_logging_config.json" # Assuming the script is run from the project root
export CUDA_VISIBLE_DEVICES="4"
# Script to launch the VLLM server for the generator model

# Default values, can be overridden by environment variables or command-line arguments
DEFAULT_MODEL="unsloth/Qwen2.5-3B-Instruct"
DEFAULT_PORT="40001"
DEFAULT_TASK="generate" # For text generation models
DEFAULT_API_KEY="EMPTY"
DEFAULT_LOG_FILE="generator_vllm_server.log" # Default log file

# LoRA adapter configuration - optional
DEFAULT_ENABLE_LORA="false"
DEFAULT_LORA_MODULES=""  # Format: "name1=/path/to/adapter1 name2=/path/to/adapter2"
DEFAULT_MAX_LORAS="1"
DEFAULT_MAX_LORA_RANK="16"
DEFAULT_LORA_DTYPE="auto"

# Use environment variables if set, otherwise use defaults
MODEL_NAME="${GENERATOR_MODEL_NAME:-$DEFAULT_MODEL}"
PORT_NUM="${GENERATOR_PORT:-$DEFAULT_PORT}"
TASK_TYPE="${GENERATOR_TASK_TYPE:-$DEFAULT_TASK}"
LOG_FILE="${GENERATOR_LOG_FILE:-$DEFAULT_LOG_FILE}"

# LoRA configuration
ENABLE_LORA="${GENERATOR_ENABLE_LORA:-$DEFAULT_ENABLE_LORA}"
LORA_MODULES="${GENERATOR_LORA_MODULES:-$DEFAULT_LORA_MODULES}"
MAX_LORAS="${GENERATOR_MAX_LORAS:-$DEFAULT_MAX_LORAS}"
MAX_LORA_RANK="${GENERATOR_MAX_LORA_RANK:-$DEFAULT_MAX_LORA_RANK}"
LORA_DTYPE="${GENERATOR_LORA_DTYPE:-$DEFAULT_LORA_DTYPE}"

echo "Starting VLLM OpenAI-compatible server for GENERATOR..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT_NUM"
echo "Task: $TASK_TYPE"
echo "LoRA Enabled: $ENABLE_LORA"
if [ "$ENABLE_LORA" = "true" ]; then
    echo "LoRA Modules: $LORA_MODULES"
    echo "Max LoRAs: $MAX_LORAS"
    echo "Max LoRA Rank: $MAX_LORA_RANK"
    echo "LoRA Data Type: $LORA_DTYPE"
fi
echo "Logging to: $LOG_FILE (via VLLM_LOGGING_CONFIG_PATH)"
echo "API Key: (Hidden for security, ensure it's set if required by your model or setup)" # VLLM uses this for client auth

# Build the command with optional LoRA arguments
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $PORT_NUM \
    --host 0.0.0.0 \
    --served-model-name $MODEL_NAME \
    --task $TASK_TYPE \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes"

# Add LoRA arguments if enabled
if [ "$ENABLE_LORA" = "true" ]; then
    VLLM_CMD="$VLLM_CMD --enable-lora"
    VLLM_CMD="$VLLM_CMD --max-loras $MAX_LORAS"
    VLLM_CMD="$VLLM_CMD --max-lora-rank $MAX_LORA_RANK"
    VLLM_CMD="$VLLM_CMD --lora-dtype $LORA_DTYPE"
    
    # Add LoRA modules if specified
    if [ -n "$LORA_MODULES" ]; then
        VLLM_CMD="$VLLM_CMD --lora-modules $LORA_MODULES"
    fi
fi

# Execute the command
eval $VLLM_CMD
    # --enable-reasoning \
    # --reasoning-parser deepseek_r1 \
    # Add other VLLM arguments below as needed, for example:
    # --tensor-parallel-size 1 \

echo "VLLM generator server script finished. If it launched successfully, it will be running in the background or on this terminal (logs in $LOG_FILE via VLLM logging config)." 