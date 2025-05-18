#!/bin/bash

export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH="./slug_search/vllm_servers/generator_logging_config.json" # Assuming the script is run from the project root
export CUDA_VISIBLE_DEVICES="5"
# Script to launch the VLLM server for the generator model

# Default values, can be overridden by environment variables or command-line arguments
DEFAULT_MODEL="unsloth/Qwen3-4B"
DEFAULT_PORT="40001"
DEFAULT_TASK="generate" # For text generation models
DEFAULT_API_KEY="EMPTY"
DEFAULT_LOG_FILE="generator_vllm_server.log" # Default log file

# Use environment variables if set, otherwise use defaults
MODEL_NAME="${GENERATOR_MODEL_NAME:-$DEFAULT_MODEL}"
PORT_NUM="${GENERATOR_PORT:-$DEFAULT_PORT}"
TASK_TYPE="${GENERATOR_TASK_TYPE:-$DEFAULT_TASK}"
LOG_FILE="${GENERATOR_LOG_FILE:-$DEFAULT_LOG_FILE}"

echo "Starting VLLM OpenAI-compatible server for GENERATOR..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT_NUM"
echo "Task: $TASK_TYPE"
echo "Logging to: $LOG_FILE (via VLLM_LOGGING_CONFIG_PATH)"
echo "API Key: (Hidden for security, ensure it's set if required by your model or setup)" # VLLM uses this for client auth

# Adjust --max-model-len, --tensor-parallel-size, --gpu-memory-utilization as needed
# For models requiring specific trust_remote_code, add --trust-remote-code
# The '--task generate' argument ensures the server is set up for generation tasks (completions, chat).

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$PORT_NUM" \
    --host "0.0.0.0" \
    --served-model-name "$MODEL_NAME" \
    --task "$TASK_TYPE" \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    # Add other VLLM arguments below as needed, for example:
    # --tensor-parallel-size 1 \

echo "VLLM generator server script finished. If it launched successfully, it will be running in the background or on this terminal (logs in $LOG_FILE via VLLM logging config)." 