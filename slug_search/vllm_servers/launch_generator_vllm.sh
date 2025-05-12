#!/bin/bash

# Script to launch the VLLM server for the generator model

# Default values, can be overridden by environment variables or command-line arguments
DEFAULT_MODEL="mistralai/Mistral-7B-Instruct-v0.1"
DEFAULT_PORT="8000"
DEFAULT_API_KEY="EMPTY" # As used in benchmarking scripts
DEFAULT_TASK="generate" # For text generation models

# Use environment variables if set, otherwise use defaults
MODEL_NAME="${GENERATOR_MODEL_NAME:-$DEFAULT_MODEL}"
PORT_NUM="${GENERATOR_PORT:-$DEFAULT_PORT}"
API_KEY="${GENERATOR_VLLM_API_KEY:-$DEFAULT_API_KEY}" # Corresponds to --api-key in vLLM
TASK_TYPE="${GENERATOR_TASK_TYPE:-$DEFAULT_TASK}"

echo "Starting VLLM OpenAI-compatible server for GENERATOR..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT_NUM"
echo "Task: $TASK_TYPE"
echo "API Key: (Hidden for security, ensure it's set if required by your model or setup)" # VLLM uses this for client auth

# Adjust --max-model-len, --tensor-parallel-size, --gpu-memory-utilization as needed
# For models requiring specific trust_remote_code, add --trust-remote-code
# The '--task generate' argument ensures the server is set up for generation tasks (completions, chat).

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$PORT_NUM" \
    --api-key "$API_KEY" \
    --host "0.0.0.0" \
    --served-model-name "generator-model" \
    --task "$TASK_TYPE" # Explicitly set task for robustness
    # Add other VLLM arguments below as needed, for example:
    # --tensor-parallel-size 1 \
    # --max-model-len 4096 \
    # --gpu-memory-utilization 0.90

echo "VLLM generator server script finished. If it launched successfully, it will be running in the background or on this terminal." 