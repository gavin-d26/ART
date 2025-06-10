#!/bin/bash
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_CONFIG_PATH="./slug_search/vllm_servers/embedder_logging_config.json" # Assuming script is run from project root
export CUDA_VISIBLE_DEVICES="3"
# Script to launch the VLLM server for the embedding model

# Default values, can be overridden by environment variables or command-line arguments
DEFAULT_MODEL="BAAI/bge-large-en-v1.5"
DEFAULT_PORT="40002"
DEFAULT_TASK="embed"    # Crucial for embedding models
DEFAULT_API_KEY="EMPTY"
DEFAULT_LOG_FILE="embedder_vllm_server.log" # Default log file

# Use environment variables if set, otherwise use defaults
MODEL_NAME="${EMBEDDER_MODEL_NAME:-$DEFAULT_MODEL}"
PORT_NUM="${EMBEDDER_PORT:-$DEFAULT_PORT}"
TASK_TYPE="${EMBEDDER_TASK_TYPE:-$DEFAULT_TASK}"
API_KEY="${EMBEDDER_API_KEY:-$DEFAULT_API_KEY}"
LOG_FILE="${EMBEDDER_LOG_FILE:-$DEFAULT_LOG_FILE}"


echo "Starting VLLM OpenAI-compatible server for EMBEDDER..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT_NUM"
echo "Task: $TASK_TYPE"
echo "Logging to: $LOG_FILE (via VLLM_LOGGING_CONFIG_PATH)"
echo "API Key: (Hidden for security, ensure it's set if required by your model or setup)"

# Adjust --max-model-len, --tensor-parallel-size, --gpu-memory-utilization as needed
# For embedding models, it's important to ensure the task type is correctly handled by VLLM's OpenAI API server
# If the server is to be used for '/v1/embeddings' endpoint, it generally infers this.
# The '--chat-template' or specific model configs might influence behavior if not a standard embedding model.
# BGE models are typically fine. '--enforce-eager' might be needed for some models as seen in datastore.py
# but it's an engine arg, not an api_server arg directly unless passed via other means or implied by model type.
# For OpenAI API server, the embedding task is handled by the /v1/embeddings endpoint.
# The `task="embed"` is an `EngineArgs` parameter if using `LLM` class directly,
# for `openai.api_server` it should correctly serve embeddings if the model is an embedding model.

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$PORT_NUM" \
    --host "0.0.0.0" \
    --served-model-name "$MODEL_NAME" \
    --task "$TASK_TYPE" \
    --gpu-memory-utilization 0.95 \
    # --max-model-len 8192 \ # BGE models can have larger sequence lengths
    # Add other VLLM arguments below as needed, for example:
    # --tensor-parallel-size 1 \
    # --enforce-eager # If needed, though typically not for OpenAI API server for embeddings.

echo "VLLM embedder server script finished. If it launched successfully, it will be running in the background or on this terminal (logs in $LOG_FILE via VLLM logging config)." 