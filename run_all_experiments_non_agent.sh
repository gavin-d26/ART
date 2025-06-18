#!/bin/bash

# Implementation of the automated benchmarking script as specified in plan.md
# This script automates the execution of slug_search/benchmarks/benchmarking.py
# for multiple pipeline configurations and top_k values.

set -e  # Exit immediately if any command fails

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================

# Virtual Environment
VENV_PATH=".venv/bin/activate"

# Pipelines to test
PIPELINES_TO_RUN=("EmbeddedRAGPipeline" "NaiveGenerationPipeline")

# Top-k values to test for retrieval pipelines
TOP_K_VALUES_TO_TRY=(5)

# Timestamp format for main output directory
MAIN_RUN_TIMESTAMP_FORMAT="+%Y%m%d_%H%M%S"

# Generator Configuration
GENERATOR_MODEL="unsloth/Qwen2.5-3B-Instruct"
GENERATOR_API_URL="http://localhost:40001/v1"
GENERATOR_API_KEY_ENV="GENERATOR_API_KEY"

# Embedder Configuration
EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
EMBEDDER_API_URL="http://localhost:40002/v1"
EMBEDDER_API_KEY_ENV="EMBEDDER_API_KEY"

# Dataset Configuration
DATASET_NAME="lucadiliello/hotpotqa"
DATASET_SPLIT="validation"
QUERY_COLUMN="question"
ANSWER_COLUMN="answers"

# Milvus Configuration
MILVUS_DB_PATH="slug_search/data/milvus_hotpotqa_fixed.db"

# Benchmarking Script Behavior
CONCURRENCY_LIMIT=70
AGENT_CONCURRENCY_LIMIT=10
MAX_QUERIES=400  # Leave empty for no limit, or set a number for testing
METRICS_TO_COMPUTE="check_answer_correctness_multi_gt;ground_truth_hit_rate;ground_truth_precision;ground_truth_count"
ENABLE_SUMMARY=true

# Agent-Specific Defaults
AGENT_QUERY_PROMPT_TEMPLATE_KEY="default_query_prompt_2"
AGENT_SYSTEM_PROMPT_KEY="qwen_2.5_3b_instruct_system_prompt"
DEFAULT_AGENT_MAX_STEPS=5
AGENT_LLM_SAMPLING_PARAMS=""
DEFAULT_AGENT_SEARCH_TOOL_TOP_K=3
DEFAULT_RAG_TOP_K_RETRIEVER=3

# Agent parameter variations
AGENT_MAX_STEPS_TO_TRY=(6)

# ============================================================================
# INITIALIZATION
# ============================================================================

echo "Starting automated benchmarking script..."

# Activate virtual environment
echo "Activating virtual environment..."
source "${VENV_PATH}"

# Export dummy API keys if they're not already set
if [ -z "${!GENERATOR_API_KEY_ENV}" ]; then
    export GENERATOR_API_KEY="EMPTY"
    echo "Set dummy GENERATOR_API_KEY"
fi

if [ -z "${!EMBEDDER_API_KEY_ENV}" ]; then
    export EMBEDDER_API_KEY="EMPTY"
    echo "Set dummy EMBEDDER_API_KEY"
fi

# Create main output directory with timestamp
MAIN_RUN_TIMESTAMP=$(date "${MAIN_RUN_TIMESTAMP_FORMAT}")
MAIN_OUTPUT_DIR="validation_run_${MAIN_RUN_TIMESTAMP}"
mkdir -p "${MAIN_OUTPUT_DIR}"
echo "Main output directory: ${MAIN_OUTPUT_DIR}"

# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================

for pipeline_name in "${PIPELINES_TO_RUN[@]}"; do
    echo "Processing pipeline: ${pipeline_name}"
    
    # Set concurrency limit based on pipeline type
    if [ "${pipeline_name}" == "AgenticToolCallingPipeline" ]; then
        CURRENT_CONCURRENCY_LIMIT="${AGENT_CONCURRENCY_LIMIT}"
    else
        CURRENT_CONCURRENCY_LIMIT="${CONCURRENCY_LIMIT}"
    fi
    
    # Construct base arguments common to all benchmarking runs
    current_base_cmd_args=(
        "--generator_model_name" "${GENERATOR_MODEL}"
        "--generator_openai_api_base_url" "${GENERATOR_API_URL}"
        "--generator_openai_api_key_env" "${GENERATOR_API_KEY_ENV}"
        "--embedding_model_name_on_vllm" "${EMBEDDING_MODEL}"
        "--embedder_openai_api_base_url" "${EMBEDDER_API_URL}"
        "--embedder_openai_api_key_env" "${EMBEDDER_API_KEY_ENV}"
        "--dataset_path" "${DATASET_NAME}"
        "--dataset_split" "${DATASET_SPLIT}"
        "--query_column" "${QUERY_COLUMN}"
        "--answer_column" "${ANSWER_COLUMN}"
        "--milvus_db_path" "${MILVUS_DB_PATH}"
        "--concurrency_limit" "${CURRENT_CONCURRENCY_LIMIT}"
    )

    # Add optional arguments based on configuration
    if [ "${ENABLE_SUMMARY}" = true ] && [ -n "${METRICS_TO_COMPUTE}" ]; then
        current_base_cmd_args+=("--metrics" "${METRICS_TO_COMPUTE}" "--summary")
    fi

    if [ -n "${MAX_QUERIES}" ]; then
        current_base_cmd_args+=("--max_queries" "${MAX_QUERIES}")
    fi
    
    # Handle different pipeline types
    if [ "${pipeline_name}" == "NaiveGenerationPipeline" ]; then
        # NaiveGenerationPipeline doesn't use retrieval, so run once without top_k variations
        EXPERIMENT_NAME="${pipeline_name}"
        EXPERIMENT_DIR="${MAIN_OUTPUT_DIR}/${EXPERIMENT_NAME}"
        mkdir -p "${EXPERIMENT_DIR}"
        
        CURRENT_RESULTS_PATH="${EXPERIMENT_DIR}/results.jsonl"
        CURRENT_LOG_FILE="${EXPERIMENT_DIR}/benchmarking.log"

        cmd_args=(
            "python" "-m" "slug_search.benchmarks.benchmarking"
            "--pipeline_name" "${pipeline_name}"
            "--results_output_path" "${CURRENT_RESULTS_PATH}"
            "${current_base_cmd_args[@]}"
        )
        
        echo "Running: ${EXPERIMENT_NAME}"
        "${cmd_args[@]}" > "${CURRENT_LOG_FILE}" 2>&1
        echo "Completed: ${EXPERIMENT_NAME}. Log: ${CURRENT_LOG_FILE}"
        
    elif [ "${pipeline_name}" == "AgenticToolCallingPipeline" ]; then
        # Agent pipeline with multiple parameter variations
        for top_k_value in "${TOP_K_VALUES_TO_TRY[@]}"; do
            for agent_max_step in "${AGENT_MAX_STEPS_TO_TRY[@]}"; do
                EXPERIMENT_NAME="${pipeline_name}_topK_${top_k_value}_steps_${agent_max_step}"
                EXPERIMENT_DIR="${MAIN_OUTPUT_DIR}/${EXPERIMENT_NAME}"
                mkdir -p "${EXPERIMENT_DIR}"

                CURRENT_RESULTS_PATH="${EXPERIMENT_DIR}/results.jsonl"
                CURRENT_LOG_FILE="${EXPERIMENT_DIR}/benchmarking.log"
                
                pipeline_specific_args=(
                    "--search_tool_top_k" "${top_k_value}"
                    "--agent_query_prompt_template_key" "${AGENT_QUERY_PROMPT_TEMPLATE_KEY}"
                    "--agent_system_prompt_key" "${AGENT_SYSTEM_PROMPT_KEY}"
                    "--max_agent_steps" "${agent_max_step}"
                    "--agent_llm_sampling_params" "${AGENT_LLM_SAMPLING_PARAMS}"
                    "--top_k_retriever" "${DEFAULT_RAG_TOP_K_RETRIEVER}"
                )

                cmd_args=(
                    "python" "-m" "slug_search.benchmarks.benchmarking"
                    "--pipeline_name" "${pipeline_name}"
                    "--results_output_path" "${CURRENT_RESULTS_PATH}"
                    "${current_base_cmd_args[@]}"
                    "${pipeline_specific_args[@]}"
                )

                echo "Running: ${EXPERIMENT_NAME}"
                "${cmd_args[@]}" > "${CURRENT_LOG_FILE}" 2>&1
                echo "Completed: ${EXPERIMENT_NAME}. Log: ${CURRENT_LOG_FILE}"
            done
        done
        
    else
        # For other pipelines that use top_k (EmbeddedRAGPipeline, EmbedderRetrieverPipeline)
        for top_k_value in "${TOP_K_VALUES_TO_TRY[@]}"; do
            EXPERIMENT_NAME="${pipeline_name}_topK_${top_k_value}"
            EXPERIMENT_DIR="${MAIN_OUTPUT_DIR}/${EXPERIMENT_NAME}"
            mkdir -p "${EXPERIMENT_DIR}"

            CURRENT_RESULTS_PATH="${EXPERIMENT_DIR}/results.jsonl"
            CURRENT_LOG_FILE="${EXPERIMENT_DIR}/benchmarking.log"
            
            pipeline_specific_args=(
                "--top_k_retriever" "${top_k_value}"
                "--search_tool_top_k" "${DEFAULT_AGENT_SEARCH_TOOL_TOP_K}"
            )
            
            cmd_args=(
                "python" "-m" "slug_search.benchmarks.benchmarking"
                "--pipeline_name" "${pipeline_name}"
                "--results_output_path" "${CURRENT_RESULTS_PATH}"
                "${current_base_cmd_args[@]}"
                "${pipeline_specific_args[@]}"
            )

            echo "Running: ${EXPERIMENT_NAME}"
            "${cmd_args[@]}" > "${CURRENT_LOG_FILE}" 2>&1
            echo "Completed: ${EXPERIMENT_NAME}. Log: ${CURRENT_LOG_FILE}"
        done
    fi
done

echo "All experiments finished. Results are in ${MAIN_OUTPUT_DIR}"
echo "Summary of experiments run:"
find "${MAIN_OUTPUT_DIR}" -name "results.jsonl" | sort 