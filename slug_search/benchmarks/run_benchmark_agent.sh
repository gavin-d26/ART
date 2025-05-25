#!/bin/bash

# Enhanced Benchmarking Script for Agentic Tool-Calling Pipeline
# Based on Phase 5 & 6 implementation plan - Production Ready System

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Environment and paths
VENV_PATH=".venv/bin/activate"
PROJECT_ROOT="$(pwd)"

# Pipeline configuration - AGENT SPECIFIC
PIPELINE_NAME="AgenticToolCallingPipeline"
GENERATOR_MODEL="unsloth/Qwen3-4B"
GENERATOR_API_URL="http://localhost:40001/v1"
GENERATOR_API_KEY_ENV="GENERATOR_API_KEY"
EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
EMBEDDER_API_URL="http://localhost:40002/v1"
EMBEDDER_API_KEY_ENV="EMBEDDER_API_KEY"

# Agent-specific configuration
AGENT_QUERY_PROMPT_TEMPLATE_KEY="default_query_prompt"  # Key for prompt template in prompts.json
UNKNOWN_TOOL_HANDLING_STRATEGY="break_loop"  # or "error_to_model"
SEARCH_TOOL_TOP_K=5
MAX_AGENT_STEPS=10
AGENT_LLM_SAMPLING_PARAMS='{"temperature": 0.1, "max_tokens": 1000}'

# Dataset configuration
DATASET_NAME="lucadiliello/hotpotqa"
DATASET_SPLIT="validation"
QUERY_COLUMN="question"
ANSWER_COLUMN="answers"

# Output configuration
MILVUS_DB_PATH="slug_search/data/milvus_hotpotqa.db"
RESULTS_OUTPUT_PATH="hotpotqa_agent_benchmark_results.jsonl"
LOG_FILE="agent_benchmarking.log"

# Performance configuration
CONCURRENCY_LIMIT=50  # Lower for agent pipeline due to complexity
# MAX_QUERIES=10  # Set to 10 as requested

# Metrics configuration
AVAILABLE_METRICS="check_answer_correctness_multi_gt;ground_truth_hit_rate;ground_truth_precision;ground_truth_count"
ENABLE_SUMMARY=true

# Summary and result configuration
SUMMARY_OUTPUT_PATH="hotpotqa_agent_benchmark_summary.json"
ENABLE_DETAILED_SUMMARY=true

# Evaluation modes (uncomment one to use)
# EVALUATION_MODE="quick_test"        # Quick health check (limited queries + key metrics)
# EVALUATION_MODE="full_analysis"     # Comprehensive evaluation (all metrics + summary)
# EVALUATION_MODE="retrieval_focus"   # Focus on retrieval performance
# EVALUATION_MODE="generation_focus"  # Focus on generation quality
EVALUATION_MODE="agent_test"          # Agent-specific testing mode

# Feature descriptions - AGENT SPECIFIC
FEATURES=(
    "Agentic tool-calling with search_documents and return_final_answer"
    "Multi-step reasoning with configurable max steps"
    "Tool usage logging and analysis"
    "Ground-truth retrieval analysis across all search calls"
    "Generation quality metrics for final answers"
    "Retrieval performance metrics"
    "Evaluation summary with agent-specific insights"
)

RESULT_FORMAT_ITEMS=(
    "query_id: Unique identifier for traceability"
    "retrieved_chunks: All retrieved chunks from all search_documents calls"
    "tool_log: Complete log of all tool calls with inputs"
    "ground_truth_analysis: Automatic verification across all retrievals"
    "Computed metrics: Generation + retrieval + agent performance"
)

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Activate virtual environment (required for all operations)
echo "Activating virtual environment..."
source "$VENV_PATH"

# Export API keys for local VLLM servers
export GENERATOR_API_KEY="EMPTY"
export EMBEDDER_API_KEY="EMPTY"

echo "Starting enhanced benchmarking for Agentic Tool-Calling Pipeline..."
echo "Pipeline: $PIPELINE_NAME | Dataset: $DATASET_NAME $DATASET_SPLIT split"
echo "Agent Config: Max Steps=$MAX_AGENT_STEPS | Tool Strategy=$UNKNOWN_TOOL_HANDLING_STRATEGY | Search Top-K=$SEARCH_TOOL_TOP_K"
echo -n "Features: "
printf "%s, " "${FEATURES[@]}" | sed 's/, $//'
echo ""

# Configure evaluation based on mode
case "$EVALUATION_MODE" in
    "quick_test")
        SELECTED_METRICS="ground_truth_hit_rate;check_answer_correctness_multi_gt"
        MAX_QUERIES=5
        ENABLE_SUMMARY=true
        echo "🚀 Mode: Quick Test (5 queries, key metrics)"
        ;;
    "full_analysis")
        SELECTED_METRICS="$AVAILABLE_METRICS"
        ENABLE_SUMMARY=true
        echo "🚀 Mode: Full Analysis (all queries, all metrics, detailed summary)"
        ;;
    "retrieval_focus")
        SELECTED_METRICS="ground_truth_hit_rate;ground_truth_precision;ground_truth_count"
        ENABLE_SUMMARY=true
        echo "🚀 Mode: Retrieval Focus (retrieval metrics only)"
        ;;
    "generation_focus")
        SELECTED_METRICS="check_answer_correctness_multi_gt"
        ENABLE_SUMMARY=true
        echo "🚀 Mode: Generation Focus (generation metrics only)"
        ;;
    "agent_test")
        SELECTED_METRICS="$AVAILABLE_METRICS"
        ENABLE_SUMMARY=true
        echo "🚀 Mode: Agent Test (comprehensive agent evaluation with $MAX_QUERIES queries)"
        ;;
    *)
        SELECTED_METRICS="$AVAILABLE_METRICS"
        ENABLE_SUMMARY=true
        echo "🚀 Mode: Default (comprehensive evaluation)"
        ;;
esac

# Build metrics argument
METRICS_ARG=""
if [ "$ENABLE_SUMMARY" = true ]; then
    METRICS_ARG="--metrics $SELECTED_METRICS --summary"
fi

# Build max queries argument if defined
MAX_QUERIES_ARG=""
if [ ! -z "$MAX_QUERIES" ]; then
    MAX_QUERIES_ARG="--max_queries $MAX_QUERIES"
fi

# Build agent sampling params argument
AGENT_SAMPLING_ARG=""
if [ ! -z "$AGENT_LLM_SAMPLING_PARAMS" ]; then
    AGENT_SAMPLING_ARG="--agent_llm_sampling_params"
fi

# Redirect stdout to log file while keeping stderr for progress bar
exec 1> >(tee -a "$LOG_FILE")

echo "$(date): Starting agent benchmarking with enhanced logging and progress tracking..." >&2

python -m slug_search.benchmarks.benchmarking \
    --pipeline_name "$PIPELINE_NAME" \
    --generator_model_name "$GENERATOR_MODEL" \
    --generator_openai_api_base_url "$GENERATOR_API_URL" \
    --generator_openai_api_key_env "$GENERATOR_API_KEY_ENV" \
    --embedding_model_name_on_vllm "$EMBEDDING_MODEL" \
    --embedder_openai_api_base_url "$EMBEDDER_API_URL" \
    --embedder_openai_api_key_env "$EMBEDDER_API_KEY_ENV" \
    --dataset_path "$DATASET_NAME" \
    --dataset_split "$DATASET_SPLIT" \
    --query_column "$QUERY_COLUMN" \
    --answer_column "$ANSWER_COLUMN" \
    --milvus_db_path "$MILVUS_DB_PATH" \
    --results_output_path "$RESULTS_OUTPUT_PATH" \
    --concurrency_limit "$CONCURRENCY_LIMIT" \
    --agent_query_prompt_template_key "$AGENT_QUERY_PROMPT_TEMPLATE_KEY" \
    --unknown_tool_handling_strategy "$UNKNOWN_TOOL_HANDLING_STRATEGY" \
    --search_tool_top_k "$SEARCH_TOOL_TOP_K" \
    --max_agent_steps "$MAX_AGENT_STEPS" \
    $AGENT_SAMPLING_ARG "$AGENT_LLM_SAMPLING_PARAMS" \
    $METRICS_ARG \
    $MAX_QUERIES_ARG 2>&1

echo ""
echo "============================================================"
echo "✅ Enhanced agent benchmarking completed successfully!"
echo "============================================================"
echo "📊 Results: ./$RESULTS_OUTPUT_PATH (with computed metrics and tool logs)"
echo "📝 Logs: ./$LOG_FILE"
if [ "$ENABLE_DETAILED_SUMMARY" = true ] && [ "$ENABLE_SUMMARY" = true ]; then
    echo "📈 Summary: ./$SUMMARY_OUTPUT_PATH (statistical analysis)"
fi
echo ""
echo "🎯 Agent Evaluation Configuration:"
echo "   • Mode: $EVALUATION_MODE"
echo "   • Pipeline: $PIPELINE_NAME"
echo "   • Metrics: $SELECTED_METRICS"
echo "   • Query Limit: $MAX_QUERIES"
echo "   • Summary Enabled: $ENABLE_SUMMARY"
echo ""
echo "🤖 Agent Configuration:"
echo "   • Max Steps: $MAX_AGENT_STEPS"
echo "   • Unknown Tool Strategy: $UNKNOWN_TOOL_HANDLING_STRATEGY"
echo "   • Search Tool Top-K: $SEARCH_TOOL_TOP_K"
echo "   • Query Prompt Template Key: $AGENT_QUERY_PROMPT_TEMPLATE_KEY"
echo "   • Sampling Params: $AGENT_LLM_SAMPLING_PARAMS"
echo ""
echo "🎯 Agent Features included:"
for feature in "${FEATURES[@]}"; do
    echo "   • $feature"
done
echo ""
echo "📋 Agent Result format includes:"
for item in "${RESULT_FORMAT_ITEMS[@]}"; do
    echo "   • $item"
done
echo ""
echo "🔍 For detailed analysis, see the evaluation summary above."
echo ""
echo "💡 Agent Mode Changes:"
echo "   • Quick Test: Set EVALUATION_MODE=\"quick_test\""
echo "   • Retrieval Focus: Set EVALUATION_MODE=\"retrieval_focus\""
echo "   • Generation Focus: Set EVALUATION_MODE=\"generation_focus\""
echo "   • Full Analysis: Set EVALUATION_MODE=\"full_analysis\""
echo "   • Agent Test: Set EVALUATION_MODE=\"agent_test\" (current)"
echo ""
echo "🛠️  Agent Configuration Changes:"
echo "   • Adjust MAX_AGENT_STEPS for longer/shorter reasoning"
echo "   • Change UNKNOWN_TOOL_HANDLING_STRATEGY to 'error_to_model' for error recovery"
echo "   • Modify SEARCH_TOOL_TOP_K for different retrieval amounts"
echo "   • Update AGENT_LLM_SAMPLING_PARAMS for different generation behavior"
echo "============================================================"

# Also log completion to stderr for immediate visibility
echo "$(date): Agent benchmarking completed successfully! Check $LOG_FILE for detailed logs." >&2 