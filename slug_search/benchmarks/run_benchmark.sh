#!/bin/bash

# Enhanced Benchmarking Script with Integrated Metrics
# Based on Phase 5 & 6 implementation plan - Production Ready System

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Environment and paths
VENV_PATH=".venv/bin/activate"
PROJECT_ROOT="$(pwd)"

# Pipeline configuration
PIPELINE_NAME="EmbeddedRAGPipeline"
GENERATOR_MODEL="unsloth/Qwen3-4B"
GENERATOR_API_URL="http://localhost:40001/v1"
GENERATOR_API_KEY_ENV="GENERATOR_API_KEY"
EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
EMBEDDER_API_URL="http://localhost:40002/v1"
EMBEDDER_API_KEY_ENV="EMBEDDER_API_KEY"

# Dataset configuration
DATASET_NAME="lucadiliello/hotpotqa"
DATASET_SPLIT="validation"
QUERY_COLUMN="question"
ANSWER_COLUMN="answers"

# Output configuration
MILVUS_DB_PATH="slug_search/data/milvus_hotpotqa.db"
RESULTS_OUTPUT_PATH="hotpotqa_benchmark_results.jsonl"
LOG_FILE="benchmarking.log"

# Performance configuration
CONCURRENCY_LIMIT=70
# MAX_QUERIES=300  # Uncomment to limit queries for testing

# Metrics configuration
AVAILABLE_METRICS="check_answer_correctness_multi_gt;ground_truth_hit_rate;ground_truth_precision;ground_truth_count"
ENABLE_SUMMARY=true

# Summary and result configuration
SUMMARY_OUTPUT_PATH="hotpotqa_benchmark_summary.json"
ENABLE_DETAILED_SUMMARY=true

# Evaluation modes (uncomment one to use)
# EVALUATION_MODE="quick_test"        # Quick health check (limited queries + key metrics)
# EVALUATION_MODE="full_analysis"     # Comprehensive evaluation (all metrics + summary)
# EVALUATION_MODE="retrieval_focus"   # Focus on retrieval performance
# EVALUATION_MODE="generation_focus"  # Focus on generation quality
EVALUATION_MODE="production"          # Production-ready comprehensive evaluation

# Feature descriptions
FEATURES=(
    "Ground-truth retrieval analysis"
    "Generation quality metrics"
    "Retrieval performance metrics"
    "Evaluation summary with insights"
)

RESULT_FORMAT_ITEMS=(
    "query_id: Unique identifier for traceability"
    "retrieved_chunks: All retrieved chunks with metadata"
    "ground_truth_analysis: Automatic verification"
    "Computed metrics: Generation + retrieval performance"
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

echo "Starting enhanced benchmarking with integrated metrics computation..."
echo "Pipeline: $PIPELINE_NAME | Dataset: $DATASET_NAME $DATASET_SPLIT split"
echo -n "Features: "
printf "%s, " "${FEATURES[@]}" | sed 's/, $//'
echo ""

# Configure evaluation based on mode
case "$EVALUATION_MODE" in
    "quick_test")
        SELECTED_METRICS="ground_truth_hit_rate;check_answer_correctness_multi_gt"
        MAX_QUERIES=50
        ENABLE_SUMMARY=true
        echo "🚀 Mode: Quick Test (50 queries, key metrics)"
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
    "production")
        SELECTED_METRICS="$AVAILABLE_METRICS"
        ENABLE_SUMMARY=true
        echo "🚀 Mode: Production (comprehensive evaluation)"
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
    $METRICS_ARG \
    $MAX_QUERIES_ARG

echo ""
echo "============================================================"
echo "✅ Enhanced benchmarking completed successfully!"
echo "============================================================"
echo "📊 Results: ./$RESULTS_OUTPUT_PATH (with computed metrics)"
echo "📝 Logs: ./$LOG_FILE"
if [ "$ENABLE_DETAILED_SUMMARY" = true ] && [ "$ENABLE_SUMMARY" = true ]; then
    echo "📈 Summary: ./$SUMMARY_OUTPUT_PATH (statistical analysis)"
fi
echo ""
echo "🎯 Evaluation Configuration:"
echo "   • Mode: $EVALUATION_MODE"
echo "   • Metrics: $SELECTED_METRICS"
if [ ! -z "$MAX_QUERIES" ]; then
    echo "   • Query Limit: $MAX_QUERIES"
else
    echo "   • Query Limit: All queries"
fi
echo "   • Summary Enabled: $ENABLE_SUMMARY"
echo ""
echo "🎯 Features included:"
for feature in "${FEATURES[@]}"; do
    echo "   • $feature"
done
echo ""
echo "📋 Result format includes:"
for item in "${RESULT_FORMAT_ITEMS[@]}"; do
    echo "   • $item"
done
echo ""
echo "🔍 For detailed analysis, see the evaluation summary above."
echo ""
echo "💡 Quick Mode Changes:"
echo "   • Quick Test: Set EVALUATION_MODE=\"quick_test\""
echo "   • Retrieval Focus: Set EVALUATION_MODE=\"retrieval_focus\""
echo "   • Generation Focus: Set EVALUATION_MODE=\"generation_focus\""
echo "   • Full Analysis: Set EVALUATION_MODE=\"full_analysis\""
echo "============================================================" 