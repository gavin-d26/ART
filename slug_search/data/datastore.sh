#!/bin/bash

# Specify which GPU to use (0, 1, 2, etc. or multiple: "0,1")
# export CUDA_VISIBLE_DEVICES=3

# Enhanced Datastore Creation Script with Unique ID System
# Based on Phase 1 implementation plan - Critical ID System Fix

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Environment and paths
VENV_PATH=".venv/bin/activate"
PROJECT_ROOT="$(pwd)"

# Dataset configuration
DATASET_NAME="lucadiliello/hotpotqa"
DATASET_DISPLAY_NAME="HotPotQA"
TEXT_COLUMN="context"
MILVUS_DB_PATH="slug_search/data/milvus_hotpotqa_fixed.db"

# Model configuration
EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
PREPROCESS_FUNCTION="preprocess_and_chunk_text"
GPU_MEMORY_UTILIZATION="0.90"

# Split configuration
SPLITS=("train" "validation")
# MAX_DOCS=1  # Uncomment to limit documents for testing

# Derived dataset identifier (for ID generation)
DATASET_IDENTIFIER=$(echo "$DATASET_NAME" | sed 's/.*\///')

# Enhanced features
ENHANCED_FEATURES=(
    "Unique ID generation across splits"
    "Rich metadata structure for evaluation"
    "Ground-truth verification support"
    "Phase 1 implementation (ID collision fix)"
)

# Metadata structure
METADATA_FIELDS=(
    "chunk_id: Unique across all splits"
    "original_doc_id: Document identifier"
    "split_name: train/validation"
    "dataset_name: $DATASET_NAME"
    "derived_dataset_identifier: $DATASET_IDENTIFIER"
)

# Evaluation capabilities
EVALUATION_FEATURES=(
    "Ground-truth verification"
    "Retrieval tracking"
    "Cross-split analysis"
)

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Activate virtual environment (required for all operations)
echo "Activating virtual environment..."
source "$VENV_PATH"

echo "============================================================"
echo "🚀 Creating Milvus datastore for $DATASET_DISPLAY_NAME dataset"
echo "============================================================"
echo "✨ Enhanced Features:"
for feature in "${ENHANCED_FEATURES[@]}"; do
    echo "   • $feature"
done
echo ""

# Store the script directory for relative path resolution
SCRIPT_DIR=$(dirname "$0")

# Build max docs argument if defined
MAX_DOCS_ARG=""
if [ ! -z "$MAX_DOCS" ]; then
    MAX_DOCS_ARG="--max_docs $MAX_DOCS"
fi

# Process each split
for i in "${!SPLITS[@]}"; do
    SPLIT="${SPLITS[$i]}"
    SPLIT_UPPER=$(echo "$SPLIT" | tr '[:lower:]' '[:upper:]')
    ID_FORMAT="${DATASET_IDENTIFIER}_${SPLIT}_{id}_chunk_{index}"
    
    echo "📊 Processing $SPLIT_UPPER split..."
    echo "   • Dataset: $DATASET_NAME"
    echo "   • Split: $SPLIT"
    echo "   • ID format: $ID_FORMAT"
    echo ""
    
    # Build arguments for first split (with --drop_old_db)
    DROP_DB_ARG=""
    if [ $i -eq 0 ]; then
        DROP_DB_ARG="--drop_old_db"
    fi
    
    python "$SCRIPT_DIR/datastore.py" \
        --dataset_name "$DATASET_NAME" \
        --split_name "$SPLIT" \
        --text_column "$TEXT_COLUMN" \
        --milvus_db_path "$MILVUS_DB_PATH" \
        --model "$EMBEDDING_MODEL" \
        --preprocess_function "$PREPROCESS_FUNCTION" \
        $DROP_DB_ARG \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        $MAX_DOCS_ARG
    
    echo ""
    echo "✅ $SPLIT_UPPER split processing completed!"
    echo ""
done

# After all splits are processed, create a training DB copy
TRAINING_DB_PATH="${MILVUS_DB_PATH%.*}_training.${MILVUS_DB_PATH##*.}"

if [ -f "$TRAINING_DB_PATH" ]; then
    echo "Deleting existing training DB at $TRAINING_DB_PATH..."
    rm "$TRAINING_DB_PATH"
fi

# Pauses script execution for 10 seconds to ensure previous operations complete before copying the database
sleep 10
if [ -f "$MILVUS_DB_PATH.lock" ]; then
    rm "$MILVUS_DB_PATH.lock"
fi

cp "$MILVUS_DB_PATH" "$TRAINING_DB_PATH"
echo "✅ Training DB created at: $TRAINING_DB_PATH"

echo "============================================================"
echo "🎉 Datastore creation finished successfully!"
echo "============================================================"
echo "📁 Database location: $MILVUS_DB_PATH"
echo ""
echo "🔧 Enhanced metadata structure includes:"
for field in "${METADATA_FIELDS[@]}"; do
    echo "   • $field"
done
echo ""
echo "🎯 Ready for evaluation with:"
for capability in "${EVALUATION_FEATURES[@]}"; do
    echo "   • $capability"
done
echo ""
echo "▶️  Next step: Run benchmarking with:"
echo "   ./slug_search/benchmarks/run_benchmark.sh"
echo "============================================================"