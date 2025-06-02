#!/bin/bash

# Specify which GPU to use (if needed, but not required for BM25)
# export CUDA_VISIBLE_DEVICES=3

# Enhanced Datastore Creation Script for Elasticsearch + BM25
# Based on Phase 1 implementation plan - BM25/Elasticsearch version

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

# Elasticsearch configuration
ELASTICSEARCH_HOST="http://localhost:40005"
ELASTICSEARCH_INDEX_NAME="hotpotqa_bm25_prod"

# Model/Preprocessing configuration
PREPROCESS_FUNCTION="preprocess_and_chunk_text"

# Split configuration
SPLITS=("train" "validation")
MAX_DOCS=1  # Uncomment to limit documents for testing

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
echo "🚀 Creating Elasticsearch datastore (BM25) for $DATASET_DISPLAY_NAME dataset"
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
    
    # Build arguments for first split (with --drop_old_index)
    DROP_INDEX_ARG=""
    if [ $i -eq 0 ]; then
        DROP_INDEX_ARG="--drop_old_index"
    fi
    
    python "$SCRIPT_DIR/datastore_bm25.py" \
        --dataset_name "$DATASET_NAME" \
        --split_name "$SPLIT" \
        --text_column "$TEXT_COLUMN" \
        --elasticsearch_host "$ELASTICSEARCH_HOST" \
        --elasticsearch_index_name "$ELASTICSEARCH_INDEX_NAME" \
        --preprocess_function "$PREPROCESS_FUNCTION" \
        $DROP_INDEX_ARG \
        $MAX_DOCS_ARG
    
    echo ""
    echo "✅ $SPLIT_UPPER split processing completed!"
    echo ""
done

echo "============================================================"
echo "🎉 Datastore creation finished successfully!"
echo "============================================================"
echo "📁 Elasticsearch Index: $ELASTICSEARCH_INDEX_NAME (Host: $ELASTICSEARCH_HOST)"
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
echo "▶️  Next step: Run benchmarking with your BM25 pipeline."
echo "============================================================" 