#!/bin/bash

# Activate the conda environment
# conda init && conda activate art

# Script to run the Python datastore script with example arguments.
# Adjust the arguments below as needed for your use case.

SCRIPT_DIR=$(dirname "$0")
python "$SCRIPT_DIR/datastore.py" \
    --dataset_name "lucadiliello/hotpotqa" \
    --split_name "train" \
    --text_column "context" \
    --milvus_db_path "slug_search/data/milvus_hotpotqa.db" \
    --model "intfloat/e5-mistral-7b-instruct" \
    --drop_old_db \
    --max_docs 1 \
    --gpu-memory-utilization 0.02 \
    # --metadata_columns id title \
    
    # Add other VLLM or script-specific arguments here if needed
    # Example: --tensor_parallel_size 1
    # Example: --embedding_dim 4096

echo "Datastore script execution finished."
