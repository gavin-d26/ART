#!/bin/bash

# creates a milvus datastore for the hotpotqa dataset

SCRIPT_DIR=$(dirname "$0")
python "$SCRIPT_DIR/datastore.py" \
    --dataset_name "lucadiliello/hotpotqa" \
    --split_name "train" \
    --text_column "context" \
    --milvus_db_path "slug_search/data/milvus_hotpotqa.db" \
    --model "BAAI/bge-large-en-v1.5" \
    --drop_old_db \
    --gpu-memory-utilization 0.4 \
    # --max_docs 1 \

echo "Datastore script execution finished for train split."


python "$SCRIPT_DIR/datastore.py" \
    --dataset_name "lucadiliello/hotpotqa" \
    --split_name "validation" \
    --text_column "context" \
    --milvus_db_path "slug_search/data/milvus_hotpotqa.db" \
    --model "BAAI/bge-large-en-v1.5" \
    --gpu-memory-utilization 0.4 \

echo "Datastore script execution finished for validation split."