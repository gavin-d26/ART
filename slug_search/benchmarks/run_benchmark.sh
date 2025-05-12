#!/bin/bash

# Export dummy API keys for local VLLM servers (replace with actual if needed, or manage securely)
export GENERATOR_VLLM_API_KEY="EMPTY"
export EMBEDDER_VLLM_API_KEY="EMPTY"

# Ensure this script is executable: chmod +x slug_search/benchmarks/run_benchmark.sh
# Run from the project root directory: ./slug_search/benchmarks/run_benchmark.sh

python slug_search/benchmarks/benchmarking.py \
    --pipeline_name "build_vllm_embedded_rag_pipeline" \
    --generator_model_name "mistralai/Mistral-7B-Instruct-v0.1" \
    --generator_openai_api_base_url "http://localhost:8000/v1" \
    --generator_openai_api_key_env "GENERATOR_VLLM_API_KEY" \
    --embedding_model_name_on_vllm "BAAI/bge-large-en-v1.5" \
    --embedder_openai_api_base_url "http://localhost:8001/v1" \
    --embedder_openai_api_key_env "EMBEDDER_VLLM_API_KEY" \
    --dataset_path "lucadiliello/hotpotqa" \
    --dataset_split "train" \
    --query_column "question" \
    --answer_column "answer" \
    --milvus_db_path "slug_search/data/milvus_hotpotqa.db" \
    --max_queries 10 \
    --results_output_path "hotpotqa_benchmark_results.csv"

echo "Benchmarking script finished. Check ./benchmarking.log and ./hotpotqa_benchmark_results.csv (in project root)" 