#!/bin/bash

# Export dummy API keys for local VLLM servers (replace with actual if needed, or manage securely)
export GENERATOR_API_KEY="EMPTY"
export EMBEDDER_API_KEY="EMPTY"

# Ensure this script is executable: chmod +x slug_search/benchmarks/run_benchmark.sh
# Run from the project root directory: ./slug_search/benchmarks/run_benchmark.sh

python -m slug_search.benchmarks.benchmarking \
    --pipeline_name "EmbeddedRAGPipeline" \
    --generator_model_name "unsloth/Qwen3-4B" \
    --generator_openai_api_base_url "http://localhost:40001/v1" \
    --generator_openai_api_key_env "GENERATOR_API_KEY" \
    --embedding_model_name_on_vllm "BAAI/bge-large-en-v1.5" \
    --embedder_openai_api_base_url "http://localhost:40002/v1" \
    --embedder_openai_api_key_env "EMBEDDER_API_KEY" \
    --dataset_path "lucadiliello/hotpotqa" \
    --dataset_split "validation" \
    --query_column "question" \
    --answer_column "answers" \
    --milvus_db_path "slug_search/data/milvus_hotpotqa.db" \
    --max_queries 10 \
    --results_output_path "hotpotqa_benchmark_results.jsonl"

echo "Benchmarking script finished. Check ./benchmarking.log and ./hotpotqa_benchmark_results.jsonl (in project root)" 