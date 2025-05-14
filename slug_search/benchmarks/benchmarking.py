import argparse
import asyncio
import logging
import os
import pandas as pd
from datasets import load_dataset
from importlib import import_module
import json

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmarking.log")],
)
logger = logging.getLogger(__name__)

TIMEOUT = 60.0 * 20


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking script for RAG setups.")
    parser.add_argument(
        "--pipeline_name",
        type=str,
        required=True,
        help="Name of the pipeline function in pipelines.py to use (e.g., 'build_vllm_embedded_rag_pipeline').",
    )
    # Generator VLLM Server Arguments
    parser.add_argument(
        "--generator_model_name",
        type=str,
        required=True,
        help="Model name for the generator (e.g., 'gpt-3.5-turbo' or a VLLM compatible model name).",
    )
    parser.add_argument(
        "--generator_openai_api_base_url",
        type=str,
        required=True,
        help="Base URL for the Generator VLLM OpenAI-compatible API (e.g., 'http://localhost:8000/v1').",
    )
    parser.add_argument(
        "--generator_openai_api_key_env",
        type=str,
        required=True,
        help="Name of the environment variable holding the API key for the Generator VLLM API.",
    )
    # Embedder VLLM Server Arguments
    parser.add_argument(
        "--embedding_model_name_on_vllm",  # Renamed from retriever_model
        type=str,
        required=True,
        help="Model name for query embeddings served by VLLM (e.g., 'BAAI/bge-large-en-v1.5').",
    )
    parser.add_argument(
        "--embedder_openai_api_base_url",
        type=str,
        required=True,
        help="Base URL for the Embedder VLLM OpenAI-compatible API (e.g., 'http://localhost:8001/v1').",
    )
    parser.add_argument(
        "--embedder_openai_api_key_env",
        type=str,
        required=True,
        help="Name of the environment variable holding the API key for the Embedder VLLM API.",
    )
    # Dataset Arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path or name of the Hugging Face dataset (e.g., 'lucadiliello/hotpotqa').",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., 'train', 'validation', 'test').",
    )
    parser.add_argument(
        "--query_column",
        type=str,
        required=True,
        help="Column in the dataset containing the queries/questions.",
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        required=True,
        help="Column in the dataset containing the ground truth answers.",
    )
    # Milvus Arguments
    parser.add_argument(
        "--milvus_db_path",
        type=str,
        default="./milvus_pipeline.db",
        help="Path for the Milvus Lite DB file.",
    )
    # Script Behavior Arguments
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Optional: Maximum number of queries from the dataset to process.",
    )
    parser.add_argument(
        "--results_output_path",
        type=str,
        default="benchmark_results.jsonl",
        help="Path to save the benchmarking results JSONL file.",
    )
    parser.add_argument(
        "--concurrency_limit",
        type=int,
        default=50,  # Default value if not provided
        help="Maximum number of concurrent queries to process.",
    )
    return parser.parse_args()


# --- Main Benchmarking Logic ---
async def main():
    args = parse_args()
    logger.info("Starting benchmarking script with arguments: %s", args)

    # 0. Define concurrency limit
    CONCURRENCY_LIMIT = args.concurrency_limit
    logger.info(f"Using concurrency limit: {CONCURRENCY_LIMIT}")

    # 1. Load API Keys
    generator_api_key_env_name = args.generator_openai_api_key_env
    if not os.getenv(generator_api_key_env_name):  # Check if the actual env var is set
        logger.error(
            f"Environment variable {generator_api_key_env_name} not set for Generator API key."
        )
        return
    logger.info(
        f"Using environment variable '{generator_api_key_env_name}' for Generator API key."
    )

    embedder_api_key_env_name = args.embedder_openai_api_key_env
    if not os.getenv(embedder_api_key_env_name):  # Check if the actual env var is set
        logger.error(
            f"Environment variable {embedder_api_key_env_name} not set for Embedder API key."
        )
        return
    logger.info(
        f"Using environment variable '{embedder_api_key_env_name}' for Embedder API key."
    )

    # 2. Load Dataset (same as before)
    logger.info(f"Loading dataset: {args.dataset_path}, split: {args.dataset_split}")
    try:
        dataset = load_dataset(args.dataset_path, split=args.dataset_split)
        df = dataset.to_pandas()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    if args.query_column not in df.columns:
        logger.error(
            f"Query column '{args.query_column}' not found. Available: {df.columns.tolist()}"
        )
        return
    if args.answer_column not in df.columns:
        logger.error(
            f"Answer column '{args.answer_column}' not found. Available: {df.columns.tolist()}"
        )
        return

    if args.max_queries and args.max_queries < len(df):
        logger.info(f"Using a subset of {args.max_queries} queries.")
        df = df.head(args.max_queries)
    logger.info(f"Loaded {len(df)} queries for benchmarking.")

    # 3. Dynamically Load and Build Pipeline
    try:
        pipelines_module = import_module("slug_search.benchmarks.pipelines")
        # Get the pipeline CLASS from the module
        PipelineClass = getattr(pipelines_module, args.pipeline_name)
    except (ModuleNotFoundError, AttributeError) as e:
        logger.error(
            f"Failed to load pipeline class: '{args.pipeline_name}' from 'pipelines.py': {e}"
        )
        return

    logger.info(f"Instantiating Haystack pipeline from class '{args.pipeline_name}'...")
    try:
        # Instantiate the pipeline class
        rag_pipeline_instance = PipelineClass(  # Renamed from rag_pipeline
            milvus_path=args.milvus_db_path,
            # Embedder VLLM params
            query_embedding_model_name=args.embedding_model_name_on_vllm,
            query_embedder_api_base=args.embedder_openai_api_base_url,
            query_embedder_api_key_env_var=embedder_api_key_env_name,
            # Generator VLLM params
            generator_model_name=args.generator_model_name,
            generator_api_base=args.generator_openai_api_base_url,
            generator_api_key_env_var=generator_api_key_env_name,
            # top_k_retriever can be added here if EmbeddedRAGPipeline expects it and it's in args
            # For now, assuming it has a default or is not needed by NaivePipeline
            timeout=TIMEOUT,
        )
        logger.info("Haystack pipeline instance created successfully.")
    except Exception as e:
        logger.error(f"Error instantiating Haystack pipeline: {e}", exc_info=True)
        return

    # 4. Run Pipeline for each query
    results = []
    logger.info(f"Starting RAG pipeline runs for {len(df)} queries...")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def process_query_wrapper(query_text, actual_answer, pipeline_instance, p_id):
        async with semaphore:  # Acquire semaphore
            logger.info(
                f'Semaphore acquired for query ID {p_id}. Starting processing: "{query_text[:100]}..."'
            )
            try:
                # pipeline_input_data is now constructed inside the pipeline's run_pipeline method
                # Use run_pipeline() for non-blocking execution, passing only the query
                generated_answer = {}
                pipeline_output = await pipeline_instance.run_pipeline(query=query_text)

                generated_answer = pipeline_output.get(
                    "generation", "Error: Could not extract answer"
                )

                # Store other information from pipeline_output (e.g., "generation_tokens") as metadata
                generated_tokens = pipeline_output.get("generation_tokens", [])

                logger.info(
                    f"Finished query ID {p_id}. Query: {query_text}, Actual: {actual_answer}, Generated: {generated_answer}, Tokens: {generated_tokens}"
                )
                return {
                    "query": query_text,
                    "actual_answer": actual_answer,
                    "generated_answer": generated_answer,
                    "generated_tokens": generated_tokens,
                }
            except Exception as e:
                logger.error(
                    f'Error running pipeline for query ID {p_id} ("{query_text}"): {e}',
                    exc_info=True,
                )
                return {
                    "query": query_text,
                    "actual_answer": actual_answer,
                    "generated_answer": f"Error: {e}",
                    "generated_tokens": 0,
                }
            # Semaphore is released automatically when exiting the 'async with' block

    # Create tasks for all queries
    tasks = []
    for index, row in df.iterrows():
        query_text = row[args.query_column]
        actual_answer = row[args.answer_column]
        # Pass the single rag_pipeline_instance to each task
        # Also, ensure actual_answer is in a format suitable for JSON (e.g. list of strings if it's complex)
        # The original .tolist() might be problematic if actual_answer is not a Series/array object here.
        # Assuming actual_answer from df row is already suitable or simple type. If it's a list/array:
        current_actual_answer = actual_answer
        if hasattr(
            actual_answer, "tolist"
        ):  # Check if it's a pandas Series or numpy array
            current_actual_answer = actual_answer.tolist()

        tasks.append(
            process_query_wrapper(  # Use the wrapper
                query_text, current_actual_answer, rag_pipeline_instance, index + 1
            )
        )  # index + 1 for 1-based logging

    logger.info(
        f"Gathering results for {len(tasks)} queries concurrently with a limit of {CONCURRENCY_LIMIT}..."
    )
    results_list = await asyncio.gather(*tasks)

    # Filter out None results if any task failed critically before returning a dict (though current process_query always returns a dict)
    results = [res for res in results_list if isinstance(res, dict)]

    # 5. Save Results
    try:
        with open(args.results_output_path, "w") as f:
            for result_item in results:
                f.write(json.dumps(result_item) + "\n")
        logger.info(f"Benchmarking results saved to {args.results_output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {args.results_output_path}: {e}")

    logger.info(
        f"Benchmarking script finished. Check ./benchmarking.log and {args.results_output_path} (in project root)"
    )


if __name__ == "__main__":
    asyncio.run(main())
