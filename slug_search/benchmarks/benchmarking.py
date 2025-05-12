import argparse
import asyncio
import logging
import os
import pandas as pd
from datasets import load_dataset
from importlib import import_module

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmarking.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
        default="benchmark_results.csv",
        help="Path to save the benchmarking results CSV file.",
    )
    return parser.parse_args()


# --- Main Benchmarking Logic ---
async def main():
    args = parse_args()
    logger.info("Starting benchmarking script with arguments: %s", args)

    # 1. Load API Keys
    generator_api_key = os.getenv(args.generator_openai_api_key_env)
    if not generator_api_key:
        logger.error(
            f"Environment variable {args.generator_openai_api_key_env} not set for Generator API key."
        )
        return
    logger.info(
        f"Successfully loaded Generator API key from env var {args.generator_openai_api_key_env}."
    )

    embedder_api_key = os.getenv(args.embedder_openai_api_key_env)
    if not embedder_api_key:
        logger.error(
            f"Environment variable {args.embedder_openai_api_key_env} not set for Embedder API key."
        )
        return
    logger.info(
        f"Successfully loaded Embedder API key from env var {args.embedder_openai_api_key_env}."
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
        pipeline_builder_fn = getattr(pipelines_module, args.pipeline_name)
    except (ModuleNotFoundError, AttributeError) as e:
        logger.error(
            f"Failed to load pipeline: '{args.pipeline_name}' from 'pipelines.py': {e}"
        )
        return

    logger.info(f"Building Haystack pipeline using '{args.pipeline_name}'...")
    try:
        rag_pipeline = pipeline_builder_fn(
            milvus_path=args.milvus_db_path,
            # Embedder VLLM params
            query_embedding_model_name=args.embedding_model_name_on_vllm,
            query_embedder_api_base=args.embedder_openai_api_base_url,
            query_embedder_api_key=embedder_api_key,
            # Generator VLLM params
            generator_model_name=args.generator_model_name,
            generator_api_base=args.generator_openai_api_base_url,
            generator_api_key=generator_api_key,
        )
        logger.info("Haystack pipeline built successfully.")
    except Exception as e:
        logger.error(f"Error building Haystack pipeline: {e}", exc_info=True)
        return

    # 4. Run Pipeline for each query
    results = []
    logger.info(f"Starting RAG pipeline runs for {len(df)} queries...")

    async def process_query(query_text, actual_answer, rag_pipeline_instance, p_id):
        logger.info(f'Starting processing for query ID {p_id}: "{query_text[:100]}..."')
        try:
            pipeline_input_data = {
                "query_text_embedder": {"text": query_text},  # For OpenAITextEmbedder
                "prompt_builder": {"query": query_text},  # For PromptBuilder
            }
            # Use run_async() for non-blocking execution
            pipeline_output = await rag_pipeline_instance.run_async(
                data=pipeline_input_data
            )

            generated_answer = "Error: Could not extract answer"
            # Component name in pipelines.py is "llm_generator"
            llm_output_key = "llm_generator"

            if (
                llm_output_key in pipeline_output
                and "replies" in pipeline_output[llm_output_key]
                and pipeline_output[llm_output_key]["replies"]
            ):
                generated_answer = pipeline_output[llm_output_key]["replies"][0]

            pipeline_meta = (
                pipeline_output.get(llm_output_key, {}).get("meta", {})
                if pipeline_output and llm_output_key in pipeline_output
                else {}
            )

            logger.info(
                f"Finished query ID {p_id}. Query: {query_text}, Actual: {actual_answer}, Generated: {generated_answer}"
            )
            return {
                "query": query_text,
                "actual_answer": actual_answer,
                "generated_answer": generated_answer,
                "pipeline_output_metadata": pipeline_meta,
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
                "pipeline_output_metadata": {},
            }

    # Create tasks for all queries
    tasks = []
    for index, row in df.iterrows():
        query_text = row[args.query_column]
        actual_answer = row[args.answer_column]
        # Pass the single rag_pipeline instance to each task
        tasks.append(
            process_query(query_text, actual_answer, rag_pipeline, index + 1)
        )  # index + 1 for 1-based logging

    logger.info(f"Gathering results for {len(tasks)} queries concurrently...")
    results_list = await asyncio.gather(*tasks)

    # Filter out None results if any task failed critically before returning a dict (though current process_query always returns a dict)
    results = [res for res in results_list if isinstance(res, dict)]

    # 5. Save Results (same as before)
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(args.results_output_path, index=False)
        logger.info(f"Benchmarking results saved to {args.results_output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {args.results_output_path}: {e}")

    logger.info("Benchmarking script finished.")


if __name__ == "__main__":
    asyncio.run(main())
