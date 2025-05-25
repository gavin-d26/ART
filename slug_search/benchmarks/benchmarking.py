import argparse
import asyncio
import logging
import os
import pandas as pd
import sys
from datasets import load_dataset
from importlib import import_module
import json
from tqdm.asyncio import tqdm

# Import Phase 1 utilities for ground-truth analysis
from slug_search.data.datastore import (
    extract_document_id_from_query_metadata,
    check_if_ground_truth_retrieved,
)

# Import Phase 5 metrics module for dynamic function discovery
import slug_search.benchmarks.metrics as metrics_module

# Import Phase 5 analysis functions
from slug_search.benchmarks.analysis import (
    generate_evaluation_summary,
    print_summary,
    save_summary,
)

# Import search tools configuration
from slug_search.training.search_tools import configure_search_tools


# --- Logging Setup ---
# Create a custom logger that redirects all output to log file
class LogRedirector:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


# Configure logging to capture everything
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("benchmarking.log"),
        logging.StreamHandler(sys.stderr),  # Keep stderr for progress bar
    ],
)
logger = logging.getLogger(__name__)

# Redirect stdout to logger (but keep stderr for progress bar)
sys.stdout = LogRedirector(logger, logging.INFO)

TIMEOUT = 60.0 * 20


# Phase 5: Dynamically discover all metric functions
def get_available_metrics():
    """Dynamically discover all metric functions from metrics module."""
    import inspect

    available_metrics = {}

    # Get all functions from the metrics module
    for name, obj in inspect.getmembers(metrics_module, inspect.isfunction):
        # Skip private functions and the evaluate_results function
        if not name.startswith("_") and name != "evaluate_results":
            available_metrics[name] = obj

    return available_metrics


# Get available metrics dynamically - must be before parse_args()
AVAILABLE_METRICS = get_available_metrics()


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
    # Phase 3: Ground-truth analysis arguments
    parser.add_argument(
        "--enable_ground_truth_analysis",
        action="store_true",
        default=True,
        help="Enable ground-truth retrieval analysis (requires dataset_name and split_name).",
    )
    # Phase 5: Metrics computation arguments
    available_metrics_list = list(AVAILABLE_METRICS.keys())
    generation_metrics = [
        m for m in available_metrics_list if not m.startswith("ground_truth_")
    ]
    retrieval_metrics = [
        m for m in available_metrics_list if m.startswith("ground_truth_")
    ]

    help_text = (
        "Semicolon-separated list of metrics to compute. "
        f"Available metrics ({len(available_metrics_list)} total): "
        f"Generation metrics: {', '.join(generation_metrics)}; "
        f"Retrieval metrics: {', '.join(retrieval_metrics)}. "
        "Example: 'check_answer_correctness_multi_gt;ground_truth_hit_rate;ground_truth_precision'. "
        "If not provided, only benchmarking results are saved."
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help=help_text,
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate evaluation summary with statistics and insights after benchmarking. Requires --metrics to be specified. Creates both console output and JSON summary file.",
    )

    # Agent-specific arguments for AgenticToolCallingPipeline
    parser.add_argument(
        "--agent_query_prompt_template_key",
        type=str,
        default="default_query_prompt",
        help="Key to load prompt template from prompts.json for the agent (template should contain {{query}} placeholder).",
    )
    parser.add_argument(
        "--unknown_tool_handling_strategy",
        type=str,
        choices=["error_to_model", "break_loop"],
        default="break_loop",
        help="Strategy for handling unknown tool calls: 'error_to_model' or 'break_loop'.",
    )
    parser.add_argument(
        "--search_tool_top_k",
        type=int,
        default=3,
        help="Top-k documents to retrieve for the search tool.",
    )
    parser.add_argument(
        "--max_agent_steps",
        type=int,
        default=5,
        help="Maximum number of agent steps before termination.",
    )
    parser.add_argument(
        "--agent_llm_sampling_params",
        type=str,
        default=None,
        help="JSON string of sampling parameters for the agent LLM (e.g., '{\"temperature\": 0.7}').",
    )
    parser.add_argument(
        "--top_k_retriever",
        type=int,
        default=3,
        help="Top-k documents to retrieve by the main retriever in the RAG pipeline (distinct from agent's search_tool_top_k).",
    )

    return parser.parse_args()


# --- Phase 5: Metrics Computation Functions ---
def compute_metrics_for_result(result, metric_names):
    """Compute specified metrics for a single result."""
    if not metric_names:
        return result

    for metric_name in metric_names:
        if metric_name not in AVAILABLE_METRICS:
            logger.warning(f"Unknown metric: {metric_name}")
            continue

        metric_fn = AVAILABLE_METRICS[metric_name]
        try:
            if metric_name.startswith("ground_truth_"):
                # Retrieval metric
                metric_value = metric_fn(result.get("ground_truth_analysis", {}))
            else:
                # Generation metric
                metric_value = metric_fn(
                    result.get("generated_answer", ""), result.get("actual_answer", "")
                )
            result[metric_name] = metric_value
        except Exception as e:
            logger.error(f"Error computing {metric_name}: {e}")
            result[metric_name] = None

    return result


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
        # Ensure index is 0-based and sequential for ID consistency
        df = df.reset_index(drop=True)
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
        # Configure search tools globally before pipeline instantiation
        configure_search_tools(
            milvus_db_path=args.milvus_db_path,
            embedding_model_name=args.embedding_model_name_on_vllm,
            embedder_api_base=args.embedder_openai_api_base_url,
            embedder_api_key_env_var=embedder_api_key_env_name,
            top_k=args.search_tool_top_k,
        )

        # Parse agent LLM sampling params if provided
        agent_llm_sampling_params = None
        if args.agent_llm_sampling_params:
            try:
                agent_llm_sampling_params = json.loads(args.agent_llm_sampling_params)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse agent_llm_sampling_params: {e}. Using defaults."
                )

        # Prepare all possible parameters - pipelines will accept what they need via **kwargs
        pipeline_params = {
            "milvus_path": args.milvus_db_path,
            "query_embedding_model_name": args.embedding_model_name_on_vllm,
            "embedding_model_name": args.embedding_model_name_on_vllm,  # Alternative name
            "query_embedder_api_base": args.embedder_openai_api_base_url,
            "embedder_api_base": args.embedder_openai_api_base_url,  # Alternative name
            "query_embedder_api_key_env_var": embedder_api_key_env_name,
            "embedder_api_key_env_var": embedder_api_key_env_name,  # Alternative name
            "generator_model_name": args.generator_model_name,
            "generator_api_base": args.generator_openai_api_base_url,
            "generator_api_key_env_var": generator_api_key_env_name,
            "top_k_retriever": args.top_k_retriever,
            "timeout": TIMEOUT,
            # Agent-specific parameters
            "agent_llm_model_name": args.generator_model_name,
            "agent_llm_api_base_url": args.generator_openai_api_base_url,
            "agent_llm_api_key_env_var": generator_api_key_env_name,
            "agent_query_prompt_template_key": args.agent_query_prompt_template_key,
            "unknown_tool_handling_strategy": args.unknown_tool_handling_strategy,
            "search_tool_top_k": args.search_tool_top_k,
            "max_agent_steps": args.max_agent_steps,
            "agent_llm_sampling_params": agent_llm_sampling_params,
        }

        # Instantiate the pipeline class - it will accept what it needs via **kwargs
        rag_pipeline_instance = PipelineClass(**pipeline_params)
        logger.info("Haystack pipeline instance created successfully.")
    except Exception as e:
        logger.error(f"Error instantiating Haystack pipeline: {e}", exc_info=True)
        return

    # 4. Run Pipeline for each query
    results = []
    logger.info(f"Starting RAG pipeline runs for {len(df)} queries...")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def process_query_wrapper(
        query_text, actual_answer, pipeline_instance, p_id, query_row, index
    ):
        async with semaphore:  # Acquire semaphore
            logger.debug(f'Processing query ID {p_id}: "{query_text[:50]}..."')
            try:
                # pipeline_input_data is now constructed inside the pipeline's run_pipeline method
                # Use run_pipeline() for non-blocking execution, passing only the query
                pipeline_output = await pipeline_instance.run_pipeline(query=query_text)

                generated_answer = pipeline_output.get(
                    "generation", "Error: Could not extract answer"
                )

                # Store other information from pipeline_output (e.g., "generation_tokens") as metadata
                generated_tokens = pipeline_output.get("generation_tokens", 0)

                # Phase 3: Extract retrieved chunks from pipeline output
                retrieved_chunks = pipeline_output.get("retrieved_chunks", [])

                # Extract tool_log from pipeline output (for agent pipelines)
                tool_log = pipeline_output.get("tool_log", [])

                # Phase 3: Ground-truth analysis (if enabled)
                ground_truth_analysis = {}
                query_document_id = None

                if args.enable_ground_truth_analysis:
                    try:
                        # Extract the document ID for this query to check ground-truth retrieval
                        query_document_id = extract_document_id_from_query_metadata(
                            query_row, args.dataset_path, args.dataset_split, index
                        )

                        # Check if ground-truth chunks were retrieved
                        ground_truth_analysis = check_if_ground_truth_retrieved(
                            retrieved_chunks, query_document_id
                        )

                        logger.debug(
                            f"Query ID {p_id}: Ground-truth analysis - "
                            f"Retrieved: {ground_truth_analysis.get('ground_truth_retrieved', False)}, "
                            f"GT chunks: {ground_truth_analysis.get('num_ground_truth_chunks', 0)}, "
                            f"Total chunks: {ground_truth_analysis.get('total_retrieved', 0)}"
                        )
                    except Exception as gt_error:
                        logger.warning(
                            f"Ground-truth analysis failed for query ID {p_id}: {gt_error}"
                        )
                        ground_truth_analysis = {"error": str(gt_error)}

                logger.debug(f"Completed query ID {p_id}. Tokens: {generated_tokens}")

                # Phase 3: Enhanced result format
                result = {
                    "query_id": query_document_id
                    or f"query_{p_id}",  # Use document ID as query ID
                    "query": query_text,
                    "actual_answer": actual_answer,
                    "generated_answer": generated_answer,
                    "generation_tokens": generated_tokens,
                    "retrieved_chunks": retrieved_chunks,
                }

                # Add tool_log if available (for agent pipelines)
                if tool_log:
                    result["tool_log"] = tool_log

                # Add ground-truth analysis if enabled and successful
                if args.enable_ground_truth_analysis and ground_truth_analysis:
                    result["ground_truth_analysis"] = ground_truth_analysis

                return result

            except Exception as e:
                logger.error(
                    f'Error running pipeline for query ID {p_id} ("{query_text[:50]}..."): {e}',
                    exc_info=True,
                )
                return {
                    "query_id": f"query_{p_id}",
                    "query": query_text,
                    "actual_answer": actual_answer,
                    "generated_answer": f"Error: {e}",
                    "generation_tokens": 0,
                    "retrieved_chunks": [],
                    "tool_log": [],
                }
            # Semaphore is released automatically when exiting the 'async with' block

    # Create tasks for all queries
    tasks = []
    for index, row in df.iterrows():
        query_text = row[args.query_column]
        actual_answer = row[args.answer_column]
        current_actual_answer = actual_answer
        if hasattr(actual_answer, "tolist"):
            current_actual_answer = actual_answer.tolist()
        query_row = row.to_dict()
        tasks.append(
            process_query_wrapper(
                query_text,
                current_actual_answer,
                rag_pipeline_instance,
                index,
                query_row,
                index,
            )
        )

    logger.info(
        f"Gathering results for {len(tasks)} queries concurrently with a limit of {CONCURRENCY_LIMIT}..."
    )

    # Use tqdm progress bar for task completion tracking
    progress_bar = tqdm(
        total=len(tasks),
        desc="Processing queries",
        unit="query",
        file=sys.stderr,  # Use stderr to avoid logging redirection
        dynamic_ncols=True,
        leave=True,
    )

    # Process tasks with progress tracking
    results_list = []
    completed_tasks = []

    # Convert tasks to asyncio tasks for better control
    async_tasks = [asyncio.create_task(task) for task in tasks]

    # Process tasks as they complete
    for completed_task in asyncio.as_completed(async_tasks):
        try:
            result = await completed_task
            results_list.append(result)
            progress_bar.update(1)
        except Exception as e:
            logger.error(f"Task failed: {e}")
            # Add error result to maintain consistency
            results_list.append(
                {
                    "query_id": f"error_{len(results_list)}",
                    "query": "Error",
                    "actual_answer": "Error",
                    "generated_answer": f"Task failed: {e}",
                    "generation_tokens": 0,
                    "retrieved_chunks": [],
                    "tool_log": [],
                }
            )
            progress_bar.update(1)

    progress_bar.close()
    logger.info(f"Completed processing {len(results_list)} queries")

    # Filter out None results if any task failed critically before returning a dict (though current process_query always returns a dict)
    results = [res for res in results_list if isinstance(res, dict)]

    # Phase 5: Compute metrics if requested
    if args.metrics:
        logger.info(f"Computing metrics: {args.metrics}")
        metric_names = [
            name.strip() for name in args.metrics.split(";") if name.strip()
        ]
        logger.info(f"Parsed metric names: {metric_names}")

        # Compute metrics for each result
        for i, result in enumerate(results):
            try:
                results[i] = compute_metrics_for_result(result, metric_names)
            except Exception as e:
                logger.error(f"Error computing metrics for result {i}: {e}")

        logger.info(f"Metrics computation completed for {len(results)} results")

    # 5. Save Results
    try:
        with open(args.results_output_path, "w") as f:
            for result_item in results:
                f.write(json.dumps(result_item) + "\n")
        logger.info(f"Benchmarking results saved to {args.results_output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {args.results_output_path}: {e}")

    # Phase 5: Generate summary if requested
    if args.summary and args.metrics:
        logger.info("Generating evaluation summary...")
        try:
            summary = generate_evaluation_summary(args.results_output_path)

            # Print summary to console
            print_summary(summary)

            # Save summary to file
            summary_path = args.results_output_path.replace(".jsonl", "_summary.json")
            save_summary(summary, summary_path)
            logger.info(f"Evaluation summary saved to {summary_path}")

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
    elif args.summary and not args.metrics:
        logger.warning(
            "Summary requested but no metrics computed. Use --metrics to enable summary generation."
        )

    logger.info(
        f"Benchmarking script finished. Check ./benchmarking.log and {args.results_output_path} (in project root)"
    )


if __name__ == "__main__":
    asyncio.run(main())
