# Slug Search

This directory contains scripts and data for a project involving Large Language Models (LLMs), including components for serving models via VLLM, benchmarking various LLM pipelines, and managing data for these processes.

## 🚀 Quick Start

### Prerequisites

1. **Activate virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Start VLLM servers** (in separate terminals):
   ```bash
   # Generator server (port 8000)
   python -m vllm.entrypoints.openai.api_server \
     --model meta-llama/Llama-3.2-3B-Instruct \
     --port 8000

   # Embedder server (port 8001)  
   python -m vllm.entrypoints.openai.api_server \
     --model BAAI/bge-large-en-v1.5 \
     --port 8001 \
     --task embed
   ```

3. **Set environment variables**:
   ```bash
   export VLLM_API_KEY="your-api-key"
   ```

### Basic Usage

**1. Create Vector Store**:
```bash
python slug_search/data/datastore.py \
  --dataset_name lucadiliello/hotpotqa \
  --split_name train \
  --text_column context \
  --drop_old_db
```

**2. Run Evaluation**:
```bash
python slug_search/benchmarks/benchmarking.py \
  --pipeline_name EmbeddedRAGPipeline \
  --dataset_path lucadiliello/hotpotqa \
  --dataset_split validation \
  --query_column question \
  --answer_column answer \
  --generator_model_name meta-llama/Llama-3.2-3B-Instruct \
  --generator_openai_api_base_url http://localhost:8000/v1 \
  --generator_openai_api_key_env VLLM_API_KEY \
  --embedding_model_name_on_vllm BAAI/bge-large-en-v1.5 \
  --embedder_openai_api_base_url http://localhost:8001/v1 \
  --embedder_openai_api_key_env VLLM_API_KEY \
  --metrics "check_answer_correctness_multi_gt;ground_truth_hit_rate;ground_truth_precision" \
  --summary \
  --max_queries 10
```

### Available Metrics
- `check_answer_correctness_multi_gt`: Generation quality
- `ground_truth_hit_rate`: Whether ground-truth chunks retrieved
- `ground_truth_precision`: Proportion of relevant chunks
- `ground_truth_count`: Number of ground-truth chunks

### Common Commands

**Quick test (10 queries)**:
```bash
--max_queries 10 --metrics "ground_truth_hit_rate;check_answer_correctness_multi_gt"
```

**Full evaluation with summary**:
```bash
--metrics "check_answer_correctness_multi_gt;ground_truth_hit_rate;ground_truth_precision;ground_truth_count" --summary
```

### Output Files
- `benchmark_results.jsonl`: Individual query results with metrics
- `benchmark_results_summary.json`: Statistical summary (with `--summary`)
- `benchmarking.log`: Detailed execution logs

### Troubleshooting
- **"Unknown metric"**: Check spelling, use `;` to separate
- **Memory issues**: Lower `--concurrency_limit` or `--max_queries`
- **API errors**: Verify VLLM servers are running and API keys set

## Directory Structure

- **`benchmarks/`**: Contains scripts and utilities for benchmarking different LLM interaction pipelines.
- **`data/`**: Houses data-related scripts, a test notebook, and a Milvus vector database.
- **`vllm_servers/`**: Includes shell scripts for launching VLLM (Very Large Language Model) servers, compatible with the OpenAI API format. These servers are primarily intended for use with the benchmarking scripts in the `benchmarks/` directory.
- **`training/`**: Contains scripts and configuration for training/fine-tuning models using the `art` library.

## Files and Usage

### 1. Starting VLLM Servers (`vllm_servers/`)

These scripts launch VLLM instances to serve LLMs as OpenAI-compatible API endpoints. They are primarily used by the benchmarking scripts.

**Common Configuration:**
- Scripts are typically run from the project root (e.g., `bash slug_search/vllm_servers/launch_generator_vllm.sh`).
- **Logging:** VLLM logging is configured via `VLLM_CONFIGURE_LOGGING=1` and `VLLM_LOGGING_CONFIG_PATH` (pointing to `generator_logging_config.json` or `embedder_logging_config.json`). Logs are output to files specified in these JSON configs (e.g., `generator_vllm_server.log`, `embedder_vllm_server.log` in the project root).
- **GPU Allocation:** The `CUDA_VISIBLE_DEVICES` environment variable can be set within the scripts (or globally) to specify which GPU(s) to use.

-   **`launch_generator_vllm.sh`**: Shell script to start a VLLM server for text generation tasks.
    *   **Usage**: `bash slug_search/vllm_servers/launch_generator_vllm.sh`
    *   **Key Environment Variables (can override defaults in script):**
        *   `GENERATOR_MODEL_NAME`: Hugging Face model name for generation (Default: `unsloth/Qwen3-4B`).
        *   `GENERATOR_PORT`: Port for the server (Default: `40001`).
        *   `GENERATOR_TASK_TYPE`: Task type for VLLM (Default: `generate`).
        *   `GENERATOR_LOG_FILE`: (Deprecated by VLLM_LOGGING_CONFIG_PATH) Name for the log file.
    *   **VLLM Arguments:** The script calls `python -m vllm.entrypoints.openai.api_server` with arguments like:
        *   `--model`: The model name.
        *   `--port`: The server port.
        *   `--host`: Server host (Default: `0.0.0.0`).
        *   `--served-model-name`: Name for the served model.
        *   `--task`: VLLM task type.
        *   `--gpu-memory-utilization`: GPU memory utilization (Default: `0.95`).
        *   `--max-model-len`: Maximum model length (Default: `8192`).
        *   `--enable-auto-tool-choice`, `--tool-call-parser hermes`: For models with tool-calling capabilities.
    *   *Note*: Modify the script to adjust `CUDA_VISIBLE_DEVICES`, `gpu-memory-utilization`, `tensor-parallel-size`, `max-model-len`, or add other VLLM arguments as needed.

-   **`launch_embedder_vllm.sh`**: Shell script to start a VLLM server for creating text embeddings.
    *   **Usage**: `bash slug_search/vllm_servers/launch_embedder_vllm.sh`
    *   **Key Environment Variables (can override defaults in script):**
        *   `EMBEDDER_MODEL_NAME`: Hugging Face model name for embeddings (Default: `BAAI/bge-large-en-v1.5`).
        *   `EMBEDDER_PORT`: Port for the server (Default: `40002`).
        *   `EMBEDDER_TASK_TYPE`: Task type for VLLM (Default: `embed`).
        *   `EMBEDDER_LOG_FILE`: (Deprecated by VLLM_LOGGING_CONFIG_PATH) Name for the log file.
    *   **VLLM Arguments:** Similar to the generator, including:
        *   `--model`, `--port`, `--host`, `--served-model-name`, `--task`.
        *   `--gpu-memory-utilization`: (Default: `0.40`).
    *   *Note*: Modify the script to adjust `CUDA_VISIBLE_DEVICES`, `gpu-memory-utilization`, or other VLLM arguments. The `--task embed` argument is crucial for embedding models.

### 2. Creating Vector Datastores (`data/`)

These scripts handle the creation and population of a Milvus vector database with embeddings.

-   **`datastore.sh`**: Shell script that automates the creation of the Milvus vector database using `datastore.py`.
    *   **Usage**: `bash slug_search/data/datastore.sh`
    *   **Functionality**:
        *   By default, it processes both 'train' and 'validation' splits of the `lucadiliello/hotpotqa` dataset.
        *   It uses `datastore.py` to load data, chunk text from the `context` column, generate embeddings using `BAAI/bge-large-en-v1.5` (via VLLM's direct batch processing), and insert them into `slug_search/data/milvus_hotpotqa.db`.
        *   The first call to `datastore.py` (for the 'train' split) includes the `--drop_old_db` flag to recreate the database.
    *   **Customization**: Modify `datastore.sh` to change dataset name, splits, text column, Milvus path, embedding model, or other parameters passed to `datastore.py`.

-   **`datastore.py`**: Python script to create and populate a Milvus vector database. This script uses VLLM's `LLM` class for direct batch embedding and does **not** require a separate VLLM server to be running.
    *   **Direct Usage**: `python slug_search/data/datastore.py [ARGUMENTS]`
    *   Run `python slug_search/data/datastore.py --help` for a full list of arguments.
    *   **Key Arguments**:
        *   `--dataset_name`: HF dataset name (Default: `lucadiliello/hotpotqa`).
        *   `--split_name`: Dataset split (Default: `train`).
        *   `--text_column`: Column with text to embed (Default: `context`).
        *   `--milvus_db_path`: Path for Milvus Lite DB file (Default: `./milvus_pipeline.db`, but `datastore.sh` overrides this to `slug_search/data/milvus_hotpotqa.db`).
        *   `--model`: HF model name for VLLM embeddings (Default: `BAAI/bge-large-en-v1.5`). This is a VLLM `EngineArgs` parameter.
        *   `--task`: VLLM engine task, should be `embed`.
        *   `--preprocess_function`: Name of the preprocessing function in the script (Default: `preprocess_and_chunk_text`).
        *   `--drop_old_db`: Flag to drop an existing Milvus DB.
        *   `--max_docs`: Maximum number of documents to process.
        *   `--metadata_columns`: List of dataset columns to include as metadata.
        *   Other VLLM engine arguments (e.g., `--gpu-memory-utilization`, `--tensor-parallel-size`) can be passed directly.
    *   **Functionality**:
        1.  Loads data from Hugging Face datasets.
        2.  Preprocesses and chunks text using the specified function (e.g., `preprocess_and_chunk_text`).
        3.  Initializes VLLM's `LLM` engine with the specified embedding model and arguments.
        4.  Generates embeddings for the text chunks in batches.
        5.  Writes the documents (text chunks and their embeddings) to the Milvus database.

-   **`milvus_hotpotqa.db`**: A Milvus vector database file generated by `datastore.sh` (via `datastore.py`). It contains chunked text and corresponding embeddings from the HotpotQA dataset.

-   **`test_data.ipynb`**: A Jupyter Notebook likely used for testing data integrity, datastore connections, or experimenting with the data in `milvus_hotpotqa.db`.

### 3. Executing Benchmarks and Adding Pipelines (`benchmarks/`)

This section describes how to run benchmarks on different LLM pipelines and how to define new pipelines.

-   **`run_benchmark.sh`**: A shell script to automate the execution of benchmarks using `benchmarking.py`.
    *   **Usage**: `bash slug_search/benchmarks/run_benchmark.sh` (typically run from the project root).
    *   **Functionality**:
        *   Exports dummy API keys (`GENERATOR_API_KEY`, `EMBEDDER_API_KEY`) set to "EMPTY" for local VLLM server usage.
        *   Executes `python -m slug_search.benchmarks.benchmarking` with a predefined set of arguments.
    *   **Configuration**:
        *   You **must** ensure the VLLM servers (generator and embedder) are running and accessible at the URLs specified in the script's arguments (or override them).
        *   Modify the arguments passed to `benchmarking.py` within this script to change:
            *   `--pipeline_name`: The class name of the pipeline to run (from `pipelines.py`).
            *   Generator and Embedder model names and API endpoints (`--generator_model_name`, `--generator_openai_api_base_url`, etc.).
            *   Dataset details (`--dataset_path`, `--dataset_split`, `--query_column`, `--answer_column`).
            *   Milvus DB path (`--milvus_db_path`).
            *   Output path (`--results_output_path`, Default: `hotpotqa_benchmark_results.jsonl`).
            *   Concurrency (`--concurrency_limit`, Default: 70).
            *   `--max_queries`: Limit the number of queries processed.
    *   **Output**: Results are saved to a `.jsonl` file (e.g., `hotpotqa_benchmark_results.jsonl`) and logs are written to `benchmarking.log` in the project root.

-   **`benchmarking.py`**: A Python script for running benchmarks on Haystack `AsyncPipeline`s.
    *   **Direct Usage**: `python -m slug_search.benchmarks.benchmarking [ARGUMENTS]`
    *   Run `python -m slug_search.benchmarks.benchmarking --help` for a full list of arguments.
    *   **Functionality**:
        1.  Parses command-line arguments for pipeline configuration, model endpoints, dataset details, etc.
        2.  Loads API keys from environment variables.
        3.  Loads the specified dataset.
        4.  Dynamically imports and instantiates the specified pipeline class from `slug_search.benchmarks.pipelines`.
        5.  Asynchronously runs the pipeline for each query from the dataset, up to `max_queries` and with a given `concurrency_limit`.
        6.  Collects results (generated answer, actual answer, tokens used).
        7.  Saves results to a JSONL file.
    *   **Key Arguments**: See `run_benchmark.sh` configuration or run with `--help`.

-   **`pipelines.py`**: Python script defining various Haystack `AsyncPipeline` classes.
    *   **Structure**:
        *   An abstract base class `Pipe` defines the interface for pipelines (`__init__` and `run_pipeline`).
        *   Concrete pipeline classes inherit from `Pipe` (e.g., `EmbeddedRAGPipeline`, `NaiveGenerationPipeline`, `EmbedderRetrieverPipeline`).
    *   **Existing Pipelines**:
        *   `EmbeddedRAGPipeline`: Implements Retrieval Augmented Generation. It uses an `OpenAITextEmbedder` for query embedding (via a VLLM embedder server), a `MilvusEmbeddingRetriever` to fetch documents from the Milvus DB, a `ChatPromptBuilder` to construct the prompt, and an `OpenAIChatGenerator` (via a VLLM generator server) to produce the answer.
        *   `NaiveGenerationPipeline`: A simpler pipeline that sends the query directly to an `OpenAIChatGenerator` (via a VLLM generator server) after basic prompt formatting.
        *   `EmbedderRetrieverPipeline`: A pipeline that embeds a query and retrieves documents from Milvus, returning the retrieved documents.
    *   **Adding a New Pipeline**:
        1.  Create a new class that inherits from `slug_search.benchmarks.pipelines.Pipe`.
        2.  Implement the `__init__` method:
            *   Accept necessary configuration parameters (e.g., Milvus path, model names, API URLs, API key environment variable names, timeout values).
            *   Initialize Haystack components (DocumentStore, Embedders, Retrievers, PromptBuilders, Generators). Ensure you use `haystack.utils.Secret.from_env_var("YOUR_ENV_VAR_NAME")` for API keys.
            *   Construct your `haystack.AsyncPipeline`.
            *   Add components to the pipeline using `pipeline.add_component("name", component_instance)`.
            *   Connect components using `pipeline.connect("component_A.output_socket", "component_B.input_socket")`.
            *   Store the constructed pipeline in `self.pipeline`.
        3.  Implement the `async def run_pipeline(self, query: str) -> dict:` method:
            *   Check if `self.pipeline` is initialized.
            *   Prepare the input data dictionary for the pipeline's `run_async` method. The keys should match the input component names and their expected input data structures (e.g., `{"query_text_embedder": {"text": query}, "prompt_builder": {"query": query}}`).
            *   Call `output = await self.pipeline.run_async(data=pipeline_input_data)`.
            *   Extract the desired information from the `output` dictionary (e.g., generated text, retrieved documents, token counts) and return it as a dictionary. The `benchmarking.py` script expects at least a `"generation"` key for pipelines that generate text, and can also process `"generation_tokens"`.
        4.  After defining your new pipeline class in `pipelines.py`, you can use it in `benchmarking.py` by passing its class name to the `--pipeline_name` argument (e.g., via `run_benchmark.sh`).

-   **`metrics.py`**: Contains functions for evaluating pipeline outputs against ground truth, likely used for more detailed analysis beyond simple generation. (Usage not detailed in `benchmarking.py` execution flow, may require separate integration).

### 4. Running Training Scripts (`training/`)

This directory contains scripts for fine-tuning or training models using the `art` library. The main script is `train.py`.

-   **`train.py`**: Python script to run training for a model using the `art` library.
    *   **Usage**: `python -m slug_search.training.train [ARGUMENTS]`
    *   **Key Functionality**:
        1.  **Configuration**:
            *   Loads global `TrainingConfig` and `ProjectPolicyConfig` which define parameters like batch sizes, learning rates, model details, dataset paths, prompt templates, and verifiers.
            *   Loads prompt templates from `prompts.json` and chat templates from `chat_templates.json`.
            *   Search tools (Milvus retriever) are configured globally at the start of the script.
        2.  **Arguments**:
            *   `--debug`: Runs in a debug mode with minimal samples and steps. Creates/uses project `slug_search_project_debug` and model `slug-search-agent-debug`.
            *   `--prompt_template`: Key for the prompt template to use from `prompts.json` (Default: `default_query_prompt`).
            *   `--verifier`: Verifier to use during training (Default: `check_answer_correctness_multi_gt`).
            *   `--chat_template`: Key for a custom chat template from `chat_templates.json`.
        3.  **Training Process (`run_training` function)**:
            *   Initializes an `art.TrainableModel` with a name, project name, base model, and the policy configuration.
            *   Registers the model with a backend (e.g., `LocalBackend`).
            *   Uses `load_and_iterate_hf_dataset` to load training data.
            *   Iterates through epochs and batches:
                *   Performs validation at specified `eval_steps` using `run_validation`.
                *   For each scenario in a batch, generates multiple trajectories per group using `art.rollout` (defined in `training/rollout.py`).
                *   Calls `model.train()` with the gathered trajectory groups and training configuration (e.g., learning rate).
            *   Runs a final validation after training.
        4.  **Validation Process (`run_validation` function)**:
            *   Loads validation data using `load_and_iterate_hf_dataset`.
            *   Gathers trajectories for validation scenarios using `art.gather_trajectories`.
            *   Logs trajectories if a backend is available.
            *   Calculates and logs average metrics (including reward) to a file in `validation_logs/PROJECT_NAME-MODEL_NAME.log`.
    *   **Output**:
        *   Trained model/LoRA adapter saved by the `art` library under `.art/PROJECT_NAME/models/MODEL_NAME/`.
        *   Validation logs in `validation_logs/`.
    *   **Prerequisites for Training**:
        *   Ensure VLLM servers (especially the embedder for search tools, if used by the agent) are running if the agent's tools require them. The default configuration in `train.py` configures search tools to use an embedder at `http://localhost:40002/v1`.
        *   The base model specified in `ProjectPolicyConfig` (default `unsloth/Qwen3-0.6B`) needs to be accessible.
        *   Datasets specified in `ProjectPolicyConfig` need to be accessible.
    *   **Customization**:
        *   Modify `TrainingConfig` and `ProjectPolicyConfig` directly in `train.py` for persistent changes.
        *   Use command-line arguments for temporary changes to prompt template, verifier, or to run in debug mode.
        *   Update `prompts.json` or `chat_templates.json` to add or modify templates.
        *   Modify `rollout.py` to change how trajectories are generated or how tools/verifiers are used within a rollout.
        *   The `search_tools.py` can be modified to change the retriever configuration.
        *   `dataloaders.py` can be adjusted if different data loading or preprocessing is needed.

-   **Other files in `training/`**:
    *   `rollout.py`: Defines the `rollout` function used to generate a single trajectory for a given model and scenario (e.g., a query). This involves model inference, tool usage, and verification.
    *   `data_types.py`: Defines Pydantic models for configuration (`TrainingConfig`, `ProjectPolicyConfig`, `SearchQuery`).
    *   `dataloaders.py`: Contains `load_and_iterate_hf_dataset` for loading and batching data from Hugging Face datasets.
    *   `search_tools.py`: Configures and provides search/retrieval tools for the agent (e.g., Milvus retriever via Haystack).
    *   `prompts.json`: JSON file containing various prompt templates.
    *   `chat_templates.json`: JSON file containing custom chat templates for models.
    *   `verifiers.py`: Likely contains verifier functions (though the default `check_answer_correctness_multi_gt` might be an `art` built-in or defined elsewhere).

## Setup and Prerequisites

1.  **Python Environment**: Ensure a Python environment with all dependencies listed in `pyproject.toml` (and `uv.lock`) is active.
2.  **VLLM Installation**: VLLM must be installed and correctly configured to use available GPUs. Refer to the official VLLM documentation.
3.  **API Keys**: If using models hosted via APIs that require keys (even for local VLLM OpenAI-compatible servers if auth is enabled), ensure the respective environment variables (e.g., `GENERATOR_API_KEY`, `EMBEDDER_API_KEY`) are set. The `run_benchmark.sh` script exports "EMPTY" as a placeholder for local VLLM usage.
4.  **Data**:
    *   To populate the vector database, run `bash data/datastore.sh`. This will download the HotpotQA dataset and the BGE embedding model. This step uses VLLM's batch embedding capabilities and does not require the servers from `vllm_servers/` to be active.
    *   Ensure sufficient disk space for datasets, models, and the Milvus DB.
5.  **VLLM Servers (for Benchmarking)**: Before running benchmarks in the `benchmarks/` directory that rely on local VLLM instances (i.e., when providing local OpenAI API compatible endpoints), launch the required servers using the scripts in `vllm_servers/`. For example:
    *   `bash vllm_servers/launch_generator_vllm.sh`
    *   `bash vllm_servers/launch_embedder_vllm.sh`
    *   Ensure the ports and model names in the benchmark configurations match those used by the VLLM servers.