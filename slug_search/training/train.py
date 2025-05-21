import gc
import art
from art.local import LocalBackend
import asyncio
from dotenv import load_dotenv
from typing import List
import argparse  # Added for command line argument parsing
import json
from pathlib import Path
import polars as pl
from datetime import datetime  # Added for timestamping

from .rollout import rollout
from .data_types import ProjectPolicyConfig, TrainingConfig, SearchQuery
from .dataloaders import load_and_iterate_hf_dataset
from .search_tools import configure_search_tools

load_dotenv()

# Configure search tools
configure_search_tools(
    milvus_db_path="slug_search/data/milvus_hotpotqa.db",
    embedding_model_name="BAAI/bge-large-en-v1.5",
    embedder_api_base="http://localhost:40002/v1",
    embedder_api_key_env_var="EMBEDDER_API_KEY",
)

BASE_MODEL_NAME = "unsloth/Qwen3-0.6B"  # Define base model name as a constant

# Global configurations
training_config = TrainingConfig(
    trajectories_per_group=2,
    groups_per_step=2,
    learning_rate=1e-5,
    eval_steps=10,
    num_epochs=1,
)

project_policy_config = ProjectPolicyConfig(
    base_model=BASE_MODEL_NAME,
    max_tool_calls=5,
    max_tokens=1200,
    log_to_openpipe=False,
    use_tools=True,
    training_config=training_config,
    prompt_template="default_query_prompt",
    verifier="check_answer_correctness_multi_gt",
    # Training dataset
    training_dataset_path="lucadiliello/hotpotqa",
    training_dataset_split="train",
    training_prompt_column="question",
    training_answer_column="answers",
    max_training_samples=20,
    # Validation dataset
    val_dataset_path="lucadiliello/hotpotqa",
    val_dataset_split="validation",
    val_prompt_column="question",
    val_answer_column="answers",
    max_val_samples=20,
)

# Prompt template configuration
PROMPTS_JSON_PATH = Path(__file__).parent / "prompts.json"
with open(PROMPTS_JSON_PATH, "r") as f:
    prompt_templates = json.load(f)

CHAT_TEMPLATE_PATH = Path(__file__).parent / "chat_templates.json"
with open(CHAT_TEMPLATE_PATH, "r") as f:
    chat_templates = json.load(f)


def get_prompt_template(key: str) -> str:
    if key not in prompt_templates:
        raise ValueError(
            f"Prompt template key '{key}' not found in prompts.json. Available keys: {list(prompt_templates.keys())}"
        )
    return prompt_templates[key]


# run a validation step between training epochs/steps.
async def run_validation(model: art.TrainableModel, global_step: int):
    assert hasattr(model, "config")
    config = model.config
    train_cfg = config.training_config
    prompt_template_key = config.prompt_template
    prompt_template = None
    if prompt_template_key:
        prompt_template = get_prompt_template(prompt_template_key)

    print(f"Running validation for model: {model.name}")

    # Use dataloader for validation data
    val_iterator = load_and_iterate_hf_dataset(
        dataset_path=config.val_dataset_path,
        dataset_split=config.val_dataset_split,
        prompt_column=config.val_prompt_column,
        answer_column=config.val_answer_column,
        prompt_template=prompt_template,
        batch_size=config.max_val_samples,  # Process all val samples in one batch
        num_epochs=1,  # Single pass
        max_samples=config.max_val_samples,
    )

    all_val_scenarios = []
    for batch, _, _, _ in val_iterator:
        all_val_scenarios.extend(batch)

    if not all_val_scenarios:
        print("No validation scenarios found. Skipping validation.")
        return

    val_trajectories = await art.gather_trajectories(
        (rollout(model, scenario) for scenario in all_val_scenarios),
        pbar_desc=f"Validation for {model.name}",
        max_exceptions=config.max_val_samples,  # Allow exceptions for all scenarios
    )

    valid_trajectories = [t for t in val_trajectories if isinstance(t, art.Trajectory)]

    if not valid_trajectories:
        print(
            "No valid trajectories generated during validation. Skipping metrics calculation."
        )
        return

    if getattr(model, "_backend", None) is not None and hasattr(model, "log"):
        await model.log(valid_trajectories)

    metrics_list = []
    for t in valid_trajectories:
        metric_data = {}
        if getattr(t, "metrics", None):
            metric_data.update(t.metrics)
        if getattr(t, "reward", None) is not None:
            metric_data["reward"] = t.reward
        if metric_data:
            metrics_list.append(metric_data)

    if not metrics_list:
        print(
            "No metrics or rewards found in valid trajectories. Skipping metrics calculation."
        )
        return

    metrics_df = pl.DataFrame(metrics_list)

    avg_metrics = metrics_df.select(
        [
            pl.mean(c).alias(c)
            for c in metrics_df.columns
            if metrics_df[c].dtype != pl.Null
        ]
    ).with_columns(pl.lit(len(valid_trajectories)).alias("n_trajectories"))

    # Log metrics to file
    project_name = model.project
    model_name = model.name
    log_dir = Path("validation_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{project_name}-{model_name}.log"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"Timestamp: {timestamp}, Global Step: {global_step}\nMetrics:\n{avg_metrics.to_pandas().to_string()}\n---\n"

    with open(log_file_path, "a") as f:
        f.write(log_entry)

    print(f"Validation metrics logged to {log_file_path}")

    return avg_metrics


# The model/LoRA is saved at .art/project_name/models/model_name
# Function to run training.
async def run_training(
    model_name: str,
    project_name: str,
    current_project_policy_config: ProjectPolicyConfig,
):
    model = art.TrainableModel(
        name=model_name,
        project=project_name,
        base_model=current_project_policy_config.base_model,
        config=current_project_policy_config,
    )

    backend = LocalBackend()
    await model.register(backend)

    print(f"Registered model: {model.name} in project: {model.project}")

    assert hasattr(model, "config")
    config = model.config
    assert hasattr(config, "training_config") and config.training_config is not None
    train_cfg = config.training_config
    prompt_template = config.prompt_template
    if prompt_template is not None:
        prompt_template = get_prompt_template(prompt_template)

    # Use dataloader
    train_iterator = load_and_iterate_hf_dataset(
        dataset_path=config.training_dataset_path,
        dataset_split=config.training_dataset_split,
        prompt_column=config.training_prompt_column,
        answer_column=config.training_answer_column,
        prompt_template=prompt_template,
        batch_size=train_cfg.groups_per_step,
        num_epochs=train_cfg.num_epochs,
        max_samples=config.max_training_samples,
    )

    for batch, epoch, global_step, epoch_step in train_iterator:
        print(
            f"Epoch: {epoch}, Global Step: {global_step}, Epoch Step: {epoch_step}, Batch Size: {len(batch)}"
        )
        # Run validation after each eval_steps
        if (global_step > 0 and global_step % train_cfg.eval_steps == 0): # fmt: skip # ensure global_step > 0
            validation_results = await run_validation(model, global_step)
            if validation_results is not None:
                print("Validation Metrics from run_validation return:")
                print(validation_results)
        trajectory_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    (
                        rollout(model, scenario)
                        for _ in range(train_cfg.trajectories_per_group)
                    )
                )
                for scenario in batch
            ),
            pbar_desc=f"Gathering trajectories for step {global_step}",
        )

        if not trajectory_groups:
            print(
                "No trajectory groups were generated. Skipping training for this step."
            )
            continue

        await model.train(
            trajectory_groups,
            config=art.TrainConfig(learning_rate=train_cfg.learning_rate),
        )

    # Run validation after training
    if (global_step > 0 and global_step % train_cfg.eval_steps == 0): # fmt: skip # ensure global_step > 0
        validation_results = await run_validation(model, global_step)
        if validation_results is not None:
            print("Validation Metrics from run_validation return:")
            print(validation_results)

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training or debugging for slug-search agent."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with minimal configuration.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="default_query_prompt",
        help=f"Prompt template key to use from prompts.json (default: 'default_query_prompt').",
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default="check_answer_correctness_multi_gt",
        help="Verifier to use for training.",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Chat template to use for training.",
    )
    args = parser.parse_args()

    if args.prompt_template:
        project_policy_config.prompt_template = args.prompt_template

    if args.verifier:
        project_policy_config.verifier = args.verifier

    if args.chat_template:
        if args.chat_template not in chat_templates:
            raise ValueError(
                f"Chat template key '{args.chat_template}' not found in chat_templates.json. Available keys: {list(chat_templates.keys())}"
            )
        project_policy_config.custom_chat_template = chat_templates[args.chat_template]

    if args.debug:
        model_name = "slug-search-agent-debug"
        project_name = "slug_search_project_debug"

        # Create a deep copy of the global project_policy_config for modification
        debug_project_policy_config = project_policy_config.model_copy(deep=True)

        # Modify attributes for debugging
        debug_project_policy_config.max_training_samples = 1
        debug_project_policy_config.max_val_samples = 1
        # Also modify the nested training_config for debugging
        debug_project_policy_config.training_config.eval_steps = 1
        debug_project_policy_config.training_config.trajectories_per_group = 1
        debug_project_policy_config.training_config.groups_per_step = 1
        debug_project_policy_config.training_config.num_epochs = 1

        print("Running in DEBUG mode with minimal configuration.")
        asyncio.run(run_training(model_name, project_name, debug_project_policy_config))
    else:
        model_name = "slug-search-agent-001"
        project_name = "slug_search_project"
        asyncio.run(run_training(model_name, project_name, project_policy_config))

    # garbage collection
    gc.collect()
