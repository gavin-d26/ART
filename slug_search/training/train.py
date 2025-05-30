import argparse  # Added for command line argument parsing
import asyncio
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from tabulate import tabulate

import art
from art.local import LocalBackend

import wandb  # Only import if needed in the function

from .data_types import ProjectPolicyConfig, SearchQuery, TrainingConfig
from .dataloaders import load_and_iterate_hf_dataset
from .rollout import rollout
from .search_tools import configure_search_tools

load_dotenv()

# Configure search tools (will be updated with command line args later)
configure_search_tools(
    milvus_db_path="slug_search/data/milvus_hotpotqa_training.db",
    embedding_model_name="BAAI/bge-large-en-v1.5",
    embedder_api_base="http://localhost:40002/v1",
    embedder_api_key_env_var="EMBEDDER_API_KEY",
)

BASE_MODEL_NAME = "unsloth/Qwen3-4B"  # Define base model name as a constant

# Global configurations
training_config = TrainingConfig(
    trajectories_per_group=2,
    groups_per_step=2,
    learning_rate=1e-5,
    eval_steps=10,
    num_epochs=1,
    beta=0.001,
)

project_policy_config = ProjectPolicyConfig(
    base_model=BASE_MODEL_NAME,
    temperature=1.0,
    top_p=1.0,
    max_tool_calls=5,
    max_tokens=4096,
    log_to_openpipe=False,
    use_tools=True,
    training_config=training_config,
    prompt_template="default_query_prompt_2",
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

# Global dict for wandb runs
_wandb_runs = {}


def get_prompt_template(key: str) -> str:
    if key not in prompt_templates:
        raise ValueError(
            f"Prompt template key '{key}' not found in prompts.json. Available keys: {list(prompt_templates.keys())}"
        )
    return prompt_templates[key]


def log_metrics_from_trajectory_groups(
    trajectory_groups,
    project_name,
    model_name,
    global_step,
    split="val",
    config=None,
    commit=False,
):
    """
    Logs metrics from trajectory groups to a local file and optionally to wandb if WANDB_API_KEY_NO_ART is set.
    Args:
        trajectory_groups (list of lists): List of TrajectoryGroup (each a list of trajectories).
        project_name (str): Project name.
        model_name (str): Model name.
        global_step (int): Global step.
        split (str): 'train' or 'validation'.
        config (dict): Project policy configuration.
        commit (bool): Whether to commit the metrics to wandb.
    """
    assert split in {"train", "val"}
    all_metrics = {"reward": [], "exception_rate": []}
    n_trajectories = 0
    for group in trajectory_groups:
        for t in group:
            n_trajectories += 1
            if isinstance(t, BaseException):
                all_metrics["exception_rate"].append(1)
                continue
            else:
                all_metrics["exception_rate"].append(0)
            # Add reward metric
            if hasattr(t, "reward") and t.reward is not None:
                all_metrics["reward"].append(t.reward)
            # Collect other custom metrics
            if hasattr(t, "metrics") and t.metrics:
                for metric, value in t.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(float(value))
    # Calculate averages for all metrics
    averages = {}
    for metric, values in all_metrics.items():
        if values:
            averages[metric] = sum(values) / len(values)

    # Calculate reward std dev within groups
    def reward_std_dev(groups):
        import math

        rewards = []
        for group in groups:
            group_rewards = [
                t.reward
                for t in group
                if hasattr(t, "reward")
                and t.reward is not None
                and not isinstance(t, BaseException)
            ]
            if group_rewards:
                mean = sum(group_rewards) / len(group_rewards)
                var = sum((r - mean) ** 2 for r in group_rewards) / len(group_rewards)
                rewards.append(math.sqrt(var))
        if rewards:
            return sum(rewards) / len(rewards)
        return 0.0

    averages["reward_std_dev"] = reward_std_dev(trajectory_groups)
    averages["n_trajectories"] = n_trajectories
    # Local log file with split in name
    log_dir = Path("validation_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{project_name}-{model_name}-{split}.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_table = tabulate(
        [list(averages.values())], headers=list(averages.keys()), tablefmt="github"
    )
    log_entry = f"Timestamp: {timestamp}, Global Step: {global_step}, Split: {split}\nMetrics:\n{metrics_table}\n---\n"
    with open(log_file_path, "a") as f:
        f.write(log_entry)
    print(f"{split.capitalize()} metrics logged to {log_file_path}")
    # Optionally log to wandb if WANDB_API_KEY_NO_ART is set
    wandb_api_key = os.environ.get("WANDB_API_KEY_NO_ART")
    if wandb_api_key:
        import wandb

        global _wandb_runs
        run = _wandb_runs.get(model_name)
        if run is None or getattr(run, "_is_finished", False):
            run = wandb.init(
                project=project_name,
                name=model_name,
                id=model_name,
                resume="allow",
                config=config,
            )
            _wandb_runs[model_name] = run
            print(f"Wandb run initialized! You can view it at {run.url}")
        # Namespace metrics
        namespaced_metrics = {f"{split}/{k}": v for k, v in averages.items()}
        run.log(namespaced_metrics, step=global_step, commit=commit)
        if split == "val" and commit:
            run.finish()


# run a validation step between training epochs/steps.
async def run_validation(model: art.TrainableModel, global_step: int, commit=False):
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

    # Gather trajectory groups for validation, as in training
    trajectory_groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(
                (
                    rollout(model, scenario)
                    for _ in range(train_cfg.trajectories_per_group)
                )
            )
            for scenario in all_val_scenarios
        ),
        pbar_desc=f"Validation for {model.name}",
    )

    # Filter out empty groups
    valid_trajectory_groups = [g for g in trajectory_groups if g]
    if not valid_trajectory_groups:
        print(
            "No valid trajectory groups generated during validation. Skipping metrics calculation."
        )
        return

    log_metrics_from_trajectory_groups(
        valid_trajectory_groups,
        model.project,
        model.name,
        global_step,
        split="val",
        config=model.config.model_dump(),
        commit=commit,
    )


# The model/LoRA is saved at .art/project_name/models/model_name
# Function to run training.
async def run_training(
    model_name: str,
    project_name: str,
    current_project_policy_config: ProjectPolicyConfig,
):
    # Convert vLLM config to internal config
    internal_config = None
    if current_project_policy_config.vllm_config:
        vllm_dict = current_project_policy_config.vllm_config.model_dump()
        # Only include non-default values to keep config clean
        engine_args = {}

        # Only add values that differ from defaults or are explicitly set
        if vllm_dict.get("enforce_eager", False):
            engine_args["enforce_eager"] = True
        if vllm_dict.get("gpu_memory_utilization", 0.95) != 0.95:
            engine_args["gpu_memory_utilization"] = vllm_dict["gpu_memory_utilization"]
        if vllm_dict.get("max_model_len") is not None:
            engine_args["max_model_len"] = vllm_dict["max_model_len"]
        if vllm_dict.get("max_num_seqs") is not None:
            engine_args["max_num_seqs"] = vllm_dict["max_num_seqs"]

        if engine_args:
            internal_config = art.dev.InternalModelConfig(engine_args=engine_args)
            print(f"Using vLLM engine args: {engine_args}")

    model = art.TrainableModel(
        name=model_name,
        project=project_name,
        base_model=current_project_policy_config.base_model,
        config=current_project_policy_config,
        _internal_config=internal_config,
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
        if (global_step % train_cfg.eval_steps == 0): # fmt: skip # ensure global_step > 0
            await run_validation(model, global_step, commit=False)
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
            pbar_desc=f"Gathering trajectories for Global Step {global_step}",
        )

        if not trajectory_groups:
            print(
                "No trajectory groups were generated. Skipping training for this step."
            )
            continue

        # Log metrics using the new function
        log_metrics_from_trajectory_groups(
            trajectory_groups,
            model.project,
            model.name,
            global_step,
            split="train",
            config=model.config.model_dump(),
            commit=True,
        )

        await model.train(
            trajectory_groups,
            config=art.TrainConfig(learning_rate=train_cfg.learning_rate),
        )

    # Run validation after training
    if (global_step > 0 and global_step % train_cfg.eval_steps == 0): # fmt: skip # ensure global_step > 0
        await run_validation(model, global_step + 1, commit=True)

    model._backend.close()
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
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top documents to retrieve for search (default: 3).",
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

    # Reconfigure search tools with command line arguments
    configure_search_tools(
        milvus_db_path="slug_search/data/milvus_hotpotqa_training.db",
        embedding_model_name="BAAI/bge-large-en-v1.5",
        embedder_api_base="http://localhost:40002/v1",
        embedder_api_key_env_var="EMBEDDER_API_KEY",
        top_k=args.top_k,
    )

    if args.debug:
        model_name = "slug-search-agent-debug-3"
        project_name = "slug_search_project_debug"

        # Create a deep copy of the global project_policy_config for modification
        debug_project_policy_config = project_policy_config.model_copy(deep=True)

        # Modify attributes for debugging
        debug_project_policy_config.max_training_samples = 30
        debug_project_policy_config.max_val_samples = 10
        # Also modify the nested training_config for debugging
        debug_project_policy_config.training_config.eval_steps = 2
        debug_project_policy_config.training_config.trajectories_per_group = 8
        debug_project_policy_config.training_config.groups_per_step = 4
        debug_project_policy_config.training_config.num_epochs = 1
        debug_project_policy_config.training_config.learning_rate = 1e-6

        # Configure vLLM settings for debug mode
        debug_project_policy_config.vllm_config.enforce_eager = True
        debug_project_policy_config.vllm_config.gpu_memory_utilization = 0.95
        debug_project_policy_config.vllm_config.max_model_len = (
            4096  # Smaller context for faster debugging
        )

        print("Running in DEBUG mode with minimal configuration.")
        print(
            f"Debug vLLM config: enforce_eager={debug_project_policy_config.vllm_config.enforce_eager}, "
            f"gpu_memory_utilization={debug_project_policy_config.vllm_config.gpu_memory_utilization}"
        )
        asyncio.run(run_training(model_name, project_name, debug_project_policy_config))
        print("Debug training finished.")
    else:
        model_name = "slug-search-agent-001"
        project_name = "slug_search_project"
        print(
            f"Production vLLM config: enforce_eager={project_policy_config.vllm_config.enforce_eager}, "
            f"gpu_memory_utilization={project_policy_config.vllm_config.gpu_memory_utilization}"
        )
        asyncio.run(run_training(model_name, project_name, project_policy_config))
        print("Production training finished.")

    # garbage collection
    gc.collect()
    sys.exit(0)  # Force exit
