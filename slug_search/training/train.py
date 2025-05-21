import gc
import art
from art.local import LocalBackend
import asyncio
from dotenv import load_dotenv
from typing import List
import argparse  # Added for command line argument parsing
import json
from pathlib import Path

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


def get_prompt_template(key: str) -> str:
    if key not in prompt_templates:
        raise ValueError(
            f"Prompt template key '{key}' not found in prompts.json. Available keys: {list(prompt_templates.keys())}"
        )
    return prompt_templates[key]


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
    args = parser.parse_args()

    if args.prompt_template:
        project_policy_config.prompt_template = args.prompt_template

    if args.debug:
        model_name = "slug-search-agent-debug"
        project_name = "slug_search_project_debug"

        # Create a deep copy of the global project_policy_config for modification
        debug_project_policy_config = project_policy_config.model_copy(deep=True)

        # Modify attributes for debugging
        debug_project_policy_config.max_training_samples = 1
        debug_project_policy_config.max_val_samples = 1
        # Also modify the nested training_config for debugging
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
