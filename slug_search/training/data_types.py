from typing import List, Optional
from pydantic import BaseModel

# fmt: off

class VLLMConfig(BaseModel):
    """Configuration for vLLM engine arguments"""
    enforce_eager: bool = False
    gpu_memory_utilization: float = 0.95
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None


class TrainingConfig(BaseModel):
    trajectories_per_group: int = 6
    groups_per_step: int = 1  # This will be the batch_size for the dataloader
    learning_rate: float = 1.2e-5
    eval_steps: int = 30
    num_epochs: int = 1  # Number of times to iterate over the dataset
    beta: float = 0.04
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    lr_scheduler_type: str = "constant_with_warmup"


class PeftArgs(BaseModel):
    r: int = 8,
    lora_alpha: int = 16


class ProjectPolicyConfig(BaseModel):
    base_model: str  # No default value; must be provided
    # TODO: add sampling configs
    temperature: float = 1.0
    top_p: float = 1.0 # TODO: check R1-searcher
    max_tool_calls: int = 5
    max_tokens: int = 2048
    log_to_openpipe: bool = False
    use_tools: bool = True
    training_config: TrainingConfig | None = None
    vllm_config: VLLMConfig = VLLMConfig()  # Add vLLM configuration
    peft_args: PeftArgs = PeftArgs()
    custom_chat_template: str | None = None
    system_prompt: str | None = None
    prompt_template: str | None = None  # The prompt template string to use for training
    verifier: str | None = None  # The verifier to use for training. A function name from verifiers.py (see slug_search/training/verifiers.py)
    # Training dataset specific parameters
    training_dataset_path: str = "default/dataset-path"  # Placeholder, needs to be configured
    training_dataset_split: str = "train"  # Placeholder
    training_prompt_column: str = "prompt"  # Placeholder
    training_answer_column: str = "answer"  # Placeholder
    max_training_samples: Optional[int] = None  # Max samples to load from HF dataset, overrides training_dataset_size for loader
    max_training_steps: int = 100
    # Validation dataset specific parameters
    val_dataset_path: str = "default/dataset-path"  # Placeholder, needs to be configured
    val_dataset_split: str = "validation"  # Placeholder
    val_prompt_column: str = "prompt"  # Placeholder
    val_answer_column: str = "answer"  # Placeholder
    max_val_samples: Optional[int] = None  # Max samples to load from HF dataset, overrides val_dataset_size for loader
    max_val_steps: int = 100


class SearchQuery(BaseModel):
    id: int
    query: str
    answer: str  # The answer is a json string since it could be a list of strings.
