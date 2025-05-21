import math
import random
import json
from typing import List, Generator, Tuple
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from .data_types import SearchQuery


def load_and_iterate_hf_dataset(
    dataset_path: str,
    dataset_split: str,
    prompt_column: str,
    answer_column: str,
    prompt_template: str | None = None,
    batch_size: int = 1,
    num_epochs: int = 1,
    initial_step: int = 0,
    use_tqdm: bool = True,
    max_samples: int | None = None,
) -> Generator[Tuple[List[SearchQuery], int, int, int], None, None]:
    """
    Loads a Hugging Face dataset, extracts prompt and answer columns, creates SearchQuery objects,
    and generates batches over multiple epochs with deterministic shuffling.

    Args:
        dataset_path: Path or name of the Hugging Face dataset.
        dataset_split: Dataset split to use (e.g., 'train', 'validation', 'test').
        prompt_column: Column in the dataset containing the prompts/questions.
        answer_column: Column in the dataset containing the ground truth answers.
        prompt_template: Template for formatting the prompt. Must contain {{query}} as a placeholder for the prompt text. If None, the prompt is used as is.
        batch_size: The number of SearchQuery objects in each batch. Defaults to 1.
        num_epochs: The number of times to iterate over the dataset. Defaults to 1.
        initial_step: The global step number to start from. Defaults to 0.
        use_tqdm: Whether to display a progress bar. Defaults to True.
        max_samples: Optional maximum number of samples to load from the dataset.

    Yields:
        A tuple containing:
        - batch (List[SearchQuery]): The list of SearchQuery objects for the current batch.
        - epoch (int): The current epoch number (0-indexed).
        - global_step (int): The overall step number across all epochs.
        - epoch_step (int): The step number within the current epoch (0-indexed).
    """
    try:
        hf_dataset: Dataset = load_dataset(dataset_path, split=dataset_split)  # type: ignore
    except Exception as e:
        print(
            f"Failed to load dataset: {dataset_path} with split {dataset_split}. Error: {e}"
        )
        return

    # Use dataset slicing if max_samples is provided
    if max_samples is not None:
        try:
            hf_dataset = hf_dataset.select(range(min(len(hf_dataset), max_samples)))
        except Exception as e:
            print(
                f"Warning: Could not slice dataset for max_samples={max_samples}: {e}"
            )
            # fallback to using the full dataset

    processed_dataset: List[SearchQuery] = []
    for i, record in enumerate(hf_dataset):
        try:
            prompt_text = record.get(prompt_column)
            answer_text = record.get(answer_column)
            if prompt_text is None:
                print(
                    f"Warning: Prompt column '{prompt_column}' not found or is null in record {i}. Skipping."
                )
                continue
            if answer_text is None:
                print(
                    f"Warning: Answer column '{answer_column}' not found or is null in record {i}. Skipping."
                )
                continue
            # Format the prompt using the prompt_template if provided
            if prompt_template is not None:
                formatted_prompt = prompt_template.replace(
                    "{{query}}", str(prompt_text)
                )
            else:
                formatted_prompt = str(prompt_text)
            processed_dataset.append(
                SearchQuery(
                    id=i,
                    query=formatted_prompt,
                    answer=json.dumps(answer_text), # return the answer as a json string since it could be a list of strings. # fmt: skip
                )
            )
        except Exception as e:
            print(f"Warning: Error processing record {i}: {e}. Skipping record.")
            continue

    dataset_size = len(processed_dataset)
    if dataset_size == 0:
        print("No data loaded or processed. Exiting dataloader.")
        return

    steps_per_epoch = math.ceil(dataset_size / batch_size)
    total_steps = steps_per_epoch * num_epochs

    progress_bar = None
    if use_tqdm:
        progress_bar = tqdm(
            initial=initial_step,
            total=total_steps,
            desc=f"Iterating HF dataset ({dataset_split})",
            unit="batch",
        )

    for epoch in range(num_epochs):
        indices = list(range(dataset_size))
        random.seed(epoch)  # Deterministic shuffle per epoch
        random.shuffle(indices)

        for i in range(0, dataset_size, batch_size):
            epoch_step = i // batch_size
            global_step = epoch * steps_per_epoch + epoch_step

            if global_step < initial_step:
                if progress_bar:
                    pass
                continue

            batch_indices = indices[i : i + batch_size]
            batch = [processed_dataset[idx] for idx in batch_indices]
            yield batch, epoch, global_step, epoch_step

            if progress_bar:
                progress_bar.update(1)

    if progress_bar:
        progress_bar.close()


if __name__ == "__main__":
    print("Testing dataloader...")
    try:
        print("Attempting to load 'glue/mrpc' for testing the dataloader structure.")
        data_iterator = load_and_iterate_hf_dataset(
            dataset_path="lucadiliello/hotpotqa",
            dataset_split="train",
            prompt_column="question",
            answer_column="answers",
            batch_size=5,
            num_epochs=2,
            max_samples=20,
            initial_step=2,
        )
        for batch_data, epoch_num, global_s, epoch_s in data_iterator:
            print(
                f"Epoch: {epoch_num}, Global Step: {global_s}, Epoch Step: {epoch_s}, Batch Size: {len(batch_data)}"
            )
            # for item in batch_data:
            #     print(f"  ID: {item.id}, Query: {item.query[:30]}..., Answer: {item.answer[:30]}...")
        print("Dataloader test finished.")
    except ImportError:
        print("The 'datasets' library is not installed. Skipping live example usage.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
