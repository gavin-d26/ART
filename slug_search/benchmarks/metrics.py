import json
from typing import Callable, List, Union, Dict, Any


# --- Generation Metric Functions ---
def check_answer_correctness_multi_gt(
    answer: str, correct_answer: Union[str, list]
) -> bool:
    """
    Check if generated answer contains all required ground truth elements.
    Adapted from slug_search.training.verifiers.

    Args:
        answer: Generated answer string
        correct_answer: Either a JSON string of list or a list directly

    Returns:
        True if answer contains all ground truth elements, False otherwise
    """
    if not isinstance(answer, str) or answer.startswith("Error:"):
        return False

    # Handle both JSON string and direct list input
    if isinstance(correct_answer, str):
        try:
            correct_answer_list = json.loads(correct_answer)
        except json.JSONDecodeError:
            # If it's not valid JSON, treat as single string
            correct_answer_list = [correct_answer]
    elif isinstance(correct_answer, list):
        correct_answer_list = correct_answer
    else:
        # Single item that's not a string or list
        correct_answer_list = [str(correct_answer)]

    # Check if all ground truth elements are present in the answer
    for answer_item in correct_answer_list:
        if str(answer_item) not in answer:
            return False
    return True


# --- Retrieval Evaluation Metrics ---


def ground_truth_hit_rate(ground_truth_analysis: Dict) -> float:
    """
    Calculate hit rate (whether any ground truth chunk was retrieved).

    Args:
        ground_truth_analysis: Output from check_if_ground_truth_retrieved()

    Returns:
        1.0 if any ground truth chunk was retrieved, 0.0 otherwise
    """
    return 1.0 if ground_truth_analysis.get("ground_truth_retrieved", False) else 0.0


def ground_truth_precision(ground_truth_analysis: Dict) -> float:
    """
    Calculate precision of ground truth retrieval.

    Args:
        ground_truth_analysis: Output from check_if_ground_truth_retrieved()

    Returns:
        Proportion of retrieved chunks that are ground truth
    """
    total = ground_truth_analysis.get("total_retrieved", 0)
    if total == 0:
        return 0.0
    gt_count = ground_truth_analysis.get("num_ground_truth_chunks", 0)
    return gt_count / total


def ground_truth_count(ground_truth_analysis: Dict) -> int:
    """
    Count of ground truth chunks retrieved.

    Args:
        ground_truth_analysis: Output from check_if_ground_truth_retrieved()

    Returns:
        Number of ground truth chunks retrieved
    """
    return ground_truth_analysis.get("num_ground_truth_chunks", 0)


# --- Enhanced Evaluator Function ---


def evaluate_results(
    results_file_path: str,
    metric_functions: List[Callable],
    output_file_path: str,
) -> None:
    """
    Enhanced evaluation function that handles both generation and retrieval metrics.

    Args:
        results_file_path: Path to the .jsonl file containing generation results.
                           Each line should be a JSON object with at least
                           "generated_answer" (str) and "actual_answer" (str or List[str]).
                           For retrieval metrics, should also contain "ground_truth_analysis" (Dict).
        metric_functions: A list of functions, where each function takes either:
                          - (generated_answer: str, actual_answer: Union[str, List[str]]) for generation metrics
                          - (ground_truth_analysis: Dict) for retrieval metrics
                          and returns a float or int.
        output_file_path: Path to save the evaluated results (as .jsonl).
    """
    evaluated_results = []
    try:
        with open(results_file_path, "r", encoding="utf-8") as f_in:
            for line_number, line in enumerate(f_in, 1):
                try:
                    result_item = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(
                        f"Skipping invalid JSON line {line_number}: {line.strip()} - Error: {e}"
                    )
                    # Optionally, store problematic lines or handle them differently
                    malformed_item = {
                        "error": "JSONDecodeError",
                        "line": line_number,
                        "content": line.strip(),
                        "details": str(e),
                    }
                    evaluated_results.append(malformed_item)
                    continue

                generated_answer = result_item.get("generated_answer")
                actual_answer = result_item.get(
                    "actual_answer"
                )  # This can be str or list
                ground_truth_analysis = result_item.get("ground_truth_analysis", {})

                # It's crucial that generated_answer is a string for most metrics.
                # The benchmarking script aims for this, but errors might make it non-string.
                if not isinstance(generated_answer, str):
                    # If generated_answer is None or not a string, treat as an error state for metrics.
                    # Some metrics might handle None, but it's safer to standardize.
                    print(
                        f"Warning: 'generated_answer' is not a string (type: {type(generated_answer)}) for item at line {line_number} (query: '{result_item.get('query', 'N/A')}'). Metrics might yield 0 or error."
                    )
                    # Force it to an empty string if it's None, to avoid downstream errors in metric_fn,
                    # or let metric_fn handle None if they are designed to.
                    # Most of our current metrics will return 0.0 if generated_answer.startswith("Error:") or is not str.
                    # So, if it's None, they might fail. Let's ensure it's at least an empty string if None.
                    if generated_answer is None:
                        generated_answer = (
                            ""  # Or a special marker like "GENERATION_UNAVAILABLE"
                        )

                # Apply all metric functions
                for metric_fn in metric_functions:
                    metric_name = metric_fn.__name__
                    try:
                        # Check if this is a generation metric or retrieval metric
                        if metric_name.startswith("ground_truth_"):
                            # Retrieval metric - pass ground_truth_analysis
                            metric_value = metric_fn(ground_truth_analysis)
                        else:
                            # Generation metric - pass generated and actual answers
                            current_gen_ans = (
                                generated_answer
                                if isinstance(generated_answer, str)
                                else str(generated_answer or "")
                            )
                            metric_value = metric_fn(current_gen_ans, actual_answer)

                        result_item[metric_name] = metric_value
                    except Exception as e:
                        print(
                            f"Error calculating metric '{metric_name}' for query '{result_item.get('query', 'N/A')}' (line {line_number}): {e}"
                        )
                        result_item[metric_name] = (
                            None  # Or some error placeholder like "METRIC_ERROR"
                        )

                evaluated_results.append(result_item)

        # Save results
        with open(output_file_path, "w", encoding="utf-8") as f_out:
            for item in evaluated_results:
                f_out.write(json.dumps(item) + "\n")
        print(f"Evaluation complete. Results saved to {output_file_path}")

    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")


if __name__ == "__main__":
    # Create a dummy results.jsonl for testing
    dummy_results_content = [
        {
            "query": "What is the capital of France?",
            "generated_answer": "Paris",
            "actual_answer": "Paris",
            "tokens": 1,
        },
        {
            "query": "What is 2+2?",
            "generated_answer": "5",
            "actual_answer": "4",
            "tokens": 1,
        },
        {
            "query": "Best fruit?",
            "generated_answer": "banana",
            "actual_answer": ["orange", "banana"],
            "tokens": 1,
        },
        {
            "query": "Error query",
            "generated_answer": "Error: Model failed",
            "actual_answer": "Some answer",
            "tokens": 0,
        },
        {
            "query": "List query",
            "generated_answer": "item1",
            "actual_answer": ["item1", "item2"],
            "tokens": 1,
        },
        {
            "query": "None gen",
            "generated_answer": None,
            "actual_answer": "Valid answer",
            "tokens": 0,
        },
        {
            "query": "Numerical actual",
            "generated_answer": "123",
            "actual_answer": 123,
            "tokens": 1,
        },
    ]
    dummy_input_path = "dummy_benchmark_results.jsonl"
    dummy_output_path = "evaluated_dummy_results.jsonl"

    with open(dummy_input_path, "w", encoding="utf-8") as f:
        for item in dummy_results_content:
            f.write(json.dumps(item) + "\n")
        # Add an invalid JSON line for testing error handling
        f.write("this is not valid json\n")
        f.write(
            json.dumps(
                {
                    "query": "Last valid",
                    "generated_answer": "yes",
                    "actual_answer": "yes",
                }
            )
            + "\n"
        )

    # Add some dummy ground truth analysis data for testing retrieval metrics
    dummy_results_with_gt = []
    for item in dummy_results_content:
        # Add dummy ground truth analysis
        generated_answer = item.get("generated_answer", "") or ""
        item["ground_truth_analysis"] = {
            "ground_truth_retrieved": (True if "Paris" in generated_answer else False),
            "num_ground_truth_chunks": (1 if "Paris" in generated_answer else 0),
            "num_other_chunks": 2,
            "total_retrieved": 3,
        }
        dummy_results_with_gt.append(item)

    # Rewrite the dummy file with ground truth analysis
    with open(dummy_input_path, "w", encoding="utf-8") as f:
        for item in dummy_results_with_gt:
            f.write(json.dumps(item) + "\n")
        # Add an invalid JSON line for testing error handling
        f.write("this is not valid json\n")
        f.write(
            json.dumps(
                {
                    "query": "Last valid",
                    "generated_answer": "yes",
                    "actual_answer": "yes",
                    "ground_truth_analysis": {
                        "ground_truth_retrieved": False,
                        "num_ground_truth_chunks": 0,
                        "num_other_chunks": 1,
                        "total_retrieved": 1,
                    },
                }
            )
            + "\n"
        )

    # Test both generation and retrieval metrics
    metrics_to_run = [
        ground_truth_hit_rate,
        ground_truth_precision,
        ground_truth_count,  # Retrieval metrics
    ]

    print(
        f"\nRunning example evaluation with both generation and retrieval metrics on '{dummy_input_path}'..."
    )
    evaluate_results(dummy_input_path, metrics_to_run, dummy_output_path)

    print(f"\nDummy evaluation finished. Check '{dummy_output_path}'.")
    print("Contents of the evaluated file:")
    try:
        with open(dummy_output_path, "r", encoding="utf-8") as f:
            for line in f:
                print(line.strip())
    except FileNotFoundError:
        print(f"Could not find output file {dummy_output_path} to display.")

    # Clean up dummy files
    import os

    try:
        os.remove(dummy_input_path)
        # os.remove(dummy_output_path) # Keep for inspection or remove
        print(
            f"\nCleaned up {dummy_input_path}. You may want to inspect or remove {dummy_output_path}."
        )
    except OSError as e:
        print(f"Error cleaning up dummy files: {e}")
