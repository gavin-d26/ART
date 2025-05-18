import json
from typing import Callable, List, Union, Dict, Any

# --- Metric Functions ---


def exact_match(generated_answer: str, actual_answer: Union[str, List[str]]) -> float:
    """
    Checks for exact match between generated_answer and actual_answer.
    Returns 0.0 if generated_answer indicates an error or is not a string.
    If actual_answer is a list, checks if generated_answer is in the list (elements converted to string).
    If actual_answer is a string, checks for direct equality.
    Handles cases where actual_answer might be other basic types by converting to string.
    """
    if not isinstance(generated_answer, str) or generated_answer.startswith("Error:"):
        return 0.0

    if isinstance(actual_answer, list):
        # Ensure all elements in actual_answer are strings for comparison
        return 1.0 if generated_answer in [str(a) for a in actual_answer] else 0.0
    elif isinstance(actual_answer, str):
        return 1.0 if generated_answer == actual_answer else 0.0
    elif actual_answer is None:
        return 0.0
    else:  # If actual_answer is neither list nor string (e.g. int, float directly)
        return 1.0 if generated_answer == str(actual_answer) else 0.0


def rouge_l_score(generated_answer: str, actual_answer: Union[str, List[str]]) -> float:
    """
    Placeholder for ROUGE-L F1 score.
    Requires 'rouge_score' library for actual implementation.
    Returns -1.0 as a placeholder.
    """
    if not isinstance(generated_answer, str) or generated_answer.startswith("Error:"):
        return 0.0
    # Actual implementation would be:
    # from rouge_score import rouge_scorer
    # scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # ref_str = ""
    # if isinstance(actual_answer, list):
    #     ref_str = str(actual_answer[0]) if actual_answer else ""
    # elif isinstance(actual_answer, str):
    #     ref_str = actual_answer
    # else:
    #     ref_str = str(actual_answer)
    # if not generated_answer.strip() and not ref_str.strip(): return 1.0
    # if not ref_str.strip() or not generated_answer.strip(): return 0.0
    # scores = scorer.score(ref_str, generated_answer)
    # return scores['rougeL'].fmeasure
    print(
        "Warning: rouge_l_score is a placeholder. Install and import 'rouge_score' for actual calculation."
    )
    return -1.0


def bleu_score(generated_answer: str, actual_answer: Union[str, List[str]]) -> float:
    """
    Placeholder for BLEU score.
    Requires 'nltk' library for actual implementation.
    Returns -1.0 as a placeholder.
    """
    if not isinstance(generated_answer, str) or generated_answer.startswith("Error:"):
        return 0.0
    # Actual implementation would be:
    # from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    # candidate_tokens = generated_answer.split()
    # reference_tokens_list = []
    # if isinstance(actual_answer, list):
    #     reference_tokens_list = [str(ref).split() for ref in actual_answer if str(ref).strip()]
    # elif isinstance(actual_answer, str):
    #     reference_tokens_list = [actual_answer.split()] if actual_answer.strip() else []
    # else:
    #     ref_str = str(actual_answer)
    #     reference_tokens_list = [ref_str.split()] if ref_str.strip() else []
    # if not candidate_tokens and not any(reference_tokens_list): return 1.0
    # if not reference_tokens_list or not candidate_tokens : return 0.0
    # chencherry = SmoothingFunction()
    # return sentence_bleu(reference_tokens_list, candidate_tokens, smoothing_function=chencherry.method1)
    print(
        "Warning: bleu_score is a placeholder. Install and import 'nltk' for actual calculation."
    )
    return -1.0


# --- Evaluator Function ---


def evaluate_results(
    results_file_path: str,
    metric_functions: List[Callable[[str, Union[str, List[str]]], float]],
    output_file_path: str,
) -> None:
    """
    Evaluates generation results from a .jsonl file using a list of metric functions.

    Args:
        results_file_path: Path to the .jsonl file containing generation results.
                           Each line should be a JSON object with at least
                           "generated_answer" (str) and "actual_answer" (str or List[str]).
        metric_functions: A list of functions, where each function takes
                          (generated_answer: str, actual_answer: Union[str, List[str]])
                          and returns a float.
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

                for metric_fn in metric_functions:
                    metric_name = metric_fn.__name__
                    try:
                        # Ensure generated_answer is a string before passing to metrics that expect strings
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

    metrics_to_run = [exact_match, rouge_l_score, bleu_score]

    print(f"\nRunning example evaluation on '{dummy_input_path}'...")
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
