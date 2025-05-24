import json
import numpy as np
from typing import Dict, List, Any


def generate_evaluation_summary(results_file_path: str) -> Dict[str, Any]:
    """Generate summary statistics from evaluation results."""
    results = []
    with open(results_file_path, "r") as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    summary = {"total_queries": len(results), "metrics_summary": {}, "insights": []}

    # Calculate statistics for each metric
    for result in results:
        for key, value in result.items():
            if key.endswith(
                (
                    "_match",
                    "_rate",
                    "_precision",
                    "_count",
                    "_score",
                    "_correctness_multi_gt",
                )
            ) and isinstance(value, (int, float)):
                if key not in summary["metrics_summary"]:
                    summary["metrics_summary"][key] = []
                summary["metrics_summary"][key].append(value)

    # Compute mean, std, min, max for each metric
    for metric, values in summary["metrics_summary"].items():
        if values:  # Only compute if we have values
            summary["metrics_summary"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

    # Generate insights
    summary["insights"] = generate_insights(summary["metrics_summary"])

    return summary


def generate_insights(metrics_summary: Dict[str, Dict[str, float]]) -> List[str]:
    """Generate actionable insights from metrics."""
    insights = []

    # Ground truth retrieval insights
    if "ground_truth_hit_rate" in metrics_summary:
        hit_rate = metrics_summary["ground_truth_hit_rate"]["mean"]
        if hit_rate > 0.8:
            insights.append(f"Excellent retrieval performance: {hit_rate:.1%} hit rate")
        elif hit_rate > 0.6:
            insights.append(f"Good retrieval performance: {hit_rate:.1%} hit rate")
        else:
            insights.append(
                f"Retrieval needs improvement: only {hit_rate:.1%} hit rate"
            )

    # Generation quality insights
    if "check_answer_correctness_multi_gt" in metrics_summary:
        correctness = metrics_summary["check_answer_correctness_multi_gt"]["mean"]
        if correctness > 0.7:
            insights.append(
                f"Strong generation quality: {correctness:.1%} correct answers"
            )
        elif correctness > 0.5:
            insights.append(
                f"Moderate generation quality: {correctness:.1%} correct answers"
            )
        else:
            insights.append(
                f"Generation quality needs improvement: {correctness:.1%} correct answers"
            )

    # Precision insights
    if "ground_truth_precision" in metrics_summary:
        precision = metrics_summary["ground_truth_precision"]["mean"]
        if precision > 0.8:
            insights.append(
                f"High retrieval precision: {precision:.1%} of retrieved chunks are relevant"
            )
        elif precision > 0.5:
            insights.append(
                f"Moderate retrieval precision: {precision:.1%} of retrieved chunks are relevant"
            )
        else:
            insights.append(
                f"Low retrieval precision: only {precision:.1%} of retrieved chunks are relevant"
            )

    # Ground truth count insights
    if "ground_truth_count" in metrics_summary:
        avg_count = metrics_summary["ground_truth_count"]["mean"]
        max_count = metrics_summary["ground_truth_count"]["max"]
        insights.append(
            f"Average {avg_count:.1f} ground truth chunks retrieved per query (max: {max_count:.0f})"
        )

    return insights


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted summary to console."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Queries: {summary['total_queries']}")
    print()

    if summary["metrics_summary"]:
        print("METRICS SUMMARY:")
        print("-" * 40)
        for metric, stats in summary["metrics_summary"].items():
            print(f"{metric}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
            print()

    if summary["insights"]:
        print("KEY INSIGHTS:")
        print("-" * 40)
        for i, insight in enumerate(summary["insights"], 1):
            print(f"{i}. {insight}")
        print()

    print("=" * 60)


def save_summary(summary: Dict[str, Any], output_path: str) -> None:
    """Save summary to JSON file."""
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
