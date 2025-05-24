# Benchmarking Guide: Flexible Evaluation System

This system supports **any dataset, pipeline, or metric** out of the box. Simply add new components and they work immediately - no configuration needed!

## 🚀 Quick Start

### Basic Benchmarking with Key Metrics
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
  --summary
```

### Benchmarking with All Available Metrics
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
  --metrics "check_answer_correctness_multi_gt;ground_truth_hit_rate;ground_truth_precision;ground_truth_count" \
  --summary \
  --max_queries 100
```

## 🔧 System Flexibility

### ✅ **Add Any Dataset** - Works Immediately
```bash
# Any HuggingFace dataset
--dataset_path microsoft/ms_marco --dataset_split train --query_column query --answer_column answers

# Any local dataset  
--dataset_path ./my_dataset.jsonl --query_column question --answer_column ground_truth
```

### ✅ **Add Any Pipeline** - Works Immediately
```python
# In pipelines.py - add any new pipeline class
class MyCustomPipeline:
    def __init__(self, **kwargs):  # Accepts any parameters
        # Your initialization
        
    async def run_pipeline(self, query: str) -> dict:
        # Your logic
        return {"generation": "...", "retrieved_chunks": [...]}
```
```bash
# Use immediately in benchmarking
--pipeline_name MyCustomPipeline --my_custom_param value
```

### ✅ **Add Any Metric** - Works Immediately  
```python
# In metrics.py - add any new metric function
def my_custom_metric(ground_truth_analysis: Dict) -> float:
    """Automatically discovered and available!"""
    return some_calculation()
```
```bash
# Use immediately in benchmarking
--metrics "my_custom_metric;ground_truth_hit_rate"
```

## 📊 Current Available Metrics

**Auto-discovered from `metrics.py`**: `check_answer_correctness_multi_gt`, `ground_truth_hit_rate`, `ground_truth_precision`, `ground_truth_count`

## 🎯 Common Usage Patterns

### Quick Health Check
```bash
--metrics "ground_truth_hit_rate;check_answer_correctness_multi_gt" --max_queries 50
```

### Full Analysis
```bash
--metrics "check_answer_correctness_multi_gt;ground_truth_hit_rate;ground_truth_precision;ground_truth_count" --summary
```

## 📈 Output & Results

### Files Generated
- **`benchmark_results.jsonl`**: Individual query results with computed metrics
- **`benchmark_results_summary.json`**: Statistical summary (when using `--summary`)

### Summary Example
```
EVALUATION SUMMARY
============================================================
Total Queries: 100

METRICS SUMMARY:
ground_truth_hit_rate: Mean: 0.85, Std: 0.36
check_answer_correctness_multi_gt: Mean: 0.72, Std: 0.45

KEY INSIGHTS:
1. Excellent retrieval performance: 85.0% hit rate
2. Good generation quality: 72.0% correct answers
```

## 🛠️ Key Options

```bash
--max_queries 50                    # Limit for testing
--concurrency_limit 10              # Control memory usage  
--results_output_path custom.jsonl  # Custom output file
--summary                           # Generate statistical summary
```

## 🚨 Quick Troubleshooting

- **"Unknown metric"**: Check spelling, use `;` to separate metrics
- **"Summary but no metrics"**: Must use `--metrics` with `--summary`  
- **Memory issues**: Lower `--concurrency_limit` or `--max_queries`

## 💡 Pro Tips

- **Test first**: Use `--max_queries 10` for quick validation
- **Any dataset works**: System handles ID generation automatically
- **Any pipeline works**: Uses **kwargs for maximum compatibility
- **Any metric works**: Just add function to `metrics.py` - auto-discovered!