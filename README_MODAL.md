# Slug Search Training on Modal

This setup allows you to run your training script on Modal cloud with GPU support, exactly matching your local environment including the embedder server.

## Quick Start

1. **Setup Modal (one-time)**:
   ```bash
   ./setup_modal.sh
   ```

2. **Run your exact training command**:
   ```bash
   modal run train_modal.py::debug --log-file="final_test_$(date +%H%M%S).log"
   ```

This replicates your local command:
```bash
python -m slug_search.training.train --debug --chat_template "qwen3_preserve_thinking" 2>&1 | tee debug80.log
```

## What's Included

The Modal setup provides:

- ✅ **Exact dependency matching** using your `uv.lock` file
- ✅ **Embedder vLLM server** (BAAI/bge-large-en-v1.5) on localhost:40002
- ✅ **All your hardcoded configurations** from train.py
- ✅ **Persistent storage** for models, logs, and data
- ✅ **Real-time output streaming** 
- ✅ **Log file creation** (debug80.log)
- ✅ **H100 GPU** with 64GB RAM (configurable)

## Usage Commands

### Debug Training (Recommended)
```bash
modal run train_modal.py::debug
```
This runs both the embedder server and your training script in the same container, exactly matching your local setup.

### Production Training  
```bash
modal run train_modal.py::production
```

### Custom Training
```bash
modal run train_modal.py::custom --debug=true --chat_template="qwen3_preserve_thinking"
```

### Debug Without Embedder (Not Recommended)
```bash
modal run train_modal.py::debug_no_embedder
```

### Emergency Stop
```bash
modal run train_modal.py::stop
```

### View Log Files
```bash
modal run train_modal.py::logs                                # View last 100 lines of debug80.log
modal run train_modal.py::logs --log_file="production.log"    # View specific log file
modal run train_modal.py::logs --tail_lines=50                # View last 50 lines
modal run train_modal.py::logs --tail_lines=0                 # View entire file
```

### List Files
```bash
modal run train_modal.py::files                               # List all files in workspace
```

## Architecture

### Container Setup
- **Image**: Debian Slim with Python 3.10
- **Dependencies**: Installed via `uv sync --frozen` using your lockfile
- **GPU**: Single H100 shared between embedder and training
- **Memory**: 64GB RAM
- **Storage**: Persistent Modal volumes for data, logs, and models

### Process Flow
1. **Embedder Server**: Starts first on localhost:40002
2. **Health Check**: Waits for embedder to be ready
3. **Training**: Runs your exact command with real-time output
4. **Cleanup**: Gracefully shuts down embedder on completion

### Memory Management
- **Embedder**: Uses 40% of GPU memory (`--gpu-memory-utilization 0.4`)
- **Training**: Uses remaining GPU memory for model training
- **RAM**: 64GB for handling large models and datasets

## Configuration

### GPU Types
Edit `train_modal.py` to change GPU:
```python
gpu=modal.gpu.A100(count=1)  # A100
gpu=modal.gpu.T4(count=1)    # T4 (cheaper)
gpu=modal.gpu.H100(count=1)  # H100 (fastest)
```

### Environment Variables
Managed through Modal secrets:
- `wandb`: WANDB_API_KEY, WANDB_API_KEY_NO_ART
- `huggingface`: HF_TOKEN
- `openai`: OPENAI_API_KEY  
- `embedder`: EMBEDDER_API_KEY

### Persistent Storage
Three Modal volumes:
- `slug-search-data`: Your training data and Milvus DB
- `slug-search-logs`: Training logs and metrics
- `slug-search-art`: Model checkpoints and artifacts

## Hardcoded Configurations

The Modal script preserves all your hardcoded settings from `train.py`:

### Debug Mode Settings
- Model: `slug-search-agent-debug-80`
- Project: `slug_search_project_debug`
- Max training samples: 10,240
- Max val samples: 50
- Chat template: Applied from your arguments
- PEFT args: r=128, lora_alpha=256
- Training config: 8 trajectories/group, 128 groups/step
- vLLM config: enforce_eager=True, gpu_memory_utilization=0.80

### Search Tools Configuration
- Milvus DB: `slug_search/data/milvus_hotpotqa_fixed.db`
- Embedding model: `BAAI/bge-large-en-v1.5`
- Embedder API: `http://localhost:40002/v1`
- Top K: 5 (or your custom value)

## Monitoring

### Real-time Output
All output is streamed in real-time to your terminal:
```
[EMBEDDER] Model loaded successfully
[EMBEDDER] Server running on localhost:40002
✅ Embedder server is ready!
=== Starting Training ===
Running in DEBUG mode with minimal configuration...
```

### Log Files
- **debug80.log**: Complete training output (persisted in volume)
- **embedder_vllm_server.log**: Embedder server logs (JSON format)

### Progress Tracking
Training metrics are logged to:
- Local files in `/workspace/validation_logs/`
- W&B (if configured)
- Console output with progress bars

## Stopping Training

### Method 1: Keyboard Interrupt (Recommended)
If you can see the output streaming:
```bash
# Press Ctrl+C while the script is running
^C
```

### Method 2: Modal CLI Commands
```bash
# List running apps
modal app list

# Stop the entire app
modal app stop slug-search-training

# List running functions
modal function list

# Stop specific function
modal function stop <function-id>
```

### Method 3: Emergency Stop Command
```bash
# Emergency stop via script
modal run train_modal.py::stop
```

### Method 4: Modal Dashboard
1. Go to [modal.com](https://modal.com)
2. Navigate to your apps → "slug-search-training"
3. Click "Stop" on the running function

## Troubleshooting

### Common Issues

1. **Embedder timeout**: Increase startup timeout in the script
2. **GPU memory errors**: Reduce embedder memory utilization
3. **Network issues**: Check Modal internet connectivity
4. **Volume access**: Ensure volumes are properly mounted
5. **Can't stop training**: Use emergency stop or Modal dashboard

### Debug Commands

Check embedder health:
```python
import requests
response = requests.get("http://localhost:40002/health")
print(response.status_code)
```

Monitor GPU usage:
```bash
nvidia-smi
```

### Getting Help

1. Check Modal logs: `modal logs list`
2. View function logs: `modal logs <function_id>`
3. Monitor resource usage in Modal dashboard

## Cost Optimization

### GPU Selection
- **H100**: Fastest but most expensive (~$4-6/hour)
- **A100**: Good balance (~$1-3/hour)  
- **T4**: Cheapest but slower (~$0.50-1/hour)

### Optimization Tips
- Use `keep_warm=0` if not running frequently
- Set appropriate timeouts to avoid runaway costs
- Monitor usage in Modal dashboard
- Consider spot instances for non-critical runs

## Files

- `train_modal.py`: Main Modal script
- `setup_modal.sh`: One-time setup script  
- `README_MODAL.md`: This documentation

Your training script runs exactly as it would locally, with the same arguments, configurations, and behavior. The only difference is it's running on powerful cloud GPUs! 