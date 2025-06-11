# LoRA Adapter Configuration for VLLM Generator Server

This document explains how to configure LoRA (Low-Rank Adaptation) adapters with the VLLM generator server.

## Environment Variables

The following environment variables can be set to configure LoRA adapters:

### Basic LoRA Configuration
- `GENERATOR_ENABLE_LORA`: Set to `"true"` to enable LoRA adapter support (default: `"false"`)
- `GENERATOR_LORA_MODULES`: Space-separated list of LoRA adapters in format `"name1=/path/to/adapter1 name2=/path/to/adapter2"` (default: empty)

### Advanced LoRA Configuration
- `GENERATOR_MAX_LORAS`: Maximum number of LoRA adapters that can be loaded simultaneously (default: `"1"`)
- `GENERATOR_MAX_LORA_RANK`: Maximum rank for LoRA adapters (default: `"16"`)
- `GENERATOR_LORA_DTYPE`: Data type for LoRA computations - `"auto"`, `"float16"`, or `"bfloat16"` (default: `"auto"`)

## Usage Examples

### Example 1: Basic LoRA Setup
```bash
# Enable LoRA with a single adapter
export GENERATOR_ENABLE_LORA="true"
export GENERATOR_LORA_MODULES="my-adapter=/path/to/my/lora/adapter"
./slug_search/vllm_servers/launch_generator_vllm.sh
```

### Example 2: Multiple LoRA Adapters
```bash
# Enable LoRA with multiple adapters
export GENERATOR_ENABLE_LORA="true"
export GENERATOR_LORA_MODULES="sql-lora=/path/to/sql/adapter chat-lora=/path/to/chat/adapter"
export GENERATOR_MAX_LORAS="2"
./slug_search/vllm_servers/launch_generator_vllm.sh
```

### Example 3: Custom LoRA Configuration
```bash
# Enable LoRA with custom settings
export GENERATOR_ENABLE_LORA="true"
export GENERATOR_LORA_MODULES="custom-adapter=/path/to/custom/adapter"
export GENERATOR_MAX_LORAS="3"
export GENERATOR_MAX_LORA_RANK="32"
export GENERATOR_LORA_DTYPE="float16"
./slug_search/vllm_servers/launch_generator_vllm.sh
```

## LoRA Adapter Format

LoRA adapters should be in the standard format expected by VLLM:
- Directory containing `adapter_config.json`
- Directory containing `adapter_model.bin` or `adapter_model.safetensors`
- Compatible with the base model architecture

## API Usage

Once the server is running with LoRA adapters, you can specify which adapter to use in your API requests:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:40001/v1",
    api_key="EMPTY"
)

# Use a specific LoRA adapter
response = client.chat.completions.create(
    model="my-adapter",  # Use the LoRA adapter name
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Dynamic LoRA Loading (Optional)

VLLM also supports dynamic loading/unloading of LoRA adapters at runtime if enabled:

```bash
# Enable dynamic LoRA loading (security risk in production)
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

Then use the `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` endpoints.

## Notes

- LoRA adapters reduce memory usage compared to full fine-tuning
- Multiple adapters can be served efficiently from the same base model
- Adapter switching happens per-request with minimal overhead
- Ensure adapter compatibility with your base model architecture 