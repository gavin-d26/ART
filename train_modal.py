import modal
import os
from pathlib import Path

# Create a Modal image with uv and exact dependencies from lockfile
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "git", "procps")  # procps provides pkill
    .run_commands(
        [
            # Install uv
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "export PATH=$HOME/.local/bin:$PATH",
        ]
    )
    .env({"PATH": "/root/.local/bin:$PATH"})
    # Copy only dependency files first (for better caching)
    .add_local_file("pyproject.toml", "/workspace/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/workspace/uv.lock", copy=True)
    .add_local_file("README.md", "/workspace/README.md", copy=True)
    # Install dependencies using exact lockfile (heavy operation - cached if deps don't change)
    .run_commands(
        [
            "cd /workspace && /root/.local/bin/uv sync --all-extras",
            "cd /workspace && /root/.local/bin/uv add requests",  # For health checking embedder
        ]
    )
    # Copy code files last (these change frequently, so they go in final layers)
    .add_local_dir("slug_search", "/workspace/slug_search", copy=True)
    .add_local_dir("src", "/workspace/src", copy=True)
)

app = modal.App("slug-search-training", image=image)

# Global process tracking for emergency shutdown
_running_processes = {}


@app.function(
    timeout=60,
    cpu=1.0,
    memory=1024,
)
def emergency_stop():
    """
    Emergency stop function to terminate all running training processes
    """
    import os
    import signal
    import subprocess

    try:
        # Kill all python processes related to training
        subprocess.run(["pkill", "-f", "slug_search.training.train"], check=False)
        subprocess.run(
            ["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False
        )

        return {"status": "stopped", "message": "Training processes terminated"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to stop: {str(e)}"}


@app.function(
    gpu="A100-40GB",  # Dedicated GPU for embedder
    timeout=3600 * 12,  # 12 hours timeout
    cpu=4.0,
    memory=16384,  # 16GB RAM for embedder
    allow_concurrent_inputs=100,  # Allow multiple embedding requests
    # keep_warm=1,  # Removed to avoid unnecessary A100 allocation
    secrets=[
        modal.Secret.from_name("embedder"),
    ],
)
def run_embedder_server():
    """
    Run the vLLM embedder server (equivalent to launch_embedder_vllm.sh)
    """
    import subprocess
    import os
    import time
    import signal
    import sys

    os.chdir("/workspace")

    # Set environment variables matching your script
    env = os.environ.copy()
    env.update(
        {
            "VLLM_CONFIGURE_LOGGING": "1",
            "VLLM_LOGGING_CONFIG_PATH": "./slug_search/vllm_servers/embedder_logging_config.json",
            # Note: CUDA_VISIBLE_DEVICES not needed in Modal as each function gets its own GPU
        }
    )

    # Configuration matching your defaults
    model_name = env.get("EMBEDDER_MODEL_NAME", "BAAI/bge-large-en-v1.5")
    port_num = env.get("EMBEDDER_PORT", "40002")
    task_type = env.get("EMBEDDER_TASK_TYPE", "embed")
    api_key = env.get("EMBEDDER_API_KEY", "EMPTY")

    cmd = [
        "/root/.local/bin/uv",
        "run",
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--port",
        port_num,
        "--host",
        "0.0.0.0",
        "--served-model-name",
        model_name,
        "--task",
        task_type,
        "--gpu-memory-utilization",
        "0.95",
    ]

    print(f"=== Starting VLLM Embedder Server ===")
    print(f"Model: {model_name}")
    print(f"Port: {port_num}")
    print(f"Task: {task_type}")
    print(f"Command: {' '.join(cmd)}")

    # Start the embedder server
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait for server to start and log output
    startup_timeout = 300  # 5 minutes for model loading
    start_time = time.time()
    server_ready = False

    def signal_handler(sig, frame):
        print("Shutting down embedder server...")
        process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue

            print(f"[EMBEDDER] {line.rstrip()}")

            # Check if server is ready
            if "Application startup complete" in line or "Uvicorn running on" in line:
                server_ready = True
                print("🚀 Embedder server is ready!")

            # Timeout check during startup
            if not server_ready and (time.time() - start_time) > startup_timeout:
                print("❌ Embedder server startup timeout!")
                process.terminate()
                raise Exception("Embedder server failed to start within timeout")

        # If we get here, the process ended
        return_code = process.poll()
        if return_code != 0:
            raise Exception(f"Embedder server failed with return code {return_code}")

    except KeyboardInterrupt:
        print("Interrupted, shutting down embedder server...")
        process.terminate()
        process.wait()

    return "Embedder server completed"


@app.function(
    gpu="A100-80GB",  # Single GPU for both embedder and training - cost optimized
    timeout=3600 * 12,  # 12 hours timeout
    cpu=8.0,
    memory=32768 * 2,  # 64GB RAM
    volumes={
        # Persistent storage for model data and logs - mount outside copied directories
        "/data": modal.Volume.from_name("slug-search-data", create_if_missing=True),
        "/logs": modal.Volume.from_name("slug-search-logs", create_if_missing=True),
        "/art": modal.Volume.from_name("slug-search-art", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("wandb"),  # For WANDB_API_KEY_NO_ART
        modal.Secret.from_name("embedder"),  # For EMBEDDER_API_KEY
    ],
)
def run_training_debug_with_embedder(log_file: str = "debug80.log"):
    """
    Run training in debug mode with embedder server in the same container
    Matches your exact local command:
    python -m slug_search.training.train --debug --chat_template "qwen3_preserve_thinking" 2>&1 | tee debug80.log
    """
    import subprocess
    import sys
    import threading
    import time

    # Change to workspace directory
    os.chdir("/workspace")

    # Create symlinks so your code can find data in expected locations
    import subprocess

    subprocess.run(["ln", "-sf", "/data", "slug_search/data"], check=True)
    subprocess.run(["ln", "-sf", "/logs", "validation_logs"], check=True)
    subprocess.run(["ln", "-sf", "/art", ".art"], check=True)
    print(
        "Created symlinks: /data -> slug_search/data, /logs -> validation_logs, /art -> .art"
    )

    # Set up environment variables
    env = os.environ.copy()
    env.update(
        {
            "VLLM_CONFIGURE_LOGGING": "1",
            "VLLM_LOGGING_CONFIG_PATH": "./slug_search/vllm_servers/embedder_logging_config.json",
        }
    )

    print("=== Starting Embedder Server ===")

    # Start embedder server in background
    embedder_cmd = [
        "/root/.local/bin/uv",
        "run",
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "BAAI/bge-large-en-v1.5",
        "--port",
        "40002",
        "--host",
        "0.0.0.0",
        "--served-model-name",
        "BAAI/bge-large-en-v1.5",
        "--task",
        "embed",
        "--gpu-memory-utilization",
        "0.05",  # Leave 80% GPU memory for training model
    ]

    print(f"Embedder command: {' '.join(embedder_cmd)}")

    embedder_process = subprocess.Popen(
        embedder_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for embedder to be ready by monitoring output
    def wait_for_embedder():
        max_wait = 120  # 5 minutes
        start_time = time.time()

        # Wait for embedder process to show "Application startup complete" in its output
        while time.time() - start_time < max_wait:
            if embedder_process.poll() is not None:
                print("❌ Embedder process ended unexpectedly!")
                return False

            print(f"Waiting for embedder startup... ({int(time.time() - start_time)}s)")
            time.sleep(10)

            # Simple time-based check - vLLM usually starts within 60-120 seconds
            if time.time() - start_time > 120:  # 2 minutes should be enough
                print("✅ Embedder server should be ready!")
                return True
        return False

    # Monitor embedder output in background
    def monitor_embedder():
        while embedder_process.poll() is None:
            line = embedder_process.stdout.readline()
            if line:
                print(f"[EMBEDDER] {line.rstrip()}")

    embedder_thread = threading.Thread(target=monitor_embedder, daemon=True)
    embedder_thread.start()

    if not wait_for_embedder():
        embedder_process.terminate()
        raise Exception("Embedder server failed to start within timeout")

    print("=== Starting Training ===")

    # Command exactly matching your local usage
    training_cmd = [
        "/root/.local/bin/uv",
        "run",
        "python",
        "-m",
        "slug_search.training.train",
        "--debug",
        "--chat_template",
        "qwen3_preserve_thinking",
    ]

    print(f"Training command: {' '.join(training_cmd)}")
    print(f"Working directory: {os.getcwd()}")

    # Create log file for output (equivalent to tee debug80.log)
    # Save to persistent logs volume so it survives function completion
    log_file_path = f"/logs/{log_file}"

    try:
        # Run the training with real-time output and logging
        with open(log_file_path, "w") as log_file:
            training_process = subprocess.Popen(
                training_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
            )

            # Stream output in real-time and write to log
            for line in iter(training_process.stdout.readline, ""):
                print(line.rstrip())  # Print to console
                log_file.write(line)  # Write to log file
                log_file.flush()  # Ensure immediate write

            training_process.wait()

            if training_process.returncode != 0:
                raise Exception(
                    f"Training failed with return code {training_process.returncode}"
                )

        print("=== Training completed successfully! ===")

    finally:
        # Clean up embedder server
        print("=== Shutting down embedder server ===")
        embedder_process.terminate()
        embedder_process.wait()

    # Read and return the log content
    with open(log_file_path, "r") as f:
        log_content = f.read()

    return {
        "status": "success",
        "log_file": log_file_path,
        "log_preview": (
            log_content[-2000:] if len(log_content) > 2000 else log_content
        ),  # Last 2000 chars
    }


@app.function(
    gpu="A100-80GB",  # Cost optimized GPU choice
    timeout=3600 * 12,  # 12 hours timeout
    cpu=8.0,
    memory=32768 * 2,  # 64GB RAM
    volumes={
        # Persistent storage for model data and logs - mount outside copied directories
        "/data": modal.Volume.from_name("slug-search-data", create_if_missing=True),
        "/logs": modal.Volume.from_name("slug-search-logs", create_if_missing=True),
        "/art": modal.Volume.from_name("slug-search-art", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("wandb"),  # For WANDB_API_KEY_NO_ART
        modal.Secret.from_name("embedder"),  # For EMBEDDER_API_KEY
    ],
)
def run_training_debug(log_file: str = "debug80.log"):
    """
    Run training in debug mode matching your exact local command:
    python -m slug_search.training.train --debug --chat_template "qwen3_preserve_thinking" 2>&1 | tee debug80.log
    """
    import subprocess
    import sys

    # Change to workspace directory
    os.chdir("/workspace")

    # Set up environment variables that might be needed
    # These will be populated from Modal secrets if available
    env = os.environ.copy()

    # Command exactly matching your local usage
    cmd = [
        "/root/.local/bin/uv",
        "run",
        "python",
        "-m",
        "slug_search.training.train",
        "--debug",
        "--chat_template",
        "qwen3_preserve_thinking",
    ]

    print(f"=== Running command: {' '.join(cmd)} ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path will include: {os.environ.get('PYTHONPATH', 'Not set')}")

    # Create log file for output (equivalent to tee debug80.log)
    # Save to persistent logs volume so it survives function completion
    log_file_path = f"/logs/{log_file}"

    # Run the training with real-time output and logging
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        # Stream output in real-time and write to log
        for line in iter(process.stdout.readline, ""):
            print(line.rstrip())  # Print to console
            log_file.write(line)  # Write to log file
            log_file.flush()  # Ensure immediate write

        process.wait()

        if process.returncode != 0:
            raise Exception(f"Training failed with return code {process.returncode}")

    print("=== Training completed successfully! ===")

    # Read and return the log content
    with open(log_file_path, "r") as f:
        log_content = f.read()

    return {
        "status": "success",
        "log_file": log_file_path,
        "log_preview": (
            log_content[-2000:] if len(log_content) > 2000 else log_content
        ),  # Last 2000 chars
    }


@app.function(
    gpu="A100-80GB",
    timeout=3600 * 12,
    cpu=8.0,
    memory=32768,
    volumes={
        "/data": modal.Volume.from_name("slug-search-data", create_if_missing=True),
        "/logs": modal.Volume.from_name("slug-search-logs", create_if_missing=True),
        "/art": modal.Volume.from_name("slug-search-art", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("wandb"),  # For WANDB_API_KEY_NO_ART
        modal.Secret.from_name("embedder"),  # For EMBEDDER_API_KEY
    ],
)
def run_training_production():
    """
    Run training in production mode
    python -m slug_search.training.train
    """
    import subprocess
    import sys

    os.chdir("/workspace")

    env = os.environ.copy()

    cmd = ["/root/.local/bin/uv", "run", "python", "-m", "slug_search.training.train"]

    print(f"=== Running PRODUCTION command: {' '.join(cmd)} ===")

    # Save to persistent logs volume so it survives function completion
    log_file_path = "/logs/production.log"

    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        for line in iter(process.stdout.readline, ""):
            print(line.rstrip())
            log_file.write(line)
            log_file.flush()

        process.wait()

        if process.returncode != 0:
            raise Exception(
                f"Production training failed with return code {process.returncode}"
            )

    print("=== Production training completed successfully! ===")

    with open(log_file_path, "r") as f:
        log_content = f.read()

    return {
        "status": "success",
        "log_file": log_file_path,
        "log_preview": log_content[-2000:] if len(log_content) > 2000 else log_content,
    }


@app.function(
    gpu="A100-80GB",
    timeout=3600 * 12,
    cpu=8.0,
    memory=32768,
    volumes={
        "/data": modal.Volume.from_name("slug-search-data", create_if_missing=True),
        "/logs": modal.Volume.from_name("slug-search-logs", create_if_missing=True),
        "/art": modal.Volume.from_name("slug-search-art", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("wandb"),  # For WANDB_API_KEY_NO_ART
        modal.Secret.from_name("embedder"),  # For EMBEDDER_API_KEY
    ],
)
def run_training_custom(
    debug: bool = False,
    chat_template: str = None,
    prompt_template: str = None,
    system_prompt: str = None,
    verifier: str = None,
    top_k: int = None,
):
    """
    Run training with custom parameters
    """
    import subprocess
    import sys

    os.chdir("/workspace")

    env = os.environ.copy()

    cmd = ["/root/.local/bin/uv", "run", "python", "-m", "slug_search.training.train"]

    # Add arguments
    if debug:
        cmd.append("--debug")
    if chat_template:
        cmd.extend(["--chat_template", chat_template])
    if prompt_template:
        cmd.extend(["--prompt_template", prompt_template])
    if system_prompt:
        cmd.extend(["--system_prompt", system_prompt])
    if verifier:
        cmd.extend(["--verifier", verifier])
    if top_k is not None:
        cmd.extend(["--top_k", str(top_k)])

    print(f"=== Running CUSTOM command: {' '.join(cmd)} ===")

    mode = "debug" if debug else "custom"
    # Save to persistent logs volume so it survives function completion
    log_file_path = f"/logs/{mode}.log"

    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        for line in iter(process.stdout.readline, ""):
            print(line.rstrip())
            log_file.write(line)
            log_file.flush()

        process.wait()

        if process.returncode != 0:
            raise Exception(
                f"Custom training failed with return code {process.returncode}"
            )

    print("=== Custom training completed successfully! ===")

    with open(log_file_path, "r") as f:
        log_content = f.read()

    return {
        "status": "success",
        "log_file": log_file_path,
        "log_preview": log_content[-2000:] if len(log_content) > 2000 else log_content,
    }


# Local entrypoints for easy usage
@app.local_entrypoint()
def debug(log_file: str = "debug80.log"):
    """
    Run debug training with embedder server (RECOMMENDED - matches your exact local setup):
    python -m slug_search.training.train --debug --chat_template "qwen3_preserve_thinking" 2>&1 | tee debug80.log
    """
    print("🚀 Starting DEBUG training with embedder on Modal...")
    result = run_training_debug_with_embedder.remote(log_file=log_file)
    print("✅ Debug training completed!")
    print("\n=== Training Log Preview ===")
    print(result["log_preview"])
    return result


@app.local_entrypoint()
def debug_no_embedder(log_file: str = "debug80.log"):
    """
    Run debug training without embedder server (will fail if training needs embeddings):
    """
    print("🚀 Starting DEBUG training (NO EMBEDDER) on Modal...")
    result = run_training_debug.remote(log_file=log_file)
    print("✅ Debug training completed!")
    print("\n=== Training Log Preview ===")
    print(result["log_preview"])
    return result


@app.local_entrypoint()
def production():
    """
    Run production training
    """
    print("🚀 Starting PRODUCTION training on Modal...")
    result = run_training_production.remote()
    print("✅ Production training completed!")
    print("\n=== Training Log Preview ===")
    print(result["log_preview"])
    return result


@app.local_entrypoint()
def custom(
    debug: bool = False,
    chat_template: str = None,
    prompt_template: str = None,
    system_prompt: str = None,
    verifier: str = None,
    top_k: int = None,
):
    """
    Run training with custom parameters
    """
    print(f"🚀 Starting {'DEBUG' if debug else 'CUSTOM'} training on Modal...")
    result = run_training_custom.remote(
        debug=debug,
        chat_template=chat_template,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        verifier=verifier,
        top_k=top_k,
    )
    print("✅ Custom training completed!")
    print("\n=== Training Log Preview ===")
    print(result["log_preview"])
    return result


@app.local_entrypoint()
def stop():
    """
    Emergency stop - terminates all running training processes
    """
    print("🛑 Sending emergency stop signal...")
    result = emergency_stop.remote()
    print(f"📋 Result: {result}")
    return result


@app.function(
    timeout=300,
    cpu=1.0,
    memory=2048,
    volumes={
        "/data": modal.Volume.from_name("slug-search-data", create_if_missing=True),
        "/logs": modal.Volume.from_name("slug-search-logs", create_if_missing=True),
        "/art": modal.Volume.from_name("slug-search-art", create_if_missing=True),
    },
)
def view_logs(log_file: str = "debug80.log", tail_lines: int = 100):
    """
    View log files from the training runs
    """
    import os
    from pathlib import Path

    os.chdir("/workspace")

    # Common log file locations (now in persistent volumes)
    log_paths = [
        f"/logs/{log_file}",  # Main log location (persistent)
        f"/workspace/{log_file}",  # Legacy location (ephemeral)
        f"/workspace/validation_logs/{log_file}",  # Training validation logs
    ]

    # Find the log file
    found_log = None
    for path in log_paths:
        if os.path.exists(path):
            found_log = path
            break

    if not found_log:
        # List available log files
        all_logs = []
        for search_dir in [
            "/logs",  # Persistent log volume
            "/workspace",
            "/workspace/validation_logs",
        ]:
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.endswith(".log"):
                        all_logs.append(os.path.join(search_dir, file))

        return {
            "status": "not_found",
            "requested": log_file,
            "available_logs": all_logs,
            "message": f"Log file '{log_file}' not found. Available log files listed above.",
        }

    # Read the log file
    try:
        with open(found_log, "r") as f:
            if tail_lines > 0:
                lines = f.readlines()
                content = "".join(lines[-tail_lines:])
                total_lines = len(lines)
            else:
                content = f.read()
                total_lines = len(content.split("\n"))

        return {
            "status": "success",
            "log_file": found_log,
            "total_lines": total_lines,
            "showing_lines": (
                min(tail_lines, total_lines) if tail_lines > 0 else total_lines
            ),
            "content": content,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error reading log file: {str(e)}"}


@app.function(
    timeout=60,
    cpu=1.0,
    memory=1024,
    volumes={
        "/data": modal.Volume.from_name("slug-search-data", create_if_missing=True),
        "/logs": modal.Volume.from_name("slug-search-logs", create_if_missing=True),
        "/art": modal.Volume.from_name("slug-search-art", create_if_missing=True),
    },
)
def list_files(directory: str = "/workspace"):
    """
    List files in the workspace directories
    """
    import os

    os.chdir("/workspace")

    result = {}

    # Directories to check (including persistent volumes)
    dirs_to_check = [
        "/logs",  # Persistent log volume
        "/data",  # Persistent data volume
        "/art",  # Persistent art volume
        "/workspace",
        "/workspace/validation_logs",
    ]

    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            try:
                files = os.listdir(dir_path)
                result[dir_path] = {
                    "files": files,
                    "log_files": [f for f in files if f.endswith(".log")],
                    "count": len(files),
                }
            except Exception as e:
                result[dir_path] = {"error": str(e)}
        else:
            result[dir_path] = {"status": "not_exists"}

    return result


@app.local_entrypoint()
def logs(log_file: str = "debug80.log", tail_lines: int = 100):
    """
    View training log files

    Examples:
    modal run train_modal.py::logs                           # View last 100 lines of debug80.log
    modal run train_modal.py::logs --log_file="production.log"  # View production.log
    modal run train_modal.py::logs --tail_lines=50              # View last 50 lines
    modal run train_modal.py::logs --tail_lines=0               # View entire file
    """
    print(f"📋 Fetching log file: {log_file}")
    result = view_logs.remote(log_file=log_file, tail_lines=tail_lines)

    if result["status"] == "success":
        print(f"✅ Found log file: {result['log_file']}")
        print(
            f"📊 Total lines: {result['total_lines']}, Showing: {result['showing_lines']}"
        )
        print("\n" + "=" * 80)
        print(result["content"])
        print("=" * 80)
    elif result["status"] == "not_found":
        print(f"❌ Log file not found: {result['requested']}")
        print("\n📁 Available log files:")
        for log in result["available_logs"]:
            print(f"  - {log}")
    else:
        print(f"❌ Error: {result['message']}")

    return result


@app.local_entrypoint()
def files(directory: str = "/workspace"):
    """
    List files in workspace directories
    """
    print("📁 Listing files in workspace...")
    result = list_files.remote(directory=directory)

    for dir_path, info in result.items():
        print(f"\n📂 {dir_path}:")
        if "error" in info:
            print(f"   ❌ Error: {info['error']}")
        elif "status" in info and info["status"] == "not_exists":
            print(f"   📭 Directory does not exist")
        else:
            print(f"   📊 Total files: {info['count']}")
            if info["log_files"]:
                print(f"   📄 Log files: {', '.join(info['log_files'])}")
            if info["files"]:
                print(
                    f"   📋 All files: {', '.join(info['files'][:10])}{'...' if len(info['files']) > 10 else ''}"
                )

    return result
