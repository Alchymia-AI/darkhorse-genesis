import sys
import os

# vLLM to target Metal (MPS) for Apple Silicon
os.environ["VLLM_TARGET_DEVICE"] = "mps"

# 1. Bypass transformers version check
try:
    import transformers
    transformers.__version__ = "5.0.0.dev0"
    print(f"DEBUG: Manually set transformers.__version__ to {transformers.__version__}")
except ImportError:
    print("Warning: Transformers not found in environment.")

# 2. Force V0 Engine (V1 support for MPS is experimental)
try:
    from vllm.entrypoints.openai.api_server import main as vllm_v0_main
    print("DEBUG: Switching to vLLM V0 Engine for Metal (MPS) compatibility.")
except ImportError:
    from vllm.entrypoints.cli.main import main as vllm_v0_main

if __name__ == "__main__":
    # Simulate the command line arguments
    # Added memory constraints and remote code trust for stability
    sys.argv = [
        "vllm", 
        "serve", 
        "./hf_model_out", 
        "--tokenizer", "./hf_model_out",
        "--load-format", "safetensors",
        "--enforce-eager",
        "--kv-cache-dtype", "auto", 
        "--max-model-len", "2048",         # Reduced for stability during init
        "--gpu-memory-utilization", "0.7", # Leave room for OS/UI
        "--trust-remote-code",             # Required for custom architectures
        "--distributed-executor-backend", "mp" # Ensure multiprocess works on macOS
    ]
    
    print(f"Launching vLLM (Metal/MPS Mode) with args: {sys.argv}")
    
    try:
        vllm_v0_main()
    except Exception as e:
        print(f"\n[ENGINE ERROR] {type(e).__name__}: {e}")
        print("\nTip: Ensure you are running on macOS with a version of vLLM compiled with Metal support.")