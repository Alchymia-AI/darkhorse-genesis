import os
import torch
import importlib
from vllm.model_executor.models import ModelRegistry
from vllm import LLM, SamplingParams
from transformers import AutoConfig, LlamaConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

"""
Alchymia AI - Darkhorse Genesis vLLM Integration
Version: 1.9 (MLA CPU Fallback & Error Handling)
"""

# 1. Create a custom config class to fix the consistency error
MODEL_TYPE = "darkhorse_genesis"

class DarkhorseConfig(LlamaConfig):
    model_type = MODEL_TYPE

# Register the custom config properly
if MODEL_TYPE not in CONFIG_MAPPING:
    CONFIG_MAPPING.register(MODEL_TYPE, DarkhorseConfig)
    AutoConfig.register(MODEL_TYPE, DarkhorseConfig)

def get_vllm_base_arch():
    """
    Find the actual class object in vLLM for DeepseekV3 or Llama.
    Priority: DeepseekV3 (for MLA support), fallback to Llama (for compatibility).
    """
    try:
        from vllm.model_executor.models.deepseek_v3 import DeepseekV3ForCausalLM
        # Check if we are on CPU; if so, DeepSeekV3 will raise NotImplementedError for MLA
        if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            print("INFO: GPU/MPS not detected. Falling back to Llama architecture to avoid MLA CPU error.")
            from vllm.model_executor.models.llama import LlamaForCausalLM
            return LlamaForCausalLM, "LlamaForCausalLM"
        return DeepseekV3ForCausalLM, "DeepseekV3ForCausalLM"
    except ImportError:
        try:
            from vllm.model_executor.models.llama import LlamaForCausalLM
            return LlamaForCausalLM, "LlamaForCausalLM"
        except ImportError:
            return None, None

BaseClass, BaseClassName = get_vllm_base_arch()

# 2. Define our Class as a Subclass of a Validated Class
if BaseClass:
    class DarkhorseGenesisForCausalLM(BaseClass):
        """ 
        Inheriting from valid vLLM architecture. 
        If inheriting from DeepseekV3, it uses MLA. 
        If inheriting from Llama, it treats weights as standard attention.
        """
        pass
else:
    raise RuntimeError("Could not find a base vLLM architecture to inherit from.")

# 3. SURGICAL REGISTRY INJECTION
def force_register_darkhorse():
    custom_arch_name = "DarkhorseGenesisForCausalLM"
    ModelRegistry.register_model(custom_arch_name, DarkhorseGenesisForCausalLM)
    
    if hasattr(ModelRegistry, "_models"):
        ModelRegistry._models[custom_arch_name] = DarkhorseGenesisForCausalLM
        ModelRegistry._models[MODEL_TYPE] = DarkhorseGenesisForCausalLM
    print(f"INFO: Architecture {custom_arch_name} (Base: {BaseClassName}) force-injected.")

def run_darkhorse_inference(model_path: str, prompt: str):
    # --- HARDWARE DETECTION ---
    if torch.backends.mps.is_available():
        os.environ["VLLM_TARGET_DEVICE"] = "mps"
        print("INFO: Setting VLLM_TARGET_DEVICE to Metal (MPS)")
    elif torch.cuda.is_available():
        os.environ["VLLM_TARGET_DEVICE"] = "cuda"
        print("INFO: Setting VLLM_TARGET_DEVICE to CUDA")
    else:
        print("WARNING: No GPU/MPS detected. Architecture will attempt CPU execution.")

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=512,
    )

    try:
        force_register_darkhorse()

        # Initialize engine
        llm = LLM(
            model=model_path,
            enforce_eager=True,           
            gpu_memory_utilization=0.4,   
            max_model_len=2048,
            trust_remote_code=True
        )

        outputs = llm.generate([prompt], sampling_params)

        for output in outputs:
            print(f"\n[Darkhorse Output]: {output.outputs[0].text}")

    except Exception as e:
        error_msg = str(e)
        if "MLA is not supported on CPU" in error_msg:
            print("\n[CRITICAL ERROR]: MLA execution failed on this hardware.")
            print("REMEDY: Please ensure you are using a GPU-enabled vLLM build or set BaseClass to LlamaForCausalLM.")
        else:
            print(f"\nCRITICAL ERROR: {e}")

if __name__ == "__main__":
    MODEL_PATH = "./hf_model_out"
    TEST_PROMPT = "Write a technical summary of the Darkhorse model."

    if os.path.exists(MODEL_PATH):
        run_darkhorse_inference(MODEL_PATH, TEST_PROMPT)
    else:
        print(f"Model path {MODEL_PATH} not found.")