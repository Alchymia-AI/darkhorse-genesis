import os
import json
import torch
import argparse
import numpy as np
from safetensors.torch import load_file
from gguf import GGUFWriter, GGMLQuantizationType

"""
Darkhorse Genesis GGUF Converter 
Alchymia AI Research Lab

"""

def get_ggml_type(tensor):
    """Determines the appropriate GGML type based on tensor dtype."""
    if tensor.dtype == torch.float16:
        return GGMLQuantizationType.F16
    elif tensor.dtype == torch.float32:
        return GGMLQuantizationType.F32
    return GGMLQuantizationType.F32

def convert_to_gguf(model_dir, output_path):
    print(f"🚀 Initializing GGUF Conversion for Darkhorse Genesis...")
    
    # 1. Load Configurations
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config.json in {model_dir}")
        
    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. Initialize GGUF Writer
    writer = GGUFWriter(output_path, "darkhorse_genesis")

    # 3. Write Architecture Metadata 
    # Using add_key to avoid version-specific wrapper issues
    writer.add_string("general.architecture", "darkhorse_genesis")
    writer.add_string("general.name", "Darkhorse Genesis")
    writer.add_uint32("darkhorse.context_length", config.get("model_max_length", 131072))
    writer.add_uint32("darkhorse.embedding_length", config["hidden_size"])
    writer.add_uint32("darkhorse.block_count", config["num_hidden_layers"])
    writer.add_uint32("darkhorse.head_count", config["num_attention_heads"])
    
    # MoE Specifics
    writer.add_uint32("darkhorse.expert_count", config.get("moe_num_experts", 8))
    writer.add_uint32("darkhorse.expert_used_count", config.get("moe_top_k", 2))
    
    # Darkhorse-specific MLA (Multi-Head Latent Attention) metadata
    writer.add_uint32("darkhorse.kv_lora_rank", config.get("kv_lora_rank", 64))
    writer.add_uint32("darkhorse.q_lora_rank", config.get("q_lora_rank", 128))

    # 4. Load Tensors and Write to GGUF
    # Handle both single file and potential shards
    tensor_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    
    print(f"Found {len(tensor_files)} safetensor files. Processing...")

    for ts_file in tensor_files:
        ts_path = os.path.join(model_dir, ts_file)
        state_dict = load_file(ts_path)

        for name, tensor in state_dict.items():
            new_name = name
            
            # Map MLA Projections to GGUF-standardized naming for latent attention
            if "kv_down" in name: new_name = name.replace("kv_down", "attn_kv_a")
            elif "kv_up" in name: new_name = name.replace("kv_up", "attn_kv_b")
            elif "q_down" in name: new_name = name.replace("q_down", "attn_q_a")
            elif "q_up" in name: new_name = name.replace("q_up", "attn_q_b")
            
            # Map MoE Experts
            if "experts" in name:
                new_name = name.replace("experts", "ffn_expert")

            # Convert to numpy and ensure correct alignment/type
            data = tensor.detach().cpu().numpy()
            
            # Check for GGML type compatibility
            ggml_type = get_ggml_type(tensor)
            
            writer.add_tensor(new_name, data, raw_dtype=ggml_type)
            print(f"Mapped: {name} -> {new_name}")

    # 5. Finalize
    print("Writing GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"✅ Conversion complete! GGUF saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darkhorse to GGUF Converter")
    parser.add_argument("--model-dir", type=str, required=True, help="HF model directory")
    parser.add_argument("--output", type=str, default="darkhorse-genesis.gguf", help="Output GGUF file path")
    
    args = parser.parse_args()
    convert_to_gguf(args.model_dir, args.output)