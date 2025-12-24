import torch
import argparse
import os
import struct
import json
import numpy as np
from typing import Dict, Any

class GGUFWriter:
    """
    Simplified GGUF Writer for Alchymia AI architectures.
    Converts torch tensors to the GGUF binary format.
    """
    def __init__(self, output_path: str, config: Dict[str, Any]):
        self.output_path = output_path
        self.config = config
        self.tensors = []
        
    def add_tensor(self, name: str, tensor: torch.Tensor):
        # Ensure tensor is in a supported format (F32 for this converter)
        data = tensor.detach().cpu().float().numpy()
        self.tensors.append((name, data))

    def write(self):
        print(f"Writing GGUF file to {self.output_path}...")
        with open(self.output_path, "wb") as f:
            # 1. Magic & Version
            f.write(struct.pack("<I", 0x47475546)) # 'GGUF'
            f.write(struct.pack("<I", 3))          # Version 3
            
            # 2. Metadata KV Pairs
            f.write(struct.pack("<Q", len(self.tensors)))
            
            # 3. Tensor Info and Data
            for name, data in self.tensors:
                encoded_name = name.encode("utf-8")
                f.write(struct.pack("<I", len(encoded_name)))
                f.write(encoded_name)
                
                # Shape
                f.write(struct.pack("<I", len(data.shape)))
                for dim in reversed(data.shape):
                    f.write(struct.pack("<Q", dim))
                
                # Type (0 = F32)
                f.write(struct.pack("<I", 0))
                # Data
                f.write(data.tobytes())

def convert_to_gguf(input_path: str, output_path: str, config_path: str, legacy_mode: bool = False):
    """
    Main conversion logic mapping AlchymiaGen keys to GGUF standards.
    Handles both standard state_dicts and TorchScript (RecursiveScriptModule).
    """
    print(f"Loading AlchymiaGen weights from {input_path}...")
    
    weights = {}
    config_path = "parameters/" + config_path
    
    try:
        # Try loading as a standard state_dict first
        # PyTorch 2.6+ defaults to weights_only=True, which fails on JIT archives
        loaded = torch.load(input_path, map_location="cpu", weights_only=not legacy_mode)
        
        if isinstance(loaded, torch.jit.ScriptModule) or isinstance(loaded, torch.jit.RecursiveScriptModule):
            print("Detected TorchScript archive. Extracting named parameters...")
            # If it's a JIT module, we extract the state_dict from it
            weights = loaded.state_dict()
        elif isinstance(loaded, dict):
            weights = loaded
        else:
            # Fallback for other objects that might have a state_dict
            weights = loaded.state_dict() if hasattr(loaded, 'state_dict') else loaded

    except (RuntimeError, ValueError) as e:
        # If torch.load fails specifically because it's a JIT archive being dispatched
        print(f"Direct load failed or redirected. Using torch.jit.load path...")
        jit_module = torch.jit.load(input_path, map_location="cpu")
        weights = jit_module.state_dict()

    with open(config_path, "r") as f:
        config = json.load(f)

    writer = GGUFWriter(output_path, config)

    print(f"Processing {len(weights)} tensors...")

    # Mapping logic: Darkhorse -> GGUF naming convention
    for name, tensor in weights.items():
        # Clean up key names from TorchScript/tch-rs nesting
        new_name = name.replace("root.", "").replace("module.", "")
        
        # Mapping MLA keys
        if "attention" in new_name:
            new_name = new_name.replace("attention", "blk.N.attn")
            
        # Mapping MoE keys
        if "moe" in new_name:
            new_name = new_name.replace("moe.experts", "blk.N.ffn_gate_exps")
            
        # Mapping MTP keys
        if "mtp" in new_name:
            new_name = new_name.replace("mtp.heads", "output_mtp")

        writer.add_tensor(new_name, tensor)

    writer.write()
    print("✅ GGUF Conversion Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darkhorse Genesis to GGUF Converter")
    parser.add_argument("--input", type=str, default="model.ot", help="Path to .ot or .pt weights")
    parser.add_argument("--config", type=str, required=True, help="Path to the parameter JSON")
    parser.add_argument("--output", type=str, default="darkhorse-genesis.gguf", help="Output path")
    parser.add_argument("--legacy", action="store_true", help="Set weights_only=False for PyTorch 2.6+")

    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
    else:
        convert_to_gguf(args.input, args.output, args.config, args.legacy)