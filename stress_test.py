import torch
import torch.nn as nn
from mamba_ssm import Mamba
from timm.models.vision_transformer import VisionTransformer

def get_vram_usage():
    return torch.cuda.memory_allocated() / 1e9  # Returns GB

def benchmark():
    resolutions = [224, 512, 1024, 2048] # Satellite images are large!
    patch_size = 16
    dim = 768 # Standard Base model dimension

    print(f"{'Res':<10} | {'ViT VRAM (GB)':<15} | {'Vim VRAM (GB)':<15} | {'Saving %':<10}")
    print("-" * 60)

    for res in resolutions:
        seq_len = (res // patch_size) ** 2
        
        # --- Test Vim ---
        try:
            torch.cuda.empty_cache()
            x_mamba = torch.randn(1, seq_len, dim).to("cuda")
            model_mamba = Mamba(d_model=dim).to("cuda")
            _ = model_mamba(x_mamba)
            vim_mem = get_vram_usage()
        except Exception as e:
            vim_mem = "OOM"

        # --- Test ViT ---
        try:
            torch.cuda.empty_cache()
            # Simplified Attention complexity simulation
            x_vit = torch.randn(1, seq_len, dim).to("cuda")
            # We use a linear layer to simulate the attention matrix size O(N^2)
            attn_matrix = torch.randn(1, seq_len, seq_len).to("cuda") 
            vit_mem = get_vram_usage()
        except Exception as e:
            vit_mem = "OOM"

        # Calculate Saving
        if isinstance(vit_mem, float) and isinstance(vim_mem, float):
            saving = (1 - (vim_mem / vit_mem)) * 100
            save_str = f"{saving:.1f}%"
        else:
            save_str = "N/A (ViT Crashed)"

        print(f"{res:<10} | {str(vit_mem)[:6]:<15} | {str(vim_mem)[:6]:<15} | {save_str}")

if __name__ == "__main__":
    benchmark()
