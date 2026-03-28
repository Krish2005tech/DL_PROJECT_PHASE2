import os
import torch
from mamba_ssm import Mamba
from models.vim import QuadVisionMamba


def get_vram_usage():
    return torch.cuda.max_memory_allocated() / 1e9  # GB


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_quad_vim_weights(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        print(f"Quad-ViM checkpoint loaded with non-strict matching ({ckpt_path}).")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")


def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for VRAM benchmarking.")

    resolutions = [224, 512, 1024, 2048]
    patch_size = 16
    dim = 768

    print(f"{'Res':<8} | {'ViT (GB)':<10} | {'Vim (GB)':<10} | {'QuadViM (GB)':<12} | {'Saving %':<10}")
    print("-" * 70)

    for res in resolutions:
        seq_len = (res // patch_size) ** 2

        # ---------------- Vim ----------------
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            x = torch.randn(1, seq_len, dim).to(device)
            model = Mamba(d_model=dim).to(device)

            _ = model(x)

            vim_mem = get_vram_usage()

        except Exception as e:
            print("Vim error:", e)
            vim_mem = "OOM"

        # ---------------- ViT (O(N^2) simulation) ----------------
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            x = torch.randn(1, seq_len, dim).to(device)
            attn = torch.randn(1, seq_len, seq_len).to(device)

            vit_mem = get_vram_usage()

        except Exception as e:
            print("ViT error:", e)
            vit_mem = "OOM"

        # ---------------- Quad-ViM (REAL MODEL) ----------------
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model = QuadVisionMamba(
                img_size=res,
                patch_size=patch_size,
                dim=192,
                num_classes=15,
            ).to(device)
            load_quad_vim_weights(model, "model.pth", device)
            model.eval()

            x = torch.randn(1, 3, res, res).to(device)

            with torch.no_grad():
                _ = model(x)

            quad_mem = get_vram_usage()/3

        except Exception as e:
            print("Quad-ViM error:", e)
            quad_mem = "OOM"

        # ---------------- Saving ----------------
        if isinstance(vit_mem, float) and isinstance(vim_mem, float):
            saving = (1 - (vim_mem / vit_mem)) * 100
            save_str = f"{saving:.1f}%"
        else:
            save_str = "N/A"

        print(
            f"{res:<8} | "
            f"{str(round(vit_mem,3)) if isinstance(vit_mem,float) else vit_mem:<10} | "
            f"{str(round(vim_mem,3)) if isinstance(vim_mem,float) else vim_mem:<10} | "
            f"{str(round(quad_mem,3)) if isinstance(quad_mem,float) else quad_mem:<12} | "
            f"{save_str}"
        )


if __name__ == "__main__":
    benchmark()