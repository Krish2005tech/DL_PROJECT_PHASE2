import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba

class VimModel(nn.Module):
    def __init__(self, dim=192, num_classes=4): # 🔥 Modified for 4-class rotation
        super().__init__()

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=16, stride=16)

        self.mamba_fwd = Mamba(d_model=dim)
        self.mamba_bwd = Mamba(d_model=dim)

        self.norm = nn.LayerNorm(dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        B, D, H, W = x.shape

        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.norm(x)

        # Standard Bidirectional Vim Scan
        fwd = self.mamba_fwd(x)
        bwd = self.mamba_bwd(torch.flip(x, dims=[1]))

        out = fwd + torch.flip(bwd, dims=[1])

        out = out.transpose(1, 2)
        out = self.pool(out).squeeze(-1)

        return self.fc(out)
