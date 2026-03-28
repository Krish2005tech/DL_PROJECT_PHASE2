import torch
import torch.nn as nn
from einops import rearrange
from models.quad_vim_block import QuadMambaBlock

class QuadVimModel(nn.Module):
    def __init__(self, dim=192, num_classes=4): # 🔥 Modified for 4-class rotation
        super().__init__()

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.block = QuadMambaBlock(dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        B, D, H, W = x.shape

        x = rearrange(x, 'b d h w -> b (h w) d')

        # Your Novelty: 4-way spatial scanning
        x = self.block(x, H, W)

        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)

        return self.fc(x)
