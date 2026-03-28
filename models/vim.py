import torch
import torch.nn as nn
from models.quad_vim_block import QuadMambaBlock

class QuadVisionMamba(nn.Module):
    def __init__(self, img_size=224, patch_size=16, channels=3, dim=192, depth=12, num_classes=15):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        
        # 1. Patch Embedding: Turn images into sequences
        self.patch_embed = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        
        # 2. Stack of Quad-Mamba Layers
        self.layers = nn.ModuleList([
            QuadMambaBlock(dim=dim) for _ in range(depth)
        ])
        
        # 3. Head: Normalization and Classification
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: [Batch, 3, H, W]
        x = self.patch_embed(x)             # -> [B, D, h_grid, w_grid]
        x = x.flatten(2).transpose(1, 2)    # -> [B, Seq_Len, D]
        
        # Pass through our 4-way scanning blocks
        for layer in self.layers:
            x = layer(x, self.grid_size, self.grid_size)
            
        x = self.norm(x.mean(dim=1))        # Global Average Pooling
        return self.head(x)
