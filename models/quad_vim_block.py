import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange

class QuadMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mamba_tl_br = Mamba(d_model=dim)
        self.mamba_br_tl = Mamba(d_model=dim)
        self.mamba_tr_bl = Mamba(d_model=dim)
        self.mamba_bl_tr = Mamba(d_model=dim)

        self.norm = nn.LayerNorm(dim)
        self.fuse = nn.Linear(dim * 4, dim)
        self.act = nn.SiLU()

    def forward(self, x, h, w):
        B, L, D = x.shape
        assert L == h * w

        x_norm = self.norm(x)

        x_2d = rearrange(x_norm, 'b (h w) d -> b d h w', h=h, w=w)

        p1 = x_norm
        p2 = rearrange(torch.flip(x_2d, dims=[2,3]), 'b d h w -> b (h w) d')
        p3 = rearrange(torch.flip(x_2d, dims=[3]), 'b d h w -> b (h w) d')
        p4 = rearrange(torch.flip(x_2d, dims=[2]), 'b d h w -> b (h w) d')

        out1 = self.mamba_tl_br(p1)
        out2 = self.mamba_br_tl(p2)
        out3 = self.mamba_tr_bl(p3)
        out4 = self.mamba_bl_tr(p4)

        out = torch.cat([out1, out2, out3, out4], dim=-1)
        out = self.act(self.fuse(out))

        return x + out
