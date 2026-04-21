"""
Patch transformer layer for attention-based depth refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchTransformer(nn.Module):
    """
    Transformer layer operating on image patches for depth refinement.
    """
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
        self.dim = dim
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        x = rearrange(x, 'b h w c -> b (h w) c')
        
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        x = rearrange(x, 'b h w c -> b c h w')
        
        return x