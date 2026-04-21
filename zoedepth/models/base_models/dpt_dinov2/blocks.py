"""
Transformer blocks for DPT (Dense Prediction Transformer).
Based on Vision Transformer with patching and DINOv2 features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DPTBlock(nn.Module):
    """
    Single DPT block with attention and MLP.
    """
    
    def __init__(self, dim, num_heads=16, mlp_ratio=4., drop=0., drop_path=0.):
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
        self.drop_path = nn.Identity()
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DPTBlocks(nn.Module):
    """
    Stack of DPT blocks.
    """
    
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., drop=0., drop_path_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            DPTBlock(dim, num_heads, mlp_ratio, drop, drop_path_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using conv (DINOv2 style).
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class ViTAdapter(nn.Module):
    """
    Vision Transformer adapter for DPT.
    """
    
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=1024, depth=24, num_heads=16):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        self.blocks = DPTBlocks(embed_dim, depth, num_heads)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        return x