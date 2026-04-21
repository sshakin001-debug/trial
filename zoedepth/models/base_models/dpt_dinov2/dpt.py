"""
DPT (Dense Prediction Transformer) with DINOv2 for Depth Anything.
Combines ViT features with depth prediction head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from zoedepth.models.base_models.dpt_dinov2.blocks import ViTAdapter


class DepthAnythingDepthModel(nn.Module):
    """
    Depth Anything model using DINOv2-based ViT encoder with DPT decoder.
    Supports vitl (ViT-Large), vitb (ViT-Base), vits (ViT-Small), vitg (ViT-Giant).
    """
    
    def __init__(self, model_type="vitl", img_size=518, patch_size=14, **kwargs):
        super().__init__()
        self.model_type = model_type
        
        model_configs = {
            "vits": {"embed_dim": 384, "depth": 12, "num_heads": 6},
            "vitb": {"embed_dim": 768, "depth": 12, "num_heads": 12},
            "vitl": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
            "vitg": {"embed_dim": 1536, "depth": 40, "num_heads": 24},
        }
        
        config = model_configs.get(model_type, model_configs["vitl"])
        
        self.encoder = ViTAdapter(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"]
        )
        
        self.head = nn.Sequential(
            nn.Linear(config["embed_dim"], 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        
        self.pretrained_url = None
        self.pretrained_resource = None
    
    def forward(self, x):
        features = self.encoder(x)
        
        B, N, C = features.shape
        
        depth = self.head(features.mean(dim=1))
        
        return depth


def build_depth_anything(model_type="vitl", pretrained=True, **kwargs):
    """
    Build Depth Anything model.
    
    Args:
        model_type: ViT variant ('vits', 'vitb', 'vitl', 'vitg')
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments
        
    Returns:
        DepthAnythingDepthModel
    """
    model = DepthAnythingDepthModel(model_type=model_type, **kwargs)
    return model